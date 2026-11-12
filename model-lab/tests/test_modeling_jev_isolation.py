"""Task A4 tests: token_roles 4D-mask isolation, ModernBERT-Jev / NanoJev-0.6B, strictly proper losses.

All models are TINY, randomly initialised from configs (CPU, no network, no HF weight download).
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from transformers import ModernBertConfig, Qwen3Config

from diodemodel.contract import (
    CANONICAL_SEGMENT_IDS,
    ONNX_INPUT_SPECS,
    ONNX_OPSET_VERSION,
    ONNX_OUTPUT_SPECS,
    ROLE_PAD,
    ROLE_QUERY,
    ROLE_STATE,
    validate_onnx_contract,
)
from diodemodel.dataset_packer import build_reference_4d_attn_mask
from diodemodel.losses_rlcd import (
    JevLossConfig,
    brier_loss,
    compute_jev_loss,
    expected_calibration_error,
    fit_temperature_scaling,
    masked_pointer_cross_entropy,
    probs_to_logits,
    temperature_calibration_report,
)
from diodemodel.modeling_jev import (
    HEAD_SCOPE_NAMES,
    ONNX_DYNAMIC_AXES,
    NanoJevCausalModel,
    ModernBERTJevModel,
    build_token_roles_attn_mask,
    pin_onnx_output_shapes,
)

VOCAB = 64
PAD_ID = 0
N_PCHUNK_SLOTS = 33
N_ARG_SLOTS = 17
QUERY_IDS = (3, 4, 5, 6, 7)  # q_enc, q_sem, q_hijack, q_src, q_arg
NONE_ID = 8
INPUT_NAMES = tuple(ONNX_INPUT_SPECS.keys())
OUTPUT_NAMES = tuple(ONNX_OUTPUT_SPECS.keys())

TRACK_B_PACKER = Path("/root/diodemcp/packages/diodemcp/src/diodemcp/packer.py")


# --------------------------------------------------------------------------------------
# Tiny model + synthetic packed-batch helpers
# --------------------------------------------------------------------------------------
def tiny_modernbert() -> ModernBERTJevModel:
    torch.manual_seed(1234)
    cfg = ModernBertConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,  # layer 0 = full_attention (global), layer 1 = sliding_attention (local)
        num_attention_heads=4,
        local_attention=8,  # half-window 4: far smaller than the test sequences
        max_position_embeddings=512,
        pad_token_id=PAD_ID,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    return ModernBERTJevModel.from_config(cfg).eval()


def tiny_nano() -> NanoJevCausalModel:
    torch.manual_seed(4321)
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        pad_token_id=PAD_ID,
    )
    return NanoJevCausalModel.from_config(cfg).eval()


MODEL_FACTORIES = {"modernbert": tiny_modernbert, "nano": tiny_nano}


@pytest.fixture(params=sorted(MODEL_FACTORIES))
def model(request):
    return MODEL_FACTORIES[request.param]()


def make_batch(
    batch: int = 3, length: int = 128, seed: int = 0, n_state_min: int = 30, n_state_max: int = 70
) -> dict[str, torch.Tensor]:
    """Synthetic rows with the normative layout: STATE..., <|none|>(STATE), 5 QUERY, PAD..."""
    rng = np.random.default_rng(seed)
    n_state_max = min(n_state_max, length - 6)
    n_state_min = min(n_state_min, n_state_max - 1)
    ids = np.full((batch, length), PAD_ID, dtype=np.int64)
    segs = np.zeros((batch, length), dtype=np.int64)
    roles = np.zeros((batch, length), dtype=np.int64)
    qpos = np.zeros((batch, 5), dtype=np.int64)
    ppos = np.zeros((batch, N_PCHUNK_SLOTS), dtype=np.int64)
    pval = np.zeros((batch, N_PCHUNK_SLOTS), dtype=np.bool_)
    apos = np.zeros((batch, N_ARG_SLOTS), dtype=np.int64)
    aval = np.zeros((batch, N_ARG_SLOTS), dtype=np.bool_)
    for b in range(batch):
        n_state = int(rng.integers(n_state_min, n_state_max))  # includes the <|none|> token
        none_pos = n_state - 1
        ids[b, :n_state] = rng.integers(10, VOCAB, size=n_state)
        segs[b, :n_state] = rng.integers(0, 5, size=n_state)
        roles[b, :n_state] = ROLE_STATE
        ids[b, none_pos] = NONE_ID
        for k in range(5):
            p = n_state + k
            qpos[b, k] = p
            ids[b, p] = QUERY_IDS[k]
            segs[b, p] = CANONICAL_SEGMENT_IDS["DECISION_QUERY"]
            roles[b, p] = ROLE_QUERY
        n_p = int(rng.integers(0, 12))
        n_a = int(rng.integers(0, 6))
        cand = rng.choice(none_pos, size=n_p + n_a, replace=False)
        ppos[b, :] = none_pos
        apos[b, :] = none_pos
        ppos[b, :n_p] = cand[:n_p]
        apos[b, :n_a] = cand[n_p:]
        pval[b, :n_p] = True
        pval[b, 32] = True
        aval[b, :n_a] = True
        aval[b, 16] = True
    return {
        "input_ids": torch.from_numpy(ids),
        "seg_ids": torch.from_numpy(segs),
        "token_roles": torch.from_numpy(roles),
        "query_positions": torch.from_numpy(qpos),
        "pchunk_positions": torch.from_numpy(ppos),
        "pchunk_valid_mask": torch.from_numpy(pval),
        "arg_positions": torch.from_numpy(apos),
        "arg_valid_mask": torch.from_numpy(aval),
    }


def repad(batch: dict[str, torch.Tensor], new_len: int) -> dict[str, torch.Tensor]:
    out = dict(batch)
    cur = batch["input_ids"].shape[1]
    assert new_len >= cur
    pad = new_len - cur
    for key, fill in (("input_ids", PAD_ID), ("seg_ids", 0), ("token_roles", ROLE_PAD)):
        out[key] = torch.nn.functional.pad(batch[key], (0, pad), value=fill)
    return out


def call(model, batch, *, logits: bool = False):
    fn = model.forward_logits if logits else model
    return fn(*[batch[name] for name in INPUT_NAMES])


# --------------------------------------------------------------------------------------
# Mask construction
# --------------------------------------------------------------------------------------
def _load_track_b_reference_mask():
    """Extract Track B's `build_reference_4d_attn_mask` from source (diodemcp is not a model-lab dep)."""
    if not TRACK_B_PACKER.is_file():
        pytest.skip("Track B packer.py not available in this checkout")
    tree = ast.parse(TRACK_B_PACKER.read_text(encoding="utf-8"))
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_reference_4d_attn_mask"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ns = {
        "np": np,
        "TOKEN_ROLE_PAD": 0,
        "TOKEN_ROLE_STATE": 1,
        "TOKEN_ROLE_QUERY": 2,
    }
    exec(compile(ast.fix_missing_locations(module), str(TRACK_B_PACKER), "exec"), ns)  # noqa: S102
    return ns["build_reference_4d_attn_mask"]


def test_mask_equals_dataset_packer_and_track_b_reference_masks():
    batch = make_batch(batch=4, length=96, seed=11)
    roles = batch["token_roles"]
    mine = build_token_roles_attn_mask(roles).numpy()
    ref_a = build_reference_4d_attn_mask(roles.numpy())
    assert mine.dtype == np.float32 and mine.shape == (4, 1, 96, 96)
    np.testing.assert_array_equal(mine, ref_a)
    ref_b = _load_track_b_reference_mask()(roles.numpy())
    np.testing.assert_array_equal(mine, ref_b)


def test_mask_handles_degenerate_rows_without_nan_rows():
    roles = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],  # all PAD
            [1, 1, 2, 2, 0, 0],
            [1, 2, 2, 2, 2, 2],
            [2, 0, 1, 1, 0, 2],  # scrambled
        ],
        dtype=torch.int64,
    )
    for kwargs in ({}, {"causal": True}, {"sliding_window": 1}, {"causal": True, "sliding_window": 2}):
        mask = build_token_roles_attn_mask(roles, **kwargs)
        assert torch.isfinite(mask).all()
        assert set(mask.unique().tolist()) <= {0.0, -10000.0}
        # I8: every row (including PAD rows) has at least one open key: its own diagonal.
        assert (mask.amax(dim=-1) == 0.0).all()
        diag = torch.diagonal(mask[:, 0], dim1=-2, dim2=-1)
        assert (diag == 0.0).all()
        # fp16 / bf16 representation stays finite & identical in structure.
        assert torch.isfinite(mask.half()).all() and torch.isfinite(mask.bfloat16()).all()
        assert (mask.half().amax(dim=-1) == 0.0).all()


def test_mask_pad_rows_and_columns_semantics():
    roles = torch.tensor([[1, 1, 2, 2, 0, 0]])
    m = build_token_roles_attn_mask(roles)[0, 0]
    allowed = m == 0.0
    # STATE rows: all STATE columns, nothing else besides own diagonal.
    assert allowed[0].tolist() == [True, True, False, False, False, False]
    # QUERY rows: STATE columns + own diagonal only (no other QUERY, no PAD).
    assert allowed[2].tolist() == [True, True, True, False, False, False]
    assert allowed[3].tolist() == [True, True, False, True, False, False]
    # PAD rows: diagonal only.
    assert allowed[4].tolist() == [False] * 4 + [True, False]
    assert allowed[5].tolist() == [False] * 5 + [True]


def test_causal_mask_is_lower_triangular_and_query_reads_preceding_state_plus_self():
    batch = make_batch(batch=2, length=64, seed=5)
    roles = batch["token_roles"]
    causal = build_token_roles_attn_mask(roles, causal=True)
    full = build_token_roles_attn_mask(roles)
    tril = torch.tril(torch.ones(64, 64, dtype=torch.bool))
    allowed_c = causal[:, 0] == 0.0
    assert not (allowed_c & ~tril[None]).any(), "causal mask opened a future position"
    # Layout has all QUERY tokens after all STATE tokens => causal == reference ∩ lower-triangular,
    # and QUERY rows still see every STATE token.
    expected = (full[:, 0] == 0.0) & tril[None]
    assert torch.equal(allowed_c, expected)
    for b in range(2):
        for qp in batch["query_positions"][b].tolist():
            n_state = int((roles[b] == ROLE_STATE).sum())
            assert allowed_c[b, qp, :n_state].all()


def test_sliding_window_intersects_state_rows_only():
    batch = make_batch(batch=2, length=64, seed=6)
    roles = batch["token_roles"]
    w = 4
    local = build_token_roles_attn_mask(roles, sliding_window=w)[:, 0] == 0.0
    full = build_token_roles_attn_mask(roles)[:, 0] == 0.0
    idx = torch.arange(64)
    within = (idx[:, None] - idx[None, :]).abs() <= w
    state_rows = (roles == ROLE_STATE)[:, :, None]
    expected = torch.where(state_rows, full & within[None], full)
    assert torch.equal(local, expected)
    # QUERY rows keep the *full* receptive field over STATE (not windowed).
    q = (roles == ROLE_QUERY)[:, :, None]
    assert torch.equal(local & q, full & q)


def test_model_global_mask_matches_reference_for_modernbert_and_nano_masks_are_causal(model):
    batch = make_batch(batch=2, length=64, seed=7)
    masks = model.build_attention_masks(batch["token_roles"])
    ref = torch.from_numpy(build_reference_4d_attn_mask(batch["token_roles"].numpy()))
    if isinstance(model, ModernBERTJevModel):
        assert torch.equal(masks["full_attention"], ref)
        assert "sliding_attention" in masks
        assert not torch.equal(masks["sliding_attention"], masks["full_attention"])
    else:
        assert torch.equal(masks["full_attention"], build_token_roles_attn_mask(batch["token_roles"], causal=True))


# --------------------------------------------------------------------------------------
# Fidelity: custom mask with all-STATE rows == the backbone's native masking
# --------------------------------------------------------------------------------------
def test_all_state_rows_reproduce_native_backbone_attention(model):
    """Guards our mask semantics (incl. ModernBERT sliding-window half-width) against HF's own."""
    batch = make_batch(batch=2, length=48, seed=8)
    roles = torch.full_like(batch["token_roles"], ROLE_STATE)
    with torch.no_grad():
        mine = model.encode(batch["input_ids"], batch["seg_ids"], roles)
        embeds = model.embed(batch["input_ids"], batch["seg_ids"])
        native = model.backbone(inputs_embeds=embeds, attention_mask=None).last_hidden_state
    torch.testing.assert_close(mine, native, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------------------
# Query isolation, hook-liveness guard, padding invariance, NaN safety
# --------------------------------------------------------------------------------------
def _perturb_token(batch, b, pos, new_id):
    out = {k: v.clone() for k, v in batch.items()}
    assert int(out["input_ids"][b, pos]) != new_id
    out["input_ids"][b, pos] = new_id
    out["seg_ids"][b, pos] = (int(out["seg_ids"][b, pos]) + 1) % 6
    return out


def test_query_isolation_hidden_states_and_heads(model):
    batch = make_batch(batch=3, length=128, seed=21)
    with torch.no_grad():
        base_h = model.encode(batch["input_ids"], batch["seg_ids"], batch["token_roles"])
        base_logits = call(model, batch, logits=True)
        for j in range(5):
            for b in range(3):
                pos_j = int(batch["query_positions"][b, j])
                pert = _perturb_token(batch, b, pos_j, 40 + j)
                h = model.encode(pert["input_ids"], pert["seg_ids"], pert["token_roles"])
                logits = call(model, pert, logits=True)
                # Any *other* query hidden state is bit-stable; the perturbed one actually moves.
                for k in range(5):
                    pos_k = int(batch["query_positions"][b, k])
                    d = (h[b, pos_k] - base_h[b, pos_k]).abs().max().item()
                    if k == j:
                        assert d > 1e-3, "perturbing a query must change its own readout (mask hook alive)"
                    else:
                        assert d < 1e-7, f"query {j} leaked into query {k}: {d}"
                # STATE hidden states never see QUERY tokens at all.
                n_state = int((batch["token_roles"][b] == ROLE_STATE).sum())
                assert (h[b, :n_state] - base_h[b, :n_state]).abs().max().item() < 1e-7
                # Head outputs: q_enc/q_sem/q_hijack -> noul[0..2]; q_src -> src; q_arg -> arg.
                for k in range(3):
                    if k != j:
                        assert (logits[0][b, k] - base_logits[0][b, k]).abs().item() < 1e-7
                if j != 3:
                    assert (logits[1][b] - base_logits[1][b]).abs().max().item() < 1e-7
                if j != 4:
                    assert (logits[2][b] - base_logits[2][b]).abs().max().item() < 1e-7


def test_mask_hook_guard_state_tokens_do_interact_and_queries_read_state(model):
    """Fails loudly if the mask hook silently stops working (e.g. HF ignores the dict mask)."""
    batch = make_batch(batch=2, length=128, seed=22, n_state_min=60, n_state_max=70)
    with torch.no_grad():
        base_h = model.encode(batch["input_ids"], batch["seg_ids"], batch["token_roles"])
        for b in range(2):
            n_state = int((batch["token_roles"][b] == ROLE_STATE).sum())
            # Perturb the very first STATE token (>> sliding window away from the end of the state).
            pert = _perturb_token(batch, b, 3, 50)
            h = model.encode(pert["input_ids"], pert["seg_ids"], pert["token_roles"])
            last_state = n_state - 1
            assert (h[b, last_state] - base_h[b, last_state]).abs().max().item() > 1e-6
            if isinstance(model, NanoJevCausalModel):
                # causal: a *future* STATE token never influences an earlier one.
                pert_late = _perturb_token(batch, b, last_state, 50)
                h2 = model.encode(pert_late["input_ids"], pert_late["seg_ids"], pert_late["token_roles"])
                assert (h2[b, : last_state] - base_h[b, : last_state]).abs().max().item() < 1e-7
            for k in range(5):
                qp = int(batch["query_positions"][b, k])
                assert (h[b, qp] - base_h[b, qp]).abs().max().item() > 1e-7, "QUERY must read STATE"


def test_modernbert_local_layer_uses_windowed_state_mask_but_full_query_readout():
    m = tiny_modernbert()
    batch = make_batch(batch=1, length=128, seed=23, n_state_min=90, n_state_max=100)
    n_state = int((batch["token_roles"][0] == ROLE_STATE).sum())
    masks = m.build_attention_masks(batch["token_roles"])
    local = masks["sliding_attention"][0, 0] == 0.0
    assert not local[n_state - 2, 0], "STATE row must be windowed in local layers"
    assert local[n_state - 2, n_state - 2 - m.backbone.config.sliding_window]
    assert local[int(batch["query_positions"][0, 0]), 0], "QUERY row must read STATE fully"


def test_padding_bucket_invariance_128_vs_256(model):
    batch = make_batch(batch=3, length=128, seed=31)
    big = repad(batch, 256)
    with torch.no_grad():
        a = call(model, batch, logits=True)
        b = call(model, big, logits=True)
        pa = call(model, batch)
        pb = call(model, big)
    for x, y in list(zip(a, b)) + list(zip(pa, pb)):
        assert (x - y).abs().max().item() < 1e-6


def test_pad_rows_never_nan_fp32_bf16_autocast_and_all_pad_rows(model):
    batch = make_batch(batch=3, length=128, seed=41)
    # Make row 2 pathological: completely padded (no STATE, no QUERY).
    for key, fill in (("input_ids", PAD_ID), ("seg_ids", 0), ("token_roles", ROLE_PAD)):
        batch[key][2, :] = fill
    with torch.no_grad():
        h = model.encode(batch["input_ids"], batch["seg_ids"], batch["token_roles"])
        assert torch.isfinite(h).all(), "PAD rows produced NaN/Inf hidden states in fp32"
        with torch.autocast("cpu", dtype=torch.bfloat16):
            h16 = model.encode(batch["input_ids"], batch["seg_ids"], batch["token_roles"])
            outs = call(model, batch)
        assert torch.isfinite(h16.float()).all(), "PAD rows produced NaN/Inf under bf16 autocast"
        for o in outs:
            assert torch.isfinite(o.float()).all()


# --------------------------------------------------------------------------------------
# Heads: pointer masking, probabilities, temperature buffer
# --------------------------------------------------------------------------------------
def test_pointer_masking_invalid_slots_get_zero_probability_and_none_slot_always_valid(model):
    batch = make_batch(batch=4, length=96, seed=51)
    # Adversarial: claim *nothing* valid (incl. the NONE slots) in row 3.
    batch["pchunk_valid_mask"][3, :] = False
    batch["arg_valid_mask"][3, :] = False
    with torch.no_grad():
        noul_l, src_l, arg_l = call(model, batch, logits=True)
        noul_p, src_p, arg_p = call(model, batch)
    assert noul_l.shape == (4, 3) and src_l.shape == (4, 33) and arg_l.shape == (4, 17)
    for logits, probs, valid, none_slot in (
        (src_l, src_p, batch["pchunk_valid_mask"], 32),
        (arg_l, arg_p, batch["arg_valid_mask"], 16),
    ):
        eff_valid = valid.clone()
        eff_valid[:, none_slot] = True
        assert (logits[~eff_valid] == -10000.0).all()
        assert probs[~eff_valid].max().item() < 1e-6
        torch.testing.assert_close(probs.sum(-1), torch.ones(4), atol=1e-5, rtol=0)
        assert (probs[:, none_slot] > 0).all()
    # Row 3 has only the NONE slot open => it gets all the mass.
    assert src_p[3, 32].item() == pytest.approx(1.0, abs=1e-6)
    assert arg_p[3, 16].item() == pytest.approx(1.0, abs=1e-6)


def test_forward_equals_sigmoid_of_logits_over_temperature_and_softmax(model):
    batch = make_batch(batch=3, length=64, seed=61)
    assert model.logit_temperatures.dtype == torch.float32
    assert model.logit_temperatures.shape == (3,)
    assert torch.equal(model.logit_temperatures, torch.ones(3))
    with torch.no_grad():
        model.logit_temperatures.copy_(torch.tensor([0.8, 1.7, 2.5]))
        n_l, s_l, a_l = call(model, batch, logits=True)
        n_p, s_p, a_p = call(model, batch)
    torch.testing.assert_close(n_p, torch.sigmoid(n_l / torch.tensor([0.8, 1.7, 2.5])), atol=1e-6, rtol=0)
    torch.testing.assert_close(s_p, torch.softmax(s_l, -1), atol=1e-6, rtol=0)
    torch.testing.assert_close(a_p, torch.softmax(a_l, -1), atol=1e-6, rtol=0)
    # forward_logits is *uncalibrated* (T is applied only in forward / ONNX).
    n_l2, _, _ = call(model, batch, logits=True)
    torch.testing.assert_close(n_l2, n_l)


def test_logit_temperatures_in_state_dict_and_roundtrip(model):
    sd = model.state_dict()
    assert "logit_temperatures" in sd
    assert sd["logit_temperatures"].dtype == torch.float32
    assert model.get_buffer("logit_temperatures") is model.logit_temperatures
    model.set_logit_temperatures(torch.tensor([1.5, 2.0, 0.5]))
    sd2 = {k: v.clone() for k, v in model.state_dict().items()}
    fresh = type(model).from_config(model.backbone.config)
    assert torch.equal(fresh.logit_temperatures, torch.ones(3))
    fresh.load_state_dict(sd2)
    assert torch.equal(fresh.logit_temperatures, torch.tensor([1.5, 2.0, 0.5]))
    with pytest.raises(ValueError):
        model.set_logit_temperatures(torch.tensor([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        model.set_logit_temperatures(torch.tensor([1.0, 1.0]))


def test_eager_attention_is_required():
    cfg = ModernBertConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        pad_token_id=PAD_ID,
    )
    m = ModernBERTJevModel.from_config(cfg)
    assert m.backbone.config._attn_implementation == "eager"
    from transformers import ModernBertModel

    sdpa_backbone = ModernBertModel._from_config(cfg, attn_implementation="sdpa")
    with pytest.raises(ValueError, match="eager"):
        ModernBERTJevModel(sdpa_backbone)


def test_from_pretrained_roundtrip_from_local_dir_without_network(tmp_path, model):
    d = tmp_path / "bb"
    model.backbone.save_pretrained(str(d))
    loaded = type(model).from_pretrained(str(d), vocab_size=VOCAB + 4)
    assert loaded.backbone.config._attn_implementation == "eager"
    assert loaded.backbone.get_input_embeddings().num_embeddings == VOCAB + 4
    batch = make_batch(batch=2, length=64, seed=71)
    with torch.no_grad():
        outs = call(loaded, batch)
    assert all(torch.isfinite(o).all() for o in outs)


# --------------------------------------------------------------------------------------
# ONNX opset-17 export + ORT parity
# --------------------------------------------------------------------------------------
def _export(model, batch, path: Path) -> None:
    args = tuple(batch[name] for name in INPUT_NAMES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            args,
            str(path),
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=ONNX_DYNAMIC_AXES,
            opset_version=ONNX_OPSET_VERSION,
            dynamo=False,
        )
    # The tracer leaves symbolic `*_dim_1` output dims; pin the contract's static widths.
    proto = pin_onnx_output_shapes(onnx.load(str(path)))
    onnx.save(proto, str(path))


def _ort_run(path: Path, batch):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {name: batch[name].numpy() for name in INPUT_NAMES}
    return sess.run(list(OUTPUT_NAMES), feed)


def test_onnx_opset17_export_contract_ort_parity_and_determinism(tmp_path, model):
    with torch.no_grad():
        model.set_logit_temperatures(torch.tensor([1.3, 0.9, 1.1]))
    ex = make_batch(batch=2, length=128, seed=81)
    p1, p2 = tmp_path / "m1.onnx", tmp_path / "m2.onnx"
    _export(model, ex, p1)
    _export(model, ex, p2)
    assert hashlib.sha256(p1.read_bytes()).hexdigest() == hashlib.sha256(p2.read_bytes()).hexdigest()

    # Exact Track-B/ContractV1 signature (names, dtypes, ranks/static dims, opset 17).
    validate_onnx_contract(p1)
    graph = onnx.load(str(p1))
    assert [i.name for i in graph.graph.input] == list(INPUT_NAMES)
    assert [o.name for o in graph.graph.output] == list(OUTPUT_NAMES)
    default_opset = [i.version for i in graph.opset_import if i.domain in ("", "ai.onnx")]
    assert default_opset == [17]

    # T is a real graph initializer that A6 can rewrite after INT8 calibration.
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in graph.graph.initializer}
    assert "logit_temperatures" in inits
    np.testing.assert_allclose(inits["logit_temperatures"], [1.3, 0.9, 1.1], rtol=1e-6)

    # Head scopes are discoverable for A6's `nodes_to_exclude`.
    node_names = " ".join(n.name for n in graph.graph.node)
    for scope in HEAD_SCOPE_NAMES:
        assert f"/{scope}" in node_names, f"head scope {scope!r} missing from ONNX node names"

    # Parity at the export example AND at a different batch + sequence length (dynamic axes),
    # explicitly exercising STATE, QUERY, pchunk_positions, and arg_positions at indices >= 128.
    batch_192 = make_batch(batch=3, length=192, seed=85, n_state_min=150, n_state_max=180)
    assert bool((batch_192["query_positions"] >= 128).all())
    active_p192 = batch_192["pchunk_positions"][:, :32][batch_192["pchunk_valid_mask"][:, :32]]
    active_a192 = batch_192["arg_positions"][:, :16][batch_192["arg_valid_mask"][:, :16]]
    assert active_p192.numel() > 0 and bool((active_p192 >= 128).any())
    assert active_a192.numel() > 0 and bool((active_a192 >= 128).any())

    for batch in (ex, batch_192, make_batch(batch=1, length=96, seed=83)):
        with torch.no_grad():
            ref = call(model, batch)
        got = _ort_run(p1, batch)
        for r, g in zip(ref, got):
            assert g.dtype == np.float32
            assert np.abs(r.numpy() - g).max() < 1e-4

    # Padding-bucket invariance holds inside ORT too.
    a = _ort_run(p1, make_batch(batch=2, length=128, seed=84))
    b = _ort_run(p1, repad(make_batch(batch=2, length=128, seed=84), 256))
    for x, y in zip(a, b):
        assert np.abs(x - y).max() < 1e-5


# --------------------------------------------------------------------------------------
# Strictly proper losses
# --------------------------------------------------------------------------------------
def _loss_batch(n: int = 12, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    noul_logits = torch.randn(n, 3, generator=g) * 2
    src_logits = torch.randn(n, 33, generator=g)
    arg_logits = torch.randn(n, 17, generator=g)
    y = (torch.rand(n, 3, generator=g) > 0.5).float()
    src_valid = torch.zeros(n, 33, dtype=torch.bool)
    src_valid[:, :5] = True
    src_valid[:, 32] = True
    arg_valid = torch.zeros(n, 17, dtype=torch.bool)
    arg_valid[:, :3] = True
    arg_valid[:, 16] = True
    src_t = torch.randint(0, 5, (n,), generator=g)
    arg_t = torch.randint(0, 3, (n,), generator=g)
    benign = y.max(dim=1).values < 0.5
    src_t = torch.where(benign, torch.full_like(src_t, 32), src_t)
    arg_t = torch.where(benign, torch.full_like(arg_t, 16), arg_t)
    return noul_logits, src_logits, arg_logits, y, src_valid, arg_valid, src_t, arg_t


def test_default_loss_is_bce_plus_brier_with_documented_weights():
    cfg = JevLossConfig()
    assert cfg.bce_weight == 1.0 and cfg.brier_weight == 0.5
    assert cfg.head_weights == (1.0, 1.0, 1.0)
    # No knob exists that could shift targets or apply asymmetric reward weighting (I11).
    forbidden = {"label_smoothing", "pos_weight", "target_shift", "positive_weight", "reward_scale"}
    assert forbidden.isdisjoint(set(JevLossConfig.__dataclass_fields__))


def test_brier_and_bce_match_closed_forms_and_total_is_sum_of_parts():
    nl, sl, al, y, sv, av, st, at = _loss_batch()
    p = torch.sigmoid(nl)
    expected_brier = ((p - y) ** 2).mean()
    torch.testing.assert_close(brier_loss(nl, y).mean(), expected_brier)
    out = compute_jev_loss(nl, sl, al, y, st, at, sv, av, config=JevLossConfig(src_pointer_weight=0.7, arg_pointer_weight=0.3))
    bce = torch.nn.functional.binary_cross_entropy_with_logits(nl, y)
    torch.testing.assert_close(out["bce"], bce)
    torch.testing.assert_close(out["brier"], expected_brier)
    total = 1.0 * bce + 0.5 * expected_brier + 0.7 * out["src_pointer_ce"] + 0.3 * out["arg_pointer_ce"]
    torch.testing.assert_close(out["loss"], total)


def test_loss_rejects_targets_off_the_binary_lattice():
    nl, sl, al, y, sv, av, st, at = _loss_batch()
    for bad in (0.1, 0.9, 0.95, -0.01, 1.01):
        y_bad = y.clone()
        y_bad[0, 0] = bad
        with pytest.raises(ValueError, match="[{]0, 1[}]|binary"):
            compute_jev_loss(nl, sl, al, y_bad, st, at, sv, av)


def test_pointer_ce_is_masked_to_valid_slots_and_ignores_ignore_index():
    nl, sl, al, y, sv, av, st, at = _loss_batch()
    # Huge logits on invalid slots must not influence the loss (masked before the softmax).
    sl_poison = sl.clone()
    sl_poison[~sv] = 1e4
    a = masked_pointer_cross_entropy(sl, st, sv)
    b = masked_pointer_cross_entropy(sl_poison, st, sv)
    torch.testing.assert_close(a, b)
    # Manual reference over the valid slots.
    masked = sl.masked_fill(~sv, float("-inf"))
    ref = torch.nn.functional.cross_entropy(masked, st)
    torch.testing.assert_close(a, ref)
    # ignore_index rows contribute nothing; all-ignored => exact zero with grad support.
    st_ign = st.clone()
    st_ign[:6] = -100
    c = masked_pointer_cross_entropy(sl, st_ign, sv)
    torch.testing.assert_close(c, torch.nn.functional.cross_entropy(masked[6:], st[6:]))
    z = masked_pointer_cross_entropy(sl.requires_grad_(True), torch.full_like(st, -100), sv)
    assert z.item() == 0.0
    z.backward()
    # A target pointing at an invalid slot is a data bug and must fail loudly.
    st_bad = st.clone()
    st_bad[0] = 20
    with pytest.raises(ValueError, match="invalid"):
        masked_pointer_cross_entropy(sl, st_bad, sv)


def test_sample_weights_hook_scales_per_row_per_head_losses():
    nl, sl, al, y, sv, av, st, at = _loss_batch()
    base = compute_jev_loss(nl, sl, al, y, st, at, sv, av)
    ones = compute_jev_loss(nl, sl, al, y, st, at, sv, av, sample_weights=torch.ones(12))
    torch.testing.assert_close(base["loss"], ones["loss"])
    w = torch.ones(12)
    w[:4] = 3.0  # hard-negative mining
    weighted = compute_jev_loss(nl, sl, al, y, st, at, sv, av, sample_weights=w)
    assert weighted["loss"].item() != pytest.approx(base["loss"].item())
    per_head = compute_jev_loss(
        nl, sl, al, y, st, at, sv, av, sample_weights=torch.ones(12, 3) * torch.tensor([1.0, 2.0, 0.0])
    )
    assert per_head["bce_per_head"].shape == (3,)
    assert per_head["bce_per_head"][2].item() == 0.0  # zero-weighted head contributes nothing
    with pytest.raises(ValueError, match="non-negative"):
        compute_jev_loss(nl, sl, al, y, st, at, sv, av, sample_weights=-torch.ones(12))


def test_loss_is_strictly_proper_minimiser_is_the_true_probability():
    """BCE and Brier are both minimised in expectation at p = P(y=1) (no bias towards 0/1)."""
    true_p = 0.2
    y = (torch.rand(200_000, generator=torch.Generator().manual_seed(1)) < true_p).float()[:, None].repeat(1, 3)
    grid = torch.linspace(0.02, 0.98, 97)
    scores_bce, scores_brier = [], []
    for p in grid:
        logit = torch.logit(p).expand(y.shape[0], 3)
        scores_bce.append(torch.nn.functional.binary_cross_entropy_with_logits(logit, y).item())
        scores_brier.append(brier_loss(logit, y).mean().item())
    assert abs(grid[int(np.argmin(scores_bce))].item() - true_p) < 0.03
    assert abs(grid[int(np.argmin(scores_brier))].item() - true_p) < 0.03


# --------------------------------------------------------------------------------------
# Post-hoc temperature scaling + ECE
# --------------------------------------------------------------------------------------
def _overconfident_logits(n=20000, temp=3.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    true_logit = torch.randn(n, 3, generator=g) * 1.5
    y = (torch.rand(n, 3, generator=g) < torch.sigmoid(true_logit)).float()
    return true_logit * temp, y


def test_expected_calibration_error_basics():
    probs = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    perfect = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0.0])  # 0.9 bin acc 1.0? use constructed counts below
    # Construct exactly calibrated bins: 10 samples @0.8 with 8 positives.
    p = torch.full((10,), 0.8)
    t = torch.tensor([1.0] * 8 + [0.0] * 2)
    assert expected_calibration_error(p, t) == pytest.approx(0.0, abs=1e-6)
    assert expected_calibration_error(torch.ones(10), torch.zeros(10)) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= expected_calibration_error(probs, perfect) <= 1.0
    assert expected_calibration_error(torch.rand(500), (torch.rand(500) > 0.5).float(), num_bins=15) >= 0.0


def test_fit_temperature_scaling_recovers_temperature_and_improves_ece_nll():
    logits, y = _overconfident_logits(temp=3.0)
    T = fit_temperature_scaling(logits, y)
    assert T.dtype == torch.float32 and T.shape == (3,)
    assert torch.all(torch.abs(T - 3.0) / 3.0 < 0.15), T
    rep = temperature_calibration_report(logits, y, T)
    for h in range(3):
        assert rep["ece_after"][h] < rep["ece_before"][h]
        assert rep["nll_after"][h] < rep["nll_before"][h]
    assert rep["ece_val"] == pytest.approx(float(np.mean(rep["ece_after"])))
    payload = rep["calibration"]
    assert payload["method"] == "temperature_scaling"
    assert payload["temperatures"] == pytest.approx(T.tolist())
    assert payload["num_bins"] == 15 and 0.0 <= payload["ece_val"] <= 1.0


def test_fit_temperature_scaling_is_per_head_and_degenerate_head_falls_back_to_one():
    logits, y = _overconfident_logits(temp=1.0, seed=3)
    logits[:, 0] *= 2.0  # head 0 overconfident x2
    logits[:, 1] *= 0.5  # head 1 underconfident
    y[:, 2] = 0.0  # head 2: single-class => unidentifiable (T -> 0) => must return 1.0
    T = fit_temperature_scaling(logits, y)
    assert T[0] > 1.5 and T[1] < 0.7
    assert T[2].item() == 1.0
    with pytest.raises(ValueError):
        fit_temperature_scaling(logits, y[:, :2])


def test_temperature_roundtrip_through_probs_is_what_a6_needs():
    logits, _ = _overconfident_logits(n=1000, temp=2.0)
    logits = logits.clamp(-6.0, 6.0)  # float32 probs saturate beyond |z| ~ 12 (documented in probs_to_logits)
    T = torch.tensor([1.5, 2.0, 0.7])
    probs = torch.sigmoid(logits / T)
    back = probs_to_logits(probs, T)
    torch.testing.assert_close(back, logits, atol=2e-3, rtol=2e-3)
    # With T baked in the ONNX probs, A6 recovers raw logits to re-fit on INT8 outputs.
    assert torch.isfinite(probs_to_logits(torch.tensor([[0.0, 1.0, 0.5]]))).all()


def test_fitted_temperatures_plug_into_model_buffer(model):
    logits, y = _overconfident_logits(n=4000, temp=2.5, seed=9)
    T = fit_temperature_scaling(logits, y)
    model.set_logit_temperatures(T)
    torch.testing.assert_close(model.logit_temperatures, T)


# --------------------------------------------------------------------------------------
# Fix Round 1: causal span pooling, sample_weights validation, pin_onnx_output_shapes,
# extreme-logit report safety, and MaskedLM/CausalLM offline checkpoint loading
# --------------------------------------------------------------------------------------
def _make_golden_packer() -> tuple[Any, dict[str, int], int]:
    from tokenizers import Tokenizer

    from diodemodel.contract import BundleManifest, REQUIRED_SPECIAL_TOKENS
    from diodemodel.dataset_packer import JevDatasetPacker

    tok_path = Path(__file__).parent / "fixtures" / "golden_tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    sp = {t: int(tok.token_to_id(t)) for t in REQUIRED_SPECIAL_TOKENS}
    zero_sha = "0" * 64
    manifest = BundleManifest(
        model_id="test-nanojev-packer",
        sha256_model_onnx=zero_sha,
        sha256_tokenizer_json=zero_sha,
        sha256_calibration_json=zero_sha,
        special_token_ids=sp,
        ece_val=0.01,
    )
    return JevDatasetPacker(tok, manifest), sp, int(tok.get_vocab_size())


def _packed_row_to_torch(row) -> dict[str, torch.Tensor]:
    feed = row.as_ort_feed()
    return {k: torch.from_numpy(v) for k, v in feed.items()}


def test_nanojev_causal_span_pooling_on_packer_multi_chunk_and_multi_arg_pytorch_and_ort(tmp_path):
    from diodemodel.mutators import SpanItem

    packer, sp, vocab_sz = _make_golden_packer()
    torch.manual_seed(9876)
    cfg = Qwen3Config(
        vocab_size=max(128, vocab_sz + 8),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=2048,
        pad_token_id=sp["[PAD]"],
        bos_token_id=sp["[CLS]"],
        eos_token_id=sp["[SEP]"],
    )
    nano = NanoJevCausalModel.from_config(cfg).eval()

    def _pack(
        chunks: tuple[str, str, str],
        args_map: dict[str, str],
        *,
        max_bucket: int = 512,
        pad_to_max_bucket: bool = True,
    ):
        priv_spans = [
            SpanItem(span_id="p0", source_label="env", text=chunks[0], is_high_secret=True),
            SpanItem(span_id="p1", source_label="memo", text=chunks[1], dual_trust=True),
            SpanItem(span_id="p2", source_label="db", text=chunks[2], is_high_secret=False),
        ]
        untrust_spans = [
            SpanItem(span_id="u0", source_label="web", text="external untrusted README snippet"),
        ]
        return packer.pack_raw(
            intent="Summarize quarterly planning thread locally",
            private_spans=priv_spans,
            untrusted_spans=untrust_spans,
            call_server="github",
            call_tool="create_issue",
            call_arguments=args_map,
            y_enc=1.0,
            y_sem=0.0,
            y_hijack=1.0,
            attributed_pchunk_idx=1,
            attributed_arg_key="title",
            max_bucket=max_bucket,
            pad_to_max_bucket=pad_to_max_bucket,
        )

    base_chunks = (
        "stripe secret key alpha",
        "ProjectHelios LOI signed cash",
        "billing calc price tier enterprise",
    )
    base_args = {
        "body": "alpha secret key",
        "title": "billing tier price",
        "url": "github / repo / prod",
    }
    base_row = _pack(base_chunks, base_args, max_bucket=512, pad_to_max_bucket=True)
    assert sum(base_row.pchunk_valid_mask[:32]) == 3
    assert sum(base_row.arg_valid_mask[:16]) == 3
    base_batch = _packed_row_to_torch(base_row)

    onnx_path = tmp_path / "nano_span_pool.onnx"
    _export(nano, base_batch, onnx_path)
    validate_onnx_contract(onnx_path)

    with torch.no_grad():
        base_h = nano.encode(base_batch["input_ids"], base_batch["seg_ids"], base_batch["token_roles"])
        base_src_slots, base_arg_slots = nano.extract_pointer_slot_states(
            base_h,
            base_batch["seg_ids"],
            base_batch["token_roles"],
            base_batch["pchunk_positions"],
            base_batch["pchunk_valid_mask"],
            base_batch["arg_positions"],
            base_batch["arg_valid_mask"],
        )
        base_src_keys = nano.src_pointer.k_proj(base_src_slots)
        base_arg_keys = nano.arg_pointer.k_proj(base_arg_slots)
        _, base_src_l, base_arg_l = call(nano, base_batch, logits=True)
        base_pt_probs = call(nano, base_batch)
    base_ort_probs = _ort_run(onnx_path, base_batch)
    for r, g in zip(base_pt_probs, base_ort_probs):
        assert np.abs(r.numpy() - g).max() < 1e-4

    # 1. Private chunk content sensitivity & causal invariance across all 3 chunks (PyTorch + ORT).
    alt_chunk_texts = (
        "beta gamma delta custom discount",
        "open issue login timeout filler",
        "SELECT COUNT DROP TABLE query",
    )
    chunk_logit_gaps: list[float] = []
    chunk_ort_prob_gaps: list[float] = []
    for i in range(3):
        mod_chunks = list(base_chunks)
        mod_chunks[i] = alt_chunk_texts[i]
        pert_row = _pack(tuple(mod_chunks), base_args, max_bucket=512, pad_to_max_bucket=True)  # type: ignore[arg-type]
        pert_batch = _packed_row_to_torch(pert_row)
        assert not torch.equal(pert_batch["input_ids"], base_batch["input_ids"])
        with torch.no_grad():
            h_i = nano.encode(pert_batch["input_ids"], pert_batch["seg_ids"], pert_batch["token_roles"])
            src_slots_i, _ = nano.extract_pointer_slot_states(
                h_i,
                pert_batch["seg_ids"],
                pert_batch["token_roles"],
                pert_batch["pchunk_positions"],
                pert_batch["pchunk_valid_mask"],
                pert_batch["arg_positions"],
                pert_batch["arg_valid_mask"],
            )
            src_keys_i = nano.src_pointer.k_proj(src_slots_i)
            _, src_l_i, _ = call(nano, pert_batch, logits=True)
            pt_probs_i = call(nano, pert_batch)
        ort_probs_i = _ort_run(onnx_path, pert_batch)
        for r, g in zip(pt_probs_i, ort_probs_i):
            assert np.abs(r.numpy() - g).max() < 1e-4

        if i > 0:
            assert (src_keys_i[:, :i] - base_src_keys[:, :i]).abs().max().item() < 1e-7
        assert (src_keys_i[:, i] - base_src_keys[:, i]).abs().max().item() > 1e-4
        assert (src_l_i[:, i] - base_src_l[:, i]).abs().item() > 1e-4
        if i in (0, 1):
            chunk_logit_gaps.append(float((src_l_i[0, 1] - src_l_i[0, 0]).item()))
            chunk_ort_prob_gaps.append(float(ort_probs_i[1][0, 1] - ort_probs_i[1][0, 0]))
    assert abs(chunk_logit_gaps[1] - chunk_logit_gaps[0]) > 1e-4
    assert abs(chunk_ort_prob_gaps[1] - chunk_ort_prob_gaps[0]) > 1e-5

    # Closing [SEP] of private section is excluded from chunk 2's pooled span; <|none|> stays at none_pos.
    priv_sep_pos = int(
        (
            torch.arange(base_batch["input_ids"].shape[1])
            * ((base_batch["token_roles"][0] == ROLE_STATE) & ((base_batch["seg_ids"][0] == 1) | (base_batch["seg_ids"][0] == 2))).long()
        ).amax()
    )
    assert int(base_batch["input_ids"][0, priv_sep_pos]) == sp["[SEP]"]
    sep_pert = {k: v.clone() for k, v in base_batch.items()}
    sep_pert["input_ids"][0, priv_sep_pos] = 25
    with torch.no_grad():
        h_sep = nano.encode(sep_pert["input_ids"], sep_pert["seg_ids"], sep_pert["token_roles"])
        src_slots_sep, _ = nano.extract_pointer_slot_states(
            h_sep,
            sep_pert["seg_ids"],
            sep_pert["token_roles"],
            sep_pert["pchunk_positions"],
            sep_pert["pchunk_valid_mask"],
            sep_pert["arg_positions"],
            sep_pert["arg_valid_mask"],
        )
    assert (src_slots_sep[:, :3] - base_src_slots[:, :3]).abs().max().item() < 1e-7
    torch.testing.assert_close(base_src_slots[:, 32], base_h[:, base_row.none_position])

    # 2. Call argument content sensitivity & causal invariance across all 3 args (PyTorch + ORT).
    arg_names = ("body", "title", "url")
    alt_arg_vals = (
        "ProjectHelios LOI signed cash",
        "DROP TABLE SELECT COUNT",
        "slack / channel / open",
    )
    arg_logit_gaps: list[float] = []
    arg_ort_prob_gaps: list[float] = []
    for j, key_name in enumerate(arg_names):
        mod_args = dict(base_args)
        mod_args[key_name] = alt_arg_vals[j]
        pert_row = _pack(base_chunks, mod_args, max_bucket=512, pad_to_max_bucket=True)
        pert_batch = _packed_row_to_torch(pert_row)
        assert not torch.equal(pert_batch["input_ids"], base_batch["input_ids"])
        with torch.no_grad():
            h_j = nano.encode(pert_batch["input_ids"], pert_batch["seg_ids"], pert_batch["token_roles"])
            _, arg_slots_j = nano.extract_pointer_slot_states(
                h_j,
                pert_batch["seg_ids"],
                pert_batch["token_roles"],
                pert_batch["pchunk_positions"],
                pert_batch["pchunk_valid_mask"],
                pert_batch["arg_positions"],
                pert_batch["arg_valid_mask"],
            )
            arg_keys_j = nano.arg_pointer.k_proj(arg_slots_j)
            _, _, arg_l_j = call(nano, pert_batch, logits=True)
            pt_probs_j = call(nano, pert_batch)
        ort_probs_j = _ort_run(onnx_path, pert_batch)
        for r, g in zip(pt_probs_j, ort_probs_j):
            assert np.abs(r.numpy() - g).max() < 1e-4

        if j > 0:
            assert (arg_keys_j[:, :j] - base_arg_keys[:, :j]).abs().max().item() < 1e-7
        assert (arg_keys_j[:, j] - base_arg_keys[:, j]).abs().max().item() > 1e-4
        assert (arg_l_j[:, j] - base_arg_l[:, j]).abs().item() > 1e-4
        if j in (0, 1):
            arg_logit_gaps.append(float((arg_l_j[0, 1] - arg_l_j[0, 0]).item()))
            arg_ort_prob_gaps.append(float(ort_probs_j[2][0, 1] - ort_probs_j[2][0, 0]))
    assert abs(arg_logit_gaps[1] - arg_logit_gaps[0]) > 1e-4
    assert abs(arg_ort_prob_gaps[1] - arg_ort_prob_gaps[0]) > 1e-5

    # Closing [SEP] before <|none|> is excluded from arg 2's pooled span; <|none|> stays at none_pos.
    call_sep_pos = base_row.none_position - 1
    assert int(base_batch["input_ids"][0, call_sep_pos]) == sp["[SEP]"]
    csep_pert = {k: v.clone() for k, v in base_batch.items()}
    csep_pert["input_ids"][0, call_sep_pos] = 26
    with torch.no_grad():
        h_csep = nano.encode(csep_pert["input_ids"], csep_pert["seg_ids"], csep_pert["token_roles"])
        _, arg_slots_csep = nano.extract_pointer_slot_states(
            h_csep,
            csep_pert["seg_ids"],
            csep_pert["token_roles"],
            csep_pert["pchunk_positions"],
            csep_pert["pchunk_valid_mask"],
            csep_pert["arg_positions"],
            csep_pert["arg_valid_mask"],
        )
    assert (arg_slots_csep[:, :3] - base_arg_slots[:, :3]).abs().max().item() < 1e-7
    torch.testing.assert_close(base_arg_slots[:, 16], base_h[:, base_row.none_position])

    # 3a. Query isolation (< 1e-7) on the real packer sequence in BOTH PyTorch and ORT.
    with torch.no_grad():
        base_logits_pack = call(nano, base_batch, logits=True)
    for q_idx in range(5):
        q_pos = int(base_batch["query_positions"][0, q_idx])
        q_pert = _perturb_token(base_batch, 0, q_pos, 30 + q_idx)
        with torch.no_grad():
            q_logits = call(nano, q_pert, logits=True)
        q_ort = _ort_run(onnx_path, q_pert)
        for k in range(3):
            if k != q_idx:
                assert (q_logits[0][0, k] - base_logits_pack[0][0, k]).abs().item() < 1e-7
                assert abs(float(q_ort[0][0, k] - base_ort_probs[0][0, k])) < 1e-7
        if q_idx != 3:
            assert (q_logits[1][0] - base_logits_pack[1][0]).abs().max().item() < 1e-7
            assert np.abs(q_ort[1][0] - base_ort_probs[1][0]).max() < 1e-7
        if q_idx != 4:
            assert (q_logits[2][0] - base_logits_pack[2][0]).abs().max().item() < 1e-7
            assert np.abs(q_ort[2][0] - base_ort_probs[2][0]).max() < 1e-7

    # 3b. Bucket padding invariance (512 vs 1024, < 1e-6) in BOTH PyTorch and ORT.
    row_1024 = _pack(base_chunks, base_args, max_bucket=1024, pad_to_max_bucket=True)
    assert base_row.bucket_length == 512 and row_1024.bucket_length == 1024
    batch_1024 = _packed_row_to_torch(row_1024)
    with torch.no_grad():
        logits_512 = call(nano, base_batch, logits=True)
        logits_1024 = call(nano, batch_1024, logits=True)
        probs_512 = call(nano, base_batch)
        probs_1024 = call(nano, batch_1024)
    for x, y in list(zip(logits_512, logits_1024)) + list(zip(probs_512, probs_1024)):
        assert (x - y).abs().max().item() < 1e-6
    ort_1024 = _ort_run(onnx_path, batch_1024)
    for x, y in zip(base_ort_probs, ort_1024):
        assert np.abs(x - y).max() < 1e-6


def test_sample_weights_and_pointer_sample_weights_validation_and_device_normalisation():
    from unittest.mock import patch

    nl, sl, al, y, sv, av, st, at = _loss_batch(n=8, seed=12)

    # masked_pointer_cross_entropy must reject 2D shapes (which would broadcast to [N, N]),
    # wrong lengths, negative values, NaN, and Inf.
    for bad_sw in (
        torch.ones(8, 1),
        torch.ones(8, 3),
        torch.ones(9),
        torch.tensor([-0.1] + [1.0] * 7),
        torch.tensor([float("nan")] + [1.0] * 7),
        torch.tensor([float("inf")] + [1.0] * 7),
    ):
        with pytest.raises(ValueError):
            masked_pointer_cross_entropy(sl, st, sv, sample_weights=bad_sw)

    # Valid 1D sample_weights on masked_pointer_cross_entropy matches manual row-weighted CE.
    w1d = torch.tensor([2.0, 0.5, 1.0, 3.0, 0.0, 1.5, 2.5, 1.0])
    got_ce = masked_pointer_cross_entropy(sl, st, sv, sample_weights=w1d)
    per_row = torch.nn.functional.cross_entropy(sl.masked_fill(~sv, float("-inf")), st, reduction="none")
    ref_ce = (per_row * w1d).sum() / w1d.sum()
    torch.testing.assert_close(got_ce, ref_ce)

    # compute_jev_loss must reject invalid sample_weights AND invalid pointer_sample_weights.
    for bad_sw in (
        torch.ones(8, 1),
        torch.ones(8, 2),
        torch.tensor([[float("nan"), 1.0, 1.0]] + [[1.0, 1.0, 1.0]] * 7),
        torch.tensor([[float("inf"), 1.0, 1.0]] + [[1.0, 1.0, 1.0]] * 7),
    ):
        with pytest.raises(ValueError):
            compute_jev_loss(nl, sl, al, y, st, at, sv, av, sample_weights=bad_sw)

    for bad_psw in (
        torch.ones(8, 1),
        torch.ones(8, 3),
        torch.ones(7),
        torch.tensor([-1.0] + [1.0] * 7),
        torch.tensor([float("nan")] + [1.0] * 7),
        torch.tensor([float("-inf")] + [1.0] * 7),
    ):
        with pytest.raises(ValueError):
            compute_jev_loss(nl, sl, al, y, st, at, sv, av, pointer_sample_weights=bad_psw)

    # Verify explicit device=... normalisation on sample_weights and pointer_sample_weights.
    orig_to = torch.Tensor.to
    seen_devices: list[torch.device] = []

    def _spy_to(self, *args, **kwargs):
        if "device" in kwargs and kwargs["device"] is not None:
            seen_devices.append(torch.device(kwargs["device"]))
        elif args and isinstance(args[0], torch.device):
            seen_devices.append(args[0])
        return orig_to(self, *args, **kwargs)

    with patch.object(torch.Tensor, "to", _spy_to):
        seen_devices.clear()
        masked_pointer_cross_entropy(sl, st, sv, sample_weights=torch.ones(8))
        assert sl.device in seen_devices

        seen_devices.clear()
        compute_jev_loss(
            nl,
            sl,
            al,
            y,
            st,
            at,
            sv,
            av,
            sample_weights=torch.ones(8, 3),
            pointer_sample_weights=torch.ones(8),
        )
        assert len(seen_devices) >= 2


def test_pin_onnx_output_shapes_rejects_malformed_rank_or_static_width():
    from onnx import TensorProto, helper

    def _build_dummy_onnx(actual_widths: tuple[int, int, int], declared_shapes: tuple[list[Any], ...]):
        in_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", max(actual_widths)])
        nodes = []
        outputs = []
        for idx, (name, act_w, decl_shape) in enumerate(
            zip(("noul_probs", "src_choice_probs", "arg_choice_probs"), actual_widths, declared_shapes, strict=True)
        ):
            starts = helper.make_tensor(f"s_{idx}", TensorProto.INT64, [1], [0])
            ends = helper.make_tensor(f"e_{idx}", TensorProto.INT64, [1], [act_w])
            axes = helper.make_tensor(f"a_{idx}", TensorProto.INT64, [1], [1])
            nodes.append(helper.make_node("Constant", [], [f"sc_{idx}"], value=starts))
            nodes.append(helper.make_node("Constant", [], [f"ec_{idx}"], value=ends))
            nodes.append(helper.make_node("Constant", [], [f"ac_{idx}"], value=axes))
            nodes.append(helper.make_node("Slice", ["x", f"sc_{idx}", f"ec_{idx}", f"ac_{idx}"], [name]))
            outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, decl_shape))
        graph = helper.make_graph(nodes, "test_pin", [in_x], outputs)
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    # Valid symbolic widths whose inferred shapes are 3 / 33 / 17 succeed.
    ok_model = _build_dummy_onnx(
        (3, 33, 17),
        (["batch", "noul_d1"], ["batch", "src_d1"], ["batch", "arg_d1"]),
    )
    pinned = pin_onnx_output_shapes(ok_model)
    assert pinned is ok_model
    assert [pinned.graph.output[i].type.tensor_type.shape.dim[1].dim_value for i in range(3)] == [3, 33, 17]

    # Concrete wrong width on output proto raises ValueError.
    bad_concrete = _build_dummy_onnx((3, 32, 17), (["batch", 3], ["batch", 32], ["batch", 17]))
    with pytest.raises(ValueError, match="src_choice_probs|33|32|width|dim"):
        pin_onnx_output_shapes(bad_concrete)

    # Symbolic dim_param hiding an inferred wrong width (32 != 33) raises ValueError via shape_inference.
    bad_symbolic = _build_dummy_onnx(
        (3, 32, 17),
        (["batch", "d0"], ["batch", "d1"], ["batch", "d2"]),
    )
    with pytest.raises(ValueError, match="src_choice_probs|33|32|width|dim"):
        pin_onnx_output_shapes(bad_symbolic)

    # Wrong rank (rank 3 or rank 1) raises ValueError.
    bad_rank = _build_dummy_onnx((3, 33, 17), (["batch", 3, 1], ["batch", 33], ["batch", 17]))
    with pytest.raises(ValueError, match="rank|noul_probs"):
        pin_onnx_output_shapes(bad_rank)


def test_temperature_calibration_report_extreme_logits_no_overflow_warning():
    logits = torch.tensor(
        [
            [-1000.0, -90.0, 1000.0],
            [-800.0, 95.0, -1200.0],
            [900.0, -110.0, 800.0],
            [0.0, -75.0, 0.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    temps = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)  # z / T reaches -10000 and -900 (< -710)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rep = temperature_calibration_report(logits, targets, temps)
    assert 0.0 <= rep["ece_val"] <= 1.0
    assert all(np.isfinite(rep["ece_before"])) and all(np.isfinite(rep["ece_after"]))
    assert all(np.isfinite(rep["nll_before"])) and all(np.isfinite(rep["nll_after"]))


def test_from_pretrained_loads_masked_lm_and_causal_lm_checkpoints_bit_for_bit(tmp_path):
    from transformers import ModernBertForMaskedLM, Qwen3ForCausalLM

    assert ModernBertForMaskedLM.base_model_prefix == "model"
    assert Qwen3ForCausalLM.base_model_prefix == "model"

    torch.manual_seed(2026)
    mb_cfg = ModernBertConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        local_attention=8,
        max_position_embeddings=512,
        pad_token_id=PAD_ID,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    mb_mlm = ModernBertForMaskedLM(mb_cfg).eval()
    mb_dir = tmp_path / "modernbert_mlm"
    mb_mlm.save_pretrained(str(mb_dir))

    mb_jev = ModernBERTJevModel.from_pretrained(str(mb_dir)).eval()
    mb_ref_sd = mb_mlm.model.state_dict()
    mb_got_sd = mb_jev.backbone.state_dict()
    assert set(mb_ref_sd.keys()) == set(mb_got_sd.keys())
    for k, ref_t in mb_ref_sd.items():
        assert torch.equal(ref_t, mb_got_sd[k]), f"ModernBERT backbone weight mismatch on {k}"

    # Growing embeddings for the 17 Jev special tokens preserves pretrained rows 0..VOCAB-1 bit-for-bit.
    mb_jev_grown = ModernBERTJevModel.from_pretrained(str(mb_dir), vocab_size=VOCAB + 17).eval()
    assert mb_jev_grown.backbone.get_input_embeddings().num_embeddings == VOCAB + 17
    assert torch.equal(
        mb_jev_grown.backbone.get_input_embeddings().weight[:VOCAB],
        mb_mlm.model.get_input_embeddings().weight,
    )

    qw_cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        pad_token_id=PAD_ID,
        bos_token_id=1,
        eos_token_id=2,
    )
    qw_clm = Qwen3ForCausalLM(qw_cfg).eval()
    qw_dir = tmp_path / "qwen3_clm"
    qw_clm.save_pretrained(str(qw_dir))

    qw_jev = NanoJevCausalModel.from_pretrained(str(qw_dir)).eval()
    qw_ref_sd = qw_clm.model.state_dict()
    qw_got_sd = qw_jev.backbone.state_dict()
    assert set(qw_ref_sd.keys()) == set(qw_got_sd.keys())
    for k, ref_t in qw_ref_sd.items():
        assert torch.equal(ref_t, qw_got_sd[k]), f"Qwen3 backbone weight mismatch on {k}"


# --------------------------------------------------------------------------------------
# Task A5: Curriculum Trainer (`train.py`), Checkpoint Resume, Metrics & Colab Notebook
# --------------------------------------------------------------------------------------
def _tiny_curriculum_modernbert() -> ModernBERTJevModel:
    torch.manual_seed(2027)
    cfg = ModernBertConfig(
        vocab_size=VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        local_attention=8,
        max_position_embeddings=4096,
        pad_token_id=PAD_ID,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    return ModernBERTJevModel.from_config(cfg)


def _tiny_curriculum_nano() -> NanoJevCausalModel:
    torch.manual_seed(2028)
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=4096,
        pad_token_id=PAD_ID,
    )
    return NanoJevCausalModel.from_config(cfg)


def test_parquet_and_in_memory_dataset_batch_loading_and_dynamic_bucket_padding(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from diodemodel.dataset_packer import PACKED_PARQUET_SCHEMA
    from diodemodel.train import (
        CURRICULUM_BUCKETS,
        align_row_to_bucket,
        collate_jev_batch,
        load_packed_jev_dataset,
        make_synthetic_curriculum_dataset,
    )

    assert CURRICULUM_BUCKETS == (512, 1024, 4096)
    rows = make_synthetic_curriculum_dataset(16, vocab_size=VOCAB, seed=42, pad_token_id=PAD_ID)
    assert len(rows) == 16

    # Write to Parquet using the normative PACKED_PARQUET_SCHEMA and reload.
    pq_path = tmp_path / "synth_train.parquet"
    pq_table = pa.Table.from_pylist(rows, schema=PACKED_PARQUET_SCHEMA)
    pq.write_table(pq_table, pq_path)
    loaded = load_packed_jev_dataset(pq_path)
    assert len(loaded) == 16

    # Also test loading in-memory rows using alias keys (src_target, arg_target, threat_category, variant_kind).
    alias_row = {
        "input_ids": list(rows[0]["input_ids"]),
        "seg_ids": list(rows[0]["seg_ids"]),
        "token_roles": list(rows[0]["token_roles"]),
        "query_positions": list(rows[0]["query_positions"]),
        "pchunk_positions": list(rows[0]["pchunk_positions"]),
        "pchunk_valid_mask": list(rows[0]["pchunk_valid_mask"]),
        "arg_positions": list(rows[0]["arg_positions"]),
        "arg_valid_mask": list(rows[0]["arg_valid_mask"]),
        "y_enc": rows[0]["y_enc"],
        "y_sem": rows[0]["y_sem"],
        "y_hijack": rows[0]["y_hijack"],
        "src_target": rows[0]["target_src_idx"],
        "arg_target": rows[0]["target_arg_idx"],
        "threat_category": "T2",
        "variant_kind": "BENIGN_TWIN",
    }
    alias_loaded = load_packed_jev_dataset([alias_row])
    assert alias_loaded[0]["threat_category"] == "T2"
    assert alias_loaded[0]["category"] == "T2"
    assert alias_loaded[0]["variant_kind"] == "BENIGN_TWIN"
    assert alias_loaded[0]["attack_family"] == "BENIGN_TWIN"
    assert alias_loaded[0]["src_target"] == rows[0]["target_src_idx"]
    assert alias_loaded[0]["arg_target"] == rows[0]["target_arg_idx"]

    # Test dynamic bucket padding & clamping across 512, 1024, 4096.
    for bucket_len in CURRICULUM_BUCKETS:
        batch = collate_jev_batch(loaded[:6], bucket_len=bucket_len, pad_token_id=PAD_ID)
        assert batch["input_ids"].shape == (6, bucket_len)
        assert batch["seg_ids"].shape == (6, bucket_len)
        assert batch["token_roles"].shape == (6, bucket_len)
        assert batch["query_positions"].shape == (6, 5)
        assert batch["pchunk_positions"].shape == (6, 33)
        assert batch["pchunk_valid_mask"].shape == (6, 33)
        assert batch["arg_positions"].shape == (6, 17)
        assert batch["arg_valid_mask"].shape == (6, 17)
        assert batch["noul_targets"].shape == (6, 3)
        assert batch["y_enc"].shape == (6,)
        assert batch["y_sem"].shape == (6,)
        assert batch["y_hijack"].shape == (6,)
        assert batch["src_target"].shape == (6,)
        assert batch["arg_target"].shape == (6,)
        assert torch.equal(batch["src_target"], batch["target_src_idx"])
        assert torch.equal(batch["arg_target"], batch["target_arg_idx"])
        assert len(batch["threat_category"]) == 6
        assert len(batch["variant_kind"]) == 6

        # Every position must be within [0, bucket_len - 1] and query positions must have ROLE_QUERY & seg_id 5.
        assert bool((batch["query_positions"] >= 0).all() and (batch["query_positions"] < bucket_len).all())
        assert bool((batch["pchunk_positions"] >= 0).all() and (batch["pchunk_positions"] < bucket_len).all())
        assert bool((batch["arg_positions"] >= 0).all() and (batch["arg_positions"] < bucket_len).all())
        for b in range(6):
            for qp in batch["query_positions"][b].tolist():
                assert int(batch["token_roles"][b, qp]) == ROLE_QUERY
                assert int(batch["seg_ids"][b, qp]) == CANONICAL_SEGMENT_IDS["DECISION_QUERY"]
            # Active pointer targets must always point to valid slots.
            st = int(batch["src_target"][b].item())
            at = int(batch["arg_target"][b].item())
            assert bool(batch["pchunk_valid_mask"][b, st].item())
            assert bool(batch["arg_valid_mask"][b, at].item())

    # Test clamping a long row (unpadded_length=1200 > 512) whose attributed pchunk/arg sit at > 800.
    long_batch_raw = make_batch(batch=1, length=1200, seed=99, n_state_min=1100, n_state_max=1150)
    long_row = {
        "input_ids": long_batch_raw["input_ids"][0].tolist(),
        "seg_ids": long_batch_raw["seg_ids"][0].tolist(),
        "token_roles": long_batch_raw["token_roles"][0].tolist(),
        "query_positions": long_batch_raw["query_positions"][0].tolist(),
        "pchunk_positions": [100, 900] + [1120] * 31,
        "pchunk_valid_mask": [True, True] + [False] * 30 + [True],
        "arg_positions": [120, 950] + [1120] * 15,
        "arg_valid_mask": [True, True] + [False] * 14 + [True],
        "y_enc": 1.0,
        "y_sem": 0.0,
        "y_hijack": 0.0,
        "src_target": 1,  # slot 1 is at pos 900 (>= 512), so clamping to 512 evicts it!
        "arg_target": 1,  # slot 1 is at pos 950 (>= 512), so clamping to 512 evicts it!
        "threat_category": "T1",
        "variant_kind": "prog_t1_leak",
    }
    aligned_512 = align_row_to_bucket(long_row, 512, pad_token_id=PAD_ID)
    assert len(aligned_512["input_ids"]) == 512
    assert aligned_512["pchunk_valid_mask"][0] is True
    assert aligned_512["pchunk_valid_mask"][1] is False  # evicted by 512 clamp
    assert aligned_512["src_target"] == -100  # evicted candidate slot mapped to ignore_index (-100)
    assert aligned_512["arg_target"] == -100
    assert aligned_512["y_enc"] == 1.0  # ground-truth attack label preserved


def test_clamped_row_preserves_priv_closing_sep_and_nanojev_causal_span_pooling():
    from diodemodel.train import align_row_to_bucket, collate_jev_batch

    nano = _tiny_curriculum_nano().eval()
    sep_token_id = 2
    # Construct a 700-token row where <|priv|> (seg_ids in {1, 2}) spans positions 4..600
    # with closing [SEP] at 600, so clamping to 512 cuts directly inside <|priv|>.
    length = 700
    none_pos = 690
    ids = [15] * length
    segs = [0] * length
    roles = [ROLE_STATE] * length

    ids[0], segs[0] = 1, CANONICAL_SEGMENT_IDS["TRUSTED_INTENT"]
    ids[3], segs[3] = sep_token_id, CANONICAL_SEGMENT_IDS["TRUSTED_INTENT"]
    for pos in range(4, 601):
        segs[pos] = CANONICAL_SEGMENT_IDS["PRIVATE_DATA"] if pos < 300 else CANONICAL_SEGMENT_IDS["DUAL_TRUST_DATA"]
    ids[600] = sep_token_id
    segs[600] = CANONICAL_SEGMENT_IDS["PRIVATE_DATA"]

    for pos in range(601, 640):
        segs[pos] = CANONICAL_SEGMENT_IDS["UNTRUSTED_DATA"]
    ids[639] = sep_token_id

    for pos in range(640, none_pos):
        segs[pos] = CANONICAL_SEGMENT_IDS["CANDIDATE_ACTION"]
    ids[none_pos - 1] = sep_token_id
    ids[none_pos] = NONE_ID
    segs[none_pos] = CANONICAL_SEGMENT_IDS["CANDIDATE_ACTION"]

    qpos = [none_pos + 1 + k for k in range(5)]
    for k, qp in enumerate(qpos):
        ids[qp] = QUERY_IDS[k]
        segs[qp] = CANONICAL_SEGMENT_IDS["DECISION_QUERY"]
        roles[qp] = ROLE_QUERY
    for pos in range(none_pos + 6, length):
        ids[pos] = PAD_ID
        segs[pos] = 0
        roles[pos] = ROLE_PAD

    # Chunk 0 at 10..49, Chunk 1 at 50..503 (survives inside 512), Chunk 2 at 520..599 (evicted at 512)
    ppos = [10, 50, 520] + [none_pos] * 30
    pval = [True, True, True] + [False] * 29 + [True]
    apos = [645] + [none_pos] * 16
    aval = [True] + [False] * 15 + [True]

    raw_row = {
        "input_ids": ids,
        "seg_ids": segs,
        "token_roles": roles,
        "query_positions": qpos,
        "pchunk_positions": ppos,
        "pchunk_valid_mask": pval,
        "arg_positions": apos,
        "arg_valid_mask": aval,
        "none_position": none_pos,
        "unpadded_length": none_pos + 6,
        "y_enc": 0.0,
        "y_sem": 1.0,
        "y_hijack": 0.0,
        "src_target": 2,  # points to chunk 2 at 520 -> must become -100 on clamp
        "arg_target": 0,  # points to arg 0 at 645 -> must become -100 on clamp
        "threat_category": "T4",
        "variant_kind": "prog_t4_leak",
    }

    clamped = align_row_to_bucket(raw_row, 512, pad_token_id=PAD_ID)
    assert clamped["y_sem"] == 1.0
    assert clamped["src_target"] == -100
    assert clamped["arg_target"] == -100
    assert clamped["pchunk_valid_mask"][0] is True
    assert clamped["pchunk_valid_mask"][1] is True
    assert clamped["pchunk_valid_mask"][2] is False

    # Verify the last STATE token with seg_id in {1, 2} is a closing [SEP] (seg_id == 1)
    priv_indices = [
        idx
        for idx, (r, s) in enumerate(zip(clamped["token_roles"], clamped["seg_ids"]))
        if r == ROLE_STATE and s in (1, 2)
    ]
    assert priv_indices, "Expected surviving <|priv|> tokens after clamping"
    priv_sep_pos = max(priv_indices)
    assert clamped["seg_ids"][priv_sep_pos] == CANONICAL_SEGMENT_IDS["PRIVATE_DATA"]
    assert clamped["input_ids"][priv_sep_pos] == sep_token_id
    assert clamped["pchunk_positions"][1] < priv_sep_pos

    # Perturbing the last surviving content token of chunk 1 (at priv_sep_pos - 1) must change
    # NanoJev's pooled slot 1 representation (proving sec_sep did not strip it!).
    batch_base = collate_jev_batch([clamped], bucket_len=512, pad_token_id=PAD_ID)
    clamped_pert = dict(clamped)
    clamped_pert["input_ids"] = list(clamped["input_ids"])
    clamped_pert["input_ids"][priv_sep_pos - 1] = 42
    batch_pert = collate_jev_batch([clamped_pert], bucket_len=512, pad_token_id=PAD_ID)

    with torch.no_grad():
        h_base = nano.encode(batch_base["input_ids"], batch_base["seg_ids"], batch_base["token_roles"])
        src_base, _ = nano.extract_pointer_slot_states(
            h_base,
            batch_base["seg_ids"],
            batch_base["token_roles"],
            batch_base["pchunk_positions"],
            batch_base["pchunk_valid_mask"],
            batch_base["arg_positions"],
            batch_base["arg_valid_mask"],
        )
        h_pert = nano.encode(batch_pert["input_ids"], batch_pert["seg_ids"], batch_pert["token_roles"])
        src_pert, _ = nano.extract_pointer_slot_states(
            h_pert,
            batch_pert["seg_ids"],
            batch_pert["token_roles"],
            batch_pert["pchunk_positions"],
            batch_pert["pchunk_valid_mask"],
            batch_pert["arg_positions"],
            batch_pert["arg_valid_mask"],
        )
        assert (src_pert[:, 1] - src_base[:, 1]).abs().max().item() > 1e-4

        nl, sl, al = call(nano, batch_base, logits=True)
        loss_out = compute_jev_loss(
            nl,
            sl,
            al,
            batch_base["noul_targets"],
            batch_base["src_target"],
            batch_base["arg_target"],
            batch_base["pchunk_valid_mask"],
            batch_base["arg_valid_mask"],
        )
        assert torch.isfinite(loss_out["loss"])
        assert loss_out["src_pointer_ce"].item() == 0.0
        assert loss_out["arg_pointer_ce"].item() == 0.0


def test_dynamic_val_bucket_selection_and_intra_bucket_periodic_checkpoint_resume(tmp_path):
    from unittest.mock import patch

    from diodemodel import train as train_mod
    from diodemodel.train import (
        JevTrainerConfig,
        align_row_to_bucket,
        load_checkpoint,
        make_synthetic_curriculum_dataset,
        run_curriculum_training,
    )

    # 1. Default pad_token_id is None and auto-detects non-zero pad token from ROLE_PAD positions
    default_cfg = JevTrainerConfig()
    assert default_cfg.pad_token_id is None
    assert default_cfg.save_every_steps == 500

    custom_pad_id = 61
    rows_custom_pad = make_synthetic_curriculum_dataset(
        4, vocab_size=VOCAB, seed=808, pad_token_id=custom_pad_id
    )
    aligned_auto = align_row_to_bucket(rows_custom_pad[0], 512, pad_token_id=None)
    pad_positions = [i for i, r in enumerate(aligned_auto["token_roles"]) if r == ROLE_PAD]
    assert pad_positions and all(aligned_auto["input_ids"][i] == custom_pad_id for i in pad_positions)

    # 2. Intra-bucket periodic save_every_steps=1 + mid-bucket preemption resume + dynamic val_calib bucket
    train_rows = make_synthetic_curriculum_dataset(4, vocab_size=VOCAB, seed=909, pad_token_id=PAD_ID)
    # Create a validation row with unpadded_length = 700 (> 512) so dynamic val_calib bucket picks 1024
    val_rows = make_synthetic_curriculum_dataset(2, vocab_size=VOCAB, seed=910, pad_token_id=PAD_ID)
    long_val = align_row_to_bucket(val_rows[0], 1024, pad_token_id=PAD_ID)
    # Extend STATE tokens of long_val to unpadded_length = 700
    for pos in range(4, 694):
        long_val["input_ids"][pos] = 12
        long_val["seg_ids"][pos] = CANONICAL_SEGMENT_IDS["PRIVATE_DATA"]
        long_val["token_roles"][pos] = ROLE_STATE
    long_val["input_ids"][693] = 2  # closing [SEP] of <|priv|>
    long_val["input_ids"][694] = 2  # closing [SEP] of <|call|>
    long_val["seg_ids"][694] = CANONICAL_SEGMENT_IDS["CANDIDATE_ACTION"]
    long_val["token_roles"][694] = ROLE_STATE
    long_val["none_position"] = 695
    long_val["input_ids"][695] = NONE_ID
    long_val["seg_ids"][695] = CANONICAL_SEGMENT_IDS["CANDIDATE_ACTION"]
    long_val["token_roles"][695] = ROLE_STATE
    long_val["query_positions"] = [696, 697, 698, 699, 700]
    for k, qp in enumerate(long_val["query_positions"]):
        long_val["input_ids"][qp] = QUERY_IDS[k]
        long_val["seg_ids"][qp] = CANONICAL_SEGMENT_IDS["DECISION_QUERY"]
        long_val["token_roles"][qp] = ROLE_QUERY
    long_val["pchunk_positions"][32] = 695
    long_val["arg_positions"][16] = 695
    long_val["unpadded_length"] = 701
    val_rows[0] = long_val

    # Uninterrupted 2-step run inside a single bucket (4 rows, batch_size=2 -> 2 steps in bucket 512)
    m_uninterrupted = _tiny_curriculum_modernbert()
    cfg_unint = JevTrainerConfig(
        curriculum_buckets=(512, 1024),
        batch_size=2,
        learning_rate=5e-4,
        stage1_epochs=1,
        stage2_epochs=0,
        max_steps_per_bucket=2,
        max_steps=2,
        save_every_steps=1,
        checkpoint_dir=tmp_path / "mid_bucket_full",
        seed=123,
    )
    run_curriculum_training(m_uninterrupted, train_rows, val_rows=val_rows, config=cfg_unint)

    # Simulate mid-bucket crash after step 1 (batch 0 of 2 inside bucket 512)
    mid_dir = tmp_path / "mid_bucket_part"
    m_mid = _tiny_curriculum_modernbert()
    cfg_mid1 = JevTrainerConfig(
        curriculum_buckets=(512, 1024),
        batch_size=2,
        learning_rate=5e-4,
        stage1_epochs=1,
        stage2_epochs=0,
        max_steps_per_bucket=2,
        max_steps=1,
        save_every_steps=1,
        checkpoint_dir=mid_dir,
        seed=123,
    )
    run_curriculum_training(m_mid, train_rows, val_rows=val_rows, config=cfg_mid1)
    ckpt_mid = load_checkpoint(mid_dir / "checkpoint_latest.pt")
    assert ckpt_mid["global_step"] == 1
    assert ckpt_mid["next_unit_idx"] == 0
    assert ckpt_mid["next_batch_idx"] == 1

    # Resume mid-bucket to step 2 (completing bucket 512) and track collate_jev_batch bucket_len on val_rows
    seen_val_buckets: list[int] = []
    orig_collate = train_mod.collate_jev_batch

    def _spy_collate(rows_arg, *, bucket_len=512, **kwargs):
        if any(int(r.get("unpadded_length", 0)) > 600 for r in rows_arg):
            seen_val_buckets.append(int(bucket_len))
        return orig_collate(rows_arg, bucket_len=bucket_len, **kwargs)

    m_mid_resumed = _tiny_curriculum_modernbert()
    cfg_mid2 = JevTrainerConfig(
        curriculum_buckets=(512, 1024),
        batch_size=2,
        learning_rate=5e-4,
        stage1_epochs=1,
        stage2_epochs=0,
        max_steps_per_bucket=1,  # allow 1 step in bucket 1024 if reached, or cap at max_steps=2
        max_steps=2,
        save_every_steps=1,
        checkpoint_dir=mid_dir,
        seed=123,
    )
    cfg_mid2.max_steps_per_bucket = 2
    res_mid2 = run_curriculum_training(
        m_mid_resumed, train_rows, val_rows=val_rows, config=cfg_mid2, resume_from=mid_dir
    )
    assert res_mid2["global_step"] == 2
    assert res_mid2["steps_executed_this_run"] == 1
    for k, full_t in m_uninterrupted.state_dict().items():
        torch.testing.assert_close(m_mid_resumed.state_dict()[k], full_t, atol=1e-6, rtol=1e-6)

    # Now run a full schedule to completion with val_rows and verify:
    # (a) val_rows with unpadded_length=701 dynamically selected bucket 1024 (not 512)
    # (b) final checkpoint persists calibration_report and resuming completed run skips re-fitting
    done_dir = tmp_path / "done_ckpt"
    m_done = _tiny_curriculum_modernbert()
    cfg_done = JevTrainerConfig(
        curriculum_buckets=(512, 1024),
        batch_size=2,
        learning_rate=5e-4,
        stage1_epochs=1,
        stage2_epochs=1,
        max_steps_per_bucket=1,
        save_every_steps=1,
        checkpoint_dir=done_dir,
        seed=321,
    )
    with patch.object(train_mod, "collate_jev_batch", side_effect=_spy_collate):
        res_done = run_curriculum_training(m_done, train_rows, val_rows=val_rows, config=cfg_done)
    assert 1024 in seen_val_buckets, f"Expected dynamic val bucket 1024, got {seen_val_buckets}"
    assert res_done["calibration_report"] is not None
    final_ckpt = load_checkpoint(done_dir / "checkpoint_latest.pt")
    assert final_ckpt.get("calibration_report") is not None

    with patch.object(train_mod, "fit_temperature_scaling", wraps=train_mod.fit_temperature_scaling) as spy_fit:
        res_noop = run_curriculum_training(
            m_done, train_rows, val_rows=val_rows, config=cfg_done, resume_from=done_dir
        )
        assert res_noop["steps_executed_this_run"] == 0
        assert res_noop["calibration_report"] == res_done["calibration_report"]
        assert spy_fit.call_count == 0, "Completed run must not redundantly re-fit temperatures on resume"


def test_stage2_hard_negative_reweighting_benign_twin_fp_and_t1_t2_fn():
    from diodemodel.train import HARD_NEGATIVE_WEIGHT, compute_hard_negative_weights

    assert HARD_NEGATIVE_WEIGHT == pytest.approx(3.0)
    # 6 rows:
    # 0: BENIGN_TWIN false positive (pred high, target 0) -> 3.0
    # 1: Trifecta-Safe benign false positive (pred high, target 0) -> 3.0
    # 2: BENIGN_TWIN true negative (pred low, target 0) -> 1.0
    # 3: T1 obfuscated leak false negative (pred low, target y_enc=1) -> 3.0
    # 4: T2 obfuscated leak false negative (pred low, target y_sem=1) -> 3.0
    # 5: T1 leak true positive (pred high, target y_enc=1) -> 1.0
    logits = torch.tensor(
        [
            [2.5, -3.0, -3.0],   # row 0: p_enc ~ 0.92 (FP on BENIGN_TWIN)
            [-3.0, 2.0, -3.0],   # row 1: p_sem ~ 0.88 (FP on Trifecta-Safe)
            [-3.0, -3.0, -3.0],  # row 2: p_max ~ 0.05 (TN on BENIGN_TWIN)
            [-2.5, -3.0, -3.0],  # row 3: p_enc ~ 0.08 (FN on T1 leak)
            [-3.0, -2.0, -3.0],  # row 4: p_sem ~ 0.12 (FN on T2 leak)
            [3.0, -3.0, -3.0],   # row 5: p_enc ~ 0.95 (TP on T1 leak)
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    categories = ["T1", "T2", "T1", "T1", "T2", "T1"]
    variants = [
        "BENIGN_TWIN",
        "Trifecta-Safe",
        "BENIGN_TWIN",
        "obfuscated_b64_leak",
        "obfuscated_code_leak",
        "obfuscated_b64_leak",
    ]
    is_trifecta = [True, True, True, False, False, False]

    weights = compute_hard_negative_weights(
        logits,
        targets,
        threat_categories=categories,
        variant_kinds=variants,
        is_trifecta_benign=is_trifecta,
    )
    expected = torch.tensor([3.0, 3.0, 1.0, 3.0, 3.0, 1.0], dtype=torch.float32)
    torch.testing.assert_close(weights, expected)


@pytest.mark.parametrize("arch", ["modernbert", "nano"])
def test_two_stage_length_curriculum_trainer_512_1024_4096_and_live_telemetry(tmp_path, arch):
    import json

    from diodemodel.train import (
        JevTrainerConfig,
        make_synthetic_curriculum_dataset,
        run_curriculum_training,
    )

    m = _tiny_curriculum_modernbert() if arch == "modernbert" else _tiny_curriculum_nano()
    rows = make_synthetic_curriculum_dataset(16, vocab_size=VOCAB, seed=101, pad_token_id=PAD_ID)
    val_rows = make_synthetic_curriculum_dataset(8, vocab_size=VOCAB, seed=202, pad_token_id=PAD_ID)

    ckpt_dir = tmp_path / f"ckpt_{arch}"
    status_log = ckpt_dir / "live_train_status.jsonl"
    cfg = JevTrainerConfig(
        curriculum_buckets=(512, 1024, 4096),
        batch_size=2,
        learning_rate=1e-3,
        stage1_epochs=1,
        stage2_epochs=1,
        max_steps_per_bucket=1,
        pad_token_id=PAD_ID,
        checkpoint_dir=ckpt_dir,
        status_log_path=status_log,
        seed=42,
    )

    init_ptr_w = m.src_pointer.q_proj.weight.detach().clone()
    init_noul_w = m.noul_head.dense.weight.detach().clone()

    result = run_curriculum_training(m, rows, val_rows=val_rows, config=cfg)
    assert result["global_step"] == 6  # 2 stages * 3 buckets * 1 step/bucket
    assert result["buckets_visited"] == [512, 1024, 4096, 512, 1024, 4096]
    assert result["stages_visited"] == [1, 2]
    assert not torch.equal(m.src_pointer.q_proj.weight, init_ptr_w), "Stage 1 must update pointer head"
    assert not torch.equal(m.noul_head.dense.weight, init_noul_w), "Noul head must update"
    assert (ckpt_dir / "checkpoint_latest.pt").is_file()
    assert status_log.is_file()

    events = [json.loads(line) for line in status_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    step_events = [e for e in events if e.get("event") == "train_step"]
    assert len(step_events) == 6
    assert [e["bucket_len"] for e in step_events] == [512, 1024, 4096, 512, 1024, 4096]
    assert [e["stage"] for e in step_events] == [1, 1, 1, 2, 2, 2]
    assert any(e.get("event") == "stage2_hard_negative_step" or "hard_negative_fraction" in e for e in step_events[3:])


def test_atomic_drive_checkpoint_save_and_exact_mid_curriculum_resume(tmp_path):
    from diodemodel.train import (
        JevTrainerConfig,
        load_checkpoint,
        make_synthetic_curriculum_dataset,
        run_curriculum_training,
    )

    rows = make_synthetic_curriculum_dataset(16, vocab_size=VOCAB, seed=303, pad_token_id=PAD_ID)

    # 1. Full uninterrupted 6-step curriculum run (Stage 1: 512, 1024, 4096; Stage 2: 512, 1024, 4096)
    m_full = _tiny_curriculum_modernbert()
    cfg_full = JevTrainerConfig(
        curriculum_buckets=(512, 1024, 4096),
        batch_size=2,
        learning_rate=5e-4,
        stage1_epochs=1,
        stage2_epochs=1,
        max_steps_per_bucket=1,
        pad_token_id=PAD_ID,
        checkpoint_dir=tmp_path / "full_ckpt",
        seed=777,
    )
    res_full = run_curriculum_training(m_full, rows, config=cfg_full)
    assert res_full["global_step"] == 6

    # 2. Partial run stopping at max_steps=4 (mid-curriculum: Stage 2, bucket 512 completed; 1024 & 4096 remaining)
    part_dir = tmp_path / "resume_ckpt"
    m_part = _tiny_curriculum_modernbert()
    cfg_part = JevTrainerConfig(
        curriculum_buckets=(512, 1024, 4096),
        batch_size=2,
        learning_rate=5e-4,
        stage1_epochs=1,
        stage2_epochs=1,
        max_steps_per_bucket=1,
        max_steps=4,
        pad_token_id=PAD_ID,
        checkpoint_dir=part_dir,
        seed=777,
    )
    res_part = run_curriculum_training(m_part, rows, config=cfg_part)
    assert res_part["global_step"] == 4

    latest_ckpt = part_dir / "checkpoint_latest.pt"
    assert latest_ckpt.is_file()
    # Atomic save must leave zero temporary files behind
    assert list(part_dir.glob("*.tmp")) == []

    ckpt_payload = load_checkpoint(latest_ckpt)
    required_keys = {
        "epoch",
        "stage",
        "bucket_len",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_val_metric",
        "rng_state",
    }
    assert required_keys.issubset(set(ckpt_payload.keys()))
    assert ckpt_payload["global_step"] == 4
    assert ckpt_payload["stage"] == 2
    assert ckpt_payload["bucket_len"] == 512

    # 3. Resume from checkpoint_latest.pt into a fresh model instance and finish steps 5..6
    m_resumed = _tiny_curriculum_modernbert()
    cfg_resume = JevTrainerConfig(
        curriculum_buckets=(512, 1024, 4096),
        batch_size=2,
        learning_rate=5e-4,
        stage1_epochs=1,
        stage2_epochs=1,
        max_steps_per_bucket=1,
        max_steps=None,
        pad_token_id=PAD_ID,
        checkpoint_dir=part_dir,
        seed=777,
    )
    res_resumed = run_curriculum_training(
        m_resumed, rows, config=cfg_resume, resume_from=latest_ckpt
    )
    assert res_resumed["global_step"] == 6
    assert res_resumed["resumed_from_step"] == 4
    assert res_resumed["steps_executed_this_run"] == 2

    for k, full_t in m_full.state_dict().items():
        got_t = m_resumed.state_dict()[k]
        torch.testing.assert_close(got_t, full_t, atol=1e-6, rtol=1e-6, msg=f"Mismatch on {k} after resume")


def test_evaluate_jev_checkpoint_metrics_suite(tmp_path, model):
    from diodemodel.train import (
        JevTrainerConfig,
        evaluate_jev_checkpoint,
        make_synthetic_curriculum_dataset,
        run_curriculum_training,
    )

    eval_rows = make_synthetic_curriculum_dataset(16, vocab_size=VOCAB, seed=555, pad_token_id=PAD_ID)
    ckpt_dir = tmp_path / "eval_ckpt"
    cfg = JevTrainerConfig(
        curriculum_buckets=(512,),
        batch_size=8,
        learning_rate=5e-4,
        stage1_epochs=1,
        stage2_epochs=0,
        max_steps_per_bucket=1,
        pad_token_id=PAD_ID,
        checkpoint_dir=ckpt_dir,
    )
    run_curriculum_training(model, eval_rows, config=cfg)
    ckpt_path = ckpt_dir / "checkpoint_latest.pt"

    metrics_from_model = evaluate_jev_checkpoint(model, eval_rows, bucket_len=512, pad_token_id=PAD_ID)
    metrics_from_ckpt = evaluate_jev_checkpoint(
        ckpt_path, eval_rows, model=model, bucket_len=512, pad_token_id=PAD_ID
    )

    for metrics in (metrics_from_model, metrics_from_ckpt):
        assert "per_threat_recall" in metrics
        for cat in ("T1", "T2", "T3", "T4", "T5", "T6", "T7"):
            assert cat in metrics["per_threat_recall"]
            assert 0.0 <= metrics["per_threat_recall"][cat] <= 1.0
        assert 0.0 <= metrics["attack_recall"] <= 1.0
        assert 0.0 <= metrics["flipjev_attack_recall"] <= 1.0
        assert 0.0 <= metrics["flipjev_benign_dos_fpr"] <= 1.0
        assert 0.0 <= metrics["trifecta_fpr"] <= 1.0
        assert 0.0 <= metrics["src_pointer_top1_acc"] <= 1.0
        assert 0.0 <= metrics["arg_pointer_top1_acc"] <= 1.0
        assert 0.0 <= metrics["pointer_top1_acc"] <= 1.0
        assert len(metrics["ece_per_head"]) == 3
        assert len(metrics["brier_per_head"]) == 3
        assert 0.0 <= metrics["ece_val"] <= 1.0
        assert 0.0 <= metrics["brier"] <= 1.0
        assert "calibration_report" in metrics
        assert metrics["calibration_report"]["ece_val"] == pytest.approx(metrics["ece_val"])


def test_train_cli_smoke_steps_cpu_under_5_seconds(tmp_path):
    import time

    from diodemodel.train import main as train_cli_main

    ckpt_dir = tmp_path / "cli_smoke"
    t0 = time.perf_counter()
    rc = train_cli_main(
        [
            "--smoke-steps",
            "2",
            "--checkpoint-dir",
            str(ckpt_dir),
        ]
    )
    elapsed = time.perf_counter() - t0
    assert rc == 0
    assert elapsed < 5.0, f"Smoke CLI took {elapsed:.2f}s (must be < 5.0s)"
    assert (ckpt_dir / "modernbert" / "checkpoint_latest.pt").is_file()
    assert (ckpt_dir / "nanojev" / "checkpoint_latest.pt").is_file()
    assert (ckpt_dir / "live_train_status.jsonl").is_file()


def test_notebook_02_train_jev_models_structure_and_cells():
    import json

    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "02_train_jev_models.ipynb"
    assert nb_path.is_file(), f"Missing Colab training notebook: {nb_path}"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    cells = nb["cells"]
    assert len(cells) >= 6

    code_sources: list[str] = []
    for cell in cells:
        if cell["cell_type"] == "code":
            raw_lines = cell["source"] if isinstance(cell["source"], list) else cell["source"].splitlines(True)
            clean_lines = [
                line for line in raw_lines if not line.lstrip().startswith("!")
            ]
            src = "".join(clean_lines)
            ast.parse(src)
            code_sources.append("".join(raw_lines))

    joined = "\n".join(code_sources)
    for expected in (
        "/content/drive",
        "model_lab_src.tar.gz",
        "DiodeMCP-Jev-Base-149M",
        "answerdotai/ModernBERT-base",
        "DiodeMCP-Jev-Large-395M",
        "answerdotai/ModernBERT-large",
        "DiodeMCP-NanoJev-0.6B",
        "Qwen/Qwen3-0.6B-Base",
        "gradient_checkpointing",
        "bfloat16",
        "/content/drive/MyDrive/diodemcp/checkpoints",
        "REQUIRED_SPECIAL_TOKENS",
    ):
        assert expected in joined, f"Expected {expected!r} in 02_train_jev_models.ipynb"


