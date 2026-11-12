"""Task A4 tests: token_roles 4D-mask isolation, ModernBERT-Jev / NanoJev-0.6B, strictly proper losses.

All models are TINY, randomly initialised from configs (CPU, no network, no HF weight download).
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
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

    # Parity at the export example AND at a different batch + sequence length (dynamic axes).
    for batch in (ex, make_batch(batch=3, length=192, seed=82), make_batch(batch=1, length=96, seed=83)):
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
