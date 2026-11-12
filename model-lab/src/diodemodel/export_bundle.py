"""Eager ONNX opset=17 exporter, MatMul/Gemm-only INT8 dynamic quantizer, post-INT8 ORT temperature
calibrator, and DiodeModelBundle v1 packager (Rulings RA3, R5, R6, I7, I8, I11).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Final
import warnings

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.quant_utils import add_pre_process_metadata
import torch
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerBase

from diodemodel.contract import (
    CANONICAL_SEGMENT_IDS,
    DEFAULT_LENGTH_BUCKETS,
    ONNX_INPUT_SPECS,
    ONNX_OPSET_VERSION,
    ONNX_OUTPUT_SPECS,
    REQUIRED_SPECIAL_TOKENS,
    ROLE_PAD,
    ROLE_QUERY,
    ROLE_STATE,
    BundleManifest,
    clamp_thresholds,
    compute_sha256,
    validate_bundle_dir,
    validate_onnx_contract,
)
from diodemodel.dataset_packer import PackedJevRow
from diodemodel.losses_rlcd import (
    DEFAULT_ECE_BINS,
    fit_temperature_scaling,
    probs_to_logits,
    temperature_calibration_report,
)
from diodemodel.modeling_jev import (
    ARG_SLOTS,
    HEAD_SCOPE_NAMES,
    NUM_NOUL_HEADS,
    ONNX_DYNAMIC_AXES,
    SRC_SLOTS,
    _JevModelBase,
    pin_onnx_output_shapes,
)
from diodemodel.tokenizer_builder import build_jev_tokenizer
from diodemodel.train import (
    CURRICULUM_BUCKETS,
    _build_tiny_smoke_modernbert,
    _build_tiny_smoke_nanojev,
    _instantiate_model_from_checkpoint_payload,
    _select_batch_bucket,
    collate_jev_batch,
    load_checkpoint,
    load_packed_jev_dataset,
    make_synthetic_curriculum_dataset,
)

ONNX_INPUT_NAMES: Final[tuple[str, ...]] = tuple(ONNX_INPUT_SPECS.keys())
ONNX_OUTPUT_NAMES: Final[tuple[str, ...]] = tuple(ONNX_OUTPUT_SPECS.keys())
QUANTIZE_OP_TYPES: Final[tuple[str, ...]] = ("MatMul", "Gemm")
HEAD_EXCLUSION_SCOPES: Final[tuple[str, ...]] = (*HEAD_SCOPE_NAMES, "logit_temperatures")


@dataclass(frozen=True)
class ExportBundleConfig:
    """Configuration for DiodeModelBundle v1 export, INT8 quantization, and post-INT8 calibration."""

    model_id: str = "diodemcp-jev-base-int8"
    max_seq_len: int = 8192
    length_buckets: tuple[int, ...] = tuple(DEFAULT_LENGTH_BUCKETS)
    thresholds: Mapping[str, float] | None = None
    opset_version: int = ONNX_OPSET_VERSION
    weight_type: QuantType = QuantType.QInt8
    per_channel: bool = False
    calib_batch_size: int = 16
    calib_buckets: tuple[int, ...] = CURRICULUM_BUCKETS
    num_ece_bins: int = DEFAULT_ECE_BINS
    intra_op_threads: int = 4
    keep_fp32_onnx: bool = False


def make_dummy_export_batch(
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    vocab_size: int = 64,
    pad_token_id: int = 0,
    seed: int = 81,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Create a canonical 8-input dummy batch for deterministic opset-17 ONNX tracing."""
    if batch_size < 1 or seq_len < 16:
        raise ValueError(f"Expected batch_size >= 1 and seq_len >= 16, got {batch_size}, {seq_len}")
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    eff_vocab = max(16, int(vocab_size))
    pad_id = int(pad_token_id) if 0 <= int(pad_token_id) < eff_vocab else 0

    ids = np.full((batch_size, seq_len), pad_id, dtype=np.int64)
    segs = np.zeros((batch_size, seq_len), dtype=np.int64)
    roles = np.full((batch_size, seq_len), ROLE_PAD, dtype=np.int64)
    qpos = np.zeros((batch_size, 5), dtype=np.int64)
    ppos = np.zeros((batch_size, SRC_SLOTS), dtype=np.int64)
    pval = np.zeros((batch_size, SRC_SLOTS), dtype=np.bool_)
    apos = np.zeros((batch_size, ARG_SLOTS), dtype=np.int64)
    aval = np.zeros((batch_size, ARG_SLOTS), dtype=np.bool_)

    for b in range(batch_size):
        n_state = min(seq_len - 6, max(24, seq_len // 2 + b * 4))
        none_pos = n_state - 1
        priv_sep = max(8, n_state // 2)
        ids[b, :n_state] = rng.integers(1, eff_vocab, size=n_state)
        roles[b, :n_state] = ROLE_STATE
        segs[b, :4] = CANONICAL_SEGMENT_IDS["TRUSTED_INTENT"]
        segs[b, 4 : priv_sep + 1] = CANONICAL_SEGMENT_IDS["PRIVATE_DATA"]
        segs[b, priv_sep + 1 : priv_sep + 5] = CANONICAL_SEGMENT_IDS["UNTRUSTED_DATA"]
        segs[b, priv_sep + 5 : n_state] = CANONICAL_SEGMENT_IDS["CANDIDATE_ACTION"]

        for k in range(5):
            qp = n_state + k
            qpos[b, k] = qp
            ids[b, qp] = min(eff_vocab - 1, 3 + k)
            segs[b, qp] = CANONICAL_SEGMENT_IDS["DECISION_QUERY"]
            roles[b, qp] = ROLE_QUERY

        ppos[b, :] = none_pos
        ppos[b, 0] = 5
        ppos[b, 1] = min(priv_sep - 1, 7)
        pval[b, 0] = True
        pval[b, 1] = True
        pval[b, SRC_SLOTS - 1] = True

        apos[b, :] = none_pos
        apos[b, 0] = min(none_pos - 2, priv_sep + 6)
        apos[b, 1] = min(none_pos - 1, priv_sep + 7)
        aval[b, 0] = True
        aval[b, 1] = True
        aval[b, ARG_SLOTS - 1] = True

    return {
        "input_ids": torch.from_numpy(ids).to(dev),
        "seg_ids": torch.from_numpy(segs).to(dev),
        "token_roles": torch.from_numpy(roles).to(dev),
        "query_positions": torch.from_numpy(qpos).to(dev),
        "pchunk_positions": torch.from_numpy(ppos).to(dev),
        "pchunk_valid_mask": torch.from_numpy(pval).to(dev),
        "arg_positions": torch.from_numpy(apos).to(dev),
        "arg_valid_mask": torch.from_numpy(aval).to(dev),
    }


def verify_onnx_no_bool_where(model_or_path: onnx.ModelProto | Path | str) -> None:
    """Verify that no ONNX ``Where`` node selects between ``BOOL`` data tensors (unsupported on ORT CPU)."""
    proto = (
        onnx.load(str(model_or_path))
        if isinstance(model_or_path, (str, Path))
        else model_or_path
    )
    try:
        inferred = onnx.shape_inference.infer_shapes(proto)
    except Exception:
        inferred = proto

    elem_types: dict[str, int] = {}
    for vi in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type:
            elem_types[vi.name] = int(vi.type.tensor_type.elem_type)
    for init in inferred.graph.initializer:
        elem_types[init.name] = int(init.data_type)
    for node in inferred.graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type:
                    elem_types[node.output[0]] = int(attr.t.data_type)

    for node in inferred.graph.node:
        if node.op_type != "Where":
            continue
        data_tensors = [*node.input[1:3], *node.output[:1]]
        for t_name in data_tensors:
            if elem_types.get(t_name) == TensorProto.BOOL:
                raise ValueError(
                    f"ONNX Where node {node.name or '<unnamed>'!r} operates on BOOL tensor "
                    f"{t_name!r}, which lacks an ORT CPU kernel"
                )


def collect_excluded_onnx_nodes(model_or_path: onnx.ModelProto | Path | str) -> list[str]:
    """Return node names to exclude from INT8 quantization so only backbone weight MatMul/Gemms are quantized.

    Excludes:
    * All nodes under ``HEAD_SCOPE_NAMES`` (``/noul_head``, ``/src_pointer``, ``/arg_pointer``) and
      ``logit_temperatures`` (plus ``<name>_MatMul`` aliases created when ORT decomposes ``Gemm``).
    * All nodes outside ``/backbone/`` (such as 4D mask construction and segment/token embedding ops).
    * Any ``MatMul``/``Gemm`` node whose second input is not a constant weight initializer (e.g. attention
      ``Q @ K^T`` and ``attn_probs @ V``).
    """
    proto = (
        onnx.load(str(model_or_path))
        if isinstance(model_or_path, (str, Path))
        else model_or_path
    )
    init_names = {init.name for init in proto.graph.initializer}
    excluded: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            excluded.append(name)

    for node in proto.graph.node:
        in_head_scope = any(
            f"/{scope}" in node.name
            or node.name.startswith(f"{scope}/")
            or node.name == scope
            for scope in HEAD_EXCLUSION_SCOPES
        )
        touches_temp = any("logit_temperatures" in t for t in (*node.input, *node.output))
        outside_backbone = not node.name.startswith("/backbone/")
        non_weight_matmul = node.op_type in QUANTIZE_OP_TYPES and (
            len(node.input) < 2 or node.input[1] not in init_names
        )

        if in_head_scope or touches_temp or outside_backbone or non_weight_matmul:
            _add(node.name)
            _add(f"{node.name}_MatMul")

    return excluded


def export_fp32_onnx(
    model: _JevModelBase,
    output_path: Path | str,
    *,
    dummy_batch: Mapping[str, torch.Tensor] | None = None,
    opset_version: int = ONNX_OPSET_VERSION,
) -> Path:
    """Export ``ModernBERTJevModel`` or ``NanoJevCausalModel`` to an opset=17 FP32 ONNX graph."""
    if opset_version != ONNX_OPSET_VERSION:
        raise ValueError(
            f"DiodeModelBundle v1 requires opset_version == {ONNX_OPSET_VERSION}, got {opset_version}"
        )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    orig_device = next(model.parameters()).device
    cpu_model = model.to("cpu").eval()
    try:
        vocab_size = int(cpu_model.backbone.get_input_embeddings().num_embeddings)
        pad_id = int(getattr(cpu_model.backbone.config, "pad_token_id", 0) or 0)
        batch = (
            {k: v.detach().to("cpu") for k, v in dummy_batch.items()}
            if dummy_batch is not None
            else make_dummy_export_batch(
                batch_size=2,
                seq_len=128,
                vocab_size=vocab_size,
                pad_token_id=pad_id,
                device="cpu",
            )
        )
        args = tuple(batch[name] for name in ONNX_INPUT_NAMES)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                cpu_model,
                args,
                str(out_path),
                input_names=list(ONNX_INPUT_NAMES),
                output_names=list(ONNX_OUTPUT_NAMES),
                dynamic_axes=ONNX_DYNAMIC_AXES,
                opset_version=opset_version,
                do_constant_folding=True,
                dynamo=False,
            )
    finally:
        model.to(orig_device)
        if was_training:
            model.train()

    proto = onnx.load(str(out_path))
    pin_onnx_output_shapes(proto)
    add_pre_process_metadata(proto)
    verify_onnx_no_bool_where(proto)
    onnx.save(proto, str(out_path))
    validate_onnx_contract(out_path)
    return out_path


def quantize_onnx_matmul_int8(
    fp32_onnx_path: Path | str,
    int8_onnx_path: Path | str,
    *,
    weight_type: QuantType = QuantType.QInt8,
    per_channel: bool = False,
) -> dict[str, Any]:
    """Quantize ONLY backbone ``MatMul`` and ``Gemm`` weight projections to INT8 while keeping
    ``LayerNorm``, ``Softmax``, 4D mask ops, and ``HEAD_SCOPE_NAMES`` in pure FP32 (`I8`)."""
    src_path = Path(fp32_onnx_path)
    dst_path = Path(int8_onnx_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    fp32_proto = onnx.load(str(src_path))
    add_pre_process_metadata(fp32_proto)
    pin_onnx_output_shapes(fp32_proto)
    onnx.save(fp32_proto, str(src_path))

    nodes_to_exclude = collect_excluded_onnx_nodes(fp32_proto)
    quantize_dynamic(
        model_input=src_path,
        model_output=dst_path,
        op_types_to_quantize=list(QUANTIZE_OP_TYPES),
        per_channel=per_channel,
        reduce_range=False,
        weight_type=weight_type,
        nodes_to_exclude=nodes_to_exclude,
        extra_options={"MatMulConstBOnly": True},
    )

    q_proto = onnx.load(str(dst_path))
    pin_onnx_output_shapes(q_proto)
    verify_onnx_no_bool_where(q_proto)
    onnx.save(q_proto, str(dst_path))
    validate_onnx_contract(dst_path)

    quantized_matmul_nodes = [
        n.name for n in q_proto.graph.node if n.op_type == "MatMulInteger"
    ]
    dql_nodes = [
        n.name for n in q_proto.graph.node if n.op_type == "DynamicQuantizeLinear"
    ]
    return {
        "op_types_to_quantize": list(QUANTIZE_OP_TYPES),
        "weight_type": str(weight_type),
        "per_channel": bool(per_channel),
        "nodes_to_exclude": nodes_to_exclude,
        "quantized_matmul_nodes": quantized_matmul_nodes,
        "dynamic_quantize_linear_nodes": dql_nodes,
    }


def read_onnx_logit_temperatures(onnx_path_or_model: Path | str | onnx.ModelProto) -> np.ndarray:
    """Read the 3-element FP32 ``logit_temperatures`` initializer from an ONNX model."""
    proto = (
        onnx.load(str(onnx_path_or_model))
        if isinstance(onnx_path_or_model, (str, Path))
        else onnx_path_or_model
    )
    for init in proto.graph.initializer:
        if init.name == "logit_temperatures":
            arr = np.asarray(numpy_helper.to_array(init), dtype=np.float32).reshape(-1)
            if arr.shape != (NUM_NOUL_HEADS,):
                raise ValueError(
                    f"Initializer 'logit_temperatures' expected shape ({NUM_NOUL_HEADS},), got {arr.shape}"
                )
            return arr
    raise ValueError("ONNX graph is missing required 'logit_temperatures' initializer")


def bake_onnx_logit_temperatures(
    onnx_path: Path | str,
    temperatures: Any,
) -> np.ndarray:
    """Overwrite the FP32 ``logit_temperatures`` initializer inside ``onnx_path`` in-place (`I8`, `I11`)."""
    path = Path(onnx_path)
    t_arr = np.asarray(
        temperatures.detach().cpu().numpy()
        if isinstance(temperatures, torch.Tensor)
        else temperatures,
        dtype=np.float32,
    ).reshape(-1)
    if t_arr.shape != (NUM_NOUL_HEADS,):
        raise ValueError(f"temperatures must have shape ({NUM_NOUL_HEADS},), got {t_arr.shape}")
    if not bool(np.all(np.isfinite(t_arr))) or bool(np.any(t_arr <= 0.0)):
        raise ValueError(f"temperatures must be finite and > 0, got {t_arr.tolist()}")

    proto = onnx.load(str(path))
    found = False
    for init in proto.graph.initializer:
        if init.name == "logit_temperatures":
            init.CopyFrom(numpy_helper.from_array(t_arr, name="logit_temperatures"))
            found = True
            break
    if not found:
        raise ValueError(f"Initializer 'logit_temperatures' not found in {path}")

    pin_onnx_output_shapes(proto)
    onnx.save(proto, str(path))
    validate_onnx_contract(path)
    return t_arr


def create_ort_cpu_session(
    onnx_path: Path | str,
    *,
    intra_op_threads: int = 4,
) -> ort.InferenceSession:
    """Create a deterministic CPU ONNX Runtime session matching Track B's ``OrtJevPredictor`` settings."""
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = max(1, int(intra_op_threads))
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def calibrate_int8_onnx_temperatures(
    int8_onnx_path: Path | str,
    val_calib_rows: Sequence[Mapping[str, Any] | PackedJevRow] | Path | str,
    *,
    batch_size: int = 16,
    calib_buckets: Sequence[int] = CURRICULUM_BUCKETS,
    pad_token_id: int | None = None,
    num_bins: int = DEFAULT_ECE_BINS,
    thresholds: Mapping[str, float] | None = None,
    intra_op_threads: int = 4,
) -> dict[str, Any]:
    """Run ``val_calib`` through the quantized INT8 ORT session, recover pre-temperature INT8 logits via
    ``probs_to_logits(p, T_prev)``, fit per-head ``(T_enc, T_sem, T_hijack)`` via ``fit_temperature_scaling``,
    and bake the fitted FP32 temperatures directly into ``model_int8.onnx`` (`I8`, `I11`)."""
    onnx_path = Path(int8_onnx_path)
    rows = load_packed_jev_dataset(val_calib_rows)
    if not rows:
        raise ValueError("calibrate_int8_onnx_temperatures requires at least 1 calibration row")

    t_prev_np = read_onnx_logit_temperatures(onnx_path)
    t_prev = torch.from_numpy(t_prev_np.copy())

    sess = create_ort_cpu_session(onnx_path, intra_op_threads=intra_op_threads)
    eff_bs = max(1, int(batch_size))

    prob_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    for start in range(0, len(rows), eff_bs):
        batch_slice = rows[start : start + eff_bs]
        bucket = _select_batch_bucket(batch_slice, calib_buckets)
        batch = collate_jev_batch(
            batch_slice,
            bucket_len=bucket,
            pad_token_id=pad_token_id,
            device="cpu",
        )
        feed = {name: batch[name].numpy() for name in ONNX_INPUT_NAMES}
        noul_probs_np, _, _ = sess.run(list(ONNX_OUTPUT_NAMES), feed)
        prob_batches.append(torch.from_numpy(np.asarray(noul_probs_np, dtype=np.float32)))
        target_batches.append(batch["noul_targets"].detach().cpu())

    int8_probs_prev = torch.cat(prob_batches, dim=0)
    val_targets = torch.cat(target_batches, dim=0)

    # Invert sigmoid(z_int8 / T_prev) -> true post-INT8 pre-temperature logits z_int8
    int8_raw_logits = probs_to_logits(int8_probs_prev, t_prev)
    fitted_temps = fit_temperature_scaling(int8_raw_logits, val_targets)
    calib_rep = temperature_calibration_report(
        int8_raw_logits, val_targets, fitted_temps, num_bins=num_bins
    )

    # Compute pre/post Brier on post-INT8 logits
    z_f64 = int8_raw_logits.double()
    y_f64 = val_targets.double()
    t_fit_f64 = fitted_temps.double()
    p_unscaled = torch.sigmoid(z_f64)
    p_calibrated = torch.sigmoid(z_f64 / t_fit_f64)
    brier_before = [
        float(((p_unscaled[:, h] - y_f64[:, h]) ** 2).mean().item())
        for h in range(NUM_NOUL_HEADS)
    ]
    brier_after = [
        float(((p_calibrated[:, h] - y_f64[:, h]) ** 2).mean().item())
        for h in range(NUM_NOUL_HEADS)
    ]
    brier_val = float(np.mean(brier_after))
    nll_val = float(np.mean(calib_rep["nll_after"]))

    # Bake fitted temperatures into model_int8.onnx
    bake_onnx_logit_temperatures(onnx_path, fitted_temps)

    clamped_thresh = clamp_thresholds(
        dict(thresholds) if thresholds is not None else {"warn_min": 0.20, "critical_min": 0.80}
    )
    t_list = [float(x) for x in fitted_temps.detach().cpu().tolist()]
    calibration_payload: dict[str, Any] = {
        "method": "temperature_scaling",
        "post_int8_calibrated": True,
        "temperatures": t_list,
        "temperatures_by_head": {
            "T_enc": t_list[0],
            "T_sem": t_list[1],
            "T_hijack": t_list[2],
        },
        "previous_temperatures": [float(x) for x in t_prev.tolist()],
        "ece_before": [float(x) for x in calib_rep["ece_before"]],
        "ece_after": [float(x) for x in calib_rep["ece_after"]],
        "ece_val": float(calib_rep["ece_val"]),
        "brier_before": brier_before,
        "brier_after": brier_after,
        "brier_val": brier_val,
        "nll_before": [float(x) for x in calib_rep["nll_before"]],
        "nll_after": [float(x) for x in calib_rep["nll_after"]],
        "nll_val": nll_val,
        "num_bins": int(num_bins),
        "num_calibration_rows": len(rows),
        "thresholds": clamped_thresh,
    }
    return {
        "fitted_temperatures": fitted_temps,
        "previous_temperatures": t_prev,
        "int8_raw_logits": int8_raw_logits,
        "val_targets": val_targets,
        "calibration_payload": calibration_payload,
        "report": calib_rep,
    }


def _materialize_bundle_tokenizer(
    tokenizer_source: Path | str | Tokenizer | PreTrainedTokenizerBase,
    target_tokenizer_json: Path,
    *,
    expected_special_token_ids: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Write ``tokenizer.json`` into the bundle directory and return verified 17 ``special_token_ids``."""
    target_tokenizer_json.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(tokenizer_source, Tokenizer):
        tokenizer_source.save(str(target_tokenizer_json))
    elif isinstance(tokenizer_source, PreTrainedTokenizerBase):
        if hasattr(tokenizer_source, "backend_tokenizer"):
            tokenizer_source.backend_tokenizer.save(str(target_tokenizer_json))
        else:
            with tempfile.TemporaryDirectory() as tmp_d:
                tokenizer_source.save_pretrained(tmp_d)
                shutil.copyfile(Path(tmp_d) / "tokenizer.json", target_tokenizer_json)
    else:
        src_path = Path(tokenizer_source)
        if src_path.is_file():
            if src_path.resolve() != target_tokenizer_json.resolve():
                shutil.copyfile(src_path, target_tokenizer_json)
        elif src_path.is_dir() and (src_path / "tokenizer.json").is_file():
            src_tok = src_path / "tokenizer.json"
            if src_tok.resolve() != target_tokenizer_json.resolve():
                shutil.copyfile(src_tok, target_tokenizer_json)
        else:
            with tempfile.TemporaryDirectory() as tmp_d:
                build_jev_tokenizer(str(tokenizer_source), Path(tmp_d))
                shutil.copyfile(Path(tmp_d) / "tokenizer.json", target_tokenizer_json)

    rust_tok = Tokenizer.from_file(str(target_tokenizer_json))
    resolved_ids: dict[str, int] = {}
    for tok_str in REQUIRED_SPECIAL_TOKENS:
        tid = rust_tok.token_to_id(tok_str)
        if tid is None:
            raise ValueError(
                f"Tokenizer {target_tokenizer_json} is missing required special token {tok_str!r}"
            )
        resolved_ids[tok_str] = int(tid)

    if expected_special_token_ids is not None:
        for tok_str in REQUIRED_SPECIAL_TOKENS:
            if tok_str in expected_special_token_ids:
                exp_id = int(expected_special_token_ids[tok_str])
                if resolved_ids[tok_str] != exp_id:
                    raise ValueError(
                        f"Special token ID mismatch for {tok_str!r}: "
                        f"expected {exp_id}, tokenizer.json has {resolved_ids[tok_str]}"
                    )
    return resolved_ids


def export_diode_bundle(
    model_or_checkpoint: _JevModelBase | Path | str,
    bundle_dir: Path | str,
    *,
    tokenizer_source: Path | str | Tokenizer | PreTrainedTokenizerBase,
    model_id: str = "diodemcp-jev-base-int8",
    val_calib_rows: Sequence[Mapping[str, Any] | PackedJevRow] | Path | str | None = None,
    model: _JevModelBase | None = None,
    special_token_ids: Mapping[str, int] | None = None,
    thresholds: Mapping[str, float] | None = None,
    length_buckets: Sequence[int] = DEFAULT_LENGTH_BUCKETS,
    max_seq_len: int = 8192,
    opset_version: int = ONNX_OPSET_VERSION,
    weight_type: QuantType = QuantType.QInt8,
    per_channel: bool = False,
    calib_batch_size: int = 16,
    calib_buckets: Sequence[int] = CURRICULUM_BUCKETS,
    num_ece_bins: int = DEFAULT_ECE_BINS,
    intra_op_threads: int = 4,
    keep_fp32_onnx: bool = False,
    config: ExportBundleConfig | None = None,
) -> dict[str, Any]:
    """Export a Jev model or checkpoint to a complete, SHA256-verified ``DiodeModelBundle v1`` directory.

    Pipeline:
    1. Materializes ``tokenizer.json`` and extracts the 17 ``REQUIRED_SPECIAL_TOKENS``.
    2. Exports ``model`` in eager mode with ``dynamo=False, opset_version=17`` and pins static output shapes.
    3. Quantizes ONLY backbone ``MatMul`` and ``Gemm`` weight projections to INT8 (`model_int8.onnx`) while
       excluding ``HEAD_SCOPE_NAMES`` (``/noul_head``, ``/src_pointer``, ``/arg_pointer``), ``logit_temperatures``,
       4D mask ops, ``LayerNorm``, and ``Softmax``.
    4. Runs ``val_calib_rows`` through the quantized INT8 ORT session, inverts probabilities to recover
       post-INT8 pre-temperature logits, fits ``(T_enc, T_sem, T_hijack)`` via ``fit_temperature_scaling``,
       and bakes the fitted FP32 temperatures into ``model_int8.onnx``.
    5. Writes ``calibration.json`` and ``manifest.json`` (computing ``sha256_model_onnx``,
       ``sha256_tokenizer_json``, and ``sha256_calibration_json`` AFTER baking temperatures) and validates
       the bundle with ``contract.validate_bundle_dir(bundle_dir, verify_sha256=True)``.
    """
    if config is not None:
        model_id = config.model_id
        max_seq_len = config.max_seq_len
        length_buckets = config.length_buckets
        thresholds = config.thresholds if thresholds is None else thresholds
        opset_version = config.opset_version
        weight_type = config.weight_type
        per_channel = config.per_channel
        calib_batch_size = config.calib_batch_size
        calib_buckets = config.calib_buckets
        num_ece_bins = config.num_ece_bins
        intra_op_threads = config.intra_op_threads
        keep_fp32_onnx = config.keep_fp32_onnx

    out_dir = Path(bundle_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model or checkpoint
    if isinstance(model_or_checkpoint, (str, Path)):
        ckpt_payload = load_checkpoint(
            model_or_checkpoint, model=model, restore_rng=False, map_location="cpu"
        )
        export_model = (
            model.to("cpu").eval()
            if model is not None
            else _instantiate_model_from_checkpoint_payload(ckpt_payload, device="cpu")
        )
    else:
        export_model = model_or_checkpoint

    # 2. Write and verify tokenizer.json
    tokenizer_json_path = out_dir / "tokenizer.json"
    resolved_special_ids = _materialize_bundle_tokenizer(
        tokenizer_source,
        tokenizer_json_path,
        expected_special_token_ids=special_token_ids,
    )
    pad_token_id = int(resolved_special_ids["[PAD]"])

    # 3. Export FP32 opset-17 ONNX and quantize backbone MatMul/Gemm to INT8
    int8_onnx_path = out_dir / "model_int8.onnx"
    with tempfile.TemporaryDirectory(prefix="diodemodel_fp32_") as tmp_dir:
        fp32_path = (
            (out_dir / "model_fp32.onnx")
            if keep_fp32_onnx
            else (Path(tmp_dir) / "model_fp32.onnx")
        )
        export_fp32_onnx(
            export_model,
            fp32_path,
            opset_version=opset_version,
        )
        quant_info = quantize_onnx_matmul_int8(
            fp32_path,
            int8_onnx_path,
            weight_type=weight_type,
            per_channel=per_channel,
        )

    # 4. Post-INT8 temperature scaling on val_calib_rows
    clamped_thresh = clamp_thresholds(
        dict(thresholds) if thresholds is not None else {"warn_min": 0.20, "critical_min": 0.80}
    )
    int8_raw_logits: torch.Tensor | None = None
    val_targets: torch.Tensor | None = None

    if val_calib_rows is not None:
        calib_result = calibrate_int8_onnx_temperatures(
            int8_onnx_path,
            val_calib_rows,
            batch_size=calib_batch_size,
            calib_buckets=calib_buckets,
            pad_token_id=pad_token_id,
            num_bins=num_ece_bins,
            thresholds=clamped_thresh,
            intra_op_threads=intra_op_threads,
        )
        fitted_temperatures: torch.Tensor = calib_result["fitted_temperatures"]
        export_model.set_logit_temperatures(fitted_temperatures)
        calibration_payload = calib_result["calibration_payload"]
        int8_raw_logits = calib_result["int8_raw_logits"]
        val_targets = calib_result["val_targets"]
    else:
        t_cur = read_onnx_logit_temperatures(int8_onnx_path)
        fitted_temperatures = torch.from_numpy(t_cur.copy())
        t_list = [float(x) for x in t_cur.tolist()]
        calibration_payload = {
            "method": "temperature_scaling",
            "post_int8_calibrated": False,
            "temperatures": t_list,
            "temperatures_by_head": {
                "T_enc": t_list[0],
                "T_sem": t_list[1],
                "T_hijack": t_list[2],
            },
            "previous_temperatures": t_list,
            "ece_before": [0.0, 0.0, 0.0],
            "ece_after": [0.0, 0.0, 0.0],
            "ece_val": 0.0,
            "brier_before": [0.0, 0.0, 0.0],
            "brier_after": [0.0, 0.0, 0.0],
            "brier_val": 0.0,
            "nll_before": [0.0, 0.0, 0.0],
            "nll_after": [0.0, 0.0, 0.0],
            "nll_val": 0.0,
            "num_bins": int(num_ece_bins),
            "num_calibration_rows": 0,
            "thresholds": clamped_thresh,
        }

    # 5. Write calibration.json, compute all 3 SHA256 digests AFTER baking temperatures, and write manifest.json
    calibration_json_path = out_dir / "calibration.json"
    calibration_json_path.write_text(
        json.dumps(calibration_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    sha_model = compute_sha256(int8_onnx_path)
    sha_tok = compute_sha256(tokenizer_json_path)
    sha_calib = compute_sha256(calibration_json_path)

    manifest_obj = BundleManifest(
        bundle_version="1.0",
        model_id=model_id,
        sha256_model_onnx=sha_model,
        sha256_tokenizer_json=sha_tok,
        sha256_calibration_json=sha_calib,
        max_seq_len=int(max_seq_len),
        length_buckets=[int(b) for b in length_buckets],
        max_pchunks=SRC_SLOTS - 1,
        max_args=ARG_SLOTS - 1,
        special_token_ids=resolved_special_ids,
        segment_ids=dict(CANONICAL_SEGMENT_IDS),
        thresholds=clamped_thresh,
        ece_val=float(calibration_payload["ece_val"]),
    )
    manifest_json_path = out_dir / "manifest.json"
    manifest_json_path.write_text(
        json.dumps(manifest_obj.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # 6. Validate complete bundle directory
    validated_manifest = validate_bundle_dir(out_dir, verify_sha256=True)

    return {
        "bundle_dir": out_dir,
        "manifest": validated_manifest,
        "calibration": calibration_payload,
        "quantization": quant_info,
        "fitted_temperatures": fitted_temperatures,
        "int8_raw_logits": int8_raw_logits,
        "val_targets": val_targets,
    }


def build_smoke_jev_tokenizer() -> tuple[Tokenizer, dict[str, int]]:
    """Build a self-contained deterministic 17-special-token fast Rust tokenizer for offline smoke runs."""
    special_ids = {tok: idx for idx, tok in enumerate(REQUIRED_SPECIAL_TOKENS)}
    vocab: dict[str, int] = dict(special_ids)
    vocab["[UNK]"] = len(vocab)
    for idx in range(len(vocab), 64):
        vocab[f"tok_{idx}"] = idx

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.add_special_tokens(
        [
            AddedToken(
                s,
                single_word=False,
                lstrip=False,
                rstrip=False,
                normalized=False,
                special=True,
            )
            for s in REQUIRED_SPECIAL_TOKENS
        ]
    )
    return tok, special_ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m diodemodel.export_bundle",
        description="Export, INT8-quantize, post-INT8 calibrate, and package a DiodeModelBundle v1.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast offline CPU smoke export across tiny ModernBERTJevModel and NanoJevCausalModel.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to atomic checkpoint (.pt) to export.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json, tokenizer dir, or HF model ID.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="DiodeMCP-Jev-Base-149M",
        help="Model ID for manifest.json.",
    )
    parser.add_argument(
        "--val-parquet",
        type=Path,
        default=None,
        help="Optional packed val_calib Parquet/JSONL path for post-INT8 temperature scaling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Target bundle directory (or parent directory in --smoke mode).",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=4,
        help="ORT CPU intra-op threads during calibration.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if args.smoke:
        t0 = time.perf_counter()
        tok, special_ids = build_smoke_jev_tokenizer()
        pad_id = int(special_ids["[PAD]"])
        val_rows = make_synthetic_curriculum_dataset(
            12, vocab_size=64, seed=42, pad_token_id=pad_id
        )
        tmp_ctx = (
            tempfile.TemporaryDirectory(prefix="diodemodel_export_smoke_")
            if args.output_dir is None
            else None
        )
        root_out = Path(args.output_dir) if args.output_dir is not None else Path(tmp_ctx.name)  # type: ignore[union-attr]
        root_out.mkdir(parents=True, exist_ok=True)
        try:
            specs = (
                ("DiodeMCP-Jev-Base-149M", _build_tiny_smoke_modernbert(64, pad_id)),
                ("DiodeMCP-NanoJev-0.6B", _build_tiny_smoke_nanojev(64, pad_id)),
            )
            for m_id, m_obj in specs:
                export_diode_bundle(
                    m_obj,
                    root_out / m_id,
                    tokenizer_source=tok,
                    model_id=m_id,
                    val_calib_rows=val_rows,
                    special_token_ids=special_ids,
                    calib_buckets=(128, 256),
                    intra_op_threads=args.intra_op_threads,
                )
            elapsed = time.perf_counter() - t0
            print(f"Export smoke OK: 2 INT8 bundles validated in {elapsed:.2f}s")
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()
        return 0

    if args.checkpoint is None or args.tokenizer is None or args.output_dir is None:
        raise ValueError(
            "--checkpoint, --tokenizer, and --output-dir are required when not running --smoke"
        )
    export_diode_bundle(
        args.checkpoint,
        args.output_dir,
        tokenizer_source=args.tokenizer,
        model_id=args.model_id,
        val_calib_rows=args.val_parquet,
        intra_op_threads=args.intra_op_threads,
    )
    return 0


__all__ = [
    "ExportBundleConfig",
    "HEAD_EXCLUSION_SCOPES",
    "ONNX_INPUT_NAMES",
    "ONNX_OUTPUT_NAMES",
    "QUANTIZE_OP_TYPES",
    "bake_onnx_logit_temperatures",
    "build_smoke_jev_tokenizer",
    "calibrate_int8_onnx_temperatures",
    "collect_excluded_onnx_nodes",
    "create_ort_cpu_session",
    "export_diode_bundle",
    "export_fp32_onnx",
    "make_dummy_export_batch",
    "quantize_onnx_matmul_int8",
    "read_onnx_logit_temperatures",
    "verify_onnx_no_bool_where",
]


if __name__ == "__main__":
    raise SystemExit(main())
