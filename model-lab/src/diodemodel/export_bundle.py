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
TWO_GIB_BYTES: Final[int] = 2 * 1024 * 1024 * 1024
DOWN_PROJ_SUBSTRINGS: Final[tuple[str, ...]] = ("mlp/down_proj", "mlp.down_proj")


@dataclass(frozen=True)
class ExportBundleConfig:
    """Configuration for DiodeModelBundle v1 export, INT8 quantization, and post-INT8 calibration."""

    model_id: str = "diodemcp-jev-base-int8"
    max_seq_len: int = 8192
    length_buckets: tuple[int, ...] = tuple(DEFAULT_LENGTH_BUCKETS)
    thresholds: Mapping[str, float] | None = None
    opset_version: int = ONNX_OPSET_VERSION
    weight_type: QuantType = QuantType.QInt8
    per_channel: bool = True
    reduce_range: bool = False
    calib_batch_size: int = 16
    calib_buckets: tuple[int, ...] = CURRICULUM_BUCKETS
    calib_max_rows: int | None = None
    flip_max_rows: int | None = None
    min_temp: float = 0.0
    max_flip_rate: float | None = None
    num_ece_bins: int = DEFAULT_ECE_BINS
    intra_op_threads: int = 4
    keep_fp32_onnx: bool = False


def _is_attack_calib_row(row: Mapping[str, Any] | PackedJevRow) -> bool:
    if isinstance(row, PackedJevRow):
        return max(float(row.y_enc), float(row.y_sem), float(row.y_hijack)) > 0.5
    if "noul_targets" in row:
        return max(float(v) for v in row["noul_targets"]) > 0.5
    return (
        max(
            float(row.get("y_enc", 0.0)),
            float(row.get("y_sem", 0.0)),
            float(row.get("y_hijack", 0.0)),
        )
        > 0.5
    )


def _calib_row_unpadded_length(row: Mapping[str, Any] | PackedJevRow) -> int:
    if isinstance(row, PackedJevRow):
        return int(row.unpadded_length)
    return int(row.get("unpadded_length", len(row.get("input_ids", ()))))


def select_stratified_calib_rows(
    rows: Sequence[Mapping[str, Any] | PackedJevRow],
    *,
    max_rows: int,
    max_bucket_len: int,
) -> list[Mapping[str, Any] | PackedJevRow]:
    """Select an interleaved, balanced attack/benign calibration subset within ``max_bucket_len``."""
    eligible = [r for r in rows if _calib_row_unpadded_length(r) <= max_bucket_len]
    if not eligible:
        eligible = list(rows)
    if max_rows <= 0 or len(eligible) <= max_rows:
        atk_all = [r for r in eligible if _is_attack_calib_row(r)]
        ben_all = [r for r in eligible if not _is_attack_calib_row(r)]
    else:
        target_atk = max_rows // 2
        target_ben = max_rows - target_atk
        atk_pool = [r for r in eligible if _is_attack_calib_row(r)]
        ben_pool = [r for r in eligible if not _is_attack_calib_row(r)]
        atk_all = atk_pool[:target_atk]
        ben_all = ben_pool[:target_ben]
        if len(atk_all) < target_atk:
            extra = target_atk - len(atk_all)
            ben_all = ben_pool[: target_ben + extra]
        elif len(ben_all) < target_ben:
            extra = target_ben - len(ben_all)
            atk_all = atk_pool[: target_atk + extra]

    interleaved: list[Mapping[str, Any] | PackedJevRow] = []
    for idx in range(max(len(atk_all), len(ben_all))):
        if idx < len(atk_all):
            interleaved.append(atk_all[idx])
        if idx < len(ben_all):
            interleaved.append(ben_all[idx])
    return interleaved


def collect_nodes_with_substrings(
    model_or_path: onnx.ModelProto | Path | str,
    substrings: Sequence[str],
) -> list[str]:
    """Return quantizable ONNX node names (and ORT ``_MatMul`` aliases) matching any substring."""
    if not substrings:
        return []
    needles = tuple(s.lower() for s in substrings if s)
    if not needles:
        return []
    proto = (
        onnx.load(str(model_or_path), load_external_data=False)
        if isinstance(model_or_path, (str, Path))
        else model_or_path
    )
    matched: list[str] = []
    seen: set[str] = set()
    for node in proto.graph.node:
        if node.op_type not in QUANTIZE_OP_TYPES:
            continue
        haystack = " ".join((node.name, *node.input, *node.output)).lower()
        if any(needle in haystack for needle in needles):
            for candidate in (node.name, f"{node.name}_MatMul"):
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    matched.append(candidate)
    return matched


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
        onnx.load(str(model_or_path), load_external_data=False)
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
        onnx.load(str(model_or_path), load_external_data=False)
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
    first_param = next(model.parameters())
    orig_device = first_param.device
    orig_dtype = first_param.dtype
    cpu_model = model.to(device="cpu", dtype=torch.float32).eval()
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
        model.to(device=orig_device, dtype=orig_dtype)
        if was_training:
            model.train()

    proto = onnx.load(str(out_path), load_external_data=False)
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
    per_channel: bool = True,
    reduce_range: bool = False,
    extra_excluded_substrings: Sequence[str] = (),
) -> dict[str, Any]:
    """Quantize ONLY backbone ``MatMul`` and ``Gemm`` weight projections to INT8 while keeping
    ``LayerNorm``, ``Softmax``, 4D mask ops, and ``HEAD_SCOPE_NAMES`` in pure FP32 (`I8`)."""
    src_path = Path(fp32_onnx_path)
    dst_path = Path(int8_onnx_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.resolve() != dst_path.resolve():
        for leftover in dst_path.parent.glob(f"{dst_path.name}*"):
            if leftover.is_file():
                leftover.unlink()

    fp32_proto = onnx.load(str(src_path), load_external_data=False)
    add_pre_process_metadata(fp32_proto)
    pin_onnx_output_shapes(fp32_proto)
    onnx.save(fp32_proto, str(src_path))

    base_excluded = collect_excluded_onnx_nodes(fp32_proto)
    extra_excluded = collect_nodes_with_substrings(fp32_proto, extra_excluded_substrings)
    seen_excl = set(base_excluded)
    nodes_to_exclude = list(base_excluded)
    for name in extra_excluded:
        if name not in seen_excl:
            seen_excl.add(name)
            nodes_to_exclude.append(name)

    quantize_dynamic(
        model_input=src_path,
        model_output=dst_path,
        op_types_to_quantize=list(QUANTIZE_OP_TYPES),
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=weight_type,
        nodes_to_exclude=nodes_to_exclude,
        extra_options={"MatMulConstBOnly": True},
    )

    q_proto = onnx.load(str(dst_path), load_external_data=False)
    pin_onnx_output_shapes(q_proto)
    verify_onnx_no_bool_where(q_proto)
    onnx.save(q_proto, str(dst_path))
    validate_onnx_contract(dst_path)

    for init in q_proto.graph.initializer:
        if init.data_location == TensorProto.EXTERNAL:
            raise ValueError(
                f"INT8 ONNX initializer {init.name!r} uses external data; single-file <2 GiB bundle required"
            )
    onnx_bytes = int(dst_path.stat().st_size)
    if onnx_bytes >= TWO_GIB_BYTES:
        raise ValueError(
            f"Quantized model_int8.onnx size ({onnx_bytes} bytes) exceeds 2 GiB single-file limit"
        )

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
        "reduce_range": bool(reduce_range),
        "nodes_to_exclude": nodes_to_exclude,
        "extra_excluded_nodes": extra_excluded,
        "quantized_matmul_nodes": quantized_matmul_nodes,
        "dynamic_quantize_linear_nodes": dql_nodes,
        "onnx_bytes": onnx_bytes,
    }


quantize_onnx_int8 = quantize_onnx_matmul_int8


def read_onnx_logit_temperatures(onnx_path_or_model: Path | str | onnx.ModelProto) -> np.ndarray:
    """Read the 3-element FP32 ``logit_temperatures`` initializer from an ONNX model."""
    is_path = isinstance(onnx_path_or_model, (str, Path))
    proto = (
        onnx.load(str(onnx_path_or_model), load_external_data=False)
        if is_path
        else onnx_path_or_model
    )
    for init in proto.graph.initializer:
        if init.name == "logit_temperatures":
            if init.data_location == TensorProto.EXTERNAL and is_path:
                onnx.external_data_helper.load_external_data_for_tensor(
                    init, str(Path(onnx_path_or_model).parent)  # type: ignore[arg-type]
                )
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

    proto = onnx.load(str(path), load_external_data=False)
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


def _fit_threshold_pair_from_scores(
    scores: np.ndarray,
    benign_mask: np.ndarray,
    *,
    warn_fpr_target: float = 0.015,
    critical_fpr_target: float = 0.001,
) -> dict[str, float]:
    """Fit smallest ``warn_min`` in ``[0.15, 0.45]`` and ``critical_min`` in ``[0.55, 0.92]``."""
    b_scores = np.asarray(scores[benign_mask], dtype=np.float64)
    if b_scores.size == 0:
        return {"warn_min": 0.20, "critical_min": 0.80}

    # Candidate grid in [0.15, 0.45] including observed benign points + eps
    warn_cands = np.unique(
        np.concatenate(
            [
                np.linspace(0.15, 0.45, 301),
                np.clip(b_scores + 1e-4, 0.15, 0.45),
            ]
        )
    )
    warn_min = 0.45
    for cand in warn_cands:
        fpr = float((b_scores >= cand).mean())
        if fpr <= float(warn_fpr_target) + 1e-12:
            warn_min = float(cand)
            break
    warn_min = max(0.15, min(0.45, float(warn_min)))

    min_crit = max(0.55, warn_min + 0.15 + 1e-4)
    if min_crit >= 0.92:
        critical_min = 0.92
    else:
        crit_cands = np.unique(
            np.concatenate(
                [
                    np.linspace(min_crit, 0.92, 371),
                    np.clip(b_scores + 1e-4, min_crit, 0.92),
                ]
            )
        )
        critical_min = 0.92
        for cand in crit_cands:
            fpr = float((b_scores >= cand).mean())
            if fpr <= float(critical_fpr_target) + 1e-12:
                critical_min = float(cand)
                break
        critical_min = max(min_crit, min(0.92, float(critical_min)))

    return {"warn_min": round(warn_min, 6), "critical_min": round(critical_min, 6)}


def fit_data_thresholds(
    probs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
) -> dict[str, Any]:
    """Fit data-driven ``warn_min``, ``critical_min``, and ``per_head_thresholds`` from calibration probs."""
    p_np = np.asarray(probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else probs, dtype=np.float64)
    y_np = np.asarray(targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets, dtype=np.float64)
    if p_np.ndim != 2 or p_np.shape[1] != NUM_NOUL_HEADS:
        raise ValueError(f"Expected probs shape (N, {NUM_NOUL_HEADS}), got {p_np.shape}")

    p_max = p_np.max(axis=-1)
    y_max = y_np.max(axis=-1)
    overall = _fit_threshold_pair_from_scores(p_max, y_max < 0.5)

    head_names = ("enc", "sem", "hijack")
    per_head: dict[str, dict[str, float]] = {}
    for h_idx, h_name in enumerate(head_names):
        per_head[h_name] = _fit_threshold_pair_from_scores(p_np[:, h_idx], y_np[:, h_idx] < 0.5)

    return {
        "warn_min": overall["warn_min"],
        "critical_min": overall["critical_min"],
        "per_head_thresholds": per_head,
    }


def _compute_decision_bands(p_max: np.ndarray, *, warn_min: float, critical_min: float) -> np.ndarray:
    """Map max risk probabilities to 3 decision bands: 0=ALLOW, 1=WARN, 2=BLOCK."""
    bands = np.zeros(p_max.shape, dtype=np.int64)
    bands[p_max >= float(warn_min)] = 1
    bands[p_max >= float(critical_min)] = 2
    return bands


def _compute_fp32_int8_band_flip_rate(
    fp32_onnx_path: Path | str,
    int8_onnx_path: Path | str,
    rows: Sequence[Mapping[str, Any] | PackedJevRow] | None,
    *,
    fitted_temperatures: torch.Tensor,
    warn_min: float,
    critical_min: float,
    batch_size: int = 16,
    calib_buckets: Sequence[int] = CURRICULUM_BUCKETS,
    pad_token_id: int = 0,
    intra_op_threads: int = 4,
    vocab_size: int = 64,
) -> float:
    """Compute fraction of rows whose 3-band decision differs between FP32 ONNX and INT8 ONNX."""
    t_fp32_init = torch.from_numpy(read_onnx_logit_temperatures(fp32_onnx_path).copy())
    fp32_sess = create_ort_cpu_session(fp32_onnx_path, intra_op_threads=intra_op_threads)
    int8_sess = create_ort_cpu_session(int8_onnx_path, intra_op_threads=intra_op_threads)

    fp32_probs_all: list[torch.Tensor] = []
    int8_probs_all: list[torch.Tensor] = []

    if rows:
        eff_bs = max(1, int(batch_size))
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
            fp32_out, _, _ = fp32_sess.run(list(ONNX_OUTPUT_NAMES), feed)
            int8_out, _, _ = int8_sess.run(list(ONNX_OUTPUT_NAMES), feed)
            fp32_raw = probs_to_logits(
                torch.from_numpy(np.asarray(fp32_out, dtype=np.float32)), t_fp32_init
            )
            fp32_cal = torch.sigmoid(fp32_raw.double() / fitted_temperatures.double()).float()
            fp32_probs_all.append(fp32_cal)
            int8_probs_all.append(torch.from_numpy(np.asarray(int8_out, dtype=np.float32)))
    else:
        dummy = make_dummy_export_batch(
            batch_size=4, seq_len=64, vocab_size=vocab_size, pad_token_id=pad_token_id, device="cpu"
        )
        feed = {name: dummy[name].numpy() for name in ONNX_INPUT_NAMES}
        fp32_out, _, _ = fp32_sess.run(list(ONNX_OUTPUT_NAMES), feed)
        int8_out, _, _ = int8_sess.run(list(ONNX_OUTPUT_NAMES), feed)
        fp32_raw = probs_to_logits(
            torch.from_numpy(np.asarray(fp32_out, dtype=np.float32)), t_fp32_init
        )
        fp32_cal = torch.sigmoid(fp32_raw.double() / fitted_temperatures.double()).float()
        fp32_probs_all.append(fp32_cal)
        int8_probs_all.append(torch.from_numpy(np.asarray(int8_out, dtype=np.float32)))

    fp32_pmax = torch.cat(fp32_probs_all, dim=0).amax(dim=-1).numpy()
    int8_pmax = torch.cat(int8_probs_all, dim=0).amax(dim=-1).numpy()
    fp32_bands = _compute_decision_bands(fp32_pmax, warn_min=warn_min, critical_min=critical_min)
    int8_bands = _compute_decision_bands(int8_pmax, warn_min=warn_min, critical_min=critical_min)
    return float((fp32_bands != int8_bands).mean())


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

    fitted_thresh_info = fit_data_thresholds(p_calibrated, val_targets)
    fitted_pair = {
        "warn_min": fitted_thresh_info["warn_min"],
        "critical_min": fitted_thresh_info["critical_min"],
    }
    calib_thresholds = (
        clamp_thresholds(dict(thresholds)) if thresholds is not None else dict(fitted_pair)
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
        "thresholds": calib_thresholds,
        "fitted_thresholds": fitted_pair,
        "per_head_thresholds": fitted_thresh_info["per_head_thresholds"],
    }
    return {
        "fitted_temperatures": fitted_temps,
        "previous_temperatures": t_prev,
        "int8_raw_logits": int8_raw_logits,
        "val_targets": val_targets,
        "calibration_payload": calibration_payload,
        "fitted_thresholds": fitted_thresh_info,
        "report": calib_rep,
    }


def _apply_min_temperature_floor(
    int8_onnx_path: Path | str,
    cal_res: dict[str, Any],
    *,
    min_temp: float,
    explicit_thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Clamp post-INT8 temperatures at ``min_temp``, re-bake ONNX weights, and refresh metrics."""
    payload = dict(cal_res["calibration_payload"])
    if min_temp > 0.0:
        payload["min_temp"] = float(min_temp)
    fitted_temps: torch.Tensor = cal_res["fitted_temperatures"]

    if min_temp > 0.0 and bool((fitted_temps < float(min_temp)).any()):
        clamped_temps = torch.clamp(fitted_temps, min=float(min_temp))
        bake_onnx_logit_temperatures(int8_onnx_path, clamped_temps)

        int8_raw_logits: torch.Tensor = cal_res["int8_raw_logits"]
        val_targets: torch.Tensor = cal_res["val_targets"]
        calib_rep = temperature_calibration_report(
            int8_raw_logits,
            val_targets,
            clamped_temps,
            num_bins=int(payload["num_bins"]),
        )
        y_f64 = val_targets.double()
        p_calibrated = torch.sigmoid(int8_raw_logits.double() / clamped_temps.double())
        brier_after = [
            float(((p_calibrated[:, h] - y_f64[:, h]) ** 2).mean().item())
            for h in range(NUM_NOUL_HEADS)
        ]
        fitted_thresh_info = fit_data_thresholds(p_calibrated, val_targets)
        fitted_pair = {
            "warn_min": fitted_thresh_info["warn_min"],
            "critical_min": fitted_thresh_info["critical_min"],
        }
        t_list = [float(x) for x in clamped_temps.detach().cpu().tolist()]
        payload["unclamped_temperatures"] = list(payload["temperatures"])
        payload["temperatures"] = t_list
        payload["temperatures_by_head"] = {
            "T_enc": t_list[0],
            "T_sem": t_list[1],
            "T_hijack": t_list[2],
        }
        payload["ece_before"] = [float(x) for x in calib_rep["ece_before"]]
        payload["ece_after"] = [float(x) for x in calib_rep["ece_after"]]
        payload["ece_val"] = float(calib_rep["ece_val"])
        payload["brier_after"] = brier_after
        payload["brier_val"] = float(np.mean(brier_after))
        payload["nll_after"] = [float(x) for x in calib_rep["nll_after"]]
        payload["nll_val"] = float(np.mean(calib_rep["nll_after"]))
        payload["thresholds"] = (
            dict(explicit_thresholds) if explicit_thresholds is not None else dict(fitted_pair)
        )
        payload["fitted_thresholds"] = fitted_pair
        payload["per_head_thresholds"] = fitted_thresh_info["per_head_thresholds"]

        cal_res = dict(cal_res)
        cal_res["fitted_temperatures"] = clamped_temps
        cal_res["report"] = calib_rep

    cal_res = dict(cal_res)
    cal_res["calibration_payload"] = payload
    return cal_res


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
    per_channel: bool = True,
    reduce_range: bool = False,
    calib_batch_size: int = 16,
    calib_buckets: Sequence[int] = CURRICULUM_BUCKETS,
    calib_max_rows: int | None = None,
    flip_max_rows: int | None = None,
    min_temp: float = 0.0,
    max_flip_rate: float | None = None,
    extra_excluded_substrings: Sequence[str] = (),
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
       4D mask ops, ``LayerNorm``, and ``Softmax`` (with optional automatic ``mlp.down_proj`` outlier fallback
       when ``max_flip_rate`` is set).
    4. Runs ``val_calib_rows`` through the quantized INT8 ORT session, inverts probabilities to recover
       post-INT8 pre-temperature logits, fits ``(T_enc, T_sem, T_hijack)`` via ``fit_temperature_scaling``,
       applies optional ``min_temp`` clamping, and bakes the fitted FP32 temperatures into ``model_int8.onnx``.
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
        reduce_range = config.reduce_range
        calib_batch_size = config.calib_batch_size
        calib_buckets = config.calib_buckets
        calib_max_rows = config.calib_max_rows if calib_max_rows is None else calib_max_rows
        flip_max_rows = config.flip_max_rows if flip_max_rows is None else flip_max_rows
        min_temp = config.min_temp if min_temp == 0.0 else min_temp
        max_flip_rate = config.max_flip_rate if max_flip_rate is None else max_flip_rate
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

    # 3-4. Export FP32 opset-17 ONNX, quantize backbone MatMul/Gemm to INT8, calibrate & check FP32<->INT8 flips
    int8_onnx_path = out_dir / "model_int8.onnx"
    kept_fp32_path: Path | None = None
    int8_raw_logits: torch.Tensor | None = None
    val_targets: torch.Tensor | None = None
    explicit_thresholds = clamp_thresholds(dict(thresholds)) if thresholds is not None else None
    eff_calib_buckets = tuple(sorted(int(b) for b in calib_buckets)) or CURRICULUM_BUCKETS
    loaded_calib_rows = (
        load_packed_jev_dataset(val_calib_rows) if val_calib_rows is not None else None
    )
    if loaded_calib_rows is not None and calib_max_rows is not None and calib_max_rows > 0:
        loaded_calib_rows = select_stratified_calib_rows(
            loaded_calib_rows,
            max_rows=int(calib_max_rows),
            max_bucket_len=eff_calib_buckets[-1],
        )
    flip_eval_rows = (
        loaded_calib_rows[: int(flip_max_rows)]
        if (loaded_calib_rows is not None and flip_max_rows is not None and flip_max_rows > 0)
        else loaded_calib_rows
    )

    eff_vocab = int(
        getattr(getattr(getattr(export_model, "backbone", None), "config", None), "vocab_size", 64)
    )
    fallback_used = False

    with tempfile.TemporaryDirectory(prefix="diodemodel_fp32_") as tmp_dir:
        if keep_fp32_onnx:
            fp32_dir = out_dir.parent / f"{out_dir.name}_fp32"
            fp32_dir.mkdir(parents=True, exist_ok=True)
            fp32_path = fp32_dir / "model_fp32.onnx"
            kept_fp32_path = fp32_path
        else:
            fp32_path = Path(tmp_dir) / "model_fp32.onnx"
        export_fp32_onnx(
            export_model,
            fp32_path,
            opset_version=opset_version,
        )

        def _quantize_calibrate_and_check(
            substrings: Sequence[str],
        ) -> tuple[
            dict[str, Any],
            torch.Tensor,
            dict[str, Any],
            torch.Tensor | None,
            torch.Tensor | None,
            float,
        ]:
            q_info = quantize_onnx_matmul_int8(
                fp32_path,
                int8_onnx_path,
                weight_type=weight_type,
                per_channel=per_channel,
                reduce_range=reduce_range,
                extra_excluded_substrings=substrings,
            )
            if loaded_calib_rows is not None:
                c_res = calibrate_int8_onnx_temperatures(
                    int8_onnx_path,
                    loaded_calib_rows,
                    batch_size=calib_batch_size,
                    calib_buckets=eff_calib_buckets,
                    pad_token_id=pad_token_id,
                    num_bins=num_ece_bins,
                    thresholds=explicit_thresholds,
                    intra_op_threads=intra_op_threads,
                )
                if min_temp > 0.0:
                    c_res = _apply_min_temperature_floor(
                        int8_onnx_path,
                        c_res,
                        min_temp=min_temp,
                        explicit_thresholds=explicit_thresholds,
                    )
                f_temps: torch.Tensor = c_res["fitted_temperatures"]
                export_model.set_logit_temperatures(f_temps)
                c_payload = dict(c_res["calibration_payload"])
                z_raw: torch.Tensor | None = c_res["int8_raw_logits"]
                y_tgt: torch.Tensor | None = c_res["val_targets"]
            else:
                t_cur = read_onnx_logit_temperatures(int8_onnx_path)
                f_temps = torch.from_numpy(t_cur.copy())
                t_list = [float(x) for x in t_cur.tolist()]
                fallback_thresh = (
                    explicit_thresholds
                    if explicit_thresholds is not None
                    else {"warn_min": 0.20, "critical_min": 0.80}
                )
                c_payload = {
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
                    "thresholds": fallback_thresh,
                    "per_head_thresholds": {
                        "enc": dict(fallback_thresh),
                        "sem": dict(fallback_thresh),
                        "hijack": dict(fallback_thresh),
                    },
                }
                z_raw = None
                y_tgt = None

            eff_t = c_payload["thresholds"]
            flip_r = _compute_fp32_int8_band_flip_rate(
                fp32_path,
                int8_onnx_path,
                flip_eval_rows,
                fitted_temperatures=f_temps,
                warn_min=float(eff_t["warn_min"]),
                critical_min=float(eff_t["critical_min"]),
                batch_size=calib_batch_size,
                calib_buckets=eff_calib_buckets,
                pad_token_id=pad_token_id,
                intra_op_threads=intra_op_threads,
                vocab_size=eff_vocab,
            )
            return q_info, f_temps, c_payload, z_raw, y_tgt, flip_r

        (
            quant_info,
            fitted_temperatures,
            calibration_payload,
            int8_raw_logits,
            val_targets,
            fp32_int8_band_flip_rate,
        ) = _quantize_calibrate_and_check(extra_excluded_substrings)

        if (
            max_flip_rate is not None
            and fp32_int8_band_flip_rate > float(max_flip_rate)
            and not any(s in extra_excluded_substrings for s in DOWN_PROJ_SUBSTRINGS)
        ):
            fallback_used = True
            (
                quant_info,
                fitted_temperatures,
                calibration_payload,
                int8_raw_logits,
                val_targets,
                fp32_int8_band_flip_rate,
            ) = _quantize_calibrate_and_check((*extra_excluded_substrings, *DOWN_PROJ_SUBSTRINGS))

        calibration_payload["fp32_int8_band_flip_rate"] = fp32_int8_band_flip_rate
        calibration_payload["per_channel"] = bool(per_channel)
        calibration_payload["reduce_range"] = bool(reduce_range)
        calibration_payload["down_proj_fallback_used"] = bool(fallback_used)

    clamped_thresh = clamp_thresholds(dict(calibration_payload["thresholds"]))

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
        "fp32_onnx_path": kept_fp32_path,
        "manifest": validated_manifest,
        "calibration": calibration_payload,
        "quantization": quant_info,
        "fitted_temperatures": fitted_temperatures,
        "int8_raw_logits": int8_raw_logits,
        "val_targets": val_targets,
        "fp32_int8_band_flip_rate": fp32_int8_band_flip_rate,
        "fallback_used": fallback_used,
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
    parser.add_argument("--calib-max-rows", type=int, default=None)
    parser.add_argument("--flip-max-rows", type=int, default=None)
    parser.add_argument("--min-temp", type=float, default=0.0)
    parser.add_argument("--max-flip-rate", type=float, default=None)
    parser.add_argument("--calib-buckets", type=str, default=None)
    parser.add_argument(
        "--reduce-range",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use 7-bit reduce_range quantization (prevents AVX2 VPMADDUBSW overflow).",
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
    parsed_buckets = (
        tuple(int(x.strip()) for x in args.calib_buckets.split(",") if x.strip())
        if args.calib_buckets
        else CURRICULUM_BUCKETS
    )
    export_diode_bundle(
        args.checkpoint,
        args.output_dir,
        tokenizer_source=args.tokenizer,
        model_id=args.model_id,
        val_calib_rows=args.val_parquet,
        calib_buckets=parsed_buckets,
        calib_max_rows=args.calib_max_rows,
        flip_max_rows=args.flip_max_rows,
        min_temp=args.min_temp,
        max_flip_rate=args.max_flip_rate,
        reduce_range=args.reduce_range,
        intra_op_threads=args.intra_op_threads,
    )
    return 0


__all__ = [
    "DOWN_PROJ_SUBSTRINGS",
    "ExportBundleConfig",
    "HEAD_EXCLUSION_SCOPES",
    "ONNX_INPUT_NAMES",
    "ONNX_OUTPUT_NAMES",
    "QUANTIZE_OP_TYPES",
    "TWO_GIB_BYTES",
    "bake_onnx_logit_temperatures",
    "build_smoke_jev_tokenizer",
    "calibrate_int8_onnx_temperatures",
    "collect_excluded_onnx_nodes",
    "collect_nodes_with_substrings",
    "create_ort_cpu_session",
    "export_diode_bundle",
    "export_fp32_onnx",
    "fit_data_thresholds",
    "make_dummy_export_batch",
    "quantize_onnx_int8",
    "quantize_onnx_matmul_int8",
    "read_onnx_logit_temperatures",
    "select_stratified_calib_rows",
    "verify_onnx_no_bool_where",
]


if __name__ == "__main__":
    raise SystemExit(main())
