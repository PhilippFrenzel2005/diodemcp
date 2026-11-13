"""CPU-only INT8 ONNX exporter for DiodeMCP-NanoJev-0.6B with >2 GiB FP32 staging
and automatic ``mlp.down_proj`` exclusion fallback.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Final

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.quant_utils import add_pre_process_metadata
import numpy as np
from tokenizers import Tokenizer
import torch

from diodemodel.contract import (
    CANONICAL_SEGMENT_IDS,
    DEFAULT_LENGTH_BUCKETS,
    ONNX_OPSET_VERSION,
    BundleManifest,
    clamp_thresholds,
    compute_sha256,
    validate_bundle_dir,
    validate_onnx_contract,
)
from diodemodel.dataset_packer import PackedJevRow
from diodemodel.export_bundle import (
    QUANTIZE_OP_TYPES,
    _compute_fp32_int8_band_flip_rate,
    _materialize_bundle_tokenizer,
    bake_onnx_logit_temperatures,
    calibrate_int8_onnx_temperatures,
    collect_excluded_onnx_nodes,
    export_fp32_onnx,
    fit_data_thresholds,
    verify_onnx_no_bool_where,
)
from diodemodel.losses_rlcd import temperature_calibration_report
from diodemodel.modeling_jev import ARG_SLOTS, NUM_NOUL_HEADS, SRC_SLOTS, pin_onnx_output_shapes
from diodemodel.train import (
    _instantiate_model_from_checkpoint_payload,
    load_checkpoint,
    load_packed_jev_dataset,
)

TWO_GIB_BYTES: Final[int] = 2 * 1024 * 1024 * 1024
DOWN_PROJ_SUBSTRINGS: Final[tuple[str, ...]] = ("mlp/down_proj", "mlp.down_proj")
DEFAULT_CALIB_BUCKETS: Final[tuple[int, ...]] = (256, 512)


def _is_attack_row(row: Mapping[str, Any] | PackedJevRow) -> bool:
    if isinstance(row, PackedJevRow):
        return max(float(row.y_enc), float(row.y_sem), float(row.y_hijack)) > 0.5
    if "noul_targets" in row:
        return max(float(v) for v in row["noul_targets"]) > 0.5
    return max(
        float(row.get("y_enc", 0.0)),
        float(row.get("y_sem", 0.0)),
        float(row.get("y_hijack", 0.0)),
    ) > 0.5


def _row_unpadded_length(row: Mapping[str, Any] | PackedJevRow) -> int:
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
    eligible = [r for r in rows if _row_unpadded_length(r) <= max_bucket_len]
    if not eligible:
        eligible = list(rows)
    if max_rows <= 0 or len(eligible) <= max_rows:
        atk_all = [r for r in eligible if _is_attack_row(r)]
        ben_all = [r for r in eligible if not _is_attack_row(r)]
    else:
        target_atk = max_rows // 2
        target_ben = max_rows - target_atk
        atk_pool = [r for r in eligible if _is_attack_row(r)]
        ben_pool = [r for r in eligible if not _is_attack_row(r)]
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
    onnx_path: Path | str,
    substrings: Sequence[str],
) -> list[str]:
    """Return quantizable ONNX node names (and ORT ``_MatMul`` aliases) matching any substring."""
    if not substrings:
        return []
    needles = tuple(s.lower() for s in substrings if s)
    proto = onnx.load(str(onnx_path), load_external_data=False)
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


def _quantize_fp32_to_single_int8(
    fp32_onnx_path: Path,
    int8_onnx_path: Path,
    *,
    per_channel: bool = True,
    reduce_range: bool = True,
    extra_excluded_substrings: Sequence[str] = (),
) -> tuple[dict[str, Any], list[str]]:
    """Quantize backbone MatMul/Gemm nodes from ``fp32_onnx_path`` into a single ``<2 GiB`` ONNX file."""
    int8_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    for leftover in int8_onnx_path.parent.glob(f"{int8_onnx_path.name}*"):
        if leftover.is_file():
            leftover.unlink()

    fp32_proto = onnx.load(str(fp32_onnx_path), load_external_data=False)
    add_pre_process_metadata(fp32_proto)
    pin_onnx_output_shapes(fp32_proto)
    onnx.save(fp32_proto, str(fp32_onnx_path))

    base_excluded = collect_excluded_onnx_nodes(fp32_proto)
    extra_excluded = collect_nodes_with_substrings(fp32_onnx_path, extra_excluded_substrings)
    seen = set(base_excluded)
    nodes_to_exclude = list(base_excluded)
    for name in extra_excluded:
        if name not in seen:
            seen.add(name)
            nodes_to_exclude.append(name)

    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        op_types_to_quantize=list(QUANTIZE_OP_TYPES),
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude,
        extra_options={"MatMulConstBOnly": True},
    )

    q_proto = onnx.load(str(int8_onnx_path), load_external_data=False)
    pin_onnx_output_shapes(q_proto)
    verify_onnx_no_bool_where(q_proto)
    onnx.save(q_proto, str(int8_onnx_path))
    validate_onnx_contract(int8_onnx_path)

    for init in q_proto.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError(
                f"INT8 ONNX initializer {init.name!r} uses external data; single-file <2 GiB bundle required"
            )
    size_bytes = int8_onnx_path.stat().st_size
    if size_bytes >= TWO_GIB_BYTES:
        raise ValueError(
            f"Quantized model_int8.onnx size ({size_bytes} bytes) exceeds 2 GiB single-file limit"
        )

    quant_info = {
        "op_types_to_quantize": list(QUANTIZE_OP_TYPES),
        "weight_type": str(QuantType.QInt8),
        "per_channel": bool(per_channel),
        "reduce_range": bool(reduce_range),
        "nodes_to_exclude_count": len(nodes_to_exclude),
        "extra_excluded_nodes": extra_excluded,
        "quantized_matmul_count": sum(1 for n in q_proto.graph.node if n.op_type == "MatMulInteger"),
        "onnx_bytes": size_bytes,
    }
    return quant_info, extra_excluded


def _apply_min_temperature_floor(
    int8_onnx_path: Path | str,
    cal_res: dict[str, Any],
    *,
    min_temp: float,
) -> dict[str, Any]:
    """Clamp post-INT8 temperatures at ``min_temp``, re-bake ONNX weights, and refresh metrics."""
    payload = dict(cal_res["calibration_payload"])
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
        payload["thresholds"] = dict(fitted_pair)
        payload["fitted_thresholds"] = fitted_pair
        payload["per_head_thresholds"] = fitted_thresh_info["per_head_thresholds"]

        cal_res = dict(cal_res)
        cal_res["fitted_temperatures"] = clamped_temps
        cal_res["report"] = calib_rep

    cal_res = dict(cal_res)
    cal_res["calibration_payload"] = payload
    return cal_res


def export_nanojev_with_outlier_fallback(
    *,
    checkpoint_path: Path | str | None,
    tokenizer_path: Path | str,
    val_calib_path: Path | str,
    output_bundle: Path | str,
    model_id: str = "DiodeMCP-NanoJev-0.6B",
    max_flip_rate: float = 0.05,
    calib_max_rows: int = 512,
    flip_max_rows: int = 32,
    calib_batch_size: int = 16,
    min_temp: float = 0.75,
    calib_buckets: Sequence[int] = DEFAULT_CALIB_BUCKETS,
    length_buckets: Sequence[int] = tuple(DEFAULT_LENGTH_BUCKETS),
    max_seq_len: int = 8192,
    intra_op_threads: int | None = None,
    per_channel: bool = True,
    reduce_range: bool = True,
    staging_dir: Path | str | None = None,
    reuse_fp32: bool = False,
) -> dict[str, Any]:
    """Export a NanoJev checkpoint on CPU to a verified single-file INT8 ``DiodeModelBundle v1``."""
    out_dir = Path(output_bundle)
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = max(1, int(intra_op_threads if intra_op_threads is not None else (os.cpu_count() or 4)))

    tokenizer_json_path = out_dir / "tokenizer.json"
    resolved_special_ids = _materialize_bundle_tokenizer(
        tokenizer_path,
        tokenizer_json_path,
    )
    pad_token_id = int(resolved_special_ids["[PAD]"])
    eff_vocab = int(Tokenizer.from_file(str(tokenizer_json_path)).get_vocab_size())

    eff_buckets = tuple(sorted(int(b) for b in calib_buckets)) or DEFAULT_CALIB_BUCKETS
    all_val_rows = load_packed_jev_dataset(val_calib_path)
    val_rows = select_stratified_calib_rows(
        all_val_rows,
        max_rows=calib_max_rows,
        max_bucket_len=eff_buckets[-1],
    )
    flip_rows = val_rows[:flip_max_rows] if flip_max_rows > 0 else val_rows

    int8_onnx_path = out_dir / "model_int8.onnx"

    tmp_ctx = (
        tempfile.TemporaryDirectory(prefix="nanojev_fp32_") if staging_dir is None else None
    )
    stage_root = Path(staging_dir) if staging_dir is not None else Path(tmp_ctx.name)  # type: ignore[union-attr]
    stage_root.mkdir(parents=True, exist_ok=True)

    try:
        fp32_path = stage_root / "model_fp32.onnx"
        reused_fp32 = bool(reuse_fp32 and fp32_path.is_file())
        if not reused_fp32:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required when FP32 ONNX is not reused")
            ckpt_payload = load_checkpoint(checkpoint_path, restore_rng=False, map_location="cpu")
            export_model = _instantiate_model_from_checkpoint_payload(ckpt_payload, device="cpu").float()
            eff_vocab = int(
                getattr(
                    getattr(getattr(export_model, "backbone", None), "config", None),
                    "vocab_size",
                    eff_vocab,
                )
            )
            export_fp32_onnx(
                export_model,
                fp32_path,
                opset_version=ONNX_OPSET_VERSION,
            )
            del export_model, ckpt_payload

        def _quantize_and_calibrate(
            extra_substrings: Sequence[str],
        ) -> tuple[dict[str, Any], list[str], dict[str, Any], float]:
            q_info, extra_nodes = _quantize_fp32_to_single_int8(
                fp32_path,
                int8_onnx_path,
                per_channel=per_channel,
                reduce_range=reduce_range,
                extra_excluded_substrings=extra_substrings,
            )
            cal_res = calibrate_int8_onnx_temperatures(
                int8_onnx_path,
                val_rows,
                batch_size=calib_batch_size,
                calib_buckets=eff_buckets,
                pad_token_id=pad_token_id,
                intra_op_threads=threads,
            )
            cal_res = _apply_min_temperature_floor(
                int8_onnx_path,
                cal_res,
                min_temp=min_temp,
            )
            cal_payload = cal_res["calibration_payload"]
            eff_thresh = cal_payload["thresholds"]
            flip_rate = _compute_fp32_int8_band_flip_rate(
                fp32_path,
                int8_onnx_path,
                flip_rows,
                fitted_temperatures=cal_res["fitted_temperatures"],
                warn_min=float(eff_thresh["warn_min"]),
                critical_min=float(eff_thresh["critical_min"]),
                batch_size=calib_batch_size,
                calib_buckets=eff_buckets,
                pad_token_id=pad_token_id,
                intra_op_threads=threads,
                vocab_size=eff_vocab,
            )
            return q_info, extra_nodes, cal_res, flip_rate

        quant_info, extra_excluded_nodes, calib_res, flip_rate = _quantize_and_calibrate(())
        fallback_used = False

        if flip_rate > float(max_flip_rate):
            fallback_used = True
            quant_info, extra_excluded_nodes, calib_res, flip_rate = _quantize_and_calibrate(
                DOWN_PROJ_SUBSTRINGS
            )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    calibration_payload = dict(calib_res["calibration_payload"])
    calibration_payload["fp32_int8_band_flip_rate"] = float(flip_rate)
    calibration_payload["flip_eval_rows"] = len(flip_rows)
    calibration_payload["per_channel"] = bool(per_channel)
    calibration_payload["reduce_range"] = bool(reduce_range)
    calibration_payload["down_proj_fallback_used"] = bool(fallback_used)
    calibration_payload["extra_excluded_nodes_count"] = len(extra_excluded_nodes)

    clamped_thresh = clamp_thresholds(dict(calibration_payload["thresholds"]))
    calibration_payload["thresholds"] = clamped_thresh

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

    validated_manifest = validate_bundle_dir(out_dir, verify_sha256=True)

    sidecars = [
        p.name
        for p in out_dir.iterdir()
        if p.name
        not in {
            "model_int8.onnx",
            "tokenizer.json",
            "calibration.json",
            "manifest.json",
        }
    ]
    if sidecars:
        raise ValueError(f"Unexpected sidecar files in bundle directory {out_dir}: {sidecars}")

    return {
        "bundle_dir": str(out_dir),
        "model_id": validated_manifest.model_id,
        "reused_fp32": reused_fp32,
        "fallback_used": fallback_used,
        "extra_excluded_nodes": extra_excluded_nodes,
        "fp32_int8_band_flip_rate": float(flip_rate),
        "ece_val": float(validated_manifest.ece_val),
        "thresholds": dict(validated_manifest.thresholds),
        "temperatures": list(calibration_payload["temperatures"]),
        "onnx_bytes": int(int8_onnx_path.stat().st_size),
        "num_calibration_rows": int(calibration_payload["num_calibration_rows"]),
        "flip_eval_rows": len(flip_rows),
        "quantization": quant_info,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-export DiodeMCP-NanoJev-0.6B checkpoint to single-file <2 GiB INT8 bundle."
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint_best.pt")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to tokenizer.json")
    parser.add_argument("--val-calib", type=Path, required=True, help="Path to val_calib parquet/jsonl")
    parser.add_argument("--output-bundle", type=Path, required=True, help="Target bundle directory")
    parser.add_argument("--model-id", type=str, default="DiodeMCP-NanoJev-0.6B")
    parser.add_argument("--max-flip-rate", type=float, default=0.05)
    parser.add_argument("--calib-max-rows", type=int, default=512)
    parser.add_argument("--flip-max-rows", type=int, default=32)
    parser.add_argument("--calib-batch-size", type=int, default=16)
    parser.add_argument("--min-temp", type=float, default=0.75)
    parser.add_argument("--calib-buckets", type=str, default="256,512")
    parser.add_argument("--intra-op-threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--staging-dir", type=Path, default=None)
    parser.add_argument(
        "--reuse-fp32",
        action="store_true",
        help="Reuse existing model_fp32.onnx in --staging-dir without re-exporting.",
    )
    parser.add_argument(
        "--reduce-range",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 7-bit reduce_range quantization (prevents AVX2 VPMADDUBSW overflow).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    threads = max(1, int(args.intra_op_threads))
    torch.set_num_threads(threads)
    buckets = tuple(int(x.strip()) for x in args.calib_buckets.split(",") if x.strip())
    summary = export_nanojev_with_outlier_fallback(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        val_calib_path=args.val_calib,
        output_bundle=args.output_bundle,
        model_id=args.model_id,
        max_flip_rate=args.max_flip_rate,
        calib_max_rows=args.calib_max_rows,
        flip_max_rows=args.flip_max_rows,
        calib_batch_size=args.calib_batch_size,
        min_temp=args.min_temp,
        calib_buckets=buckets,
        intra_op_threads=threads,
        reduce_range=args.reduce_range,
        staging_dir=args.staging_dir,
        reuse_fp32=args.reuse_fp32,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
