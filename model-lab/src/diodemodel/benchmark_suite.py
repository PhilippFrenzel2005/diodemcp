"""DiodeModelBundle v1 ORT-INT8 benchmark suite (`DiodeBench` val + `Trifecta-Dev-Bench` + CPU latency).

Evaluates exported INT8 ONNX bundles across:
* Per-threat (`T1..T7`) attack recall & overall attack recall
* Symmetric `FlipJev` (attack recall + benign twin DoS FPR + robust accuracy)
* `Trifecta-Safe` benign twin FPR
* Source & argument pointer top-1 accuracy on active leaks
* Post-INT8 Expected Calibration Error (ECE) & Brier score
* CPU single-request inference latency (`p50_ms`, `p95_ms`) at `512` and `2048` token buckets
* Multi-model Markdown comparison table generation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Final

import numpy as np
import torch
from tokenizers import Tokenizer

from diodemodel.contract import validate_bundle_dir
from diodemodel.dataset_packer import PackedJevRow
from diodemodel.export_bundle import (
    ONNX_INPUT_NAMES,
    ONNX_OUTPUT_NAMES,
    build_smoke_jev_tokenizer,
    create_ort_cpu_session,
    export_diode_bundle,
    make_dummy_export_batch,
    read_onnx_logit_temperatures,
)
from diodemodel.losses_rlcd import probs_to_logits, temperature_calibration_report
from diodemodel.modeling_jev import ARG_SLOTS, SRC_SLOTS
from diodemodel.mutators import ALL_THREAT_CATEGORIES
from diodemodel.train import (
    CURRICULUM_BUCKETS,
    _build_tiny_smoke_modernbert,
    _build_tiny_smoke_nanojev,
    _select_batch_bucket,
    collate_jev_batch,
    load_packed_jev_dataset,
    make_synthetic_curriculum_dataset,
)

DEFAULT_LATENCY_BUCKETS: Final[tuple[int, ...]] = (512, 2048)


def evaluate_ort_bundle_dataset(
    bundle_dir: Path | str,
    eval_rows: Sequence[Mapping[str, Any] | PackedJevRow] | Path | str,
    *,
    bucket_len: int | None = None,
    batch_size: int = 16,
    threshold: float = 0.5,
    intra_op_threads: int = 4,
) -> dict[str, Any]:
    """Evaluate a validated ``DiodeModelBundle v1`` via ONNX Runtime CPU on ``eval_rows``."""
    b_dir = Path(bundle_dir)
    manifest = validate_bundle_dir(b_dir, verify_sha256=True)
    pad_token_id = int(manifest.special_token_ids["[PAD]"])

    rows = load_packed_jev_dataset(eval_rows)
    if not rows:
        raise ValueError("evaluate_ort_bundle_dataset requires at least 1 evaluation row")

    onnx_path = b_dir / "model_int8.onnx"
    sess = create_ort_cpu_session(onnx_path, intra_op_threads=intra_op_threads)
    temps_np = read_onnx_logit_temperatures(onnx_path)
    temps_t = torch.from_numpy(temps_np.copy())

    all_noul_probs: list[torch.Tensor] = []
    all_src_probs: list[torch.Tensor] = []
    all_arg_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_src_targets: list[torch.Tensor] = []
    all_arg_targets: list[torch.Tensor] = []
    all_categories: list[str] = []
    all_variants: list[str] = []
    all_trifecta: list[bool] = []
    all_flipjev: list[bool] = []

    eff_bs = max(1, int(batch_size))
    candidate_buckets = tuple(manifest.length_buckets) if manifest.length_buckets else CURRICULUM_BUCKETS

    for start in range(0, len(rows), eff_bs):
        batch_slice = rows[start : start + eff_bs]
        eff_bucket = (
            int(bucket_len)
            if bucket_len is not None
            else _select_batch_bucket(batch_slice, candidate_buckets)
        )
        batch = collate_jev_batch(
            batch_slice,
            bucket_len=eff_bucket,
            pad_token_id=pad_token_id,
            device="cpu",
        )
        feed = {name: batch[name].numpy() for name in ONNX_INPUT_NAMES}
        noul_p, src_p, arg_p = sess.run(list(ONNX_OUTPUT_NAMES), feed)

        all_noul_probs.append(torch.from_numpy(np.asarray(noul_p, dtype=np.float32)))
        all_src_probs.append(torch.from_numpy(np.asarray(src_p, dtype=np.float32)))
        all_arg_probs.append(torch.from_numpy(np.asarray(arg_p, dtype=np.float32)))
        all_targets.append(batch["noul_targets"].detach().cpu())
        all_src_targets.append(batch["src_target"].detach().cpu())
        all_arg_targets.append(batch["arg_target"].detach().cpu())
        all_categories.extend(batch["threat_category"])
        all_variants.extend(batch["variant_kind"])
        all_trifecta.extend(bool(x) for x in batch["is_trifecta_benign"].detach().cpu().tolist())
        all_flipjev.extend(bool(x) for x in batch["flipjev_applied"].detach().cpu().tolist())

    noul_probs = torch.cat(all_noul_probs, dim=0)
    src_probs = torch.cat(all_src_probs, dim=0)
    arg_probs = torch.cat(all_arg_probs, dim=0)
    noul_targets = torch.cat(all_targets, dim=0)
    src_targets = torch.cat(all_src_targets, dim=0)
    arg_targets = torch.cat(all_arg_targets, dim=0)

    # Recover pre-temperature INT8 logits from calibrated probabilities
    noul_logits = probs_to_logits(noul_probs, temps_t)

    p_max = noul_probs.amax(dim=-1).numpy()
    y_max = noul_targets.amax(dim=-1).numpy()
    y_np = noul_targets.numpy()
    src_pred = src_probs.argmax(dim=-1).numpy()
    arg_pred = arg_probs.argmax(dim=-1).numpy()
    src_t_np = src_targets.numpy()
    arg_t_np = arg_targets.numpy()

    is_attack = y_max >= 0.5
    is_benign = ~is_attack
    detected = p_max >= float(threshold)

    # 1. Per-threat (T1..T7) recall
    per_threat_recall: dict[str, float] = {}
    per_threat_counts: dict[str, int] = {}
    for cat in ALL_THREAT_CATEGORIES:
        mask = np.array([c.upper() == cat for c in all_categories], dtype=bool) & is_attack
        cnt = int(mask.sum())
        per_threat_counts[cat] = cnt
        per_threat_recall[cat] = float(detected[mask].mean()) if cnt > 0 else 0.0

    attack_recall = float(detected[is_attack].mean()) if bool(is_attack.any()) else 0.0

    # 2. Symmetric FlipJev (attack recall + benign twin DoS FPR)
    flip_mask = np.array(all_flipjev, dtype=bool)
    flip_attack_mask = flip_mask & is_attack
    flip_benign_mask = flip_mask & is_benign
    flipjev_attack_recall = (
        float(detected[flip_attack_mask].mean()) if bool(flip_attack_mask.any()) else 0.0
    )
    flipjev_benign_dos_fpr = (
        float(detected[flip_benign_mask].mean()) if bool(flip_benign_mask.any()) else 0.0
    )
    if bool(flip_attack_mask.any()) and bool(flip_benign_mask.any()):
        flipjev_robust_accuracy = 0.5 * (flipjev_attack_recall + (1.0 - flipjev_benign_dos_fpr))
    elif bool(flip_attack_mask.any()):
        flipjev_robust_accuracy = flipjev_attack_recall
    elif bool(flip_benign_mask.any()):
        flipjev_robust_accuracy = 1.0 - flipjev_benign_dos_fpr
    else:
        flipjev_robust_accuracy = 0.0

    # 3. Trifecta-Safe / Benign Twin FPR
    trifecta_mask = np.array(all_trifecta, dtype=bool) & is_benign
    if not bool(trifecta_mask.any()):
        trifecta_mask = is_benign
    trifecta_fpr = float(detected[trifecta_mask].mean()) if bool(trifecta_mask.any()) else 0.0

    # 4. Pointer top-1 accuracy on active leaks
    active_leak_src = (
        ((y_np[:, 0] >= 0.5) | (y_np[:, 1] >= 0.5))
        & (src_t_np >= 0)
        & (src_t_np < SRC_SLOTS - 1)
    )
    active_leak_arg = (
        ((y_np[:, 0] >= 0.5) | (y_np[:, 1] >= 0.5))
        & (arg_t_np >= 0)
        & (arg_t_np < ARG_SLOTS - 1)
    )
    src_pointer_top1_acc = (
        float((src_pred[active_leak_src] == src_t_np[active_leak_src]).mean())
        if bool(active_leak_src.any())
        else 0.0
    )
    arg_pointer_top1_acc = (
        float((arg_pred[active_leak_arg] == arg_t_np[active_leak_arg]).mean())
        if bool(active_leak_arg.any())
        else 0.0
    )
    pointer_top1_acc = 0.5 * (src_pointer_top1_acc + arg_pointer_top1_acc)

    # 5. Per-head & overall ECE + Brier
    calib_rep = temperature_calibration_report(noul_logits, noul_targets, temps_t)
    brier_per_head_t = ((noul_probs - noul_targets) ** 2).mean(dim=0)
    brier_per_head = [float(x) for x in brier_per_head_t.tolist()]
    brier_overall = float(brier_per_head_t.mean().item())

    return {
        "num_rows": len(rows),
        "per_threat_recall": per_threat_recall,
        "per_threat_counts": per_threat_counts,
        "attack_recall": attack_recall,
        "flipjev_attack_recall": flipjev_attack_recall,
        "flipjev_benign_dos_fpr": flipjev_benign_dos_fpr,
        "flipjev_dos_fpr": flipjev_benign_dos_fpr,
        "flipjev_robust_accuracy": float(flipjev_robust_accuracy),
        "trifecta_fpr": trifecta_fpr,
        "trifecta_safe_fpr": trifecta_fpr,
        "benign_twin_fpr": trifecta_fpr,
        "src_pointer_top1_acc": src_pointer_top1_acc,
        "arg_pointer_top1_acc": arg_pointer_top1_acc,
        "pointer_top1_acc": pointer_top1_acc,
        "ece_per_head": list(calib_rep["ece_after"]),
        "ece_val": float(calib_rep["ece_val"]),
        "brier_per_head": brier_per_head,
        "brier": brier_overall,
        "temperatures": [float(x) for x in temps_t.tolist()],
        "calibration_report": calib_rep,
    }


def measure_ort_cpu_latency(
    bundle_dir: Path | str,
    *,
    latency_buckets: Sequence[int] = DEFAULT_LATENCY_BUCKETS,
    num_runs: int = 8,
    warmup_runs: int = 2,
    intra_op_threads: int = 4,
) -> dict[str, dict[str, float]]:
    """Measure single-request (`batch_size=1`) ORT CPU latency (`p50_ms`, `p95_ms`) across buckets."""
    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}")
    b_dir = Path(bundle_dir)
    manifest = validate_bundle_dir(b_dir, verify_sha256=True)
    tok = Tokenizer.from_file(str(b_dir / "tokenizer.json"))
    vocab_size = max(16, int(tok.get_vocab_size()))
    pad_id = int(manifest.special_token_ids["[PAD]"])

    sess = create_ort_cpu_session(b_dir / "model_int8.onnx", intra_op_threads=intra_op_threads)
    results: dict[str, dict[str, float]] = {}

    for bucket in latency_buckets:
        b_len = int(bucket)
        dummy = make_dummy_export_batch(
            batch_size=1,
            seq_len=b_len,
            vocab_size=vocab_size,
            pad_token_id=pad_id,
            seed=1000 + b_len,
            device="cpu",
        )
        feed = {name: dummy[name].numpy() for name in ONNX_INPUT_NAMES}

        for _ in range(max(0, int(warmup_runs))):
            sess.run(list(ONNX_OUTPUT_NAMES), feed)

        timings_ms: list[float] = []
        for _ in range(int(num_runs)):
            t0 = time.perf_counter()
            sess.run(list(ONNX_OUTPUT_NAMES), feed)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.asarray(timings_ms, dtype=np.float64)
        results[str(b_len)] = {
            "bucket_length": float(b_len),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(np.mean(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "num_runs": float(num_runs),
        }
    return results


def run_bundle_benchmark(
    bundle_dir: Path | str,
    *,
    val_rows: Sequence[Mapping[str, Any] | PackedJevRow] | Path | str,
    trifecta_rows: Sequence[Mapping[str, Any] | PackedJevRow] | Path | str | None = None,
    latency_buckets: Sequence[int] = DEFAULT_LATENCY_BUCKETS,
    num_latency_runs: int = 8,
    warmup_runs: int = 2,
    batch_size: int = 16,
    threshold: float = 0.5,
    intra_op_threads: int = 4,
    output_json_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run full benchmark suite on a ``DiodeModelBundle v1`` (`val_calib`, `trifecta_test`, CPU latency)."""
    b_dir = Path(bundle_dir)
    manifest = validate_bundle_dir(b_dir, verify_sha256=True)

    val_metrics = evaluate_ort_bundle_dataset(
        b_dir,
        val_rows,
        batch_size=batch_size,
        threshold=threshold,
        intra_op_threads=intra_op_threads,
    )
    trifecta_metrics = (
        evaluate_ort_bundle_dataset(
            b_dir,
            trifecta_rows,
            batch_size=batch_size,
            threshold=threshold,
            intra_op_threads=intra_op_threads,
        )
        if trifecta_rows is not None
        else val_metrics
    )
    cpu_latency = measure_ort_cpu_latency(
        b_dir,
        latency_buckets=latency_buckets,
        num_runs=num_latency_runs,
        warmup_runs=warmup_runs,
        intra_op_threads=intra_op_threads,
    )

    lat_512 = cpu_latency.get("512", next(iter(cpu_latency.values())))
    lat_2048 = cpu_latency.get("2048", list(cpu_latency.values())[-1])

    report: dict[str, Any] = {
        "model_id": manifest.model_id,
        "bundle_dir": str(b_dir),
        "onnx_size_bytes": (b_dir / "model_int8.onnx").stat().st_size,
        "val_calib": val_metrics,
        "trifecta_test": trifecta_metrics,
        "cpu_latency": cpu_latency,
        "per_threat_recall": dict(trifecta_metrics["per_threat_recall"]),
        "attack_recall": float(trifecta_metrics["attack_recall"]),
        "flipjev_attack_recall": float(trifecta_metrics["flipjev_attack_recall"]),
        "flipjev_benign_dos_fpr": float(trifecta_metrics["flipjev_benign_dos_fpr"]),
        "trifecta_fpr": float(trifecta_metrics["trifecta_fpr"]),
        "trifecta_safe_fpr": float(trifecta_metrics["trifecta_safe_fpr"]),
        "src_pointer_top1_acc": float(trifecta_metrics["src_pointer_top1_acc"]),
        "arg_pointer_top1_acc": float(trifecta_metrics["arg_pointer_top1_acc"]),
        "pointer_top1_acc": float(trifecta_metrics["pointer_top1_acc"]),
        "ece_val": float(val_metrics["ece_val"]),
        "brier": float(val_metrics["brier"]),
        "temperatures": list(val_metrics["temperatures"]),
        "latency_512_p50_ms": float(lat_512["p50_ms"]),
        "latency_512_p95_ms": float(lat_512["p95_ms"]),
        "latency_2048_p50_ms": float(lat_2048["p50_ms"]),
        "latency_2048_p95_ms": float(lat_2048["p95_ms"]),
    }

    if output_json_path is not None:
        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return report


def format_benchmark_comparison_table(reports: Sequence[Mapping[str, Any]]) -> str:
    """Format a Markdown comparison table across multiple exported `DiodeModelBundle v1` reports."""
    headers = (
        "| Model ID | Attack Recall | T1..T7 Mean | FlipJev Recall | FlipJev DoS FPR | "
        "Trifecta-Safe FPR | Pointer Top-1 | Post-INT8 ECE | Brier | "
        "CPU 512ms (p50 / p95) | CPU 2048ms (p50 / p95) |"
    )
    sep = (
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [headers, sep]
    for rep in reports:
        t_recalls = rep.get("per_threat_recall", {})
        t_mean = (
            float(np.mean([float(t_recalls.get(c, 0.0)) for c in ALL_THREAT_CATEGORIES]))
            if t_recalls
            else 0.0
        )
        lines.append(
            f"| `{rep['model_id']}` "
            f"| {float(rep['attack_recall']):.3f} "
            f"| {t_mean:.3f} "
            f"| {float(rep['flipjev_attack_recall']):.3f} "
            f"| {float(rep['flipjev_benign_dos_fpr']):.3f} "
            f"| {float(rep['trifecta_safe_fpr']):.3f} "
            f"| {float(rep['pointer_top1_acc']):.3f} "
            f"| {float(rep['ece_val']):.4f} "
            f"| {float(rep['brier']):.4f} "
            f"| {float(rep['latency_512_p50_ms']):.2f} / {float(rep['latency_512_p95_ms']):.2f} "
            f"| {float(rep['latency_2048_p50_ms']):.2f} / {float(rep['latency_2048_p95_ms']):.2f} |"
        )
    return "\n".join(lines) + "\n"


def run_multi_bundle_benchmark(
    bundle_dirs: Sequence[Path | str],
    *,
    val_rows: Sequence[Mapping[str, Any] | PackedJevRow] | Path | str,
    trifecta_rows: Sequence[Mapping[str, Any] | PackedJevRow] | Path | str | None = None,
    latency_buckets: Sequence[int] = DEFAULT_LATENCY_BUCKETS,
    num_latency_runs: int = 8,
    warmup_runs: int = 2,
    batch_size: int = 16,
    threshold: float = 0.5,
    intra_op_threads: int = 4,
    output_json_path: Path | str | None = None,
    output_markdown_path: Path | str | None = None,
) -> dict[str, Any]:
    """Benchmark multiple ``DiodeModelBundle v1`` directories and generate a comparison table."""
    reports = [
        run_bundle_benchmark(
            b_dir,
            val_rows=val_rows,
            trifecta_rows=trifecta_rows,
            latency_buckets=latency_buckets,
            num_latency_runs=num_latency_runs,
            warmup_runs=warmup_runs,
            batch_size=batch_size,
            threshold=threshold,
            intra_op_threads=intra_op_threads,
        )
        for b_dir in bundle_dirs
    ]
    markdown_table = format_benchmark_comparison_table(reports)
    combined = {
        "reports": reports,
        "comparison_table_markdown": markdown_table,
    }
    if output_json_path is not None:
        out_j = Path(output_json_path)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
    if output_markdown_path is not None:
        out_m = Path(output_markdown_path)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        out_m.write_text(markdown_table, encoding="utf-8")
    return combined


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m diodemodel.benchmark_suite",
        description="Benchmark exported DiodeModelBundle v1 directories (val_calib, Trifecta, CPU latency).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast offline CPU smoke benchmark across tiny ModernBERTJevModel and NanoJevCausalModel.",
    )
    parser.add_argument(
        "--bundle-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="One or more DiodeModelBundle v1 directories to benchmark.",
    )
    parser.add_argument(
        "--val-parquet",
        type=Path,
        default=None,
        help="Packed val_calib Parquet/JSONL path.",
    )
    parser.add_argument(
        "--trifecta-parquet",
        type=Path,
        default=None,
        help="Optional packed Trifecta-Dev-Bench Parquet/JSONL path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for JSON benchmark report.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional output path for Markdown comparison table.",
    )
    parser.add_argument(
        "--num-latency-runs",
        type=int,
        default=4,
        help="Number of timed ORT CPU latency runs per bucket.",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=4,
        help="ORT CPU intra-op threads.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if args.smoke:
        tok, special_ids = build_smoke_jev_tokenizer()
        pad_id = int(special_ids["[PAD]"])
        val_rows = make_synthetic_curriculum_dataset(
            12, vocab_size=64, seed=42, pad_token_id=pad_id
        )
        trifecta_rows = make_synthetic_curriculum_dataset(
            12, vocab_size=64, seed=99, pad_token_id=pad_id
        )
        with tempfile.TemporaryDirectory(prefix="diodemodel_bench_smoke_") as tmp_d:
            root = Path(tmp_d)
            specs = (
                ("DiodeMCP-Jev-Base-149M", _build_tiny_smoke_modernbert(64, pad_id)),
                ("DiodeMCP-NanoJev-0.6B", _build_tiny_smoke_nanojev(64, pad_id)),
            )
            b_dirs: list[Path] = []
            for m_id, m_obj in specs:
                b_dir = root / m_id
                export_diode_bundle(
                    m_obj,
                    b_dir,
                    tokenizer_source=tok,
                    model_id=m_id,
                    val_calib_rows=val_rows,
                    special_token_ids=special_ids,
                    calib_buckets=(128, 256),
                    intra_op_threads=args.intra_op_threads,
                )
                b_dirs.append(b_dir)

            out = run_multi_bundle_benchmark(
                b_dirs,
                val_rows=val_rows,
                trifecta_rows=trifecta_rows,
                latency_buckets=(512, 2048),
                num_latency_runs=2,
                warmup_runs=1,
                intra_op_threads=args.intra_op_threads,
                output_json_path=args.output_json,
                output_markdown_path=args.output_markdown,
            )
            print(out["comparison_table_markdown"])
        return 0

    if not args.bundle_dirs or args.val_parquet is None:
        raise ValueError("--bundle-dirs and --val-parquet are required when not running --smoke")

    out = run_multi_bundle_benchmark(
        args.bundle_dirs,
        val_rows=args.val_parquet,
        trifecta_rows=args.trifecta_parquet,
        num_latency_runs=args.num_latency_runs,
        intra_op_threads=args.intra_op_threads,
        output_json_path=args.output_json,
        output_markdown_path=args.output_markdown,
    )
    print(out["comparison_table_markdown"])
    return 0


__all__ = [
    "DEFAULT_LATENCY_BUCKETS",
    "evaluate_ort_bundle_dataset",
    "format_benchmark_comparison_table",
    "measure_ort_cpu_latency",
    "run_bundle_benchmark",
    "run_multi_bundle_benchmark",
]


if __name__ == "__main__":
    raise SystemExit(main())
