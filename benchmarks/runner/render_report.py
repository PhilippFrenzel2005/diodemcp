"""Render Markdown, CSV, and JSON reports for Suite A and Suite B benchmark runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
from typing import Any


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100.0:.1f}%"


def _fmt_pct_ci(value: float, ci: Sequence[float]) -> str:
    lo = ci[0] * 100.0 if len(ci) == 2 else 0.0
    hi = ci[1] * 100.0 if len(ci) == 2 else 0.0
    return f"{value * 100.0:.1f}% ({lo:.1f}–{hi:.1f}%)"


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _write_suite_a_csv(
    suite_a_summaries: Sequence[Mapping[str, Any]],
    output_csv: Path,
) -> None:
    fieldnames = [
        "contender_id",
        "operating_point",
        "total_traces",
        "recall",
        "recall_ci_low",
        "recall_ci_high",
        "fpr",
        "fpr_ci_low",
        "fpr_ci_high",
        "precision",
        "f1",
        "auroc",
        "pr_auc",
        *(f"V{v}_recall" for v in range(1, 7)),
        *(f"B{v}_fpr" for v in range(1, 7)),
        "latency_p50_ms",
        "latency_p95_ms",
        "peak_rss_mb",
        "cold_start_ms",
        "error_count",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for summary in suite_a_summaries:
            for op_point in ("protective", "hard"):
                op_data = summary[op_point]
                overall = op_data["overall"]
                per_vec = op_data["per_vector"]
                row: dict[str, Any] = {
                    "contender_id": summary["contender_id"],
                    "operating_point": op_point,
                    "total_traces": summary["total_traces"],
                    "recall": f"{overall['recall']:.6f}",
                    "recall_ci_low": f"{overall['recall_ci_95'][0]:.6f}",
                    "recall_ci_high": f"{overall['recall_ci_95'][1]:.6f}",
                    "fpr": f"{overall['fpr']:.6f}",
                    "fpr_ci_low": f"{overall['fpr_ci_95'][0]:.6f}",
                    "fpr_ci_high": f"{overall['fpr_ci_95'][1]:.6f}",
                    "precision": f"{overall['precision']:.6f}",
                    "f1": f"{overall['f1']:.6f}",
                    "auroc": f"{overall['auroc']:.6f}" if overall["auroc"] is not None else "",
                    "pr_auc": f"{overall['pr_auc']:.6f}" if overall["pr_auc"] is not None else "",
                    "latency_p50_ms": f"{summary['latency_p50_ms']:.3f}",
                    "latency_p95_ms": f"{summary['latency_p95_ms']:.3f}",
                    "peak_rss_mb": f"{summary['peak_rss_mb']:.2f}",
                    "cold_start_ms": f"{summary['cold_start_ms']:.2f}",
                    "error_count": summary["error_count"],
                }
                for v in range(1, 7):
                    row[f"V{v}_recall"] = f"{per_vec[f'V{v}']['recall']:.6f}"
                    row[f"B{v}_fpr"] = f"{per_vec[f'B{v}']['fpr']:.6f}"
                writer.writerow(row)


def _render_overview_table(
    lines: list[str],
    title: str,
    suite_a_summaries: Sequence[Mapping[str, Any]],
    op_key: str,
) -> None:
    lines.append(f"### {title}")
    lines.append("")
    lines.append(
        "| Contender | Attack Recall (95% CI) | Benign FPR (95% CI) | F1 | AUROC | PR-AUC | p50 (ms) | p95 (ms) | Peak RSS (MB) | Cold Start (ms) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in suite_a_summaries:
        ov = s[op_key]["overall"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{s['contender_id']}`",
                    _fmt_pct_ci(ov["recall"], ov["recall_ci_95"]),
                    _fmt_pct_ci(ov["fpr"], ov["fpr_ci_95"]),
                    _fmt_float(ov["f1"]),
                    _fmt_float(ov["auroc"]),
                    _fmt_float(ov["pr_auc"]),
                    _fmt_float(s["latency_p50_ms"], 2),
                    _fmt_float(s["latency_p95_ms"], 2),
                    _fmt_float(s["peak_rss_mb"], 1),
                    _fmt_float(s["cold_start_ms"], 1),
                ]
            )
            + " |"
        )
    lines.append("")


def _render_per_vector_table(
    lines: list[str],
    title: str,
    suite_a_summaries: Sequence[Mapping[str, Any]],
    op_key: str,
) -> None:
    lines.append(f"#### {title}")
    lines.append("")
    lines.append(
        "| Contender | V1 Recall | V2 Recall | V3 Recall | V4 Recall | V5 Recall | V6 Recall | B1 FPR | B2 FPR | B3 FPR | B4 FPR | B5 FPR | B6 FPR |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in suite_a_summaries:
        pv = s[op_key]["per_vector"]
        v_cols = [_fmt_pct(pv[f"V{v}"]["recall"]) for v in range(1, 7)]
        b_cols = [_fmt_pct(pv[f"B{v}"]["fpr"]) for v in range(1, 7)]
        lines.append(f"| `{s['contender_id']}` | " + " | ".join(v_cols + b_cols) + " |")
    lines.append("")


def render_competitive_report(
    suite_a_summaries: Sequence[Mapping[str, Any]],
    suite_b_summaries: Sequence[Mapping[str, Any]],
    hardware_meta: Mapping[str, Any],
    output_md: Path,
    output_csv: Path,
    output_json: Path,
) -> None:
    """Write COMPETITIVE_BENCHMARK_REPORT.md, suite_a_summary.csv, and JSON summary."""
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "hardware_meta": dict(hardware_meta),
        "suite_a": [dict(s) for s in suite_a_summaries],
        "suite_b": [dict(s) for s in suite_b_summaries],
    }
    output_json.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_suite_a_csv(suite_a_summaries, output_csv)

    lines: list[str] = [
        "# Competitive MCP Security Benchmark Report",
        "",
        "## Execution Environment",
        "",
    ]
    if hardware_meta:
        for key, value in hardware_meta.items():
            lines.append(f"- **{key}:** `{value}`")
    else:
        lines.append("- **Environment:** Offline CPU benchmark")
    lines.append("")

    lines.append("## Suite A: 600-Trace Multi-Turn Replay Benchmark")
    lines.append("")
    _render_overview_table(
        lines,
        "Protective Gate (WARN + CRITICAL)",
        suite_a_summaries,
        "protective",
    )
    _render_per_vector_table(
        lines,
        "Per-Vector Attack Recall & Matched-Sink Benign FPR — Protective Gate (WARN + CRITICAL)",
        suite_a_summaries,
        "protective",
    )
    _render_overview_table(
        lines,
        "Autonomous Hard-Block (CRITICAL Only)",
        suite_a_summaries,
        "hard",
    )
    _render_per_vector_table(
        lines,
        "Per-Vector Attack Recall & Matched-Sink Benign FPR — Autonomous Hard-Block (CRITICAL Only)",
        suite_a_summaries,
        "hard",
    )

    lines.append("## Suite B: Live MCP `stdio` Wire-Protocol Sandbox (30 Scenarios)")
    lines.append("")
    lines.append(
        "| Contender | Attack Blocked | Attack Recall | Benign FP | Benign FPR | p50_ms | p95_ms | Overhead p50 (ms) | Overhead p95 (ms) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for sb in suite_b_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{sb.get('contender_id', 'unknown')}`",
                    f"{sb.get('attack_blocked', 0)}/{sb.get('attack_total', 0)}",
                    _fmt_pct(float(sb.get("attack_recall", 0.0))),
                    f"{sb.get('benign_fp', 0)}/{sb.get('benign_total', 0)}",
                    _fmt_pct(float(sb.get("benign_fpr", 0.0))),
                    _fmt_float(sb.get("p50_ms"), 2),
                    _fmt_float(sb.get("p95_ms"), 2),
                    _fmt_float(sb.get("overhead_p50_ms"), 2),
                    _fmt_float(sb.get("overhead_p95_ms"), 2),
                ]
            )
            + " |"
        )
    lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")
