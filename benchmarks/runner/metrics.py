"""Pure-stdlib statistical metrics engine for Suite A competitive benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

from benchmarks.contenders.protocol import TraceEvalResult
from benchmarks.traces.schema import MultiTurnTrace

_WILSON_Z_95 = 1.959963984540054


def sanitize_score(score: float) -> float:
    """Sanitize risk score to [0.0, 1.0], mapping non-finite values fail-closed to 1.0."""
    try:
        val = float(score)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(val):
        return 1.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def wilson_ci_95(successes: int, total: int) -> tuple[float, float]:
    """Return 95% Wilson score confidence interval `(low, high)` in `[0.0, 1.0]`."""
    if total <= 0:
        return (0.0, 0.0)
    k = max(0, min(successes, total))
    n = float(total)
    p = k / n
    z2 = _WILSON_Z_95 * _WILSON_Z_95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (_WILSON_Z_95 * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def compute_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute exact ROC-AUC via Mann-Whitney U with average-rank tie handling."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length.")
    n = len(labels)
    if n == 0:
        return None

    n_pos = sum(1 for y in labels if int(y) == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    paired = sorted(
        ((sanitize_score(s), int(y)) for y, s in zip(labels, scores)),
        key=lambda item: item[0],
    )
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and paired[j][0] == paired[i][0]:
            j += 1
        # 1-based average rank for indices i .. j-1
        avg_rank = (i + 1 + j) / 2.0
        pos_in_tie = sum(1 for k in range(i, j) if paired[k][1] == 1)
        rank_sum_pos += pos_in_tie * avg_rank
        i = j

    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return u_stat / (n_pos * n_neg)


def compute_pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute step-wise Average Precision (PR-AUC) in pure stdlib Python."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length.")
    n = len(labels)
    if n == 0:
        return None

    n_pos = sum(1 for y in labels if int(y) == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    paired = sorted(
        ((sanitize_score(s), int(y)) for y, s in zip(labels, scores)),
        key=lambda item: item[0],
        reverse=True,
    )
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and paired[j][0] == paired[i][0]:
            j += 1
        for k in range(i, j):
            if paired[k][1] == 1:
                tp += 1
            else:
                fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return ap


def percentile(values: Sequence[float], q: float) -> float:
    """Compute linear-interpolated percentile `q` in `[0, 100]`."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (max(0.0, min(100.0, q)) / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def compute_binary_metrics(
    labels: Sequence[int],
    predictions: Sequence[bool],
    scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute confusion counts, recall/FPR with Wilson 95% CIs, precision, F1, AUROC, and PR-AUC."""
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length.")
    if scores is not None and len(scores) != len(labels):
        raise ValueError("scores must match labels length.")

    tp = fp = tn = fn = 0
    for y_raw, pred_raw in zip(labels, predictions):
        y = 1 if int(y_raw) == 1 else 0
        pred = bool(pred_raw)
        if y == 1 and pred:
            tp += 1
        elif y == 0 and pred:
            fp += 1
        elif y == 0 and not pred:
            tn += 1
        else:
            fn += 1

    positives = tp + fn
    negatives = fp + tn
    total = positives + negatives

    recall = (tp / positives) if positives > 0 else 0.0
    fpr = (fp / negatives) if negatives > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = ((tp + tn) / total) if total > 0 else 0.0

    auroc = compute_roc_auc(labels, scores) if scores is not None else None
    pr_auc = compute_pr_auc(labels, scores) if scores is not None else None

    return {
        "total": total,
        "positives": positives,
        "negatives": negatives,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "recall_ci_95": list(wilson_ci_95(tp, positives)),
        "fpr": fpr,
        "fpr_ci_95": list(wilson_ci_95(fp, negatives)),
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "auroc": auroc,
        "pr_auc": pr_auc,
    }


def _build_operating_point_summary(
    traces: Sequence[MultiTurnTrace],
    ordered_results: Sequence[TraceEvalResult],
    *,
    hard_mode: bool,
) -> dict[str, Any]:
    labels = [t.label for t in traces]
    preds = [r.blocked_hard if hard_mode else r.blocked_protective for r in ordered_results]
    scores = [sanitize_score(r.risk_score) for r in ordered_results]

    overall = compute_binary_metrics(labels, preds, scores)
    per_vector: dict[str, Any] = {}
    paired_vectors: dict[str, Any] = {}

    for v in range(1, 7):
        v_key = f"V{v}"
        b_key = f"B{v}"
        v_idx = [i for i, t in enumerate(traces) if t.split_group == v_key]
        b_idx = [i for i, t in enumerate(traces) if t.split_group == b_key]
        pair_idx = [i for i, t in enumerate(traces) if t.vector_id == v]

        per_vector[v_key] = compute_binary_metrics(
            [labels[i] for i in v_idx],
            [preds[i] for i in v_idx],
            [scores[i] for i in v_idx],
        )
        per_vector[b_key] = compute_binary_metrics(
            [labels[i] for i in b_idx],
            [preds[i] for i in b_idx],
            [scores[i] for i in b_idx],
        )
        paired_vectors[f"vector_{v}"] = compute_binary_metrics(
            [labels[i] for i in pair_idx],
            [preds[i] for i in pair_idx],
            [scores[i] for i in pair_idx],
        )

    return {
        "overall": overall,
        "per_vector": per_vector,
        "paired_vectors": paired_vectors,
    }


def summarize_contender_suite_a(
    contender_id: str,
    traces: Sequence[MultiTurnTrace],
    results: Sequence[TraceEvalResult],
    *,
    peak_rss_mb: float,
    cold_start_ms: float,
) -> dict[str, Any]:
    """Summarize Suite A results across both protective and hard operating points."""
    by_id = {r.trace_id: r for r in results}
    if len(by_id) != len(results):
        raise ValueError("Duplicate trace_id found in results.")

    ordered_results: list[TraceEvalResult] = []
    for trace in traces:
        res = by_id.get(trace.trace_id)
        if res is None:
            raise ValueError(f"Missing result for trace_id={trace.trace_id!r}")
        ordered_results.append(res)

    latencies = [max(0.0, float(r.latency_ms)) for r in ordered_results]
    error_count = sum(1 for r in ordered_results if r.blocked_stage == "error")

    return {
        "contender_id": contender_id,
        "total_traces": len(traces),
        "error_count": error_count,
        "cold_start_ms": float(cold_start_ms),
        "peak_rss_mb": float(peak_rss_mb),
        "latency_p50_ms": percentile(latencies, 50.0),
        "latency_p95_ms": percentile(latencies, 95.0),
        "latency_mean_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "latency_max_ms": max(latencies) if latencies else 0.0,
        "protective": _build_operating_point_summary(traces, ordered_results, hard_mode=False),
        "hard": _build_operating_point_summary(traces, ordered_results, hard_mode=True),
    }
