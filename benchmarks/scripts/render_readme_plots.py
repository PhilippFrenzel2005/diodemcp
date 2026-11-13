#!/usr/bin/env python3
"""Generate sanitizer-safe dark & light SVG benchmark plots from Suite A & Suite B results."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
ASSETS_DIR = REPO_ROOT / ".github" / "assets"

FONT_SANS = "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def compute_roc_points(jsonl_path: Path) -> list[tuple[float, float]]:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scored: list[tuple[float, int]] = []
    pos_total = 0
    neg_total = 0
    for r in rows:
        is_pos = 1 if str(r["trace_id"]).startswith("V") else 0
        pos_total += is_pos
        neg_total += 1 - is_pos
        scored.append((float(r["risk_score"]), is_pos))

    scored.sort(key=lambda x: x[0], reverse=True)
    pts: list[tuple[float, float]] = [(0.0, 0.0)]
    tp = 0
    fp = 0
    idx = 0
    n = len(scored)
    while idx < n:
        thresh = scored[idx][0]
        while idx < n and scored[idx][0] == thresh:
            if scored[idx][1] == 1:
                tp += 1
            else:
                fp += 1
            idx += 1
        pts.append((fp / neg_total, tp / pos_total))
    if pts[-1] != (1.0, 1.0):
        pts.append((1.0, 1.0))
    return pts


def render_roc_pareto(sa_by_id: dict[str, dict], dark: bool) -> str:
    bg1 = "#0b0f17" if dark else "#f8fafc"
    bg2 = "#0f172a" if dark else "#f1f5f9"
    card_bg = "#111827" if dark else "#ffffff"
    border = "#334155" if dark else "#cbd5e1"
    outer_border = "#1e293b" if dark else "#cbd5e1"
    grid = "#1e293b" if dark else "#e2e8f0"
    fg = "#f8fafc" if dark else "#0f172a"
    sub = "#94a3b8" if dark else "#475569"
    muted = "#64748b" if dark else "#64748b"

    c_nano = "#10b981" if dark else "#059669"
    c_base = "#22d3ee" if dark else "#0284c7"
    c_tier0 = "#f59e0b" if dark else "#d97706"
    c_llm = "#f43f5e" if dark else "#e11d48"
    c_regex = "#a78bfa" if dark else "#7c3aed"
    c_cisco = "#94a3b8" if dark else "#64748b"

    curves = [
        ("diodemcp-nanojev-0.6b", "NanoJev-0.6B (0.999)", c_nano, 3.0, ""),
        ("diodemcp-base-149m", "Base-149M (0.990)", c_base, 2.5, ""),
        ("diodemcp-tier0-only", "Tier-0 Rules (0.979)", c_tier0, 1.8, "5,3"),
        ("protectai-llm-guard", "ProtectAI LLM-Guard (0.742)", c_llm, 2.2, ""),
        ("stateless-regex-policy", "Stateless Regex (0.716)", c_regex, 1.6, "4,4"),
        ("cisco-mcp-scanner-defs", "Cisco mcp-scanner (0.562)", c_cisco, 2.0, ""),
    ]

    # Left plot box: x=86..496 (w=410), y=122..362 (h=240)
    lx0, ly0, lw, lh = 86.0, 122.0, 410.0, 240.0
    # Right plot box: x=612..1026 (w=414), y=122..362 (h=240) -> FPR 0..65%
    rx0, ry0, rw, rh = 612.0, 122.0, 414.0, 240.0

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="430" viewBox="0 0 1080 430" role="img" aria-label="ROC Curves and Security-Usability Pareto Frontier">',
        "  <defs>",
        f'    <linearGradient id="rpBg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="{bg1}"/><stop offset="100%" stop-color="{bg2}"/></linearGradient>',
        f'    <linearGradient id="nanoArea" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{c_nano}" stop-opacity="0.18"/><stop offset="100%" stop-color="{c_nano}" stop-opacity="0.01"/></linearGradient>',
        "  </defs>",
        f'  <rect x="1" y="1" width="1078" height="428" rx="16" fill="url(#rpBg)" stroke="{outer_border}" stroke-width="1.5"/>',
        f'  <text x="32" y="36" fill="{fg}" font-family="{FONT_SANS}" font-size="18" font-weight="800">ROC Separation &amp; Autonomous Security–Usability Pareto Frontier</text>',
        f'  <text x="32" y="56" fill="{sub}" font-family="{FONT_MONO}" font-size="11.5">Suite A · 600 Multi-Turn MCP Trajectories (360 Attacks V1–V6 vs. 240 Matched-Sink Benign Developer Twins B1–B6)</text>',
        # Left Card
        f'  <rect x="28" y="72" width="500" height="334" rx="12" fill="{card_bg}" stroke="{border}"/>',
        f'  <text x="48" y="98" fill="{fg}" font-family="{FONT_SANS}" font-size="14" font-weight="700">Empirical ROC Curves (Risk Score Sweep)</text>',
        f'  <text x="508" y="98" text-anchor="end" fill="{c_nano}" font-family="{FONT_MONO}" font-size="11" font-weight="700">AUROC 0.999 vs 0.742</text>',
        # Right Card
        f'  <rect x="552" y="72" width="500" height="334" rx="12" fill="{card_bg}" stroke="{border}"/>',
        f'  <text x="572" y="98" fill="{fg}" font-family="{FONT_SANS}" font-size="14" font-weight="700">Hard-Block Operating Points</text>',
        f'  <text x="1032" y="98" text-anchor="end" fill="{c_base}" font-family="{FONT_MONO}" font-size="11" font-weight="700">TOP-LEFT IS IDEAL</text>',
    ]

    # Left grid & axes
    for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        gx = lx0 + frac * lw
        gy = ly0 + (1.0 - frac) * lh
        lines.append(f'  <line x1="{gx:.1f}" y1="{ly0:.1f}" x2="{gx:.1f}" y2="{ly0+lh:.1f}" stroke="{grid}" stroke-width="1"/>')
        lines.append(f'  <line x1="{lx0:.1f}" y1="{gy:.1f}" x2="{lx0+lw:.1f}" y2="{gy:.1f}" stroke="{grid}" stroke-width="1"/>')
        lines.append(f'  <text x="{gx:.1f}" y="{ly0+lh+16:.1f}" text-anchor="middle" fill="{muted}" font-family="{FONT_MONO}" font-size="10">{int(frac*100)}%</text>')
        lines.append(f'  <text x="{lx0-8:.1f}" y="{gy+3.5:.1f}" text-anchor="end" fill="{muted}" font-family="{FONT_MONO}" font-size="10">{int(frac*100)}%</text>')

    # Diagonal chance line
    lines.append(f'  <line x1="{lx0:.1f}" y1="{ly0+lh:.1f}" x2="{lx0+lw:.1f}" y2="{ly0:.1f}" stroke="{muted}" stroke-width="1" stroke-dasharray="4,4"/>')
    lines.append(f'  <text x="{lx0+lw/2:.1f}" y="{ly0+lh+34:.1f}" text-anchor="middle" fill="{sub}" font-family="{FONT_SANS}" font-size="11" font-weight="600">Benign Developer False-Positive Rate (FPR)</text>')
    lines.append(f'  <text x="44" y="{ly0+lh/2:.1f}" text-anchor="middle" fill="{sub}" font-family="{FONT_SANS}" font-size="11" font-weight="600" transform="rotate(-90 44 {ly0+lh/2:.1f})">Attack Recall (TPR)</text>')

    # Plot ROC curves (reverse order so NanoJev & Base render on top)
    nano_pts = compute_roc_points(RESULTS_DIR / "suite_a_diodemcp-nanojev-0.6b.jsonl")
    nano_poly = " ".join(f"{lx0 + x*lw:.1f},{ly0 + (1.0-y)*lh:.1f}" for x, y in nano_pts)
    nano_area = f"{lx0:.1f},{ly0+lh:.1f} " + nano_poly + f" {lx0+lw:.1f},{ly0+lh:.1f}"
    lines.append(f'  <polygon points="{nano_area}" fill="url(#nanoArea)"/>')

    for cid, _label, color, stroke_w, dash in reversed(curves):
        pts = compute_roc_points(RESULTS_DIR / f"suite_a_{cid}.jsonl")
        poly = " ".join(f"{lx0 + x*lw:.1f},{ly0 + (1.0-y)*lh:.1f}" for x, y in pts)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'  <polyline points="{poly}" fill="none" stroke="{color}" stroke-width="{stroke_w}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
        )

    # Legend box inside bottom-right of ROC plot
    leg_bg = "#0b1120" if dark else "#f8fafc"
    lines.append(f'  <rect x="274" y="226" width="214" height="128" rx="8" fill="{leg_bg}" fill-opacity="0.94" stroke="{border}"/>')
    for idx, (_cid, label, color, stroke_w, dash) in enumerate(curves):
        ly = 244 + idx * 19
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'  <line x1="286" y1="{ly-4}" x2="308" y2="{ly-4}" stroke="{color}" stroke-width="{stroke_w}"{dash_attr}/>')
        weight = "700" if idx < 2 else "400"
        txt_col = fg if idx < 2 else sub
        lines.append(f'  <text x="316" y="{ly}" fill="{txt_col}" font-family="{FONT_MONO}" font-size="10" font-weight="{weight}">{_esc(label)}</text>')

    # Right subplot: Pareto Frontier (x: FPR 0..60%, y: Recall 0..100%)
    max_fpr = 0.60
    # Ideal zone highlight: FPR 0..3%, Recall 88..100%
    iz_w = (0.035 / max_fpr) * rw
    iz_h = 0.14 * rh
    iz_fill = "#064e3b" if dark else "#dcfce7"
    iz_stroke = "#10b981" if dark else "#059669"
    lines.append(f'  <rect x="{rx0:.1f}" y="{ry0:.1f}" width="{iz_w:.1f}" height="{iz_h:.1f}" fill="{iz_fill}" fill-opacity="0.45" stroke="{iz_stroke}" stroke-dasharray="3,2"/>')
    lines.append(f'  <text x="{rx0:.1f}" y="{ry0 - 7:.1f}" fill="{c_nano}" font-family="{FONT_MONO}" font-size="9.5" font-weight="700">Shaded: Ideal Zone (&gt;=90% Recall, &lt;=2% FPR)</text>')

    for fpr_tick in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
        gx = rx0 + (fpr_tick / max_fpr) * rw
        lines.append(f'  <line x1="{gx:.1f}" y1="{ry0:.1f}" x2="{gx:.1f}" y2="{ry0+rh:.1f}" stroke="{grid}" stroke-width="1"/>')
        lines.append(f'  <text x="{gx:.1f}" y="{ry0+rh+16:.1f}" text-anchor="middle" fill="{muted}" font-family="{FONT_MONO}" font-size="10">{int(round(fpr_tick*100))}%</text>')

    for rec_tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        gy = ry0 + (1.0 - rec_tick) * rh
        lines.append(f'  <line x1="{rx0:.1f}" y1="{gy:.1f}" x2="{rx0+rw:.1f}" y2="{gy:.1f}" stroke="{grid}" stroke-width="1"/>')
        lines.append(f'  <text x="{rx0-8:.1f}" y="{gy+3.5:.1f}" text-anchor="end" fill="{muted}" font-family="{FONT_MONO}" font-size="10">{int(rec_tick*100)}%</text>')

    lines.append(f'  <text x="{rx0+rw/2:.1f}" y="{ry0+rh+34:.1f}" text-anchor="middle" fill="{sub}" font-family="{FONT_SANS}" font-size="11" font-weight="600">Benign Developer False-Positive Rate (0% - 60%)</text>')
    lines.append(f'  <text x="568" y="{ry0+rh/2:.1f}" text-anchor="middle" fill="{sub}" font-family="{FONT_SANS}" font-size="11" font-weight="600" transform="rotate(-90 568 {ry0+rh/2:.1f})">Autonomous Recall</text>')

    points = [
        ("diodemcp-nanojev-0.6b", "NanoJev-0.6B (95.6%, 0.4%)", c_nano, 7.5, 646.0, 138.0),
        ("diodemcp-base-149m", "Base-149M (94.4%, 1.2%)", c_base, 7.0, 646.0, 162.0),
        ("diodemcp-tier0-only", "Tier-0 Rules (94.4%, 15.4%)", c_tier0, 5.5, 734.0, 190.0),
        ("protectai-llm-guard", "LLM-Guard (92.8%, 55.8%)", c_llm, 6.5, 838.0, 164.0),
        ("stateless-regex-policy", "Stateless Regex (48.6%, 5.4%)", c_regex, 5.5, 664.0, 248.0),
        ("cisco-mcp-scanner-extended", "Cisco Extended (39.2%, 10.8%)", c_cisco, 5.0, 702.0, 274.0),
        ("cisco-mcp-scanner-defs", "Cisco Native (12.5%, 0.0%)", c_cisco, 5.5, 628.0, 330.0),
    ]

    for cid, label, color, radius, tx, ty in points:
        ov = sa_by_id[cid]["hard"]["overall"]
        fpr = float(ov["fpr"])
        rec = float(ov["recall"])
        px = rx0 + (fpr / max_fpr) * rw
        py = ry0 + (1.0 - rec) * rh
        lines.append(f'  <line x1="{px:.1f}" y1="{py:.1f}" x2="{tx-4:.1f}" y2="{ty-3:.1f}" stroke="{color}" stroke-width="1" stroke-opacity="0.65"/>')
        if cid in ("diodemcp-nanojev-0.6b", "diodemcp-base-149m"):
            lines.append(f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="{radius+4:.1f}" fill="{color}" fill-opacity="0.22"/>')
        lines.append(f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{color}" stroke="{card_bg}" stroke-width="1.5"/>')
        weight = "700" if "NanoJev" in label or "Base-149M" in label or "LLM-Guard" in label else "500"
        tcol = fg if weight == "700" else sub
        lines.append(f'  <text x="{tx:.1f}" y="{ty:.1f}" fill="{tcol}" font-family="{FONT_MONO}" font-size="10" font-weight="{weight}">{_esc(label)}</text>')

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _recall_cell_style(val: float, dark: bool) -> tuple[str, str]:
    if val >= 0.90:
        return ("#064e3b", "#6ee7b7") if dark else ("#dcfce7", "#065f46")
    if val >= 0.70:
        return ("#164e63", "#67e8f9") if dark else ("#e0f2fe", "#075985")
    if val >= 0.35:
        return ("#78350f", "#fcd34d") if dark else ("#fef3c7", "#92400e")
    return ("#4c0519", "#fda4af") if dark else ("#ffe4e6", "#9f1239")


def _fpr_cell_style(val: float, dark: bool) -> tuple[str, str]:
    if val <= 0.025:
        return ("#064e3b", "#6ee7b7") if dark else ("#dcfce7", "#065f46")
    if val <= 0.08:
        return ("#1e3a5f", "#93c5fd") if dark else ("#e0f2fe", "#0369a1")
    if val <= 0.25:
        return ("#78350f", "#fcd34d") if dark else ("#fef3c7", "#92400e")
    return ("#4c0519", "#fda4af") if dark else ("#ffe4e6", "#9f1239")


def render_vector_heatmap(sa_by_id: dict[str, dict], dark: bool) -> str:
    bg1 = "#0b0f17" if dark else "#f8fafc"
    bg2 = "#0f172a" if dark else "#f1f5f9"
    card_bg = "#111827" if dark else "#ffffff"
    border = "#334155" if dark else "#cbd5e1"
    outer_border = "#1e293b" if dark else "#cbd5e1"
    fg = "#f8fafc" if dark else "#0f172a"
    sub = "#94a3b8" if dark else "#475569"
    muted = "#64748b" if dark else "#64748b"

    rows = [
        ("diodemcp-nanojev-0.6b", "DiodeMCP NanoJev-0.6B", True),
        ("diodemcp-base-149m", "DiodeMCP Base-149M", True),
        ("diodemcp-tier0-only", "DiodeMCP Tier-0 Rules", False),
        ("protectai-llm-guard", "ProtectAI LLM-Guard", False),
        ("stateless-regex-policy", "Stateless Regex Policy", False),
        ("cisco-mcp-scanner-defs", "Cisco mcp-scanner", False),
    ]
    v_cols = [
        ("V1", "Indirect"),
        ("V2", "Unicode"),
        ("V3", "tools/list"),
        ("V4", "Encoded"),
        ("V5", "SQL/Exec"),
        ("V6", "X-Server"),
    ]
    b_cols = [
        ("B1", "Webhook"),
        ("B2", "Issue/PR"),
        ("B3", "Catalog"),
        ("B4", "Base64"),
        ("B5", "DB Ops"),
        ("B6", "SecTests"),
    ]

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="404" viewBox="0 0 1080 404" role="img" aria-label="Per-Vector Attack Recall and Matched-Sink Benign FPR Heatmap">',
        "  <defs>",
        f'    <linearGradient id="hmBg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="{bg1}"/><stop offset="100%" stop-color="{bg2}"/></linearGradient>',
        "  </defs>",
        f'  <rect x="1" y="1" width="1078" height="402" rx="16" fill="url(#hmBg)" stroke="{outer_border}" stroke-width="1.5"/>',
        f'  <text x="32" y="36" fill="{fg}" font-family="{FONT_SANS}" font-size="18" font-weight="800">Per-Vector Attack Recall (V1–V6) &amp; Matched-Sink Benign Twin FPR (B1–B6)</text>',
        f'  <text x="32" y="56" fill="{sub}" font-family="{FONT_MONO}" font-size="11.5">Autonomous Hard-Block Mode · Left block: Attack Recall (Higher = Better) · Right block: Benign Dev FPR (Lower = Better)</text>',
        f'  <rect x="24" y="72" width="1032" height="308" rx="12" fill="{card_bg}" stroke="{border}"/>',
    ]

    # Column geometry: row labels x=40..226; V1..V6 starting at x=232, cell_w=64, gap=5 -> 6*69=414 -> ends 641; divider at 651; B1..B6 starting at x=662, cell_w=62, gap=5 -> ends 1034
    vx0 = 232
    bx0 = 664
    cw = 60
    gap = 6
    y_top = 136
    rh = 34
    rgap = 5

    c_green = "#34d399" if dark else "#047857"
    c_cyan = "#38bdf8" if dark else "#0369a1"
    lines.append(f'  <text x="{vx0 + 3*cw + 2.5*gap:.1f}" y="94" text-anchor="middle" fill="{c_green}" font-family="{FONT_MONO}" font-size="11" font-weight="700">ATTACK RECALL BY THREAT VECTOR (60 TRACES EACH)</text>')
    lines.append(f'  <text x="{bx0 + 3*cw + 2.5*gap:.1f}" y="94" text-anchor="middle" fill="{c_cyan}" font-family="{FONT_MONO}" font-size="11" font-weight="700">BENIGN DEV FPR BY MATCHED SINK (40 TWINS EACH)</text>')
    lines.append(f'  <line x1="650" y1="84" x2="650" y2="366" stroke="{border}" stroke-width="1.5" stroke-dasharray="4,3"/>')

    for i, (code, short) in enumerate(v_cols):
        cx = vx0 + i * (cw + gap) + cw / 2
        lines.append(f'  <text x="{cx:.1f}" y="113" text-anchor="middle" fill="{fg}" font-family="{FONT_MONO}" font-size="11" font-weight="700">{code}</text>')
        lines.append(f'  <text x="{cx:.1f}" y="126" text-anchor="middle" fill="{muted}" font-family="{FONT_SANS}" font-size="9.5">{_esc(short)}</text>')

    for i, (code, short) in enumerate(b_cols):
        cx = bx0 + i * (cw + gap) + cw / 2
        lines.append(f'  <text x="{cx:.1f}" y="113" text-anchor="middle" fill="{fg}" font-family="{FONT_MONO}" font-size="11" font-weight="700">{code}</text>')
        lines.append(f'  <text x="{cx:.1f}" y="126" text-anchor="middle" fill="{muted}" font-family="{FONT_SANS}" font-size="9.5">{_esc(short)}</text>')

    for r_idx, (cid, rlabel, bold) in enumerate(rows):
        ry = y_top + r_idx * (rh + rgap)
        pv = sa_by_id[cid]["hard"]["per_vector"]
        fw = "700" if bold else "500"
        lcol = fg if bold else sub
        lines.append(f'  <text x="40" y="{ry + 22}" fill="{lcol}" font-family="{FONT_SANS}" font-size="12.5" font-weight="{fw}">{_esc(rlabel)}</text>')

        for i, (code, _short) in enumerate(v_cols):
            val = float(pv[code]["recall"])
            cfill, ctxt = _recall_cell_style(val, dark)
            cx = vx0 + i * (cw + gap)
            lines.append(f'  <rect x="{cx}" y="{ry}" width="{cw}" height="{rh}" rx="6" fill="{cfill}"/>')
            lines.append(f'  <text x="{cx + cw/2:.1f}" y="{ry + 21}" text-anchor="middle" fill="{ctxt}" font-family="{FONT_MONO}" font-size="11.5" font-weight="700">{val*100:.0f}%</text>')

        for i, (code, _short) in enumerate(b_cols):
            val = float(pv[code]["fpr"])
            cfill, ctxt = _fpr_cell_style(val, dark)
            cx = bx0 + i * (cw + gap)
            lines.append(f'  <rect x="{cx}" y="{ry}" width="{cw}" height="{rh}" rx="6" fill="{cfill}"/>')
            lines.append(f'  <text x="{cx + cw/2:.1f}" y="{ry + 21}" text-anchor="middle" fill="{ctxt}" font-family="{FONT_MONO}" font-size="11.5" font-weight="700">{val*100:.1f}%</text>')

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_live_wire_suite_b(sb_by_id: dict[str, dict], dark: bool) -> str:
    bg1 = "#0b0f17" if dark else "#f8fafc"
    bg2 = "#0f172a" if dark else "#f1f5f9"
    card_bg = "#111827" if dark else "#ffffff"
    border = "#334155" if dark else "#cbd5e1"
    outer_border = "#1e293b" if dark else "#cbd5e1"
    track = "#1e293b" if dark else "#e2e8f0"
    fg = "#f8fafc" if dark else "#0f172a"
    sub = "#94a3b8" if dark else "#475569"
    muted = "#64748b" if dark else "#64748b"

    cat_colors = [
        ("tools_list_poisoning", "tools/list Poison", "#06b6d4" if dark else "#0891b2"),
        ("rugpull_pin_mutation", "Catalog Rug-Pull", "#10b981" if dark else "#059669"),
        ("cross_server_exfil", "Cross-Server Exfil", "#3b82f6" if dark else "#2563eb"),
        ("error_and_oversize_injection", "Error/Oversize RPC", "#8b5cf6" if dark else "#7c3aed"),
    ]

    contenders = [
        ("diodemcp-nanojev-0.6b", "DiodeMCP NanoJev-0.6B", True),
        ("diodemcp-base-149m", "DiodeMCP Base-149M", True),
        ("protectai-llm-guard", "ProtectAI LLM-Guard", False),
        ("diodemcp-tier0-only", "DiodeMCP Tier-0 Rules", False),
        ("cisco-mcp-scanner-defs", "Cisco mcp-scanner", False),
        ("stateless-regex-policy", "Stateless Regex Policy", False),
    ]

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="390" viewBox="0 0 1080 390" role="img" aria-label="Suite B Live MCP stdio Wire Recall and p50 Latency">',
        "  <defs>",
        f'    <linearGradient id="wbBg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="{bg1}"/><stop offset="100%" stop-color="{bg2}"/></linearGradient>',
        "  </defs>",
        f'  <rect x="1" y="1" width="1078" height="388" rx="16" fill="url(#wbBg)" stroke="{outer_border}" stroke-width="1.5"/>',
        f'  <text x="32" y="36" fill="{fg}" font-family="{FONT_SANS}" font-size="18" font-weight="800">Suite B Live MCP stdio Wire Benchmark (30 Real Child-Process Sessions)</text>',
        f'  <text x="32" y="56" fill="{sub}" font-family="{FONT_MONO}" font-size="11.5">20 Live Wire Attacks (5 per category) + 10 Benign Sessions · 0.0% Benign Wire FPR across DiodeMCP tiers</text>',
        # Left Card
        f'  <rect x="28" y="72" width="536" height="294" rx="12" fill="{card_bg}" stroke="{border}"/>',
        f'  <text x="48" y="98" fill="{fg}" font-family="{FONT_SANS}" font-size="14" font-weight="700">Live Wire Attacks Blocked by Attack Surface (/20)</text>',
        # Right Card
        f'  <rect x="584" y="72" width="468" height="294" rx="12" fill="{card_bg}" stroke="{border}"/>',
        f'  <text x="604" y="98" fill="{fg}" font-family="{FONT_SANS}" font-size="14" font-weight="700">Wire p50 Overhead (Log ms)</text>',
        f'  <text x="1032" y="98" text-anchor="end" fill="{ "#34d399" if dark else "#047857" }" font-family="{FONT_MONO}" font-size="11" font-weight="700">40.6x FASTER vs LLM-GUARD</text>',
    ]

    # Category legend row on left card
    lx_cursor = 48
    for _ckey, clabel, ccol in cat_colors:
        lines.append(f'  <rect x="{lx_cursor}" y="110" width="10" height="10" rx="2" fill="{ccol}"/>')
        lines.append(f'  <text x="{lx_cursor+14}" y="119" fill="{sub}" font-family="{FONT_MONO}" font-size="10">{_esc(clabel)}</text>')
        lx_cursor += 124

    bar_x0 = 216.0
    full_w = 240.0  # 20 attacks -> 12px per attack
    px_per_hit = full_w / 20.0

    # Log latency scale on right card: 0.05 ms to 500 ms (4 decades)
    rx_bar0 = 760.0
    rx_full = 200.0
    log_min = math.log10(0.05)
    log_max = math.log10(500.0)

    for idx, (cid, label, bold) in enumerate(contenders):
        ry = 140 + idx * 36
        rec = sb_by_id[cid]
        fw = "700" if bold else "500"
        tcol = fg if bold else sub
        lines.append(f'  <text x="48" y="{ry+15}" fill="{tcol}" font-family="{FONT_SANS}" font-size="12" font-weight="{fw}">{_esc(label)}</text>')
        lines.append(f'  <rect x="{bar_x0}" y="{ry}" width="{full_w}" height="20" rx="5" fill="{track}"/>')

        cur_x = bar_x0
        for ckey, _clabel, ccol in cat_colors:
            hits = int(rec["per_category"][ckey]["blocked"])
            if hits > 0:
                seg_w = hits * px_per_hit
                lines.append(f'  <rect x="{cur_x:.1f}" y="{ry}" width="{seg_w:.1f}" height="20" rx="3" fill="{ccol}"/>')
                cur_x += seg_w

        tot_blk = int(rec["attack_blocked"])
        pct = float(rec["attack_recall"]) * 100.0
        scol = ("#34d399" if dark else "#047857") if tot_blk >= 19 else (fg if bold else sub)
        lines.append(f'  <text x="{bar_x0 + full_w + 10:.1f}" y="{ry+15}" fill="{scol}" font-family="{FONT_MONO}" font-size="11.5" font-weight="{fw}">{pct:.0f}% ({tot_blk}/20)</text>')

        # Right card: Latency bar
        ov_ms = max(float(rec["overhead_p50_ms"]), 0.05)
        frac = (math.log10(ov_ms) - log_min) / (log_max - log_min)
        lw_bar = max(4.0, frac * rx_full)
        if cid in ("diodemcp-nanojev-0.6b", "diodemcp-base-149m"):
            lbar_col = "#10b981" if cid == "diodemcp-nanojev-0.6b" else "#22d3ee"
            if not dark:
                lbar_col = "#059669" if cid == "diodemcp-nanojev-0.6b" else "#0284c7"
        elif ov_ms > 100.0:
            lbar_col = "#f43f5e" if dark else "#e11d48"
        else:
            lbar_col = "#64748b"

        short_lbl = label.replace("DiodeMCP ", "").replace("ProtectAI ", "").replace(" Policy", "")
        lines.append(f'  <text x="604" y="{ry+15}" fill="{tcol}" font-family="{FONT_SANS}" font-size="12" font-weight="{fw}">{_esc(short_lbl)}</text>')
        lines.append(f'  <rect x="{rx_bar0}" y="{ry}" width="{rx_full}" height="20" rx="5" fill="{track}"/>')
        lines.append(f'  <rect x="{rx_bar0}" y="{ry}" width="{lw_bar:.1f}" height="20" rx="5" fill="{lbar_col}"/>')
        ms_txt = f"{float(rec['overhead_p50_ms']):.2f} ms"
        lines.append(f'  <text x="{rx_bar0 + rx_full + 10:.1f}" y="{ry+15}" fill="{tcol}" font-family="{FONT_MONO}" font-size="11.5" font-weight="{fw}">{ms_txt}</text>')

    lines.append(f'  <text x="604" y="354" fill="{muted}" font-family="{FONT_MONO}" font-size="10">95% wire recall at 7.8–11.8 ms p50 (vs. 317.2 ms LLM-Guard)</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    sa_list = json.loads((RESULTS_DIR / "suite_a_summary.json").read_text(encoding="utf-8"))["suite_a"]
    sb_list = json.loads((RESULTS_DIR / "suite_b_summary.json").read_text(encoding="utf-8"))["suite_b"]
    sa_by_id = {item["contender_id"]: item for item in sa_list}
    sb_by_id = {item["contender_id"]: item for item in sb_list}

    outputs = {
        "plot-roc-pareto-dark.svg": render_roc_pareto(sa_by_id, dark=True),
        "plot-roc-pareto-light.svg": render_roc_pareto(sa_by_id, dark=False),
        "plot-vector-heatmap-dark.svg": render_vector_heatmap(sa_by_id, dark=True),
        "plot-vector-heatmap-light.svg": render_vector_heatmap(sa_by_id, dark=False),
        "plot-live-wire-dark.svg": render_live_wire_suite_b(sb_by_id, dark=True),
        "plot-live-wire-light.svg": render_live_wire_suite_b(sb_by_id, dark=False),
    }
    for fname, content in outputs.items():
        (ASSETS_DIR / fname).write_text(content, encoding="utf-8")
        print(f"Wrote {ASSETS_DIR / fname} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
