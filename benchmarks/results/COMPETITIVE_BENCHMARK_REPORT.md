# Competitive MCP Security Benchmark Report

## Execution Environment

- **Host:** `eclipse.hmsrv`
- **CPU:** `Intel(R) Core(TM) i3-10100 CPU @ 3.60GHz`
- **CPU Threads Pinned:** `4`
- **RAM:** `31.1 GiB`
- **OS / Kernel:** `Linux 6.8.0-149-generic (x86_64)`
- **Python:** `3.12.3`
- **Offline Mode:** `HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, NO_PROXY=*`

## Suite A: 600-Trace Multi-Turn Replay Benchmark

### Protective Gate (WARN + CRITICAL)

| Contender | Attack Recall (95% CI) | Benign FPR (95% CI) | F1 | AUROC | PR-AUC | p50 (ms) | p95 (ms) | Peak RSS (MB) | Cold Start (ms) |
|---|---|---|---|---|---|---|---|---|---|
| `diodemcp-nanojev-0.6b` | 100.0% (98.9–100.0%) | 35.4% (29.6–41.7%) | 0.894 | 0.979 | 0.984 | 514.11 | 531.59 | 2949.3 | 6198.3 |
| `diodemcp-base-149m` | 100.0% (98.9–100.0%) | 46.2% (40.1–52.6%) | 0.866 | 0.971 | 0.972 | 153.02 | 166.36 | 1564.1 | 1666.3 |
| `diodemcp-tier0-only` | 100.0% (98.9–100.0%) | 26.7% (21.5–32.6%) | 0.918 | 0.971 | 0.972 | 29.42 | 47.04 | 76.4 | 0.0 |
| `protectai-llm-guard` | 92.8% (89.6–95.0%) | 55.8% (49.5–62.0%) | 0.807 | 0.742 | 0.752 | 1019.22 | 1186.04 | 1864.9 | 4518.0 |
| `cisco-mcp-scanner-defs` | 12.5% (9.5–16.3%) | 0.0% (0.0–1.6%) | 0.222 | 0.562 | 0.650 | 0.72 | 0.83 | 237.4 | 2513.8 |
| `cisco-mcp-scanner-extended` | 39.2% (34.3–44.3%) | 10.8% (7.5–15.4%) | 0.535 | 0.613 | 0.696 | 0.85 | 0.98 | 237.5 | 2498.7 |
| `stateless-regex-policy` | 48.6% (43.5–53.8%) | 5.4% (3.2–9.0%) | 0.639 | 0.716 | 0.761 | 0.34 | 0.39 | 31.9 | 0.0 |

#### Per-Vector Attack Recall & Matched-Sink Benign FPR — Protective Gate (WARN + CRITICAL)

| Contender | V1 Recall | V2 Recall | V3 Recall | V4 Recall | V5 Recall | V6 Recall | B1 FPR | B2 FPR | B3 FPR | B4 FPR | B5 FPR | B6 FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `diodemcp-nanojev-0.6b` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 55.0% | 32.5% | 0.0% | 35.0% | 65.0% | 25.0% |
| `diodemcp-base-149m` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 47.5% | 75.0% | 37.5% | 42.5% | 47.5% | 27.5% |
| `diodemcp-tier0-only` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 35.0% | 32.5% | 0.0% | 35.0% | 32.5% | 25.0% |
| `protectai-llm-guard` | 100.0% | 98.3% | 100.0% | 100.0% | 58.3% | 100.0% | 47.5% | 67.5% | 50.0% | 35.0% | 35.0% | 100.0% |
| `cisco-mcp-scanner-defs` | 0.0% | 0.0% | 75.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `cisco-mcp-scanner-extended` | 76.7% | 16.7% | 75.0% | 0.0% | 16.7% | 50.0% | 32.5% | 0.0% | 0.0% | 0.0% | 32.5% | 0.0% |
| `stateless-regex-policy` | 50.0% | 25.0% | 75.0% | 0.0% | 66.7% | 75.0% | 0.0% | 0.0% | 0.0% | 0.0% | 32.5% | 0.0% |

### Autonomous Hard-Block (CRITICAL Only)

| Contender | Attack Recall (95% CI) | Benign FPR (95% CI) | F1 | AUROC | PR-AUC | p50 (ms) | p95 (ms) | Peak RSS (MB) | Cold Start (ms) |
|---|---|---|---|---|---|---|---|---|---|
| `diodemcp-nanojev-0.6b` | 100.0% (98.9–100.0%) | 28.7% (23.4–34.8%) | 0.913 | 0.979 | 0.984 | 514.11 | 531.59 | 2949.3 | 6198.3 |
| `diodemcp-base-149m` | 94.4% (91.6–96.4%) | 20.8% (16.2–26.4%) | 0.907 | 0.971 | 0.972 | 153.02 | 166.36 | 1564.1 | 1666.3 |
| `diodemcp-tier0-only` | 94.4% (91.6–96.4%) | 20.8% (16.2–26.4%) | 0.907 | 0.971 | 0.972 | 29.42 | 47.04 | 76.4 | 0.0 |
| `protectai-llm-guard` | 92.8% (89.6–95.0%) | 55.8% (49.5–62.0%) | 0.807 | 0.742 | 0.752 | 1019.22 | 1186.04 | 1864.9 | 4518.0 |
| `cisco-mcp-scanner-defs` | 12.5% (9.5–16.3%) | 0.0% (0.0–1.6%) | 0.222 | 0.562 | 0.650 | 0.72 | 0.83 | 237.4 | 2513.8 |
| `cisco-mcp-scanner-extended` | 39.2% (34.3–44.3%) | 10.8% (7.5–15.4%) | 0.535 | 0.613 | 0.696 | 0.85 | 0.98 | 237.5 | 2498.7 |
| `stateless-regex-policy` | 48.6% (43.5–53.8%) | 5.4% (3.2–9.0%) | 0.639 | 0.716 | 0.761 | 0.34 | 0.39 | 31.9 | 0.0 |

#### Per-Vector Attack Recall & Matched-Sink Benign FPR — Autonomous Hard-Block (CRITICAL Only)

| Contender | V1 Recall | V2 Recall | V3 Recall | V4 Recall | V5 Recall | V6 Recall | B1 FPR | B2 FPR | B3 FPR | B4 FPR | B5 FPR | B6 FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `diodemcp-nanojev-0.6b` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 35.0% | 32.5% | 0.0% | 27.5% | 52.5% | 25.0% |
| `diodemcp-base-149m` | 100.0% | 100.0% | 100.0% | 100.0% | 66.7% | 100.0% | 35.0% | 32.5% | 0.0% | 0.0% | 32.5% | 25.0% |
| `diodemcp-tier0-only` | 100.0% | 100.0% | 100.0% | 100.0% | 66.7% | 100.0% | 35.0% | 32.5% | 0.0% | 0.0% | 32.5% | 25.0% |
| `protectai-llm-guard` | 100.0% | 98.3% | 100.0% | 100.0% | 58.3% | 100.0% | 47.5% | 67.5% | 50.0% | 35.0% | 35.0% | 100.0% |
| `cisco-mcp-scanner-defs` | 0.0% | 0.0% | 75.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `cisco-mcp-scanner-extended` | 76.7% | 16.7% | 75.0% | 0.0% | 16.7% | 50.0% | 32.5% | 0.0% | 0.0% | 0.0% | 32.5% | 0.0% |
| `stateless-regex-policy` | 50.0% | 25.0% | 75.0% | 0.0% | 66.7% | 75.0% | 0.0% | 0.0% | 0.0% | 0.0% | 32.5% | 0.0% |

## Suite B: Live MCP `stdio` Wire-Protocol Sandbox (30 Scenarios)

| Contender | Attack Blocked | Attack Recall | Benign FP | Benign FPR | p50_ms | p95_ms | Overhead p50 (ms) | Overhead p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| `raw-stdio-baseline` | 0/20 | 0.0% | 0/10 | 0.0% | 0.14 | 0.22 | 0.00 | 0.00 |
| `diodemcp-nanojev-0.6b` | 13/20 | 65.0% | 0/10 | 0.0% | 8.03 | 248.55 | 7.89 | 248.33 |
| `diodemcp-base-149m` | 15/20 | 75.0% | 1/10 | 10.0% | 7.75 | 68.73 | 7.61 | 68.52 |
| `diodemcp-tier0-only` | 8/20 | 40.0% | 0/10 | 0.0% | 7.14 | 18.67 | 7.00 | 18.46 |
| `protectai-llm-guard` | 20/20 | 100.0% | 3/10 | 30.0% | 317.61 | 344.26 | 317.47 | 344.04 |
| `cisco-mcp-scanner-defs` | 8/20 | 40.0% | 0/10 | 0.0% | 64.55 | 66.54 | 0.00 | 0.00 |
| `stateless-regex-policy` | 13/20 | 65.0% | 0/10 | 0.0% | 0.51 | 0.81 | 0.37 | 0.59 |
