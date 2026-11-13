# DiodeMCP Competitive MCP Security Benchmark Sandbox

This directory contains the **offline, deterministic, reproducible competitive benchmark sandbox** comparing single-model `DiodeMCP` (`NanoJev-0.6B` INT8, `Jev-Base-149M` INT8, and `Tier-0 Only` ablation) against leading locally installable open-source MCP security defenses:

- **`ProtectAI LLM-Guard` (`llm-guard==0.3.16`):** `PromptInjection` (`protectai/deberta-v3-base-prompt-injection-v2`, threshold `0.5`) on tool descriptions and untrusted tool outputs + `Secrets` and `Regex`/PII scanners on outgoing candidate tool arguments.
- **`Cisco AI Defense mcp-scanner` (`cisco-ai-mcp-scanner==4.8.7`):** Offline `YaraAnalyzer` + heuristic rules in both native MCP definition scan mode (`tools/list`) and extended payload YARA mode (`tools/list` + `prior_turns` + `candidate_call`).
- **`Stateless Regex/Policy Baseline`:** Gitleaks-grade secret regexes, destructive shell/SQL rules, dotfile path checks, and `_TOOL_POISONING_RE` without cross-turn taint tracking or `tools/list` pin hashing.

## Layout

- `benchmarks/traces/` — Frozen schemas (`schema.py`), deterministic 600-trace Suite A generator (`build_suite_a.py`, `seed=20261113`), and 30 live MCP `stdio` wire scenarios (`suite_b_scenarios.py`).
- `benchmarks/contenders/` — Persistent JSONL stdin/stdout worker processes (`diodemcp_worker.py`, `llm_guard_worker.py`, `cisco_scanner_worker.py`, `stateless_policy_worker.py`) and IPC schema (`protocol.py`).
- `benchmarks/wire/` — Real JSON-RPC 2.0 `stdio` mock MCP servers (`mock_mcp_servers.py`), competitor `stdio` proxy (`competitor_stdio_proxy.py`), and Suite B orchestrator (`run_suite_b.py`).
- `benchmarks/runner/` — Sequential Suite A runner (`run_suite_a.py`), pure-stdlib statistical metrics (`metrics.py`), and Markdown/CSV/JSON report renderer (`render_report.py`).
- `benchmarks/scripts/` — CPU INT8 ONNX exporter for `NanoJev-0.6B` (`export_nanojev_cpu.py`), remote `eclipse` virtualenv bootstrap (`bootstrap_eclipse.sh`), and offline sequential runner (`run_all.sh`).

## Running Unit Tests Locally

```bash
PYTHONPATH=. .venv/bin/python -m pytest benchmarks/tests/ -v
```

## Reproducing Full Benchmarks on `eclipse`

1. **Bootstrap isolated virtualenvs & pre-warm Hugging Face / YARA caches (online once):**

   ```bash
   bash /srv/homelab/diodemcp-bench/repo/benchmarks/scripts/bootstrap_eclipse.sh
   ```

   This creates four isolated Python 3.12 virtualenvs under `/srv/homelab/diodemcp-bench/.venvs/`:
   - `venv-diodemcp` (zero PyTorch; ONNX Runtime CPU only)
   - `venv-llm-guard` (CPU-only PyTorch + `llm-guard==0.3.16`)
   - `venv-cisco` (`cisco-ai-mcp-scanner==4.8.7`)
   - `venv-baseline` (stdlib baseline)

2. **Run Suite A (600 traces) + Suite B (30 live wire scenarios) strictly offline and sequentially:**

   ```bash
   bash /srv/homelab/diodemcp-bench/repo/benchmarks/scripts/run_all.sh
   ```

   `run_all.sh` pins `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `TOKENIZERS_PARALLELISM=false`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `NO_PROXY=*`, runs 5 untimed warmup evaluations per contender, records `perf_counter_ns()` latency and peak `VmHWM`/`VmRSS` memory, and writes final reports to `/srv/homelab/diodemcp-bench/results/`:
   - `COMPETITIVE_BENCHMARK_REPORT.md`
   - `suite_a_summary.csv`
   - `suite_a_summary.json`
   - `suite_b_summary.json`
