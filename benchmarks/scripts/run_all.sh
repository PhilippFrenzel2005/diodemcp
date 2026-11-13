#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT="${BENCH_ROOT:-/srv/homelab/diodemcp-bench}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENVS_ROOT="${VENVS_ROOT:-${BENCH_ROOT}/.venvs}"
BUNDLES_ROOT="${BUNDLES_ROOT:-${BENCH_ROOT}/bundles}"
DATA_DIR="${DATA_DIR:-${BENCH_ROOT}/data}"
RESULTS_DIR="${RESULTS_DIR:-${BENCH_ROOT}/results}"
THREADS="${THREADS:-4}"
WARMUP="${WARMUP:-5}"

# Strictly offline execution + consistent 4-thread CPU pinning
export HF_HOME="${HF_HOME:-${BENCH_ROOT}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_PROXY="*"
export no_proxy="*"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PY_DIODEMCP="${VENVS_ROOT}/venv-diodemcp/bin/python"
if [[ ! -x "${PY_DIODEMCP}" ]]; then
  echo "ERROR: venv-diodemcp python not found at ${PY_DIODEMCP}" >&2
  exit 1
fi

mkdir -p "${DATA_DIR}" "${RESULTS_DIR}"

echo "[1/4] Generating deterministic 600-trace Suite A dataset (seed=20261113)..."
SUITE_A_JSONL="${DATA_DIR}/suite_a_600.jsonl"
"${PY_DIODEMCP}" -c "
from pathlib import Path
from benchmarks.traces.build_suite_a import SUITE_A_SEED, write_suite_a_jsonl
out = write_suite_a_jsonl(Path('${SUITE_A_JSONL}'), seed=SUITE_A_SEED)
print(f'Wrote Suite A traces to {out}')
"

echo "[2/4] Running Suite A sequentially across all contenders..."
"${PY_DIODEMCP}" -m benchmarks.runner.run_suite_a \
  --traces "${SUITE_A_JSONL}" \
  --output-dir "${RESULTS_DIR}" \
  --venvs-root "${VENVS_ROOT}" \
  --bundles-root "${BUNDLES_ROOT}" \
  --repo-root "${REPO_ROOT}" \
  --warmup "${WARMUP}" \
  --threads "${THREADS}"

echo "[3/4] Running Suite B (30 live MCP stdio scenarios) sequentially across all contenders..."
"${PY_DIODEMCP}" -m benchmarks.wire.run_suite_b \
  --output-dir "${RESULTS_DIR}" \
  --venvs-root "${VENVS_ROOT}" \
  --bundles-root "${BUNDLES_ROOT}" \
  --repo-root "${REPO_ROOT}" \
  --warmup "${WARMUP}" \
  --threads "${THREADS}"

echo "[4/4] Rendering Markdown, CSV, and JSON competitive benchmark reports..."
"${PY_DIODEMCP}" -c "
import json
import os
import platform
import subprocess
from pathlib import Path
from benchmarks.runner.render_report import render_competitive_report

results_dir = Path('${RESULTS_DIR}')
suite_a = json.loads((results_dir / 'suite_a_summary.json').read_text(encoding='utf-8'))['suite_a']
suite_b = json.loads((results_dir / 'suite_b_summary.json').read_text(encoding='utf-8'))['suite_b']

cpu_model = platform.processor() or 'unknown'
try:
    for line in Path('/proc/cpuinfo').read_text(encoding='utf-8').splitlines():
        if line.startswith('model name'):
            cpu_model = line.split(':', 1)[1].strip()
            break
except OSError:
    pass

mem_gib = 'unknown'
try:
    for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
        if line.startswith('MemTotal:'):
            kb = int(line.split()[1])
            mem_gib = f'{kb / (1024 * 1024):.1f} GiB'
            break
except Exception:
    pass

hardware_meta = {
    'Host': platform.node(),
    'CPU': cpu_model,
    'CPU Threads Pinned': '${THREADS}',
    'RAM': mem_gib,
    'OS / Kernel': f'{platform.system()} {platform.release()} ({platform.machine()})',
    'Python': platform.python_version(),
    'Offline Mode': 'HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, NO_PROXY=*',
}

render_competitive_report(
    suite_a,
    suite_b,
    hardware_meta,
    results_dir / 'COMPETITIVE_BENCHMARK_REPORT.md',
    results_dir / 'suite_a_summary.csv',
    results_dir / 'competitive_benchmark_summary.json',
)
print('Rendered report to', results_dir / 'COMPETITIVE_BENCHMARK_REPORT.md')
"

echo "Done! Results written to ${RESULTS_DIR}"
