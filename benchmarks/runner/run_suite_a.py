"""Sequential Suite A benchmark orchestrator over persistent contender workers."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

from benchmarks.contenders.protocol import (
    TraceEvalResult,
    decode_ipc_message,
    encode_ipc_message,
)
from benchmarks.runner.metrics import summarize_contender_suite_a
from benchmarks.traces.schema import MultiTurnTrace

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SuiteAContenderSpec:
    """Configuration for spawning one isolated Suite A contender worker."""

    contender_id: str
    python_bin: Path
    worker_module: str
    worker_args: tuple[str, ...] = ()
    env_overrides: dict[str, str] = field(default_factory=dict)


def build_default_suite_a_contenders(
    venvs_root: Path,
    bundles_root: Path,
    *,
    threads: int = 4,
) -> list[SuiteAContenderSpec]:
    """Return the canonical 7-contender Suite A lineup on eclipse."""
    py_diodemcp = venvs_root / "venv-diodemcp" / "bin" / "python"
    py_llm_guard = venvs_root / "venv-llm-guard" / "bin" / "python"
    py_cisco = venvs_root / "venv-cisco" / "bin" / "python"
    py_baseline = venvs_root / "venv-baseline" / "bin" / "python"

    return [
        SuiteAContenderSpec(
            contender_id="diodemcp-nanojev-0.6b",
            python_bin=py_diodemcp,
            worker_module="benchmarks.contenders.diodemcp_worker",
            worker_args=(
                "--mode",
                "ort",
                "--bundle-dir",
                str(bundles_root / "DiodeMCP-NanoJev-0.6B"),
                "--threads",
                str(threads),
            ),
        ),
        SuiteAContenderSpec(
            contender_id="diodemcp-base-149m",
            python_bin=py_diodemcp,
            worker_module="benchmarks.contenders.diodemcp_worker",
            worker_args=(
                "--mode",
                "ort",
                "--bundle-dir",
                str(bundles_root / "DiodeMCP-Jev-Base-149M"),
                "--threads",
                str(threads),
            ),
        ),
        SuiteAContenderSpec(
            contender_id="diodemcp-tier0-only",
            python_bin=py_diodemcp,
            worker_module="benchmarks.contenders.diodemcp_worker",
            worker_args=("--mode", "tier0", "--threads", str(threads)),
        ),
        SuiteAContenderSpec(
            contender_id="protectai-llm-guard",
            python_bin=py_llm_guard,
            worker_module="benchmarks.contenders.llm_guard_worker",
            worker_args=("--threads", str(threads)),
        ),
        SuiteAContenderSpec(
            contender_id="cisco-mcp-scanner-defs",
            python_bin=py_cisco,
            worker_module="benchmarks.contenders.cisco_scanner_worker",
            worker_args=("--scan-mode", "definitions_only"),
        ),
        SuiteAContenderSpec(
            contender_id="cisco-mcp-scanner-extended",
            python_bin=py_cisco,
            worker_module="benchmarks.contenders.cisco_scanner_worker",
            worker_args=("--scan-mode", "extended_payload"),
        ),
        SuiteAContenderSpec(
            contender_id="stateless-regex-policy",
            python_bin=py_baseline,
            worker_module="benchmarks.contenders.stateless_policy_worker",
        ),
    ]


def load_traces_jsonl(traces_path: Path) -> list[MultiTurnTrace]:
    """Load Suite A traces from a JSONL file."""
    traces: list[MultiTurnTrace] = []
    with traces_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                traces.append(MultiTurnTrace.from_dict(json.loads(stripped)))
            except Exception as exc:
                raise ValueError(f"Invalid trace at {traces_path}:{line_no}: {exc}") from exc
    return traces


def _read_pid_rss_mb(pid: int) -> float:
    proc_status = Path(f"/proc/{pid}/status")
    if not proc_status.is_file():
        return 0.0
    peak_kb = 0
    try:
        for line in proc_status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(("VmHWM:", "VmRSS:")):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    peak_kb = max(peak_kb, int(parts[1]))
    except OSError:
        return 0.0
    return peak_kb / 1024.0


def _build_worker_env(
    repo_root: Path,
    *,
    threads: int,
    overrides: dict[str, str],
) -> dict[str, str]:
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    repo_str = str(repo_root.resolve())
    env["PYTHONPATH"] = f"{repo_str}{os.pathsep}{existing_pp}" if existing_pp else repo_str
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["NUMEXPR_NUM_THREADS"] = str(threads)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.update(overrides)
    return env


def run_single_contender_suite_a(
    spec: SuiteAContenderSpec,
    traces: Sequence[MultiTurnTrace],
    output_dir: Path,
    *,
    repo_root: Path = DEFAULT_REPO_ROOT,
    warmup_count: int = 5,
    threads: int = 4,
    rpc_timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run one Suite A contender sequentially in an isolated worker subprocess."""
    if not traces:
        raise ValueError("Cannot run Suite A with zero traces.")

    cmd = [
        str(spec.python_bin),
        "-m",
        spec.worker_module,
        "--contender-id",
        spec.contender_id,
        *spec.worker_args,
    ]
    env = _build_worker_env(repo_root, threads=threads, overrides=spec.env_overrides)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
        cwd=str(repo_root),
        env=env,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    stderr_tail: deque[str] = deque(maxlen=40)

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        try:
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))
        except Exception:
            pass

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    def _rpc(payload: dict[str, Any]) -> dict[str, Any]:
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(encode_ipc_message(payload) + "\n")
        proc.stdin.flush()

        # Simple blocking readline (workers respond synchronously per request).
        deadline = time.monotonic() + rpc_timeout_s
        line = proc.stdout.readline()
        if not line:
            tail_txt = "\n".join(stderr_tail)
            raise RuntimeError(
                f"Worker {spec.contender_id} exited unexpectedly (rc={proc.poll()}).\n{tail_txt}"
            )
        if time.monotonic() > deadline:
            raise TimeoutError(f"Worker {spec.contender_id} timed out on IPC request")
        return decode_ipc_message(line)

    peak_rss_mb = 0.0
    cold_start_ms = 0.0
    results: list[TraceEvalResult] = []
    trace_dicts = [t.to_dict() for t in traces]

    try:
        ping_resp = _rpc({"cmd": "ping"})
        if not ping_resp.get("ok"):
            raise RuntimeError(f"Worker {spec.contender_id} failed ping: {ping_resp}")
        cold_start_ms = float(ping_resp.get("cold_start_ms", 0.0))
        peak_rss_mb = max(
            peak_rss_mb,
            float(ping_resp.get("rss_mb", 0.0)),
            _read_pid_rss_mb(proc.pid),
        )

        # Untimed warmup traces
        for idx in range(max(0, warmup_count)):
            warm_trace = trace_dicts[idx % len(trace_dicts)]
            warm_resp = _rpc({"cmd": "eval_trace", "trace": warm_trace})
            peak_rss_mb = max(
                peak_rss_mb,
                float(warm_resp.get("rss_mb", 0.0)),
                _read_pid_rss_mb(proc.pid),
            )

        # Timed evaluation traces
        for idx, trace_dict in enumerate(trace_dicts):
            resp = _rpc({"cmd": "eval_trace", "trace": trace_dict})
            peak_rss_mb = max(
                peak_rss_mb,
                float(resp.get("rss_mb", 0.0)),
                _read_pid_rss_mb(proc.pid),
            )
            raw_res = resp.get("result")
            if not isinstance(raw_res, dict):
                raise RuntimeError(
                    f"Worker {spec.contender_id} returned invalid result at index {idx}: {resp}"
                )
            results.append(TraceEvalResult.from_dict(raw_res))

        try:
            _rpc({"cmd": "shutdown"})
        except Exception:
            pass
    finally:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        stderr_thread.join(timeout=2.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_trace_path = output_dir / f"suite_a_{spec.contender_id}.jsonl"
    with per_trace_path.open("w", encoding="utf-8") as fh:
        for res in results:
            fh.write(encode_ipc_message(res.to_dict()) + "\n")

    return summarize_contender_suite_a(
        spec.contender_id,
        traces,
        results,
        peak_rss_mb=peak_rss_mb,
        cold_start_ms=cold_start_ms,
    )


def run_suite_a(
    contenders: Sequence[SuiteAContenderSpec],
    traces: Sequence[MultiTurnTrace],
    output_dir: Path,
    *,
    repo_root: Path = DEFAULT_REPO_ROOT,
    warmup_count: int = 5,
    threads: int = 4,
) -> list[dict[str, Any]]:
    """Run all Suite A contenders strictly sequentially and persist `suite_a_summary.json`."""
    summaries: list[dict[str, Any]] = []
    for spec in contenders:
        sys.stderr.write(f"[suite_a] Running contender: {spec.contender_id} ({len(traces)} traces)\n")
        sys.stderr.flush()
        summary = run_single_contender_suite_a(
            spec,
            traces,
            output_dir,
            repo_root=repo_root,
            warmup_count=warmup_count,
            threads=threads,
        )
        summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "suite_a_summary.json"
    summary_path.write_text(
        json.dumps({"suite_a": summaries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sequential Suite A benchmark runner")
    parser.add_argument("--traces", type=Path, required=True, help="Path to suite_a_600.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON/JSONL results")
    parser.add_argument(
        "--venvs-root",
        type=Path,
        default=Path("/srv/homelab/diodemcp-bench/.venvs"),
        help="Root directory containing venv-* environments",
    )
    parser.add_argument(
        "--bundles-root",
        type=Path,
        default=Path("/srv/homelab/diodemcp-bench/bundles"),
        help="Root directory containing DiodeModelBundle v1 directories",
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--contenders",
        type=str,
        default="",
        help="Optional comma-separated contender IDs to run",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    traces = load_traces_jsonl(args.traces)
    all_specs = build_default_suite_a_contenders(
        args.venvs_root, args.bundles_root, threads=args.threads
    )
    if args.contenders.strip():
        wanted = {c.strip() for c in args.contenders.split(",") if c.strip()}
        specs = [s for s in all_specs if s.contender_id in wanted]
        if not specs:
            raise SystemExit(f"No matching contenders in {sorted(wanted)}")
    else:
        specs = all_specs

    run_suite_a(
        specs,
        traces,
        args.output_dir,
        repo_root=args.repo_root,
        warmup_count=args.warmup,
        threads=args.threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
