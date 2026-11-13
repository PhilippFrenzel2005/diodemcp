"""Sequential Suite B live MCP stdio wire-protocol benchmark orchestrator."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from diodemcp.engine import write_active_bundle_config
from diodemcp.ipc_client import EngineClient, ensure_daemon_running
from diodemcp.transport import TransportError, default_transport, read_admin_secret

from benchmarks.contenders.protocol import decode_ipc_message, encode_ipc_message
from benchmarks.runner.metrics import percentile
from benchmarks.traces.schema import WireScenario
from benchmarks.traces.suite_b_scenarios import (
    build_suite_b_scenarios,
    get_server_init_state,
    get_server_poison_mode,
)

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SuiteBContenderSpec:
    """Configuration for one Suite B contender."""

    contender_id: str
    kind: str  # raw_stdio | diodemcp_shim | competitor_proxy | cisco_stdio
    python_bin: Path
    diodemcp_bin: Path | None = None
    bundle_dir: Path | None = None
    tier0_only: bool = False
    proxy_mode: str | None = None


def build_default_suite_b_contenders(
    venvs_root: Path,
    bundles_root: Path,
) -> list[SuiteBContenderSpec]:
    """Return the canonical 7-contender Suite B lineup on eclipse."""
    py_diodemcp = venvs_root / "venv-diodemcp" / "bin" / "python"
    bin_diodemcp = venvs_root / "venv-diodemcp" / "bin" / "diodemcp"
    py_llm_guard = venvs_root / "venv-llm-guard" / "bin" / "python"
    py_cisco = venvs_root / "venv-cisco" / "bin" / "python"
    py_baseline = venvs_root / "venv-baseline" / "bin" / "python"

    return [
        SuiteBContenderSpec(
            contender_id="raw-stdio-baseline",
            kind="raw_stdio",
            python_bin=py_diodemcp,
        ),
        SuiteBContenderSpec(
            contender_id="diodemcp-nanojev-0.6b",
            kind="diodemcp_shim",
            python_bin=py_diodemcp,
            diodemcp_bin=bin_diodemcp,
            bundle_dir=bundles_root / "DiodeMCP-NanoJev-0.6B",
        ),
        SuiteBContenderSpec(
            contender_id="diodemcp-base-149m",
            kind="diodemcp_shim",
            python_bin=py_diodemcp,
            diodemcp_bin=bin_diodemcp,
            bundle_dir=bundles_root / "DiodeMCP-Jev-Base-149M",
        ),
        SuiteBContenderSpec(
            contender_id="diodemcp-tier0-only",
            kind="diodemcp_shim",
            python_bin=py_diodemcp,
            diodemcp_bin=bin_diodemcp,
            tier0_only=True,
        ),
        SuiteBContenderSpec(
            contender_id="protectai-llm-guard",
            kind="competitor_proxy",
            python_bin=py_llm_guard,
            proxy_mode="llm_guard",
        ),
        SuiteBContenderSpec(
            contender_id="cisco-mcp-scanner-defs",
            kind="cisco_stdio",
            python_bin=py_cisco,
        ),
        SuiteBContenderSpec(
            contender_id="stateless-regex-policy",
            kind="competitor_proxy",
            python_bin=py_baseline,
            proxy_mode="stateless_policy",
        ),
    ]


def _build_env(repo_root: Path, threads: int) -> dict[str, str]:
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    repo_str = str(repo_root.resolve())
    env["PYTHONPATH"] = f"{repo_str}{os.pathsep}{existing_pp}" if existing_pp else repo_str
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["NUMEXPR_NUM_THREADS"] = str(threads)
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _is_firewall_block_response(resp: Mapping[str, Any]) -> bool:
    """Return True if a JSON-RPC response represents a firewall/proxy security block."""
    err = resp.get("error")
    if isinstance(err, Mapping):
        code = err.get("code")
        msg = str(err.get("message") or "").lower()
        if code == -32001 or "blocked by" in msg or "diodemcp" in msg:
            return True

    res = resp.get("result")
    if isinstance(res, Mapping) and res.get("isError") is True:
        if "incident_id" in res:
            return True
        for blk in res.get("content") or []:
            if isinstance(blk, Mapping):
                txt = str(blk.get("text") or "").lower()
                if "diodemcp" in txt or "blocked by" in txt:
                    return True
    return False


class _JsonRpcStdioClient:
    """Minimal synchronous JSON-RPC 2.0 stdio client for a wrapped MCP server."""

    def __init__(self, cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
        self._cmd = cmd
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(cwd),
            env=env,
        )
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._next_id = 1

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        try:
            for line in self._proc.stderr:
                self._stderr_tail.append(line.rstrip("\n"))
        except Exception:
            pass

    def initialize(self) -> None:
        resp = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "diodemcp-suite-b", "version": "1.0"},
            },
        )
        if "error" in resp:
            raise RuntimeError(f"MCP initialize failed: {resp}")
        self.notify("notifications/initialized", {})

    def notify(self, method: str, params: dict[str, Any]) -> None:
        assert self._proc.stdin is not None
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        req_id = self._next_id
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

        while True:
            line = self._proc.stdout.readline()
            if not line:
                tail = "\n".join(self._stderr_tail)
                raise RuntimeError(
                    f"MCP stdio server exited before responding to {method} "
                    f"(rc={self._proc.poll()}).\n{tail}"
                )
            stripped = line.strip()
            if not stripped:
                continue
            msg = json.loads(stripped)
            if isinstance(msg, dict) and msg.get("id") == req_id:
                return msg

    def close(self) -> None:
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=4.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._stderr_thread.join(timeout=2.0)


def _shutdown_diodemcp_daemon(state_dir: Path) -> None:
    """Cleanly stop any `DiodeEngineServer` daemon associated with `state_dir`."""
    cfg = default_transport(state_dir)
    if not cfg.addr_file.exists() or not cfg.admin_secret_file.exists():
        return
    try:
        secret = read_admin_secret(cfg)
        EngineClient(cfg, auto_spawn=False).shutdown(admin_secret=secret)
    except (TransportError, ValueError, OSError):
        return
    # Wait briefly for the daemon process to finish unlinking socket/pid files
    # before TemporaryDirectory cleanup runs.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            EngineClient(cfg, auto_spawn=False).status()
        except (TransportError, ValueError, OSError):
            break
        time.sleep(0.05)


def _build_wrapped_server_cmd(
    spec: SuiteBContenderSpec,
    *,
    server_name: str,
    mock_cmd: list[str],
    state_dir: Path | None,
    ws_root: Path,
    threads: int,
) -> list[str]:
    if spec.kind == "raw_stdio":
        return mock_cmd
    if spec.kind == "diodemcp_shim":
        assert state_dir is not None
        if spec.diodemcp_bin is not None and spec.diodemcp_bin.is_file():
            shim_prefix = [str(spec.diodemcp_bin), "shim"]
        else:
            shim_prefix = [str(spec.python_bin), "-m", "diodemcp.cli", "shim"]
        return [
            *shim_prefix,
            "--server",
            server_name,
            "--state-dir",
            str(state_dir),
            "--workspace-root",
            str(ws_root),
            "--",
            *mock_cmd,
        ]
    if spec.kind == "competitor_proxy":
        assert spec.proxy_mode is not None
        return [
            str(spec.python_bin),
            "-m",
            "benchmarks.wire.competitor_stdio_proxy",
            "--mode",
            spec.proxy_mode,
            "--threads",
            str(threads),
            "--",
            *mock_cmd,
        ]
    raise ValueError(f"Unsupported wrapped command kind: {spec.kind!r}")


def _run_cisco_suite_b(
    spec: SuiteBContenderSpec,
    scenarios: Sequence[WireScenario],
    *,
    repo_root: Path,
    env: dict[str, str],
    warmup_count: int,
) -> dict[str, Any]:
    worker_cmd = [
        str(spec.python_bin),
        "-m",
        "benchmarks.contenders.cisco_scanner_worker",
        "--contender-id",
        spec.contender_id,
        "--scan-mode",
        "definitions_only",
    ]
    proc = subprocess.Popen(
        worker_cmd,
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
    stderr_tail: deque[str] = deque(maxlen=40)

    def _drain() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_tail.append(line.rstrip("\n"))

    t_err = threading.Thread(target=_drain, daemon=True)
    t_err.start()

    def _rpc(payload: dict[str, Any]) -> dict[str, Any]:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(encode_ipc_message(payload) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(
                f"Cisco worker exited early (rc={proc.poll()}): {' '.join(stderr_tail)}"
            )
        return decode_ipc_message(line)

    try:
        ping_resp = _rpc({"cmd": "ping"})
        if not ping_resp.get("ok"):
            raise RuntimeError(f"Cisco worker ping failed: {ping_resp}")

        with tempfile.TemporaryDirectory(prefix="suiteb_cisco_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            # Untimed warmup scans
            warm_state = tmp_root / "warm_state.json"
            warm_state.write_text('{"tools_list_calls":0,"poison_mode":"none"}', encoding="utf-8")
            for _ in range(max(0, warmup_count)):
                _rpc(
                    {
                        "cmd": "scan_stdio",
                        "command": sys.executable,
                        "args": [
                            "-m",
                            "benchmarks.wire.mock_mcp_servers",
                            "--profile",
                            "fs_git",
                            "--state-file",
                            str(warm_state),
                        ],
                        "env": env,
                    }
                )

            scenario_results: list[dict[str, Any]] = []
            benign_latencies_ms: list[float] = []

            for scenario in scenarios:
                scen_dir = tmp_root / scenario.scenario_id
                scen_dir.mkdir(parents=True, exist_ok=True)
                state_files: dict[str, Path] = {}
                for srv in scenario.servers:
                    sf = scen_dir / f"{srv}.json"
                    sf.write_text(
                        json.dumps(get_server_init_state(scenario, srv)),
                        encoding="utf-8",
                    )
                    state_files[srv] = sf

                seen_session_servers: set[tuple[int, str]] = set()
                scen_blocked = False
                scen_lat_ms = 0.0
                for step in scenario.steps:
                    if step.method != "tools/list":
                        continue
                    key = (step.session_index, step.server)
                    if key in seen_session_servers:
                        continue
                    seen_session_servers.add(key)
                    poison_mode = get_server_poison_mode(scenario, step.server)
                    resp = _rpc(
                        {
                            "cmd": "scan_stdio",
                            "command": str(spec.python_bin),
                            "args": [
                                "-m",
                                "benchmarks.wire.mock_mcp_servers",
                                "--profile",
                                step.server,
                                "--state-file",
                                str(state_files[step.server]),
                                "--poison-mode",
                                poison_mode,
                            ],
                            "env": env,
                        }
                    )
                    lat = float(resp.get("latency_ms", 0.0))
                    scen_lat_ms += lat
                    if scenario.label == 0:
                        benign_latencies_ms.append(lat)
                    if resp.get("blocked_protective"):
                        scen_blocked = True

                scenario_results.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "category": scenario.category,
                        "label": scenario.label,
                        "blocked": scen_blocked,
                        "latency_ms": scen_lat_ms,
                    }
                )

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
        proc.wait(timeout=10.0)
        t_err.join(timeout=2.0)

    return _aggregate_suite_b_results(
        spec.contender_id,
        scenario_results,
        benign_latencies_ms,
        raw_p50_ms=None,
        raw_p95_ms=None,
    )


def _aggregate_suite_b_results(
    contender_id: str,
    scenario_results: Sequence[Mapping[str, Any]],
    benign_latencies_ms: Sequence[float],
    *,
    raw_p50_ms: float | None,
    raw_p95_ms: float | None,
) -> dict[str, Any]:
    attacks = [r for r in scenario_results if int(r["label"]) == 1]
    benigns = [r for r in scenario_results if int(r["label"]) == 0]
    atk_blocked = sum(1 for r in attacks if r["blocked"])
    ben_fp = sum(1 for r in benigns if r["blocked"])

    p50_ms = percentile(benign_latencies_ms, 50.0)
    p95_ms = percentile(benign_latencies_ms, 95.0)
    if raw_p50_ms is None:
        overhead_p50 = 0.0
        overhead_p95 = 0.0
    else:
        overhead_p50 = max(0.0, p50_ms - raw_p50_ms)
        overhead_p95 = max(0.0, p95_ms - raw_p95_ms)

    per_category: dict[str, dict[str, Any]] = {}
    for item in scenario_results:
        cat = str(item["category"])
        bucket = per_category.setdefault(cat, {"total": 0, "blocked": 0})
        bucket["total"] += 1
        if item["blocked"]:
            bucket["blocked"] += 1

    return {
        "contender_id": contender_id,
        "total_scenarios": len(scenario_results),
        "attack_total": len(attacks),
        "attack_blocked": atk_blocked,
        "attack_recall": (atk_blocked / len(attacks)) if attacks else 0.0,
        "benign_total": len(benigns),
        "benign_fp": ben_fp,
        "benign_fpr": (ben_fp / len(benigns)) if benigns else 0.0,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "overhead_p50_ms": overhead_p50,
        "overhead_p95_ms": overhead_p95,
        "per_category": per_category,
        "scenarios": list(scenario_results),
    }


def run_single_contender_suite_b(
    spec: SuiteBContenderSpec,
    scenarios: Sequence[WireScenario],
    *,
    repo_root: Path = DEFAULT_REPO_ROOT,
    warmup_count: int = 5,
    threads: int = 4,
    raw_p50_ms: float | None = None,
    raw_p95_ms: float | None = None,
) -> dict[str, Any]:
    """Run all Suite B scenarios sequentially for one contender."""
    env = _build_env(repo_root, threads)
    if spec.kind == "cisco_stdio":
        return _run_cisco_suite_b(
            spec,
            scenarios,
            repo_root=repo_root,
            env=env,
            warmup_count=warmup_count,
        )

    with tempfile.TemporaryDirectory(
        prefix=f"suiteb_{spec.contender_id}_",
        ignore_cleanup_errors=True,
    ) as tmp_dir:
        tmp_root = Path(tmp_dir)
        state_dir: Path | None = None
        if spec.kind == "diodemcp_shim":
            state_dir = tmp_root / "diodemcp_state"
            state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            write_active_bundle_config(
                state_dir,
                bundle_dir=spec.bundle_dir,
                tier0_only=spec.tier0_only,
            )
            ensure_daemon_running(
                default_transport(state_dir),
                bundle_dir=spec.bundle_dir if not spec.tier0_only else None,
            )

        try:
            # Untimed warmup calls
            if warmup_count > 0:
                warm_ws = tmp_root / "workspaces" / "_warmup"
                warm_ws.mkdir(parents=True, exist_ok=True)
                (warm_ws / ".git").mkdir(exist_ok=True)
                warm_state = tmp_root / "warm_mock.json"
                warm_state.write_text('{"tools_list_calls":0,"poison_mode":"none"}', encoding="utf-8")
                mock_py = str(spec.python_bin)
                warm_mock_cmd = [
                    mock_py,
                    "-m",
                    "benchmarks.wire.mock_mcp_servers",
                    "--profile",
                    "fs_git",
                    "--state-file",
                    str(warm_state),
                ]
                warm_cmd = _build_wrapped_server_cmd(
                    spec,
                    server_name="fs_git",
                    mock_cmd=warm_mock_cmd,
                    state_dir=state_dir,
                    ws_root=warm_ws,
                    threads=threads,
                )
                client = _JsonRpcStdioClient(warm_cmd, cwd=repo_root, env=env)
                try:
                    client.initialize()
                    for _ in range(warmup_count):
                        client.request("tools/call", {"name": "git_status", "arguments": {}})
                finally:
                    client.close()
                if state_dir is not None:
                    (state_dir / "tool_pins.json").unlink(missing_ok=True)

            scenario_results: list[dict[str, Any]] = []
            benign_latencies_ms: list[float] = []

            for scenario in scenarios:
                if state_dir is not None:
                    # Isolate tool_pins.json across scenarios while keeping daemon/ORT warm.
                    (state_dir / "tool_pins.json").unlink(missing_ok=True)

                ws_root = tmp_root / "workspaces" / scenario.scenario_id
                ws_root.mkdir(parents=True, exist_ok=True)
                (ws_root / ".git").mkdir(exist_ok=True)

                mock_state_dir = tmp_root / "mock_states" / scenario.scenario_id
                mock_state_dir.mkdir(parents=True, exist_ok=True)
                state_files: dict[str, Path] = {}
                for srv in scenario.servers:
                    sf = mock_state_dir / f"{srv}.json"
                    sf.write_text(
                        json.dumps(get_server_init_state(scenario, srv)),
                        encoding="utf-8",
                    )
                    state_files[srv] = sf

                active_clients: dict[str, _JsonRpcStdioClient] = {}
                cur_session: int | None = None
                scen_blocked = False
                scen_lat_ms = 0.0

                def _close_active_clients() -> None:
                    for cl in active_clients.values():
                        cl.close()
                    active_clients.clear()

                try:
                    for step in scenario.steps:
                        if cur_session is not None and step.session_index != cur_session:
                            _close_active_clients()
                        cur_session = step.session_index

                        client = active_clients.get(step.server)
                        if client is None:
                            poison_mode = get_server_poison_mode(scenario, step.server)
                            mock_cmd = [
                                str(spec.python_bin),
                                "-m",
                                "benchmarks.wire.mock_mcp_servers",
                                "--profile",
                                step.server,
                                "--state-file",
                                str(state_files[step.server]),
                                "--poison-mode",
                                poison_mode,
                            ]
                            wrapped_cmd = _build_wrapped_server_cmd(
                                spec,
                                server_name=step.server,
                                mock_cmd=mock_cmd,
                                state_dir=state_dir,
                                ws_root=ws_root,
                                threads=threads,
                            )
                            client = _JsonRpcStdioClient(wrapped_cmd, cwd=repo_root, env=env)
                            client.initialize()
                            active_clients[step.server] = client

                        t0 = time.perf_counter_ns()
                        resp = client.request(step.method, dict(step.params))
                        step_lat_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                        scen_lat_ms += step_lat_ms
                        if scenario.label == 0:
                            benign_latencies_ms.append(step_lat_ms)
                        if _is_firewall_block_response(resp):
                            scen_blocked = True
                finally:
                    _close_active_clients()

                scenario_results.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "category": scenario.category,
                        "label": scenario.label,
                        "blocked": scen_blocked,
                        "latency_ms": scen_lat_ms,
                    }
                )
        finally:
            if state_dir is not None:
                _shutdown_diodemcp_daemon(state_dir)

    return _aggregate_suite_b_results(
        spec.contender_id,
        scenario_results,
        benign_latencies_ms,
        raw_p50_ms=raw_p50_ms,
        raw_p95_ms=raw_p95_ms,
    )


def run_suite_b(
    contenders: Sequence[SuiteBContenderSpec],
    scenarios: Sequence[WireScenario],
    output_dir: Path,
    *,
    repo_root: Path = DEFAULT_REPO_ROOT,
    warmup_count: int = 5,
    threads: int = 4,
) -> list[dict[str, Any]]:
    """Run all Suite B contenders sequentially and persist `suite_b_summary.json`."""
    summaries: list[dict[str, Any]] = []
    raw_p50: float | None = None
    raw_p95: float | None = None

    for spec in contenders:
        sys.stderr.write(f"[suite_b] Running contender: {spec.contender_id} ({len(scenarios)} scenarios)\n")
        sys.stderr.flush()
        summary = run_single_contender_suite_b(
            spec,
            scenarios,
            repo_root=repo_root,
            warmup_count=warmup_count,
            threads=threads,
            raw_p50_ms=raw_p50,
            raw_p95_ms=raw_p95,
        )
        if spec.kind == "raw_stdio" and raw_p50 is None:
            raw_p50 = float(summary["p50_ms"])
            raw_p95 = float(summary["p95_ms"])
        summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "suite_b_summary.json"
    summary_path.write_text(
        json.dumps({"suite_b": summaries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sequential Suite B live MCP stdio benchmark runner")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--venvs-root",
        type=Path,
        default=Path("/srv/homelab/diodemcp-bench/.venvs"),
    )
    parser.add_argument(
        "--bundles-root",
        type=Path,
        default=Path("/srv/homelab/diodemcp-bench/bundles"),
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--contenders", type=str, default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    scenarios = build_suite_b_scenarios()
    all_specs = build_default_suite_b_contenders(args.venvs_root, args.bundles_root)
    if args.contenders.strip():
        wanted = {c.strip() for c in args.contenders.split(",") if c.strip()}
        specs = [s for s in all_specs if s.contender_id in wanted]
        if not specs:
            raise SystemExit(f"No matching contenders in {sorted(wanted)}")
    else:
        specs = all_specs

    run_suite_b(
        specs,
        scenarios,
        args.output_dir,
        repo_root=args.repo_root,
        warmup_count=args.warmup,
        threads=args.threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
