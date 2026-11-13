"""Persistent DiodeMCP contender worker over JSONL stdin/stdout IPC."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from diodemcp.ort_predictor import OrtJevPredictor
from diodemcp.predictor import PredictorProtocol, Tier0DeterministicPredictor
from diodemcp.shim import _TOOL_POISONING_RE, _cap_ipc_string, _normalize_for_screening
from diodemcp.transport import default_transport
from diodemcp.types import CapabilityFlag

from benchmarks.contenders.protocol import TraceEvalResult
from benchmarks.contenders.stateless_policy_worker import run_worker_loop
from benchmarks.traces.schema import MultiTurnTrace, ToolDefinition

_PATH_KEYS: frozenset[str] = frozenset(
    {
        "path",
        "file_path",
        "filepath",
        "target_file",
        "target_path",
        "absolute_path",
        "source",
        "destination",
        "src",
        "dst",
        "uri",
        "filename",
        "file",
    }
)


def _collect_candidate_paths(args: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for k, v in args.items():
        if str(k).lower() in _PATH_KEYS and isinstance(v, str) and v.strip():
            paths.append(v.strip())
    return paths


def _materialize_relative_file(ws_root: Path, raw_path: str, content: str) -> None:
    """Materialize workspace-relative or /workspace/app/... files safely inside `ws_root`."""
    cleaned = raw_path.strip()
    if not cleaned or "://" in cleaned or cleaned.startswith("~"):
        return
    if cleaned.startswith("/workspace/app/"):
        rel = cleaned[len("/workspace/app/") :]
    elif cleaned.startswith("/"):
        return
    else:
        rel = cleaned
    rel = rel.lstrip("/")
    if not rel or rel == ".":
        return
    try:
        candidate = (ws_root / rel).resolve()
        ws_resolved = ws_root.resolve()
        candidate.relative_to(ws_resolved)
    except (ValueError, OSError):
        return
    if candidate.exists():
        return
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(content, encoding="utf-8", errors="replace")
    except OSError:
        pass


def _resolve_eval_cwd(ws_root: Path, raw_cwd: str) -> Path:
    """Map trace cwd like `/workspace/app/sub` into `ws_root` while keeping `.git` root."""
    cleaned = (raw_cwd or "").strip()
    if cleaned.startswith("/workspace/app/"):
        sub = cleaned[len("/workspace/app/") :].strip("/")
        if sub:
            target = (ws_root / sub).resolve()
            try:
                target.relative_to(ws_root.resolve())
                target.mkdir(parents=True, exist_ok=True)
                return target
            except (ValueError, OSError):
                pass
    return ws_root


def _screen_catalog(
    tools: Sequence[ToolDefinition],
    pins: dict[tuple[str, str], str],
) -> dict[str, list[str]]:
    """Mirror `diodemcp.shim._McpShimCore._screen_metadata_entries` for `tools/list`."""
    suspicious_by_server: dict[str, list[str]] = {}
    for td in tools:
        pin_key = (td.server, f"tool:{td.name}")
        canonical_obj = {
            "name": td.name,
            "title": None,
            "description": td.description,
            "inputSchema": td.input_schema,
            "outputSchema": None,
            "annotations": None,
        }
        text_parts = [td.name, td.description]
        if td.input_schema:
            text_parts.append(
                json.dumps(td.input_schema, sort_keys=True, ensure_ascii=False)
            )
        try:
            canonical_json = json.dumps(
                canonical_obj,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except Exception:
            canonical_json = "\n".join(text_parts)

        digest = hashlib.sha256(canonical_json.encode("utf-8", errors="replace")).hexdigest()
        prev_digest = pins.get(pin_key)
        if prev_digest is None:
            pins[pin_key] = digest
            is_rug_pull = False
        else:
            is_rug_pull = prev_digest != digest

        raw_combined = "\n".join(p for p in text_parts if p)
        normalized_combined = _normalize_for_screening(raw_combined)
        is_poisoned = _TOOL_POISONING_RE.search(normalized_combined) is not None
        if is_poisoned or is_rug_pull:
            tag = "rug_pull" if is_rug_pull else "poisoned_tool"
            suspicious_by_server.setdefault(td.server, []).append(
                f"[{td.server}:{td.name}:{tag}] {normalized_combined}".strip()
            )
    return suspicious_by_server


class DiodeMcpEvaluator:
    """In-process evaluator keeping `OrtJevPredictor`/`Tier0` warm while isolating every trace."""

    def __init__(
        self,
        contender_id: str,
        mode: str,
        *,
        bundle_dir: Path | str | None = None,
        threads: int = 4,
    ) -> None:
        if mode not in ("tier0", "ort"):
            raise ValueError(f"Unsupported mode {mode!r}; expected 'tier0' or 'ort'")
        self.contender_id = contender_id
        self.mode = mode
        t0 = time.perf_counter_ns()
        if mode == "tier0":
            self.predictor: PredictorProtocol = Tier0DeterministicPredictor()
        else:
            if bundle_dir is None:
                raise ValueError("--bundle-dir is required when --mode ort")
            self.predictor = OrtJevPredictor(bundle_dir, intra_op_threads=threads)
        self.cold_start_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

    def evaluate_trace(self, trace: MultiTurnTrace) -> TraceEvalResult:
        """Evaluate a single trace in a completely isolated temporary state & workspace dir."""
        from diodemcp.engine import DiodeEngineServer

        t_start = time.perf_counter_ns()
        tools_list_tainted = False

        with tempfile.TemporaryDirectory(prefix="dmcp_trace_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            state_dir = tmp_root / "state"
            state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            ws_root = tmp_root / "workspace"
            ws_root.mkdir(parents=True, exist_ok=True)
            (ws_root / ".git").mkdir(exist_ok=True)

            # Materialize referenced workspace/secret files inside ws_root
            for turn in trace.prior_turns:
                for p in _collect_candidate_paths(turn.arguments):
                    _materialize_relative_file(ws_root, p, turn.output_text[:4096])
            for p in _collect_candidate_paths(trace.candidate_call.arguments):
                _materialize_relative_file(ws_root, p, "")

            eval_cwd = _resolve_eval_cwd(ws_root, trace.candidate_call.cwd)

            cfg = default_transport(state_dir)
            engine = DiodeEngineServer(cfg, predictor=self.predictor)

            pins: dict[tuple[str, str], str] = {}
            if trace.initial_tools_catalog is not None:
                init_suspicious = _screen_catalog(trace.initial_tools_catalog, pins)
                for srv, entries in init_suspicious.items():
                    tools_list_tainted = True
                    engine.record_output(
                        ws_root,
                        srv,
                        "__tools_list__",
                        {"source_tool": f"{srv}:__tools_list__"},
                        _cap_ipc_string("\n".join(entries)),
                        extra_flags=CapabilityFlag.READS_UNTRUSTED,
                    )

            cur_suspicious = _screen_catalog(trace.tools_catalog, pins)
            for srv, entries in cur_suspicious.items():
                tools_list_tainted = True
                engine.record_output(
                    ws_root,
                    srv,
                    "__tools_list__",
                    {"source_tool": f"{srv}:__tools_list__"},
                    _cap_ipc_string("\n".join(entries)),
                    extra_flags=CapabilityFlag.READS_UNTRUSTED,
                )

            for intent in trace.user_intents:
                engine.record_intent(ws_root, intent)

            for turn in trace.prior_turns:
                extra_flags = (
                    CapabilityFlag.READS_UNTRUSTED
                    if turn.is_error
                    else CapabilityFlag.LOCAL_SAFE
                )
                engine.record_output(
                    ws_root,
                    turn.server,
                    turn.tool,
                    dict(turn.arguments),
                    _cap_ipc_string(turn.output_text),
                    extra_flags=extra_flags,
                )

            decision = engine.evaluate_call(
                eval_cwd,
                trace.candidate_call.server,
                trace.candidate_call.tool,
                dict(trace.candidate_call.arguments),
                client_kind="headless",
            )

        latency_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0
        band = str(decision.get("band") or "safe")
        action = str(decision.get("action") or "allow")
        p_max = float(decision.get("p_max", 0.0))
        blocked_protective = (band in ("warn", "critical")) or (action != "allow")
        blocked_hard = band == "critical"
        blocked_stage = "candidate_call" if blocked_protective else "none"
        reason = str(decision.get("reason") or f"band={band}, action={action}")

        return TraceEvalResult(
            trace_id=trace.trace_id,
            contender_id=self.contender_id,
            blocked_protective=blocked_protective,
            blocked_hard=blocked_hard,
            risk_score=p_max,
            latency_ms=latency_ms,
            blocked_stage=blocked_stage,
            reason=reason,
            extra={
                "band": band,
                "action": action,
                "scores": dict(decision.get("scores") or {}),
                "tools_list_tainted": tools_list_tainted,
            },
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DiodeMCP Suite A contender worker")
    parser.add_argument("--contender-id", required=True)
    parser.add_argument("--mode", choices=("tier0", "ort"), required=True)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(list(argv) if argv is not None else None)

    evaluator = DiodeMcpEvaluator(
        contender_id=args.contender_id,
        mode=args.mode,
        bundle_dir=args.bundle_dir,
        threads=args.threads,
    )
    return run_worker_loop(evaluator, args.contender_id, evaluator.cold_start_ms)


if __name__ == "__main__":
    raise SystemExit(main())
