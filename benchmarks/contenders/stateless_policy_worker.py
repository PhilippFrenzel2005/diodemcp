"""Stateless regex & static policy MCP firewall worker (pure Python stdlib)."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
from typing import Any

from benchmarks.contenders.protocol import (
    TraceEvalResult,
    decode_ipc_message,
    encode_ipc_message,
)
from benchmarks.traces.schema import MultiTurnTrace, ToolDefinition

# Tool-poisoning patterns (stateless: no NFKC/Cf stripping, no pin-hash tracking).
_TOOL_POISONING_RE = re.compile(
    r"(?:"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|system)\s+instructions"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|above)"
    r"|<important>|</important>|<system>|</system>|<\|intent\|>|<\|missing_intent\|>|<\|im_start\|>|\[inst\]"
    r"|~/\.cursor/mcp\.json|~/\.claude\b|~/\.diodemcp\b"
    r"|\b(?:important|critical|urgent)\s*:\s*(?:ignore|before|first|always|do\s+not|never)\b"
    r"|\b(?:do\s+not|don't|never)\s+(?:tell|inform|mention|show|reveal)\s+(?:to\s+)?(?:the\s+)?user\b"
    r"|\b(?:before|prior\s+to)\s+(?:calling|using|invoking)\s+this\s+tool\b"
    r"|\b(?:read|cat|open|include|exfiltrate|upload|send|steal)\b[^\n]{0,160}"
    r"(?:\.env\b|id_rsa|id_ed25519|\.ssh|\.aws|admin\.secret|\.diodemcp|credentials|/etc/passwd)"
    r"|\b(?:send|post|upload|exfiltrate|forward|curl|wget)\b[^\n]{0,160}https?://[^\s\"'<>]+"
    r")",
    re.IGNORECASE,
)

# Gitleaks-grade secret regexes.
_GITLEAKS_SECRET_RE = re.compile(
    r"(?:"
    r"-----BEGIN (?:[A-Z0-9 ]{1,32} )?PRIVATE KEY-----"
    r"|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"
    r"|\bgh[pousr]_[A-Za-z0-9]{12,}\b"
    r"|\bgithub_pat_[A-Za-z0-9_]{16,}\b"
    r"|\b[sr]k_(?:live|test)_[A-Za-z0-9]{8,}\b"
    r"|\bwhsec_[A-Za-z0-9]{12,}\b"
    r"|\bAIza[0-9A-Za-z_-]{20,}\b"
    r"|\bxox[baprs]-[0-9A-Za-z-]{10,}\b"
    r"|\bglpat-[0-9A-Za-z_-]{16,}\b"
    r"|\bnpm_[A-Za-z0-9]{20,}\b"
    r"|\bpypi-[A-Za-z0-9_-]{20,}\b"
    r"|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
    r"|://[^:/@\s]+:([^@\s]{6,})@"
    r"|\b(?:api_key|secret_key|private_key|auth_token|access_token|password|passwd)\b\s*[:=]\s*[\"']?[A-Za-z0-9_/+=-]{12,}"
    r")",
    re.IGNORECASE,
)

_DESTRUCTIVE_SHELL_SQL_RE = re.compile(
    r"(?:"
    r"\brm\s+-[A-Za-z]*[rRf]{2,}\b"
    r"|\brm\s+-[A-Za-z]*r[A-Za-z]*\s+-[A-Za-z]*f\b"
    r"|\bgit\s+push\b[^\n;|&]{0,120}(?:--force-with-lease|--force|-f)\b"
    r"|\bgit\s+reset\s+--hard\b"
    r"|\bmkfs(?:\.\w+)?\b"
    r"|\bdd\s+if="
    r"|(?:curl|wget)\b[^\n]{0,512}\|\s*(?:sudo\s+)?(?:ba|z|da|k)?sh\b"
    r"|\bDROP\s+(?:TABLE|DATABASE|SCHEMA|INDEX)\b"
    r"|\bTRUNCATE\s+TABLE\b"
    r"|\bDELETE\s+FROM\s+[A-Za-z0-9_.\"`]+\s*(?:;|$)"
    r")",
    re.IGNORECASE,
)

_SENSITIVE_DOTFILE_PATH_RE = re.compile(
    r"(?:"
    r"(?:^|[/\\\"'\s=@,(])\.env(?:\.[A-Za-z0-9_.-]+)?(?:$|[/\\\"'\s;|&><,)])"
    r"|(?:^|[~/\\\"'\s=@,(])\.ssh[/\\]"
    r"|(?:^|[~/\\\"'\s=@,(])\.aws[/\\]"
    r"|(?:^|[~/\\\"'\s=@,(])\.kube[/\\]"
    r"|(?:^|[~/\\\"'\s=@,(])(?:\.git-credentials|\.netrc|\.npmrc|\.pypirc)"
    r"|(?:^|[~/\\\"'\s=@,(])\.diodemcp(?:[/\\]|$)"
    r"|(?:^|[~/\\\"'\s=@,(])\.cursor[/\\]mcp\.json"
    r"|(?:^|[~/\\\"'\s=@,(])\.claude(?:[/\\]|\.json)"
    r"|\bid_(?:rsa|ed25519|ecdsa|dsa)\b"
    r"|/etc/(?:passwd|shadow)\b"
    r")",
    re.IGNORECASE,
)


def read_process_peak_rss_mb() -> float:
    """Return peak RSS memory in MiB from /proc/self/status (VmHWM/VmRSS) and rusage."""
    peak_kb = 0
    try:
        peak_kb = max(peak_kb, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    except Exception:
        pass
    proc_status = Path("/proc/self/status")
    if proc_status.is_file():
        try:
            for line in proc_status.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith(("VmHWM:", "VmRSS:")):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        peak_kb = max(peak_kb, int(parts[1]))
        except OSError:
            pass
    return peak_kb / 1024.0


def setup_pristine_ipc_stdout() -> Any:
    """Duplicate fd 1 for clean JSONL IPC and redirect stdout -> stderr."""
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(proto_fd, "w", encoding="utf-8", buffering=1, closefd=True)


def run_worker_loop(
    evaluator: Any,
    contender_id: str,
    cold_start_ms: float,
    *,
    proto_out: Any = None,
    extra_handlers: Mapping[str, Any] | None = None,
) -> int:
    """Read JSONL commands from stdin and emit responses over pristine IPC stdout."""
    own_proto = proto_out is None
    if own_proto:
        proto_out = setup_pristine_ipc_stdout()
    try:
        for raw_line in sys.stdin:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                msg = decode_ipc_message(stripped)
            except Exception as exc:
                proto_out.write(
                    encode_ipc_message({"ok": False, "error": f"Invalid IPC JSON: {exc}"}) + "\n"
                )
                proto_out.flush()
                continue

            cmd = msg.get("cmd")
            if cmd == "ping":
                proto_out.write(
                    encode_ipc_message(
                        {
                            "ok": True,
                            "contender_id": contender_id,
                            "cold_start_ms": cold_start_ms,
                            "rss_mb": read_process_peak_rss_mb(),
                        }
                    )
                    + "\n"
                )
                proto_out.flush()
            elif cmd == "eval_trace":
                try:
                    trace = MultiTurnTrace.from_dict(msg["trace"])
                    result = evaluator.evaluate_trace(trace)
                    proto_out.write(
                        encode_ipc_message(
                            {
                                "ok": True,
                                "result": result.to_dict(),
                                "rss_mb": read_process_peak_rss_mb(),
                            }
                        )
                        + "\n"
                    )
                except Exception as exc:
                    trace_id = str((msg.get("trace") or {}).get("trace_id") or "unknown")
                    err_res = TraceEvalResult(
                        trace_id=trace_id,
                        contender_id=contender_id,
                        blocked_protective=True,
                        blocked_hard=True,
                        risk_score=1.0,
                        latency_ms=0.0,
                        blocked_stage="error",
                        reason=f"Worker error: {exc}",
                    )
                    proto_out.write(
                        encode_ipc_message(
                            {
                                "ok": False,
                                "error": str(exc),
                                "result": err_res.to_dict(),
                                "rss_mb": read_process_peak_rss_mb(),
                            }
                        )
                        + "\n"
                    )
                proto_out.flush()
            elif cmd == "shutdown":
                proto_out.write(encode_ipc_message({"ok": True}) + "\n")
                proto_out.flush()
                return 0
            elif extra_handlers and cmd in extra_handlers:
                try:
                    resp = dict(extra_handlers[cmd](msg))
                    resp.setdefault("ok", True)
                    resp.setdefault("rss_mb", read_process_peak_rss_mb())
                except Exception as exc:
                    resp = {"ok": False, "error": str(exc), "rss_mb": read_process_peak_rss_mb()}
                proto_out.write(encode_ipc_message(resp) + "\n")
                proto_out.flush()
            else:
                proto_out.write(
                    encode_ipc_message({"ok": False, "error": f"Unknown cmd: {cmd!r}"}) + "\n"
                )
                proto_out.flush()
        return 0
    finally:
        if own_proto:
            proto_out.close()


def _flatten_strings(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        if value:
            out.append(value)
    elif isinstance(value, Mapping):
        for k, v in value.items():
            if isinstance(k, str) and k:
                out.append(k)
            _flatten_strings(v, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _flatten_strings(item, out)


class StatelessPolicyEvaluator:
    """Stateless regex/policy evaluator for Suite A and Suite B proxy."""

    def __init__(self, contender_id: str = "stateless-regex-policy") -> None:
        t0 = time.perf_counter_ns()
        self.contender_id = contender_id
        self.cold_start_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

    def check_tools_list(self, tools: Sequence[ToolDefinition | Mapping[str, Any]]) -> str | None:
        """Return match reason if any tool metadata triggers `_TOOL_POISONING_RE`, else None."""
        for item in tools:
            if isinstance(item, ToolDefinition):
                name = item.name
                desc = item.description
                schema = item.input_schema
            else:
                name = str(item.get("name") or "")
                desc = str(item.get("description") or "")
                schema = item.get("inputSchema") or item.get("input_schema") or {}
            parts = [name, desc]
            if schema:
                parts.append(json.dumps(schema, sort_keys=True, ensure_ascii=False))
            text = "\n".join(p for p in parts if p)
            if _TOOL_POISONING_RE.search(text):
                return f"tool_poisoning_regex:{name}"
        return None

    def check_candidate_args(self, arguments: Mapping[str, Any]) -> str | None:
        """Return rule name if outgoing candidate arguments match secret/destructive/dotfile regexes."""
        parts: list[str] = []
        _flatten_strings(arguments, parts)
        combined = "\n".join(parts)
        if not combined:
            return None
        if _GITLEAKS_SECRET_RE.search(combined):
            return "gitleaks_secret_regex"
        if _DESTRUCTIVE_SHELL_SQL_RE.search(combined):
            return "destructive_shell_or_sql_regex"
        if _SENSITIVE_DOTFILE_PATH_RE.search(combined):
            return "sensitive_dotfile_path_regex"
        return None

    def evaluate_trace(self, trace: MultiTurnTrace) -> TraceEvalResult:
        """Evaluate a trace statelessly (tool catalog + candidate args; ignores prior_turns)."""
        t0 = time.perf_counter_ns()
        tools_hit = self.check_tools_list(trace.tools_catalog)
        if tools_hit is not None:
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return TraceEvalResult(
                trace_id=trace.trace_id,
                contender_id=self.contender_id,
                blocked_protective=True,
                blocked_hard=True,
                risk_score=1.0,
                latency_ms=latency_ms,
                blocked_stage="tools_list",
                reason=f"Stateless regex match: {tools_hit}",
            )

        args_hit = self.check_candidate_args(trace.candidate_call.arguments)
        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        if args_hit is not None:
            return TraceEvalResult(
                trace_id=trace.trace_id,
                contender_id=self.contender_id,
                blocked_protective=True,
                blocked_hard=True,
                risk_score=1.0,
                latency_ms=latency_ms,
                blocked_stage="candidate_call",
                reason=f"Stateless regex match: {args_hit}",
            )

        return TraceEvalResult(
            trace_id=trace.trace_id,
            contender_id=self.contender_id,
            blocked_protective=False,
            blocked_hard=False,
            risk_score=0.0,
            latency_ms=latency_ms,
            blocked_stage="none",
            reason="No stateless rule matched",
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stateless regex/policy Suite A contender worker")
    parser.add_argument("--contender-id", default="stateless-regex-policy")
    args = parser.parse_args(list(argv) if argv is not None else None)

    evaluator = StatelessPolicyEvaluator(contender_id=args.contender_id)
    return run_worker_loop(evaluator, args.contender_id, evaluator.cold_start_ms)


if __name__ == "__main__":
    raise SystemExit(main())
