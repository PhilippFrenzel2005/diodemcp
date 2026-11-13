"""Persistent Cisco AI Defense mcp-scanner contender worker over JSONL stdin/stdout IPC."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import sys
import time
from typing import Any, Protocol

from benchmarks.contenders.protocol import TraceEvalResult
from benchmarks.contenders.stateless_policy_worker import (
    run_worker_loop,
    setup_pristine_ipc_stdout,
)
from benchmarks.traces.schema import MultiTurnTrace

SEVERITY_RANK: dict[str, int] = {
    "NONE": 0,
    "INFO": 1,
    "UNKNOWN": 1,
    "LOW": 2,
    "MEDIUM": 3,
    "HIGH": 4,
    "CRITICAL": 5,
}

SEVERITY_SCORE: dict[str, float] = {
    "NONE": 0.0,
    "INFO": 0.15,
    "UNKNOWN": 0.15,
    "LOW": 0.35,
    "MEDIUM": 0.70,
    "HIGH": 0.95,
    "CRITICAL": 0.95,
}


@dataclass(frozen=True)
class CiscoFinding:
    """Normalized offline finding record."""

    severity: str
    summary: str
    analyzer: str = "yara"
    threat_category: str = ""


class CiscoAnalyzerBackend(Protocol):
    """Backend interface for Cisco mcp-scanner (injectable for fast unit tests)."""

    def analyze_yara(self, content: str, *, context: dict[str, Any]) -> list[CiscoFinding]: ...
    def analyze_schema(self, content: str, *, context: dict[str, Any]) -> list[CiscoFinding]: ...
    def scan_stdio_server(
        self, command: str, args: list[str], env: dict[str, str] | None = None
    ) -> list[CiscoFinding]: ...


class RealCiscoAnalyzerBackend:
    """Offline `cisco-ai-mcp-scanner==4.8.7` backend (YARA + SchemaReferenceAnalyzer)."""

    def __init__(self) -> None:
        from mcpscanner import AnalyzerEnum, Config, Scanner, YaraAnalyzer
        from mcpscanner.core.analyzers.schema_reference_analyzer import (
            SchemaReferenceAnalyzer,
        )
        from mcpscanner.core.mcp_models import StdioServer

        self._AnalyzerEnum = AnalyzerEnum
        self._Config = Config
        self._Scanner = Scanner
        self._StdioServer = StdioServer
        self._yara = YaraAnalyzer(config=None, rules_dir=None)
        self._schema = SchemaReferenceAnalyzer()
        self._loop = asyncio.new_event_loop()

    @staticmethod
    def _normalize_findings(raw_findings: Any) -> list[CiscoFinding]:
        out: list[CiscoFinding] = []
        if not raw_findings:
            return out
        for item in raw_findings:
            sev = str(getattr(item, "severity", "UNKNOWN") or "UNKNOWN").strip().upper()
            summary = str(getattr(item, "summary", "") or "")
            analyzer = str(getattr(item, "analyzer", "") or "")
            cat = str(getattr(item, "threat_category", "") or "")
            out.append(
                CiscoFinding(
                    severity=sev,
                    summary=summary,
                    analyzer=analyzer,
                    threat_category=cat,
                )
            )
        return out

    def analyze_yara(self, content: str, *, context: dict[str, Any]) -> list[CiscoFinding]:
        if not content.strip():
            return []
        raw = self._loop.run_until_complete(self._yara.analyze(content, context=context))
        return self._normalize_findings(raw)

    def analyze_schema(self, content: str, *, context: dict[str, Any]) -> list[CiscoFinding]:
        if not content.strip():
            return []
        raw = self._loop.run_until_complete(self._schema.analyze(content, context=context))
        return self._normalize_findings(raw)

    def scan_stdio_server(
        self, command: str, args: list[str], env: dict[str, str] | None = None
    ) -> list[CiscoFinding]:
        server = self._StdioServer(command=command, args=args, env=env)
        scanner = self._Scanner(self._Config())
        scan_results = self._loop.run_until_complete(
            scanner.scan_stdio_server_tools(
                server,
                analyzers=[self._AnalyzerEnum.YARA, self._AnalyzerEnum.SCHEMA],
            )
        )
        findings: list[CiscoFinding] = []
        for item in scan_results or []:
            nested = getattr(item, "findings", None)
            if nested is not None:
                findings.extend(self._normalize_findings(nested))
            elif hasattr(item, "severity"):
                findings.extend(self._normalize_findings([item]))
        return findings


def _summarize_findings(
    findings: Sequence[CiscoFinding],
) -> tuple[bool, bool, float, str]:
    max_rank = 0
    max_sev = "NONE"
    summaries: list[str] = []
    for f in findings:
        sev = f.severity.strip().upper()
        rank = SEVERITY_RANK.get(sev, 1)
        if rank > max_rank:
            max_rank = rank
            max_sev = sev
        if rank >= SEVERITY_RANK["LOW"]:
            summaries.append(f"[{sev}:{f.analyzer}] {f.summary}")

    blocked_protective = max_rank >= SEVERITY_RANK["MEDIUM"]
    blocked_hard = max_rank >= SEVERITY_RANK["HIGH"]
    risk_score = SEVERITY_SCORE.get(max_sev, 0.0)
    reason = "; ".join(summaries[:5]) if summaries else "No Cisco mcp-scanner findings"
    return blocked_protective, blocked_hard, risk_score, reason


class CiscoScannerEvaluator:
    """Suite A and Suite B evaluator wrapping Cisco mcp-scanner offline analyzers."""

    def __init__(
        self,
        contender_id: str,
        scan_mode: str = "definitions_only",
        *,
        backend: CiscoAnalyzerBackend | None = None,
    ) -> None:
        if scan_mode not in ("definitions_only", "extended_payload"):
            raise ValueError(
                f"Unsupported scan_mode {scan_mode!r}; expected 'definitions_only' or 'extended_payload'"
            )
        self.contender_id = contender_id
        self.scan_mode = scan_mode
        t0 = time.perf_counter_ns()
        self.backend: CiscoAnalyzerBackend = (
            backend if backend is not None else RealCiscoAnalyzerBackend()
        )
        self.cold_start_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

    def evaluate_trace(self, trace: MultiTurnTrace) -> TraceEvalResult:
        t0 = time.perf_counter_ns()
        all_findings: list[CiscoFinding] = []
        first_stage = "none"

        # 1. MCP tool definition scan (YaraAnalyzer on description + SchemaReferenceAnalyzer on schema)
        for td in trace.tools_catalog:
            desc_findings = self.backend.analyze_yara(
                td.description,
                context={"tool_name": td.name, "content_type": "description"},
            )
            schema_payload = json.dumps(
                {"name": td.name, "description": td.description, "inputSchema": td.input_schema},
                ensure_ascii=False,
                sort_keys=True,
            )
            schema_findings = self.backend.analyze_schema(
                schema_payload,
                context={"tool_name": td.name, "content_type": "parameters"},
            )
            tool_findings = [*desc_findings, *schema_findings]
            if tool_findings:
                all_findings.extend(tool_findings)
                if first_stage == "none" and any(
                    SEVERITY_RANK.get(f.severity.upper(), 1) >= SEVERITY_RANK["MEDIUM"]
                    for f in tool_findings
                ):
                    first_stage = "tools_list"

        # 2. Optional Extended Payload YARA scan on prior_turns & candidate_call
        if self.scan_mode == "extended_payload":
            for turn in trace.prior_turns:
                if not turn.output_text.strip():
                    continue
                turn_findings = self.backend.analyze_yara(
                    turn.output_text,
                    context={"tool_name": turn.tool, "content_type": "content"},
                )
                if turn_findings:
                    all_findings.extend(turn_findings)
                    if first_stage == "none" and any(
                        SEVERITY_RANK.get(f.severity.upper(), 1) >= SEVERITY_RANK["MEDIUM"]
                        for f in turn_findings
                    ):
                        first_stage = "prior_output"

            cand_payload = json.dumps(
                {
                    "server": trace.candidate_call.server,
                    "tool": trace.candidate_call.tool,
                    "arguments": trace.candidate_call.arguments,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            cand_findings = self.backend.analyze_yara(
                cand_payload,
                context={"tool_name": trace.candidate_call.tool, "content_type": "parameters"},
            )
            if cand_findings:
                all_findings.extend(cand_findings)
                if first_stage == "none" and any(
                    SEVERITY_RANK.get(f.severity.upper(), 1) >= SEVERITY_RANK["MEDIUM"]
                    for f in cand_findings
                ):
                    first_stage = "candidate_call"

        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        blocked_protective, blocked_hard, risk_score, reason = _summarize_findings(all_findings)
        blocked_stage = first_stage if blocked_protective else "none"

        return TraceEvalResult(
            trace_id=trace.trace_id,
            contender_id=self.contender_id,
            blocked_protective=blocked_protective,
            blocked_hard=blocked_hard,
            risk_score=risk_score,
            latency_ms=latency_ms,
            blocked_stage=blocked_stage,
            reason=reason,
            extra={"scan_mode": self.scan_mode, "finding_count": len(all_findings)},
        )

    def handle_scan_stdio_cmd(self, msg: Mapping[str, Any]) -> dict[str, Any]:
        """Handle Suite B `scan_stdio` IPC command against a live MCP stdio server."""
        command = str(msg["command"])
        args = [str(x) for x in msg.get("args", [])]
        env = dict(msg["env"]) if isinstance(msg.get("env"), Mapping) else None

        t0 = time.perf_counter_ns()
        findings = self.backend.scan_stdio_server(command, args, env=env)
        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        blocked_protective, blocked_hard, risk_score, reason = _summarize_findings(findings)
        return {
            "ok": True,
            "blocked_protective": blocked_protective,
            "blocked_hard": blocked_hard,
            "risk_score": risk_score,
            "latency_ms": latency_ms,
            "reason": reason,
            "finding_count": len(findings),
        }


def main(argv: Sequence[str] | None = None) -> int:
    proto_out = setup_pristine_ipc_stdout()
    try:
        parser = argparse.ArgumentParser(description="Cisco mcp-scanner Suite A/B worker")
        parser.add_argument("--contender-id", default="cisco-mcp-scanner-defs")
        parser.add_argument(
            "--scan-mode",
            choices=("definitions_only", "extended_payload"),
            default="definitions_only",
        )
        args = parser.parse_args(list(argv) if argv is not None else None)

        evaluator = CiscoScannerEvaluator(
            contender_id=args.contender_id,
            scan_mode=args.scan_mode,
        )
        return run_worker_loop(
            evaluator,
            args.contender_id,
            evaluator.cold_start_ms,
            proto_out=proto_out,
            extra_handlers={"scan_stdio": evaluator.handle_scan_stdio_cmd},
        )
    except Exception as exc:
        sys.stderr.write(f"[cisco_scanner_worker] Fatal startup error: {exc}\n")
        sys.stderr.flush()
        return 1
    finally:
        proto_out.close()


if __name__ == "__main__":
    raise SystemExit(main())
