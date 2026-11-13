"""Persistent ProtectAI LLM-Guard contender worker over JSONL stdin/stdout IPC."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import sys
import time
from typing import Any, Protocol

from benchmarks.contenders.protocol import TraceEvalResult
from benchmarks.contenders.stateless_policy_worker import (
    run_worker_loop,
    setup_pristine_ipc_stdout,
)
from benchmarks.traces.schema import MultiTurnTrace, ToolDefinition


class LlmGuardBackend(Protocol):
    """Backend interface for LLM-Guard scanning (injectable for fast unit tests)."""

    def scan_prompt_injection(self, text: str) -> tuple[bool, float]: ...
    def scan_secrets(self, text: str) -> tuple[bool, float]: ...
    def scan_pii(self, text: str) -> tuple[bool, float]: ...


class RealLlmGuardBackend:
    """Offline `llm-guard==0.3.16` backend (`PromptInjection` + `Secrets` + `Anonymize`)."""

    def __init__(self, *, threads: int = 4) -> None:
        import torch

        torch.set_num_threads(threads)

        from llm_guard.input_scanners import Anonymize, PromptInjection, Secrets
        from llm_guard.input_scanners.prompt_injection import MatchType
        from llm_guard.vault import Vault

        self._prompt_injection = PromptInjection(
            threshold=0.5,
            match_type=MatchType.FULL,
        )
        self._secrets = Secrets(redact_mode="all")
        self._vault = Vault()
        self._anonymize = Anonymize(self._vault, threshold=0.5)

    @staticmethod
    def _clamp_score(is_valid: bool, raw_score: float) -> float:
        val = float(raw_score)
        if val < 0.0:
            # llm-guard returns -1.0 when is_valid=True
            return 0.0 if is_valid else 0.5
        return max(0.0, min(1.0, val))

    def scan_prompt_injection(self, text: str) -> tuple[bool, float]:
        if not text.strip():
            return True, 0.0
        _sanitized, is_valid, raw_score = self._prompt_injection.scan(text)
        return bool(is_valid), self._clamp_score(bool(is_valid), raw_score)

    def scan_secrets(self, text: str) -> tuple[bool, float]:
        if not text.strip():
            return True, 0.0
        _sanitized, is_valid, raw_score = self._secrets.scan(text)
        return bool(is_valid), self._clamp_score(bool(is_valid), raw_score)

    def scan_pii(self, text: str) -> tuple[bool, float]:
        if not text.strip():
            return True, 0.0
        _sanitized, is_valid, raw_score = self._anonymize.scan(text)
        return bool(is_valid), self._clamp_score(bool(is_valid), raw_score)


class LlmGuardEvaluator:
    """Evaluates Suite A traces and Suite B wire messages using LLM-Guard scanners."""

    def __init__(
        self,
        contender_id: str = "protectai-llm-guard",
        *,
        threads: int = 4,
        backend: LlmGuardBackend | None = None,
    ) -> None:
        self.contender_id = contender_id
        t0 = time.perf_counter_ns()
        self.backend: LlmGuardBackend = (
            backend if backend is not None else RealLlmGuardBackend(threads=threads)
        )
        self.cold_start_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

    def scan_tools_list(
        self, tools: Sequence[ToolDefinition | Mapping[str, Any]]
    ) -> tuple[bool, float, str | None]:
        """Scan `tools/list` descriptions via `PromptInjection`."""
        max_score = 0.0
        for item in tools:
            if isinstance(item, ToolDefinition):
                name, desc = item.name, item.description
            else:
                name = str(item.get("name") or "")
                desc = str(item.get("description") or "")
            text = f"{name}\n{desc}".strip()
            if not text:
                continue
            is_valid, score = self.backend.scan_prompt_injection(text)
            norm_score = max(0.0, min(1.0, float(score)))
            max_score = max(max_score, norm_score)
            if not is_valid:
                return False, max_score, f"PromptInjection(tools_list:{name}, score={norm_score:.3f})"
        return True, max_score, None

    def scan_tool_output(self, output_text: str, source_name: str = "tool") -> tuple[bool, float, str | None]:
        """Scan untrusted MCP tool output via `PromptInjection` (`post_mcp_call`)."""
        if not output_text.strip():
            return True, 0.0, None
        is_valid, score = self.backend.scan_prompt_injection(output_text)
        norm_score = max(0.0, min(1.0, float(score)))
        if not is_valid:
            return False, norm_score, f"PromptInjection(output:{source_name}, score={norm_score:.3f})"
        return True, norm_score, None

    def scan_candidate_args(self, arguments: Mapping[str, Any]) -> tuple[bool, float, str | None]:
        """Scan outgoing `tools/call` arguments via `PromptInjection`, `Secrets`, and PII (`Anonymize`)."""
        serialized = json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True)
        if not serialized.strip() or serialized == "{}":
            return True, 0.0, None

        max_score = 0.0
        for scanner_name, fn in (
            ("PromptInjection", self.backend.scan_prompt_injection),
            ("Secrets", self.backend.scan_secrets),
            ("AnonymizePII", self.backend.scan_pii),
        ):
            is_valid, score = fn(serialized)
            norm_score = max(0.0, min(1.0, float(score)))
            max_score = max(max_score, norm_score)
            if not is_valid:
                return False, max_score, f"{scanner_name}(candidate_call, score={norm_score:.3f})"
        return True, max_score, None

    def evaluate_trace(self, trace: MultiTurnTrace) -> TraceEvalResult:
        """Evaluate a full `MultiTurnTrace` with pre/post MCP guardrail semantics."""
        t0 = time.perf_counter_ns()
        max_risk = 0.0
        blocked_stage = "none"
        reasons: list[str] = []

        # 1. tools/list description check
        for td in trace.tools_catalog:
            text = f"{td.name}\n{td.description}".strip()
            if not text:
                continue
            is_valid, score = self.backend.scan_prompt_injection(text)
            norm = max(0.0, min(1.0, float(score)))
            max_risk = max(max_risk, norm)
            if not is_valid:
                if blocked_stage == "none":
                    blocked_stage = "tools_list"
                reasons.append(f"PromptInjection(tools_list:{td.name}, score={norm:.3f})")

        # 2. prior_turns post_mcp_call check
        for turn in trace.prior_turns:
            if not turn.output_text.strip():
                continue
            is_valid, score = self.backend.scan_prompt_injection(turn.output_text)
            norm = max(0.0, min(1.0, float(score)))
            max_risk = max(max_risk, norm)
            if not is_valid:
                if blocked_stage == "none":
                    blocked_stage = "prior_output"
                reasons.append(f"PromptInjection(prior_output:{turn.tool}, score={norm:.3f})")

        # 3. candidate_call pre_mcp_call check
        ok_args, args_score, args_reason = self.scan_candidate_args(trace.candidate_call.arguments)
        max_risk = max(max_risk, args_score)
        if not ok_args and args_reason:
            if blocked_stage == "none":
                blocked_stage = "candidate_call"
            reasons.append(args_reason)

        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        blocked = blocked_stage != "none"
        return TraceEvalResult(
            trace_id=trace.trace_id,
            contender_id=self.contender_id,
            blocked_protective=blocked,
            blocked_hard=blocked,
            risk_score=max_risk,
            latency_ms=latency_ms,
            blocked_stage=blocked_stage,
            reason="; ".join(reasons) if reasons else "All LLM-Guard scanners passed",
        )


def main(argv: Sequence[str] | None = None) -> int:
    # Redirect stdout -> stderr BEFORE importing llm_guard/structlog so IPC stdout stays pristine.
    proto_out = setup_pristine_ipc_stdout()
    try:
        parser = argparse.ArgumentParser(description="ProtectAI LLM-Guard Suite A contender worker")
        parser.add_argument("--contender-id", default="protectai-llm-guard")
        parser.add_argument("--threads", type=int, default=4)
        args = parser.parse_args(list(argv) if argv is not None else None)

        evaluator = LlmGuardEvaluator(contender_id=args.contender_id, threads=args.threads)
        return run_worker_loop(
            evaluator,
            args.contender_id,
            evaluator.cold_start_ms,
            proto_out=proto_out,
        )
    except Exception as exc:
        sys.stderr.write(f"[llm_guard_worker] Fatal startup error: {exc}\n")
        sys.stderr.flush()
        return 1
    finally:
        proto_out.close()


if __name__ == "__main__":
    raise SystemExit(main())
