"""JSONL worker IPC protocol and evaluation result schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from typing import Any

VALID_BLOCKED_STAGES: frozenset[str] = frozenset(
    {"none", "tools_list", "prior_output", "candidate_call", "error"}
)


@dataclass(frozen=True)
class TraceEvalResult:
    """Per-trace evaluation result emitted by every contender worker."""

    trace_id: str
    contender_id: str
    blocked_protective: bool
    blocked_hard: bool
    risk_score: float
    latency_ms: float
    blocked_stage: str
    reason: str
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace_id or not self.contender_id:
            raise ValueError("TraceEvalResult requires non-empty trace_id and contender_id.")
        if self.blocked_stage not in VALID_BLOCKED_STAGES:
            raise ValueError(
                f"blocked_stage must be one of {sorted(VALID_BLOCKED_STAGES)}, "
                f"got {self.blocked_stage!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "trace_id": self.trace_id,
            "contender_id": self.contender_id,
            "blocked_protective": bool(self.blocked_protective),
            "blocked_hard": bool(self.blocked_hard),
            "risk_score": float(self.risk_score),
            "latency_ms": float(self.latency_ms),
            "blocked_stage": self.blocked_stage,
            "reason": self.reason,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TraceEvalResult:
        """Deserialize from dict."""
        return cls(
            trace_id=str(data["trace_id"]),
            contender_id=str(data["contender_id"]),
            blocked_protective=bool(data["blocked_protective"]),
            blocked_hard=bool(data["blocked_hard"]),
            risk_score=float(data["risk_score"]),
            latency_ms=float(data["latency_ms"]),
            blocked_stage=str(data["blocked_stage"]),
            reason=str(data.get("reason", "")),
            extra=dict(data.get("extra") or {}),
        )


def encode_ipc_message(payload: Mapping[str, Any]) -> str:
    """Encode a mapping as a single-line JSONL message without trailing newline."""
    return json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))


def decode_ipc_message(line: str) -> dict[str, Any]:
    """Decode one JSONL IPC line into a dictionary."""
    stripped = line.strip()
    if not stripped:
        raise ValueError("Empty IPC line.")
    decoded = json.loads(stripped)
    if not isinstance(decoded, dict):
        raise ValueError(f"IPC payload must be a JSON object, got {type(decoded).__name__}")
    return decoded
