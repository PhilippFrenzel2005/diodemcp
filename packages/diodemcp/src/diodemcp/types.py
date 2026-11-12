"""Core Pydantic schemas, bitflags, and risk band types for DiodeMCP."""

from __future__ import annotations

import time
from enum import IntFlag, StrEnum
from typing import Any

from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

from diodemcp.constants import CRITICAL_THRESHOLD, WARN_THRESHOLD


class CapabilityFlag(IntFlag):
    """Non-exclusive capability bitflags across T1-T7 threat surfaces."""

    LOCAL_SAFE = 0
    READS_PRIVATE = 1
    HIGH_SECRET = 2
    READS_UNTRUSTED = 4
    WRITES_EXTERNAL = 8
    MUTATES_SENSITIVE = 16

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        max_mask = (
            cls.READS_PRIVATE
            | cls.HIGH_SECRET
            | cls.READS_UNTRUSTED
            | cls.WRITES_EXTERNAL
            | cls.MUTATES_SENSITIVE
        ).value

        def _validate(value: Any) -> CapabilityFlag:
            if isinstance(value, cls):
                return value
            if isinstance(value, int) and not isinstance(value, bool):
                if 0 <= value <= max_mask:
                    return cls(value)
                raise ValueError(f"CapabilityFlag value {value} out of valid mask range 0..{max_mask}")
            raise ValueError(f"Expected CapabilityFlag or int, got {type(value).__name__}")

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                int,
                return_schema=core_schema.int_schema(),
                when_used="json",
            ),
        )


class RiskBand(StrEnum):
    """Calibrated 3-band routing levels."""

    SAFE = "safe"
    WARN = "warn"
    CRITICAL = "critical"


class TaintSpan(BaseModel):
    """A private, untrusted, or dual-trust content span recorded in the session ledger."""

    span_id: str
    source_label: str
    text: str
    flags: CapabilityFlag
    dual_trust: bool = False
    simhash: int | None = None
    timestamp: float = Field(default_factory=time.time)


class CandidateCall(BaseModel):
    """A proposed MCP or native IDE tool invocation awaiting firewall evaluation."""

    server: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class DecisionScores(BaseModel):
    """Calibrated probabilities and surgical attribution returned by a predictor."""

    p_enc: float = Field(ge=0.0, le=1.0)
    p_sem: float = Field(ge=0.0, le=1.0)
    p_hijack: float = Field(ge=0.0, le=1.0)
    attributed_span_idx: int | None = Field(default=None, ge=0)
    attributed_arg_key: str | None = None

    @property
    def p_max(self) -> float:
        return max(self.p_enc, self.p_sem, self.p_hijack)

    @property
    def band(self) -> RiskBand:
        score = self.p_max
        if score < WARN_THRESHOLD:
            return RiskBand.SAFE
        if score < CRITICAL_THRESHOLD:
            return RiskBand.WARN
        return RiskBand.CRITICAL
