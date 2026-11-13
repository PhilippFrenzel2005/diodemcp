"""Frozen dataclass schemas for Suite A multi-turn traces and Suite B wire scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    """MCP tool catalog entry active in a benchmark session."""

    server: str
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.server or not self.name:
            raise ValueError("ToolDefinition requires non-empty server and name.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "server": self.server,
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolDefinition:
        """Deserialize from dict."""
        return cls(
            server=str(data["server"]),
            name=str(data["name"]),
            description=str(data.get("description", "")),
            input_schema=dict(data.get("input_schema") or {}),
        )


@dataclass(frozen=True)
class PriorTurn:
    """Completed tool call and output recorded earlier in a session."""

    server: str
    tool: str
    arguments: dict[str, Any]
    output_text: str
    is_error: bool = False

    def __post_init__(self) -> None:
        if not self.server or not self.tool:
            raise ValueError("PriorTurn requires non-empty server and tool.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "server": self.server,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "output_text": self.output_text,
            "is_error": bool(self.is_error),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PriorTurn:
        """Deserialize from dict."""
        return cls(
            server=str(data["server"]),
            tool=str(data["tool"]),
            arguments=dict(data.get("arguments") or {}),
            output_text=str(data.get("output_text", "")),
            is_error=bool(data.get("is_error", False)),
        )


@dataclass(frozen=True)
class CandidateAction:
    """Proposed final sink tool call evaluated by each benchmark contender."""

    server: str
    tool: str
    arguments: dict[str, Any]
    cwd: str = "/workspace/app"

    def __post_init__(self) -> None:
        if not self.server or not self.tool:
            raise ValueError("CandidateAction requires non-empty server and tool.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "server": self.server,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "cwd": self.cwd,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateAction:
        """Deserialize from dict."""
        return cls(
            server=str(data["server"]),
            tool=str(data["tool"]),
            arguments=dict(data.get("arguments") or {}),
            cwd=str(data.get("cwd", "/workspace/app")),
        )


@dataclass(frozen=True)
class MultiTurnTrace:
    """Complete deterministic Suite A replay trace."""

    trace_id: str
    split_group: str
    vector_id: int
    label: int
    user_intents: tuple[str, ...]
    tools_catalog: tuple[ToolDefinition, ...]
    prior_turns: tuple[PriorTurn, ...]
    candidate_call: CandidateAction
    initial_tools_catalog: tuple[ToolDefinition, ...] | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.trace_id or not self.split_group:
            raise ValueError("MultiTurnTrace requires non-empty trace_id and split_group.")
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label!r}")
        if not (1 <= self.vector_id <= 6):
            raise ValueError(f"vector_id must be in 1..6, got {self.vector_id!r}")
        if not self.user_intents:
            raise ValueError("MultiTurnTrace requires at least one user intent.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize trace to JSON-compatible dict."""
        return {
            "trace_id": self.trace_id,
            "split_group": self.split_group,
            "vector_id": self.vector_id,
            "label": self.label,
            "user_intents": list(self.user_intents),
            "tools_catalog": [td.to_dict() for td in self.tools_catalog],
            "prior_turns": [pt.to_dict() for pt in self.prior_turns],
            "candidate_call": self.candidate_call.to_dict(),
            "initial_tools_catalog": (
                [td.to_dict() for td in self.initial_tools_catalog]
                if self.initial_tools_catalog is not None
                else None
            ),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MultiTurnTrace:
        """Deserialize trace from dict."""
        raw_init = data.get("initial_tools_catalog")
        return cls(
            trace_id=str(data["trace_id"]),
            split_group=str(data["split_group"]),
            vector_id=int(data["vector_id"]),
            label=int(data["label"]),
            user_intents=tuple(str(x) for x in data["user_intents"]),
            tools_catalog=tuple(ToolDefinition.from_dict(x) for x in data.get("tools_catalog", ())),
            prior_turns=tuple(PriorTurn.from_dict(x) for x in data.get("prior_turns", ())),
            candidate_call=CandidateAction.from_dict(data["candidate_call"]),
            initial_tools_catalog=(
                tuple(ToolDefinition.from_dict(x) for x in raw_init)
                if raw_init is not None
                else None
            ),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class WireStep:
    """Single MCP JSON-RPC step executed over stdio in Suite B."""

    server: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    session_index: int = 1
    expect_blocked: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not self.server or not self.method:
            raise ValueError("WireStep requires non-empty server and method.")
        if self.session_index < 1:
            raise ValueError("session_index must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "server": self.server,
            "method": self.method,
            "params": dict(self.params),
            "session_index": self.session_index,
            "expect_blocked": bool(self.expect_blocked),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireStep:
        """Deserialize from dict."""
        return cls(
            server=str(data["server"]),
            method=str(data["method"]),
            params=dict(data.get("params") or {}),
            session_index=int(data.get("session_index", 1)),
            expect_blocked=bool(data.get("expect_blocked", False)),
            note=str(data.get("note", "")),
        )


@dataclass(frozen=True)
class WireScenario:
    """Live MCP stdio wire-protocol scenario for Suite B."""

    scenario_id: str
    category: str
    label: int
    servers: tuple[str, ...]
    steps: tuple[WireStep, ...]
    description: str

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.category:
            raise ValueError("WireScenario requires non-empty scenario_id and category.")
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label!r}")
        if not self.servers or not self.steps:
            raise ValueError("WireScenario requires non-empty servers and steps.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize scenario to JSON-compatible dict."""
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "label": self.label,
            "servers": list(self.servers),
            "steps": [s.to_dict() for s in self.steps],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireScenario:
        """Deserialize scenario from dict."""
        return cls(
            scenario_id=str(data["scenario_id"]),
            category=str(data["category"]),
            label=int(data["label"]),
            servers=tuple(str(x) for x in data["servers"]),
            steps=tuple(WireStep.from_dict(x) for x in data["steps"]),
            description=str(data.get("description", "")),
        )
