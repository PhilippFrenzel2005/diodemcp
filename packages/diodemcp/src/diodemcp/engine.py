"""Persistent workspace taint engine, 3-band approval router, and sync/async IPC clients (`C4`, `C5`, `C6`, `C7`, `I13`)."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import sys
import threading
import time
from typing import Any, Literal
import urllib.parse

from filelock import FileLock, Timeout as FileLockTimeout

from diodemcp.constants import (
    CRITICAL_THRESHOLD,
    DEFAULT_IDLE_TIMEOUT_SECONDS,
    WARN_THRESHOLD,
)
from diodemcp.ledger import (
    MAX_OUTBOUND_BYTES_PER_DEST,
    MAX_OUTBOUND_CALLS_PER_DEST,
    SessionLedger,
    _reassemble_outbound_window,
    _utf8_truncate_tail,
    compute_workspace_key,
)
from diodemcp.ort_predictor import OrtJevPredictor
from diodemcp.packer import flatten_call_arguments
from diodemcp.predictor import (
    PredictorProtocol,
    Tier0DeterministicPredictor,
    _intent_explicitly_authorizes_copy,
)
from diodemcp.redactor import (
    build_redacted_audit_preview,
    redact_call_arguments,
)
from diodemcp.registry import CapabilityRegistry, _normalize_server_and_tool
from diodemcp.transport import (
    MAX_IPC_LINE_BYTES,
    PRIVILEGED_METHODS,
    TransportAuthError,
    TransportConfig,
    TransportError,
    TransportPermissionError,
    atomic_write_private_json,
    canonical_json_sha256,
    compute_admin_hmac,
    compute_client_hmac,
    compute_server_hmac,
    default_transport,
    ensure_admin_secret,
    ensure_private_dir,
    send_rpc_async,
    send_rpc_sync,
    unix_path_handle,
)
from diodemcp.types import CandidateCall, CapabilityFlag, DecisionScores, RiskBand, TaintSpan

DEFAULT_APPROVAL_TTL_SECONDS: float = 120.0
MAX_STORED_INCIDENTS: int = 256

_URL_EXTRACT_RE = re.compile(r"""https?://[^\s"'<>]+""", re.IGNORECASE)

_TARGET_PATH_KEYS: tuple[str, ...] = (
    "file_path",
    "filepath",
    "path",
    "target_file",
    "targetfile",
    "target_path",
    "absolute_path",
    "absolutepath",
    "notebook_path",
    "destination",
    "dst",
    "to_path",
    "uri",
    "filename",
    "file",
)

_WRITE_BODY_KEYS: frozenset[str] = frozenset(
    {
        "content",
        "contents",
        "code_content",
        "codecontent",
        "replacement_content",
        "replacementcontent",
        "file_text",
        "new_string",
        "new_str",
        "newtext",
        "new_source",
        "diff",
        "patch",
        "edits",
    }
)


def write_active_bundle_config(
    state_dir: Path | str,
    *,
    bundle_dir: Path | str | None = None,
    tier0_only: bool = False,
) -> Path:
    """Persist `<state_dir>/active_bundle.json` (`0600`) for daemon predictor initialization."""
    resolved_state = ensure_private_dir(Path(state_dir).expanduser())
    cfg_path = resolved_state / "active_bundle.json"
    resolved_bundle = (
        str(Path(bundle_dir).expanduser().resolve()) if bundle_dir is not None else None
    )
    atomic_write_private_json(
        cfg_path,
        {
            "bundle_dir": resolved_bundle,
            "tier0_only": bool(tier0_only),
            "updated_at": time.time(),
        },
    )
    return cfg_path


def _load_configured_predictor(
    state_dir: Path,
    explicit_bundle_dir: Path | str | None = None,
) -> tuple[PredictorProtocol, str]:
    """Load `OrtJevPredictor` when configured/present, else fall back to `Tier0DeterministicPredictor`."""
    if explicit_bundle_dir is not None:
        ort_pred = OrtJevPredictor(explicit_bundle_dir)
        return ort_pred, f"ort:{ort_pred.manifest.model_id}"

    active_cfg_file = state_dir / "active_bundle.json"
    if active_cfg_file.is_file() and not active_cfg_file.is_symlink():
        data = json.loads(active_cfg_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if data.get("tier0_only") is True:
                return Tier0DeterministicPredictor(), "tier0"
            cfg_bundle = data.get("bundle_dir")
            if isinstance(cfg_bundle, str) and cfg_bundle.strip():
                ort_pred = OrtJevPredictor(Path(cfg_bundle))
                return ort_pred, f"ort:{ort_pred.manifest.model_id}"

    default_current = state_dir / "models" / "current"
    if default_current.is_dir():
        ort_pred = OrtJevPredictor(default_current)
        return ort_pred, f"ort:{ort_pred.manifest.model_id}"

    return Tier0DeterministicPredictor(), "tier0"


def _extract_primary_path_arg(arguments: Mapping[str, Any]) -> str | None:
    for key in _TARGET_PATH_KEYS:
        val = arguments.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for k, val in arguments.items():
        if str(k).lower() in _TARGET_PATH_KEYS and isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _collect_text_leaves(node: Any, out: list[str]) -> None:
    if isinstance(node, str):
        if node:
            out.append(node)
    elif isinstance(node, Mapping):
        for v in node.values():
            _collect_text_leaves(v, out)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_text_leaves(item, out)


def _extract_write_body_text(arguments: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for k, v in arguments.items():
        if str(k).lower() in _WRITE_BODY_KEYS:
            _collect_text_leaves(v, parts)
    return "\n".join(parts)


def _extract_destination_key(call: CandidateCall) -> str:
    flat = flatten_call_arguments(call.arguments, max_args=None)
    for val in flat.values():
        if not val:
            continue
        m = _URL_EXTRACT_RE.search(val)
        if m is not None:
            parsed = urllib.parse.urlsplit(m.group(0))
            if parsed.hostname:
                return parsed.hostname.lower()
    norm_srv, norm_tool = _normalize_server_and_tool(call.server, call.tool)
    return norm_srv or norm_tool or "default"


def _canonical_args_sha256(arguments: Mapping[str, Any] | None) -> str:
    normalized = dict(arguments) if arguments else {}
    return canonical_json_sha256(normalized)


def _append_jsonl_0600(path: Path, entry: dict[str, Any]) -> None:
    ensure_private_dir(path.parent)
    if path.is_symlink():
        raise TransportAuthError(f"Refusing symlinked audit log: {path}")
    line_bytes = (
        json.dumps(entry, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
    ).encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        view = memoryview(line_bytes)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        os.fsync(fd)
        if os.name == "posix" and hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
    if os.name == "posix":
        os.chmod(path, 0o600)


class DiodeEngineServer:
    """Persistent workspace taint engine and split-credential approval router (`R4`, `R7`, `C4`-`C7`)."""

    def __init__(
        self,
        config: TransportConfig | None = None,
        *,
        predictor: PredictorProtocol | None = None,
        bundle_dir: Path | str | None = None,
        idle_timeout_s: float = float(DEFAULT_IDLE_TIMEOUT_SECONDS),
        approval_ttl_s: float = DEFAULT_APPROVAL_TTL_SECONDS,
    ) -> None:
        if idle_timeout_s <= 0.0:
            raise ValueError("idle_timeout_s must be > 0")
        if approval_ttl_s <= 0.0:
            raise ValueError("approval_ttl_s must be > 0")

        self.config: TransportConfig = config if config is not None else default_transport()
        ensure_private_dir(self.config.state_dir)
        self._sessions_dir: Path = ensure_private_dir(self.config.state_dir / "sessions")
        self._audit_path: Path = self.config.state_dir / "audit.jsonl"
        self._incidents_path: Path = self.config.state_dir / "incidents.json"

        self._admin_secret: str = ensure_admin_secret(self.config)
        self._client_token: str = secrets.token_hex(32)
        self._idle_timeout_s: float = float(idle_timeout_s)
        self._approval_ttl_s: float = float(approval_ttl_s)
        self._started_at: float = time.time()

        self._tier0 = Tier0DeterministicPredictor()
        if predictor is not None:
            self._predictor: PredictorProtocol = predictor
            if isinstance(predictor, OrtJevPredictor):
                self._predictor_label = f"ort:{predictor.manifest.model_id}"
            else:
                self._predictor_label = type(predictor).__name__
        else:
            self._predictor, self._predictor_label = _load_configured_predictor(
                self.config.state_dir, bundle_dir
            )

        self._state_lock = threading.RLock()
        self._ledgers: dict[str, SessionLedger] = {}
        self._registries: dict[tuple[str, str, int], CapabilityRegistry] = {}

        # Workspace-wide trusted shapes granted via admin_secret:
        #   set[(workspace_key, norm_server, norm_tool, frozenset[str])]
        self._admin_trusted_shapes: set[tuple[str, str, str, frozenset[str]]] = set()

        # Ephemeral shim-scoped trusted shapes granted via interactive elicitation popup:
        #   set[(workspace_key, session_id, norm_server, norm_tool, frozenset[str])]
        self._session_trusted_shapes: set[tuple[str, str, str, str, frozenset[str]]] = set()
        # Keep popup session_id strictly in memory (never written to incidents.json on disk)
        self._ephemeral_elicit_sessions: dict[str, str] = {}

        self._incidents: dict[str, dict[str, Any]] = {}
        self._incident_order: list[str] = []
        self._rehydrate_incidents()

        self._bound_socket_inode: int | None = None
        self._active_connections: int = 0
        self._last_activity: float = time.monotonic()
        self._shutdown_event: asyncio.Event | None = None

    @property
    def client_token(self) -> str:
        return self._client_token

    def _thresholds(self) -> tuple[float, float]:
        manifest = getattr(self._predictor, "manifest", None)
        if manifest is not None and hasattr(manifest, "thresholds"):
            th = manifest.thresholds
            return (
                float(th.get("warn_min", WARN_THRESHOLD)),
                float(th.get("critical_min", CRITICAL_THRESHOLD)),
            )
        return WARN_THRESHOLD, CRITICAL_THRESHOLD

    def _rehydrate_incidents(self) -> None:
        if not self._incidents_path.is_file() or self._incidents_path.is_symlink():
            return
        try:
            raw = self._incidents_path.read_text(encoding="utf-8")
            if not raw.strip():
                return
            data = json.loads(raw)
            if not isinstance(data, dict):
                return
            items = data.get("incidents", [])
            if not isinstance(items, list):
                return
            for item in items[-MAX_STORED_INCIDENTS:]:
                if isinstance(item, dict) and isinstance(item.get("incident_id"), str):
                    inc_id = item["incident_id"]
                    self._incidents[inc_id] = item
                    self._incident_order.append(inc_id)
        except Exception:
            self._incidents.clear()
            self._incident_order.clear()

    def _persist_incidents(self) -> None:
        while len(self._incident_order) > MAX_STORED_INCIDENTS:
            oldest = self._incident_order.pop(0)
            self._incidents.pop(oldest, None)
        payload = {
            "incidents": [
                self._incidents[iid]
                for iid in self._incident_order
                if iid in self._incidents
            ]
        }
        atomic_write_private_json(self._incidents_path, payload)

    def _resolve_workspace(self, workspace_root: Path | str | None) -> tuple[Path, str]:
        raw = os.fspath(workspace_root) if workspace_root is not None else os.getcwd()
        real_ws = Path(os.path.realpath(raw))
        ws_key = compute_workspace_key(real_ws)
        return real_ws, ws_key

    def _get_ledger(self, workspace_root: Path | str | None) -> tuple[Path, SessionLedger]:
        real_ws, ws_key = self._resolve_workspace(workspace_root)
        ledger = self._ledgers.get(ws_key)
        if ledger is None:
            persist_file = self._sessions_dir / f"{ws_key}.json"
            ledger = SessionLedger(workspace_root=real_ws, persist_path=persist_file)
            self._ledgers[ws_key] = ledger
        return real_ws, ledger

    def _get_registry(
        self,
        real_ws: Path,
        config_path: Path | str | None = None,
    ) -> CapabilityRegistry:
        eff_cfg = (
            Path(config_path).expanduser()
            if config_path is not None
            else (self.config.state_dir / "config.toml")
        )
        mtime_ns = 0
        if eff_cfg.is_file():
            try:
                mtime_ns = eff_cfg.stat().st_mtime_ns
            except OSError:
                mtime_ns = -1
        cache_key = (str(real_ws), str(eff_cfg), mtime_ns)
        reg = self._registries.get(cache_key)
        if reg is None:
            reg = CapabilityRegistry.from_default_and_user_config(
                config_path=eff_cfg,
                workspace_root=real_ws,
            )
            if len(self._registries) >= 32:
                self._registries.clear()
            self._registries[cache_key] = reg
        return reg

    def _commit_allowed_call_effects(
        self,
        registry: CapabilityRegistry,
        ledger: SessionLedger,
        req_flags: CapabilityFlag,
        call: CandidateCall,
    ) -> None:
        if bool(req_flags & CapabilityFlag.WRITES_EXTERNAL):
            dest_key = _extract_destination_key(call)
            ledger.record_outbound_call(dest_key, call)

        norm_srv, norm_tool = _normalize_server_and_tool(call.server, call.tool)
        if registry._is_file_write_call(norm_srv, norm_tool, call.arguments):
            target_path = _extract_primary_path_arg(call.arguments)
            written_body = _extract_write_body_text(call.arguments)
            if target_path and written_body:
                ledger.record_local_write(target_path, written_body)

    def _check_rolling_outbound_floor(
        self,
        ledger: SessionLedger,
        private_spans: list[TaintSpan],
        untrusted_spans: list[TaintSpan],
        intent: str | None,
        call: CandidateCall,
    ) -> DecisionScores | None:
        if not private_spans:
            return None
        dest_key = _extract_destination_key(call)
        prior_hist = ledger._outbound_history.get(dest_key)
        if not prior_hist:
            return None

        flat = flatten_call_arguments(call.arguments, max_args=None)
        if not flat:
            return None
        bounded_entry = {
            k: _utf8_truncate_tail(v, MAX_OUTBOUND_BYTES_PER_DEST) for k, v in flat.items()
        }
        preview_hist = [*prior_hist[-(MAX_OUTBOUND_CALLS_PER_DEST - 1) :], bounded_entry]
        reassembled = _reassemble_outbound_window(preview_hist, MAX_OUTBOUND_BYTES_PER_DEST)
        if not reassembled:
            return None

        primary_arg = next(iter(call.arguments.keys()), None)
        synth_key = primary_arg if primary_arg is not None else "_rolling_outbound"
        synth_call = CandidateCall(
            server=call.server,
            tool=call.tool,
            arguments={synth_key: reassembled},
        )
        return self._tier0.evaluate(
            intent=intent,
            private_spans=private_spans,
            untrusted_spans=untrusted_spans,
            call=synth_call,
        )

    def _check_tainted_local_write_floor(
        self,
        ledger: SessionLedger,
        private_spans: list[TaintSpan],
        intent: str | None,
        call: CandidateCall,
    ) -> DecisionScores | None:
        tainted_ids: set[str] = set()
        for id_set in ledger._tainted_paths.values():
            tainted_ids.update(id_set)

        matched: list[tuple[int, TaintSpan]] = []
        for idx, sp in enumerate(private_spans):
            if sp.span_id in tainted_ids or sp.source_label.startswith("staged_write:"):
                matched.append((idx, sp))

        if not matched:
            return None

        first_arg = next(iter(call.arguments.keys()), None)
        for idx, sp in matched:
            if bool(sp.flags & CapabilityFlag.HIGH_SECRET):
                return DecisionScores(
                    p_enc=0.95,
                    p_sem=0.0,
                    p_hijack=0.0,
                    attributed_span_idx=idx,
                    attributed_span_id=sp.span_id,
                    attributed_arg_key=first_arg,
                )

        idx0, sp0 = matched[0]
        clean_intent = (intent or "").strip()
        if clean_intent and _intent_explicitly_authorizes_copy(clean_intent, sp0):
            return DecisionScores(p_enc=0.08, p_sem=0.0, p_hijack=0.0)

        return DecisionScores(
            p_enc=0.88,
            p_sem=0.0,
            p_hijack=0.0,
            attributed_span_idx=idx0,
            attributed_span_id=sp0.span_id,
            attributed_arg_key=first_arg,
        )

    def _build_reason_summary(
        self,
        call: CandidateCall,
        scores: DecisionScores,
        leaked_span: TaintSpan | None,
        private_spans: Sequence[TaintSpan],
        *,
        is_self_tamper: bool,
    ) -> str:
        if is_self_tamper:
            return (
                f"Unconditional self-protection triggered on {call.server}.{call.tool} "
                f"(protected DiodeMCP or IDE configuration target)."
            )

        safe_src = "session taint"
        if leaked_span is not None and leaked_span.source_label:
            scrubbed_map = build_redacted_audit_preview(
                {"source": leaked_span.source_label},
                None,
                leaked_span,
                private_spans=private_spans,
                max_value_chars=80,
            )
            safe_src = scrubbed_map.get("source", "private_source")

        arg_note = (
            f" via argument '{scores.attributed_arg_key}'"
            if scores.attributed_arg_key
            else ""
        )
        if max(scores.p_enc, scores.p_sem) >= scores.p_hijack:
            kind = "encoded/verbatim leak" if scores.p_enc >= scores.p_sem else "semantic leak"
            return (
                f"Potential {kind} from '{safe_src}' into external call "
                f"{call.server}.{call.tool}{arg_note} (p_max={scores.p_max:.2f})."
            )
        return (
            f"Potential untrusted prompt-injection hijack on {call.server}.{call.tool}"
            f"{arg_note} (p_hijack={scores.p_hijack:.2f})."
        )

    def _write_audit_decision(
        self,
        *,
        decision: str,
        action: str,
        band: str,
        workspace_key: str,
        call: CandidateCall,
        scores: DecisionScores,
        leaked_span: TaintSpan | None,
        private_spans: Sequence[TaintSpan],
        client_kind: str,
        is_self_tamper: bool,
        incident_id: str | None = None,
    ) -> dict[str, str]:
        args_preview = build_redacted_audit_preview(
            call.arguments,
            scores.attributed_arg_key,
            leaked_span,
            private_spans=private_spans,
        )
        safe_source: str | None = None
        if leaked_span is not None and leaked_span.source_label:
            safe_source = build_redacted_audit_preview(
                {"source": leaked_span.source_label},
                None,
                leaked_span,
                private_spans=private_spans,
                max_value_chars=120,
            ).get("source")

        record: dict[str, Any] = {
            "timestamp": time.time(),
            "event": "evaluate_call",
            "decision": decision,
            "action": action,
            "band": band,
            "incident_id": incident_id,
            "workspace_key": workspace_key,
            "server": call.server,
            "tool": call.tool,
            "client_kind": client_kind,
            "p_enc": scores.p_enc,
            "p_sem": scores.p_sem,
            "p_hijack": scores.p_hijack,
            "p_max": scores.p_max,
            "attributed_span_id": scores.attributed_span_id,
            "attributed_source": safe_source,
            "attributed_arg_key": scores.attributed_arg_key,
            "is_self_tamper": is_self_tamper,
            "args_preview": args_preview,
        }
        _append_jsonl_0600(self._audit_path, record)
        return args_preview

    def _consume_matching_approval(
        self,
        workspace_key: str,
        server: str,
        tool: str,
        args_sha256: str,
    ) -> dict[str, Any] | None:
        now = time.time()
        for inc_id in reversed(self._incident_order):
            inc = self._incidents.get(inc_id)
            if inc is None:
                continue
            if (
                not inc.get("consumed", False)
                and inc.get("approved_until") is not None
                and now <= float(inc["approved_until"])
                and inc.get("workspace_key") == workspace_key
                and inc.get("server") == server
                and inc.get("tool") == tool
                and inc.get("canonical_args_sha256") == args_sha256
            ):
                inc["consumed"] = True
                inc["consumed_at"] = now
                self._persist_incidents()
                return inc
        return None

    def evaluate_call(
        self,
        workspace_root: Path | str | None,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
        *,
        client_supports_popup: bool = False,
        client_kind: Literal["mcp_shim", "ide_hook", "headless"] = "headless",
        permission_mode: str | None = None,
        config_path: Path | str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Run full 5-step taint/self-protection/predictor pipeline and 3-band approval router."""
        if client_kind not in ("mcp_shim", "ide_hook", "headless"):
            raise ValueError(f"Invalid client_kind: {client_kind!r}")
        if arguments is not None and not isinstance(arguments, Mapping):
            raise ValueError("arguments must be a dict/mapping or None")

        with self._state_lock:
            real_ws, ledger = self._get_ledger(workspace_root)
            registry = self._get_registry(real_ws, config_path)
            ws_key = ledger.workspace_key

            call = CandidateCall(
                server=server,
                tool=tool,
                arguments=dict(arguments or {}),
            )
            req_flags = registry.classify_request(server, tool, call.arguments)
            is_self_tamper = registry.is_unconditional_critical_self_tamper(
                server, tool, call.arguments
            )
            touches_tainted_write = ledger.touches_tainted_local_write(call)
            snap = ledger.snapshot()
            priv_list = list(snap.private_spans)
            untrust_list = list(snap.untrusted_spans)

            # Step 2: Single-use out-of-band human approval replay check (exact workspace + server + tool + args SHA256)
            args_sha = _canonical_args_sha256(call.arguments)
            matched_approval = self._consume_matching_approval(
                ws_key, server, tool, args_sha
            )
            if matched_approval is not None:
                approved_scores = DecisionScores.model_validate(
                    matched_approval.get(
                        "scores",
                        {"p_enc": 0.0, "p_sem": 0.0, "p_hijack": 0.0},
                    )
                )
                leaked_sp = (
                    priv_list[approved_scores.attributed_span_idx]
                    if approved_scores.attributed_span_idx is not None
                    and 0 <= approved_scores.attributed_span_idx < len(priv_list)
                    else None
                )
                self._commit_allowed_call_effects(registry, ledger, req_flags, call)
                self._write_audit_decision(
                    decision="ALLOW_APPROVED",
                    action="allow",
                    band=str(matched_approval.get("band", "critical")),
                    workspace_key=ws_key,
                    call=call,
                    scores=approved_scores,
                    leaked_span=leaked_sp,
                    private_spans=priv_list,
                    client_kind=client_kind,
                    is_self_tamper=is_self_tamper,
                    incident_id=matched_approval["incident_id"],
                )
                return {
                    "action": "allow",
                    "band": str(matched_approval.get("band", "safe")),
                    "p_max": approved_scores.p_max,
                    "scores": approved_scores.model_dump(mode="json"),
                    "server": server,
                    "tool": tool,
                    "workspace_key": ws_key,
                    "is_self_tamper": is_self_tamper,
                    "approved_incident_id": matched_approval["incident_id"],
                }

            warn_min, critical_min = self._thresholds()

            # Step 1: Unconditional CRITICAL self-protection (C4)
            if is_self_tamper:
                t0 = self._tier0.evaluate(
                    intent=snap.intent,
                    private_spans=priv_list,
                    untrusted_spans=untrust_list,
                    call=call,
                )
                first_arg = t0.attributed_arg_key or next(iter(call.arguments.keys()), None)
                scores = DecisionScores(
                    p_enc=t0.p_enc,
                    p_sem=t0.p_sem,
                    p_hijack=1.0,
                    attributed_span_idx=t0.attributed_span_idx,
                    attributed_span_id=t0.attributed_span_id,
                    attributed_arg_key=first_arg,
                )
            else:
                # Step 4a: Fast-path gate (<0.1ms) when call cannot exfiltrate or hijack
                writes_ext = bool(req_flags & CapabilityFlag.WRITES_EXTERNAL)
                has_untrusted_any = bool(untrust_list) or any(
                    s.dual_trust or bool(s.flags & CapabilityFlag.READS_UNTRUSTED)
                    for s in priv_list
                )
                can_fast_path = ledger.should_fast_path_allow(
                    req_flags,
                    is_self_tamper=False,
                    touches_tainted_local_write=touches_tainted_write,
                ) and not (writes_ext and has_untrusted_any)

                if can_fast_path:
                    fast_scores = DecisionScores(p_enc=0.0, p_sem=0.0, p_hijack=0.0)
                    self._commit_allowed_call_effects(registry, ledger, req_flags, call)
                    self._write_audit_decision(
                        decision="ALLOW_FAST_PATH",
                        action="allow",
                        band=RiskBand.SAFE.value,
                        workspace_key=ws_key,
                        call=call,
                        scores=fast_scores,
                        leaked_span=None,
                        private_spans=priv_list,
                        client_kind=client_kind,
                        is_self_tamper=False,
                    )
                    return {
                        "action": "allow",
                        "band": RiskBand.SAFE.value,
                        "p_max": 0.0,
                        "scores": fast_scores.model_dump(mode="json"),
                        "server": server,
                        "tool": tool,
                        "workspace_key": ws_key,
                        "is_self_tamper": False,
                    }

                # Step 4b: Full predictor evaluation + C7 rolling outbound + C7 tainted local write floors
                if isinstance(self._predictor, OrtJevPredictor):
                    scores = self._predictor.evaluate_snapshot(snap, call)
                else:
                    scores = self._predictor.evaluate(
                        intent=snap.intent,
                        private_spans=priv_list,
                        untrusted_spans=untrust_list,
                        call=call,
                    )

                if writes_ext:
                    rolling_floor = self._check_rolling_outbound_floor(
                        ledger, priv_list, untrust_list, snap.intent, call
                    )
                    if rolling_floor is not None and rolling_floor.p_enc > scores.p_enc:
                        scores = DecisionScores(
                            p_enc=rolling_floor.p_enc,
                            p_sem=scores.p_sem,
                            p_hijack=scores.p_hijack,
                            attributed_span_idx=rolling_floor.attributed_span_idx,
                            attributed_span_id=rolling_floor.attributed_span_id,
                            attributed_arg_key=rolling_floor.attributed_arg_key
                            or scores.attributed_arg_key,
                        )

                    if touches_tainted_write:
                        write_floor = self._check_tainted_local_write_floor(
                            ledger, priv_list, snap.intent, call
                        )
                        if write_floor is not None and write_floor.p_enc > scores.p_enc:
                            scores = DecisionScores(
                                p_enc=write_floor.p_enc,
                                p_sem=scores.p_sem,
                                p_hijack=scores.p_hijack,
                                attributed_span_idx=write_floor.attributed_span_idx,
                                attributed_span_id=write_floor.attributed_span_id,
                                attributed_arg_key=write_floor.attributed_arg_key
                                or scores.attributed_arg_key,
                            )

                if scores.p_hijack >= 0.98:
                    is_self_tamper = True

            p_max = scores.p_max
            leaked_span: TaintSpan | None = None
            if scores.attributed_span_idx is not None and 0 <= scores.attributed_span_idx < len(
                priv_list
            ):
                leaked_span = priv_list[scores.attributed_span_idx]

            has_high_secret = bool(
                leaked_span is not None and (leaked_span.flags & CapabilityFlag.HIGH_SECRET)
            )

            # Step 3: Session trust_tool_shape check (ONLY non-CRITICAL WARN band, non-self-tamper, non-HIGH_SECRET)
            flat_arg_keys = frozenset(
                flatten_call_arguments(call.arguments, max_args=None).keys()
            )
            norm_srv, norm_tool = _normalize_server_and_tool(server, tool)
            if (
                not is_self_tamper
                and not has_high_secret
                and warn_min <= p_max < critical_min
            ):
                is_admin_trusted = (
                    ws_key,
                    norm_srv,
                    norm_tool,
                    flat_arg_keys,
                ) in self._admin_trusted_shapes
                is_shim_trusted = bool(
                    client_supports_popup
                    and session_id
                    and (ws_key, session_id, norm_srv, norm_tool, flat_arg_keys)
                    in self._session_trusted_shapes
                )
                if is_admin_trusted or is_shim_trusted:
                    self._commit_allowed_call_effects(registry, ledger, req_flags, call)
                    self._write_audit_decision(
                        decision="ALLOW_TRUSTED_SHAPE",
                        action="allow",
                        band=RiskBand.WARN.value,
                        workspace_key=ws_key,
                        call=call,
                        scores=scores,
                        leaked_span=leaked_span,
                        private_spans=priv_list,
                        client_kind=client_kind,
                        is_self_tamper=False,
                    )
                    return {
                        "action": "allow",
                        "band": RiskBand.WARN.value,
                        "p_max": p_max,
                        "scores": scores.model_dump(mode="json"),
                        "server": server,
                        "tool": tool,
                        "workspace_key": ws_key,
                        "is_self_tamper": False,
                        "trusted_shape": True,
                    }

            # Step 5: 3-Band Routing (SAFE / WARN / CRITICAL)
            if p_max < warn_min and not is_self_tamper:
                self._commit_allowed_call_effects(registry, ledger, req_flags, call)
                self._write_audit_decision(
                    decision="ALLOW",
                    action="allow",
                    band=RiskBand.SAFE.value,
                    workspace_key=ws_key,
                    call=call,
                    scores=scores,
                    leaked_span=leaked_span,
                    private_spans=priv_list,
                    client_kind=client_kind,
                    is_self_tamper=False,
                )
                return {
                    "action": "allow",
                    "band": RiskBand.SAFE.value,
                    "p_max": p_max,
                    "scores": scores.model_dump(mode="json"),
                    "server": server,
                    "tool": tool,
                    "workspace_key": ws_key,
                    "is_self_tamper": False,
                }

            band_str = (
                RiskBand.CRITICAL.value
                if (p_max >= critical_min or is_self_tamper)
                else RiskBand.WARN.value
            )
            incident_id = f"inc_{secrets.token_hex(12)}"
            redacted_args = redact_call_arguments(
                call,
                scores.attributed_arg_key,
                leaked_span,
                private_spans=priv_list,
            )
            can_redact = (redacted_args != call.arguments) and not is_self_tamper
            reason = self._build_reason_summary(
                call,
                scores,
                leaked_span,
                priv_list,
                is_self_tamper=is_self_tamper,
            )
            block_message = (
                f"Blocked by DiodeMCP (incident {incident_id}). "
                f"Review in an interactive terminal via: diodemcp audit"
            )

            bypass_ide_prompt = permission_mode in ("bypassPermissions", "dontAsk")

            if band_str == RiskBand.WARN.value:
                if client_supports_popup:
                    action_out = "elicit"
                    decision_tag = "ELICIT"
                    choices = (
                        ["allow_once", "redact", "block", "trust_shape"]
                        if can_redact
                        else ["allow_once", "block", "trust_shape"]
                    )
                    default_choice = "redact" if (p_max < 0.50 and can_redact) else "block"
                    banner = f"⚠️ DiodeMCP WARN ({p_max:.2f}): {reason}"
                elif client_kind == "ide_hook" and not bypass_ide_prompt:
                    action_out = "ask"
                    decision_tag = "ASK"
                    choices = []
                    default_choice = "block"
                    banner = f"⚠️ DiodeMCP WARN ({p_max:.2f}): {reason}"
                else:
                    action_out = "block"
                    decision_tag = "BLOCK"
                    choices = []
                    default_choice = "block"
                    banner = f"⚠️ DiodeMCP WARN ({p_max:.2f}): {reason}"
            else:
                if client_supports_popup:
                    action_out = "elicit"
                    decision_tag = "ELICIT"
                    choices = (
                        ["block", "redact", "allow_once"]
                        if can_redact
                        else ["block", "allow_once"]
                    )
                    default_choice = "block"
                    banner = f"🛑 CRITICAL DiodeMCP Alert ({p_max:.2f}): {reason}"
                else:
                    action_out = "block"
                    decision_tag = "BLOCK"
                    choices = []
                    default_choice = "block"
                    banner = f"🛑 CRITICAL DiodeMCP Alert ({p_max:.2f}): {reason}"

            args_preview = self._write_audit_decision(
                decision=decision_tag,
                action=action_out,
                band=band_str,
                workspace_key=ws_key,
                call=call,
                scores=scores,
                leaked_span=leaked_span,
                private_spans=priv_list,
                client_kind=client_kind,
                is_self_tamper=is_self_tamper,
                incident_id=incident_id,
            )

            inc_record: dict[str, Any] = {
                "incident_id": incident_id,
                "created_at": time.time(),
                "workspace_key": ws_key,
                "server": server,
                "tool": tool,
                "canonical_args_sha256": args_sha,
                "flat_arg_keys": sorted(flat_arg_keys),
                "band": band_str,
                "action_emitted": action_out,
                "p_max": round(p_max, 6),
                "scores": scores.model_dump(mode="json"),
                "is_self_tamper": is_self_tamper,
                "has_high_secret": has_high_secret,
                "client_supports_popup": bool(client_supports_popup),
                "reason": reason,
                "args_preview": args_preview,
                "approved_until": None,
                "consumed": False,
                "elicit_resolved_choice": None,
            }
            if client_supports_popup and isinstance(session_id, str) and session_id:
                self._ephemeral_elicit_sessions[incident_id] = session_id
                while len(self._ephemeral_elicit_sessions) > MAX_STORED_INCIDENTS:
                    oldest_sid = next(iter(self._ephemeral_elicit_sessions))
                    self._ephemeral_elicit_sessions.pop(oldest_sid, None)
            self._incidents[incident_id] = inc_record
            self._incident_order.append(incident_id)
            self._persist_incidents()

            response: dict[str, Any] = {
                "action": action_out,
                "band": band_str,
                "incident_id": incident_id,
                "p_max": p_max,
                "scores": scores.model_dump(mode="json"),
                "server": server,
                "tool": tool,
                "workspace_key": ws_key,
                "is_self_tamper": is_self_tamper,
                "reason": reason,
                "banner": banner,
                "message": block_message,
            }
            if action_out == "elicit":
                response["choices"] = choices
                response["default_choice"] = default_choice
                response["redacted_arguments"] = redacted_args
                response["can_redact"] = can_redact

            return response

    def record_output(
        self,
        workspace_root: Path | str | None,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None,
        output_text: str,
        *,
        config_path: Path | str | None = None,
        extra_flags: int | CapabilityFlag = CapabilityFlag.LOCAL_SAFE,
    ) -> dict[str, Any]:
        """Classify tool output via `CapabilityRegistry` and persist into workspace `SessionLedger` (`I6`)."""
        if arguments is not None and not isinstance(arguments, Mapping):
            raise ValueError("arguments must be a dict/mapping or None")

        with self._state_lock:
            real_ws, ledger = self._get_ledger(workspace_root)
            registry = self._get_registry(real_ws, config_path)
            args_dict = dict(arguments or {})

            resp_flags = registry.classify_response(
                server, tool, args_dict, output_text=output_text
            ) | CapabilityFlag(int(extra_flags))

            target_path = _extract_primary_path_arg(args_dict)
            if target_path:
                source_label = target_path
            elif isinstance(args_dict.get("url"), str) and args_dict["url"].strip():
                source_label = f"{server}:{tool}:{args_dict['url'].strip()[:120]}"
            else:
                source_label = f"{server}:{tool}"

            is_dual = bool(resp_flags & CapabilityFlag.READS_PRIVATE) and bool(
                resp_flags & CapabilityFlag.READS_UNTRUSTED
            )
            span = ledger.record_tool_output(
                source_label=source_label,
                text=output_text,
                flags=resp_flags,
                dual_trust=is_dual,
                target_path=target_path,
            )

            norm_srv, norm_tool = _normalize_server_and_tool(server, tool)
            if registry._is_file_write_call(norm_srv, norm_tool, args_dict):
                written_body = _extract_write_body_text(args_dict)
                if target_path and written_body:
                    ledger.record_local_write(target_path, written_body)

            return {
                "ok": True,
                "recorded": span is not None,
                "span_id": span.span_id if span is not None else None,
                "flags": int(resp_flags),
                "dual_trust": span.dual_trust if span is not None else False,
                "workspace_key": ledger.workspace_key,
            }

    def record_intent(
        self,
        workspace_root: Path | str | None,
        intent: str | None,
    ) -> dict[str, Any]:
        """Record current user prompt intent on the persistent workspace ledger."""
        with self._state_lock:
            _real_ws, ledger = self._get_ledger(workspace_root)
            ledger.record_intent(intent)
            return {
                "ok": True,
                "workspace_key": ledger.workspace_key,
                "has_intent": ledger.snapshot().intent is not None,
            }

    def record_local_write(
        self,
        workspace_root: Path | str | None,
        target_path: str,
        written_content: str,
    ) -> dict[str, Any]:
        """Propagate workspace private taint to `target_path` on local file edit/write (`C7`)."""
        with self._state_lock:
            _real_ws, ledger = self._get_ledger(workspace_root)
            span = ledger.record_local_write(target_path, written_content)
            return {
                "ok": True,
                "tainted": span is not None,
                "span_id": span.span_id if span is not None else None,
                "workspace_key": ledger.workspace_key,
            }

    def resolve_elicit(
        self,
        incident_id: str,
        choice: Literal["allow_once", "redact", "block", "trust_shape"],
        *,
        session_id: str | None = None,
        forwarded_arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record interactive MCP `elicitation/create` user decision (`shim.py`, scoped to shim `session_id`)."""
        if choice not in ("allow_once", "redact", "block", "trust_shape"):
            raise ValueError(f"Unsupported elicit choice: {choice!r}")

        with self._state_lock:
            inc = self._incidents.get(incident_id)
            if inc is None:
                raise ValueError(f"Unknown incident_id: {incident_id!r}")
            if not inc.get("client_supports_popup", False):
                raise ValueError(
                    f"Incident {incident_id!r} did not originate from an interactive popup"
                )
            if inc.get("elicit_resolved_choice") is not None:
                raise ValueError(f"Incident {incident_id!r} was already resolved")

            orig_session = self._ephemeral_elicit_sessions.get(incident_id)
            eff_session = session_id or orig_session
            if orig_session and session_id and orig_session != session_id:
                raise ValueError("Popup resolution session_id mismatch")

            if choice == "trust_shape":
                if (
                    inc.get("band") != RiskBand.WARN.value
                    or inc.get("is_self_tamper", False)
                    or inc.get("has_high_secret", False)
                ):
                    raise ValueError(
                        "trust_shape is strictly restricted to non-CRITICAL WARN incidents"
                    )
                if not isinstance(eff_session, str) or not eff_session:
                    raise ValueError("trust_shape popup resolution requires shim session_id")
                norm_srv, norm_tool = _normalize_server_and_tool(
                    str(inc["server"]), str(inc["tool"])
                )
                shape_keys = frozenset(str(k) for k in inc.get("flat_arg_keys", []))
                self._session_trusted_shapes.add(
                    (
                        str(inc["workspace_key"]),
                        eff_session,
                        norm_srv,
                        norm_tool,
                        shape_keys,
                    )
                )

            if choice in ("allow_once", "trust_shape", "redact") and isinstance(
                forwarded_arguments, dict
            ):
                ws_key = str(inc["workspace_key"])
                ledger = self._ledgers.get(ws_key)
                if ledger is not None:
                    call = CandidateCall(
                        server=str(inc["server"]),
                        tool=str(inc["tool"]),
                        arguments=forwarded_arguments,
                    )
                    dest_key = _extract_destination_key(call)
                    ledger.record_outbound_call(dest_key, call)

            inc["elicit_resolved_choice"] = choice
            inc["consumed"] = True
            inc["resolved_at"] = time.time()
            self._persist_incidents()

            _append_jsonl_0600(
                self._audit_path,
                {
                    "timestamp": time.time(),
                    "event": "resolve_elicit",
                    "incident_id": incident_id,
                    "choice": choice,
                    "workspace_key": inc.get("workspace_key"),
                    "server": inc.get("server"),
                    "tool": inc.get("tool"),
                    "band": inc.get("band"),
                },
            )
            return {
                "ok": True,
                "incident_id": incident_id,
                "choice": choice,
            }

    def _require_admin_secret_direct(self, admin_secret: str | None, method: str) -> None:
        if not isinstance(admin_secret, str) or not hmac.compare_digest(
            admin_secret.encode("utf-8"),
            self._admin_secret.encode("utf-8"),
        ):
            raise TransportPermissionError(
                f"Privileged operation {method!r} requires valid admin_secret"
            )

    def approve_incident(
        self,
        incident_id: str,
        *,
        admin_secret: str | None = None,
        _admin_verified: bool = False,
    ) -> dict[str, Any]:
        """Authorize exact `(workspace_key, server, tool, canonical_args_sha256)` single-use replay for 120s (`C4`)."""
        if not _admin_verified:
            self._require_admin_secret_direct(admin_secret, "approve_incident")

        with self._state_lock:
            inc = self._incidents.get(incident_id)
            if inc is None:
                raise ValueError(f"Unknown incident_id: {incident_id!r}")
            if inc.get("consumed", False):
                raise ValueError(f"Incident {incident_id!r} was already consumed")

            expires_at = time.time() + self._approval_ttl_s
            inc["approved_until"] = expires_at
            self._persist_incidents()

            _append_jsonl_0600(
                self._audit_path,
                {
                    "timestamp": time.time(),
                    "event": "approve_incident",
                    "incident_id": incident_id,
                    "workspace_key": inc.get("workspace_key"),
                    "server": inc.get("server"),
                    "tool": inc.get("tool"),
                    "expires_at": expires_at,
                },
            )
            return {
                "ok": True,
                "incident_id": incident_id,
                "server": inc.get("server"),
                "tool": inc.get("tool"),
                "expires_at": expires_at,
            }

    def trust_tool_shape(
        self,
        workspace_root: Path | str | None,
        server: str,
        tool: str,
        arg_keys: Iterable[str],
        *,
        admin_secret: str | None = None,
        _admin_verified: bool = False,
    ) -> dict[str, Any]:
        """Grant workspace-scoped WARN-band shape trust (strictly requires `admin_secret`, never bypasses CRITICAL)."""
        if not _admin_verified:
            self._require_admin_secret_direct(admin_secret, "trust_tool_shape")

        with self._state_lock:
            real_ws, ws_key = self._resolve_workspace(workspace_root)
            registry = self._get_registry(real_ws)
            key_set = frozenset(str(k) for k in arg_keys)
            probe_args = {k: "" for k in key_set}
            if registry.is_unconditional_critical_self_tamper(server, tool, probe_args):
                raise ValueError("Refusing to trust self-protection critical target")

            norm_srv, norm_tool = _normalize_server_and_tool(server, tool)
            self._admin_trusted_shapes.add((ws_key, norm_srv, norm_tool, key_set))

            _append_jsonl_0600(
                self._audit_path,
                {
                    "timestamp": time.time(),
                    "event": "trust_tool_shape",
                    "workspace_key": ws_key,
                    "server": server,
                    "tool": tool,
                    "arg_keys": sorted(key_set),
                },
            )
            return {
                "ok": True,
                "workspace_key": ws_key,
                "server": server,
                "tool": tool,
                "arg_keys": sorted(key_set),
            }

    def list_incidents(self, *, limit: int = 50) -> dict[str, Any]:
        """Return bounded list of recent redacted incidents (zero raw secrets, zero approvable tokens)."""
        bounded = max(1, min(int(limit), MAX_STORED_INCIDENTS))
        with self._state_lock:
            out: list[dict[str, Any]] = []
            for inc_id in reversed(self._incident_order):
                inc = self._incidents.get(inc_id)
                if inc is None:
                    continue
                out.append(
                    {
                        "incident_id": inc["incident_id"],
                        "created_at": inc.get("created_at"),
                        "workspace_key": inc.get("workspace_key"),
                        "server": inc.get("server"),
                        "tool": inc.get("tool"),
                        "band": inc.get("band"),
                        "action_emitted": inc.get("action_emitted"),
                        "p_max": inc.get("p_max"),
                        "reason": inc.get("reason"),
                        "args_preview": inc.get("args_preview", {}),
                        "consumed": bool(inc.get("consumed", False)),
                        "approved_until": inc.get("approved_until"),
                    }
                )
                if len(out) >= bounded:
                    break
            return {"ok": True, "incidents": out}

    def status(self) -> dict[str, Any]:
        """Return daemon health, transport mode, active predictor, and session counts."""
        with self._state_lock:
            return {
                "ok": True,
                "pid": os.getpid(),
                "mode": self.config.mode,
                "predictor": self._predictor_label,
                "uptime_s": round(max(0.0, time.time() - self._started_at), 3),
                "idle_timeout_s": self._idle_timeout_s,
                "active_workspaces": len(self._ledgers),
                "incident_count": len(self._incidents),
            }

    def _dispatch_rpc(
        self,
        method: str,
        params: dict[str, Any],
        *,
        admin_verified: bool,
    ) -> dict[str, Any]:
        if method == "ping":
            return {"pong": True, "pid": os.getpid(), "mode": self.config.mode}
        if method == "status":
            return self.status()
        if method == "evaluate_call":
            return self.evaluate_call(
                workspace_root=params.get("workspace_root"),
                server=str(params["server"]),
                tool=str(params["tool"]),
                arguments=params.get("arguments"),
                client_supports_popup=bool(params.get("client_supports_popup", False)),
                client_kind=params.get("client_kind", "headless"),
                permission_mode=params.get("permission_mode"),
                config_path=params.get("config_path"),
                session_id=params.get("session_id"),
            )
        if method == "record_output":
            return self.record_output(
                workspace_root=params.get("workspace_root"),
                server=str(params["server"]),
                tool=str(params["tool"]),
                arguments=params.get("arguments"),
                output_text=str(params.get("output_text", "")),
                config_path=params.get("config_path"),
                extra_flags=int(params.get("extra_flags", 0)),
            )
        if method == "record_intent":
            return self.record_intent(
                workspace_root=params.get("workspace_root"),
                intent=params.get("intent"),
            )
        if method == "record_local_write":
            return self.record_local_write(
                workspace_root=params.get("workspace_root"),
                target_path=str(params["target_path"]),
                written_content=str(params.get("written_content", "")),
            )
        if method == "resolve_elicit":
            return self.resolve_elicit(
                incident_id=str(params["incident_id"]),
                choice=params["choice"],
                session_id=params.get("session_id"),
                forwarded_arguments=params.get("forwarded_arguments"),
            )
        if method == "list_incidents":
            raw_limit = params.get("limit", 50)
            if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
                raise ValueError("limit must be an integer")
            return self.list_incidents(limit=raw_limit)
        if method in ("approve_incident", "approve_allow_token"):
            if not admin_verified:
                raise TransportPermissionError(f"Privileged RPC {method!r} requires admin_secret")
            inc_id = str(params.get("incident_id") or params.get("allow_token") or "")
            return self.approve_incident(inc_id, _admin_verified=True)
        if method == "trust_tool_shape":
            if not admin_verified:
                raise TransportPermissionError("Privileged RPC 'trust_tool_shape' requires admin_secret")
            raw_keys = params.get("arg_keys", [])
            if not isinstance(raw_keys, list):
                raise ValueError("arg_keys must be a list of strings")
            return self.trust_tool_shape(
                workspace_root=params.get("workspace_root"),
                server=str(params["server"]),
                tool=str(params["tool"]),
                arg_keys=[str(k) for k in raw_keys],
                _admin_verified=True,
            )
        if method == "shutdown":
            if not admin_verified:
                raise TransportPermissionError("Privileged RPC 'shutdown' requires admin_secret")
            if self._shutdown_event is not None:
                self._shutdown_event.set()
            return {"ok": True, "pid": os.getpid(), "shutting_down": True}

        raise ValueError(f"Unknown RPC method: {method!r}")

    def _process_raw_frame(self, raw_line: bytes) -> tuple[bytes, bool]:
        """Validate HMACs, dispatch RPC, and return `(response_line_bytes, close_after)`."""
        try:
            req = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            err = {
                "ok": False,
                "request_id": "",
                "code": "UNAUTHORIZED",
                "error": "Malformed JSON request",
            }
            return (json.dumps(err) + "\n").encode("utf-8"), True

        if not isinstance(req, dict):
            err = {
                "ok": False,
                "request_id": "",
                "code": "UNAUTHORIZED",
                "error": "Request root must be an object",
            }
            return (json.dumps(err) + "\n").encode("utf-8"), True

        req_id = req.get("request_id")
        nonce = req.get("client_nonce")
        method = req.get("method")
        params = req.get("params")
        client_mac = req.get("client_hmac")

        if (
            not isinstance(req_id, str)
            or not req_id
            or len(req_id) > 128
            or not isinstance(nonce, str)
            or not nonce
            or len(nonce) > 128
            or not isinstance(method, str)
            or not method
            or not isinstance(params, dict)
            or not isinstance(client_mac, str)
        ):
            err = {
                "ok": False,
                "request_id": req_id if isinstance(req_id, str) else "",
                "code": "UNAUTHORIZED",
                "error": "Missing or invalid client_token authentication fields",
            }
            return (json.dumps(err) + "\n").encode("utf-8"), True

        params_sha = canonical_json_sha256(params)
        expected_client_mac = compute_client_hmac(
            self._client_token, nonce, req_id, method, params_sha
        )
        if not hmac.compare_digest(client_mac, expected_client_mac):
            err = {
                "ok": False,
                "request_id": req_id,
                "code": "UNAUTHORIZED",
                "error": "Invalid client_token HMAC",
            }
            return (json.dumps(err) + "\n").encode("utf-8"), True

        admin_verified = False
        if method in PRIVILEGED_METHODS:
            admin_mac = req.get("admin_hmac")
            expected_admin_mac = compute_admin_hmac(
                self._admin_secret, nonce, req_id, method, params_sha
            )
            if isinstance(admin_mac, str) and hmac.compare_digest(
                admin_mac, expected_admin_mac
            ):
                admin_verified = True
            else:
                err_body = {
                    "code": "FORBIDDEN",
                    "error": f"Privileged RPC {method!r} requires valid admin_secret",
                }
                srv_mac = compute_server_hmac(
                    self._client_token,
                    nonce,
                    req_id,
                    False,
                    canonical_json_sha256(err_body),
                )
                out_err = {
                    "ok": False,
                    "request_id": req_id,
                    **err_body,
                    "server_hmac": srv_mac,
                }
                return (
                    json.dumps(out_err, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8"), False

        try:
            result = self._dispatch_rpc(method, params, admin_verified=admin_verified)
        except TransportPermissionError as exc:
            err_body = {"code": "FORBIDDEN", "error": str(exc)}
            srv_mac = compute_server_hmac(
                self._client_token,
                nonce,
                req_id,
                False,
                canonical_json_sha256(err_body),
            )
            out_err = {
                "ok": False,
                "request_id": req_id,
                **err_body,
                "server_hmac": srv_mac,
            }
            return (
                json.dumps(out_err, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8"), False
        except Exception as exc:
            err_body = {"code": "BAD_REQUEST", "error": str(exc)}
            srv_mac = compute_server_hmac(
                self._client_token,
                nonce,
                req_id,
                False,
                canonical_json_sha256(err_body),
            )
            out_err = {
                "ok": False,
                "request_id": req_id,
                **err_body,
                "server_hmac": srv_mac,
            }
            return (
                json.dumps(out_err, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8"), False

        srv_mac = compute_server_hmac(
            self._client_token,
            nonce,
            req_id,
            True,
            canonical_json_sha256(result),
        )
        out_ok = {
            "ok": True,
            "request_id": req_id,
            "result": result,
            "server_hmac": srv_mac,
        }
        close_after = method == "shutdown"
        return (
            json.dumps(out_ok, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
        ).encode("utf-8"), close_after

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._active_connections += 1
        self._last_activity = time.monotonic()
        try:
            while not reader.at_eof():
                try:
                    raw_line = await asyncio.wait_for(reader.readline(), timeout=30.0)
                except (asyncio.TimeoutError, TimeoutError, asyncio.LimitOverrunError):
                    break
                if not raw_line:
                    break

                self._last_activity = time.monotonic()
                resp_bytes, close_after = self._process_raw_frame(raw_line)
                writer.write(resp_bytes)
                await writer.drain()
                self._last_activity = time.monotonic()
                if close_after:
                    break
        except (ConnectionError, OSError):
            pass
        finally:
            self._active_connections -= 1
            self._last_activity = time.monotonic()
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _unlink_owned_endpoints(self) -> None:
        owned_addr = False
        if self.config.addr_file.exists() and not self.config.addr_file.is_symlink():
            try:
                cur = json.loads(self.config.addr_file.read_text(encoding="utf-8"))
                if isinstance(cur, dict) and cur.get("client_token") == self._client_token:
                    owned_addr = True
                    self.config.addr_file.unlink(missing_ok=True)
            except Exception:
                pass
        if self.config.mode == "unix" and self.config.socket_path.exists():
            try:
                st = self.config.socket_path.lstat()
                if (
                    self._bound_socket_inode is not None
                    and st.st_ino == self._bound_socket_inode
                ) or owned_addr:
                    self.config.socket_path.unlink(missing_ok=True)
            except OSError:
                pass

    async def serve_forever(self) -> None:
        """Bind Unix (`0600`) or loopback TCP (`127.0.0.1:0`) listener and run lock-safe idle monitor."""
        self._shutdown_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        if threading.current_thread() is threading.main_thread():
            for sig_name in ("SIGINT", "SIGTERM"):
                sig = getattr(signal, sig_name, None)
                if sig is not None:
                    try:
                        loop.add_signal_handler(sig, self._shutdown_event.set)
                    except (NotImplementedError, RuntimeError):
                        pass

        server: asyncio.AbstractServer
        if self.config.mode == "unix":
            if not hasattr(socket, "AF_UNIX"):
                raise TransportError("AF_UNIX sockets are not supported on this platform")
            ensure_private_dir(self.config.socket_path.parent)
            try:
                self.config.socket_path.unlink(missing_ok=True)
            except OSError:
                pass

            unix_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                with unix_path_handle(self.config.socket_path) as bind_target:
                    unix_sock.bind(bind_target)
                if os.name == "posix":
                    os.chmod(self.config.socket_path, 0o600)
                self._bound_socket_inode = self.config.socket_path.lstat().st_ino
                unix_sock.setblocking(False)
                server = await asyncio.start_unix_server(
                    self._handle_client,
                    sock=unix_sock,
                    limit=MAX_IPC_LINE_BYTES,
                )
            except BaseException:
                unix_sock.close()
                raise
            addr_payload: dict[str, Any] = {
                "pid": os.getpid(),
                "mode": "unix",
                "host": None,
                "port": None,
                "socket_path": str(self.config.socket_path),
                "client_token": self._client_token,
                "started_at": self._started_at,
                "protocol_version": 1,
            }
        else:
            server = await asyncio.start_server(
                self._handle_client,
                host="127.0.0.1",
                port=0,
                limit=MAX_IPC_LINE_BYTES,
            )
            bound_port = int(server.sockets[0].getsockname()[1])
            addr_payload = {
                "pid": os.getpid(),
                "mode": "tcp",
                "host": "127.0.0.1",
                "port": bound_port,
                "socket_path": None,
                "client_token": self._client_token,
                "started_at": self._started_at,
                "protocol_version": 1,
            }

        self._last_activity = time.monotonic()
        atomic_write_private_json(self.config.addr_file, addr_payload)

        cleaned_under_lock = False
        poll_interval = min(0.05, max(0.01, self._idle_timeout_s / 6.0))

        try:
            while True:
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=poll_interval,
                    )
                    break
                except (asyncio.TimeoutError, TimeoutError):
                    pass

                if (
                    self._active_connections == 0
                    and (time.monotonic() - self._last_activity) >= self._idle_timeout_s
                ):
                    lock = FileLock(str(self.config.lock_file), timeout=0.05)
                    try:
                        with lock:
                            if (
                                not self._shutdown_event.is_set()
                                and self._active_connections == 0
                                and (time.monotonic() - self._last_activity)
                                >= self._idle_timeout_s
                            ):
                                self._unlink_owned_endpoints()
                                server.close()
                                cleaned_under_lock = True
                                break
                    except FileLockTimeout:
                        continue
        finally:
            if not cleaned_under_lock:
                # Allow any in-flight shutdown response frame to drain cleanly first
                drain_deadline = time.monotonic() + 0.15
                while self._active_connections > 0 and time.monotonic() < drain_deadline:
                    await asyncio.sleep(0.005)
                try:
                    with FileLock(str(self.config.lock_file), timeout=2.0):
                        self._unlink_owned_endpoints()
                        server.close()
                except FileLockTimeout:
                    self._unlink_owned_endpoints()
                    server.close()

            try:
                await server.wait_closed()
            except Exception:
                pass


class EngineClient:
    """Synchronous (plus `.a*` async) IPC client for `DiodeEngineServer` (`hooks.py`, `cli.py`, tests)."""

    def __init__(
        self,
        config: TransportConfig | None = None,
        *,
        auto_spawn: bool = True,
        idle_timeout_s: float = float(DEFAULT_IDLE_TIMEOUT_SECONDS),
        bundle_dir: Path | str | None = None,
        timeout_s: float = 15.0,
        session_id: str | None = None,
    ) -> None:
        self.config: TransportConfig = config if config is not None else default_transport()
        self.auto_spawn: bool = auto_spawn
        self.idle_timeout_s: float = float(idle_timeout_s)
        self.bundle_dir: Path | None = (
            Path(bundle_dir).expanduser() if bundle_dir is not None else None
        )
        self.timeout_s: float = float(timeout_s)
        self.session_id: str = session_id or f"sess_{secrets.token_hex(16)}"

    def _call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        admin_secret: str | None = None,
    ) -> dict[str, Any]:
        return send_rpc_sync(
            self.config,
            method,
            params,
            admin_secret=admin_secret,
            auto_spawn=self.auto_spawn,
            idle_timeout_s=self.idle_timeout_s,
            bundle_dir=self.bundle_dir,
            timeout_s=self.timeout_s,
        )

    async def _acall(
        self,
        method: str,
        params: dict[str, Any],
        *,
        admin_secret: str | None = None,
    ) -> dict[str, Any]:
        return await send_rpc_async(
            self.config,
            method,
            params,
            admin_secret=admin_secret,
            auto_spawn=self.auto_spawn,
            idle_timeout_s=self.idle_timeout_s,
            bundle_dir=self.bundle_dir,
            timeout_s=self.timeout_s,
        )

    def status(self) -> dict[str, Any]:
        return self._call("status", {})

    async def astatus(self) -> dict[str, Any]:
        return await self._acall("status", {})

    def evaluate_call(
        self,
        workspace_root: Path | str | None = None,
        server: str = "",
        tool: str = "",
        arguments: dict[str, Any] | None = None,
        *,
        client_supports_popup: bool = False,
        client_kind: Literal["mcp_shim", "ide_hook", "headless"] = "headless",
        permission_mode: str | None = None,
        config_path: Path | str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "workspace_root": os.fspath(workspace_root)
            if workspace_root is not None
            else None,
            "server": server,
            "tool": tool,
            "arguments": dict(arguments or {}),
            "client_supports_popup": bool(client_supports_popup),
            "client_kind": client_kind,
            "permission_mode": permission_mode,
            "config_path": os.fspath(config_path) if config_path is not None else None,
            "session_id": self.session_id,
        }
        return self._call("evaluate_call", params)

    async def aevaluate_call(
        self,
        workspace_root: Path | str | None = None,
        server: str = "",
        tool: str = "",
        arguments: dict[str, Any] | None = None,
        *,
        client_supports_popup: bool = False,
        client_kind: Literal["mcp_shim", "ide_hook", "headless"] = "headless",
        permission_mode: str | None = None,
        config_path: Path | str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "workspace_root": os.fspath(workspace_root)
            if workspace_root is not None
            else None,
            "server": server,
            "tool": tool,
            "arguments": dict(arguments or {}),
            "client_supports_popup": bool(client_supports_popup),
            "client_kind": client_kind,
            "permission_mode": permission_mode,
            "config_path": os.fspath(config_path) if config_path is not None else None,
            "session_id": self.session_id,
        }
        return await self._acall("evaluate_call", params)

    def record_output(
        self,
        workspace_root: Path | str | None,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None,
        output_text: str,
        *,
        config_path: Path | str | None = None,
        extra_flags: int | CapabilityFlag = CapabilityFlag.LOCAL_SAFE,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "workspace_root": os.fspath(workspace_root)
            if workspace_root is not None
            else None,
            "server": server,
            "tool": tool,
            "arguments": dict(arguments or {}),
            "output_text": output_text,
            "config_path": os.fspath(config_path) if config_path is not None else None,
            "extra_flags": int(extra_flags),
        }
        return self._call("record_output", params)

    async def arecord_output(
        self,
        workspace_root: Path | str | None,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None,
        output_text: str,
        *,
        config_path: Path | str | None = None,
        extra_flags: int | CapabilityFlag = CapabilityFlag.LOCAL_SAFE,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "workspace_root": os.fspath(workspace_root)
            if workspace_root is not None
            else None,
            "server": server,
            "tool": tool,
            "arguments": dict(arguments or {}),
            "output_text": output_text,
            "config_path": os.fspath(config_path) if config_path is not None else None,
            "extra_flags": int(extra_flags),
        }
        return await self._acall("record_output", params)

    def record_intent(
        self,
        workspace_root: Path | str | None,
        intent: str | None,
    ) -> dict[str, Any]:
        return self._call(
            "record_intent",
            {
                "workspace_root": os.fspath(workspace_root)
                if workspace_root is not None
                else None,
                "intent": intent,
            },
        )

    async def arecord_intent(
        self,
        workspace_root: Path | str | None,
        intent: str | None,
    ) -> dict[str, Any]:
        return await self._acall(
            "record_intent",
            {
                "workspace_root": os.fspath(workspace_root)
                if workspace_root is not None
                else None,
                "intent": intent,
            },
        )

    def record_local_write(
        self,
        workspace_root: Path | str | None,
        target_path: str,
        written_content: str,
    ) -> dict[str, Any]:
        return self._call(
            "record_local_write",
            {
                "workspace_root": os.fspath(workspace_root)
                if workspace_root is not None
                else None,
                "target_path": target_path,
                "written_content": written_content,
            },
        )

    async def arecord_local_write(
        self,
        workspace_root: Path | str | None,
        target_path: str,
        written_content: str,
    ) -> dict[str, Any]:
        return await self._acall(
            "record_local_write",
            {
                "workspace_root": os.fspath(workspace_root)
                if workspace_root is not None
                else None,
                "target_path": target_path,
                "written_content": written_content,
            },
        )

    def resolve_elicit(
        self,
        incident_id: str,
        choice: Literal["allow_once", "redact", "block", "trust_shape"],
        *,
        forwarded_arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "resolve_elicit",
            {
                "incident_id": incident_id,
                "choice": choice,
                "session_id": self.session_id,
                "forwarded_arguments": forwarded_arguments,
            },
        )

    async def aresolve_elicit(
        self,
        incident_id: str,
        choice: Literal["allow_once", "redact", "block", "trust_shape"],
        *,
        forwarded_arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._acall(
            "resolve_elicit",
            {
                "incident_id": incident_id,
                "choice": choice,
                "session_id": self.session_id,
                "forwarded_arguments": forwarded_arguments,
            },
        )

    def list_incidents(self, *, limit: int = 50) -> list[dict[str, Any]]:
        res = self._call("list_incidents", {"limit": int(limit)})
        items = res.get("incidents", [])
        return list(items) if isinstance(items, list) else []

    async def alist_incidents(self, *, limit: int = 50) -> list[dict[str, Any]]:
        res = await self._acall("list_incidents", {"limit": int(limit)})
        items = res.get("incidents", [])
        return list(items) if isinstance(items, list) else []

    def approve_incident(
        self,
        incident_id: str,
        *,
        admin_secret: str | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "approve_incident",
            {"incident_id": incident_id},
            admin_secret=admin_secret,
        )

    async def aapprove_incident(
        self,
        incident_id: str,
        *,
        admin_secret: str | None = None,
    ) -> dict[str, Any]:
        return await self._acall(
            "approve_incident",
            {"incident_id": incident_id},
            admin_secret=admin_secret,
        )

    # Alias for Ruling R4 backward-compat naming (`approve_allow_token`)
    approve_allow_token = approve_incident
    aapprove_allow_token = aapprove_incident

    def trust_tool_shape(
        self,
        workspace_root: Path | str | None,
        server: str,
        tool: str,
        arg_keys: Iterable[str],
        *,
        admin_secret: str | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "trust_tool_shape",
            {
                "workspace_root": os.fspath(workspace_root)
                if workspace_root is not None
                else None,
                "server": server,
                "tool": tool,
                "arg_keys": [str(k) for k in arg_keys],
            },
            admin_secret=admin_secret,
        )

    async def atrust_tool_shape(
        self,
        workspace_root: Path | str | None,
        server: str,
        tool: str,
        arg_keys: Iterable[str],
        *,
        admin_secret: str | None = None,
    ) -> dict[str, Any]:
        return await self._acall(
            "trust_tool_shape",
            {
                "workspace_root": os.fspath(workspace_root)
                if workspace_root is not None
                else None,
                "server": server,
                "tool": tool,
                "arg_keys": [str(k) for k in arg_keys],
            },
            admin_secret=admin_secret,
        )

    def shutdown(self, *, admin_secret: str | None = None) -> dict[str, Any]:
        return self._call("shutdown", {}, admin_secret=admin_secret)

    async def ashutdown(self, *, admin_secret: str | None = None) -> dict[str, Any]:
        return await self._acall("shutdown", {}, admin_secret=admin_secret)


class AsyncEngineClient:
    """First-class async IPC client for `shim.py` (`await client.evaluate_call(...)`, `await client.record_output(...)`)."""

    def __init__(
        self,
        config: TransportConfig | None = None,
        *,
        auto_spawn: bool = True,
        idle_timeout_s: float = float(DEFAULT_IDLE_TIMEOUT_SECONDS),
        bundle_dir: Path | str | None = None,
        timeout_s: float = 15.0,
        session_id: str | None = None,
    ) -> None:
        self._sync = EngineClient(
            config,
            auto_spawn=auto_spawn,
            idle_timeout_s=idle_timeout_s,
            bundle_dir=bundle_dir,
            timeout_s=timeout_s,
            session_id=session_id,
        )

    @property
    def config(self) -> TransportConfig:
        return self._sync.config

    @property
    def session_id(self) -> str:
        return self._sync.session_id

    async def status(self) -> dict[str, Any]:
        return await self._sync.astatus()

    async def evaluate_call(
        self,
        workspace_root: Path | str | None = None,
        server: str = "",
        tool: str = "",
        arguments: dict[str, Any] | None = None,
        *,
        client_supports_popup: bool = False,
        client_kind: Literal["mcp_shim", "ide_hook", "headless"] = "headless",
        permission_mode: str | None = None,
        config_path: Path | str | None = None,
    ) -> dict[str, Any]:
        return await self._sync.aevaluate_call(
            workspace_root=workspace_root,
            server=server,
            tool=tool,
            arguments=arguments,
            client_supports_popup=client_supports_popup,
            client_kind=client_kind,
            permission_mode=permission_mode,
            config_path=config_path,
        )

    async def record_output(
        self,
        workspace_root: Path | str | None,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None,
        output_text: str,
        *,
        config_path: Path | str | None = None,
        extra_flags: int | CapabilityFlag = CapabilityFlag.LOCAL_SAFE,
    ) -> dict[str, Any]:
        return await self._sync.arecord_output(
            workspace_root,
            server,
            tool,
            arguments,
            output_text,
            config_path=config_path,
            extra_flags=extra_flags,
        )

    async def record_intent(
        self,
        workspace_root: Path | str | None,
        intent: str | None,
    ) -> dict[str, Any]:
        return await self._sync.arecord_intent(workspace_root, intent)

    async def record_local_write(
        self,
        workspace_root: Path | str | None,
        target_path: str,
        written_content: str,
    ) -> dict[str, Any]:
        return await self._sync.arecord_local_write(
            workspace_root, target_path, written_content
        )

    async def resolve_elicit(
        self,
        incident_id: str,
        choice: Literal["allow_once", "redact", "block", "trust_shape"],
        *,
        forwarded_arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._sync.aresolve_elicit(
            incident_id,
            choice,
            forwarded_arguments=forwarded_arguments,
        )

    async def list_incidents(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return await self._sync.alist_incidents(limit=limit)

    async def approve_incident(
        self,
        incident_id: str,
        *,
        admin_secret: str | None = None,
    ) -> dict[str, Any]:
        return await self._sync.aapprove_incident(incident_id, admin_secret=admin_secret)

    approve_allow_token = approve_incident

    async def trust_tool_shape(
        self,
        workspace_root: Path | str | None,
        server: str,
        tool: str,
        arg_keys: Iterable[str],
        *,
        admin_secret: str | None = None,
    ) -> dict[str, Any]:
        return await self._sync.atrust_tool_shape(
            workspace_root,
            server,
            tool,
            arg_keys,
            admin_secret=admin_secret,
        )

    async def shutdown(self, *, admin_secret: str | None = None) -> dict[str, Any]:
        return await self._sync.ashutdown(admin_secret=admin_secret)


def _main_daemon_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DiodeMCP background workspace engine")
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["unix", "tcp"], default=None)
    parser.add_argument("--socket-path", type=Path, default=None)
    parser.add_argument("--addr-file", type=Path, default=None)
    parser.add_argument("--admin-secret-file", type=Path, default=None)
    parser.add_argument("--lock-file", type=Path, default=None)
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=float(DEFAULT_IDLE_TIMEOUT_SECONDS),
    )
    parser.add_argument("--bundle-dir", type=Path, default=None)

    args = parser.parse_args(list(argv) if argv is not None else None)
    base_cfg = default_transport(args.state_dir, mode=args.mode)
    cfg = TransportConfig(
        mode=args.mode if args.mode is not None else base_cfg.mode,
        state_dir=args.state_dir if args.state_dir is not None else base_cfg.state_dir,
        socket_path=args.socket_path
        if args.socket_path is not None
        else base_cfg.socket_path,
        addr_file=args.addr_file if args.addr_file is not None else base_cfg.addr_file,
        admin_secret_file=args.admin_secret_file
        if args.admin_secret_file is not None
        else base_cfg.admin_secret_file,
        lock_file=args.lock_file if args.lock_file is not None else base_cfg.lock_file,
    )

    daemon = DiodeEngineServer(
        cfg,
        bundle_dir=args.bundle_dir,
        idle_timeout_s=args.idle_timeout,
    )
    asyncio.run(daemon.serve_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main_daemon_cli())
