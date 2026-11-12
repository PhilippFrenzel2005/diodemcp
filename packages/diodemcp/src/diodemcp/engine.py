"""Persistent workspace taint engine, 3-band approval router, and sync/async IPC clients (`C4`, `C5`, `C6`, `C7`, `I13`)."""

from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import stat
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
from diodemcp.registry import (
    CapabilityRegistry,
    _collect_strings_for_self_tamper,
    _extract_path_values,
    _extract_shell_path_candidates,
    _normalize_server_and_tool,
    canonicalize_target_path,
)
from diodemcp.transport import (
    MAX_IPC_LINE_BYTES,
    MIN_CREDENTIAL_CHARS,
    PRE_AUTH_TIMEOUT_SECONDS,
    PRIVILEGED_METHODS,
    REPLAY_WINDOW_SECONDS,
    TransportAuthError,
    TransportConfig,
    TransportError,
    TransportPermissionError,
    _is_ascii_credential,
    _sanitize_surrogates_in_obj,
    atomic_write_private_json,
    canonical_json_sha256,
    compute_admin_hmac,
    compute_client_hmac,
    compute_hello_hmac,
    compute_popup_binding_sha256,
    compute_server_hmac,
    constant_time_ascii_equals,
    default_transport,
    ensure_admin_secret,
    ensure_private_dir,
    is_pid_alive,
    sanitize_error_message,
    send_rpc_async,
    send_rpc_sync,
    unix_path_handle,
)
from diodemcp.types import CandidateCall, CapabilityFlag, DecisionScores, RiskBand, TaintSpan

DEFAULT_APPROVAL_TTL_SECONDS: float = 120.0
MAX_STORED_INCIDENTS: int = 256
MAX_SEEN_NONCES: int = 4096
MAX_CONCURRENT_CONNECTIONS: int = 64
MAX_INTENT_CHARS: int = 4096
PENDING_POPUP_RETENTION_SECONDS: float = 300.0

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_URL_EXTRACT_RE = re.compile(r"""https?://[^\s"'<>]+""", re.IGNORECASE)
_UMASK_LOCK = threading.Lock()
_UNSET_SECRET: Any = object()

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


def _is_safe_0600_regular_file(path: Path) -> bool:
    if path.is_symlink() or not path.exists():
        return False
    try:
        st = path.lstat()
    except OSError:
        return False
    if not stat.S_ISREG(st.st_mode):
        return False
    if os.name == "posix":
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            return False
        if stat.S_IMODE(st.st_mode) != 0o600:
            return False
    return True


def _load_configured_predictor(
    state_dir: Path,
    explicit_bundle_dir: Path | str | None = None,
) -> tuple[PredictorProtocol, str]:
    """Load `OrtJevPredictor` when configured/present, else fall back to `Tier0DeterministicPredictor`."""
    if explicit_bundle_dir is not None:
        ort_pred = OrtJevPredictor(explicit_bundle_dir)
        return ort_pred, f"ort:{ort_pred.manifest.model_id}"

    active_cfg_file = state_dir / "active_bundle.json"
    if _is_safe_0600_regular_file(active_cfg_file):
        data = json.loads(active_cfg_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if data.get("tier0_only") is True:
                return Tier0DeterministicPredictor(), "tier0"
            cfg_bundle = data.get("bundle_dir")
            if isinstance(cfg_bundle, str) and cfg_bundle.strip():
                ort_pred = OrtJevPredictor(Path(cfg_bundle))
                return ort_pred, f"ort:{ort_pred.manifest.model_id}"

    default_current = state_dir / "models" / "current"
    if default_current.is_dir() and not default_current.is_symlink():
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
    safe_entry = _sanitize_surrogates_in_obj(entry)
    line_bytes = (
        json.dumps(safe_entry, ensure_ascii=False, separators=(",", ":")) + "\n"
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


def _merge_floor_scores(base: DecisionScores, floor: DecisionScores) -> DecisionScores:
    """Take elementwise max of `(p_enc, p_sem, p_hijack)` preserving strongest attribution."""
    new_enc = max(base.p_enc, floor.p_enc)
    new_sem = max(base.p_sem, floor.p_sem)
    new_hijack = max(base.p_hijack, floor.p_hijack)
    if (
        new_enc == base.p_enc
        and new_sem == base.p_sem
        and new_hijack == base.p_hijack
    ):
        return base

    if floor.p_max >= base.p_max:
        span_idx = (
            floor.attributed_span_idx
            if floor.attributed_span_idx is not None
            else base.attributed_span_idx
        )
        span_id = (
            floor.attributed_span_id
            if floor.attributed_span_id is not None
            else base.attributed_span_id
        )
        arg_key = floor.attributed_arg_key or base.attributed_arg_key
    else:
        span_idx = (
            base.attributed_span_idx
            if base.attributed_span_idx is not None
            else floor.attributed_span_idx
        )
        span_id = (
            base.attributed_span_id
            if base.attributed_span_id is not None
            else floor.attributed_span_id
        )
        arg_key = base.attributed_arg_key or floor.attributed_arg_key

    return DecisionScores(
        p_enc=new_enc,
        p_sem=new_sem,
        p_hijack=new_hijack,
        attributed_span_idx=span_idx,
        attributed_span_id=span_id,
        attributed_arg_key=arg_key,
    )


def _normalize_client_workspace(workspace_root: Path | str | None) -> str:
    if workspace_root is None:
        return os.path.realpath(os.getcwd())
    if not isinstance(workspace_root, (str, os.PathLike)):
        raise ValueError("workspace_root must be a path string or Path")
    raw = os.fspath(workspace_root)
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise ValueError("workspace_root must be a non-empty path")
    return os.path.abspath(os.path.expanduser(raw.strip()))


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
        self._real_state_dir_fold: str = canonicalize_target_path(
            os.fspath(self.config.state_dir), casefold=True
        ).rstrip("/")
        self._protected_exact_paths_fold: frozenset[str] = frozenset(
            canonicalize_target_path(os.fspath(p), casefold=True)
            for p in (
                self.config.socket_path,
                self.config.addr_file,
                self.config.admin_secret_file,
                self.config.lock_file,
            )
        )
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
        self._replay_lock = threading.Lock()
        self._seen_nonces: OrderedDict[tuple[str, str], float] = OrderedDict()

        self._ledgers: dict[str, SessionLedger] = {}
        self._registries: dict[tuple[str, str, int], CapabilityRegistry] = {}
        self._active_workspace_count: int = 0
        self._active_incident_count: int = 0

        # Workspace-wide trusted shapes granted via admin_secret:
        #   set[(workspace_key, norm_server, norm_tool, frozenset[str])]
        self._admin_trusted_shapes: set[tuple[str, str, str, frozenset[str]]] = set()

        # Ephemeral shim-scoped trusted shapes bound strictly to sha256(shim_secret):
        #   set[(workspace_key, popup_binding_sha256, norm_server, norm_tool, frozenset[str])]
        self._session_trusted_shapes: set[tuple[str, str, str, str, frozenset[str]]] = set()

        self._incidents: dict[str, dict[str, Any]] = {}
        self._incident_order: list[str] = []
        self._approval_monotonic_deadlines: dict[str, float] = {}
        self._rehydrate_incidents()
        self._active_incident_count = len(self._incidents)

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
        if not _is_safe_0600_regular_file(self._incidents_path):
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
            now_wall = time.time()
            now_mono = time.monotonic()
            for item in items[-MAX_STORED_INCIDENTS:]:
                if isinstance(item, dict) and isinstance(item.get("incident_id"), str):
                    inc_id = item["incident_id"]
                    self._incidents[inc_id] = item
                    self._incident_order.append(inc_id)
                    app_until = item.get("approved_until")
                    if (
                        not item.get("consumed", False)
                        and isinstance(app_until, (int, float))
                        and not isinstance(app_until, bool)
                    ):
                        rem = min(self._approval_ttl_s, max(0.0, float(app_until) - now_wall))
                        self._approval_monotonic_deadlines[inc_id] = now_mono + rem
        except Exception:
            self._incidents.clear()
            self._incident_order.clear()
            self._approval_monotonic_deadlines.clear()

    def _is_approval_unexpired(
        self,
        inc_id: str,
        inc: Mapping[str, Any],
        now_wall: float,
        now_mono: float,
    ) -> bool:
        if inc.get("consumed", False):
            return False
        approved_until = inc.get("approved_until")
        if approved_until is None or isinstance(approved_until, bool):
            return False
        try:
            wall_exp = float(approved_until)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(wall_exp) or now_wall > wall_exp:
            return False
        mono_exp = self._approval_monotonic_deadlines.get(inc_id)
        if mono_exp is not None and now_mono > mono_exp:
            return False
        return True

    def _is_incident_protected_from_eviction(
        self,
        inc_id: str,
        inc: Mapping[str, Any],
        now_wall: float,
        now_mono: float,
    ) -> bool:
        if inc.get("consumed", False):
            return False
        if self._is_approval_unexpired(inc_id, inc, now_wall, now_mono):
            return True
        if inc.get("client_supports_popup", False) and inc.get("elicit_resolved_choice") is None:
            created = inc.get("created_at", 0.0)
            try:
                age = now_wall - float(created)
            except (TypeError, ValueError):
                age = PENDING_POPUP_RETENTION_SECONDS + 1.0
            if 0.0 <= age <= PENDING_POPUP_RETENTION_SECONDS:
                return True
        return False

    def _evict_excess_incidents(self) -> None:
        if len(self._incident_order) <= MAX_STORED_INCIDENTS:
            return
        now_wall = time.time()
        now_mono = time.monotonic()
        while len(self._incident_order) > MAX_STORED_INCIDENTS:
            victim_idx: int | None = None
            for idx, cand_id in enumerate(self._incident_order):
                cand = self._incidents.get(cand_id)
                if cand is None or not self._is_incident_protected_from_eviction(
                    cand_id, cand, now_wall, now_mono
                ):
                    victim_idx = idx
                    break
            if victim_idx is None:
                victim_idx = 0
            evicted_id = self._incident_order.pop(victim_idx)
            self._incidents.pop(evicted_id, None)
            self._approval_monotonic_deadlines.pop(evicted_id, None)

    def _persist_incidents(self) -> None:
        self._evict_excess_incidents()
        self._active_incident_count = len(self._incidents)
        payload = {
            "incidents": [
                self._incidents[iid]
                for iid in self._incident_order
                if iid in self._incidents
            ]
        }
        atomic_write_private_json(self._incidents_path, payload)

    def _resolve_workspace(self, workspace_root: Path | str | None) -> tuple[Path, str]:
        if workspace_root is None:
            raise ValueError("workspace_root is required and must not be None")
        if not isinstance(workspace_root, (str, os.PathLike)):
            raise ValueError("workspace_root must be a path string or Path")
        raw = os.fspath(workspace_root)
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
            raise ValueError("workspace_root must be a non-empty path")
        expanded = os.path.expanduser(raw.strip())
        if not os.path.isabs(expanded):
            raise ValueError("workspace_root must be an absolute path")
        real_ws = Path(os.path.realpath(expanded))
        ws_key = compute_workspace_key(real_ws)
        return real_ws, ws_key

    def _get_ledger(self, workspace_root: Path | str | None) -> tuple[Path, SessionLedger]:
        real_ws, ws_key = self._resolve_workspace(workspace_root)
        ledger = self._ledgers.get(ws_key)
        if ledger is None:
            persist_file = self._sessions_dir / f"{ws_key}.json"
            ledger = SessionLedger(workspace_root=real_ws, persist_path=persist_file)
            self._ledgers[ws_key] = ledger
            self._active_workspace_count = len(self._ledgers)
        return real_ws, ledger

    def _get_registry(self, real_ws: Path) -> CapabilityRegistry:
        """Load custom policy ONLY from verified `0600` non-symlink `state_dir / 'config.toml'`; else strict default."""
        state_cfg = self.config.state_dir / "config.toml"
        strict_sentinel = self.config.state_dir / ".__default_strict_only__.toml"

        if _is_safe_0600_regular_file(state_cfg):
            try:
                mtime_ns = state_cfg.lstat().st_mtime_ns
                eff_cfg = state_cfg
            except OSError:
                mtime_ns = -1
                eff_cfg = strict_sentinel
        else:
            mtime_ns = 0
            eff_cfg = strict_sentinel

        cache_key = (str(real_ws), str(eff_cfg), mtime_ns)
        reg = self._registries.get(cache_key)
        if reg is None:
            try:
                reg = CapabilityRegistry.from_default_and_user_config(
                    config_path=eff_cfg,
                    workspace_root=real_ws,
                )
            except Exception:
                reg = CapabilityRegistry.from_default_and_user_config(
                    config_path=strict_sentinel,
                    workspace_root=real_ws,
                )
            if len(self._registries) >= 32:
                self._registries.clear()
            self._registries[cache_key] = reg
        return reg

    def _matches_active_state_path(self, folded_canon: str) -> bool:
        if not folded_canon:
            return False
        if folded_canon in self._protected_exact_paths_fold:
            return True
        if self._real_state_dir_fold:
            if (
                folded_canon == self._real_state_dir_fold
                or folded_canon.startswith(f"{self._real_state_dir_fold}/")
            ):
                return True
        return False

    def _is_active_state_self_tamper(
        self,
        registry: CapabilityRegistry,
        server: str,
        tool: str,
        args: dict[str, Any],
    ) -> bool:
        norm_srv, norm_tool = _normalize_server_and_tool(server, tool)
        for folded in registry._canonical_folded_paths(_extract_path_values(args)):
            if self._matches_active_state_path(folded):
                return True

        strings: list[str] = []
        _collect_strings_for_self_tamper(
            args,
            strings,
            skip_file_body_keys=registry._is_file_write_call(norm_srv, norm_tool, args),
        )
        for text in strings:
            for folded_tok in registry._canonical_folded_paths(
                _extract_shell_path_candidates(text)
            ):
                if self._matches_active_state_path(folded_tok):
                    return True
        return False

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
    ) -> tuple[Literal["none", "matched", "persist_failed"], dict[str, Any] | None]:
        now_wall = time.time()
        now_mono = time.monotonic()
        for inc_id in reversed(self._incident_order):
            inc = self._incidents.get(inc_id)
            if inc is None:
                continue
            if (
                self._is_approval_unexpired(inc_id, inc, now_wall, now_mono)
                and inc.get("workspace_key") == workspace_key
                and inc.get("server") == server
                and inc.get("tool") == tool
                and inc.get("canonical_args_sha256") == args_sha256
            ):
                inc["consumed"] = True
                inc["consumed_at"] = now_wall
                self._approval_monotonic_deadlines.pop(inc_id, None)
                try:
                    self._persist_incidents()
                except Exception:
                    # Fail closed if disk persistence of consumed=True fails!
                    return "persist_failed", inc
                return "matched", inc
        return "none", None

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
        config_path: Path | str | None = None,  # Deprecated/ignored (Critical 2)
        popup_binding_sha256: str | None = None,
        session_id: str | None = None,  # Legacy/ignored for popup auth (Critical 1)
    ) -> dict[str, Any]:
        """Run full 5-step taint/self-protection/predictor pipeline and 3-band approval router."""
        del config_path, session_id  # Explicitly ignored: server loads only verified state_dir/config.toml
        if client_kind not in ("mcp_shim", "ide_hook", "headless"):
            raise ValueError(f"Invalid client_kind: {client_kind!r}")
        if arguments is not None and not isinstance(arguments, Mapping):
            raise ValueError("arguments must be a dict/mapping or None")

        norm_binding: str | None = None
        if popup_binding_sha256 is not None:
            if not isinstance(popup_binding_sha256, str):
                raise ValueError("popup_binding_sha256 must be a 64-char hex string")
            cand_binding = popup_binding_sha256.strip().lower()
            if not _HEX64_RE.fullmatch(cand_binding):
                raise ValueError("popup_binding_sha256 must be a valid 64-char hex SHA-256")
            norm_binding = cand_binding

        call = CandidateCall(
            server=server,
            tool=tool,
            arguments=dict(arguments or {}),
        )
        args_sha = _canonical_args_sha256(call.arguments)
        norm_srv, norm_tool = _normalize_server_and_tool(server, tool)

        # Phase 1 (under _state_lock): fast state snapshot, single-use approval check, and fast-path gate
        with self._state_lock:
            real_ws, ledger = self._get_ledger(workspace_root)
            registry = self._get_registry(real_ws)
            ws_key = ledger.workspace_key

            req_flags = registry.classify_request(server, tool, call.arguments)
            is_self_tamper = registry.is_unconditional_critical_self_tamper(
                server, tool, call.arguments
            ) or self._is_active_state_self_tamper(
                registry, server, tool, call.arguments
            )
            writes_ext = bool(req_flags & CapabilityFlag.WRITES_EXTERNAL)
            mutates_sens = bool(req_flags & CapabilityFlag.MUTATES_SENSITIVE)
            is_pure_local_rw = registry._is_file_read_call(
                norm_srv, norm_tool, call.arguments
            ) or (
                registry._is_file_write_call(norm_srv, norm_tool, call.arguments)
                and not writes_ext
                and not mutates_sens
            )
            touches_tainted_write = ledger.touches_tainted_local_write(call)
            effective_tainted_write_sink = bool(
                touches_tainted_write and not is_pure_local_rw
            )

            snap = ledger.snapshot()
            priv_list = list(snap.private_spans)
            untrust_list = list(snap.untrusted_spans)

            # Step 2: Single-use out-of-band human approval replay check
            app_status, matched_approval = self._consume_matching_approval(
                ws_key, server, tool, args_sha
            )
            if app_status == "persist_failed":
                inc_id_fail = (
                    matched_approval["incident_id"]
                    if matched_approval is not None
                    else f"inc_{secrets.token_hex(12)}"
                )
                fail_scores = DecisionScores(p_enc=1.0, p_sem=0.0, p_hijack=0.0)
                return {
                    "action": "block",
                    "band": RiskBand.CRITICAL.value,
                    "incident_id": inc_id_fail,
                    "p_max": 1.0,
                    "scores": fail_scores.model_dump(mode="json"),
                    "server": server,
                    "tool": tool,
                    "workspace_key": ws_key,
                    "is_self_tamper": is_self_tamper,
                    "reason": "Failed to persist single-use approval consumption; failing closed.",
                    "banner": "🛑 CRITICAL DiodeMCP Alert (1.00): Failed to persist approval consumption.",
                    "message": (
                        f"Blocked by DiodeMCP (incident {inc_id_fail}). "
                        f"Review in an interactive terminal via: diodemcp audit"
                    ),
                }
            if app_status == "matched" and matched_approval is not None:
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

            if not is_self_tamper:
                has_untrusted_any = bool(untrust_list) or any(
                    s.dual_trust or bool(s.flags & CapabilityFlag.READS_UNTRUSTED)
                    for s in priv_list
                )
                can_fast_path = ledger.should_fast_path_allow(
                    req_flags,
                    is_self_tamper=False,
                    touches_tainted_local_write=effective_tainted_write_sink,
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

        # Phase 2 (OUTSIDE _state_lock): run Tier0 + ORT/Predictor over immutable snapshot
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
            if isinstance(self._predictor, OrtJevPredictor):
                scores = self._predictor.evaluate_snapshot(snap, call)
            else:
                scores = self._predictor.evaluate(
                    intent=snap.intent,
                    private_spans=priv_list,
                    untrusted_spans=untrust_list,
                    call=call,
                )
            # Critical 1: ALWAYS re-run Tier0 floor so raw secrets >= critical_min always escalate
            if self._predictor is not self._tier0:
                t0_floor = self._tier0.evaluate(
                    intent=snap.intent,
                    private_spans=priv_list,
                    untrusted_spans=untrust_list,
                    call=call,
                )
                scores = _merge_floor_scores(scores, t0_floor)

        # Phase 3 (re-acquire _state_lock): live rolling/tainted-write floors, trust_shape, and routing
        with self._state_lock:
            warn_min, critical_min = self._thresholds()
            latest_snap = ledger.snapshot()
            priv_list = list(latest_snap.private_spans)
            untrust_list = list(latest_snap.untrusted_spans)

            if not is_self_tamper:
                if writes_ext:
                    rolling_floor = self._check_rolling_outbound_floor(
                        ledger, priv_list, untrust_list, latest_snap.intent, call
                    )
                    if rolling_floor is not None:
                        scores = _merge_floor_scores(scores, rolling_floor)

                if not is_pure_local_rw and ledger.touches_tainted_local_write(call):
                    write_floor = self._check_tainted_local_write_floor(
                        ledger, priv_list, latest_snap.intent, call
                    )
                    if write_floor is not None:
                        scores = _merge_floor_scores(scores, write_floor)

            is_critical_hijack = bool(
                not is_self_tamper and scores.p_hijack >= critical_min
            )
            p_max = scores.p_max

            leaked_span: TaintSpan | None = None
            if scores.attributed_span_idx is not None and 0 <= scores.attributed_span_idx < len(
                priv_list
            ):
                leaked_span = priv_list[scores.attributed_span_idx]

            has_high_secret = bool(
                leaked_span is not None and (leaked_span.flags & CapabilityFlag.HIGH_SECRET)
            )

            # Step 3: Session trust_tool_shape check (strictly non-CRITICAL WARN, non-self-tamper, non-HIGH_SECRET)
            flat_arg_keys = frozenset(
                flatten_call_arguments(call.arguments, max_args=None).keys()
            )
            if (
                not is_self_tamper
                and not is_critical_hijack
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
                    and norm_binding is not None
                    and (ws_key, norm_binding, norm_srv, norm_tool, flat_arg_keys)
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
                if (p_max >= critical_min or is_self_tamper or is_critical_hijack)
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
                "is_critical_hijack": is_critical_hijack,
                "has_high_secret": has_high_secret,
                "client_supports_popup": bool(client_supports_popup),
                "popup_binding_sha256": norm_binding if client_supports_popup else None,
                "reason": reason,
                "args_preview": args_preview,
                "approved_until": None,
                "consumed": False,
                "elicit_resolved_choice": None,
            }
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
        config_path: Path | str | None = None,  # Deprecated/ignored (Critical 2)
        extra_flags: int | CapabilityFlag = CapabilityFlag.LOCAL_SAFE,
    ) -> dict[str, Any]:
        """Classify tool output via `CapabilityRegistry` and persist into workspace `SessionLedger` (`I6`)."""
        del config_path
        if arguments is not None and not isinstance(arguments, Mapping):
            raise ValueError("arguments must be a dict/mapping or None")

        with self._state_lock:
            real_ws, ledger = self._get_ledger(workspace_root)
            registry = self._get_registry(real_ws)
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
        """Record current user prompt intent (`<= 4096` chars) and audit update (`Important 4`)."""
        if intent is not None and not isinstance(intent, str):
            raise ValueError("intent must be a string or None")

        was_truncated = False
        capped_intent: str | None
        if intent is None:
            capped_intent = None
        else:
            cleaned = intent.replace("\x00", "")
            if len(cleaned) > MAX_INTENT_CHARS:
                capped_intent = cleaned[:MAX_INTENT_CHARS]
                was_truncated = True
            else:
                capped_intent = cleaned

        with self._state_lock:
            _real_ws, ledger = self._get_ledger(workspace_root)
            ledger.record_intent(capped_intent)
            snap = ledger.snapshot()
            stored = snap.intent
            redacted_intent_preview = (
                build_redacted_audit_preview(
                    {"intent": stored},
                    None,
                    None,
                    private_spans=list(snap.private_spans),
                    max_value_chars=120,
                ).get("intent", "")
                if stored is not None
                else None
            )
            _append_jsonl_0600(
                self._audit_path,
                {
                    "timestamp": time.time(),
                    "event": "record_intent",
                    "workspace_key": ledger.workspace_key,
                    "has_intent": stored is not None,
                    "intent_length": len(stored) if stored is not None else 0,
                    "truncated": was_truncated,
                    "intent_sha256": (
                        hashlib.sha256(
                            stored.encode("utf-8", errors="replace")
                        ).hexdigest()
                        if stored is not None
                        else None
                    ),
                    "intent_preview": redacted_intent_preview,
                },
            )
            return {
                "ok": True,
                "workspace_key": ledger.workspace_key,
                "has_intent": stored is not None,
                "truncated": was_truncated,
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
        shim_secret: str | None = None,
        forwarded_arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record interactive MCP `elicitation/create` user decision verified by `shim_secret` (`Critical 1`)."""
        if choice not in ("allow_once", "redact", "block", "trust_shape"):
            raise ValueError(f"Unsupported elicit choice: {choice!r}")

        if not _is_ascii_credential(shim_secret, min_len=MIN_CREDENTIAL_CHARS, max_len=512):
            raise TransportPermissionError(
                "resolve_elicit requires valid originating shim_secret"
            )
        presented_binding = compute_popup_binding_sha256(str(shim_secret))

        with self._state_lock:
            inc = self._incidents.get(incident_id)
            if inc is None:
                constant_time_ascii_equals(presented_binding, "0" * 64)
                raise ValueError(f"Unknown incident_id: {incident_id!r}")
            if not inc.get("client_supports_popup", False):
                constant_time_ascii_equals(presented_binding, "0" * 64)
                raise ValueError(
                    f"Incident {incident_id!r} did not originate from an interactive popup"
                )

            stored_binding = inc.get("popup_binding_sha256")
            if not isinstance(stored_binding, str) or not _HEX64_RE.fullmatch(stored_binding):
                constant_time_ascii_equals(presented_binding, "0" * 64)
                raise TransportPermissionError(
                    "Incident has no valid popup_binding_sha256; refusing unauthenticated resolution"
                )
            if not constant_time_ascii_equals(presented_binding, stored_binding):
                raise TransportPermissionError(
                    "shim_secret mismatch for popup incident resolution"
                )

            if inc.get("elicit_resolved_choice") is not None or inc.get("consumed", False):
                raise ValueError(f"Incident {incident_id!r} was already resolved")

            if choice == "trust_shape":
                if (
                    inc.get("band") != RiskBand.WARN.value
                    or inc.get("is_self_tamper", False)
                    or inc.get("is_critical_hijack", False)
                    or inc.get("has_high_secret", False)
                ):
                    raise ValueError(
                        "trust_shape is strictly restricted to non-CRITICAL WARN incidents"
                    )
                norm_srv, norm_tool = _normalize_server_and_tool(
                    str(inc["server"]), str(inc["tool"])
                )
                shape_keys = frozenset(str(k) for k in inc.get("flat_arg_keys", []))
                self._session_trusted_shapes.add(
                    (
                        str(inc["workspace_key"]),
                        stored_binding,
                        norm_srv,
                        norm_tool,
                        shape_keys,
                    )
                )

            inc["elicit_resolved_choice"] = choice
            inc["consumed"] = True
            inc["resolved_at"] = time.time()
            self._persist_incidents()

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
        if not constant_time_ascii_equals(admin_secret, self._admin_secret):
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
            if inc.get("approved_until") is not None:
                raise ValueError(
                    f"Incident {incident_id!r} was already approved and cannot be re-armed"
                )
            if inc.get("elicit_resolved_choice") is not None:
                raise ValueError(f"Incident {incident_id!r} was already resolved")

            expires_at = time.time() + self._approval_ttl_s
            mono_deadline = time.monotonic() + self._approval_ttl_s
            inc["approved_until"] = expires_at
            self._approval_monotonic_deadlines[incident_id] = mono_deadline
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
            if registry.is_unconditional_critical_self_tamper(
                server, tool, probe_args
            ) or self._is_active_state_self_tamper(registry, server, tool, probe_args):
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

    def list_incidents(
        self,
        *,
        limit: int = 50,
        admin_secret: str | None = None,
        _admin_verified: bool = False,
    ) -> dict[str, Any]:
        """Return bounded list of recent redacted incidents (strictly requires `admin_secret`)."""
        if not _admin_verified:
            self._require_admin_secret_direct(admin_secret, "list_incidents")

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
        """Return daemon health, transport mode, active predictor, and session counts without lock contention."""
        return {
            "ok": True,
            "pid": os.getpid(),
            "mode": self.config.mode,
            "predictor": self._predictor_label,
            "uptime_s": round(max(0.0, time.time() - self._started_at), 3),
            "idle_timeout_s": self._idle_timeout_s,
            "active_workspaces": self._active_workspace_count,
            "incident_count": self._active_incident_count,
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
            # Critical 2: Ignore any client-supplied config_path in params!
            return self.evaluate_call(
                workspace_root=params.get("workspace_root"),
                server=str(params["server"]),
                tool=str(params["tool"]),
                arguments=params.get("arguments"),
                client_supports_popup=bool(params.get("client_supports_popup", False)),
                client_kind=params.get("client_kind", "headless"),
                permission_mode=params.get("permission_mode"),
                popup_binding_sha256=params.get("popup_binding_sha256"),
            )
        if method == "record_output":
            # Critical 2: Ignore any client-supplied config_path in params!
            return self.record_output(
                workspace_root=params.get("workspace_root"),
                server=str(params["server"]),
                tool=str(params["tool"]),
                arguments=params.get("arguments"),
                output_text=str(params.get("output_text", "")),
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
                shim_secret=params.get("shim_secret"),
                forwarded_arguments=params.get("forwarded_arguments"),
            )
        if method == "list_incidents":
            if not admin_verified:
                raise TransportPermissionError(
                    "Privileged RPC 'list_incidents' requires admin_secret"
                )
            raw_limit = params.get("limit", 50)
            if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
                raise ValueError("limit must be an integer")
            return self.list_incidents(limit=raw_limit, _admin_verified=True)
        if method in ("approve_incident", "approve_allow_token"):
            if not admin_verified:
                raise TransportPermissionError(
                    f"Privileged RPC {method!r} requires admin_secret"
                )
            inc_id = str(params.get("incident_id") or params.get("allow_token") or "")
            return self.approve_incident(inc_id, _admin_verified=True)
        if method == "trust_tool_shape":
            if not admin_verified:
                raise TransportPermissionError(
                    "Privileged RPC 'trust_tool_shape' requires admin_secret"
                )
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
                raise TransportPermissionError(
                    "Privileged RPC 'shutdown' requires admin_secret"
                )
            if self._shutdown_event is not None:
                self._shutdown_event.set()
            return {"ok": True, "pid": os.getpid(), "shutting_down": True}

        raise ValueError(f"Unknown RPC method: {method!r}")

    def _record_replay_nonce(self, req_id: str, nonce: str, ts_val: float) -> bool:
        """Return True if `(req_id, nonce)` is fresh; False if replayed."""
        key = (req_id, nonce)
        with self._replay_lock:
            if key in self._seen_nonces:
                return False
            self._seen_nonces[key] = ts_val
            while len(self._seen_nonces) > MAX_SEEN_NONCES:
                self._seen_nonces.popitem(last=False)
            return True

    def _process_tcp_hello_line(self, raw_line: bytes) -> tuple[bytes, bool]:
        """Process 1-RTT TCP server-first challenge line, returning `(reply_bytes, ok)`."""
        try:
            obj = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            err = {
                "ok": False,
                "request_id": "",
                "code": "UNAUTHORIZED",
                "error": "Malformed TCP hello frame",
            }
            return (json.dumps(err, separators=(",", ":")) + "\n").encode("ascii"), False

        if not isinstance(obj, dict):
            err = {
                "ok": False,
                "request_id": "",
                "code": "UNAUTHORIZED",
                "error": "TCP hello must be a JSON object",
            }
            return (json.dumps(err, separators=(",", ":")) + "\n").encode("ascii"), False

        hello_nonce = obj.get("hello_nonce")
        if (
            "method" in obj
            or not _is_ascii_credential(hello_nonce, min_len=16, max_len=128)
        ):
            req_id = obj.get("request_id")
            err = {
                "ok": False,
                "request_id": req_id if isinstance(req_id, str) and len(req_id) <= 128 else "",
                "code": "UNAUTHORIZED",
                "error": "TCP connection requires server hello handshake and valid client_token",
            }
            return (
                json.dumps(err, ensure_ascii=True, separators=(",", ":")) + "\n"
            ).encode("ascii"), False

        server_nonce = secrets.token_hex(16)
        srv_hello_mac = compute_hello_hmac(
            self._client_token, str(hello_nonce), server_nonce
        )
        reply = {
            "server_nonce": server_nonce,
            "server_hmac": srv_hello_mac,
        }
        return (json.dumps(reply, separators=(",", ":")) + "\n").encode("ascii"), True

    def _process_raw_frame(self, raw_line: bytes) -> tuple[bytes, bool]:
        """Validate HMACs + replay window, dispatch RPC, and return `(response_line_bytes, close_after)`."""
        try:
            req = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            err = {
                "ok": False,
                "request_id": "",
                "code": "UNAUTHORIZED",
                "error": "Malformed JSON request",
            }
            return (json.dumps(err, separators=(",", ":")) + "\n").encode("ascii"), True

        if not isinstance(req, dict):
            err = {
                "ok": False,
                "request_id": "",
                "code": "UNAUTHORIZED",
                "error": "Request root must be an object",
            }
            return (json.dumps(err, separators=(",", ":")) + "\n").encode("ascii"), True

        req_id = req.get("request_id")
        nonce = req.get("client_nonce")
        method = req.get("method")
        params = req.get("params")
        raw_ts = req.get("timestamp")
        client_mac = req.get("client_hmac")

        if (
            not _is_ascii_credential(req_id, min_len=1, max_len=128)
            or not _is_ascii_credential(nonce, min_len=16, max_len=128)
            or not _is_ascii_credential(method, min_len=1, max_len=64)
            or not isinstance(params, dict)
            or isinstance(raw_ts, bool)
            or not isinstance(raw_ts, int)
            or not isinstance(client_mac, str)
        ):
            safe_req_id = (
                req_id if _is_ascii_credential(req_id, min_len=1, max_len=128) else ""
            )
            err = {
                "ok": False,
                "request_id": safe_req_id,
                "code": "UNAUTHORIZED",
                "error": "Missing or invalid client_token authentication fields",
            }
            return (
                json.dumps(err, ensure_ascii=True, separators=(",", ":")) + "\n"
            ).encode("ascii"), True

        now_ts = time.time()
        if abs(now_ts - float(raw_ts)) > REPLAY_WINDOW_SECONDS:
            err = {
                "ok": False,
                "request_id": req_id,
                "code": "UNAUTHORIZED",
                "error": "Stale or out-of-window RPC timestamp",
            }
            return (
                json.dumps(err, ensure_ascii=True, separators=(",", ":")) + "\n"
            ).encode("ascii"), True

        try:
            params_sha = canonical_json_sha256(params)
        except (TypeError, ValueError):
            err = {
                "ok": False,
                "request_id": req_id,
                "code": "UNAUTHORIZED",
                "error": "Invalid non-canonical JSON params",
            }
            return (
                json.dumps(err, ensure_ascii=True, separators=(",", ":")) + "\n"
            ).encode("ascii"), True

        expected_client_mac = compute_client_hmac(
            self._client_token,
            nonce,
            req_id,
            method,
            params_sha,
            timestamp=raw_ts,
        )
        if not constant_time_ascii_equals(client_mac, expected_client_mac):
            err = {
                "ok": False,
                "request_id": req_id,
                "code": "UNAUTHORIZED",
                "error": "Invalid client_token HMAC",
            }
            return (
                json.dumps(err, ensure_ascii=True, separators=(",", ":")) + "\n"
            ).encode("ascii"), True

        if not self._record_replay_nonce(req_id, nonce, float(raw_ts)):
            err = {
                "ok": False,
                "request_id": req_id,
                "code": "UNAUTHORIZED",
                "error": "Replayed RPC request frame rejected",
            }
            return (
                json.dumps(err, ensure_ascii=True, separators=(",", ":")) + "\n"
            ).encode("ascii"), True

        admin_verified = False
        if method in PRIVILEGED_METHODS:
            admin_mac = req.get("admin_hmac")
            expected_admin_mac = compute_admin_hmac(
                self._admin_secret,
                nonce,
                req_id,
                method,
                params_sha,
                timestamp=raw_ts,
            )
            if constant_time_ascii_equals(admin_mac, expected_admin_mac):
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
                    json.dumps(out_err, ensure_ascii=True, separators=(",", ":")) + "\n"
                ).encode("ascii"), False

        try:
            result = self._dispatch_rpc(method, params, admin_verified=admin_verified)
        except TransportPermissionError as exc:
            err_body = {"code": "FORBIDDEN", "error": sanitize_error_message(exc)}
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
                json.dumps(out_err, ensure_ascii=True, separators=(",", ":")) + "\n"
            ).encode("ascii"), False
        except (ValueError, TypeError, KeyError) as exc:
            err_body = {"code": "BAD_REQUEST", "error": sanitize_error_message(exc)}
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
                json.dumps(out_err, ensure_ascii=True, separators=(",", ":")) + "\n"
            ).encode("ascii"), False
        except Exception:
            err_body = {"code": "INTERNAL_ERROR", "error": "Internal daemon RPC error"}
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
                json.dumps(out_err, ensure_ascii=True, separators=(",", ":")) + "\n"
            ).encode("ascii"), False

        safe_result = _sanitize_surrogates_in_obj(result)
        srv_mac = compute_server_hmac(
            self._client_token,
            nonce,
            req_id,
            True,
            canonical_json_sha256(safe_result),
        )
        out_ok = {
            "ok": True,
            "request_id": req_id,
            "result": safe_result,
            "server_hmac": srv_mac,
        }
        close_after = method == "shutdown"
        return (
            json.dumps(out_ok, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("ascii"), close_after

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        conn_sem: asyncio.Semaphore,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._active_connections += 1
        self._last_activity = time.monotonic()
        try:
            async with conn_sem:
                loop = asyncio.get_running_loop()
                client_authenticated = False

                if self.config.mode == "tcp":
                    try:
                        hello_line = await asyncio.wait_for(
                            reader.readline(), timeout=PRE_AUTH_TIMEOUT_SECONDS
                        )
                    except (
                        asyncio.TimeoutError,
                        TimeoutError,
                        asyncio.LimitOverrunError,
                        ValueError,
                    ):
                        return
                    if not hello_line:
                        return
                    hello_reply, hello_ok = self._process_tcp_hello_line(hello_line)
                    writer.write(hello_reply)
                    await writer.drain()
                    self._last_activity = time.monotonic()
                    if not hello_ok:
                        return

                while not reader.at_eof():
                    read_timeout = (
                        30.0 if client_authenticated else PRE_AUTH_TIMEOUT_SECONDS
                    )
                    try:
                        raw_line = await asyncio.wait_for(
                            reader.readline(), timeout=read_timeout
                        )
                    except (
                        asyncio.TimeoutError,
                        TimeoutError,
                        asyncio.LimitOverrunError,
                        ValueError,
                    ):
                        break
                    if not raw_line:
                        break

                    self._last_activity = time.monotonic()
                    resp_bytes, close_after = await loop.run_in_executor(
                        executor, self._process_raw_frame, raw_line
                    )
                    if not close_after:
                        client_authenticated = True
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

    def _verify_no_live_daemon_before_bind(self) -> None:
        """Check liveness before unlinking an existing socket or overwriting `addr.json` (`Important 2`)."""
        if self.config.addr_file.exists() and not self.config.addr_file.is_symlink():
            try:
                cur = json.loads(self.config.addr_file.read_text(encoding="utf-8"))
            except Exception:
                cur = None
            if isinstance(cur, dict):
                other_pid = int(cur.get("pid", 0) or 0)
                if other_pid > 0 and other_pid != os.getpid() and is_pid_alive(other_pid):
                    if cur.get("mode") == "tcp":
                        host = cur.get("host")
                        port = cur.get("port")
                        if host == "127.0.0.1" and isinstance(port, int) and 1 <= port <= 65535:
                            try:
                                with socket.create_connection((host, port), timeout=0.5):
                                    raise TransportError(
                                        f"Live DiodeMCP TCP daemon (pid={other_pid}) is already bound"
                                    )
                            except (ConnectionRefusedError, OSError):
                                pass

        if self.config.mode == "unix" and self.config.socket_path.exists():
            if self.config.socket_path.is_symlink():
                raise TransportAuthError(
                    f"Refusing symlinked Unix socket path: {self.config.socket_path}"
                )
            st = self.config.socket_path.lstat()
            if not stat.S_ISSOCK(st.st_mode):
                raise TransportAuthError(
                    f"Refusing to unlink non-socket file at {self.config.socket_path}"
                )
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.settimeout(0.5)
                with unix_path_handle(self.config.socket_path) as target:
                    probe.connect(target)
            except (FileNotFoundError, ConnectionRefusedError, OSError):
                pass
            else:
                raise TransportError(
                    f"Refusing to clobber active Unix daemon socket at {self.config.socket_path}"
                )
            finally:
                probe.close()

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
        conn_sem = asyncio.Semaphore(MAX_CONCURRENT_CONNECTIONS)
        executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="diodemcp-rpc")
        loop = asyncio.get_running_loop()

        if threading.current_thread() is threading.main_thread():
            for sig_name in ("SIGINT", "SIGTERM"):
                sig = getattr(signal, sig_name, None)
                if sig is not None:
                    try:
                        loop.add_signal_handler(sig, self._shutdown_event.set)
                    except (NotImplementedError, RuntimeError):
                        pass

        async def _client_cb(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await self._handle_client(
                reader, writer, conn_sem=conn_sem, executor=executor
            )

        ensure_private_dir(self.config.lock_file.parent)
        bind_lock = FileLock(str(self.config.state_dir / "daemon.bind.lock"), timeout=5.0)

        server: asyncio.AbstractServer
        try:
            try:
                with bind_lock:
                    self._verify_no_live_daemon_before_bind()
                    if self.config.mode == "unix":
                        if not hasattr(socket, "AF_UNIX"):
                            raise TransportError(
                                "AF_UNIX sockets are not supported on this platform"
                            )
                        ensure_private_dir(self.config.socket_path.parent)
                        try:
                            self.config.socket_path.unlink(missing_ok=True)
                        except OSError:
                            pass

                        unix_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        try:
                            with _UMASK_LOCK:
                                prev_umask = os.umask(0o077)
                                try:
                                    with unix_path_handle(
                                        self.config.socket_path
                                    ) as bind_target:
                                        unix_sock.bind(bind_target)
                                finally:
                                    os.umask(prev_umask)
                            if os.name == "posix":
                                os.chmod(self.config.socket_path, 0o600)
                            self._bound_socket_inode = (
                                self.config.socket_path.lstat().st_ino
                            )
                            unix_sock.setblocking(False)
                            server = await asyncio.start_unix_server(
                                _client_cb,
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
                            "protocol_version": 2,
                        }
                    else:
                        server = await asyncio.start_server(
                            _client_cb,
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
                            "protocol_version": 2,
                        }

                    self._last_activity = time.monotonic()
                    atomic_write_private_json(self.config.addr_file, addr_payload)
            except FileLockTimeout as exc:
                raise TransportError("Timed out acquiring daemon bind lock") from exc
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise

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
            executor.shutdown(wait=True, cancel_futures=False)


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
        shim_secret: str | None = None,
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
        self._shim_secret: str = (
            shim_secret if shim_secret is not None else secrets.token_hex(32)
        )
        self._popup_binding_sha256: str = compute_popup_binding_sha256(self._shim_secret)

    @property
    def shim_secret(self) -> str:
        """Per-shim in-memory secret used exclusively to authenticate `resolve_elicit`."""
        return self._shim_secret

    @property
    def popup_binding_sha256(self) -> str:
        """One-way `sha256(shim_secret)` hex digest passed in `evaluate_call` for popup binding."""
        return self._popup_binding_sha256

    def _call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        admin_secret: str | None = None,
        auto_spawn: bool | None = None,
    ) -> dict[str, Any]:
        return send_rpc_sync(
            self.config,
            method,
            params,
            admin_secret=admin_secret,
            auto_spawn=self.auto_spawn if auto_spawn is None else auto_spawn,
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
        auto_spawn: bool | None = None,
    ) -> dict[str, Any]:
        return await send_rpc_async(
            self.config,
            method,
            params,
            admin_secret=admin_secret,
            auto_spawn=self.auto_spawn if auto_spawn is None else auto_spawn,
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
        popup_binding_sha256: str | None = None,
    ) -> dict[str, Any]:
        del config_path  # Ignored: server loads policy exclusively from state_dir/config.toml
        eff_binding = (
            (popup_binding_sha256 or self._popup_binding_sha256)
            if client_supports_popup
            else popup_binding_sha256
        )
        params: dict[str, Any] = {
            "workspace_root": _normalize_client_workspace(workspace_root),
            "server": server,
            "tool": tool,
            "arguments": dict(arguments or {}),
            "client_supports_popup": bool(client_supports_popup),
            "client_kind": client_kind,
            "permission_mode": permission_mode,
            "popup_binding_sha256": eff_binding,
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
        popup_binding_sha256: str | None = None,
    ) -> dict[str, Any]:
        del config_path
        eff_binding = (
            (popup_binding_sha256 or self._popup_binding_sha256)
            if client_supports_popup
            else popup_binding_sha256
        )
        params: dict[str, Any] = {
            "workspace_root": _normalize_client_workspace(workspace_root),
            "server": server,
            "tool": tool,
            "arguments": dict(arguments or {}),
            "client_supports_popup": bool(client_supports_popup),
            "client_kind": client_kind,
            "permission_mode": permission_mode,
            "popup_binding_sha256": eff_binding,
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
        del config_path
        params: dict[str, Any] = {
            "workspace_root": _normalize_client_workspace(workspace_root),
            "server": server,
            "tool": tool,
            "arguments": dict(arguments or {}),
            "output_text": output_text,
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
        del config_path
        params: dict[str, Any] = {
            "workspace_root": _normalize_client_workspace(workspace_root),
            "server": server,
            "tool": tool,
            "arguments": dict(arguments or {}),
            "output_text": output_text,
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
                "workspace_root": _normalize_client_workspace(workspace_root),
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
                "workspace_root": _normalize_client_workspace(workspace_root),
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
                "workspace_root": _normalize_client_workspace(workspace_root),
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
                "workspace_root": _normalize_client_workspace(workspace_root),
                "target_path": target_path,
                "written_content": written_content,
            },
        )

    def resolve_elicit(
        self,
        incident_id: str,
        choice: Literal["allow_once", "redact", "block", "trust_shape"],
        *,
        shim_secret: str | None = _UNSET_SECRET,
        forwarded_arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        eff_secret = self._shim_secret if shim_secret is _UNSET_SECRET else shim_secret
        return self._call(
            "resolve_elicit",
            {
                "incident_id": incident_id,
                "choice": choice,
                "shim_secret": eff_secret,
                "forwarded_arguments": forwarded_arguments,
            },
        )

    async def aresolve_elicit(
        self,
        incident_id: str,
        choice: Literal["allow_once", "redact", "block", "trust_shape"],
        *,
        shim_secret: str | None = _UNSET_SECRET,
        forwarded_arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        eff_secret = self._shim_secret if shim_secret is _UNSET_SECRET else shim_secret
        return await self._acall(
            "resolve_elicit",
            {
                "incident_id": incident_id,
                "choice": choice,
                "shim_secret": eff_secret,
                "forwarded_arguments": forwarded_arguments,
            },
        )

    def list_incidents(
        self,
        *,
        limit: int = 50,
        admin_secret: str | None = None,
    ) -> list[dict[str, Any]]:
        res = self._call(
            "list_incidents",
            {"limit": int(limit)},
            admin_secret=admin_secret,
        )
        items = res.get("incidents", [])
        return list(items) if isinstance(items, list) else []

    async def alist_incidents(
        self,
        *,
        limit: int = 50,
        admin_secret: str | None = None,
    ) -> list[dict[str, Any]]:
        res = await self._acall(
            "list_incidents",
            {"limit": int(limit)},
            admin_secret=admin_secret,
        )
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
                "workspace_root": _normalize_client_workspace(workspace_root),
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
                "workspace_root": _normalize_client_workspace(workspace_root),
                "server": server,
                "tool": tool,
                "arg_keys": [str(k) for k in arg_keys],
            },
            admin_secret=admin_secret,
        )

    def shutdown(self, *, admin_secret: str | None = None) -> dict[str, Any]:
        return self._call("shutdown", {}, admin_secret=admin_secret, auto_spawn=False)

    async def ashutdown(self, *, admin_secret: str | None = None) -> dict[str, Any]:
        return await self._acall(
            "shutdown", {}, admin_secret=admin_secret, auto_spawn=False
        )


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
        shim_secret: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._sync = EngineClient(
            config,
            auto_spawn=auto_spawn,
            idle_timeout_s=idle_timeout_s,
            bundle_dir=bundle_dir,
            timeout_s=timeout_s,
            shim_secret=shim_secret,
            session_id=session_id,
        )

    @property
    def config(self) -> TransportConfig:
        return self._sync.config

    @property
    def session_id(self) -> str:
        return self._sync.session_id

    @property
    def shim_secret(self) -> str:
        return self._sync.shim_secret

    @property
    def popup_binding_sha256(self) -> str:
        return self._sync.popup_binding_sha256

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
        popup_binding_sha256: str | None = None,
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
            popup_binding_sha256=popup_binding_sha256,
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
        shim_secret: str | None = _UNSET_SECRET,
        forwarded_arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._sync.aresolve_elicit(
            incident_id,
            choice,
            shim_secret=shim_secret,
            forwarded_arguments=forwarded_arguments,
        )

    async def list_incidents(
        self,
        *,
        limit: int = 50,
        admin_secret: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._sync.alist_incidents(limit=limit, admin_secret=admin_secret)

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
