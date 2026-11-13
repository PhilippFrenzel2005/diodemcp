"""Lightweight sync and async daemon IPC clients with zero ML/ORT/tokenizer imports."""

from __future__ import annotations

from collections.abc import Iterable
import math
import os
from pathlib import Path
import secrets
from typing import Any, Literal

from diodemcp.transport import (
    TransportConfig,
    compute_popup_binding_sha256,
    default_transport,
    send_rpc_async,
    send_rpc_sync,
    spawn_daemon_if_needed,
)
from diodemcp.types import CapabilityFlag

DEFAULT_IDLE_TIMEOUT: float = 1800.0
_UNSET_SECRET: Any = object()


def _normalize_client_workspace(workspace_root: Path | str | None) -> str:
    """Validate and normalize a client-supplied workspace path."""
    if workspace_root is None:
        return os.path.realpath(os.getcwd())
    if not isinstance(workspace_root, (str, os.PathLike)):
        raise ValueError("workspace_root must be a path string or Path")
    raw = os.fspath(workspace_root)
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise ValueError("workspace_root must be a non-empty path")
    return os.path.abspath(os.path.expanduser(raw.strip()))


def ensure_daemon_running(
    config: TransportConfig | None = None,
    *,
    idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT,
    bundle_dir: Path | str | None = None,
    startup_timeout_s: float = 12.0,
) -> dict[str, Any]:
    """Ensure the DiodeMCP background daemon is running and return its address descriptor."""
    cfg = config if config is not None else default_transport()
    return spawn_daemon_if_needed(
        cfg,
        idle_timeout_s=float(idle_timeout_s),
        bundle_dir=bundle_dir,
        startup_timeout_s=startup_timeout_s,
    )


def _validate_positive_ttl(ttl_seconds: float | None) -> float | None:
    if ttl_seconds is None:
        return None
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
        raise ValueError("ttl_seconds must be a positive finite number")
    raw_ttl = float(ttl_seconds)
    if not math.isfinite(raw_ttl) or raw_ttl <= 0.0:
        raise ValueError("ttl_seconds must be a positive finite number")
    return raw_ttl


class EngineClient:
    """Synchronous (plus `.a*` async) IPC client for `DiodeEngineServer` (`hooks.py`, `cli.py`, tests)."""

    def __init__(
        self,
        config: TransportConfig | None = None,
        *,
        auto_spawn: bool = True,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT,
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

    def _build_evaluate_params(
        self,
        workspace_root: Path | str | None,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None,
        *,
        client_supports_popup: bool,
        client_kind: Literal["mcp_shim", "ide_hook", "headless"],
        permission_mode: str | None,
        popup_binding_sha256: str | None,
    ) -> dict[str, Any]:
        if popup_binding_sha256 is _UNSET_SECRET:
            eff_binding = self._popup_binding_sha256 if client_supports_popup else None
        else:
            eff_binding = popup_binding_sha256
        return {
            "workspace_root": _normalize_client_workspace(workspace_root),
            "server": server,
            "tool": tool,
            "arguments": dict(arguments or {}),
            "client_supports_popup": bool(client_supports_popup),
            "client_kind": client_kind,
            "permission_mode": permission_mode,
            "popup_binding_sha256": eff_binding,
        }

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
        popup_binding_sha256: str | None = _UNSET_SECRET,
    ) -> dict[str, Any]:
        del config_path  # Ignored: server loads policy exclusively from state_dir/config.toml
        params = self._build_evaluate_params(
            workspace_root,
            server,
            tool,
            arguments,
            client_supports_popup=client_supports_popup,
            client_kind=client_kind,
            permission_mode=permission_mode,
            popup_binding_sha256=popup_binding_sha256,
        )
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
        popup_binding_sha256: str | None = _UNSET_SECRET,
    ) -> dict[str, Any]:
        del config_path
        params = self._build_evaluate_params(
            workspace_root,
            server,
            tool,
            arguments,
            client_supports_popup=client_supports_popup,
            client_kind=client_kind,
            permission_mode=permission_mode,
            popup_binding_sha256=popup_binding_sha256,
        )
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
        workspace_root: Path | str | None = None,
        admin_secret: str | None = None,
    ) -> list[dict[str, Any]]:
        rpc_params: dict[str, Any] = {"limit": int(limit)}
        if workspace_root is not None:
            rpc_params["workspace_root"] = _normalize_client_workspace(workspace_root)
        res = self._call(
            "list_incidents",
            rpc_params,
            admin_secret=admin_secret,
        )
        items = res.get("incidents", [])
        return list(items) if isinstance(items, list) else []

    async def alist_incidents(
        self,
        *,
        limit: int = 50,
        workspace_root: Path | str | None = None,
        admin_secret: str | None = None,
    ) -> list[dict[str, Any]]:
        rpc_params: dict[str, Any] = {"limit": int(limit)}
        if workspace_root is not None:
            rpc_params["workspace_root"] = _normalize_client_workspace(workspace_root)
        res = await self._acall(
            "list_incidents",
            rpc_params,
            admin_secret=admin_secret,
        )
        items = res.get("incidents", [])
        return list(items) if isinstance(items, list) else []

    def approve_incident(
        self,
        incident_id: str,
        *,
        ttl_seconds: float | None = None,
        admin_secret: str | None = None,
    ) -> dict[str, Any]:
        rpc_params: dict[str, Any] = {"incident_id": incident_id}
        raw_ttl = _validate_positive_ttl(ttl_seconds)
        if raw_ttl is not None:
            rpc_params["ttl_seconds"] = raw_ttl
        return self._call(
            "approve_incident",
            rpc_params,
            admin_secret=admin_secret,
        )

    async def aapprove_incident(
        self,
        incident_id: str,
        *,
        ttl_seconds: float | None = None,
        admin_secret: str | None = None,
    ) -> dict[str, Any]:
        rpc_params: dict[str, Any] = {"incident_id": incident_id}
        raw_ttl = _validate_positive_ttl(ttl_seconds)
        if raw_ttl is not None:
            rpc_params["ttl_seconds"] = raw_ttl
        return await self._acall(
            "approve_incident",
            rpc_params,
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
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT,
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
    def idle_timeout_s(self) -> float:
        return self._sync.idle_timeout_s

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
        popup_binding_sha256: str | None = _UNSET_SECRET,
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
        workspace_root: Path | str | None = None,
        admin_secret: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._sync.alist_incidents(
            limit=limit,
            workspace_root=workspace_root,
            admin_secret=admin_secret,
        )

    async def approve_incident(
        self,
        incident_id: str,
        *,
        ttl_seconds: float | None = None,
        admin_secret: str | None = None,
    ) -> dict[str, Any]:
        return await self._sync.aapprove_incident(
            incident_id,
            ttl_seconds=ttl_seconds,
            admin_secret=admin_secret,
        )

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


__all__ = [
    "AsyncEngineClient",
    "DEFAULT_IDLE_TIMEOUT",
    "EngineClient",
    "_UNSET_SECRET",
    "_normalize_client_workspace",
    "ensure_daemon_running",
]
