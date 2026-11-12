"""Hardened MCP `stdio` and Streamable-HTTP bridge shims with inline `elicitation/create` (`Task B6`).

Implements Rulings `R4`, `R7`, `C4`, `C5`, `I1`, `I3`, `I4`, `I6`, `I13`, `I14`:
1. **Pristine `stdout` FD Discipline (`I14`, `M10`):** Duplicates the real `stdout` fd (`os.dup(1)`)
   exclusively for JSON-RPC framing and redirects fd `1` + `sys.stdout` to `sys.stderr` at shim
   entry (restoring cleanly on exit). Non-JSON debug lines from a noisy child MCP server's `stdout`
   are routed to `sys.stderr`. Enforces `MAX_JSONRPC_LINE_BYTES = 64 * 1024 * 1024` (`64 MiB`)
   without crashing with `asyncio.LimitOverrunError`.
2. **Single Async Reader Loop + Disjoint `"diodemcp-<uuid4>"` Elicitation IDs (`I14`, `I3`):**
   Routes host `stdin` through a single async reader loop dispatching responses to pending
   `"diodemcp-<uuid4>"` popup futures, `notifications/cancelled` aborts, and concurrent client
   request tasks. Upstream server->client requests with `"diodemcp-"` IDs are remapped and
   upstream `elicitation/create` messages are prefixed with `[from server <name>]`.
3. **Capability Tracking & Headless Fail-Closed (`C4`):** Tracks `capabilities.elicitation` on
   `initialize` (`dict` required). Headless clients get an immediate fail-closed MCP tool error
   referencing `incident_id` and `diodemcp audit` / `diodemcp allow <incident_id>`, never exposing
   any `allow_token` or `shim_secret`.
4. **Hardened `elicitation/create` Popup Semantics (`I14`, `I13`):** Places `"block"` FIRST with
   `default="block"`; treats `"decline"`, `"cancel"`, timeout, error, or cancellation as `"block"`;
   and on `"redact"` re-evaluates `redacted_arguments` once (`client_supports_popup=False`) before
   forwarding so a second unredacted secret or self-tamper argument cannot slip through.
5. **Synchronous `ack`-Before-Forward (`I6`, `I1`):** Awaits `engine_client.record_output(...)`
   before writing `tools/call`, `resources/read`, or `prompts/get` responses to host `stdout`, and
   screens `tools/list` descriptions for prompt injection and rug-pulls.
6. **Streamable-HTTP Upstream Bridge (`I14`, `I4`):** Enforces `https://` for non-loopback hosts
   (`http://` allowed only for `127.0.0.1` / `localhost` / `[::1]`), blocks cross-origin redirect
   credential leaks, preserves custom `headers` (`Authorization`, etc.), tracks `Mcp-Session-Id`,
   and parses both `application/json` and `text/event-stream` (SSE) responses off the event loop.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import ssl
import sys
import threading
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import uuid

from diodemcp.engine import AsyncEngineClient
from diodemcp.transport import TransportError, default_transport, sanitize_error_message
from diodemcp.types import CapabilityFlag

MAX_JSONRPC_LINE_BYTES: int = 64 * 1024 * 1024  # 64 MiB frame ceiling (I14)
DEFAULT_POPUP_TIMEOUT_SECONDS: float = 120.0
ELICIT_ID_PREFIX: str = "diodemcp-"
_READ_CHUNK_SIZE: int = 65536

CHOICE_LABELS: dict[str, str] = {
    "block": "Block (Recommended)",
    "redact": "Redact & Send",
    "allow_once": "Allow Once",
    "trust_shape": "Trust Tool+Arg Shape for Session",
}

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

_PARTIAL_JSON_ID_RE = re.compile(
    rb'"id"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+|null)'
)

# Opus I1 / I6: Suspicious prompt-injection and secret-exfiltration patterns in `tools/list` metadata.
_TOOL_POISONING_RE = re.compile(
    r"(?:"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|system)\s+instructions"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|above)"
    r"|<system>|</system>|<\|intent\|>|<\|missing_intent\|>|<\|im_start\|>|\[inst\]"
    r"|\b(?:important|critical|urgent)\s*:\s*(?:ignore|before|first|always|do\s+not|never)\b"
    r"|\b(?:do\s+not|don't|never)\s+(?:tell|inform|mention|show|reveal)\s+(?:to\s+)?(?:the\s+)?user\b"
    r"|\b(?:before|prior\s+to)\s+(?:calling|using|invoking)\s+this\s+tool\b"
    r"|\b(?:read|cat|open|include|exfiltrate|upload|send|steal)\b[^\n]{0,160}"
    r"(?:\.env\b|id_rsa|id_ed25519|\.ssh|\.aws|admin\.secret|\.diodemcp|credentials|/etc/passwd)"
    r"|\b(?:send|post|upload|exfiltrate|forward|curl|wget)\b[^\n]{0,160}https?://[^\s\"'<>]+"
    r")",
    re.IGNORECASE,
)


def validate_streamable_http_url(url: str) -> str:
    """Validate Streamable-HTTP upstream URL (`http://` allowed ONLY for loopback; `https://` for remote)."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Streamable-HTTP URL must be a non-empty string")
    cleaned = url.strip()
    if "\x00" in cleaned or "\r" in cleaned or "\n" in cleaned:
        raise ValueError("Streamable-HTTP URL contains invalid control characters")

    parsed = urllib.parse.urlsplit(cleaned)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported URL scheme {parsed.scheme!r}; expected 'https://' (or 'http://' for loopback)"
        )
    if not parsed.hostname:
        raise ValueError("Streamable-HTTP URL must include a valid hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Streamable-HTTP URL must not embed userinfo credentials")

    host_lower = parsed.hostname.lower()
    if scheme == "http" and host_lower not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"Insecure cleartext http:// URL is only permitted for loopback "
            f"(127.0.0.1 / localhost / [::1]); use https:// for remote host {parsed.hostname!r}"
        )
    return cleaned


def _extract_partial_jsonrpc_id(head_bytes: bytes) -> Any:
    """Best-effort extraction of JSON-RPC `id` from the prefix of an oversized frame."""
    m = _PARTIAL_JSON_ID_RE.search(head_bytes)
    if m is None:
        return None
    raw_token = m.group(1)
    try:
        return json.loads(raw_token.decode("utf-8", errors="replace"))
    except Exception:
        return None


class _BoundedJsonLineReader:
    """Chunked line reader enforcing `MAX_JSONRPC_LINE_BYTES` without raising `LimitOverrunError`."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        *,
        max_line_bytes: int | None = None,
    ) -> None:
        self._reader = reader
        self._explicit_max_bytes = max_line_bytes
        self._buf = bytearray()
        self._discarding_oversized = False
        self._oversized_head = b""

    def _current_limit(self) -> int:
        if self._explicit_max_bytes is not None:
            return self._explicit_max_bytes
        return int(MAX_JSONRPC_LINE_BYTES)

    async def read_line(self) -> tuple[bytes | None, bool]:
        """Read next newline-delimited frame.

        Returns:
            `(None, False)` on clean EOF with empty buffer.
            `(line_bytes, False)` for a valid frame `<= max_line_bytes`.
            `(head_sample_bytes, True)` when a frame exceeded `max_line_bytes` and was drained to `\\n`.
        """
        limit = self._current_limit()
        while True:
            nl_idx = self._buf.find(b"\n")
            if nl_idx != -1:
                if self._discarding_oversized:
                    del self._buf[: nl_idx + 1]
                    head = self._oversized_head
                    self._oversized_head = b""
                    self._discarding_oversized = False
                    return head, True

                line = bytes(self._buf[:nl_idx])
                del self._buf[: nl_idx + 1]
                if line.endswith(b"\r"):
                    line = line[:-1]
                if len(line) > limit:
                    return line[:4096], True
                return line, False

            if not self._discarding_oversized and len(self._buf) > limit:
                self._oversized_head = bytes(self._buf[:4096])
                self._buf.clear()
                self._discarding_oversized = True

            try:
                chunk = await self._reader.read(_READ_CHUNK_SIZE)
            except (asyncio.LimitOverrunError, ValueError):
                chunk = b""

            if not chunk:
                if self._discarding_oversized:
                    head = self._oversized_head
                    self._oversized_head = b""
                    self._discarding_oversized = False
                    self._buf.clear()
                    return head, True
                if self._buf:
                    tail = bytes(self._buf)
                    self._buf.clear()
                    if tail.endswith(b"\r"):
                        tail = tail[:-1]
                    if len(tail) > limit:
                        return tail[:4096], True
                    return tail, False
                return None, False

            if self._discarding_oversized:
                nl_pos = chunk.find(b"\n")
                if nl_pos != -1:
                    self._buf.extend(chunk[nl_pos:])
            else:
                self._buf.extend(chunk)


def _start_fd_stream_reader(fd: int) -> tuple[asyncio.StreamReader, Callable[[], None]]:
    """Bridge an OS file descriptor (`stdin`) into an `asyncio.StreamReader` via a daemon reader thread."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=MAX_JSONRPC_LINE_BYTES)
    stop_event = threading.Event()
    fd_lock = threading.Lock()
    dup_fd: int | None = os.dup(fd)

    if sys.platform == "win32":
        try:
            import msvcrt

            msvcrt.setmode(dup_fd, os.O_BINARY)
        except Exception:
            pass

    def _close_dup_fd_once() -> None:
        nonlocal dup_fd
        with fd_lock:
            target = dup_fd
            dup_fd = None
        if target is not None:
            try:
                os.close(target)
            except OSError:
                pass

    def _thread_main() -> None:
        try:
            while not stop_event.is_set():
                with fd_lock:
                    cur_fd = dup_fd
                if cur_fd is None:
                    break
                try:
                    data = os.read(cur_fd, _READ_CHUNK_SIZE)
                except OSError:
                    break
                if not data:
                    break
                try:
                    loop.call_soon_threadsafe(reader.feed_data, data)
                except RuntimeError:
                    break
        finally:
            _close_dup_fd_once()
            try:
                loop.call_soon_threadsafe(reader.feed_eof)
            except RuntimeError:
                pass

    worker = threading.Thread(
        target=_thread_main,
        name="diodemcp-stdin-reader",
        daemon=True,
    )
    worker.start()

    def _cleanup() -> None:
        stop_event.set()
        _close_dup_fd_once()

    return reader, _cleanup


@contextmanager
def _protocol_stdout_guard(
    custom_stdout_write: Callable[[bytes], Awaitable[None] | None] | None,
):
    """Enforce `I14` pristine `stdout` FD discipline.

    Duplicates real fd 1 for MCP JSON-RPC writes and redirects fd 1 + `sys.stdout`
    to `sys.stderr` so stray `print()` or C-library writes can never corrupt `stdout`.
    """
    orig_sys_stdout = sys.stdout
    write_lock = threading.Lock()

    if custom_stdout_write is not None:
        sys.stdout = sys.stderr
        try:
            yield custom_stdout_write
        finally:
            sys.stdout = orig_sys_stdout
        return

    protocol_fd = os.dup(1)
    if sys.platform == "win32":
        try:
            import msvcrt

            msvcrt.setmode(protocol_fd, os.O_BINARY)
        except Exception:
            pass

    try:
        os.dup2(2, 1)
    except OSError:
        pass
    sys.stdout = sys.stderr

    async def _fd_write(data: bytes) -> None:
        def _sync_write() -> None:
            with write_lock:
                view = memoryview(data)
                written = 0
                while written < len(view):
                    written += os.write(protocol_fd, view[written:])

        await asyncio.to_thread(_sync_write)

    try:
        yield _fd_write
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        try:
            os.dup2(protocol_fd, 1)
        except OSError:
            pass
        try:
            os.close(protocol_fd)
        except OSError:
            pass
        sys.stdout = orig_sys_stdout


@dataclass
class _InflightClientRequest:
    """Tracks an in-flight client JSON-RPC request for cancellation & popup routing."""

    request_id: Any
    method: str
    cancelled: bool = False
    forwarded_to_upstream: bool = False
    elicit_id: str | None = None
    elicit_future: asyncio.Future[dict[str, Any]] | None = None
    task: asyncio.Task[Any] | None = None


def _extract_content_blocks_text(blocks: Any, out: list[str]) -> None:
    if isinstance(blocks, str):
        if blocks:
            out.append(blocks)
        return
    if isinstance(blocks, Mapping):
        for key in ("text", "data", "blob", "content", "description"):
            val = blocks.get(key)
            if isinstance(val, str) and val:
                out.append(val)
            elif isinstance(val, (Mapping, list, tuple)):
                _extract_content_blocks_text(val, out)
        res = blocks.get("resource")
        if isinstance(res, Mapping):
            _extract_content_blocks_text(res, out)
        return
    if isinstance(blocks, (list, tuple)):
        for item in blocks:
            _extract_content_blocks_text(item, out)


def _extract_mcp_result_text(method: str, result: Mapping[str, Any]) -> str:
    """Extract all text/blob payloads from `tools/call`, `resources/read`, or `prompts/get` results."""
    parts: list[str] = []
    if method == "tools/call":
        _extract_content_blocks_text(result.get("content"), parts)
        if "structuredContent" in result:
            try:
                parts.append(
                    json.dumps(
                        result["structuredContent"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            except Exception:
                pass
        if not parts:
            for k, v in result.items():
                if k not in ("isError", "_meta") and isinstance(v, (str, Mapping, list)):
                    _extract_content_blocks_text(v, parts)
    elif method == "resources/read":
        _extract_content_blocks_text(result.get("contents"), parts)
    elif method == "prompts/get":
        desc = result.get("description")
        if isinstance(desc, str) and desc:
            parts.append(desc)
        for msg in result.get("messages", []) if isinstance(result.get("messages"), list) else []:
            if isinstance(msg, Mapping):
                _extract_content_blocks_text(msg.get("content"), parts)
    return "\n".join(p for p in parts if p)


def _normalize_elicit_choices(
    raw_choices: Sequence[Any],
    *,
    band: str,
    can_redact: bool,
) -> list[str]:
    """Ensure `"block"` is ALWAYS first (`default="block"`) and enforce band/redact invariants (`I14`)."""
    allowed_set = {"block", "redact", "allow_once", "trust_shape"}
    filtered: list[str] = []
    for item in raw_choices:
        if not isinstance(item, str) or item not in allowed_set:
            continue
        if item == "trust_shape" and band == "critical":
            continue
        if item == "redact" and not can_redact:
            continue
        if item not in filtered:
            filtered.append(item)

    ordered: list[str] = ["block"]
    for c in filtered:
        if c != "block":
            ordered.append(c)
    return ordered


def _build_mcp_tool_error_response(
    req_id: Any,
    *,
    banner: str,
    incident_id: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    """Construct a fail-closed MCP tool error response (`isError: True`) without leaking any secret token."""
    lines = [banner.strip() or "Blocked by DiodeMCP."]
    if detail:
        lines.append(detail.strip())
    if incident_id:
        lines.append(f"Incident ID: {incident_id}")
        lines.append(
            f"Approve out-of-band in an interactive terminal via: "
            f"diodemcp allow {incident_id} (or inspect with: diodemcp audit)"
        )
    else:
        lines.append("Inspect in an interactive terminal via: diodemcp audit")

    res_obj: dict[str, Any] = {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "isError": True,
    }
    if incident_id:
        res_obj["incident_id"] = incident_id
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": res_obj,
    }


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block cross-origin or HTTPS->HTTP downgrade redirects so `Authorization` headers never leak."""

    def __init__(self, origin_scheme: str, origin_netloc: str) -> None:
        super().__init__()
        self._origin_scheme = origin_scheme.lower()
        self._origin_netloc = origin_netloc.lower()

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        new_scheme = (parsed.scheme or "").lower()
        new_netloc = (parsed.netloc or "").lower()
        if new_scheme != self._origin_scheme or new_netloc != self._origin_netloc:
            raise urllib.error.HTTPError(
                newurl,
                code,
                "Blocked cross-origin or scheme-downgrade redirect on Streamable-HTTP MCP bridge",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _parse_sse_jsonrpc_messages(raw_text: str) -> list[dict[str, Any]]:
    """Parse `text/event-stream` SSE payload into JSON-RPC message dicts."""
    messages: list[dict[str, Any]] = []
    data_lines: list[str] = []

    def _flush_event() -> None:
        if not data_lines:
            return
        joined = "\n".join(data_lines).strip()
        data_lines.clear()
        if not joined or joined == "[DONE]":
            return
        try:
            parsed = json.loads(joined)
        except Exception:
            return
        if isinstance(parsed, dict):
            messages.append(parsed)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    messages.append(item)

    for raw_line in raw_text.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            _flush_event()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            val = line[5:]
            if val.startswith(" "):
                val = val[1:]
            data_lines.append(val)
    _flush_event()
    return messages


class _McpShimCore:
    """Shared protocol state machine for `stdio` and Streamable-HTTP MCP shims (`I6`, `I14`, `C4`)."""

    def __init__(
        self,
        *,
        server_name: str,
        state_dir: Path,
        workspace_root: Path | None,
        popup_timeout_s: float,
        stdout_writer: Callable[[bytes], Awaitable[None] | None],
        engine_client: AsyncEngineClient | None = None,
    ) -> None:
        if popup_timeout_s <= 0.0:
            raise ValueError("popup_timeout_s must be > 0")
        self.server_name: str = str(server_name).strip() or "mcp_server"
        self.state_dir: Path = Path(state_dir).expanduser()
        self.workspace_root: Path = (
            Path(os.path.realpath(os.path.expanduser(os.fspath(workspace_root))))
            if workspace_root is not None
            else Path(os.path.realpath(os.getcwd()))
        )
        self.popup_timeout_s: float = float(popup_timeout_s)
        self._stdout_writer = stdout_writer
        self._stdout_lock = asyncio.Lock()

        if engine_client is not None:
            self.engine_client: AsyncEngineClient = engine_client
        else:
            cfg = default_transport(self.state_dir)
            self.engine_client = AsyncEngineClient(cfg)

        self.client_supports_popup: bool = False
        self._pending_elicitations: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._inflight_client_requests: dict[Any, _InflightClientRequest] = {}
        self._client_tasks: set[asyncio.Task[Any]] = set()
        self._tool_description_hashes: dict[str, str] = {}

        # Remap table when an upstream server originates a request with a colliding `diodemcp-` id:
        #   remapped_id -> original_upstream_id
        self._remapped_upstream_server_ids: dict[str, Any] = {}
        # Track upstream `sampling/createMessage` request IDs so client completions are screened (`I1`):
        self._pending_upstream_sampling_ids: dict[Any, dict[str, Any]] = {}

    async def write_stdout_json(self, payload: dict[str, Any]) -> None:
        """Serialize `payload` as a single JSON-RPC line and write to guarded protocol `stdout`."""
        limit = int(MAX_JSONRPC_LINE_BYTES)
        raw = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if len(raw) > limit:
            req_id = payload.get("id")
            fallback = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32603,
                    "message": f"Outbound JSON-RPC frame exceeds {limit} byte limit",
                },
            }
            raw = (
                json.dumps(fallback, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")

        async with self._stdout_lock:
            res = self._stdout_writer(raw)
            if inspect.isawaitable(res):
                await res

    def _inspect_initialize_capabilities(self, params: Any) -> None:
        """Update `client_supports_popup` strictly from `initialize.params.capabilities.elicitation` (`C4`)."""
        if not isinstance(params, Mapping):
            self.client_supports_popup = False
            return
        caps = params.get("capabilities")
        if not isinstance(caps, Mapping):
            self.client_supports_popup = False
            return
        elicit_cap = caps.get("elicitation")
        self.client_supports_popup = isinstance(elicit_cap, Mapping)

    async def _best_effort_resolve_elicit(
        self,
        incident_id: str,
        choice: str,
        *,
        forwarded_arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        try:
            return await self.engine_client.resolve_elicit(
                incident_id,
                choice,  # type: ignore[arg-type]
                forwarded_arguments=forwarded_arguments,
            )
        except Exception:
            return None

    async def _run_elicitation_popup(
        self,
        *,
        decision: Mapping[str, Any],
        tool_name: str,
        tool_args: dict[str, Any],
        inflight: _InflightClientRequest,
    ) -> tuple[bool, dict[str, Any], dict[str, Any] | None]:
        """Send `elicitation/create` (`id="diodemcp-<uuid4>"`), await user choice, and enforce re-eval on `redact`."""
        incident_id = str(decision.get("incident_id") or "")
        banner = str(decision.get("banner") or "🛑 DiodeMCP Security Alert")
        reason = str(decision.get("reason") or banner)
        band = str(decision.get("band") or "critical")
        can_redact = bool(decision.get("can_redact", False))
        redacted_args = decision.get("redacted_arguments")
        raw_choices = decision.get("choices") or ["block"]

        enum_choices = _normalize_elicit_choices(
            raw_choices if isinstance(raw_choices, Sequence) else ["block"],
            band=band,
            can_redact=can_redact,
        )

        elicit_id = f"{ELICIT_ID_PREFIX}{uuid.uuid4()}"
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_elicitations[elicit_id] = fut
        inflight.elicit_id = elicit_id
        inflight.elicit_future = fut

        elicit_request = {
            "jsonrpc": "2.0",
            "id": elicit_id,
            "method": "elicitation/create",
            "params": {
                "message": f"{banner}\nIncident ID: {incident_id}",
                "incident_id": incident_id,
                "band": band,
                "choices": enum_choices,
                "requestedSchema": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "title": "DiodeMCP Firewall Decision",
                            "description": reason,
                            "enum": enum_choices,
                            "enumNames": [CHOICE_LABELS.get(c, c) for c in enum_choices],
                            "default": "block",
                        }
                    },
                    "required": ["decision"],
                },
            },
        }

        try:
            if inflight.cancelled:
                await self._best_effort_resolve_elicit(incident_id, "block")
                return (
                    False,
                    tool_args,
                    _build_mcp_tool_error_response(
                        inflight.request_id,
                        banner=banner,
                        incident_id=incident_id,
                        detail="Request was cancelled before popup resolution; failing closed.",
                    ),
                )

            await self.write_stdout_json(elicit_request)
            elicit_resp = await asyncio.wait_for(fut, timeout=self.popup_timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            await self._best_effort_resolve_elicit(incident_id, "block")
            return (
                False,
                tool_args,
                _build_mcp_tool_error_response(
                    inflight.request_id,
                    banner=banner,
                    incident_id=incident_id,
                    detail=f"Popup timed out after {self.popup_timeout_s:.2f}s; failing closed.",
                ),
            )
        except asyncio.CancelledError:
            await self._best_effort_resolve_elicit(incident_id, "block")
            return (
                False,
                tool_args,
                _build_mcp_tool_error_response(
                    inflight.request_id,
                    banner=banner,
                    incident_id=incident_id,
                    detail="Popup cancelled by client; failing closed.",
                ),
            )
        finally:
            self._pending_elicitations.pop(elicit_id, None)
            inflight.elicit_id = None
            inflight.elicit_future = None

        if inflight.cancelled or not isinstance(elicit_resp, Mapping) or "error" in elicit_resp:
            await self._best_effort_resolve_elicit(incident_id, "block")
            return (
                False,
                tool_args,
                _build_mcp_tool_error_response(
                    inflight.request_id,
                    banner=banner,
                    incident_id=incident_id,
                    detail="Popup rejected or errored; failing closed.",
                ),
            )

        res_obj = elicit_resp.get("result")
        if not isinstance(res_obj, Mapping) or res_obj.get("action") != "accept":
            await self._best_effort_resolve_elicit(incident_id, "block")
            return (
                False,
                tool_args,
                _build_mcp_tool_error_response(
                    inflight.request_id,
                    banner=banner,
                    incident_id=incident_id,
                    detail="User declined or cancelled popup; failing closed.",
                ),
            )

        content_obj = res_obj.get("content")
        chosen_raw = (
            (content_obj.get("decision") or content_obj.get("choice"))
            if isinstance(content_obj, Mapping)
            else None
        )
        if not isinstance(chosen_raw, str) or chosen_raw not in enum_choices:
            await self._best_effort_resolve_elicit(incident_id, "block")
            return (
                False,
                tool_args,
                _build_mcp_tool_error_response(
                    inflight.request_id,
                    banner=banner,
                    incident_id=incident_id,
                    detail="Invalid or disallowed elicitation choice; failing closed.",
                ),
            )

        if chosen_raw == "block":
            await self._best_effort_resolve_elicit(incident_id, "block")
            return (
                False,
                tool_args,
                _build_mcp_tool_error_response(
                    inflight.request_id,
                    banner=banner,
                    incident_id=incident_id,
                    detail="Blocked by user via DiodeMCP popup.",
                ),
            )

        if chosen_raw == "redact":
            if not can_redact or not isinstance(redacted_args, dict):
                await self._best_effort_resolve_elicit(incident_id, "block")
                return (
                    False,
                    tool_args,
                    _build_mcp_tool_error_response(
                        inflight.request_id,
                        banner=banner,
                        incident_id=incident_id,
                        detail="Surgical redaction is unavailable for this call; failing closed.",
                    ),
                )
            resolution = await self._best_effort_resolve_elicit(
                incident_id,
                "redact",
            )
            if not isinstance(resolution, Mapping) or not resolution.get("ok"):
                return (
                    False,
                    tool_args,
                    _build_mcp_tool_error_response(
                        inflight.request_id,
                        banner=banner,
                        incident_id=incident_id,
                        detail="Failed to verify popup redaction resolution with engine; failing closed.",
                    ),
                )

            effective_redacted = (
                dict(resolution["redacted_arguments"])
                if isinstance(resolution.get("redacted_arguments"), Mapping)
                else dict(redacted_args)
            )

            # Mandatory single-pass re-evaluation of `redacted_arguments` (I14 / I13):
            try:
                reeval = await self.engine_client.evaluate_call(
                    workspace_root=self.workspace_root,
                    server=self.server_name,
                    tool=tool_name,
                    arguments=effective_redacted,
                    client_kind="mcp_shim",
                    client_supports_popup=False,
                )
            except Exception as exc:
                return (
                    False,
                    tool_args,
                    _build_mcp_tool_error_response(
                        inflight.request_id,
                        banner=banner,
                        incident_id=incident_id,
                        detail=f"Post-redaction re-evaluation failed ({sanitize_error_message(exc)}); failing closed.",
                    ),
                )

            if reeval.get("action") != "allow":
                reeval_inc = str(reeval.get("incident_id") or incident_id)
                reeval_banner = str(reeval.get("banner") or banner)
                return (
                    False,
                    tool_args,
                    _build_mcp_tool_error_response(
                        inflight.request_id,
                        banner=reeval_banner,
                        incident_id=reeval_inc,
                        detail="Redacted arguments still triggered DiodeMCP policy on re-evaluation; failing closed.",
                    ),
                )
            return True, effective_redacted, None

        if chosen_raw == "allow_once":
            resolution = await self._best_effort_resolve_elicit(
                incident_id,
                "allow_once",
                forwarded_arguments=tool_args,
            )
            if not isinstance(resolution, Mapping) or not resolution.get("ok"):
                return (
                    False,
                    tool_args,
                    _build_mcp_tool_error_response(
                        inflight.request_id,
                        banner=banner,
                        incident_id=incident_id,
                        detail="Engine rejected allow_once resolution; failing closed.",
                    ),
                )
            return True, tool_args, None

        if chosen_raw == "trust_shape":
            if band != "warn":
                await self._best_effort_resolve_elicit(incident_id, "block")
                return (
                    False,
                    tool_args,
                    _build_mcp_tool_error_response(
                        inflight.request_id,
                        banner=banner,
                        incident_id=incident_id,
                        detail="trust_shape is never permitted on CRITICAL incidents.",
                    ),
                )
            resolution = await self._best_effort_resolve_elicit(
                incident_id,
                "trust_shape",
                forwarded_arguments=tool_args,
            )
            if not isinstance(resolution, Mapping) or not resolution.get("ok"):
                return (
                    False,
                    tool_args,
                    _build_mcp_tool_error_response(
                        inflight.request_id,
                        banner=banner,
                        incident_id=incident_id,
                        detail="Engine rejected trust_shape resolution; failing closed.",
                    ),
                )
            return True, tool_args, None

        await self._best_effort_resolve_elicit(incident_id, "block")
        return (
            False,
            tool_args,
            _build_mcp_tool_error_response(
                inflight.request_id,
                banner=banner,
                incident_id=incident_id,
            ),
        )

    async def evaluate_and_gate_call(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        inflight: _InflightClientRequest,
    ) -> tuple[bool, dict[str, Any], dict[str, Any] | None]:
        """Evaluate proposed tool call via `AsyncEngineClient` and run popup or headless fail-closed gate."""
        try:
            decision = await self.engine_client.evaluate_call(
                workspace_root=self.workspace_root,
                server=self.server_name,
                tool=tool_name,
                arguments=tool_args,
                client_kind="mcp_shim",
                client_supports_popup=self.client_supports_popup,
            )
        except Exception as exc:
            return (
                False,
                tool_args,
                _build_mcp_tool_error_response(
                    inflight.request_id,
                    banner="🛑 CRITICAL DiodeMCP Alert: Firewall engine unavailable; failing closed.",
                    detail=sanitize_error_message(exc),
                ),
            )

        action = decision.get("action")
        if action == "allow":
            return True, tool_args, None

        if action == "elicit" and self.client_supports_popup:
            return await self._run_elicitation_popup(
                decision=decision,
                tool_name=tool_name,
                tool_args=tool_args,
                inflight=inflight,
            )

        incident_id = decision.get("incident_id")
        banner = str(decision.get("banner") or "🛑 Blocked by DiodeMCP")
        return (
            False,
            tool_args,
            _build_mcp_tool_error_response(
                inflight.request_id,
                banner=banner,
                incident_id=str(incident_id) if incident_id else None,
            ),
        )

    async def inspect_and_record_upstream_response(
        self,
        *,
        method: str,
        request_params: Mapping[str, Any],
        forwarded_args: dict[str, Any],
        response_msg: dict[str, Any],
    ) -> dict[str, Any]:
        """Enforce `I6` synchronous `ack`-before-forward on `tools/call`, `resources/read`, `prompts/get`, and `tools/list`."""
        res_obj = response_msg.get("result")
        if not isinstance(res_obj, Mapping):
            return response_msg

        if method == "tools/list":
            tools_list = res_obj.get("tools")
            if isinstance(tools_list, list):
                suspicious_entries: list[str] = []
                for tool_entry in tools_list:
                    if not isinstance(tool_entry, Mapping):
                        continue
                    t_name = str(tool_entry.get("name") or "")
                    t_desc = str(tool_entry.get("description") or "")
                    try:
                        schema_str = json.dumps(
                            tool_entry.get("inputSchema") or {},
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    except Exception:
                        schema_str = ""
                    combined_meta = f"{t_name}\n{t_desc}\n{schema_str}"
                    digest = hashlib.sha256(
                        combined_meta.encode("utf-8", errors="replace")
                    ).hexdigest()

                    prev_digest = self._tool_description_hashes.get(t_name)
                    self._tool_description_hashes[t_name] = digest
                    is_rug_pull = prev_digest is not None and prev_digest != digest
                    is_poisoned = _TOOL_POISONING_RE.search(combined_meta) is not None

                    if is_poisoned or is_rug_pull:
                        tag = "rug_pull" if is_rug_pull else "poisoned_description"
                        suspicious_entries.append(
                            f"[{self.server_name}:{t_name}:{tag}] {t_desc} {schema_str}".strip()
                        )

                if suspicious_entries:
                    joined_suspicious = "\n".join(suspicious_entries)
                    try:
                        await self.engine_client.record_output(
                            workspace_root=self.workspace_root,
                            server=self.server_name,
                            tool="__tools_list__",
                            arguments={
                                "source_tool": f"{self.server_name}:__tools_list__"
                            },
                            output_text=joined_suspicious,
                            extra_flags=CapabilityFlag.READS_UNTRUSTED,
                        )
                    except Exception as exc:
                        return {
                            "jsonrpc": "2.0",
                            "id": response_msg.get("id"),
                            "error": {
                                "code": -32603,
                                "message": (
                                    "Blocked by DiodeMCP: failed to record poisoned tools/list "
                                    f"taint ({sanitize_error_message(exc)})"
                                ),
                            },
                        }
            return response_msg

        if method in ("tools/call", "resources/read", "prompts/get"):
            extracted_text = _extract_mcp_result_text(method, res_obj)
            if method == "tools/call":
                tool_name = str(request_params.get("name") or "unknown_tool")
                extra_flags = CapabilityFlag.LOCAL_SAFE
                args_for_record = forwarded_args
            elif method == "resources/read":
                uri_val = str(request_params.get("uri") or "")
                tool_name = "read_resource"
                extra_flags = CapabilityFlag.READS_PRIVATE | CapabilityFlag.READS_UNTRUSTED
                args_for_record = {"uri": uri_val, "path": uri_val}
            else:
                prompt_name = str(request_params.get("name") or "prompt")
                tool_name = f"prompt:{prompt_name}"
                extra_flags = CapabilityFlag.READS_UNTRUSTED
                args_for_record = dict(request_params)

            if extracted_text or method == "tools/call":
                try:
                    await self.engine_client.record_output(
                        workspace_root=self.workspace_root,
                        server=self.server_name,
                        tool=tool_name,
                        arguments=args_for_record,
                        output_text=extracted_text,
                        extra_flags=extra_flags,
                    )
                except Exception as exc:
                    if method == "tools/call":
                        return _build_mcp_tool_error_response(
                            response_msg.get("id"),
                            banner="🛑 CRITICAL DiodeMCP Alert: Failed to persist output taint before forwarding; failing closed.",
                            detail=sanitize_error_message(exc),
                        )
                    return {
                        "jsonrpc": "2.0",
                        "id": response_msg.get("id"),
                        "error": {
                            "code": -32603,
                            "message": (
                                f"Blocked by DiodeMCP: failed to persist {method} output taint "
                                f"before forwarding; failing closed ({sanitize_error_message(exc)})"
                            ),
                        },
                    }
        return response_msg

    def handle_cancellation_notification(self, msg: Mapping[str, Any]) -> bool:
        """Process `notifications/cancelled`. Returns True if cancellation should be forwarded upstream."""
        params = msg.get("params")
        if not isinstance(params, Mapping):
            return False
        req_id = params.get("requestId")
        inflight = self._inflight_client_requests.get(req_id)
        if inflight is None:
            return False
        inflight.cancelled = True
        if inflight.elicit_future is not None and not inflight.elicit_future.done():
            inflight.elicit_future.cancel()
        return inflight.forwarded_to_upstream


def _send_http_jsonrpc_sync(
    *,
    url: str,
    static_headers: Mapping[str, str],
    session_id: str | None,
    payload: dict[str, Any],
    timeout_s: float = 30.0,
) -> tuple[int, dict[str, str], bytes]:
    """Execute one Streamable-HTTP POST request synchronously off the event loop."""
    validated_url = validate_streamable_http_url(url)
    parsed = urllib.parse.urlsplit(validated_url)
    body_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    limit = int(MAX_JSONRPC_LINE_BYTES)
    if len(body_bytes) > limit:
        raise ValueError(f"Outbound HTTP JSON-RPC request exceeds {limit} byte limit")

    req_headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    for k, v in static_headers.items():
        if isinstance(k, str) and isinstance(v, str):
            req_headers[k] = v
    if session_id:
        req_headers["Mcp-Session-Id"] = session_id

    handlers: list[urllib.request.BaseHandler] = [
        _SameOriginRedirectHandler(parsed.scheme, parsed.netloc)
    ]
    if parsed.scheme.lower() == "https":
        handlers.append(
            urllib.request.HTTPSHandler(context=ssl.create_default_context())
        )
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(
        validated_url,
        data=body_bytes,
        headers=req_headers,
        method="POST",
    )

    with opener.open(request, timeout=timeout_s) as resp:
        status_code = int(getattr(resp, "status", 200))
        resp_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
        raw_resp = resp.read(limit + 1)
        if len(raw_resp) > limit:
            raise ValueError(
                f"Upstream HTTP response exceeded {limit} byte limit"
            )
        return status_code, resp_headers, raw_resp


async def async_run_stdio_shim(
    server_name: str,
    command: list[str],
    state_dir: Path,
    *,
    workspace_root: Path | None = None,
    popup_timeout_s: float = DEFAULT_POPUP_TIMEOUT_SECONDS,
    stdin_stream: asyncio.StreamReader | None = None,
    stdout_write: Callable[[bytes], Awaitable[None] | None] | None = None,
    engine_client: AsyncEngineClient | None = None,
) -> int:
    """Run hardened MCP `stdio` shim asynchronously (`I6`, `I14`, `C4`)."""
    if not isinstance(command, list) or not command or not all(
        isinstance(arg, str) and arg for arg in command
    ):
        raise ValueError("command must be a non-empty list of strings")

    cleanup_stdin: Callable[[], None] | None = None
    with _protocol_stdout_guard(stdout_write) as guarded_stdout_write:
        core = _McpShimCore(
            server_name=server_name,
            state_dir=Path(state_dir),
            workspace_root=workspace_root,
            popup_timeout_s=popup_timeout_s,
            stdout_writer=guarded_stdout_write,
            engine_client=engine_client,
        )

        child = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
            limit=MAX_JSONRPC_LINE_BYTES,
        )
        if child.stdin is None or child.stdout is None:
            raise TransportError("Failed to open stdio pipes to child MCP server")

        if stdin_stream is None:
            stdin_stream, cleanup_stdin = _start_fd_stream_reader(0)

        child_stdin_lock = asyncio.Lock()
        pending_upstream: dict[Any, asyncio.Future[dict[str, Any]]] = {}

        async def _write_child_stdin(msg: dict[str, Any]) -> None:
            raw = (
                json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            async with child_stdin_lock:
                if child.stdin is not None and not child.stdin.is_closing():
                    child.stdin.write(raw)
                    await child.stdin.drain()

        async def _child_stdout_reader_loop() -> None:
            line_reader = _BoundedJsonLineReader(child.stdout)  # type: ignore[arg-type]
            try:
                while True:
                    raw_line, was_oversized = await line_reader.read_line()
                    if raw_line is None:
                        break
                    if was_oversized:
                        ext_id = _extract_partial_jsonrpc_id(raw_line)
                        sys.stderr.write(
                            f"[diodemcp:{core.server_name}] Dropped oversized upstream frame (> {MAX_JSONRPC_LINE_BYTES} bytes)\n"
                        )
                        sys.stderr.flush()
                        if ext_id is not None and ext_id in pending_upstream:
                            fut = pending_upstream.pop(ext_id)
                            if not fut.done():
                                fut.set_result(
                                    _build_mcp_tool_error_response(
                                        ext_id,
                                        banner="🛑 DiodeMCP Alert: Upstream response exceeded 64 MiB frame limit.",
                                    )
                                )
                        continue

                    stripped = raw_line.strip()
                    if not stripped:
                        continue

                    try:
                        msg = json.loads(stripped.decode("utf-8"))
                    except Exception:
                        # Opus M10: Drop non-JSON lines from upstream stdout to stderr
                        sys.stderr.write(
                            stripped.decode("utf-8", errors="replace") + "\n"
                        )
                        sys.stderr.flush()
                        continue

                    if not isinstance(msg, dict):
                        sys.stderr.write(
                            stripped.decode("utf-8", errors="replace") + "\n"
                        )
                        sys.stderr.flush()
                        continue

                    # Case 1: Upstream response to a client request
                    if "method" not in msg and "id" in msg:
                        resp_id = msg.get("id")
                        fut = pending_upstream.pop(resp_id, None)
                        if fut is not None and not fut.done():
                            fut.set_result(msg)
                        continue

                    # Case 2: Upstream server->client request or notification
                    up_method = str(msg.get("method") or "")
                    if "id" in msg:
                        orig_id = msg["id"]
                        # I3 / I14: Prevent upstream server from colliding with `diodemcp-` IDs!
                        if isinstance(orig_id, str) and orig_id.startswith(ELICIT_ID_PREFIX):
                            safe_id = f"upstream-{uuid.uuid4()}"
                            core._remapped_upstream_server_ids[safe_id] = orig_id
                            msg = {**msg, "id": safe_id}
                        else:
                            safe_id = orig_id

                        # I1: Track `sampling/createMessage` so client completion is screened before forwarding!
                        if up_method == "sampling/createMessage":
                            core._pending_upstream_sampling_ids[safe_id] = msg

                    # I3: Prefix upstream `elicitation/create` messages so they can never spoof DiodeMCP!
                    if up_method == "elicitation/create":
                        params_obj = msg.get("params")
                        if isinstance(params_obj, Mapping):
                            orig_text = str(params_obj.get("message") or "")
                            prefixed = f"[from server {core.server_name}] {orig_text}"
                            msg = {
                                **msg,
                                "params": {**dict(params_obj), "message": prefixed},
                            }

                    await core.write_stdout_json(msg)
            finally:
                for fut in list(pending_upstream.values()):
                    if not fut.done():
                        fut.set_exception(
                            TransportError("Upstream MCP stdio server closed stdout")
                        )
                pending_upstream.clear()

        async def _handle_client_request(msg: dict[str, Any]) -> None:
            method = str(msg.get("method") or "")
            has_id = "id" in msg
            req_id = msg.get("id")
            params = msg.get("params")
            params_dict = dict(params) if isinstance(params, Mapping) else {}

            inflight = _InflightClientRequest(request_id=req_id, method=method)
            if has_id:
                core._inflight_client_requests[req_id] = inflight

            try:
                forwarded_args: dict[str, Any] = {}
                outbound_msg = msg

                if method == "tools/call":
                    if not isinstance(params, Mapping) or not isinstance(
                        params.get("name"), str
                    ) or not str(params.get("name")).strip():
                        if has_id:
                            await core.write_stdout_json(
                                {
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "error": {
                                        "code": -32602,
                                        "message": "Invalid params: tools/call requires non-empty tool 'name'",
                                    },
                                }
                            )
                        return

                    tool_name = str(params["name"])
                    raw_args = params.get("arguments")
                    if raw_args is None:
                        tool_args: dict[str, Any] = {}
                    elif isinstance(raw_args, Mapping):
                        tool_args = dict(raw_args)
                    else:
                        if has_id:
                            await core.write_stdout_json(
                                {
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "error": {
                                        "code": -32602,
                                        "message": "Invalid params: tools/call 'arguments' must be an object",
                                    },
                                }
                            )
                        return

                    allowed, effective_args, err_resp = await core.evaluate_and_gate_call(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        inflight=inflight,
                    )
                    if not allowed:
                        if has_id and err_resp is not None:
                            await core.write_stdout_json(err_resp)
                        return
                    if inflight.cancelled:
                        if has_id:
                            await core.write_stdout_json(
                                _build_mcp_tool_error_response(
                                    req_id,
                                    banner="🛑 DiodeMCP Alert: Request cancelled before execution.",
                                )
                            )
                        return

                    forwarded_args = effective_args
                    outbound_msg = {
                        **msg,
                        "params": {**params_dict, "arguments": effective_args},
                    }

                elif method == "resources/read":
                    uri_val = str(params_dict.get("uri") or "")
                    allowed, _eff_args, err_resp = await core.evaluate_and_gate_call(
                        tool_name="read_resource",
                        tool_args={"uri": uri_val, "path": uri_val},
                        inflight=inflight,
                    )
                    if not allowed:
                        if has_id and err_resp is not None:
                            await core.write_stdout_json(err_resp)
                        return

                if not has_id:
                    await _write_child_stdin(outbound_msg)
                    return

                loop = asyncio.get_running_loop()
                up_fut: asyncio.Future[dict[str, Any]] = loop.create_future()
                pending_upstream[req_id] = up_fut
                inflight.forwarded_to_upstream = True
                try:
                    await _write_child_stdin(outbound_msg)
                    upstream_resp = await up_fut
                except Exception as exc:
                    pending_upstream.pop(req_id, None)
                    await core.write_stdout_json(
                        _build_mcp_tool_error_response(
                            req_id,
                            banner="🛑 DiodeMCP Alert: Upstream MCP server failed before responding.",
                            detail=sanitize_error_message(exc),
                        )
                    )
                    return

                checked_resp = await core.inspect_and_record_upstream_response(
                    method=method,
                    request_params=params_dict,
                    forwarded_args=forwarded_args,
                    response_msg=upstream_resp,
                )
                await core.write_stdout_json(checked_resp)
            finally:
                if has_id:
                    core._inflight_client_requests.pop(req_id, None)

        child_reader_task = asyncio.create_task(_child_stdout_reader_loop())
        host_reader = _BoundedJsonLineReader(stdin_stream)

        try:
            while True:
                raw_line, was_oversized = await host_reader.read_line()
                if raw_line is None:
                    break
                if was_oversized:
                    ext_id = _extract_partial_jsonrpc_id(raw_line)
                    await core.write_stdout_json(
                        {
                            "jsonrpc": "2.0",
                            "id": ext_id,
                            "error": {
                                "code": -32600,
                                "message": (
                                    f"Invalid Request: JSON-RPC frame exceeds "
                                    f"{ int(MAX_JSONRPC_LINE_BYTES) } byte limit"
                                ),
                            },
                        }
                    )
                    continue

                stripped = raw_line.strip()
                if not stripped:
                    continue

                try:
                    msg = json.loads(stripped.decode("utf-8"))
                except Exception:
                    await core.write_stdout_json(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": "Parse error"},
                        }
                    )
                    continue

                if not isinstance(msg, dict):
                    await core.write_stdout_json(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32600, "message": "Invalid Request"},
                        }
                    )
                    continue

                # 1. Host response to a server->client request (`diodemcp-<uuid4>` popup or upstream request)
                if "method" not in msg and "id" in msg:
                    resp_id = msg.get("id")
                    if isinstance(resp_id, str) and resp_id.startswith(ELICIT_ID_PREFIX):
                        elicit_fut = core._pending_elicitations.get(resp_id)
                        if elicit_fut is not None and not elicit_fut.done():
                            elicit_fut.set_result(msg)
                        # NEVER forward `diodemcp-*` responses to upstream!
                        continue

                    orig_up_id = core._remapped_upstream_server_ids.pop(
                        str(resp_id), resp_id
                    )
                    out_client_resp = {**msg, "id": orig_up_id}

                    # I1: If this is a client response to upstream `sampling/createMessage`,
                    # evaluate it as an outbound sink toward `server_name` before forwarding!
                    if resp_id in core._pending_upstream_sampling_ids:
                        core._pending_upstream_sampling_ids.pop(resp_id, None)
                        dummy_inflight = _InflightClientRequest(
                            request_id=orig_up_id, method="sampling/createMessage"
                        )
                        allowed, _eff, _err = await core.evaluate_and_gate_call(
                            tool_name="sampling/createMessage",
                            tool_args={"response": msg.get("result")},
                            inflight=dummy_inflight,
                        )
                        if not allowed:
                            await _write_child_stdin(
                                {
                                    "jsonrpc": "2.0",
                                    "id": orig_up_id,
                                    "error": {
                                        "code": -32603,
                                        "message": "Blocked by DiodeMCP: sampling/createMessage completion denied",
                                    },
                                }
                            )
                            continue

                    await _write_child_stdin(out_client_resp)
                    continue

                method = str(msg.get("method") or "")
                if method == "initialize":
                    core._inspect_initialize_capabilities(msg.get("params"))

                if method == "notifications/cancelled":
                    should_forward = core.handle_cancellation_notification(msg)
                    if should_forward:
                        await _write_child_stdin(msg)
                    continue

                task = asyncio.create_task(_handle_client_request(msg))
                core._client_tasks.add(task)
                task.add_done_callback(core._client_tasks.discard)

            for elicit_fut in list(core._pending_elicitations.values()):
                if not elicit_fut.done():
                    elicit_fut.cancel()
            if core._client_tasks:
                await asyncio.gather(*list(core._client_tasks), return_exceptions=True)
        finally:
            if cleanup_stdin is not None:
                cleanup_stdin()
            for task in list(core._client_tasks):
                task.cancel()
            if core._client_tasks:
                await asyncio.gather(*list(core._client_tasks), return_exceptions=True)

            try:
                if child.stdin is not None and not child.stdin.is_closing():
                    child.stdin.close()
            except Exception:
                pass

            try:
                await asyncio.wait_for(child.wait(), timeout=2.0)
            except (asyncio.TimeoutError, TimeoutError):
                try:
                    child.terminate()
                    await asyncio.wait_for(child.wait(), timeout=2.0)
                except Exception:
                    try:
                        child.kill()
                        await asyncio.wait_for(child.wait(), timeout=1.0)
                    except Exception:
                        pass

            if not child_reader_task.done():
                child_reader_task.cancel()
                await asyncio.gather(child_reader_task, return_exceptions=True)

        return int(child.returncode or 0)


async def async_run_http_bridge_shim(
    server_name: str,
    url: str,
    headers: dict[str, str],
    state_dir: Path,
    *,
    workspace_root: Path | None = None,
    popup_timeout_s: float = DEFAULT_POPUP_TIMEOUT_SECONDS,
    stdin_stream: asyncio.StreamReader | None = None,
    stdout_write: Callable[[bytes], Awaitable[None] | None] | None = None,
    engine_client: AsyncEngineClient | None = None,
) -> int:
    """Run hardened MCP Streamable-HTTP bridge shim asynchronously (`I6`, `I14`, `I4`, `C4`)."""
    validated_url = validate_streamable_http_url(url)
    static_headers = dict(headers or {})

    cleanup_stdin: Callable[[], None] | None = None
    with _protocol_stdout_guard(stdout_write) as guarded_stdout_write:
        if stdin_stream is None:
            stdin_stream, cleanup_stdin = _start_fd_stream_reader(0)

        core = _McpShimCore(
            server_name=server_name,
            state_dir=Path(state_dir),
            workspace_root=workspace_root,
            popup_timeout_s=popup_timeout_s,
            stdout_writer=guarded_stdout_write,
            engine_client=engine_client,
        )

        mcp_session_id: str | None = None
        session_lock = asyncio.Lock()

        async def _post_upstream(
            outbound_msg: dict[str, Any],
        ) -> list[dict[str, Any]]:
            nonlocal mcp_session_id
            async with session_lock:
                cur_session = mcp_session_id

            status_code, resp_headers, raw_body = await asyncio.to_thread(
                _send_http_jsonrpc_sync,
                url=validated_url,
                static_headers=static_headers,
                session_id=cur_session,
                payload=outbound_msg,
            )

            new_sid = resp_headers.get("mcp-session-id")
            if isinstance(new_sid, str) and new_sid.strip():
                async with session_lock:
                    mcp_session_id = new_sid.strip()

            if status_code in (202, 204) or not raw_body.strip():
                return []

            ctype = resp_headers.get("content-type", "").lower()
            text_body = raw_body.decode("utf-8", errors="replace")
            if "text/event-stream" in ctype:
                return _parse_sse_jsonrpc_messages(text_body)

            parsed = json.loads(text_body)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            return []

        async def _handle_client_request(msg: dict[str, Any]) -> None:
            method = str(msg.get("method") or "")
            has_id = "id" in msg
            req_id = msg.get("id")
            params = msg.get("params")
            params_dict = dict(params) if isinstance(params, Mapping) else {}

            inflight = _InflightClientRequest(request_id=req_id, method=method)
            if has_id:
                core._inflight_client_requests[req_id] = inflight

            try:
                forwarded_args: dict[str, Any] = {}
                outbound_msg = msg

                if method == "tools/call":
                    if not isinstance(params, Mapping) or not isinstance(
                        params.get("name"), str
                    ) or not str(params.get("name")).strip():
                        if has_id:
                            await core.write_stdout_json(
                                {
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "error": {
                                        "code": -32602,
                                        "message": "Invalid params: tools/call requires non-empty tool 'name'",
                                    },
                                }
                            )
                        return

                    tool_name = str(params["name"])
                    raw_args = params.get("arguments")
                    if raw_args is None:
                        tool_args: dict[str, Any] = {}
                    elif isinstance(raw_args, Mapping):
                        tool_args = dict(raw_args)
                    else:
                        if has_id:
                            await core.write_stdout_json(
                                {
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "error": {
                                        "code": -32602,
                                        "message": "Invalid params: tools/call 'arguments' must be an object",
                                    },
                                }
                            )
                        return

                    allowed, effective_args, err_resp = await core.evaluate_and_gate_call(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        inflight=inflight,
                    )
                    if not allowed:
                        if has_id and err_resp is not None:
                            await core.write_stdout_json(err_resp)
                        return
                    if inflight.cancelled:
                        if has_id:
                            await core.write_stdout_json(
                                _build_mcp_tool_error_response(
                                    req_id,
                                    banner="🛑 DiodeMCP Alert: Request cancelled before execution.",
                                )
                            )
                        return

                    forwarded_args = effective_args
                    outbound_msg = {
                        **msg,
                        "params": {**params_dict, "arguments": effective_args},
                    }

                elif method == "resources/read":
                    uri_val = str(params_dict.get("uri") or "")
                    allowed, _eff_args, err_resp = await core.evaluate_and_gate_call(
                        tool_name="read_resource",
                        tool_args={"uri": uri_val, "path": uri_val},
                        inflight=inflight,
                    )
                    if not allowed:
                        if has_id and err_resp is not None:
                            await core.write_stdout_json(err_resp)
                        return

                inflight.forwarded_to_upstream = True
                try:
                    resp_messages = await _post_upstream(outbound_msg)
                except Exception as exc:
                    if has_id:
                        await core.write_stdout_json(
                            _build_mcp_tool_error_response(
                                req_id,
                                banner="🛑 DiodeMCP Alert: Streamable-HTTP upstream request failed.",
                                detail=sanitize_error_message(exc),
                            )
                        )
                    return

                if not has_id:
                    return

                matched_resp: dict[str, Any] | None = None
                for item in resp_messages:
                    if "method" in item:
                        # Server-initiated notification or request inside SSE stream
                        up_method = str(item.get("method") or "")
                        if up_method == "elicitation/create" and isinstance(
                            item.get("params"), Mapping
                        ):
                            orig_text = str(item["params"].get("message") or "")
                            item = {
                                **item,
                                "params": {
                                    **dict(item["params"]),
                                    "message": f"[from server {core.server_name}] {orig_text}",
                                },
                            }
                        if isinstance(item.get("id"), str) and str(
                            item["id"]
                        ).startswith(ELICIT_ID_PREFIX):
                            item = {**item, "id": f"upstream-{uuid.uuid4()}"}
                        await core.write_stdout_json(item)
                    elif item.get("id") == req_id or matched_resp is None:
                        matched_resp = item

                if matched_resp is None:
                    await core.write_stdout_json(
                        _build_mcp_tool_error_response(
                            req_id,
                            banner="🛑 DiodeMCP Alert: Upstream HTTP server returned no matching JSON-RPC response.",
                        )
                    )
                    return

                checked_resp = await core.inspect_and_record_upstream_response(
                    method=method,
                    request_params=params_dict,
                    forwarded_args=forwarded_args,
                    response_msg=matched_resp,
                )
                await core.write_stdout_json(checked_resp)
            finally:
                if has_id:
                    core._inflight_client_requests.pop(req_id, None)

        host_reader = _BoundedJsonLineReader(stdin_stream)
        try:
            while True:
                raw_line, was_oversized = await host_reader.read_line()
                if raw_line is None:
                    break
                if was_oversized:
                    ext_id = _extract_partial_jsonrpc_id(raw_line)
                    await core.write_stdout_json(
                        {
                            "jsonrpc": "2.0",
                            "id": ext_id,
                            "error": {
                                "code": -32600,
                                "message": (
                                    f"Invalid Request: JSON-RPC frame exceeds "
                                    f"{int(MAX_JSONRPC_LINE_BYTES)} byte limit"
                                ),
                            },
                        }
                    )
                    continue

                stripped = raw_line.strip()
                if not stripped:
                    continue

                try:
                    msg = json.loads(stripped.decode("utf-8"))
                except Exception:
                    await core.write_stdout_json(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": "Parse error"},
                        }
                    )
                    continue

                if not isinstance(msg, dict):
                    await core.write_stdout_json(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32600, "message": "Invalid Request"},
                        }
                    )
                    continue

                if "method" not in msg and "id" in msg:
                    resp_id = msg.get("id")
                    if isinstance(resp_id, str) and resp_id.startswith(ELICIT_ID_PREFIX):
                        elicit_fut = core._pending_elicitations.get(resp_id)
                        if elicit_fut is not None and not elicit_fut.done():
                            elicit_fut.set_result(msg)
                    continue

                method = str(msg.get("method") or "")
                if method == "initialize":
                    core._inspect_initialize_capabilities(msg.get("params"))

                if method == "notifications/cancelled":
                    core.handle_cancellation_notification(msg)
                    continue

                task = asyncio.create_task(_handle_client_request(msg))
                core._client_tasks.add(task)
                task.add_done_callback(core._client_tasks.discard)

            for elicit_fut in list(core._pending_elicitations.values()):
                if not elicit_fut.done():
                    elicit_fut.cancel()
            if core._client_tasks:
                await asyncio.gather(*list(core._client_tasks), return_exceptions=True)
        finally:
            if cleanup_stdin is not None:
                cleanup_stdin()
            for task in list(core._client_tasks):
                task.cancel()
            if core._client_tasks:
                await asyncio.gather(*list(core._client_tasks), return_exceptions=True)

        return 0


def run_stdio_shim(
    server_name: str,
    command: list[str],
    state_dir: Path,
    *,
    workspace_root: Path | None = None,
    popup_timeout_s: float = DEFAULT_POPUP_TIMEOUT_SECONDS,
) -> int:
    """Synchronous entrypoint for MCP `stdio` shim (`R4`, `I6`, `I14`, `C4`)."""
    return asyncio.run(
        async_run_stdio_shim(
            server_name=server_name,
            command=command,
            state_dir=Path(state_dir),
            workspace_root=workspace_root,
            popup_timeout_s=popup_timeout_s,
        )
    )


def run_http_bridge_shim(
    server_name: str,
    url: str,
    headers: dict[str, str],
    state_dir: Path,
    *,
    workspace_root: Path | None = None,
    popup_timeout_s: float = DEFAULT_POPUP_TIMEOUT_SECONDS,
) -> int:
    """Synchronous entrypoint for MCP Streamable-HTTP bridge shim (`R4`, `I6`, `I14`, `C4`)."""
    return asyncio.run(
        async_run_http_bridge_shim(
            server_name=server_name,
            url=url,
            headers=headers,
            state_dir=Path(state_dir),
            workspace_root=workspace_root,
            popup_timeout_s=popup_timeout_s,
        )
    )


def _main_shim_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DiodeMCP stdio / Streamable-HTTP shim")
    parser.add_argument("--server", required=True, help="Logical MCP server name")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument(
        "--popup-timeout",
        type=float,
        default=DEFAULT_POPUP_TIMEOUT_SECONDS,
    )
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="HTTP header in 'Key: Value' format (repeatable)",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.url:
        from diodemcp.config_patcher import load_shim_headers_for_server

        parsed_headers: dict[str, str] = load_shim_headers_for_server(
            args.state_dir, args.server
        )
        for item in args.header:
            if ":" in item:
                k, v = item.split(":", 1)
                parsed_headers[k.strip()] = v.strip()
        return run_http_bridge_shim(
            args.server,
            args.url,
            parsed_headers,
            args.state_dir,
            workspace_root=args.workspace_root,
            popup_timeout_s=args.popup_timeout,
        )

    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    return run_stdio_shim(
        args.server,
        cmd,
        args.state_dir,
        workspace_root=args.workspace_root,
        popup_timeout_s=args.popup_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(_main_shim_cli())
