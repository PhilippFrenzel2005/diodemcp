"""Fast stdlib-first IDE native hook dispatcher (`Claude Code`, `Cursor`, `agy` — `C4`, `C5`, `I15`)."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, TextIO

from diodemcp.constants import DEFAULT_IDLE_TIMEOUT_SECONDS, MAX_LEDGER_SPAN_BYTES

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Optional hook point for unit tests that monkeypatch `diodemcp.hooks.EngineClient`
EngineClient: Any = None


def sanitize_error_message(message: Any, *, max_len: int = 240) -> str:
    """Sanitize error text to single-line bounded printable string without control chars."""
    raw = str(message or "Daemon transport error")
    cleaned = _SURROGATE_RE.sub("\ufffd", raw)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = _CONTROL_CHAR_RE.sub("", cleaned).strip()
    if not cleaned:
        return "Daemon transport error"
    if len(cleaned) > max_len:
        return cleaned[: max_len - 3] + "..."
    return cleaned


class _FastHookEngineClient:
    """Zero-ML-import synchronous IPC client for `<20ms` hook cold start (`I15`)."""

    def __init__(
        self,
        state_dir: Path | str | None = None,
        *,
        timeout_s: float = 10.0,
    ) -> None:
        self.state_dir: Path | str | None = state_dir
        self.timeout_s: float = float(timeout_s)

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        from diodemcp.transport import default_transport, send_rpc_sync

        cfg = default_transport(self.state_dir)
        return send_rpc_sync(
            cfg,
            method,
            params,
            admin_secret=None,
            auto_spawn=True,
            idle_timeout_s=float(DEFAULT_IDLE_TIMEOUT_SECONDS),
            bundle_dir=None,
            timeout_s=self.timeout_s,
        )

    def evaluate_call(
        self,
        workspace_root: Path | str | None = None,
        server: str = "native",
        tool: str = "",
        arguments: Mapping[str, Any] | None = None,
        *,
        client_supports_popup: bool = False,
        client_kind: str = "ide_hook",
        permission_mode: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        ws = extract_workspace_root(None, fallback_cwd=workspace_root)
        return self._call(
            "evaluate_call",
            {
                "workspace_root": ws,
                "server": str(server or "native"),
                "tool": str(tool or ""),
                "arguments": dict(arguments or {}),
                "client_supports_popup": bool(client_supports_popup),
                "client_kind": str(client_kind or "ide_hook"),
                "permission_mode": permission_mode,
                "popup_binding_sha256": None,
            },
        )

    def record_output(
        self,
        workspace_root: Path | str | None = None,
        server: str = "native",
        tool: str = "",
        arguments: Mapping[str, Any] | None = None,
        output_text: str = "",
        *,
        extra_flags: int = 0,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        ws = extract_workspace_root(None, fallback_cwd=workspace_root)
        return self._call(
            "record_output",
            {
                "workspace_root": ws,
                "server": str(server or "native"),
                "tool": str(tool or ""),
                "arguments": dict(arguments or {}),
                "output_text": str(output_text or ""),
                "extra_flags": int(extra_flags),
            },
        )

    def record_intent(
        self,
        workspace_root: Path | str | None = None,
        intent: str | None = None,
    ) -> dict[str, Any]:
        ws = extract_workspace_root(None, fallback_cwd=workspace_root)
        return self._call(
            "record_intent",
            {
                "workspace_root": ws,
                "intent": intent,
            },
        )

    def record_local_write(
        self,
        workspace_root: Path | str | None = None,
        target_path: str = "",
        written_content: str = "",
    ) -> dict[str, Any]:
        ws = extract_workspace_root(None, fallback_cwd=workspace_root)
        return self._call(
            "record_local_write",
            {
                "workspace_root": ws,
                "target_path": str(target_path or ""),
                "written_content": str(written_content or ""),
            },
        )


def _find_git_project_root(start_path: str) -> str | None:
    """Return nearest ancestor directory (including `start_path`) containing `.git`, or `None`."""
    try:
        cur = Path(start_path)
        for cand in (cur, *cur.parents):
            if (cand / ".git").exists():
                return str(cand)
    except OSError:
        return None
    return None


def _normalize_real_path(raw_path: str) -> str:
    expanded = os.path.expanduser(raw_path.replace("\x00", ""))
    return os.path.realpath(os.path.abspath(expanded))


def extract_workspace_root(
    payload: Mapping[str, Any] | None,
    *,
    fallback_cwd: Path | str | None = None,
) -> str:
    """Extract canonical `workspace_root` from hook payload (`CLAUDE_PROJECT_DIR`, `workspace_roots[0]`, or nearest `.git` ancestor of `cwd`) (`C5`)."""
    explicit_root: str | None = None
    cwd_candidate: str | None = None

    if isinstance(payload, Mapping):
        raw_ws = payload.get("workspace_root")
        if isinstance(raw_ws, str) and raw_ws.strip():
            explicit_root = raw_ws.strip()
        else:
            ws_roots = payload.get("workspace_roots")
            if isinstance(ws_roots, Sequence) and not isinstance(
                ws_roots, (str, bytes)
            ):
                for item in ws_roots:
                    if isinstance(item, str) and item.strip():
                        explicit_root = item.strip()
                        break

        if explicit_root is None:
            env_proj = os.environ.get("CLAUDE_PROJECT_DIR")
            if isinstance(env_proj, str) and env_proj.strip():
                explicit_root = env_proj.strip()

        if explicit_root is None:
            raw_cwd = payload.get("cwd")
            if isinstance(raw_cwd, str) and raw_cwd.strip():
                cwd_candidate = raw_cwd.strip()

    if explicit_root is not None:
        return _normalize_real_path(explicit_root)

    if cwd_candidate is not None:
        resolved_cwd = _normalize_real_path(cwd_candidate)
        git_root = _find_git_project_root(resolved_cwd)
        return git_root if git_root is not None else resolved_cwd

    if fallback_cwd is not None:
        raw_fb = os.fspath(fallback_cwd)
        if isinstance(raw_fb, str) and raw_fb.strip():
            return _normalize_real_path(raw_fb.strip())

    resolved_proc_cwd = _normalize_real_path(os.getcwd())
    if isinstance(payload, Mapping):
        git_root = _find_git_project_root(resolved_proc_cwd)
        if git_root is not None:
            return git_root
    return resolved_proc_cwd


def _normalize_ide_name(ide: str) -> str:
    cleaned = str(ide or "").strip().lower()
    if cleaned in ("claude", "claude-code", "claudecode"):
        return "claude"
    if cleaned == "cursor":
        return "cursor"
    if cleaned in ("agy", "antigravity", "antigravity-cli"):
        return "agy"
    return cleaned


def _normalize_tool_input(tool_name: str, raw_input: Any) -> dict[str, Any]:
    if isinstance(raw_input, Mapping):
        return dict(raw_input)
    if isinstance(raw_input, str):
        stripped = raw_input.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        if tool_name.strip().lower() in (
            "bash",
            "shell",
            "sh",
            "run_command",
            "execute_command",
            "cmd",
        ):
            return {"command": raw_input}
        return {"file_path": raw_input, "path": raw_input}
    return {}


def _collect_strings(node: Any, out: list[str]) -> None:
    if isinstance(node, str):
        if node:
            out.append(node)
    elif isinstance(node, Mapping):
        for v in node.values():
            _collect_strings(v, out)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_strings(item, out)


def _extract_tool_response_text(payload: Mapping[str, Any]) -> str:
    for key in (
        "tool_response",
        "tool_output",
        "result_json",
        "output",
        "response",
        "result",
        "content",
    ):
        if key in payload and payload[key] is not None:
            val = payload[key]
            if isinstance(val, str):
                if val:
                    return val
                continue
            parts: list[str] = []
            _collect_strings(val, parts)
            if parts:
                return "\n".join(parts)
            if isinstance(val, (Mapping, Sequence)) and not val:
                continue
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)
    stdio_parts = [
        str(payload[k])
        for k in ("stdout", "stderr")
        if isinstance(payload.get(k), str) and payload[k]
    ]
    if stdio_parts:
        return "\n".join(stdio_parts)
    return ""


_NATIVE_FILE_READ_TOOLS: frozenset[str] = frozenset(
    {
        "read",
        "readfile",
        "read_file",
        "read_text_file",
        "read_media_file",
        "read_multiple_files",
        "view",
        "view_file",
        "notebookread",
        "notebook_read",
        "cat",
    }
)
_NATIVE_EDITOR_VIEW_TOOLS: frozenset[str] = frozenset(
    {
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "text_editor",
    }
)
_HOOK_PATH_KEYS: frozenset[str] = frozenset(
    {
        "file_path",
        "path",
        "filepath",
        "target_file",
        "targetfile",
        "target_path",
        "absolute_path",
        "absolutepath",
        "notebook_path",
        "file",
        "filename",
        "paths",
        "files",
    }
)


def _fallback_read_text_for_post_tool(
    tool_name: str,
    tool_input: Mapping[str, Any],
    payload: Mapping[str, Any],
    workspace_root: str,
) -> str:
    tl = tool_name.strip().lower()
    if tl.startswith("mcp__"):
        parts = tl.split("__", 2)
        if len(parts) == 3:
            tl = parts[2]
    elif "." in tl:
        tl = tl.split(".", 1)[1]

    is_view_editor = tl in _NATIVE_EDITOR_VIEW_TOOLS and (
        str(tool_input.get("command") or "").strip().lower() == "view"
    )
    if tl not in _NATIVE_FILE_READ_TOOLS and not is_view_editor:
        return ""

    candidates: list[str] = []
    for mapping in (tool_input, payload):
        if not isinstance(mapping, Mapping):
            continue
        for k, v in mapping.items():
            if str(k).strip().lower() not in _HOOK_PATH_KEYS:
                continue
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())
            elif isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                for item in v:
                    if isinstance(item, str) and item.strip():
                        candidates.append(item.strip())

    read_chunks: list[str] = []
    seen_paths: set[str] = set()
    for fp in candidates:
        if fp in seen_paths:
            continue
        seen_paths.add(fp)
        content = _best_effort_read_local_file(fp, workspace_root)
        if content:
            read_chunks.append(content)
    return "\n".join(read_chunks)


def _extract_cursor_edit_text(payload: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("content", "new_string", "new_str", "text", "replacement"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    edits = payload.get("edits")
    if isinstance(edits, Sequence) and not isinstance(edits, (str, bytes)):
        for ed in edits:
            if isinstance(ed, Mapping):
                for k in ("new_string", "new_str", "text", "content", "replacement", "insert"):
                    v = ed.get(k)
                    if isinstance(v, str) and v:
                        parts.append(v)
                # Also collect any other string values except old_string
                for k, v in ed.items():
                    if (
                        str(k).strip().lower() not in ("old_string", "old_str", "range")
                        and isinstance(v, str)
                        and v
                        and v not in parts
                    ):
                        parts.append(v)
    return "\n".join(parts)


def _best_effort_read_local_file(file_path: str, workspace_root: str) -> str:
    if not file_path:
        return ""
    try:
        expanded = Path(file_path).expanduser()
        if not expanded.is_absolute():
            expanded = Path(workspace_root) / expanded
        if expanded.is_symlink() or not expanded.exists():
            return ""
        st = expanded.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_LEDGER_SPAN_BYTES:
            return ""
        return expanded.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _format_hook_decision_reason(decision: Mapping[str, Any]) -> str:
    """Format clear human/agent diagnostic with `incident_id` and zero secret tokens (`C4`)."""
    banner = str(
        decision.get("banner")
        or decision.get("reason")
        or "Blocked by DiodeMCP security policy."
    ).strip()
    incident_id = str(decision.get("incident_id") or "").strip()
    if incident_id:
        return (
            f"{banner} [Incident ID: {incident_id} — inspect via `diodemcp audit` "
            f"or approve in an interactive terminal via `diodemcp allow {incident_id}`]"
        )
    return f"{banner} [Inspect in an interactive terminal via `diodemcp audit`]"


def _write_json_line(stream: TextIO, obj: dict[str, Any]) -> None:
    stream.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def emit_fail_closed_deny(
    ide: str,
    event: str,
    reason: str,
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Emit valid IDE-native deny JSON on `stdout`, log to `stderr`, and return exit code `2` (`I15`)."""
    safe_reason = sanitize_error_message(
        reason or "DiodeMCP fail-closed hook error", max_len=360
    )
    full_msg = (
        f"🛑 DiodeMCP Fail-Closed Block ({ide}/{event}): {safe_reason}. "
        f"Inspect via `diodemcp audit`."
    )
    try:
        stderr.write(f"[diodemcp] {full_msg}\n")
        stderr.flush()
    except BaseException:
        pass

    ide_norm = _normalize_ide_name(ide)
    event_lower = str(event or "").strip().lower()
    try:
        if ide_norm == "claude":
            if event_lower in ("pretooluse", "pre_tool_use", "beforetooluse"):
                _write_json_line(
                    stdout,
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": full_msg,
                        }
                    },
                )
            else:
                _write_json_line(
                    stdout,
                    {
                        "decision": "block",
                        "reason": full_msg,
                    },
                )
        elif ide_norm == "cursor":
            _write_json_line(
                stdout,
                {
                    "permission": "deny",
                    "user_message": full_msg,
                    "agent_message": full_msg,
                },
            )
        else:
            _write_json_line(
                stdout,
                {
                    "decision": "deny",
                    "reason": full_msg,
                },
            )
    except BaseException:
        pass
    return 2


def _emit_cursor_decision(
    decision: Mapping[str, Any],
    *,
    permission_mode: str | None,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    action = str(decision.get("action") or "block").lower()
    if action == "allow":
        _write_json_line(
            stdout,
            {
                "permission": "allow",
                "user_message": "",
                "agent_message": "",
            },
        )
        return 0

    band = str(decision.get("band") or "critical").lower()
    is_self_tamper = bool(decision.get("is_self_tamper", False))
    bypass = permission_mode in ("bypassPermissions", "dontAsk")
    reason_msg = _format_hook_decision_reason(decision)

    if (
        band == "warn"
        and not is_self_tamper
        and not bypass
        and action in ("ask", "elicit")
    ):
        _write_json_line(
            stdout,
            {
                "permission": "ask",
                "user_message": reason_msg,
                "agent_message": reason_msg,
            },
        )
        return 0

    stderr.write(f"[diodemcp] {reason_msg}\n")
    stderr.flush()
    _write_json_line(
        stdout,
        {
            "permission": "deny",
            "user_message": reason_msg,
            "agent_message": reason_msg,
        },
    )
    return 2


def _emit_agy_decision(
    decision: Mapping[str, Any],
    *,
    permission_mode: str | None,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    action = str(decision.get("action") or "block").lower()
    if action == "allow":
        _write_json_line(stdout, {"decision": "allow", "reason": ""})
        return 0

    band = str(decision.get("band") or "critical").lower()
    is_self_tamper = bool(decision.get("is_self_tamper", False))
    bypass = permission_mode in ("bypassPermissions", "dontAsk")
    reason_msg = _format_hook_decision_reason(decision)
    incident_id = decision.get("incident_id")

    if (
        band == "warn"
        and not is_self_tamper
        and not bypass
        and action in ("ask", "elicit")
    ):
        out_ask: dict[str, Any] = {"decision": "ask", "reason": reason_msg}
        if incident_id:
            out_ask["incident_id"] = incident_id
        _write_json_line(stdout, out_ask)
        return 0

    stderr.write(f"[diodemcp] {reason_msg}\n")
    stderr.flush()
    out_deny: dict[str, Any] = {"decision": "deny", "reason": reason_msg}
    if incident_id:
        out_deny["incident_id"] = incident_id
    _write_json_line(stdout, out_deny)
    return 2


def run_hook(
    ide: str,
    event: str,
    *,
    stdin_text: str | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    state_dir: Path | str | None = None,
    client: Any = None,
) -> int:
    """Execute a per-IDE native hook with fail-closed `BaseException` protection (`C4`, `C5`, `I15`)."""
    out_stream = stdout if stdout is not None else sys.stdout
    err_stream = stderr if stderr is not None else sys.stderr

    try:
        ide_norm = _normalize_ide_name(ide)
        if ide_norm not in ("claude", "cursor", "agy"):
            raise ValueError(f"Unsupported IDE hook dialect: {ide!r}")

        event_clean = str(event or "").strip()
        if not event_clean:
            raise ValueError("Missing hook event name")

        if stdin_text is not None:
            raw_json = stdin_text
        else:
            in_stream = stdin if stdin is not None else sys.stdin
            raw_json = in_stream.read()

        if not isinstance(raw_json, str) or not raw_json.strip():
            raise ValueError("Empty JSON payload on hook stdin")

        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("Hook stdin JSON payload must be an object")

        ws = extract_workspace_root(payload)
        raw_perm = payload.get("permission_mode") or payload.get("permissionMode")
        permission_mode = str(raw_perm).strip() if isinstance(raw_perm, str) else None

        if client is not None:
            active_client = client
        elif EngineClient is not None:
            from diodemcp.transport import default_transport

            active_client = EngineClient(default_transport(state_dir))
        else:
            active_client = _FastHookEngineClient(state_dir)

        # --- 1. Claude Code (`claude`) ---
        if ide_norm == "claude":
            if event_clean in ("UserPromptSubmit", "user_prompt_submit"):
                prompt_val = (
                    payload.get("prompt")
                    or payload.get("user_prompt")
                    or payload.get("message")
                    or ""
                )
                active_client.record_intent(
                    workspace_root=ws,
                    intent=str(prompt_val),
                )
                _write_json_line(out_stream, {})
                return 0

            if event_clean in ("PostToolUse", "post_tool_use"):
                tool_name = str(
                    payload.get("tool_name") or payload.get("tool") or "unknown"
                ).strip()
                raw_in = (
                    payload.get("tool_input")
                    if "tool_input" in payload
                    else payload.get("arguments")
                )
                tool_input = _normalize_tool_input(tool_name, raw_in)
                output_text = _extract_tool_response_text(payload)
                if not output_text:
                    output_text = _fallback_read_text_for_post_tool(
                        tool_name, tool_input, payload, ws
                    )
                active_client.record_output(
                    workspace_root=ws,
                    server="native",
                    tool=tool_name,
                    arguments=tool_input,
                    output_text=output_text,
                )
                return 0

            if event_clean in ("PreToolUse", "pre_tool_use", "BeforeToolUse"):
                tool_name = str(
                    payload.get("tool_name") or payload.get("tool") or ""
                ).strip()
                if not tool_name:
                    raise ValueError("Missing tool_name in Claude PreToolUse payload")
                raw_in = (
                    payload.get("tool_input")
                    if "tool_input" in payload
                    else payload.get("arguments")
                )
                tool_input = _normalize_tool_input(tool_name, raw_in)
                decision = active_client.evaluate_call(
                    workspace_root=ws,
                    server="native",
                    tool=tool_name,
                    arguments=tool_input,
                    client_supports_popup=False,
                    client_kind="ide_hook",
                    permission_mode=permission_mode,
                )
                if not isinstance(decision, Mapping):
                    raise RuntimeError("Engine evaluate_call returned non-dict response")

                action = str(decision.get("action") or "block").lower()
                if action == "allow":
                    updated_input = decision.get("updatedInput")
                    if isinstance(updated_input, Mapping):
                        _write_json_line(
                            out_stream,
                            {
                                "hookSpecificOutput": {
                                    "hookEventName": "PreToolUse",
                                    "updatedInput": dict(updated_input),
                                }
                            },
                        )
                    else:
                        _write_json_line(out_stream, {})
                    return 0

                band = str(decision.get("band") or "critical").lower()
                is_self_tamper = bool(decision.get("is_self_tamper", False))
                bypass = permission_mode in ("bypassPermissions", "dontAsk")
                reason_msg = _format_hook_decision_reason(decision)

                if (
                    is_self_tamper
                    or bypass
                    or action in ("block", "deny")
                    or band == "critical"
                ):
                    err_stream.write(f"[diodemcp] {reason_msg}\n")
                    err_stream.flush()
                    _write_json_line(
                        out_stream,
                        {
                            "hookSpecificOutput": {
                                "hookEventName": "PreToolUse",
                                "permissionDecision": "deny",
                                "permissionDecisionReason": reason_msg,
                            }
                        },
                    )
                    return 2

                _write_json_line(
                    out_stream,
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "ask",
                            "permissionDecisionReason": reason_msg,
                        }
                    },
                )
                return 0

            raise ValueError(f"Unsupported Claude hook event: {event_clean!r}")

        # --- 2. Cursor (`cursor`) ---
        if ide_norm == "cursor":
            if event_clean == "beforeShellExecution":
                cmd = payload.get("command")
                if not isinstance(cmd, str):
                    ti = payload.get("tool_input")
                    cmd = ti.get("command") if isinstance(ti, Mapping) else ""
                decision = active_client.evaluate_call(
                    workspace_root=ws,
                    server="native",
                    tool="Bash",
                    arguments={"command": str(cmd or "")},
                    client_supports_popup=False,
                    client_kind="ide_hook",
                    permission_mode=permission_mode,
                )
                if not isinstance(decision, Mapping):
                    raise RuntimeError("Engine evaluate_call returned non-dict response")
                return _emit_cursor_decision(
                    decision,
                    permission_mode=permission_mode,
                    stdout=out_stream,
                    stderr=err_stream,
                )

            if event_clean == "beforeMCPExecution":
                srv = str(
                    payload.get("server_name") or payload.get("server") or "mcp"
                ).strip()
                tl = str(
                    payload.get("tool_name") or payload.get("tool") or ""
                ).strip()
                if not tl:
                    raise ValueError("Missing tool_name in Cursor beforeMCPExecution")
                raw_in = (
                    payload.get("tool_input")
                    if "tool_input" in payload
                    else payload.get("arguments")
                )
                tool_args = _normalize_tool_input(tl, raw_in)
                decision = active_client.evaluate_call(
                    workspace_root=ws,
                    server=srv,
                    tool=tl,
                    arguments=tool_args,
                    client_supports_popup=False,
                    client_kind="ide_hook",
                    permission_mode=permission_mode,
                )
                if not isinstance(decision, Mapping):
                    raise RuntimeError("Engine evaluate_call returned non-dict response")
                return _emit_cursor_decision(
                    decision,
                    permission_mode=permission_mode,
                    stdout=out_stream,
                    stderr=err_stream,
                )

            if event_clean == "beforeReadFile":
                file_path = str(
                    payload.get("file_path")
                    or payload.get("path")
                    or payload.get("target_file")
                    or ""
                ).strip()
                if not file_path:
                    raise ValueError("Missing file_path in Cursor beforeReadFile")
                decision = active_client.evaluate_call(
                    workspace_root=ws,
                    server="native",
                    tool="Read",
                    arguments={"file_path": file_path},
                    client_supports_popup=False,
                    client_kind="ide_hook",
                    permission_mode=permission_mode,
                )
                if not isinstance(decision, Mapping):
                    raise RuntimeError("Engine evaluate_call returned non-dict response")
                if str(decision.get("action") or "").lower() == "allow":
                    content_val = payload.get("content")
                    if not isinstance(content_val, str) or not content_val:
                        content_val = _best_effort_read_local_file(file_path, ws)
                    if content_val:
                        active_client.record_output(
                            workspace_root=ws,
                            server="native",
                            tool="Read",
                            arguments={"file_path": file_path},
                            output_text=content_val,
                        )
                return _emit_cursor_decision(
                    decision,
                    permission_mode=permission_mode,
                    stdout=out_stream,
                    stderr=err_stream,
                )

            if event_clean == "afterFileEdit":
                file_path = str(
                    payload.get("file_path")
                    or payload.get("path")
                    or payload.get("target_file")
                    or ""
                ).strip()
                written_text = _extract_cursor_edit_text(payload)
                if file_path and written_text:
                    active_client.record_local_write(
                        workspace_root=ws,
                        target_path=file_path,
                        written_content=written_text,
                    )
                _write_json_line(
                    out_stream,
                    {
                        "permission": "allow",
                        "user_message": "",
                        "agent_message": "",
                    },
                )
                return 0

            if event_clean == "afterShellExecution":
                cmd = payload.get("command")
                if not isinstance(cmd, str):
                    ti = payload.get("tool_input")
                    cmd = ti.get("command") if isinstance(ti, Mapping) else ""
                output_text = _extract_tool_response_text(payload)
                active_client.record_output(
                    workspace_root=ws,
                    server="native",
                    tool="Bash",
                    arguments={"command": str(cmd or "")},
                    output_text=output_text,
                )
                _write_json_line(
                    out_stream,
                    {
                        "permission": "allow",
                        "user_message": "",
                        "agent_message": "",
                    },
                )
                return 0

            if event_clean == "afterMCPExecution":
                srv = str(
                    payload.get("server_name") or payload.get("server") or "mcp"
                ).strip()
                tl = str(
                    payload.get("tool_name") or payload.get("tool") or "unknown"
                ).strip()
                raw_in = (
                    payload.get("tool_input")
                    if "tool_input" in payload
                    else payload.get("arguments")
                )
                tool_args = _normalize_tool_input(tl, raw_in)
                output_text = _extract_tool_response_text(payload)
                active_client.record_output(
                    workspace_root=ws,
                    server=srv,
                    tool=tl,
                    arguments=tool_args,
                    output_text=output_text,
                )
                _write_json_line(
                    out_stream,
                    {
                        "permission": "allow",
                        "user_message": "",
                        "agent_message": "",
                    },
                )
                return 0

            raise ValueError(f"Unsupported Cursor hook event: {event_clean!r}")

        # --- 3. Antigravity (`agy`) ---
        if event_clean in ("user_prompt_submit", "UserPromptSubmit"):
            prompt_val = (
                payload.get("prompt")
                or payload.get("user_prompt")
                or payload.get("message")
                or ""
            )
            active_client.record_intent(workspace_root=ws, intent=str(prompt_val))
            _write_json_line(out_stream, {"decision": "allow", "reason": ""})
            return 0

        if event_clean in ("post_tool_use", "PostToolUse"):
            tool_name = str(
                payload.get("tool_name") or payload.get("tool") or "unknown"
            ).strip()
            raw_in = (
                payload.get("tool_input")
                if "tool_input" in payload
                else payload.get("arguments")
            )
            tool_input = _normalize_tool_input(tool_name, raw_in)
            output_text = _extract_tool_response_text(payload)
            if not output_text:
                output_text = _fallback_read_text_for_post_tool(
                    tool_name, tool_input, payload, ws
                )
            active_client.record_output(
                workspace_root=ws,
                server="native",
                tool=tool_name,
                arguments=tool_input,
                output_text=output_text,
            )
            _write_json_line(out_stream, {"decision": "allow", "reason": ""})
            return 0

        if event_clean in ("pre_tool_use", "PreToolUse", "BeforeToolUse"):
            tool_name = str(
                payload.get("tool_name") or payload.get("tool") or ""
            ).strip()
            if not tool_name:
                raise ValueError("Missing tool_name in agy pre_tool_use payload")
            raw_in = (
                payload.get("tool_input")
                if "tool_input" in payload
                else payload.get("arguments")
            )
            tool_input = _normalize_tool_input(tool_name, raw_in)
            decision = active_client.evaluate_call(
                workspace_root=ws,
                server="native",
                tool=tool_name,
                arguments=tool_input,
                client_supports_popup=False,
                client_kind="ide_hook",
                permission_mode=permission_mode,
            )
            if not isinstance(decision, Mapping):
                raise RuntimeError("Engine evaluate_call returned non-dict response")
            return _emit_agy_decision(
                decision,
                permission_mode=permission_mode,
                stdout=out_stream,
                stderr=err_stream,
            )

        raise ValueError(f"Unsupported agy hook event: {event_clean!r}")

    except BaseException as exc:
        return emit_fail_closed_deny(
            ide,
            event,
            sanitize_error_message(exc),
            stdout=out_stream,
            stderr=err_stream,
        )


def main_hook_cli(argv: Sequence[str] | None = None) -> int:
    """Fast stdlib CLI entrypoint for `diodemcp hook <ide> <event> [--state-dir PATH]`."""
    try:
        parser = argparse.ArgumentParser(
            prog="diodemcp hook",
            description="DiodeMCP fail-closed native IDE hook handler",
        )
        parser.add_argument("ide", help="IDE dialect: claude | cursor | agy")
        parser.add_argument("event", help="Hook event name")
        parser.add_argument("--state-dir", type=Path, default=None)
        args = parser.parse_args(list(argv) if argv is not None else None)
        state_dir = args.state_dir or (
            Path(os.environ["DIODEMCP_STATE_DIR"])
            if os.environ.get("DIODEMCP_STATE_DIR")
            else None
        )
        return run_hook(args.ide, args.event, state_dir=state_dir)
    except BaseException as exc:
        raw_argv = list(argv) if argv is not None else sys.argv[2:]
        ide_guess = raw_argv[0] if len(raw_argv) >= 1 else "claude"
        ev_guess = raw_argv[1] if len(raw_argv) >= 2 else "PreToolUse"
        return emit_fail_closed_deny(
            ide_guess,
            ev_guess,
            sanitize_error_message(exc),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
