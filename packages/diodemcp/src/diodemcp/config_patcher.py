"""Atomic, idempotent, lossless IDE config & hook patcher with OAuth URL safety (`I14`, `I15`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any

from diodemcp.constants import DEFAULT_STATE_DIR
from diodemcp.transport import (
    TransportAuthError,
    atomic_write_private_bytes,
    atomic_write_private_json,
    ensure_private_dir,
)

PROTECTED_OAUTH_WARNING: str = (
    "PROTECTED via IDE Hooks; MCP OAuth transport left native"
)

CLAUDE_HOOK_EVENTS: tuple[str, ...] = (
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
)
CURSOR_HOOK_EVENTS: tuple[str, ...] = (
    "beforeShellExecution",
    "beforeMCPExecution",
    "beforeReadFile",
    "afterFileEdit",
)
AGY_HOOK_EVENTS: tuple[str, ...] = (
    "user_prompt_submit",
    "pre_tool_use",
    "post_tool_use",
)


def _resolve_targets(target: str | Sequence[str]) -> list[str]:
    if isinstance(target, str):
        parts = [p.strip().lower() for p in target.split(",") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in target if str(p).strip()]

    if not parts or "all" in parts:
        return ["claude", "cursor", "agy"]

    normalized: list[str] = []
    for p in parts:
        if p in ("claude", "claude-code", "claudecode"):
            norm = "claude"
        elif p == "cursor":
            norm = "cursor"
        elif p in ("agy", "antigravity", "antigravity-cli"):
            norm = "agy"
        else:
            raise ValueError(
                f"Unsupported IDE target {p!r}; expected claude, cursor, agy, or all"
            )
        if norm not in normalized:
            normalized.append(norm)
    return normalized


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
        if (stat.S_IMODE(st.st_mode) & 0o077) != 0:
            return False
    return True


def _load_private_json_dict(path: Path) -> dict[str, Any]:
    if not _is_safe_0600_regular_file(path):
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def load_shim_headers_for_server(state_dir: Path | str, server_name: str) -> dict[str, str]:
    """Load `0600` static HTTP headers for `server_name` from `<state_dir>/shim_headers.json`."""
    headers_path = Path(state_dir).expanduser() / "shim_headers.json"
    all_headers = _load_private_json_dict(headers_path)
    srv_map = all_headers.get(server_name)
    if not isinstance(srv_map, Mapping):
        return {}
    return {
        str(k): str(v)
        for k, v in srv_map.items()
        if isinstance(k, str) and isinstance(v, str)
    }


def read_wrap_manifest(state_dir: Path | str) -> dict[str, Any]:
    """Read `<state_dir>/wrapped_configs.json` (`0600`) for `status` reporting."""
    manifest_path = Path(state_dir).expanduser() / "wrapped_configs.json"
    data = _load_private_json_dict(manifest_path)
    if not data:
        return {"files": {}, "skipped_oauth_servers": [], "warnings": []}
    return {
        "files": data.get("files") if isinstance(data.get("files"), dict) else {},
        "skipped_oauth_servers": list(data.get("skipped_oauth_servers", []))
        if isinstance(data.get("skipped_oauth_servers"), list)
        else [],
        "warnings": list(data.get("warnings", []))
        if isinstance(data.get("warnings"), list)
        else [],
    }


def _atomic_write_config_bytes(
    target_path: Path,
    data: bytes,
    *,
    preserve_mode: int | None = None,
) -> None:
    """Atomically write `target_path` via sibling `.tmp` + `fsync` + `os.replace` (`I15`)."""
    if target_path.is_symlink():
        raise TransportAuthError(f"Refusing to modify symlinked IDE config: {target_path}")
    parent = target_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise TransportAuthError(
            f"Refusing to write inside symlinked IDE config directory: {parent}"
        )

    eff_mode = preserve_mode if preserve_mode is not None else 0o600
    tmp_path = parent / f".{target_path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, eff_mode)
    try:
        view = memoryview(data)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        os.fsync(fd)
        if os.name == "posix" and hasattr(os, "fchmod"):
            os.fchmod(fd, eff_mode)
    except BaseException:
        os.close(fd)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    else:
        os.close(fd)

    try:
        if os.name == "posix":
            os.chmod(tmp_path, eff_mode)
        os.replace(tmp_path, target_path)
        if os.name == "posix":
            os.chmod(target_path, eff_mode)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_existing_config(path: Path) -> tuple[bool, bytes, dict[str, Any], int | None]:
    if path.is_symlink():
        raise TransportAuthError(f"Refusing symlinked IDE config file: {path}")
    if not path.exists():
        return False, b"", {}, None
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise TransportAuthError(f"Expected regular IDE config file at {path}")
    mode = stat.S_IMODE(st.st_mode)
    raw_bytes = path.read_bytes()
    if not raw_bytes.strip():
        return True, raw_bytes, {}, mode
    parsed = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"IDE config root must be a JSON object: {path}")
    return True, raw_bytes, parsed, mode


def _is_diodemcp_shim_entry(entry: Any) -> bool:
    if not isinstance(entry, Mapping):
        return False
    raw_args = entry.get("args")
    if not isinstance(raw_args, Sequence) or isinstance(raw_args, (str, bytes)):
        return False
    args_list = [str(a) for a in raw_args]
    if len(args_list) >= 3 and args_list[0] == "shim" and "--server" in args_list:
        return True
    return False


def _is_diodemcp_hook_cmd(cmd: Any) -> bool:
    if not isinstance(cmd, str):
        return False
    tokens = cmd.strip().split()
    for idx in range(len(tokens) - 2):
        tok_base = os.path.basename(tokens[idx]).lower()
        if (
            "diodemcp" in tok_base
            and tokens[idx + 1] == "hook"
            and tokens[idx + 2] in ("claude", "cursor", "agy")
        ):
            return True
    return False


def _reconstruct_unwrapped_server_from_shim(
    server_name: str,
    shim_entry: Mapping[str, Any],
    shim_headers: Mapping[str, Any],
) -> dict[str, Any]:
    """Fallback reconstruction of original server config from a `diodemcp shim` entry."""
    args_list = [str(a) for a in (shim_entry.get("args") or [])]
    extra_keys = {
        k: copy.deepcopy(v)
        for k, v in shim_entry.items()
        if k not in ("command", "args")
    }
    if "--url" in args_list:
        url_idx = args_list.index("--url")
        url_val = args_list[url_idx + 1] if url_idx + 1 < len(args_list) else ""
        restored: dict[str, Any] = {**extra_keys, "url": url_val}
        saved_hdrs = shim_headers.get(server_name)
        if isinstance(saved_hdrs, Mapping) and saved_hdrs:
            restored["headers"] = {str(k): str(v) for k, v in saved_hdrs.items()}
        return restored

    if "--" in args_list:
        sep_idx = args_list.index("--")
        orig_tail = args_list[sep_idx + 1 :]
        if orig_tail:
            restored = {
                **extra_keys,
                "command": orig_tail[0],
                "args": orig_tail[1:],
            }
            return restored

    return dict(shim_entry)


def _patch_claude_hooks(
    hooks_dict: dict[str, Any],
    *,
    diodemcp_bin: str,
    state_dir: Path,
) -> None:
    for ev in CLAUDE_HOOK_EVENTS:
        cmd_str = f"{diodemcp_bin} hook claude {ev} --state-dir {state_dir}"
        ev_list = hooks_dict.get(ev)
        if not isinstance(ev_list, list):
            ev_list = []
            hooks_dict[ev] = ev_list

        updated = False
        for group in ev_list:
            if not isinstance(group, dict):
                continue
            inner = group.get("hooks")
            if not isinstance(inner, list):
                continue
            for h in inner:
                if isinstance(h, dict) and _is_diodemcp_hook_cmd(h.get("command")):
                    h["type"] = "command"
                    h["command"] = cmd_str
                    updated = True
        if not updated:
            ev_list.append(
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": cmd_str}],
                }
            )


def _unpatch_claude_hooks(
    hooks_dict: dict[str, Any],
    pre_wrap_hooks: Mapping[str, Any] | None,
) -> None:
    for ev in list(hooks_dict.keys()):
        ev_list = hooks_dict.get(ev)
        if not isinstance(ev_list, list):
            continue
        cleaned_groups: list[Any] = []
        for group in ev_list:
            if not isinstance(group, dict):
                cleaned_groups.append(group)
                continue
            inner = group.get("hooks")
            if not isinstance(inner, list):
                cleaned_groups.append(group)
                continue
            kept_inner = [
                h
                for h in inner
                if not (isinstance(h, dict) and _is_diodemcp_hook_cmd(h.get("command")))
            ]
            if kept_inner:
                new_group = dict(group)
                new_group["hooks"] = kept_inner
                cleaned_groups.append(new_group)
        if cleaned_groups:
            hooks_dict[ev] = cleaned_groups
        elif isinstance(pre_wrap_hooks, Mapping) and ev in pre_wrap_hooks:
            hooks_dict[ev] = []
        else:
            hooks_dict.pop(ev, None)


def _patch_simple_command_hooks(
    hooks_dict: dict[str, Any],
    *,
    ide_tag: str,
    events: Sequence[str],
    diodemcp_bin: str,
    state_dir: Path,
) -> None:
    for ev in events:
        cmd_str = f"{diodemcp_bin} hook {ide_tag} {ev} --state-dir {state_dir}"
        ev_list = hooks_dict.get(ev)
        if not isinstance(ev_list, list):
            ev_list = []
            hooks_dict[ev] = ev_list

        updated = False
        for item in ev_list:
            if isinstance(item, dict) and _is_diodemcp_hook_cmd(item.get("command")):
                item["command"] = cmd_str
                updated = True
        if not updated:
            ev_list.append({"command": cmd_str})


def _unpatch_simple_command_hooks(
    hooks_dict: dict[str, Any],
    pre_wrap_hooks: Mapping[str, Any] | None,
) -> None:
    for ev in list(hooks_dict.keys()):
        ev_list = hooks_dict.get(ev)
        if not isinstance(ev_list, list):
            continue
        kept = [
            item
            for item in ev_list
            if not (
                isinstance(item, dict) and _is_diodemcp_hook_cmd(item.get("command"))
            )
        ]
        if kept:
            hooks_dict[ev] = kept
        elif isinstance(pre_wrap_hooks, Mapping) and ev in pre_wrap_hooks:
            hooks_dict[ev] = []
        else:
            hooks_dict.pop(ev, None)


def _collect_target_specs(
    targets: Sequence[str],
    *,
    home_dir: Path,
    workspace_root: Path,
    include_workspace: bool,
) -> list[tuple[str, Path, bool, bool]]:
    """Return list of `(ide, config_path, has_mcp_servers, has_hooks)` to process."""
    specs: list[tuple[str, Path, bool, bool]] = []
    if "claude" in targets:
        specs.append(("claude", home_dir / ".claude.json", True, False))
        specs.append(("claude", home_dir / ".claude" / "settings.json", False, True))
        if include_workspace:
            specs.append(("claude", workspace_root / ".mcp.json", True, False))

    if "cursor" in targets:
        specs.append(("cursor", home_dir / ".cursor" / "mcp.json", True, False))
        specs.append(("cursor", home_dir / ".cursor" / "hooks.json", False, True))
        if include_workspace and (workspace_root / ".cursor" / "mcp.json").exists():
            specs.append(
                ("cursor", workspace_root / ".cursor" / "mcp.json", True, False)
            )

    if "agy" in targets:
        agy_candidates = [
            home_dir / ".gemini" / "antigravity-cli" / "mcp_config.json",
            home_dir / ".gemini" / "config" / "mcp_config.json",
            home_dir / ".gemini" / "antigravity" / "mcp_config.json",
        ]
        existing_agy = [p for p in agy_candidates if p.exists()]
        if existing_agy:
            for p in existing_agy:
                specs.append(("agy", p, True, True))
        else:
            specs.append(("agy", agy_candidates[0], True, True))

    return specs


def wrap_ide_configs(
    *,
    target: str | Sequence[str] = "all",
    state_dir: Path | str | None = None,
    home_dir: Path | str | None = None,
    workspace_root: Path | str | None = None,
    include_workspace: bool = False,
    diodemcp_bin: str = "diodemcp",
) -> dict[str, Any]:
    """Atomically backup (`0600`) and wrap target IDE MCP servers & native hooks (`I14`, `I15`)."""
    resolved_home = (
        Path(home_dir).expanduser().resolve()
        if home_dir is not None
        else Path.home().resolve()
    )
    resolved_state = ensure_private_dir(
        Path(state_dir).expanduser().resolve()
        if state_dir is not None
        else (
            resolved_home / ".diodemcp"
            if home_dir is not None
            else DEFAULT_STATE_DIR.expanduser().resolve()
        )
    )
    resolved_ws = (
        Path(workspace_root).expanduser().resolve()
        if workspace_root is not None
        else Path.cwd().resolve()
    )
    backups_dir = ensure_private_dir(resolved_state / "backups")

    manifest_path = resolved_state / "wrapped_configs.json"
    shim_headers_path = resolved_state / "shim_headers.json"

    manifest = read_wrap_manifest(resolved_state)
    files_meta: dict[str, Any] = manifest["files"]
    shim_headers: dict[str, dict[str, str]] = _load_private_json_dict(shim_headers_path)

    targets = _resolve_targets(target)
    specs = _collect_target_specs(
        targets,
        home_dir=resolved_home,
        workspace_root=resolved_ws,
        include_workspace=include_workspace,
    )

    wrapped_files: list[str] = []
    wrapped_servers: list[str] = []
    skipped_oauth_servers: list[str] = []
    warnings: list[str] = []

    for ide, cfg_path, has_mcp, has_hooks in specs:
        existed, raw_bytes, data, orig_mode = _read_existing_config(cfg_path)
        path_key = str(cfg_path)

        file_record = files_meta.get(path_key)
        if not isinstance(file_record, dict) or file_record.get("unwrapped", False):
            backup_path: str | None = None
            if existed:
                digest = hashlib.sha256(path_key.encode("utf-8")).hexdigest()[:12]
                bak_file = backups_dir / f"{digest}_{cfg_path.name}.bak"
                atomic_write_private_bytes(bak_file, raw_bytes)
                backup_path = str(bak_file)
            file_record = {
                "ide": ide,
                "path": path_key,
                "existed_before_wrap": existed,
                "backup_file": backup_path,
                "pre_wrap_data": copy.deepcopy(data),
                "wrapped_servers": {},
                "unwrapped": False,
            }
            files_meta[path_key] = file_record

        saved_srv_map: dict[str, Any] = file_record.setdefault("wrapped_servers", {})

        if has_mcp:
            mcp_servers = data.get("mcpServers")
            if isinstance(mcp_servers, dict):
                for srv_name, srv_entry in list(mcp_servers.items()):
                    if not isinstance(srv_entry, dict):
                        continue

                    if _is_diodemcp_shim_entry(srv_entry):
                        # Already shimmed: update state_dir / diodemcp_bin in place idempotently
                        srv_entry["command"] = diodemcp_bin
                        raw_args = [str(a) for a in (srv_entry.get("args") or [])]
                        if "--state-dir" in raw_args:
                            sd_idx = raw_args.index("--state-dir")
                            if sd_idx + 1 < len(raw_args):
                                raw_args[sd_idx + 1] = str(resolved_state)
                        srv_entry["args"] = raw_args
                        if srv_name not in wrapped_servers:
                            wrapped_servers.append(srv_name)
                        continue

                    raw_cmd = srv_entry.get("command")
                    if isinstance(raw_cmd, str) and raw_cmd.strip():
                        if srv_name not in saved_srv_map:
                            saved_srv_map[srv_name] = copy.deepcopy(srv_entry)
                        orig_args = [str(a) for a in (srv_entry.get("args") or [])]
                        new_entry = {
                            k: copy.deepcopy(v)
                            for k, v in srv_entry.items()
                            if k not in ("command", "args")
                        }
                        new_entry["command"] = diodemcp_bin
                        new_entry["args"] = [
                            "shim",
                            "--server",
                            srv_name,
                            "--state-dir",
                            str(resolved_state),
                            "--",
                            raw_cmd,
                            *orig_args,
                        ]
                        mcp_servers[srv_name] = new_entry
                        if srv_name not in wrapped_servers:
                            wrapped_servers.append(srv_name)
                        continue

                    url_key = (
                        "url"
                        if "url" in srv_entry
                        else ("serverUrl" if "serverUrl" in srv_entry else None)
                    )
                    if url_key is not None and isinstance(srv_entry.get(url_key), str):
                        url_val = str(srv_entry[url_key]).strip()
                        raw_hdrs = srv_entry.get("headers")
                        static_hdrs = (
                            {
                                str(k): str(v)
                                for k, v in raw_hdrs.items()
                                if isinstance(k, str)
                                and isinstance(v, str)
                                and v.strip()
                            }
                            if isinstance(raw_hdrs, Mapping)
                            else {}
                        )
                        if static_hdrs:
                            if srv_name not in saved_srv_map:
                                saved_srv_map[srv_name] = copy.deepcopy(srv_entry)
                            shim_headers[srv_name] = static_hdrs
                            new_entry = {
                                k: copy.deepcopy(v)
                                for k, v in srv_entry.items()
                                if k
                                not in (
                                    "url",
                                    "serverUrl",
                                    "headers",
                                    "command",
                                    "args",
                                )
                            }
                            new_entry["command"] = diodemcp_bin
                            new_entry["args"] = [
                                "shim",
                                "--server",
                                srv_name,
                                "--state-dir",
                                str(resolved_state),
                                "--url",
                                url_val,
                            ]
                            mcp_servers[srv_name] = new_entry
                            if srv_name not in wrapped_servers:
                                wrapped_servers.append(srv_name)
                        else:
                            # OAuth-2.1 browser-flow remote URL server: leave native (I14)
                            if srv_name not in skipped_oauth_servers:
                                skipped_oauth_servers.append(srv_name)
                            warn_str = (
                                f"Server '{srv_name}' ({url_val}): "
                                f"{PROTECTED_OAUTH_WARNING}"
                            )
                            if warn_str not in warnings:
                                warnings.append(warn_str)

        if has_hooks:
            if ide == "cursor" and "version" not in data:
                data["version"] = 1
            hooks_dict = data.get("hooks")
            if not isinstance(hooks_dict, dict):
                hooks_dict = {}
                data["hooks"] = hooks_dict

            if ide == "claude":
                _patch_claude_hooks(
                    hooks_dict,
                    diodemcp_bin=diodemcp_bin,
                    state_dir=resolved_state,
                )
            elif ide == "cursor":
                _patch_simple_command_hooks(
                    hooks_dict,
                    ide_tag="cursor",
                    events=CURSOR_HOOK_EVENTS,
                    diodemcp_bin=diodemcp_bin,
                    state_dir=resolved_state,
                )
            elif ide == "agy":
                _patch_simple_command_hooks(
                    hooks_dict,
                    ide_tag="agy",
                    events=AGY_HOOK_EVENTS,
                    diodemcp_bin=diodemcp_bin,
                    state_dir=resolved_state,
                )

        serialized = (
            json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        _atomic_write_config_bytes(cfg_path, serialized, preserve_mode=orig_mode)
        wrapped_files.append(path_key)

    atomic_write_private_json(shim_headers_path, shim_headers)
    atomic_write_private_json(
        manifest_path,
        {
            "files": files_meta,
            "skipped_oauth_servers": skipped_oauth_servers,
            "warnings": warnings,
        },
    )

    return {
        "wrapped_files": wrapped_files,
        "wrapped_servers": wrapped_servers,
        "skipped_oauth_servers": skipped_oauth_servers,
        "warnings": warnings,
        "state_dir": str(resolved_state),
    }


def unwrap_ide_configs(
    *,
    target: str | Sequence[str] = "all",
    state_dir: Path | str | None = None,
    home_dir: Path | str | None = None,
    workspace_root: Path | str | None = None,
    include_workspace: bool = False,
) -> dict[str, Any]:
    """Losslessly unwrap DiodeMCP shims & native hooks, restoring exact pre-wrap files (`I14`, `I15`)."""
    resolved_home = (
        Path(home_dir).expanduser().resolve()
        if home_dir is not None
        else Path.home().resolve()
    )
    resolved_state = ensure_private_dir(
        Path(state_dir).expanduser().resolve()
        if state_dir is not None
        else (
            resolved_home / ".diodemcp"
            if home_dir is not None
            else DEFAULT_STATE_DIR.expanduser().resolve()
        )
    )
    resolved_ws = (
        Path(workspace_root).expanduser().resolve()
        if workspace_root is not None
        else Path.cwd().resolve()
    )

    manifest_path = resolved_state / "wrapped_configs.json"
    shim_headers_path = resolved_state / "shim_headers.json"
    manifest = read_wrap_manifest(resolved_state)
    files_meta: dict[str, Any] = manifest["files"]
    shim_headers = _load_private_json_dict(shim_headers_path)

    targets = _resolve_targets(target)
    specs = _collect_target_specs(
        targets,
        home_dir=resolved_home,
        workspace_root=resolved_ws,
        include_workspace=include_workspace,
    )

    unwrapped_files: list[str] = []
    restored_servers: list[str] = []

    for ide, cfg_path, has_mcp, has_hooks in specs:
        if not cfg_path.exists():
            continue
        existed, _raw_bytes, data, orig_mode = _read_existing_config(cfg_path)
        if not existed:
            continue

        path_key = str(cfg_path)
        file_record = files_meta.get(path_key)
        pre_wrap_data = (
            file_record.get("pre_wrap_data")
            if isinstance(file_record, Mapping)
            and isinstance(file_record.get("pre_wrap_data"), dict)
            else None
        )
        saved_srv_map = (
            file_record.get("wrapped_servers")
            if isinstance(file_record, Mapping)
            and isinstance(file_record.get("wrapped_servers"), Mapping)
            else {}
        )
        existed_before_wrap = (
            bool(file_record.get("existed_before_wrap", True))
            if isinstance(file_record, Mapping)
            else True
        )
        backup_file_str = (
            file_record.get("backup_file")
            if isinstance(file_record, Mapping)
            else None
        )

        if has_mcp and isinstance(data.get("mcpServers"), dict):
            mcp_servers = data["mcpServers"]
            for srv_name, srv_entry in list(mcp_servers.items()):
                if not _is_diodemcp_shim_entry(srv_entry):
                    continue
                if srv_name in saved_srv_map:
                    mcp_servers[srv_name] = copy.deepcopy(saved_srv_map[srv_name])
                else:
                    mcp_servers[srv_name] = _reconstruct_unwrapped_server_from_shim(
                        srv_name, srv_entry, shim_headers
                    )
                if srv_name not in restored_servers:
                    restored_servers.append(srv_name)
                shim_headers.pop(srv_name, None)

        if has_hooks and isinstance(data.get("hooks"), dict):
            pre_hooks = (
                pre_wrap_data.get("hooks")
                if isinstance(pre_wrap_data, Mapping)
                and isinstance(pre_wrap_data.get("hooks"), Mapping)
                else None
            )
            if ide == "claude":
                _unpatch_claude_hooks(data["hooks"], pre_hooks)
            else:
                _unpatch_simple_command_hooks(data["hooks"], pre_hooks)

            if not data["hooks"] and (
                pre_wrap_data is None or "hooks" not in pre_wrap_data
            ):
                data.pop("hooks", None)

        if not existed_before_wrap:
            # File was created by wrap_ide_configs: delete if empty/default, else write remaining user entries
            remaining_keys = {k: v for k, v in data.items() if k != "version" or v != 1}
            if not remaining_keys:
                try:
                    cfg_path.unlink(missing_ok=True)
                except OSError:
                    pass
                if isinstance(file_record, dict):
                    file_record["unwrapped"] = True
                unwrapped_files.append(path_key)
                continue

        # Byte-faithful restore if structurally identical to pre_wrap_data and backup exists
        restored_exact_bytes = False
        if (
            pre_wrap_data is not None
            and data == pre_wrap_data
            and isinstance(backup_file_str, str)
            and backup_file_str
        ):
            bak_path = Path(backup_file_str)
            if _is_safe_0600_regular_file(bak_path):
                _atomic_write_config_bytes(
                    cfg_path,
                    bak_path.read_bytes(),
                    preserve_mode=orig_mode,
                )
                restored_exact_bytes = True

        if not restored_exact_bytes:
            serialized = (
                json.dumps(data, indent=2, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            _atomic_write_config_bytes(cfg_path, serialized, preserve_mode=orig_mode)

        if isinstance(file_record, dict):
            file_record["unwrapped"] = True
        unwrapped_files.append(path_key)

    atomic_write_private_json(shim_headers_path, shim_headers)
    atomic_write_private_json(
        manifest_path,
        {
            "files": files_meta,
            "skipped_oauth_servers": [],
            "warnings": [],
        },
    )

    return {
        "unwrapped_files": unwrapped_files,
        "restored_servers": restored_servers,
        "state_dir": str(resolved_state),
    }
