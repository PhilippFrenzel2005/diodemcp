"""Multi-domain bitflag CapabilityRegistry across T1-T7 threat surfaces."""

from __future__ import annotations

import os
import posixpath
import re
import tomllib
import urllib.parse
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from diodemcp.constants import DEFAULT_STATE_DIR
from diodemcp.types import CapabilityFlag

_WIN_EXT_UNC_RE = re.compile(r"^//\?/unc/([^/]+/[^/]+)(/.*)?$", re.IGNORECASE)
_WIN_EXT_DRIVE_RE = re.compile(r"^//[?.]/([A-Za-z]:(?:/.*)?)$")
_WIN_EXT_GENERIC_RE = re.compile(r"^//[?.]/(.+)$")
_UNC_ROOT_RE = re.compile(r"^(//[^/]+/[^/]+)(/.*)?$")
_DRIVE_ROOT_RE = re.compile(r"^([A-Za-z]:)(/.*)?$")
_FILE_URI_WIN_DRIVE_RE = re.compile(r"^/([A-Za-z]:(?:/.*)?)$")

_PATH_ARG_KEYS: frozenset[str] = frozenset(
    {
        "path",
        "file_path",
        "filepath",
        "notebook_path",
        "target_file",
        "target_path",
        "source",
        "destination",
        "src",
        "dst",
        "from_path",
        "to_path",
        "dir",
        "directory",
        "folder",
        "uri",
        "filename",
        "file",
        "paths",
        "files",
    }
)

_SQL_ARG_KEYS: frozenset[str] = frozenset(
    {"sql", "query", "statement", "command", "queries"}
)

_SHELL_ARG_KEYS: frozenset[str] = frozenset(
    {"command", "cmd", "command_line", "commandline", "script"}
)

_FILE_BODY_ARG_KEYS: frozenset[str] = frozenset(
    {
        "content",
        "contents",
        "new_string",
        "old_string",
        "new_str",
        "old_str",
        "new_source",
        "diff",
        "patch",
    }
)

_LOCAL_FILE_WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "write",
        "writefile",
        "write_file",
        "edit",
        "edit_file",
        "multiedit",
        "multi_edit",
        "str_replace",
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "replace",
        "notebookedit",
        "notebook_edit",
        "create_file",
        "delete_file",
        "remove_file",
        "move_file",
        "rename_file",
        "copy_file",
        "patch_file",
        "create_directory",
    }
)

_GENERIC_HOST_SERVERS: frozenset[str] = frozenset(
    {
        "",
        "claude",
        "claude_code",
        "claudecode",
        "cursor",
        "agy",
        "antigravity",
        "vscode",
        "ide",
        "native",
        "local",
        "hook",
        "builtin",
    }
)

_SERVER_ALIASES: dict[str, str] = {
    "brave-search": "brave_search",
    "bravesearch": "brave_search",
    "github-mcp": "github",
    "gitlab-mcp": "gitlab",
    "gworkspace-mcp": "gworkspace",
    "google-workspace": "gworkspace",
    "google_workspace": "gworkspace",
    "google-drive": "gdrive",
    "postgresql": "postgres",
    "sqlite3": "sqlite",
}

# T1: High-confidence secret headers and token prefixes for output_text promotion.
_OUTPUT_SECRET_RE = re.compile(
    r"(?:"
    r"-----BEGIN (?:[A-Z0-9 ]{1,32} )?PRIVATE KEY-----"
    r"|\bsk_(?:live|test)_[A-Za-z0-9]{8,}"
    r"|\brk_(?:live|test)_[A-Za-z0-9]{8,}"
    r"|\bwhsec_[A-Za-z0-9]{12,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{12,}"
    r"|\bgithub_pat_[A-Za-z0-9_]{16,}"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\bASIA[0-9A-Z]{16}\b"
    r"|\bAIza[0-9A-Za-z_-]{20,}"
    r"|\bxox[baprs]-[0-9A-Za-z-]{10,}"
    r"|\bglpat-[0-9A-Za-z_-]{16,}"
    r"|\bnpm_[A-Za-z0-9]{20,}"
    r"|\bpypi-[A-Za-z0-9_-]{20,}"
    r")",
)

# C4: CLI self-tamper and daemon termination patterns.
_SELF_TAMPER_CLI_RE = re.compile(
    r"(?:"
    r"\bdiodemcp(?:\.(?:exe|mjs|py))?\s+(?:allow|unwrap|model)\b"
    r"|\b(?:pkill|killall|pgrep|skill)\b[^\n;|&]{0,200}\bdiodemcp\b"
    r"|\btaskkill(?:\.exe)?\b[^\n;|&]{0,200}\bdiodemcp(?:\.exe)?\b"
    r")",
    re.IGNORECASE,
)
_KILL_VERB_RE = re.compile(r"\b(?:kill|pkill|killall|taskkill)\b", re.IGNORECASE)
_DIODEMCP_WORD_RE = re.compile(r"\bdiodemcp(?:\.exe)?\b", re.IGNORECASE)

_SELF_TAMPER_INLINE_RE = re.compile(
    r"(?:^|[~/\\\"'\s;|&><=()])(?:"
    r"\.diodemcp(?:[/\\\"'\s;|&><=()]|$)"
    r"|(?:engine\.sock|engine\.addr|admin\.secret)(?:[\"'\s;|&><=()]|$)"
    r"|\.claude\.json(?:[\"'\s;|&><=()]|$)"
    r"|\.claude(?:[/\\\"'\s;|&><=()]|$)"
    r"|\.mcp\.json(?:[\"'\s;|&><=()]|$)"
    r"|\.cursor[/\\](?:mcp|hooks)\.json(?:[\"'\s;|&><=()]|$)"
    r"|\.vscode[/\\]mcp\.json(?:[\"'\s;|&><=()]|$)"
    r"|mcp_config\.json(?:[\"'\s;|&><=()]|$)"
    r")",
    re.IGNORECASE,
)

# T5: SQL comment stripping & mutating statement detection.
_SQL_BLOCK_COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/")
_SQL_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_SQL_MUTATION_RE = re.compile(
    r"\b(?:DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT|REPLACE|MERGE|GRANT|REVOKE|CREATE|COPY|CALL|EXEC|EXECUTE|ATTACH|DETACH)\b",
    re.IGNORECASE,
)
_SQL_READ_RE = re.compile(
    r"\b(?:SELECT|WITH|SHOW|DESCRIBE|DESC|EXPLAIN|PRAGMA|VALUES)\b",
    re.IGNORECASE,
)

# Opus I1 / T2 / T6: Shell outbound network & destructive/persistence command classifiers.
_SHELL_OUTBOUND_RE = re.compile(
    r"(?:"
    r"\b(?:curl|wget|httpie|xh|scp|sftp|rsync|ssh|nc|ncat|netcat|socat|telnet|ftp|tftp|twine)\b"
    r"|\bgh\s+(?:issue|pr|gist|release|api|repo)\b"
    r"|\bglab\s+(?:issue|mr|snippet|release|api)\b"
    r"|\bgit\s+(?:push|send-email|request-pull)\b"
    r"|\b(?:npm|pnpm|yarn|poetry|uv|cargo)\s+publish\b"
    r"|\bgem\s+push\b"
    r"|\bdocker\s+push\b"
    r"|\b(?:dig|nslookup|host)\s+\S+"
    r"|/dev/(?:tcp|udp)/"
    r"|\b(?:Invoke-WebRequest|Invoke-RestMethod)\b"
    r")",
    re.IGNORECASE,
)

_SHELL_DESTRUCTIVE_RE = re.compile(
    r"(?:"
    r"\brm\s+(?:-[A-Za-z]*[rRf][A-Za-z]*\s+){1,}"
    r"|\brm\s+-[rRf]{2,}\b"
    r"|\bgit\s+push\b[^\n;|&]{0,120}(?:--force-with-lease|--force|-f)\b"
    r"|\bgit\s+reset\s+--hard\b"
    r"|\bgit\s+clean\s+-[A-Za-z]*f[A-Za-z]*\b"
    r"|\bmkfs(?:\.\w+)?\b"
    r"|\bdd\s+if="
    r"|\bchmod\s+(?:[0-7]*7[0-7]{2}|[ugoa]*\+s)\b"
    r"|\b(?:crontab|visudo|useradd|usermod|passwd)\b"
    r"|\b(?:systemctl|launchctl)\s+(?:enable|start|stop|disable|daemon-reload)\b"
    r"|\b(?:ba|z|da|k)?sh\s+-c\s+[\"']?\$\((?:curl|wget)\b"
    r"|\beval\s+[\"']?\$\((?:curl|wget)\b"
    r")",
    re.IGNORECASE,
)

_PIPE_FETCH_STAGE_RE = re.compile(r"\b(?:curl|wget|base64)\b", re.IGNORECASE)
_PIPE_EXEC_STAGE_RE = re.compile(
    r"^\s*(?:sudo\s+)?(?:env\s+)?(?:ba|z|da|k|fi)?sh\b|^\s*(?:sudo\s+)?(?:python3?|perl|ruby|node)\b",
    re.IGNORECASE,
)

_SHELL_WRITE_HINT_RE = re.compile(
    r"(?:>>?|\b(?:tee|cp|mv|install|ln|chmod|chown|touch|mkdir|truncate|dd)\b|\bsed\s+-i\b|\bperl\s+-i\b|\bopen\s*\()",
    re.IGNORECASE,
)

_SHELL_TOKEN_SPLIT_RE = re.compile(r"[\s;|&><=()$`]+")


def _resolve_posix_with_realpath(abs_posix_path: str) -> str:
    """Resolve symlinks on deepest existing prefix while preserving lexical `..` fallback."""
    current = re.sub(r"^/{2,}", "/", abs_posix_path)
    for _ in range(4):
        probe = current
        tail_parts: list[str] = []
        while probe and probe != "/" and not os.path.lexists(probe):
            parent, leaf = posixpath.split(probe)
            if parent == probe:
                break
            if leaf:
                tail_parts.append(leaf)
            probe = parent

        if probe and os.path.lexists(probe):
            real_base = os.path.realpath(probe).replace("\\", "/")
        else:
            real_base = probe or "/"

        combined = (
            posixpath.join(real_base, *reversed(tail_parts)) if tail_parts else real_base
        )
        normalized = posixpath.normpath(combined)
        if normalized.startswith("//"):
            normalized = "/" + normalized.lstrip("/")
        if normalized == current:
            break
        current = normalized
    return current


def _normalize_slashed_prefix(raw_path: str) -> str:
    cleaned = raw_path.strip()
    if not cleaned:
        return ""

    if cleaned.lower().startswith("file://"):
        parsed = urllib.parse.urlparse(cleaned)
        uri_path = urllib.parse.unquote(parsed.path or "")
        win_drive = _FILE_URI_WIN_DRIVE_RE.match(uri_path)
        if win_drive:
            cleaned = win_drive.group(1)
        elif parsed.netloc and parsed.netloc.lower() != "localhost":
            cleaned = f"//{parsed.netloc}{uri_path}"
        else:
            cleaned = uri_path

    slashed = cleaned.replace("\\", "/")

    m_ext_unc = _WIN_EXT_UNC_RE.match(slashed)
    if m_ext_unc:
        slashed = f"//{m_ext_unc.group(1)}{m_ext_unc.group(2) or ''}"
    else:
        m_ext_drive = _WIN_EXT_DRIVE_RE.match(slashed)
        if m_ext_drive:
            slashed = m_ext_drive.group(1)
        else:
            m_ext_gen = _WIN_EXT_GENERIC_RE.match(slashed)
            if m_ext_gen:
                slashed = m_ext_gen.group(1)

    if slashed == "~" or slashed.startswith("~/") or slashed.startswith("~"):
        slashed = os.path.expanduser(slashed).replace("\\", "/")

    return slashed


def _lexical_normalize_target_path(
    raw_path: str,
    workspace_root: Path | None = None,
    *,
    casefold: bool = False,
) -> str:
    """Normalize `~`, slashes, and `..` without resolving filesystem symlinks."""
    slashed = _normalize_slashed_prefix(raw_path)
    if not slashed:
        return ""

    m_unc = _UNC_ROOT_RE.match(slashed)
    if m_unc:
        unc_root = m_unc.group(1)
        unc_rest = posixpath.normpath(m_unc.group(2) or "/")
        result = unc_root if unc_rest == "/" else f"{unc_root}{unc_rest}"
        return result.casefold() if casefold else result

    m_drive = _DRIVE_ROOT_RE.match(slashed)
    if m_drive:
        drive = m_drive.group(1)
        drive_rest = posixpath.normpath(m_drive.group(2) or "/")
        result = f"{drive}/" if drive_rest == "/" else f"{drive}{drive_rest}"
        return result.casefold() if casefold else result

    if slashed.startswith("/"):
        collapsed = re.sub(r"^/{2,}", "/", slashed)
        result = posixpath.normpath(collapsed)
        if result.startswith("//"):
            result = "/" + result.lstrip("/")
        return result.casefold() if casefold else result

    if workspace_root is not None:
        ws_lex = _lexical_normalize_target_path(str(workspace_root), workspace_root=None)
        joined = f"{ws_lex.rstrip('/')}/{slashed}"
        return _lexical_normalize_target_path(joined, workspace_root=None, casefold=casefold)

    result = posixpath.normpath(slashed)
    return result.casefold() if casefold else result


def canonicalize_target_path(
    raw_path: str,
    workspace_root: Path | None = None,
    *,
    casefold: bool = False,
) -> str:
    """Canonicalize file/target paths across POSIX, Windows/UNC, symlinks, and `..` segments."""
    slashed = _normalize_slashed_prefix(raw_path)
    if not slashed:
        return ""

    m_unc = _UNC_ROOT_RE.match(slashed)
    if m_unc:
        unc_root = m_unc.group(1)
        unc_rest = posixpath.normpath(m_unc.group(2) or "/")
        result = unc_root if unc_rest == "/" else f"{unc_root}{unc_rest}"
        return result.casefold() if casefold else result

    m_drive = _DRIVE_ROOT_RE.match(slashed)
    if m_drive:
        drive = m_drive.group(1)
        drive_rest = posixpath.normpath(m_drive.group(2) or "/")
        result = f"{drive}/" if drive_rest == "/" else f"{drive}{drive_rest}"
        return result.casefold() if casefold else result

    if slashed.startswith("/"):
        result = _resolve_posix_with_realpath(slashed)
        return result.casefold() if casefold else result

    if workspace_root is not None:
        ws_canon = canonicalize_target_path(str(workspace_root), workspace_root=None)
        joined = f"{ws_canon.rstrip('/')}/{slashed}"
        return canonicalize_target_path(joined, workspace_root=None, casefold=casefold)

    result = posixpath.normpath(slashed)
    return result.casefold() if casefold else result


def _glob_body_to_regex(glob_expr: str) -> str:
    """Translate a normalized `/` glob body (`**`, `*`, `?`) into a linear-time regex."""
    i = 0
    n = len(glob_expr)
    parts: list[str] = []
    while i < n:
        if glob_expr.startswith("**/", i):
            parts.append("(?:.+/)?")
            i += 3
        elif glob_expr.startswith("/**", i) and i + 3 == n:
            parts.append("(?:/.*)?")
            i += 3
        elif glob_expr.startswith("**", i):
            parts.append(".*")
            i += 2
        elif glob_expr[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif glob_expr[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(glob_expr[i]))
            i += 1
    return "".join(parts)


def _compile_single_folded_glob(folded: str) -> re.Pattern[str]:
    is_anchored = (
        folded.startswith("/")
        or folded.startswith("//")
        or bool(_DRIVE_ROOT_RE.match(folded))
    )
    if folded.startswith("**/"):
        body = _glob_body_to_regex(folded[3:])
        return re.compile(f"^(?:.+/)?{body}$")
    if is_anchored:
        body = _glob_body_to_regex(folded)
        return re.compile(f"^{body}$")
    body = _glob_body_to_regex(folded)
    return re.compile(f"(?:^|/){body}$")


def _compile_path_globs(pattern: str) -> list[re.Pattern[str]]:
    """Compile a path glob into case-folded regex(es), resolving symlinked anchored prefixes."""
    norm = pattern.strip().replace("\\", "/")
    if norm == "~" or norm.startswith("~/"):
        norm = os.path.expanduser(norm).replace("\\", "/")

    compiled = [_compile_single_folded_glob(norm.casefold())]

    if norm.startswith("/"):
        wildcard_pos = min(
            (idx for idx in (norm.find("*"), norm.find("?")) if idx != -1),
            default=-1,
        )
        if wildcard_pos == -1:
            resolved_full = canonicalize_target_path(norm)
            if resolved_full.casefold() != norm.casefold():
                compiled.append(_compile_single_folded_glob(resolved_full.casefold()))
        elif wildcard_pos > 1:
            prefix = norm[:wildcard_pos].rstrip("/")
            suffix = norm[len(prefix) :]
            if prefix:
                resolved_prefix = canonicalize_target_path(prefix)
                resolved_pattern = f"{resolved_prefix.rstrip('/')}{suffix}"
                if resolved_pattern.casefold() != norm.casefold():
                    compiled.append(_compile_single_folded_glob(resolved_pattern.casefold()))

    return compiled


def _normalize_server_and_tool(server: str, tool: str) -> tuple[str, str]:
    """Normalize MCP and IDE native `(server, tool)` identifiers into canonical lowercase."""
    srv = server.strip().lower()
    tl = tool.strip().lower()

    if tl.startswith("mcp__"):
        parts = tl.split("__", 2)
        if len(parts) == 3 and parts[1] and parts[2]:
            if srv in _GENERIC_HOST_SERVERS:
                srv = parts[1]
            tl = parts[2]
    elif "." in tl:
        prefix, suffix = tl.split(".", 1)
        if suffix and (srv in _GENERIC_HOST_SERVERS or srv == prefix):
            srv = prefix
            tl = suffix

    srv = _SERVER_ALIASES.get(srv, srv.replace("-", "_"))
    srv = _SERVER_ALIASES.get(srv, srv)
    return srv, tl


def _collect_strings_for_self_tamper(
    value: Any,
    out: list[str],
    *,
    skip_file_body_keys: bool,
) -> None:
    if isinstance(value, str):
        if value:
            out.append(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            key_lower = k.lower() if isinstance(k, str) else ""
            if skip_file_body_keys and key_lower in _FILE_BODY_ARG_KEYS:
                continue
            if isinstance(k, str) and k:
                out.append(k)
            _collect_strings_for_self_tamper(
                v, out, skip_file_body_keys=skip_file_body_keys
            )
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_strings_for_self_tamper(
                item, out, skip_file_body_keys=skip_file_body_keys
            )


def _extract_path_values(args: dict[str, Any]) -> list[str]:
    found: list[str] = []

    def _visit_path_val(val: Any) -> None:
        if isinstance(val, str) and val.strip():
            found.append(val.strip())
        elif isinstance(val, (list, tuple, set)):
            for elem in val:
                _visit_path_val(elem)
        elif isinstance(val, dict):
            for k, v in val.items():
                if isinstance(k, str) and k.lower() in _PATH_ARG_KEYS:
                    _visit_path_val(v)

    def _walk_mapping(mapping: dict[str, Any]) -> None:
        for key, val in mapping.items():
            if key.lower() in _PATH_ARG_KEYS:
                _visit_path_val(val)
            elif isinstance(val, dict):
                _walk_mapping(val)
            elif isinstance(val, (list, tuple)):
                for elem in val:
                    if isinstance(elem, dict):
                        _walk_mapping(elem)

    _walk_mapping(args)
    return found


def _extract_keyed_strings(args: dict[str, Any], target_keys: frozenset[str]) -> list[str]:
    out: list[str] = []
    for key, val in args.items():
        if key.lower() in target_keys:
            if isinstance(val, str) and val.strip():
                out.append(val)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, str) and item.strip():
                        out.append(item)
    return out


def _extract_shell_path_candidates(command: str) -> list[str]:
    candidates: list[str] = []
    for raw_tok in _SHELL_TOKEN_SPLIT_RE.split(command):
        tok = raw_tok.strip().strip("\"'")
        if not tok or tok.startswith("-") or "://" in tok:
            continue
        if "/" in tok or "\\" in tok or tok.startswith("~") or "." in tok:
            candidates.append(tok)
    return candidates


def _strip_sql_comments(sql: str) -> str:
    without_block = _SQL_BLOCK_COMMENT_RE.sub(" ", sql)
    return _SQL_LINE_COMMENT_RE.sub(" ", without_block)


def _has_pipe_to_shell(command: str) -> bool:
    for line in command.splitlines():
        if "|" not in line:
            continue
        stages = line.split("|")
        for idx in range(len(stages) - 1):
            if _PIPE_FETCH_STAGE_RE.search(stages[idx]) and any(
                _PIPE_EXEC_STAGE_RE.search(later) for later in stages[idx + 1 :]
            ):
                return True
    return False


@dataclass
class _DomainMatcher:
    path_patterns: list[re.Pattern[str]] = field(default_factory=list)
    high_secret_path_patterns: list[re.Pattern[str]] = field(default_factory=list)
    servers: set[str] = field(default_factory=set)
    high_secret_servers: set[str] = field(default_factory=set)
    qualified_tools: set[str] = field(default_factory=set)
    standalone_tools: set[str] = field(default_factory=set)
    high_secret_qualified_tools: set[str] = field(default_factory=set)
    high_secret_standalone_tools: set[str] = field(default_factory=set)

    def extend_from_mapping(self, section: dict[str, Any]) -> None:
        for pat in section.get("paths", []):
            if isinstance(pat, str) and pat.strip():
                self.path_patterns.extend(_compile_path_globs(pat))
        for pat in section.get("high_secret_paths", []):
            if isinstance(pat, str) and pat.strip():
                self.high_secret_path_patterns.extend(_compile_path_globs(pat))
        for srv in section.get("servers", []):
            if isinstance(srv, str) and srv.strip():
                norm_srv, _ = _normalize_server_and_tool(srv, "")
                self.servers.add(norm_srv)
        for srv in section.get("high_secret_servers", []):
            if isinstance(srv, str) and srv.strip():
                norm_srv, _ = _normalize_server_and_tool(srv, "")
                self.high_secret_servers.add(norm_srv)
        for tl in section.get("tools", []):
            if isinstance(tl, str) and tl.strip():
                if "." in tl:
                    norm_srv, norm_tl = _normalize_server_and_tool("", tl)
                    self.qualified_tools.add(f"{norm_srv}.{norm_tl}")
                else:
                    self.standalone_tools.add(tl.strip().lower())
        for tl in section.get("high_secret_tools", []):
            if isinstance(tl, str) and tl.strip():
                if "." in tl:
                    norm_srv, norm_tl = _normalize_server_and_tool("", tl)
                    self.high_secret_qualified_tools.add(f"{norm_srv}.{norm_tl}")
                else:
                    self.high_secret_standalone_tools.add(tl.strip().lower())

    def matches_server_or_tool(self, norm_server: str, norm_tool: str) -> bool:
        return (
            norm_server in self.servers
            or f"{norm_server}.{norm_tool}" in self.qualified_tools
            or norm_tool in self.standalone_tools
        )

    def matches_high_secret_server_or_tool(self, norm_server: str, norm_tool: str) -> bool:
        return (
            norm_server in self.high_secret_servers
            or f"{norm_server}.{norm_tool}" in self.high_secret_qualified_tools
            or norm_tool in self.high_secret_standalone_tools
        )

    def matches_path(self, folded_canon_path: str) -> bool:
        return any(rx.search(folded_canon_path) is not None for rx in self.path_patterns)

    def matches_high_secret_path(self, folded_canon_path: str) -> bool:
        return any(
            rx.search(folded_canon_path) is not None for rx in self.high_secret_path_patterns
        )


class CapabilityRegistry:
    """Strict-Everywhere bitflag capability classifier across T1-T7 threat surfaces."""

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        self_protection: _DomainMatcher,
        private_sources: _DomainMatcher,
        untrusted_sources: _DomainMatcher,
        mixed_trust_sources: _DomainMatcher,
        dual_direction: _DomainMatcher,
        external_sinks: _DomainMatcher,
        sensitive_mutations: _DomainMatcher,
        db_servers: set[str],
        db_read_tools: set[str],
        db_mutating_tools: set[str],
        shell_tools: set[str],
        user_private_sources: _DomainMatcher,
    ) -> None:
        self.workspace_root = workspace_root
        self._self_protection = self_protection
        self._private_sources = private_sources
        self._untrusted_sources = untrusted_sources
        self._mixed_trust_sources = mixed_trust_sources
        self._dual_direction = dual_direction
        self._external_sinks = external_sinks
        self._sensitive_mutations = sensitive_mutations
        self._db_servers = db_servers
        self._db_read_tools = db_read_tools
        self._db_mutating_tools = db_mutating_tools
        self._shell_tools = shell_tools
        self._user_private_sources = user_private_sources

    @classmethod
    def from_default_and_user_config(
        cls,
        config_path: Path | None = None,
        workspace_root: Path | None = None,
    ) -> CapabilityRegistry:
        """Load built-in `registry_data.toml` and merge optional user `config.toml`."""
        default_toml_text = (
            resources.files("diodemcp")
            .joinpath("registry_data.toml")
            .read_text(encoding="utf-8")
        )
        default_data = tomllib.loads(default_toml_text)

        self_protection = _DomainMatcher()
        private_sources = _DomainMatcher()
        untrusted_sources = _DomainMatcher()
        mixed_trust_sources = _DomainMatcher()
        dual_direction = _DomainMatcher()
        external_sinks = _DomainMatcher()
        sensitive_mutations = _DomainMatcher()
        user_private_sources = _DomainMatcher()

        self_protection.extend_from_mapping(default_data.get("self_protection", {}))
        private_sources.extend_from_mapping(default_data.get("private_sources", {}))
        untrusted_sources.extend_from_mapping(default_data.get("untrusted_sources", {}))
        mixed_trust_sources.extend_from_mapping(default_data.get("mixed_trust_sources", {}))
        dual_direction.extend_from_mapping(default_data.get("dual_direction", {}))
        external_sinks.extend_from_mapping(default_data.get("external_sinks", {}))
        sensitive_mutations.extend_from_mapping(default_data.get("sensitive_mutations", {}))

        db_section = default_data.get("database", {})
        db_servers = {
            _normalize_server_and_tool(s, "")[0]
            for s in db_section.get("servers", [])
            if isinstance(s, str)
        }
        db_read_tools = {
            t.strip().lower() for t in db_section.get("read_tools", []) if isinstance(t, str)
        }
        db_mutating_tools = {
            t.strip().lower()
            for t in db_section.get("mutating_tools", [])
            if isinstance(t, str)
        }

        shell_section = default_data.get("shell", {})
        shell_tools = {
            t.strip().lower() for t in shell_section.get("tools", []) if isinstance(t, str)
        }

        resolved_config = (
            config_path
            if config_path is not None
            else (DEFAULT_STATE_DIR / "config.toml")
        )
        if resolved_config.is_file():
            user_data = tomllib.loads(resolved_config.read_text(encoding="utf-8"))
            if isinstance(user_data.get("private_sources"), dict):
                private_sources.extend_from_mapping(user_data["private_sources"])
                user_private_sources.extend_from_mapping(user_data["private_sources"])
            if isinstance(user_data.get("untrusted_sources"), dict):
                untrusted_sources.extend_from_mapping(user_data["untrusted_sources"])
            if isinstance(user_data.get("external_sinks"), dict):
                external_sinks.extend_from_mapping(user_data["external_sinks"])
            if isinstance(user_data.get("sensitive_mutations"), dict):
                sensitive_mutations.extend_from_mapping(user_data["sensitive_mutations"])

        return cls(
            workspace_root=workspace_root,
            self_protection=self_protection,
            private_sources=private_sources,
            untrusted_sources=untrusted_sources,
            mixed_trust_sources=mixed_trust_sources,
            dual_direction=dual_direction,
            external_sinks=external_sinks,
            sensitive_mutations=sensitive_mutations,
            db_servers=db_servers,
            db_read_tools=db_read_tools,
            db_mutating_tools=db_mutating_tools,
            shell_tools=shell_tools,
            user_private_sources=user_private_sources,
        )

    def _canonical_folded_paths(self, raw_paths: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for p in raw_paths:
            if not p.strip():
                continue
            for view in (
                canonicalize_target_path(p, self.workspace_root, casefold=True),
                _lexical_normalize_target_path(p, self.workspace_root, casefold=True),
            ):
                if view and view not in seen:
                    seen.add(view)
                    out.append(view)
        return out

    def _is_shell_call(self, norm_server: str, norm_tool: str) -> bool:
        return norm_tool in self._shell_tools or norm_server in {"bash", "shell", "terminal"}

    def _is_file_read_call(self, norm_server: str, norm_tool: str) -> bool:
        if norm_tool in _LOCAL_FILE_WRITE_TOOLS:
            return False
        if norm_tool in self._private_sources.standalone_tools:
            return True
        return norm_server in {"filesystem", "files"} and norm_tool not in _LOCAL_FILE_WRITE_TOOLS

    def _is_file_write_call(self, norm_server: str, norm_tool: str) -> bool:
        return norm_tool in _LOCAL_FILE_WRITE_TOOLS

    def is_unconditional_critical_self_tamper(
        self,
        server: str,
        tool: str,
        args: dict[str, Any],
    ) -> bool:
        """Return True when a call reads, mutates, or kills DiodeMCP or IDE hook/MCP configs (`C4`)."""
        norm_server, norm_tool = _normalize_server_and_tool(server, tool)
        raw_paths = _extract_path_values(args)
        for folded in self._canonical_folded_paths(raw_paths):
            if self._self_protection.matches_path(folded):
                return True

        all_strings: list[str] = []
        _collect_strings_for_self_tamper(
            args,
            all_strings,
            skip_file_body_keys=self._is_file_write_call(norm_server, norm_tool),
        )
        for text in all_strings:
            for candidate in (text, urllib.parse.unquote(text)):
                if _SELF_TAMPER_CLI_RE.search(candidate) is not None:
                    return True
                for line in candidate.splitlines():
                    if _KILL_VERB_RE.search(line) and _DIODEMCP_WORD_RE.search(line):
                        return True
                if _SELF_TAMPER_INLINE_RE.search(candidate) is not None:
                    return True
                for folded_tok in self._canonical_folded_paths(
                    _extract_shell_path_candidates(candidate)
                ):
                    if self._self_protection.matches_path(folded_tok):
                        return True
        return False

    def classify_request(
        self,
        server: str,
        tool: str,
        args: dict[str, Any],
    ) -> CapabilityFlag:
        """Classify pre-execution sink and sensitive-mutation capabilities for a proposed call."""
        norm_server, norm_tool = _normalize_server_and_tool(server, tool)
        flags = CapabilityFlag.LOCAL_SAFE

        if self.is_unconditional_critical_self_tamper(server, tool, args):
            flags |= CapabilityFlag.MUTATES_SENSITIVE | CapabilityFlag.HIGH_SECRET

        folded_paths = self._canonical_folded_paths(_extract_path_values(args))

        # Dual-direction web/search/browser tools send URL/query externally on request.
        if self._dual_direction.matches_server_or_tool(norm_server, norm_tool):
            flags |= CapabilityFlag.WRITES_EXTERNAL

        # T2 / T7: Explicit external sinks (zero exemption for git remote origin).
        if self._external_sinks.matches_server_or_tool(norm_server, norm_tool):
            flags |= CapabilityFlag.WRITES_EXTERNAL

        # T7 / T6: Explicit sensitive mutation servers/tools.
        if self._sensitive_mutations.matches_server_or_tool(norm_server, norm_tool):
            flags |= CapabilityFlag.MUTATES_SENSITIVE

        # T5: Database mutating SQL queries vs read-only SELECT queries.
        if norm_server in self._db_servers:
            if norm_tool in self._db_mutating_tools:
                flags |= CapabilityFlag.MUTATES_SENSITIVE
            for sql in _extract_keyed_strings(args, _SQL_ARG_KEYS):
                stripped_sql = _strip_sql_comments(sql)
                if _SQL_MUTATION_RE.search(stripped_sql) is not None:
                    flags |= CapabilityFlag.MUTATES_SENSITIVE

        # Opus I1 / T2 / T6: Shell command outbound network & destructive/persistence mutations.
        if self._is_shell_call(norm_server, norm_tool):
            shell_cmds = _extract_keyed_strings(args, _SHELL_ARG_KEYS)
            for cmd in shell_cmds:
                if _SHELL_OUTBOUND_RE.search(cmd) is not None:
                    flags |= CapabilityFlag.WRITES_EXTERNAL
                if _SHELL_DESTRUCTIVE_RE.search(cmd) is not None or _has_pipe_to_shell(cmd):
                    flags |= CapabilityFlag.MUTATES_SENSITIVE
                stripped_cmd_sql = _strip_sql_comments(cmd)
                if re.search(
                    r"\b(?:psql|sqlite3|mysql)\b", cmd, re.IGNORECASE
                ) and _SQL_MUTATION_RE.search(stripped_cmd_sql):
                    flags |= CapabilityFlag.MUTATES_SENSITIVE
                if _SHELL_WRITE_HINT_RE.search(cmd) is not None:
                    for folded_tok in self._canonical_folded_paths(
                        _extract_shell_path_candidates(cmd)
                    ):
                        if self._sensitive_mutations.matches_path(folded_tok):
                            flags |= CapabilityFlag.MUTATES_SENSITIVE
                        if self._external_sinks.matches_path(folded_tok):
                            flags |= CapabilityFlag.WRITES_EXTERNAL

        # T6 / T7: Local file write/edit calls inspect target paths for sensitive mutations & external sinks.
        if self._is_file_write_call(norm_server, norm_tool) or (
            norm_server in {"filesystem", "files"} and not self._is_file_read_call(norm_server, norm_tool)
        ):
            for fp in folded_paths:
                if self._sensitive_mutations.matches_path(fp):
                    flags |= CapabilityFlag.MUTATES_SENSITIVE
                if self._external_sinks.matches_path(fp):
                    flags |= CapabilityFlag.WRITES_EXTERNAL
        else:
            # User-configured external sink / sensitive mutation paths still apply to custom write tools.
            for fp in folded_paths:
                if self._external_sinks.matches_path(fp):
                    flags |= CapabilityFlag.WRITES_EXTERNAL

        return flags

    def classify_response(
        self,
        server: str,
        tool: str,
        args: dict[str, Any],
        output_text: str | None = None,
    ) -> CapabilityFlag:
        """Classify post-execution private, high-secret, and untrusted taint for a tool output."""
        norm_server, norm_tool = _normalize_server_and_tool(server, tool)
        flags = CapabilityFlag.LOCAL_SAFE

        if self.is_unconditional_critical_self_tamper(server, tool, args):
            flags |= (
                CapabilityFlag.READS_PRIVATE
                | CapabilityFlag.HIGH_SECRET
                | CapabilityFlag.MUTATES_SENSITIVE
            )

        folded_paths = self._canonical_folded_paths(_extract_path_values(args))
        is_outbound_sink_tool = self._external_sinks.matches_server_or_tool(
            norm_server, norm_tool
        )

        # Dual-direction web/search/browser tools ingest untrusted external content on response.
        if self._dual_direction.matches_server_or_tool(norm_server, norm_tool):
            flags |= CapabilityFlag.READS_UNTRUSTED

        # Opus I1: Shell outputs default to dual-trust (READS_PRIVATE | READS_UNTRUSTED).
        if self._is_shell_call(norm_server, norm_tool):
            flags |= CapabilityFlag.READS_PRIVATE | CapabilityFlag.READS_UNTRUSTED

        # T3 & T4: Mixed-trust servers & tools (unless this specific tool is purely an outbound write).
        if not is_outbound_sink_tool and self._mixed_trust_sources.matches_server_or_tool(
            norm_server, norm_tool
        ):
            flags |= CapabilityFlag.READS_PRIVATE | CapabilityFlag.READS_UNTRUSTED

        # T3: Database SELECT / read queries ingest private data on response.
        if norm_server in self._db_servers:
            sql_list = _extract_keyed_strings(args, _SQL_ARG_KEYS)
            if sql_list:
                for sql in sql_list:
                    stripped = _strip_sql_comments(sql)
                    if _SQL_READ_RE.search(stripped) is not None or (
                        _SQL_MUTATION_RE.search(stripped) is None
                        and norm_tool in self._db_read_tools
                    ):
                        flags |= CapabilityFlag.READS_PRIVATE
            elif norm_tool in self._db_read_tools:
                flags |= CapabilityFlag.READS_PRIVATE

        # T1 / T2 / Opus I1: Local & workspace file reads (Strict Everywhere: all file reads -> READS_PRIVATE).
        if self._is_file_read_call(norm_server, norm_tool):
            flags |= CapabilityFlag.READS_PRIVATE
            for fp in folded_paths:
                if self._private_sources.matches_high_secret_path(fp):
                    flags |= CapabilityFlag.READS_PRIVATE | CapabilityFlag.HIGH_SECRET
                if self._untrusted_sources.matches_path(fp):
                    flags |= CapabilityFlag.READS_UNTRUSTED
        elif not self._is_file_write_call(norm_server, norm_tool):
            # Non-file-write tools: apply server/tool rules and user-configured path rules (T7).
            if not is_outbound_sink_tool:
                if self._private_sources.matches_server_or_tool(norm_server, norm_tool):
                    flags |= CapabilityFlag.READS_PRIVATE
                if self._private_sources.matches_high_secret_server_or_tool(
                    norm_server, norm_tool
                ):
                    flags |= CapabilityFlag.READS_PRIVATE | CapabilityFlag.HIGH_SECRET
                if self._untrusted_sources.matches_server_or_tool(norm_server, norm_tool):
                    flags |= CapabilityFlag.READS_UNTRUSTED

            for fp in folded_paths:
                if self._user_private_sources.matches_path(fp):
                    flags |= CapabilityFlag.READS_PRIVATE
                if self._private_sources.matches_high_secret_path(fp):
                    flags |= CapabilityFlag.READS_PRIVATE | CapabilityFlag.HIGH_SECRET
                if self._untrusted_sources.matches_path(fp):
                    flags |= CapabilityFlag.READS_UNTRUSTED

        # T1: Dynamic output_text secret header / prefix inspection upgrades any tool response.
        if output_text and _OUTPUT_SECRET_RE.search(output_text) is not None:
            flags |= CapabilityFlag.READS_PRIVATE | CapabilityFlag.HIGH_SECRET

        return flags
