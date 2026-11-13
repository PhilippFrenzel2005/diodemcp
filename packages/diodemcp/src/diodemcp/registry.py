"""Multi-domain bitflag CapabilityRegistry across T1-T7 threat surfaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import fnmatch
from importlib import resources
import os
from pathlib import Path
import posixpath
import re
import tomllib
from typing import Any
import urllib.parse

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
        "targetfile",
        "target_path",
        "absolute_path",
        "absolutepath",
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
        "codecontent",
        "replacementcontent",
        "targetcontent",
        "file_text",
        "new_string",
        "old_string",
        "new_str",
        "old_str",
        "newtext",
        "oldtext",
        "new_source",
        "diff",
        "patch",
    }
)

_EDITOR_MULTIPLEX_TOOLS: frozenset[str] = frozenset(
    {
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "text_editor",
    }
)

_LOCAL_FILE_WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "write",
        "writefile",
        "write_file",
        "write_to_file",
        "edit",
        "edit_file",
        "multiedit",
        "multi_edit",
        "replace_file_content",
        "multi_replace_file_content",
        "replace_in_file",
        "apply_diff",
        "apply_patch",
        "insert_content",
        "insert_edit_into_file",
        "create_new_file",
        "str_replace",
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "text_editor",
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
        "windsurf",
        "cline",
        "copilot",
        "vscode",
        "ide",
        "native",
        "local",
        "hook",
        "builtin",
    }
)

_SERVER_ALIASES: dict[str, str] = {
    "brave_search": "brave_search",
    "bravesearch": "brave_search",
    "github_mcp": "github",
    "gitlab_mcp": "gitlab",
    "gworkspace_mcp": "gworkspace",
    "google_workspace": "gworkspace",
    "google_drive": "gdrive",
    "postgresql": "postgres",
    "sqlite3": "sqlite",
}

_KNOWN_DOMAIN_LIST_KEYS: frozenset[str] = frozenset(
    {
        "paths",
        "high_secret_paths",
        "servers",
        "high_secret_servers",
        "tools",
        "high_secret_tools",
    }
)

_USER_CONFIG_DOMAIN_SECTIONS: frozenset[str] = frozenset(
    {
        "private_sources",
        "untrusted_sources",
        "external_sinks",
        "sensitive_mutations",
        "mixed_trust_sources",
        "dual_direction",
    }
)

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
_DIODEMCP_SUBCMDS = r"(?:allow|approve|unwrap|model|trust|shutdown)"
_CLI_FLAG_TOKEN = (
    r"--?[A-Za-z0-9][A-Za-z0-9_-]*"
    r"(?:=(?:\"[^\"]*\"|'[^']*'|[^\s;|&]+)"
    r"|\s+(?!" + _DIODEMCP_SUBCMDS + r"\b)(?:\"[^\"]*\"|'[^']*'|[^\s;|&-]+))?"
)
_SELF_TAMPER_CLI_RE = re.compile(
    r"(?:"
    r"(?:\bpython[0-9.]*\s+-m\s+)?"
    r"\bdiodemcp(?:\.(?:exe|mjs|py|cli))?(?:(?:@|==)[A-Za-z0-9._+-]+)?"
    r"(?:\s+" + _CLI_FLAG_TOKEN + r"){0,6}"
    r"\s+" + _DIODEMCP_SUBCMDS + r"\b"
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
    r"|\.cursor(?:[/\\\"'\s;|&><=()]|$)"
    r"|\.vscode[/\\]mcp\.json(?:[\"'\s;|&><=()]|$)"
    r"|mcp_config\.json(?:[\"'\s;|&><=()]|$)"
    r")",
    re.IGNORECASE,
)

# T5: SQL comment stripping (preserving MySQL/MariaDB /*!...*/ & /*M!...*/ comments) & DML/DDL detection.
_SQL_BLOCK_COMMENT_RE = re.compile(r"/\*(?![Mm]?!)[\s\S]*?\*/")
_MYSQL_EXEC_COMMENT_OPEN_RE = re.compile(r"/\*[Mm]?![0-9]{0,6}")
_MYSQL_EXEC_COMMENT_CLOSE_RE = re.compile(r"\*/")
_SQL_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_SQL_MUTATION_RE = re.compile(
    r"(?:"
    r"\b(?:DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT|REPLACE|MERGE|GRANT|REVOKE|CREATE|COPY|CALL|EXEC|EXECUTE|ATTACH|DETACH)\b"
    r"|\bINTO\s+(?:OUTFILE|DUMPFILE)\b"
    r")",
    re.IGNORECASE,
)
_SQL_READ_RE = re.compile(
    r"\b(?:SELECT|WITH|SHOW|DESCRIBE|DESC|EXPLAIN|PRAGMA|VALUES|RETURNING)\b",
    re.IGNORECASE,
)

_DIODEMCP_PATH_COMPONENT_RE = re.compile(
    r"(?:^|[~/\\\"'\s;|&><=():])\.diodemcp(?:[/\\\"'\s;|&><=()._-]|$)",
    re.IGNORECASE,
)

# Escalate-only dynamic metadata heuristics for unknown/unlisted tools.
_CAMEL_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")
_TOOL_VERBS_EXTERNAL: frozenset[str] = frozenset(
    {
        "send",
        "post",
        "share",
        "upload",
        "publish",
        "forward",
        "transfer",
        "pay",
        "webhook",
        "notify",
        "dispatch",
        "submit",
        "broadcast",
        "reply",
    }
)
_TOOL_VERBS_MUTATE: frozenset[str] = frozenset(
    {
        "create",
        "delete",
        "remove",
        "drop",
        "transfer",
        "pay",
        "exec",
        "execute",
        "run",
        "write",
        "update",
        "modify",
        "alter",
        "patch",
        "grant",
        "revoke",
        "reset",
    }
)
_KEY_HINTS_EXTERNAL: frozenset[str] = frozenset(
    {
        "to",
        "cc",
        "bcc",
        "recipient",
        "recipients",
        "email",
        "url",
        "uri",
        "webhook",
        "iban",
        "amount",
        "channel",
        "body",
        "message",
        "endpoint",
        "href",
        "target_url",
    }
)
_KEY_HINTS_MUTATE: frozenset[str] = frozenset(
    {
        "command",
        "cmd",
        "sql",
        "iban",
        "amount",
    }
)

_CANONICAL_IDE_TOOL_CASING: dict[str, str] = {
    "bash": "Bash",
    "bashoutput": "BashOutput",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "multiedit": "MultiEdit",
    "glob": "Glob",
    "grep": "Grep",
    "notebookread": "NotebookRead",
    "notebookedit": "NotebookEdit",
}

# Opus I1 / T2 / T6: Shell outbound network & destructive/persistence command classifiers.
_SHELL_OUTBOUND_RE = re.compile(
    r"(?:"
    r"\b(?:curl|wget|httpie|xh|scp|sftp|rsync|ssh|nc|ncat|netcat|socat|telnet|ftp|tftp|twine)\b"
    r"|\bgh\s+(?:issue|pr|gist|release|api|repo)\b"
    r"|\bglab\s+(?:issue|mr|snippet|release|api)\b"
    r"|\bgit\s+(?:push|send-email|request-pull)\b"
    r"|\bgit\s+(?:clone|fetch|ls-remote)\b[^\n;|&]*https?://(?:[^/\s:@]+:[^/\s@]*@|[^\s?#]+\?[^\s]+)"
    r"|\b(?:npm|pnpm|yarn|poetry|uv|cargo)\s+publish\b"
    r"|\bgem\s+push\b"
    r"|\bdocker\s+push\b"
    r"|(?:^|[\s;|&(])(?:dig|nslookup|host)\s+(?!(?:localhost|127\.0\.0\.1)\b)\S+"
    r"|/dev/(?:tcp|udp)/"
    r"|\b(?:Invoke-WebRequest|Invoke-RestMethod)\b"
    r"|\b(?:iwr|irm)\s+\S+"
    r"|\bpython[0-9.]*\b[^\n;|&]{0,80}\s-c\b[^\n;|&]*\b(?:urllib|http\.client|requests|socket|httpx|aiohttp)\b"
    r"|\bnode\b[^\n;|&]{0,80}\s(?:-e|--eval)\b[^\n;|&]*\b(?:fetch|https?|net|axios|undici)\b"
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

_SHELL_SEGMENT_SPLIT_RE = re.compile(r"(?:\n|;|&&|\|\||\|(?!=)|(?<![<>])&(?![>&]))+")
_GIT_CONFIG_PRE_FLAG_TOKEN = (
    r"(?:(?:-f|--file|--blob)\s+(?:\"[^\"]*\"|'[^']*'|(?!-)[^\s;|&]+)"
    r"|--?[A-Za-z][A-Za-z0-9-]*(?:=(?:\"[^\"]*\"|'[^']*'|[^\s;|&]+))?)"
)
_GIT_CONFIG_READONLY_RE = re.compile(
    r"\bgit\s+config(?:\s+" + _GIT_CONFIG_PRE_FLAG_TOKEN + r"){0,6}"
    r"\s+(?:--get(?:-all|-regexp|-urlmatch)?|--list|-l)\b",
    re.IGNORECASE,
)
_GIT_CONFIG_MUTATION_RE = re.compile(
    r"\bgit\s+config\b[^\n;|&()`]{0,160}(?:"
    r"\s(?:--global|--system|-f|--file)\b"
    r"|\b(?:core\.(?:hooksPath|sshCommand|fsmonitor|editor|pager)"
    r"|alias\.[A-Za-z0-9_-]+"
    r"|credential\.[A-Za-z0-9_.-]+"
    r"|filter\.[A-Za-z0-9_.-]+"
    r"|diff\.external"
    r"|http\.extraHeader)\b"
    r")",
    re.IGNORECASE,
)

_PIPE_FETCH_STAGE_RE = re.compile(r"\b(?:curl|wget|base64)\b", re.IGNORECASE)
_PIPE_EXEC_STAGE_RE = re.compile(
    r"^\s*(?:sudo\s+)?(?:env\s+)?(?:ba|z|da|k|fi)?sh\b|^\s*(?:sudo\s+)?(?:python3?|perl|ruby|node)\b",
    re.IGNORECASE,
)

_SHELL_WRITE_HINT_RE = re.compile(
    r"(?:"
    r">>?"
    r"|\b(?:tee|cp|mv|install|ln|chmod|chown|touch|mkdir|truncate|dd|patch)\b"
    r"|\bsed\b[^\n;|&]{0,80}(?:\s-[A-Za-z]*i|\s--in-place)\b"
    r"|\b(?:perl|ruby)\b[^\n;|&]{0,80}\s-[A-Za-z]*i"
    r"|\bopen\s*\("
    r"|\.(?:write_text|write_bytes)\s*\("
    r"|\b(?:writeFileSync|appendFileSync|copyFileSync)\s*\("
    r"|\bshutil\.(?:copy\w*|move)\s*\("
    r")",
    re.IGNORECASE,
)

_SHELL_HOME_ENV_RE = re.compile(
    r"(?:\$\{(?:HOME|USERPROFILE)\}|\$(?:HOME|USERPROFILE)(?![A-Za-z0-9_])|%USERPROFILE%)",
    re.IGNORECASE,
)
_QUOTED_SPAN_RE = re.compile(r"""["']([^"'\n]{1,512})["']""")
_SHELL_TOKEN_SPLIT_RE = re.compile(r"[\s;|&><=()$`',\"]+")


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

    if slashed.startswith("~"):
        slashed = os.path.expanduser(slashed).replace("\\", "/")

    return slashed


def _lexical_norm_rooted(slashed: str) -> str:
    """Lexically normalize `..` and `.` while preserving UNC, Windows drive, or POSIX root."""
    m_unc = _UNC_ROOT_RE.match(slashed)
    if m_unc:
        unc_root = m_unc.group(1)
        unc_rest = posixpath.normpath(m_unc.group(2) or "/")
        return unc_root if unc_rest == "/" else f"{unc_root}{unc_rest}"

    m_drive = _DRIVE_ROOT_RE.match(slashed)
    if m_drive:
        drive = m_drive.group(1)
        drive_rest = posixpath.normpath(m_drive.group(2) or "/")
        return f"{drive}/" if drive_rest == "/" else f"{drive}{drive_rest}"

    if slashed.startswith("/"):
        collapsed = re.sub(r"^/{2,}", "/", slashed)
        norm = posixpath.normpath(collapsed)
        if norm.startswith("//"):
            norm = "/" + norm.lstrip("/")
        return norm

    return posixpath.normpath(slashed)


def _resolve_with_existing_ancestor(slashed_path: str, root_floor: str) -> str:
    """Resolve symlinks/junctions on deepest existing prefix while preserving lexical `..` fallback."""
    current = (
        re.sub(r"^/{2,}", "/", slashed_path)
        if root_floor == "/"
        else slashed_path
    )
    floor_cmp = root_floor.rstrip("/").casefold()

    for _ in range(4):
        probe = current
        tail_parts: list[str] = []
        while (
            probe
            and probe.rstrip("/").casefold() != floor_cmp
            and not os.path.lexists(probe)
        ):
            parent, leaf = posixpath.split(probe)
            if parent == probe:
                break
            if leaf:
                tail_parts.append(leaf)
            probe = parent

        if probe and os.path.lexists(probe):
            real_base = _normalize_slashed_prefix(os.path.realpath(probe))
        else:
            real_base = probe or root_floor

        combined = (
            posixpath.join(real_base, *reversed(tail_parts)) if tail_parts else real_base
        )
        normalized = _lexical_norm_rooted(combined)
        if normalized == current:
            break
        current = normalized
    return current


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

    if (
        _UNC_ROOT_RE.match(slashed)
        or _DRIVE_ROOT_RE.match(slashed)
        or slashed.startswith("/")
    ):
        result = _lexical_norm_rooted(slashed)
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
        if os.name == "nt":
            result = _resolve_with_existing_ancestor(slashed, m_unc.group(1))
        else:
            result = _lexical_norm_rooted(slashed)
        return result.casefold() if casefold else result

    m_drive = _DRIVE_ROOT_RE.match(slashed)
    if m_drive:
        if os.name == "nt":
            result = _resolve_with_existing_ancestor(slashed, f"{m_drive.group(1)}/")
        else:
            result = _lexical_norm_rooted(slashed)
        return result.casefold() if casefold else result

    if slashed.startswith("/"):
        result = _resolve_with_existing_ancestor(slashed, "/")
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

    srv_underscored = srv.replace("-", "_")
    srv = _SERVER_ALIASES.get(srv_underscored, srv_underscored)
    return srv, tl


def canonicalize_call_head(server: str, tool: str) -> tuple[str, str]:
    """Canonicalize call head to match training format (e.g. `ide.Bash`, `github.create_issue`)."""
    raw_tl = tool.strip()
    norm_srv, norm_tl = _normalize_server_and_tool(server, tool)
    if norm_srv in _GENERIC_HOST_SERVERS:
        canon_srv = "ide"
        if norm_tl in _CANONICAL_IDE_TOOL_CASING:
            canon_tl = _CANONICAL_IDE_TOOL_CASING[norm_tl]
        else:
            canon_tl = raw_tl.split(".", 1)[1] if "." in raw_tl else (raw_tl or norm_tl)
        return canon_srv, canon_tl
    return norm_srv, norm_tl


def _expand_shell_env_vars(text: str) -> str:
    home_posix = Path.home().as_posix()
    return _SHELL_HOME_ENV_RE.sub(lambda _m: home_posix, text)


def expand_self_tamper_variants(text: str) -> list[str]:
    """Return unique candidate views with env expansion, URL decode, quote & backslash stripping."""
    expanded = _expand_shell_env_vars(text)
    unquoted = urllib.parse.unquote(expanded)
    out: list[str] = []
    seen: set[str] = set()
    for base in (text, expanded, unquoted):
        no_quotes = base.replace("'", "").replace('"', "")
        no_backslash = re.sub(r"\\(.)", r"\1", no_quotes)
        for variant in (base, no_quotes, no_backslash):
            if variant and variant not in seen:
                seen.add(variant)
                out.append(variant)
    return out


def has_diodemcp_path_reference(text: str) -> bool:
    """Return True if text contains a `.diodemcp` path component, file, or redirection target."""
    for variant in expand_self_tamper_variants(text):
        if _DIODEMCP_PATH_COMPONENT_RE.search(variant) is not None:
            return True
    return False


_WILDCARD_CHARS_RE = re.compile(r"[*?\[\]]")
_PROTECTED_DIR_MARKERS: frozenset[str] = frozenset({".diodemcp", ".claude", ".cursor"})


def _glob_segment_is_specific(seg: str) -> bool:
    stripped = _WILDCARD_CHARS_RE.sub("", seg)
    alnum_count = sum(1 for ch in stripped if ch.isalnum())
    return alnum_count >= 2


def glob_token_matches_protected(
    raw_token: str,
    protected_targets: Sequence[str],
    workspace_root: Path | None,
) -> bool:
    """Segment-aware glob check avoiding `/` crossing and overly broad `*` false positives."""
    if not any(ch in raw_token for ch in "*?["):
        return False
    norm_glob = _lexical_normalize_target_path(raw_token, workspace_root, casefold=True)
    if not norm_glob:
        return False
    glob_segs = [s for s in norm_glob.split("/") if s]
    if not glob_segs:
        return False

    has_specific_wildcard = any(
        _glob_segment_is_specific(s)
        for s in glob_segs
        if any(ch in s for ch in "*?[")
    )
    has_protected_dir = any(s in _PROTECTED_DIR_MARKERS for s in glob_segs)
    if not has_specific_wildcard and not has_protected_dir:
        return False

    if "**" in glob_segs:
        rx = _compile_single_folded_glob(norm_glob)
        return any(rx.search(target) is not None for target in protected_targets)

    for target in protected_targets:
        t_segs = [s for s in target.split("/") if s]
        if len(glob_segs) != len(t_segs):
            continue
        if all(fnmatch.fnmatchcase(t_s, g_s) for g_s, t_s in zip(glob_segs, t_segs)):
            return True
    return False


def _collect_all_arg_and_schema_keys(
    args: dict[str, Any] | None,
    input_schema: dict[str, Any] | None = None,
) -> set[str]:
    keys: set[str] = set()

    def _visit_args(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and k:
                    keys.add(k.strip().lower())
                _visit_args(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _visit_args(item)

    def _visit_schema(schema: Any) -> None:
        if not isinstance(schema, dict):
            return
        props = schema.get("properties")
        if isinstance(props, dict):
            for k, v in props.items():
                if isinstance(k, str) and k:
                    keys.add(k.strip().lower())
                _visit_schema(v)
        items = schema.get("items")
        if isinstance(items, dict):
            _visit_schema(items)

    if args:
        _visit_args(args)
    if input_schema:
        _visit_schema(input_schema)
    return keys


def classify_tool_metadata(
    tool: str,
    args: dict[str, Any] | None = None,
    *,
    annotations: dict[str, Any] | None = None,
    input_schema: dict[str, Any] | None = None,
) -> CapabilityFlag:
    """Escalate-only heuristic capability classification from tool verbs, arg/schema keys, and MCP annotations."""
    flags = CapabilityFlag.LOCAL_SAFE

    eff_annotations = annotations
    if eff_annotations is None and isinstance(args, dict):
        raw_ann = args.get("_mcp_annotations")
        if isinstance(raw_ann, dict):
            eff_annotations = raw_ann

    if isinstance(eff_annotations, dict):
        if eff_annotations.get("openWorldHint") is True:
            flags |= CapabilityFlag.WRITES_EXTERNAL
        if eff_annotations.get("destructiveHint") is True:
            flags |= CapabilityFlag.MUTATES_SENSITIVE

    split_tool = _CAMEL_BOUNDARY_RE.sub(r"\1 \2", tool or "")
    verb_tokens = [tok for tok in re.split(r"[^a-zA-Z0-9]+", split_tool.lower()) if tok]
    for tok in verb_tokens:
        if tok in _TOOL_VERBS_EXTERNAL:
            flags |= CapabilityFlag.WRITES_EXTERNAL
        if tok in _TOOL_VERBS_MUTATE:
            flags |= CapabilityFlag.MUTATES_SENSITIVE

    arg_keys = _collect_all_arg_and_schema_keys(args, input_schema)
    ext_hits = arg_keys & _KEY_HINTS_EXTERNAL
    if (
        "uri" in ext_hits
        and verb_tokens
        and verb_tokens[0] in {"read", "list", "get", "view", "inspect", "show"}
        and not any(tok in (_TOOL_VERBS_EXTERNAL | _TOOL_VERBS_MUTATE) for tok in verb_tokens)
    ):
        uri_val = str((args or {}).get("uri") or "").strip().lower()
        if not uri_val.startswith(("http://", "https://", "ftp://", "ws://", "wss://")):
            ext_hits = ext_hits - {"uri"}
    if ext_hits:
        flags |= CapabilityFlag.WRITES_EXTERNAL
    if arg_keys & _KEY_HINTS_MUTATE:
        flags |= CapabilityFlag.MUTATES_SENSITIVE

    return flags


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
        if isinstance(val, str):
            stripped = val.strip()
            if stripped and not stripped.lower().startswith(("http://", "https://")):
                found.append(stripped)
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
    expanded = _expand_shell_env_vars(command)
    candidates: list[str] = []
    seen: set[str] = set()

    def _consider(raw_tok: str) -> None:
        tok = raw_tok.strip().strip("\"'")
        if not tok or tok.startswith("-") or "://" in tok:
            return
        if "/" in tok or "\\" in tok or tok.startswith("~") or "." in tok:
            if tok not in seen:
                seen.add(tok)
                candidates.append(tok)

    for m in _QUOTED_SPAN_RE.finditer(expanded):
        _consider(m.group(1))
    for raw_tok in _SHELL_TOKEN_SPLIT_RE.split(expanded):
        _consider(raw_tok)
    return candidates


def _strip_sql_comments(sql: str) -> str:
    without_block = _SQL_BLOCK_COMMENT_RE.sub(" ", sql)
    unwrapped_exec = _MYSQL_EXEC_COMMENT_OPEN_RE.sub(" ", without_block)
    unwrapped_exec = _MYSQL_EXEC_COMMENT_CLOSE_RE.sub(" ", unwrapped_exec)
    return _SQL_LINE_COMMENT_RE.sub(" ", unwrapped_exec)


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


def _validate_string_list(section: dict[str, Any], key: str, section_name: str) -> list[str]:
    if key not in section:
        return []
    val = section[key]
    if not isinstance(val, list) or not all(isinstance(item, str) for item in val):
        raise ValueError(
            f"[{section_name}].{key} must be a list of strings, got {type(val).__name__}"
        )
    return val


_NON_SECRET_ENV_TEMPLATE_BASENAMES: frozenset[str] = frozenset(
    {
        ".env.example",
        ".env.sample",
        ".env.template",
        ".env.dist",
        ".env.schema",
    }
)


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

    def extend_from_mapping(
        self,
        section: Any,
        *,
        section_name: str = "config",
    ) -> None:
        if not isinstance(section, dict):
            raise ValueError(
                f"[{section_name}] must be a TOML table/dict, got {type(section).__name__}"
            )
        unknown_keys = set(section.keys()) - _KNOWN_DOMAIN_LIST_KEYS
        if unknown_keys:
            sorted_unknown = ", ".join(sorted(str(k) for k in unknown_keys))
            raise ValueError(f"Unknown keys in [{section_name}]: {sorted_unknown}")

        for pat in _validate_string_list(section, "paths", section_name):
            if pat.strip():
                self.path_patterns.extend(_compile_path_globs(pat))
        for pat in _validate_string_list(section, "high_secret_paths", section_name):
            if pat.strip():
                self.high_secret_path_patterns.extend(_compile_path_globs(pat))
        for srv in _validate_string_list(section, "servers", section_name):
            if srv.strip():
                norm_srv, _ = _normalize_server_and_tool(srv, "")
                self.servers.add(norm_srv)
        for srv in _validate_string_list(section, "high_secret_servers", section_name):
            if srv.strip():
                norm_srv, _ = _normalize_server_and_tool(srv, "")
                self.high_secret_servers.add(norm_srv)
        for tl in _validate_string_list(section, "tools", section_name):
            if tl.strip():
                if "." in tl:
                    norm_srv, norm_tl = _normalize_server_and_tool("", tl)
                    self.qualified_tools.add(f"{norm_srv}.{norm_tl}")
                else:
                    self.standalone_tools.add(tl.strip().lower())
        for tl in _validate_string_list(section, "high_secret_tools", section_name):
            if tl.strip():
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
        base = posixpath.basename(folded_canon_path.rstrip("/"))
        if base in _NON_SECRET_ENV_TEMPLATE_BASENAMES:
            return False
        return any(
            rx.search(folded_canon_path) is not None for rx in self.high_secret_path_patterns
        )


def _is_editor_view_mode(norm_tool: str, args: dict[str, Any] | None) -> bool:
    if norm_tool not in _EDITOR_MULTIPLEX_TOOLS or not isinstance(args, dict):
        return False
    for key, val in args.items():
        if key.lower() == "command" and isinstance(val, str):
            return val.strip().lower() == "view"
    return False


_DEFAULT_PROBE_TARGET_PATTERNS: tuple[str, ...] = (
    "~/.diodemcp/incidents.json",
    "~/.diodemcp/config.toml",
    "~/.diodemcp/active_bundle.json",
    "~/.diodemcp/audit.jsonl",
    "~/.diodemcp/admin.secret",
    "~/.diodemcp/engine.sock",
    "~/.diodemcp/engine.addr",
    "~/.diodemcp/daemon.lock",
    "~/.claude/settings.json",
    "~/.claude.json",
    "~/.cursor/mcp.json",
    "~/.mcp.json",
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
        user_sensitive_mutations: _DomainMatcher,
        local_safe: _DomainMatcher | None = None,
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
        self._user_sensitive_mutations = user_sensitive_mutations
        self._local_safe = local_safe if local_safe is not None else _DomainMatcher()
        self._protected_probe_targets: tuple[str, ...] = tuple(
            self._canonical_folded_paths(list(_DEFAULT_PROBE_TARGET_PATTERNS))
        )

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
        user_sensitive_mutations = _DomainMatcher()

        self_protection.extend_from_mapping(
            default_data.get("self_protection", {}), section_name="self_protection"
        )
        private_sources.extend_from_mapping(
            default_data.get("private_sources", {}), section_name="private_sources"
        )
        untrusted_sources.extend_from_mapping(
            default_data.get("untrusted_sources", {}), section_name="untrusted_sources"
        )
        mixed_trust_sources.extend_from_mapping(
            default_data.get("mixed_trust_sources", {}), section_name="mixed_trust_sources"
        )
        dual_direction.extend_from_mapping(
            default_data.get("dual_direction", {}), section_name="dual_direction"
        )
        external_sinks.extend_from_mapping(
            default_data.get("external_sinks", {}), section_name="external_sinks"
        )
        sensitive_mutations.extend_from_mapping(
            default_data.get("sensitive_mutations", {}), section_name="sensitive_mutations"
        )

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
        local_safe = _DomainMatcher()
        local_safe.extend_from_mapping(
            default_data.get("local_safe", {}), section_name="local_safe"
        )

        resolved_config = (
            config_path
            if config_path is not None
            else (DEFAULT_STATE_DIR / "config.toml")
        )
        if resolved_config.is_file():
            user_data = tomllib.loads(resolved_config.read_text(encoding="utf-8"))
            for key in user_data:
                if key in _USER_CONFIG_DOMAIN_SECTIONS and not isinstance(
                    user_data[key], dict
                ):
                    raise ValueError(
                        f"[{key}] in {resolved_config} must be a TOML table/dict"
                    )
            if "private_sources" in user_data:
                private_sources.extend_from_mapping(
                    user_data["private_sources"], section_name="private_sources"
                )
                user_private_sources.extend_from_mapping(
                    user_data["private_sources"], section_name="private_sources"
                )
            if "untrusted_sources" in user_data:
                untrusted_sources.extend_from_mapping(
                    user_data["untrusted_sources"], section_name="untrusted_sources"
                )
            if "external_sinks" in user_data:
                external_sinks.extend_from_mapping(
                    user_data["external_sinks"], section_name="external_sinks"
                )
            if "sensitive_mutations" in user_data:
                sensitive_mutations.extend_from_mapping(
                    user_data["sensitive_mutations"], section_name="sensitive_mutations"
                )
                user_sensitive_mutations.extend_from_mapping(
                    user_data["sensitive_mutations"], section_name="sensitive_mutations"
                )
            if "mixed_trust_sources" in user_data:
                mixed_trust_sources.extend_from_mapping(
                    user_data["mixed_trust_sources"], section_name="mixed_trust_sources"
                )
            if "dual_direction" in user_data:
                dual_direction.extend_from_mapping(
                    user_data["dual_direction"], section_name="dual_direction"
                )

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
            user_sensitive_mutations=user_sensitive_mutations,
            local_safe=local_safe,
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

    def _is_file_read_call(
        self,
        norm_server: str,
        norm_tool: str,
        args: dict[str, Any] | None = None,
    ) -> bool:
        if _is_editor_view_mode(norm_tool, args):
            return True
        if norm_tool in _LOCAL_FILE_WRITE_TOOLS:
            return False
        if norm_tool in self._private_sources.standalone_tools:
            return True
        return norm_server in {"filesystem", "files"} and norm_tool not in _LOCAL_FILE_WRITE_TOOLS

    def _is_file_write_call(
        self,
        norm_server: str,
        norm_tool: str,
        args: dict[str, Any] | None = None,
    ) -> bool:
        if _is_editor_view_mode(norm_tool, args):
            return False
        return norm_tool in _LOCAL_FILE_WRITE_TOOLS

    def _is_sql_db_call(
        self,
        norm_server: str,
        norm_tool: str,
        args: dict[str, Any] | None = None,
    ) -> bool:
        if norm_server in self._db_servers:
            return True
        if norm_tool in (self._db_read_tools | self._db_mutating_tools) and isinstance(
            args, dict
        ):
            return bool(_extract_keyed_strings(args, _SQL_ARG_KEYS))
        return False

    def _is_explicitly_registered_tool(
        self,
        norm_server: str,
        norm_tool: str,
        args: dict[str, Any] | None = None,
    ) -> bool:
        return (
            self._local_safe.matches_server_or_tool(norm_server, norm_tool)
            or self._is_shell_call(norm_server, norm_tool)
            or self._is_file_read_call(norm_server, norm_tool, args)
            or self._is_file_write_call(norm_server, norm_tool, args)
            or self._is_sql_db_call(norm_server, norm_tool, args)
            or self._external_sinks.matches_server_or_tool(norm_server, norm_tool)
            or self._sensitive_mutations.matches_server_or_tool(norm_server, norm_tool)
            or self._dual_direction.matches_server_or_tool(norm_server, norm_tool)
            or self._untrusted_sources.matches_server_or_tool(norm_server, norm_tool)
            or self._mixed_trust_sources.matches_server_or_tool(norm_server, norm_tool)
            or f"{norm_server}.{norm_tool}" in self._private_sources.qualified_tools
            or norm_tool in self._private_sources.standalone_tools
        )

    def is_known_safe_tool(
        self,
        server: str,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        annotations: dict[str, Any] | None = None,
        input_schema: dict[str, Any] | None = None,
    ) -> bool:
        """Return True ONLY for explicitly registered local/safe tools whose request is `LOCAL_SAFE`."""
        norm_server, norm_tool = _normalize_server_and_tool(server, tool)
        req_flags = self.classify_request(
            server,
            tool,
            args or {},
            annotations=annotations,
            input_schema=input_schema,
        )
        if req_flags != CapabilityFlag.LOCAL_SAFE:
            return False
        if self._local_safe.matches_server_or_tool(norm_server, norm_tool):
            return True
        if self._is_file_read_call(norm_server, norm_tool, args):
            return True
        if self._is_file_write_call(norm_server, norm_tool, args):
            return True
        if self._is_sql_db_call(norm_server, norm_tool, args) and (
            norm_tool in self._db_read_tools or norm_tool in self._db_mutating_tools
        ):
            return True
        return False

    def is_unconditional_critical_self_tamper(
        self,
        server: str,
        tool: str,
        args: dict[str, Any],
    ) -> bool:
        """Return True when a call reads, mutates, or kills DiodeMCP or IDE hook/MCP configs (`C4`)."""
        norm_server, norm_tool = _normalize_server_and_tool(server, tool)
        raw_paths = _extract_path_values(args)
        for raw_p in raw_paths:
            if has_diodemcp_path_reference(raw_p):
                return True
            if glob_token_matches_protected(
                raw_p, self._protected_probe_targets, self.workspace_root
            ):
                return True
        for folded in self._canonical_folded_paths(raw_paths):
            if self._self_protection.matches_path(folded):
                return True

        all_strings: list[str] = []
        _collect_strings_for_self_tamper(
            args,
            all_strings,
            skip_file_body_keys=self._is_file_write_call(norm_server, norm_tool, args),
        )
        for text in all_strings:
            for candidate in expand_self_tamper_variants(text):
                if _SELF_TAMPER_CLI_RE.search(candidate) is not None:
                    return True
                for line in candidate.splitlines():
                    if _KILL_VERB_RE.search(line) and _DIODEMCP_WORD_RE.search(line):
                        return True
                if _SELF_TAMPER_INLINE_RE.search(candidate) is not None:
                    return True
                if _DIODEMCP_PATH_COMPONENT_RE.search(candidate) is not None:
                    return True
                for raw_tok in _extract_shell_path_candidates(candidate):
                    if glob_token_matches_protected(
                        raw_tok, self._protected_probe_targets, self.workspace_root
                    ):
                        return True
                for split_tok in _SHELL_TOKEN_SPLIT_RE.split(candidate):
                    if split_tok and glob_token_matches_protected(
                        split_tok, self._protected_probe_targets, self.workspace_root
                    ):
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
        *,
        annotations: dict[str, Any] | None = None,
        input_schema: dict[str, Any] | None = None,
    ) -> CapabilityFlag:
        """Classify pre-execution sink and sensitive-mutation capabilities for a proposed call."""
        norm_server, norm_tool = _normalize_server_and_tool(server, tool)
        flags = CapabilityFlag.LOCAL_SAFE

        if self.is_unconditional_critical_self_tamper(server, tool, args):
            flags |= CapabilityFlag.MUTATES_SENSITIVE | CapabilityFlag.HIGH_SECRET

        folded_paths = self._canonical_folded_paths(_extract_path_values(args))

        is_mcp_protocol_read = norm_tool in {
            "read_resource",
            "list_resources",
            "list_prompts",
            "get_prompt",
        } and not any(
            str(v).strip().lower().startswith(("http://", "https://", "ftp://", "ws://", "wss://"))
            for v in (args.get("uri"), args.get("url"))
            if v is not None
        )

        # Dual-direction web/search/browser tools send URL/query externally on request.
        if not is_mcp_protocol_read and self._dual_direction.matches_server_or_tool(
            norm_server, norm_tool
        ):
            flags |= CapabilityFlag.WRITES_EXTERNAL

        # T2 / T7: Explicit external sinks (zero exemption for git remote origin).
        if not is_mcp_protocol_read and self._external_sinks.matches_server_or_tool(
            norm_server, norm_tool
        ):
            flags |= CapabilityFlag.WRITES_EXTERNAL

        # T7 / T6: Explicit sensitive mutation servers/tools.
        if self._sensitive_mutations.matches_server_or_tool(norm_server, norm_tool):
            flags |= CapabilityFlag.MUTATES_SENSITIVE

        # T5: Database mutating SQL queries vs read-only SELECT queries.
        if self._is_sql_db_call(norm_server, norm_tool, args):
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
                for seg in _SHELL_SEGMENT_SPLIT_RE.split(cmd):
                    for m_git in _GIT_CONFIG_MUTATION_RE.finditer(seg):
                        if _GIT_CONFIG_READONLY_RE.match(seg, m_git.start()) is None:
                            flags |= CapabilityFlag.MUTATES_SENSITIVE
                            break
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
        is_file_read = self._is_file_read_call(norm_server, norm_tool, args)
        if self._is_file_write_call(norm_server, norm_tool, args) or (
            norm_server in {"filesystem", "files"} and not is_file_read
        ):
            for fp in folded_paths:
                if self._sensitive_mutations.matches_path(fp):
                    flags |= CapabilityFlag.MUTATES_SENSITIVE
                if self._external_sinks.matches_path(fp):
                    flags |= CapabilityFlag.WRITES_EXTERNAL
        else:
            # User-configured external sink / sensitive mutation paths apply to custom non-read tools.
            for fp in folded_paths:
                if self._external_sinks.matches_path(fp):
                    flags |= CapabilityFlag.WRITES_EXTERNAL
                if not is_file_read and self._user_sensitive_mutations.matches_path(fp):
                    flags |= CapabilityFlag.MUTATES_SENSITIVE

        # Escalate-only dynamic metadata classification:
        # - MCP annotations (`openWorldHint`, `destructiveHint`) always escalate.
        # - Unknown/unlisted tools also escalate from verb/arg-key/schema heuristics.
        if not self._is_explicitly_registered_tool(norm_server, norm_tool, args):
            flags |= classify_tool_metadata(
                norm_tool,
                args,
                annotations=annotations,
                input_schema=input_schema,
            )
        else:
            flags |= classify_tool_metadata(
                "",
                None,
                annotations=annotations if annotations is not None else (
                    args.get("_mcp_annotations") if isinstance(args, dict) else None
                ),
            )

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

        # T3: Database SELECT / RETURNING / read queries ingest private data on response.
        if self._is_sql_db_call(norm_server, norm_tool, args):
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
        if self._is_file_read_call(norm_server, norm_tool, args):
            flags |= CapabilityFlag.READS_PRIVATE
            for fp in folded_paths:
                if self._private_sources.matches_high_secret_path(fp):
                    flags |= CapabilityFlag.READS_PRIVATE | CapabilityFlag.HIGH_SECRET
                if self._untrusted_sources.matches_path(fp):
                    flags |= CapabilityFlag.READS_UNTRUSTED
        elif not self._is_file_write_call(norm_server, norm_tool, args):
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
                if self._mixed_trust_sources.matches_path(fp):
                    flags |= CapabilityFlag.READS_PRIVATE | CapabilityFlag.READS_UNTRUSTED

        # T1: Dynamic output_text secret header / prefix inspection upgrades any tool response.
        if output_text and _OUTPUT_SECRET_RE.search(output_text) is not None:
            flags |= CapabilityFlag.READS_PRIVATE | CapabilityFlag.HIGH_SECRET

        # Unknown / unlisted MCP server/tool responses default to READS_UNTRUSTED
        # while explicitly safe/known local-write/sink/db/sensitive tools stay clean.
        if flags == CapabilityFlag.LOCAL_SAFE:
            is_known_clean = (
                self._local_safe.matches_server_or_tool(norm_server, norm_tool)
                or self._is_file_write_call(norm_server, norm_tool, args)
                or is_outbound_sink_tool
                or (norm_server in self._db_servers)
                or self._sensitive_mutations.matches_server_or_tool(norm_server, norm_tool)
                or self._is_explicitly_registered_tool(norm_server, norm_tool, args)
            )
            if not is_known_clean:
                flags |= CapabilityFlag.READS_UNTRUSTED

        return flags
