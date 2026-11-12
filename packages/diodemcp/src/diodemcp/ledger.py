"""Workspace-scoped persistent session taint ledger, SimHash dedup, and split-exfil accumulator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import time
from typing import Any
import urllib.parse

from diodemcp.constants import MAX_LEDGER_SPAN_BYTES
from diodemcp.packer import flatten_call_arguments
from diodemcp.predictor import (
    _KNOWN_SECRET_RE,
    _KV_ASSIGN_RE,
    _SENSITIVE_KEY_WORDS,
    _URI_PASSWORD_RE,
    _detect_code_or_data_copy,
    _expand_arg_views_detailed,
    _extract_secret_candidates,
    _secret_matches_views,
)
from diodemcp.registry import canonicalize_target_path
from diodemcp.types import CandidateCall, CapabilityFlag, TaintSpan

MAX_SPAN_BYTES: int = MAX_LEDGER_SPAN_BYTES
MAX_OUTBOUND_CALLS_PER_DEST: int = 16
MAX_OUTBOUND_BYTES_PER_DEST: int = 8192

_SIMHASH_WORD_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_LONG_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{23,}")
_PRINTABLE_RUN_RE = re.compile(r"[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]{8,}")
_SHELL_TOKEN_SPLIT_RE = re.compile(r"""[\s;|&><()"'`=]+""")
_EMBEDDED_URL_OR_QUERY_RE = re.compile(r"""(?:https?://[^\s"'<>]+|\?[^\s"'<>]+)""", re.IGNORECASE)
_WIN_DRIVE_ROOT_FOLD_RE = re.compile(r"^[a-z]:$")
_UNC_SHARE_ROOT_FOLD_RE = re.compile(r"^//[^/]+(?:/[^/]+)?$")
_REPO_WIDE_PUBLISH_CMD_RE = re.compile(
    r"(?:"
    r"\bgit\s+(?:push|send-email|bundle)\b"
    r"|\b(?:npm|pnpm|yarn|poetry|uv|cargo)\s+publish\b"
    r"|\btwine\s+upload\b"
    r"|\bgh\s+(?:release|pr)\s+create\b"
    r")",
    re.IGNORECASE,
)

_BOILERPLATE_LINES: frozenset[str] = frozenset(
    {
        "from __future__ import annotations",
        "#!/usr/bin/env python3",
        "#!/usr/bin/env python",
        "#!/usr/bin/env bash",
        "#!/usr/bin/env node",
        'if __name__ == "__main__":',
        "if __name__ == '__main__':",
    }
)

_PATH_HINT_KEYS: frozenset[str] = frozenset(
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
    }
)


def compute_workspace_key(
    workspace_root: Path | str | None = None,
    *,
    uid: int | str | None = None,
) -> str:
    """Compute canonical 24-hex workspace key `sha256(f'{uid}:{realpath(workspace_root)}')[:24]`."""
    raw_root = os.fspath(workspace_root) if workspace_root is not None else os.getcwd()
    real_root = os.path.realpath(raw_root)
    if uid is not None:
        resolved_uid = str(uid)
    elif hasattr(os, "getuid"):
        resolved_uid = str(os.getuid())
    else:
        resolved_uid = os.environ.get("USERNAME") or os.environ.get("USER") or "0"
    payload = f"{resolved_uid}:{real_root}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def compute_simhash64(text: str) -> int:
    """Compute a deterministic process-independent 64-bit SimHash over word & 4-gram shingles."""
    if not text:
        return 0
    normalized = " ".join(text[:MAX_SPAN_BYTES].lower().split())
    if not normalized:
        return 0

    features: list[str] = [m.group(0) for m in _SIMHASH_WORD_RE.finditer(normalized)]
    if len(normalized) >= 4:
        step = 2 if len(normalized) <= 4096 else 4
        for idx in range(0, len(normalized) - 3, step):
            features.append(f"4g:{normalized[idx : idx + 4]}")

    if not features:
        features = [normalized]

    weights = [0] * 64
    for feat in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        hv = int.from_bytes(digest, "big", signed=False)
        for bit in range(64):
            if (hv >> bit) & 1:
                weights[bit] += 1
            else:
                weights[bit] -= 1

    fingerprint = 0
    for bit, score in enumerate(weights):
        if score > 0:
            fingerprint |= 1 << bit
    return fingerprint


def _hamming_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _utf8_truncate_head(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _utf8_truncate_tail(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    tail = encoded[-max_bytes:]
    # Advance past any leading UTF-8 continuation bytes (0b10xxxxxx)
    idx = 0
    while idx < len(tail) and (tail[idx] & 0xC0) == 0x80:
        idx += 1
    return tail[idx:].decode("utf-8", errors="ignore")


def _extract_printable_secret_lines(text: str) -> list[str]:
    preserved: list[str] = []
    seen: set[str] = set()

    def _keep(snippet: str) -> None:
        clean = snippet.strip()
        if clean and clean not in seen:
            seen.add(clean)
            preserved.append(clean)

    for run_match in _PRINTABLE_RUN_RE.finditer(text):
        run = run_match.group(0)
        if "PRIVATE KEY-----" in run:
            _keep(run[:4096])
            continue
        for m in _KNOWN_SECRET_RE.finditer(run):
            _keep(m.group(0))
        for m in _URI_PASSWORD_RE.finditer(run):
            _keep(run[:512])
        for m in _KV_ASSIGN_RE.finditer(run):
            key_l = m.group(1).lower()
            if any(w in key_l for w in _SENSITIVE_KEY_WORDS):
                _keep(m.group(0))
    return preserved


def _sanitize_and_cap_span_text(text: str, max_bytes: int = MAX_SPAN_BYTES) -> str:
    """Replace binary/null-byte blobs with safe SHA256 summary + preserved secrets, capped at 64 KiB."""
    raw_bytes = text.encode("utf-8", errors="replace")
    null_count = text.count("\x00")
    ctrl_count = sum(1 for ch in text if ord(ch) < 32 and ch not in "\t\n\r")

    if null_count > 0 or (len(text) > 0 and (ctrl_count / len(text)) > 0.10):
        digest = hashlib.sha256(raw_bytes).hexdigest()
        header = f"[binary_blob bytes={len(raw_bytes)} sha256={digest}]"
        preserved_secrets = _extract_printable_secret_lines(text)
        if preserved_secrets:
            joined_secrets = "\n".join(preserved_secrets)
            candidate = f"{header}\n{joined_secrets}"
        elif len(text) <= 4096 and (null_count / max(len(text), 1)) < 0.05:
            printable = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+", " ", text).strip()
            candidate = f"{header}\n{printable}" if printable else header
        else:
            candidate = header
        return _utf8_truncate_head(candidate, max_bytes)

    if len(raw_bytes) <= max_bytes:
        return text

    # Retain 48 KiB head + truncation marker (+ middle secret lines if any) + 16 KiB tail (up to max_bytes)
    marker = "\n...[TRUNCATED]...\n"
    marker_bytes = len(marker.encode("utf-8"))
    if max_bytes <= marker_bytes + 64:
        return _utf8_truncate_head(text, max_bytes)

    default_head = max(32, (max_bytes * 3) // 4)
    default_tail = max(32, max_bytes - default_head - marker_bytes)

    head_slice = raw_bytes[:default_head].decode("utf-8", errors="ignore")
    tail_slice = raw_bytes[-default_tail:].decode("utf-8", errors="ignore")
    mid_slice = raw_bytes[default_head : len(raw_bytes) - default_tail].decode(
        "utf-8", errors="ignore"
    )
    mid_secrets = _extract_printable_secret_lines(mid_slice) if mid_slice else []

    mid_block = ""
    if mid_secrets:
        candidate_mid = "\n" + "\n".join(mid_secrets) + "\n"
        max_mid = max_bytes // 4
        if len(candidate_mid.encode("utf-8")) > max_mid:
            candidate_mid = _utf8_truncate_head(candidate_mid, max_mid)
        mid_block = candidate_mid

    mid_bytes = len(mid_block.encode("utf-8"))
    head_budget = max(32, default_head - mid_bytes)
    head_part = _utf8_truncate_head(head_slice, head_budget)
    used_bytes = len(head_part.encode("utf-8")) + marker_bytes + mid_bytes
    tail_budget = max(0, max_bytes - used_bytes)
    tail_part = _utf8_truncate_tail(tail_slice, tail_budget) if tail_budget > 0 else ""
    return f"{head_part}{marker}{mid_block}{tail_part}"


def _atomic_write_json_0600(target_path: Path, payload: dict[str, Any]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tmp_name = f".{target_path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    tmp_path = target_path.parent / tmp_name
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, serialized)
        os.fsync(fd)
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
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target_path)
        os.chmod(target_path, 0o600)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _reassemble_outbound_window(history: list[dict[str, str]], max_bytes: int = MAX_OUTBOUND_BYTES_PER_DEST) -> str:
    """Reconstruct chunked/query-param/path/arg outbound streams from up to 16 calls (`C7`)."""
    if not history:
        return ""

    param_streams: dict[str, list[str]] = {}
    all_param_values: list[str] = []
    path_tail_stream: list[str] = []
    key_streams: dict[str, list[str]] = {}
    raw_lines: list[str] = []

    for call_flat in history:
        for arg_key, val in call_flat.items():
            if not val:
                continue
            key_streams.setdefault(arg_key, []).append(val)
            raw_lines.append(f"{arg_key}={val}")

            # Extract URL query parameters & path segments (including URLs embedded in shell commands)
            url_candidates: list[str] = []
            if val.startswith(("http://", "https://", "?")) and not any(
                ch in val for ch in ('"', "'", " ")
            ):
                url_candidates.append(val)
            else:
                url_candidates.extend(
                    m.group(0).rstrip(".,;:)]}>'\"")
                    for m in _EMBEDDED_URL_OR_QUERY_RE.finditer(val)
                )

            for url_cand in url_candidates:
                try:
                    parsed = urllib.parse.urlsplit(url_cand)
                except ValueError:
                    continue
                if parsed.query:
                    for qk, qv in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
                        param_streams.setdefault(f"{arg_key}?{qk}", []).append(qv)
                        all_param_values.append(qv)
                if parsed.path and parsed.path != "/":
                    tail_seg = parsed.path.rstrip("/").rsplit("/", 1)[-1]
                    if tail_seg:
                        path_tail_stream.append(urllib.parse.unquote(tail_seg))

    sections: list[str] = []
    for pk, chunks in param_streams.items():
        if len(chunks) >= 2:
            sections.append(f"[param:{pk}]={''.join(chunks)}")
    if len(all_param_values) >= 2:
        sections.append(f"[all_params]={''.join(all_param_values)}")
    if len(path_tail_stream) >= 2:
        sections.append(f"[path_tail]={''.join(path_tail_stream)}")
    for ak, vals in key_streams.items():
        if len(vals) >= 2:
            sections.append(f"[concat:{ak}]={''.join(vals)}")
    sections.extend(raw_lines)

    # Prioritize reassembled streams at the end so tail-capping preserves latest chunks & streams
    raw_block = "\n".join(raw_lines)
    stream_block = "\n".join(sections[: len(sections) - len(raw_lines)])
    combined = f"{raw_block}\n{stream_block}".strip() if stream_block else raw_block
    return _utf8_truncate_tail(combined, max_bytes)


@dataclass(frozen=True)
class LedgerSnapshot:
    """Immutable point-in-time snapshot of a workspace `SessionLedger`."""

    intent: str | None = None
    private_spans: tuple[TaintSpan, ...] = ()
    untrusted_spans: tuple[TaintSpan, ...] = ()
    tainted_paths: frozenset[str] = field(default_factory=frozenset)
    outbound_accumulator: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "private_spans",
            tuple(s.model_copy(deep=True) for s in self.private_spans),
        )
        object.__setattr__(
            self,
            "untrusted_spans",
            tuple(s.model_copy(deep=True) for s in self.untrusted_spans),
        )
        object.__setattr__(self, "tainted_paths", frozenset(self.tainted_paths))
        object.__setattr__(self, "outbound_accumulator", dict(self.outbound_accumulator))


_SHELL_REDIRECT_RE = re.compile(
    r"""(?:^|[^0-9<>])>>?\s*("[^"\n]+"|'[^'\n]+'|[^\s;|&><]+)"""
)
_SHELL_TEE_RE = re.compile(
    r"""\btee\b(?:\s+-[A-Za-z]+)*\s+("[^"\n]+"|'[^'\n]+'|[^\s;|&><]+)"""
)
_SHELL_CHAIN_SPLIT_RE = re.compile(r"(?:\n|;|&&|\|\|)+")
_COPY_CMD_NAMES: frozenset[str] = frozenset({"cp", "mv", "install", "ln"})


def _extract_shell_copy_redirect_ops(
    command: str,
) -> list[tuple[list[str], str, bool]]:
    """Return `(src_tokens, dst_token, has_inline_write_content)` for shell copy/redirect ops."""
    ops: list[tuple[list[str], str, bool]] = []
    for seg in _SHELL_CHAIN_SPLIT_RE.split(command):
        seg_s = seg.strip()
        if not seg_s:
            continue

        redir_dsts: list[str] = []
        for rx in (_SHELL_REDIRECT_RE, _SHELL_TEE_RE):
            for m in rx.finditer(seg_s):
                tok = m.group(1).strip().strip("\"'")
                if tok and not tok.startswith("/dev/null") and not tok.startswith("&"):
                    redir_dsts.append(tok)
        if redir_dsts:
            try:
                toks = shlex.split(seg_s)
            except ValueError:
                toks = seg_s.split()
            src_toks: list[str] = []
            for t in toks:
                if t in redir_dsts or t.startswith((">", "|", "-")) or "://" in t:
                    continue
                clean_t = t.strip().strip("\"'")
                if clean_t:
                    src_toks.append(clean_t)
            for dst in redir_dsts:
                ops.append((src_toks, dst, True))

        for pipe_part in seg_s.split("|"):
            part_s = pipe_part.strip()
            if not part_s:
                continue
            try:
                toks = shlex.split(part_s)
            except ValueError:
                toks = part_s.split()
            idx = 0
            while idx < len(toks) and "=" in toks[idx] and not toks[idx].startswith(("-", "/")):
                idx += 1
            if idx >= len(toks):
                continue
            cmd_name = os.path.basename(toks[idx]).lower()
            if cmd_name not in _COPY_CMD_NAMES:
                continue
            pos_args: list[str] = []
            target_dir: str | None = None
            j = idx + 1
            while j < len(toks):
                t = toks[j]
                if t in {">", ">>", "<"}:
                    j += 2
                    continue
                if t in {"-t", "--target-directory"} and j + 1 < len(toks):
                    target_dir = toks[j + 1]
                    j += 2
                    continue
                if t.startswith("--target-directory="):
                    target_dir = t.split("=", 1)[1]
                    j += 1
                    continue
                if t.startswith("-"):
                    j += 1
                    continue
                pos_args.append(t)
                j += 1
            if target_dir and pos_args:
                ops.append((pos_args, target_dir, False))
            elif len(pos_args) >= 2:
                ops.append((pos_args[:-1], pos_args[-1], False))
    return ops


def _extract_recordable_read_path(
    source_label: str,
    target_path: str | None,
    workspace_root: Path | None,
) -> str | None:
    cand: str | None = None
    if target_path and target_path.strip():
        cand = target_path.strip()
    elif source_label and source_label.strip():
        sl = source_label.strip()
        if (
            sl in {"unknown", "corrupt_ledger_fail_closed", "evicted_secrets_summary"}
            or sl.startswith("staged_write:")
            or "://" in sl
        ):
            return None
        if ":" in sl and not (len(sl) >= 3 and sl[1] == ":" and sl[2] in ("/", "\\")):
            return None
        cand = sl
    if not cand or cand.startswith(("http://", "https://")):
        return None
    canon = canonicalize_target_path(cand, workspace_root)
    if not canon:
        return None
    if workspace_root is not None:
        ws_canon = canonicalize_target_path(os.fspath(workspace_root), None)
        if canon.casefold().rstrip("/") == ws_canon.casefold().rstrip("/"):
            return None
    if os.path.isdir(canon):
        return None
    return canon


class SessionLedger:
    """Workspace-keyed persistent taint ledger with trust-scoped dedup and `HIGH_SECRET` pinning."""

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        *,
        persist_path: Path | str | None = None,
        max_private_spans: int = 128,
        max_untrusted_spans: int = 128,
        max_span_bytes: int = MAX_SPAN_BYTES,
        uid: int | str | None = None,
    ) -> None:
        if max_private_spans < 1 or max_untrusted_spans < 1:
            raise ValueError("max_private_spans and max_untrusted_spans must be >= 1")
        self._workspace_root = (
            Path(os.path.realpath(os.fspath(workspace_root)))
            if workspace_root is not None
            else None
        )
        self.workspace_key: str = compute_workspace_key(workspace_root, uid=uid)
        self._persist_path: Path | None = Path(persist_path) if persist_path is not None else None
        self._max_private_spans = max_private_spans
        self._max_untrusted_spans = max_untrusted_spans
        self._max_span_bytes = max_span_bytes

        self._intent: str | None = None
        self._intent_history: list[str] = []
        self._private_spans: list[TaintSpan] = []
        self._untrusted_spans: list[TaintSpan] = []
        self._tainted_paths: dict[str, set[str]] = {}
        self._private_read_paths: dict[str, set[str]] = {}
        self._untrusted_tainted_paths: set[str] = set()
        self._outbound_history: dict[str, list[dict[str, str]]] = {}
        self._span_sha256: dict[str, str] = {}
        self._evicted_secret_candidates: list[str] = []
        self._span_counter: int = 0

        if self._persist_path is not None and self._persist_path.exists():
            self._rehydrate_from_disk(self._persist_path)

    @property
    def persist_path(self) -> Path | None:
        return self._persist_path

    def clear(self) -> None:
        """Reset all session taint state (explicit new session / `/clear`) and persist."""
        self._intent = None
        self._intent_history.clear()
        self._private_spans.clear()
        self._untrusted_spans.clear()
        self._tainted_paths.clear()
        self._private_read_paths.clear()
        self._untrusted_tainted_paths.clear()
        self._outbound_history.clear()
        self._span_sha256.clear()
        self._evicted_secret_candidates.clear()
        self._persist_if_configured()

    def _next_span_id(self) -> str:
        self._span_counter += 1
        return f"span_{self.workspace_key[:8]}_{self._span_counter:06d}_{secrets.token_hex(3)}"

    def _rehydrate_from_disk(self, path: Path) -> None:
        try:
            raw_bytes = path.read_bytes()
            if not raw_bytes:
                raise ValueError("Empty ledger file")
            data = json.loads(raw_bytes.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Ledger JSON root is not an object")

            raw_intent = data.get("intent")
            if raw_intent is not None and not isinstance(raw_intent, str):
                raise ValueError("Ledger 'intent' field must be a string or null")

            raw_intent_hist = data.get("intent_history")
            if raw_intent_hist is not None and (
                not isinstance(raw_intent_hist, list)
                or not all(isinstance(x, str) for x in raw_intent_hist)
            ):
                raise ValueError("Ledger 'intent_history' must be a list of strings")

            raw_priv = data.get("private_spans", [])
            raw_untrust = data.get("untrusted_spans", [])
            raw_tp = data.get("tainted_paths", {})
            raw_prp = data.get("private_read_paths", {})
            raw_utp = data.get("untrusted_tainted_paths", [])
            raw_evicted = data.get("evicted_secret_candidates", [])
            raw_out = data.get("outbound_history", {})

            if not isinstance(raw_priv, list) or not isinstance(raw_untrust, list):
                raise ValueError("Ledger span collections must be lists")
            if (
                not isinstance(raw_tp, dict)
                or not isinstance(raw_prp, dict)
                or not isinstance(raw_out, dict)
            ):
                raise ValueError("Ledger path/outbound maps must be objects")
            if not isinstance(raw_utp, list) or not all(isinstance(x, str) for x in raw_utp):
                raise ValueError("Ledger 'untrusted_tainted_paths' must be list[str]")
            if not isinstance(raw_evicted, list) or not all(
                isinstance(x, str) for x in raw_evicted
            ):
                raise ValueError("Ledger 'evicted_secret_candidates' must be list[str]")

            parsed_priv = [TaintSpan.model_validate(item) for item in raw_priv]
            parsed_untrust = [TaintSpan.model_validate(item) for item in raw_untrust]

            parsed_tp: dict[str, set[str]] = {}
            for k, vals in raw_tp.items():
                if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
                    raise ValueError("Ledger 'tainted_paths' values must be list[str]")
                parsed_tp[str(k)] = {v for v in vals}

            parsed_prp: dict[str, set[str]] = {}
            for k, vals in raw_prp.items():
                if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
                    raise ValueError("Ledger 'private_read_paths' values must be list[str]")
                parsed_prp[str(k)] = {v for v in vals}

            parsed_out: dict[str, list[dict[str, str]]] = {}
            for dest, entries in raw_out.items():
                if not isinstance(entries, list):
                    raise ValueError("Ledger 'outbound_history' values must be lists")
                bounded_entries: list[dict[str, str]] = []
                for entry in entries[-MAX_OUTBOUND_CALLS_PER_DEST:]:
                    if not isinstance(entry, dict) or not all(
                        isinstance(av, str) for av in entry.values()
                    ):
                        raise ValueError("Ledger 'outbound_history' entry must be dict[str, str]")
                    bounded_entries.append({str(ak): av for ak, av in entry.items()})
                parsed_out[str(dest)] = bounded_entries

            default_counter = len(parsed_priv) + len(parsed_untrust)
            raw_counter = data.get("span_counter", default_counter)
            if isinstance(raw_counter, bool) or not isinstance(raw_counter, int):
                raise ValueError("Ledger 'span_counter' must be an integer")

            clean_intent = (
                raw_intent.strip()
                if isinstance(raw_intent, str)
                and raw_intent.strip()
                and raw_intent.strip() != "<|missing_intent|>"
                else None
            )
            if raw_intent_hist is not None:
                hist = [
                    x.strip()
                    for x in raw_intent_hist
                    if x.strip() and x.strip() != "<|missing_intent|>"
                ][-3:]
            elif clean_intent is not None:
                hist = [clean_intent]
            else:
                hist = []

            self._intent_history = hist
            self._intent = "\n".join(hist) if hist else clean_intent
            self._private_spans = parsed_priv
            self._untrusted_spans = parsed_untrust
            self._tainted_paths = parsed_tp
            self._private_read_paths = parsed_prp
            self._untrusted_tainted_paths = {x for x in raw_utp if x}
            self._evicted_secret_candidates = list(raw_evicted)
            self._outbound_history = parsed_out
            self._span_counter = raw_counter
            self._span_sha256 = {
                sp.span_id: hashlib.sha256(sp.text.encode("utf-8")).hexdigest()
                for sp in (*parsed_priv, *parsed_untrust)
            }
        except Exception:
            # Fail-closed (C5): corrupt persisted session file must never revert to empty fast-path
            self._intent = None
            self._intent_history = []
            self._untrusted_spans = []
            self._tainted_paths = {}
            self._private_read_paths = {}
            self._untrusted_tainted_paths = set()
            self._evicted_secret_candidates = []
            self._outbound_history = {}
            sentinel = TaintSpan(
                span_id=self._next_span_id(),
                source_label="corrupt_ledger_fail_closed",
                text="[CORRUPT_PERSISTED_LEDGER_FAIL_CLOSED]",
                flags=CapabilityFlag.READS_PRIVATE
                | CapabilityFlag.HIGH_SECRET
                | CapabilityFlag.READS_UNTRUSTED,
                dual_trust=True,
                simhash=0,
            )
            self._private_spans = [sentinel]
            self._span_sha256 = {
                sentinel.span_id: hashlib.sha256(sentinel.text.encode("utf-8")).hexdigest()
            }

    def _persist_if_configured(self) -> None:
        if self._persist_path is None:
            return
        payload: dict[str, Any] = {
            "workspace_key": self.workspace_key,
            "intent": self._intent,
            "intent_history": self._intent_history,
            "span_counter": self._span_counter,
            "private_spans": [s.model_dump(mode="json") for s in self._private_spans],
            "untrusted_spans": [s.model_dump(mode="json") for s in self._untrusted_spans],
            "tainted_paths": {k: sorted(v) for k, v in sorted(self._tainted_paths.items())},
            "private_read_paths": {
                k: sorted(v) for k, v in sorted(self._private_read_paths.items())
            },
            "untrusted_tainted_paths": sorted(self._untrusted_tainted_paths),
            "evicted_secret_candidates": self._evicted_secret_candidates,
            "outbound_history": self._outbound_history,
        }
        _atomic_write_json_0600(self._persist_path, payload)

    def record_intent(self, intent: str | None) -> None:
        """Maintain a rolling window of the last 3 non-empty user intents."""
        if intent is None:
            self._intent = None
            self._intent_history.clear()
        else:
            trimmed = _utf8_truncate_head(intent.strip(), self._max_span_bytes)
            if not trimmed or trimmed == "<|missing_intent|>":
                self._intent = None
                self._intent_history.clear()
            else:
                if not self._intent_history or self._intent_history[-1] != trimmed:
                    self._intent_history.append(trimmed)
                    if len(self._intent_history) > 3:
                        del self._intent_history[:-3]
                combined = "\n".join(self._intent_history)
                if len(combined.encode("utf-8")) > self._max_span_bytes:
                    combined = _utf8_truncate_tail(combined, self._max_span_bytes)
                self._intent = combined
        self._persist_if_configured()

    def _find_trust_scoped_duplicate(
        self,
        pool: list[TaintSpan],
        cleaned_text: str,
        cand_sha: str,
        flags: CapabilityFlag,
        is_dual: bool,
        simhash: int,
    ) -> TaintSpan | None:
        is_high_sec = bool(flags & CapabilityFlag.HIGH_SECRET)
        cand_secrets: set[str] | None = None

        for existing in pool:
            # I3: Dedup MUST NEVER cross (flags, dual_trust) trust boundaries
            if existing.flags != flags or existing.dual_trust != is_dual:
                continue

            existing_sha = self._span_sha256.get(existing.span_id)
            if existing_sha is None:
                existing_sha = hashlib.sha256(existing.text.encode("utf-8")).hexdigest()
                self._span_sha256[existing.span_id] = existing_sha
            if existing_sha == cand_sha:
                return existing

            # Never SimHash-dedup HIGH_SECRET spans (rotated keys differ by only a few chars!)
            if is_high_sec:
                continue

            if (
                existing.simhash is not None
                and len(cleaned_text) >= 64
                and len(existing.text) >= 64
                and _hamming_distance64(existing.simhash, simhash) <= 3
            ):
                if cand_secrets is None:
                    probe = TaintSpan(
                        span_id="probe",
                        source_label="probe",
                        text=cleaned_text,
                        flags=flags,
                        dual_trust=is_dual,
                    )
                    cand_secrets = set(_extract_secret_candidates(probe))
                if set(_extract_secret_candidates(existing)) == cand_secrets:
                    return existing

        return None

    def _upsert_evicted_summary_span(self, new_secrets: list[str]) -> bool:
        added = False
        existing_set = set(self._evicted_secret_candidates)
        for sec in new_secrets:
            if sec not in existing_set:
                existing_set.add(sec)
                self._evicted_secret_candidates.append(sec)
                added = True
        if not self._evicted_secret_candidates:
            return False

        summary_text = _sanitize_and_cap_span_text(
            "\n".join(self._evicted_secret_candidates),
            self._max_span_bytes,
        )
        summary_sha = hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
        summary_simhash = compute_simhash64(summary_text)

        for idx, sp in enumerate(self._private_spans):
            if sp.source_label == "evicted_secrets_summary":
                if added:
                    updated = TaintSpan(
                        span_id=sp.span_id,
                        source_label="evicted_secrets_summary",
                        text=summary_text,
                        flags=CapabilityFlag.READS_PRIVATE | CapabilityFlag.HIGH_SECRET,
                        dual_trust=False,
                        simhash=summary_simhash,
                        timestamp=time.time(),
                    )
                    self._private_spans[idx] = updated
                    self._span_sha256[updated.span_id] = summary_sha
                return True

        summary_span = TaintSpan(
            span_id=f"evicted_summary_{self.workspace_key[:8]}",
            source_label="evicted_secrets_summary",
            text=summary_text,
            flags=CapabilityFlag.READS_PRIVATE | CapabilityFlag.HIGH_SECRET,
            dual_trust=False,
            simhash=summary_simhash,
            timestamp=time.time(),
        )
        self._private_spans.append(summary_span)
        self._span_sha256[summary_span.span_id] = summary_sha
        return True

    def _insert_private_span(self, span: TaintSpan) -> None:
        if len(self._private_spans) >= self._max_private_spans:
            evict_idx: int | None = None
            for idx, existing in enumerate(self._private_spans):
                if not bool(existing.flags & CapabilityFlag.HIGH_SECRET):
                    evict_idx = idx
                    break
            if evict_idx is not None:
                evicted = self._private_spans.pop(evict_idx)
                self._span_sha256.pop(evicted.span_id, None)
                if evicted.source_label != "evicted_secrets_summary":
                    evicted_secrets = _extract_secret_candidates(evicted)
                    if evicted_secrets and self._max_private_spans >= 2:
                        has_summary = any(
                            sp.source_label == "evicted_secrets_summary"
                            for sp in self._private_spans
                        )
                        if not has_summary:
                            # Make room for both pinned summary span and incoming span
                            extra_idx = next(
                                (
                                    i
                                    for i, sp in enumerate(self._private_spans)
                                    if not bool(sp.flags & CapabilityFlag.HIGH_SECRET)
                                ),
                                None,
                            )
                            if extra_idx is not None:
                                extra_ev = self._private_spans.pop(extra_idx)
                                self._span_sha256.pop(extra_ev.span_id, None)
                                evicted_secrets.extend(_extract_secret_candidates(extra_ev))
                        self._upsert_evicted_summary_span(evicted_secrets)
                        if len(self._private_spans) >= self._max_private_spans and not bool(
                            span.flags & CapabilityFlag.HIGH_SECRET
                        ):
                            return
                    elif evicted_secrets:
                        for sec in evicted_secrets:
                            if sec not in self._evicted_secret_candidates:
                                self._evicted_secret_candidates.append(sec)
            elif not bool(span.flags & CapabilityFlag.HIGH_SECRET):
                # All capacity slots hold pinned HIGH_SECRET spans; never evict them
                return
        self._private_spans.append(span)

    def _insert_untrusted_span(self, span: TaintSpan) -> None:
        while len(self._untrusted_spans) >= self._max_untrusted_spans:
            evicted = self._untrusted_spans.pop(0)
            self._span_sha256.pop(evicted.span_id, None)
        self._untrusted_spans.append(span)

    def record_tool_output(
        self,
        source_label: str,
        text: str,
        flags: CapabilityFlag,
        *,
        dual_trust: bool = False,
        target_path: str | None = None,
    ) -> TaintSpan | None:
        """Record a private, untrusted, or dual-trust tool output span in the ledger."""
        if not text:
            return None

        effective_flags = flags
        if target_path:
            canon_tp = canonicalize_target_path(target_path, self._workspace_root)
            if canon_tp in self._tainted_paths:
                effective_flags |= CapabilityFlag.READS_PRIVATE
                for sp in self._private_spans:
                    if sp.span_id in self._tainted_paths[canon_tp] and bool(
                        sp.flags & CapabilityFlag.HIGH_SECRET
                    ):
                        effective_flags |= CapabilityFlag.HIGH_SECRET
            if canon_tp in self._untrusted_tainted_paths:
                effective_flags |= CapabilityFlag.READS_UNTRUSTED

        if (
            "__memory_graph__" in self._untrusted_tainted_paths
            and source_label.strip().lower().startswith("memory:")
        ):
            effective_flags |= CapabilityFlag.READS_UNTRUSTED

        if bool(effective_flags & CapabilityFlag.HIGH_SECRET):
            effective_flags |= CapabilityFlag.READS_PRIVATE

        is_dual = bool(dual_trust) or (
            bool(effective_flags & CapabilityFlag.READS_PRIVATE)
            and bool(effective_flags & CapabilityFlag.READS_UNTRUSTED)
        )
        if is_dual:
            effective_flags |= CapabilityFlag.READS_PRIVATE | CapabilityFlag.READS_UNTRUSTED

        taint_mask = (
            CapabilityFlag.READS_PRIVATE
            | CapabilityFlag.HIGH_SECRET
            | CapabilityFlag.READS_UNTRUSTED
        )
        if not bool(effective_flags & taint_mask):
            return None

        cleaned = _sanitize_and_cap_span_text(text, self._max_span_bytes)
        cand_sha = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        simhash = compute_simhash64(cleaned)
        is_private_bucket = is_dual or bool(effective_flags & CapabilityFlag.READS_PRIVATE)
        target_pool = self._private_spans if is_private_bucket else self._untrusted_spans

        rec_read_path = (
            _extract_recordable_read_path(source_label, target_path, self._workspace_root)
            if is_private_bucket
            else None
        )

        dup = self._find_trust_scoped_duplicate(
            target_pool, cleaned, cand_sha, effective_flags, is_dual, simhash
        )
        if dup is not None:
            if rec_read_path:
                self._private_read_paths.setdefault(rec_read_path, set()).add(dup.span_id)
                self._persist_if_configured()
            return dup.model_copy(deep=True)

        span = TaintSpan(
            span_id=self._next_span_id(),
            source_label=source_label.strip() or "unknown",
            text=cleaned,
            flags=effective_flags,
            dual_trust=is_dual,
            simhash=simhash,
            timestamp=time.time(),
        )
        self._span_sha256[span.span_id] = cand_sha
        if is_private_bucket:
            self._insert_private_span(span)
            if rec_read_path:
                self._private_read_paths.setdefault(rec_read_path, set()).add(span.span_id)
        else:
            self._insert_untrusted_span(span)

        self._persist_if_configured()
        return span.model_copy(deep=True)

    def _written_content_overlaps_span(
        self,
        written_content: str,
        views: list[tuple[str, bool]],
        span: TaintSpan,
    ) -> bool:
        if not written_content:
            return False

        for sec in _extract_secret_candidates(span):
            matched, _ = _secret_matches_views(sec, [written_content], views)
            if matched:
                return True

        copy_score, _ = _detect_code_or_data_copy(span, views)
        if copy_score >= 0.20:
            return True

        for ident in _LONG_IDENT_RE.findall(span.text):
            if any(ident in v_text for v_text, _ in views):
                return True

        for line in span.text.splitlines():
            stripped = line.strip()
            if len(stripped) >= 24 and stripped not in _BOILERPLATE_LINES:
                if any(stripped in v_text for v_text, _ in views):
                    return True

        return False

    def record_local_write(self, target_path: str, written_content: str) -> TaintSpan | None:
        """Propagate private & untrusted taint to `target_path` when `written_content` overlaps active spans."""
        if not target_path or not written_content:
            return None
        if not self._private_spans and not self._untrusted_spans:
            return None

        canon_path = canonicalize_target_path(target_path, self._workspace_root)
        if not canon_path:
            return None

        detailed = _expand_arg_views_detailed([written_content])
        views = [(t, obf) for t, obf, _rev in detailed]

        # E10: Track untrusted writes when written content overlaps untrusted or dual-trust spans
        untrusted_pool = [
            *self._untrusted_spans,
            *(
                sp
                for sp in self._private_spans
                if sp.dual_trust or bool(sp.flags & CapabilityFlag.READS_UNTRUSTED)
            ),
        ]
        if any(
            self._written_content_overlaps_span(written_content, views, usp)
            for usp in untrusted_pool
        ):
            self._untrusted_tainted_paths.add(canon_path)
            self._persist_if_configured()

        if not self._private_spans:
            return None

        matched_spans = [
            sp
            for sp in self._private_spans
            if self._written_content_overlaps_span(written_content, views, sp)
        ]
        if not matched_spans:
            return None

        span_id_set = self._tainted_paths.setdefault(canon_path, set())
        synth_flags = CapabilityFlag.READS_PRIVATE
        synth_dual = False
        for sp in matched_spans:
            span_id_set.add(sp.span_id)
            if bool(sp.flags & CapabilityFlag.HIGH_SECRET):
                synth_flags |= CapabilityFlag.HIGH_SECRET
            if sp.dual_trust or bool(sp.flags & CapabilityFlag.READS_UNTRUSTED):
                synth_flags |= CapabilityFlag.READS_UNTRUSTED
                synth_dual = True

        staged_label = f"staged_write:{canon_path}"
        cleaned = _sanitize_and_cap_span_text(written_content, self._max_span_bytes)
        cleaned_sha = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        simhash = compute_simhash64(cleaned)

        # Update existing staged span for this path or insert a new one
        for idx, existing in enumerate(self._private_spans):
            if existing.source_label == staged_label:
                merged_flags = existing.flags | synth_flags
                merged_dual = existing.dual_trust or synth_dual
                updated = TaintSpan(
                    span_id=existing.span_id,
                    source_label=staged_label,
                    text=cleaned,
                    flags=merged_flags,
                    dual_trust=merged_dual,
                    simhash=simhash,
                    timestamp=time.time(),
                )
                self._private_spans[idx] = updated
                self._span_sha256[updated.span_id] = cleaned_sha
                span_id_set.add(updated.span_id)
                self._persist_if_configured()
                return updated.model_copy(deep=True)

        synth_span = TaintSpan(
            span_id=self._next_span_id(),
            source_label=staged_label,
            text=cleaned,
            flags=synth_flags,
            dual_trust=synth_dual,
            simhash=simhash,
            timestamp=time.time(),
        )
        self._span_sha256[synth_span.span_id] = cleaned_sha
        self._insert_private_span(synth_span)
        span_id_set.add(synth_span.span_id)
        self._persist_if_configured()
        return synth_span.model_copy(deep=True)

    def record_shell_side_effects(
        self,
        command: str,
        *,
        call_cwd: Path | str | None = None,
        is_high_secret_path_fn: Callable[[str], bool] | None = None,
    ) -> None:
        """Propagate private & untrusted path taint across shell `cp`/`mv`/redirection targets."""
        if not command:
            return
        ops = _extract_shell_copy_redirect_ops(command)
        if not ops:
            return
        bases: list[Path | None] = []
        if call_cwd is not None:
            bases.append(Path(os.fspath(call_cwd)))
        if self._workspace_root is not None and self._workspace_root not in bases:
            bases.append(self._workspace_root)
        if not bases:
            bases.append(None)

        changed = False
        has_untrust = self._has_untrusted_or_dual_taint()
        for src_toks, dst_tok, has_inline in ops:
            dst_canon = canonicalize_target_path(dst_tok, bases[0])
            if not dst_canon:
                continue

            matched_ids: set[str] = set()
            matched_hs = False
            src_untrusted = False

            for src_tok in src_toks:
                for base in bases:
                    src_canon = canonicalize_target_path(src_tok, base)
                    if not src_canon:
                        continue
                    src_fold = src_canon.casefold()
                    for tp, sids in self._tainted_paths.items():
                        if tp.casefold() == src_fold:
                            matched_ids.update(sids)
                    for rp, sids in self._private_read_paths.items():
                        if rp.casefold() == src_fold:
                            matched_ids.update(sids)
                    for utp in self._untrusted_tainted_paths:
                        if utp.casefold() == src_fold:
                            src_untrusted = True
                    if is_high_secret_path_fn is not None and is_high_secret_path_fn(src_fold):
                        matched_hs = True

            if matched_ids or matched_hs:
                tp_set = self._tainted_paths.setdefault(dst_canon, set())
                tp_set.update(matched_ids)
                if matched_hs:
                    staged_label = f"staged_write:{dst_canon}"
                    if not any(sp.source_label == staged_label for sp in self._private_spans):
                        txt = f"copied_secret_target={dst_canon} sources={','.join(src_toks)}"
                        sha = hashlib.sha256(txt.encode("utf-8")).hexdigest()
                        sp = TaintSpan(
                            span_id=self._next_span_id(),
                            source_label=staged_label,
                            text=txt,
                            flags=CapabilityFlag.READS_PRIVATE | CapabilityFlag.HIGH_SECRET,
                            dual_trust=False,
                            simhash=compute_simhash64(txt),
                            timestamp=time.time(),
                        )
                        self._span_sha256[sp.span_id] = sha
                        self._insert_private_span(sp)
                        tp_set.add(sp.span_id)
                changed = True

            if src_untrusted or (has_untrust and has_inline):
                self._untrusted_tainted_paths.add(dst_canon)
                changed = True

        if changed:
            self._persist_if_configured()

    def record_memory_write(self) -> None:
        """Taint the shared memory graph when untrusted taint is active (`E10`)."""
        if self._has_untrusted_or_dual_taint():
            self._untrusted_tainted_paths.add("__memory_graph__")
            self._persist_if_configured()

    def _is_strict_workspace_subdir(self, canon_fold: str, base: Path | None) -> bool:
        eff_base = base if base is not None else self._workspace_root
        if eff_base is None:
            return False
        canon_stripped = canon_fold.rstrip("/")
        if (
            not canon_stripped
            or _WIN_DRIVE_ROOT_FOLD_RE.match(canon_stripped) is not None
            or _UNC_SHARE_ROOT_FOLD_RE.match(canon_stripped) is not None
        ):
            return False
        ws_canon = canonicalize_target_path(os.fspath(eff_base), None)
        ws_stripped = ws_canon.casefold().rstrip("/")
        if (
            not ws_stripped
            or _WIN_DRIVE_ROOT_FOLD_RE.match(ws_stripped) is not None
            or _UNC_SHARE_ROOT_FOLD_RE.match(ws_stripped) is not None
        ):
            return False
        return canon_stripped != ws_stripped and canon_stripped.startswith(f"{ws_stripped}/")

    def _matched_ids_in_path_map(
        self,
        raw_candidate: str,
        path_map: dict[str, set[str]],
        bases: list[Path | None],
    ) -> set[str]:
        cleaned = raw_candidate.strip().lstrip("@")
        if not cleaned or cleaned.startswith(("http://", "https://")):
            return set()
        out: set[str] = set()
        for base in bases:
            canon = canonicalize_target_path(cleaned, base)
            if not canon:
                continue
            canon_fold = canon.casefold()
            allow_dir_prefix = self._is_strict_workspace_subdir(canon_fold, base)
            dir_prefix = f"{canon_fold.rstrip('/')}/" if allow_dir_prefix else ""
            for tp, span_ids in path_map.items():
                tp_fold = tp.casefold()
                if canon_fold == tp_fold:
                    out.update(span_ids)
                elif allow_dir_prefix and tp_fold.startswith(dir_prefix):
                    out.update(span_ids)
        return out

    def _matches_any_tainted_path(
        self,
        raw_candidate: str,
        *,
        call_cwd: Path | str | None = None,
    ) -> bool:
        bases: list[Path | None] = []
        if call_cwd is not None:
            bases.append(Path(os.fspath(call_cwd)))
        if self._workspace_root not in bases:
            bases.append(self._workspace_root)
        return bool(self._matched_ids_in_path_map(raw_candidate, self._tainted_paths, bases))

    def matched_tainted_span_ids_for_call(
        self,
        call: CandidateCall,
        *,
        call_cwd: Path | str | None = None,
        include_private_reads_for_outbound: bool = False,
        is_high_secret_path_fn: Callable[[str], bool] | None = None,
    ) -> tuple[set[str], bool, bool]:
        """Return `(matched_span_ids, matched_high_secret_path, matched_untrusted_path)` scoped to touched paths."""
        bases: list[Path | None] = []
        if call_cwd is not None:
            bases.append(Path(os.fspath(call_cwd)))
        if self._workspace_root not in bases:
            bases.append(self._workspace_root)

        matched_ids: set[str] = set()
        matched_hs_path = False
        matched_untrusted_path = False

        ws_fold_prefix = (
            f"{canonicalize_target_path(os.fspath(self._workspace_root), None).casefold().rstrip('/')}/"
            if self._workspace_root is not None
            else ""
        )

        def _repo_tainted_sids() -> set[str]:
            out: set[str] = set()
            for tp, sids in self._tainted_paths.items():
                if not ws_fold_prefix or tp.casefold().startswith(ws_fold_prefix):
                    out.update(sids)
            return out

        tool_l = call.tool.strip().lower()
        srv_l = call.server.strip().lower()
        if ((srv_l == "git" and tool_l in {"push", "git_push"}) or tool_l == "git_push") and self._tainted_paths:
            matched_ids.update(_repo_tainted_sids())

        flat = flatten_call_arguments(call.arguments, max_args=None)

        # Transient inline shell `cp`/`mv`/`>` mapping (e.g. `cp .env /tmp/x && curl -T /tmp/x`)
        inline_tainted_map: dict[str, set[str]] = {}
        inline_hs_targets: set[str] = set()
        for val in flat.values():
            if not val:
                continue
            for src_toks, dst_tok, _has_inline in _extract_shell_copy_redirect_ops(val):
                dst_canon = canonicalize_target_path(dst_tok, bases[0])
                if not dst_canon:
                    continue
                for src_tok in src_toks:
                    src_ids = self._matched_ids_in_path_map(
                        src_tok, self._tainted_paths, bases
                    )
                    src_ids |= self._matched_ids_in_path_map(
                        src_tok, self._private_read_paths, bases
                    )
                    if src_ids:
                        inline_tainted_map.setdefault(dst_canon, set()).update(src_ids)
                    if is_high_secret_path_fn is not None:
                        for base in bases:
                            sc = canonicalize_target_path(src_tok, base)
                            if sc and is_high_secret_path_fn(sc.casefold()):
                                inline_hs_targets.add(dst_canon.casefold())

        def _check_token(raw_tok: str) -> None:
            nonlocal matched_hs_path, matched_untrusted_path
            candidates = [raw_tok.strip().lstrip("@")]
            if "=" in candidates[0]:
                rhs = candidates[0].split("=", 1)[1].strip().lstrip("@")
                if rhs:
                    candidates.append(rhs)

            for cleaned in candidates:
                if not cleaned or cleaned.startswith(("http://", "https://")):
                    continue
                matched_ids.update(
                    self._matched_ids_in_path_map(cleaned, self._tainted_paths, bases)
                )
                for base in bases:
                    canon = canonicalize_target_path(cleaned, base)
                    if not canon:
                        continue
                    cf = canon.casefold()
                    if any(utp.casefold() == cf for utp in self._untrusted_tainted_paths):
                        matched_untrusted_path = True
                    if include_private_reads_for_outbound:
                        if cf in inline_hs_targets:
                            matched_hs_path = True
                        if is_high_secret_path_fn is not None and is_high_secret_path_fn(cf):
                            matched_hs_path = True
                if include_private_reads_for_outbound:
                    matched_ids.update(
                        self._matched_ids_in_path_map(cleaned, self._private_read_paths, bases)
                    )
                    if inline_tainted_map:
                        matched_ids.update(
                            self._matched_ids_in_path_map(cleaned, inline_tainted_map, bases)
                        )

        for flat_k, val in flat.items():
            if not val:
                continue
            leaf_key = flat_k.rsplit(".", 1)[-1].split("[", 1)[0].lower()
            if leaf_key in _PATH_HINT_KEYS:
                _check_token(val)

            if self._tainted_paths and _REPO_WIDE_PUBLISH_CMD_RE.search(val) is not None:
                matched_ids.update(_repo_tainted_sids())

            for tok in _SHELL_TOKEN_SPLIT_RE.split(val):
                tok_clean = tok.strip().lstrip("@")
                if not tok_clean or tok_clean.startswith("-"):
                    continue
                if set(tok_clean) <= {".", "/", "\\"}:
                    continue
                if "/" in tok_clean or "\\" in tok_clean or "." in tok_clean:
                    _check_token(tok_clean)

        return matched_ids, matched_hs_path, matched_untrusted_path

    def touches_tainted_local_write(self, call_or_path: CandidateCall | str | Path) -> bool:
        """Return `True` if a path or `CandidateCall` reads, uploads, or pushes a tainted local file."""
        if not self._tainted_paths:
            return False

        if isinstance(call_or_path, (str, Path)):
            return self._matches_any_tainted_path(os.fspath(call_or_path))

        matched_ids, matched_hs, matched_untrust = self.matched_tainted_span_ids_for_call(
            call_or_path,
            include_private_reads_for_outbound=False,
        )
        return bool(matched_ids or matched_hs or matched_untrust)

    def record_outbound_call(self, destination_key: str, call: CandidateCall) -> str:
        """Record outbound call args in per-destination and session-wide rolling windows (`C7`)."""
        dest = destination_key.strip() or call.server.strip() or "default"
        flat = flatten_call_arguments(call.arguments, max_args=None)
        bounded_entry = {
            k: _utf8_truncate_tail(v, MAX_OUTBOUND_BYTES_PER_DEST) for k, v in flat.items()
        }
        window = self._outbound_history.setdefault(dest, [])
        window.append(bounded_entry)
        if len(window) > MAX_OUTBOUND_CALLS_PER_DEST:
            del window[:-MAX_OUTBOUND_CALLS_PER_DEST]

        if dest != "__session__":
            sess_window = self._outbound_history.setdefault("__session__", [])
            sess_window.append(dict(bounded_entry))
            if len(sess_window) > MAX_OUTBOUND_CALLS_PER_DEST:
                del sess_window[:-MAX_OUTBOUND_CALLS_PER_DEST]

        reassembled = _reassemble_outbound_window(window, MAX_OUTBOUND_BYTES_PER_DEST)
        self._persist_if_configured()
        return reassembled

    def _has_untrusted_or_dual_taint(self) -> bool:
        if self._untrusted_spans:
            return True
        return any(
            s.dual_trust or bool(s.flags & CapabilityFlag.READS_UNTRUSTED)
            for s in self._private_spans
        )

    def should_fast_path_allow(
        self,
        call_flags: CapabilityFlag,
        *,
        is_self_tamper: bool = False,
        touches_tainted_local_write: bool = False,
        is_known_safe_tool: bool = True,
    ) -> bool:
        """Fast-path `<0.1ms` bypass gate before invoking `PredictorProtocol`."""
        if is_self_tamper or touches_tainted_local_write:
            return False

        has_taint = bool(self._private_spans) or self._has_untrusted_or_dual_taint()
        if has_taint and not is_known_safe_tool:
            return False

        writes_ext = bool(call_flags & CapabilityFlag.WRITES_EXTERNAL)
        mutates_sens = bool(call_flags & CapabilityFlag.MUTATES_SENSITIVE)

        if not writes_ext and not mutates_sens:
            return True
        if writes_ext and not mutates_sens:
            return len(self._private_spans) == 0
        if mutates_sens and not writes_ext:
            return not self._has_untrusted_or_dual_taint()
        return False

    def snapshot(self) -> LedgerSnapshot:
        """Return an immutable `LedgerSnapshot` of current session taint state."""
        outbound_acc = {
            dest: _reassemble_outbound_window(hist, MAX_OUTBOUND_BYTES_PER_DEST)
            for dest, hist in self._outbound_history.items()
            if hist
        }
        return LedgerSnapshot(
            intent=self._intent,
            private_spans=tuple(self._private_spans),
            untrusted_spans=tuple(self._untrusted_spans),
            tainted_paths=frozenset(self._tainted_paths.keys()),
            outbound_accumulator=outbound_acc,
        )
