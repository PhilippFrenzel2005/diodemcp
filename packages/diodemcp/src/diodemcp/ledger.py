"""Workspace-scoped persistent session taint ledger, SimHash dedup, and split-exfil accumulator."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
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

    # Preserve any secrets located past max_bytes so 64 KiB truncation cannot hide tail credentials
    head_text = raw_bytes[:max_bytes].decode("utf-8", errors="ignore")
    tail_text = raw_bytes[max_bytes:].decode("utf-8", errors="ignore")
    tail_secrets = _extract_printable_secret_lines(tail_text)
    if not tail_secrets:
        return head_text

    tail_suffix = "\n" + "\n".join(tail_secrets)
    suffix_bytes = tail_suffix.encode("utf-8", errors="replace")
    if len(suffix_bytes) >= max_bytes // 2:
        suffix_bytes = suffix_bytes[: max_bytes // 2]
        tail_suffix = suffix_bytes.decode("utf-8", errors="ignore")
    head_budget = max_bytes - len(tail_suffix.encode("utf-8"))
    return _utf8_truncate_head(head_text, head_budget) + tail_suffix


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
                url_candidates.extend(m.group(0) for m in _EMBEDDED_URL_OR_QUERY_RE.finditer(val))

            for url_cand in url_candidates:
                parsed = urllib.parse.urlsplit(url_cand)
                if parsed.query:
                    for qk, qv in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
                        param_streams.setdefault(f"{arg_key}?{qk}", []).append(qv)
                if parsed.path and parsed.path != "/":
                    tail_seg = parsed.path.rstrip("/").rsplit("/", 1)[-1]
                    if tail_seg:
                        path_tail_stream.append(urllib.parse.unquote(tail_seg))

    sections: list[str] = []
    for pk, chunks in param_streams.items():
        if len(chunks) >= 2:
            sections.append(f"[param:{pk}]={''.join(chunks)}")
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
        self._private_spans: list[TaintSpan] = []
        self._untrusted_spans: list[TaintSpan] = []
        self._tainted_paths: dict[str, set[str]] = {}
        self._outbound_history: dict[str, list[dict[str, str]]] = {}
        self._span_counter: int = 0

        if self._persist_path is not None and self._persist_path.exists():
            self._rehydrate_from_disk(self._persist_path)

    @property
    def persist_path(self) -> Path | None:
        return self._persist_path

    def clear(self) -> None:
        """Reset all session taint state (explicit new session / `/clear`) and persist."""
        self._intent = None
        self._private_spans.clear()
        self._untrusted_spans.clear()
        self._tainted_paths.clear()
        self._outbound_history.clear()
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

            raw_priv = data.get("private_spans", [])
            raw_untrust = data.get("untrusted_spans", [])
            raw_tp = data.get("tainted_paths", {})
            raw_out = data.get("outbound_history", {})

            if not isinstance(raw_priv, list) or not isinstance(raw_untrust, list):
                raise ValueError("Ledger span collections must be lists")
            if not isinstance(raw_tp, dict) or not isinstance(raw_out, dict):
                raise ValueError("Ledger path/outbound maps must be objects")

            parsed_priv = [TaintSpan.model_validate(item) for item in raw_priv]
            parsed_untrust = [TaintSpan.model_validate(item) for item in raw_untrust]

            parsed_tp: dict[str, set[str]] = {}
            for k, vals in raw_tp.items():
                if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
                    raise ValueError("Ledger 'tainted_paths' values must be list[str]")
                parsed_tp[str(k)] = {v for v in vals}

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

            self._intent = (
                raw_intent.strip()
                if isinstance(raw_intent, str)
                and raw_intent.strip()
                and raw_intent.strip() != "<|missing_intent|>"
                else None
            )
            self._private_spans = parsed_priv
            self._untrusted_spans = parsed_untrust
            self._tainted_paths = parsed_tp
            self._outbound_history = parsed_out
            self._span_counter = raw_counter
        except Exception:
            # Fail-closed (C5): corrupt persisted session file must never revert to empty fast-path
            self._intent = None
            self._untrusted_spans = []
            self._tainted_paths = {}
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

    def _persist_if_configured(self) -> None:
        if self._persist_path is None:
            return
        payload: dict[str, Any] = {
            "workspace_key": self.workspace_key,
            "intent": self._intent,
            "span_counter": self._span_counter,
            "private_spans": [s.model_dump(mode="json") for s in self._private_spans],
            "untrusted_spans": [s.model_dump(mode="json") for s in self._untrusted_spans],
            "tainted_paths": {k: sorted(v) for k, v in sorted(self._tainted_paths.items())},
            "outbound_history": self._outbound_history,
        }
        _atomic_write_json_0600(self._persist_path, payload)

    def record_intent(self, intent: str | None) -> None:
        """Update current user intent (trimmed; empty or `<|missing_intent|>` -> `None`)."""
        if intent is None:
            self._intent = None
        else:
            trimmed = _utf8_truncate_head(intent.strip(), self._max_span_bytes)
            self._intent = (
                None if (not trimmed or trimmed == "<|missing_intent|>") else trimmed
            )
        self._persist_if_configured()

    def _find_trust_scoped_duplicate(
        self,
        pool: list[TaintSpan],
        cleaned_text: str,
        flags: CapabilityFlag,
        is_dual: bool,
        simhash: int,
    ) -> TaintSpan | None:
        cand_sha = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        is_high_sec = bool(flags & CapabilityFlag.HIGH_SECRET)
        cand_secrets: set[str] | None = None

        for existing in pool:
            # I3: Dedup MUST NEVER cross (flags, dual_trust) trust boundaries
            if existing.flags != flags or existing.dual_trust != is_dual:
                continue

            existing_sha = hashlib.sha256(existing.text.encode("utf-8")).hexdigest()
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

    def _insert_private_span(self, span: TaintSpan) -> None:
        if len(self._private_spans) >= self._max_private_spans:
            evict_idx: int | None = None
            for idx, existing in enumerate(self._private_spans):
                if not bool(existing.flags & CapabilityFlag.HIGH_SECRET):
                    evict_idx = idx
                    break
            if evict_idx is not None:
                self._private_spans.pop(evict_idx)
            elif not bool(span.flags & CapabilityFlag.HIGH_SECRET):
                # All capacity slots hold pinned HIGH_SECRET spans; never evict them
                return
        self._private_spans.append(span)

    def _insert_untrusted_span(self, span: TaintSpan) -> None:
        while len(self._untrusted_spans) >= self._max_untrusted_spans:
            self._untrusted_spans.pop(0)
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
        simhash = compute_simhash64(cleaned)
        is_private_bucket = is_dual or bool(effective_flags & CapabilityFlag.READS_PRIVATE)
        target_pool = self._private_spans if is_private_bucket else self._untrusted_spans

        dup = self._find_trust_scoped_duplicate(
            target_pool, cleaned, effective_flags, is_dual, simhash
        )
        if dup is not None:
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
        if is_private_bucket:
            self._insert_private_span(span)
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
        """Propagate taint to `target_path` if `written_content` overlaps any active private span (`C7`)."""
        if not target_path or not written_content or not self._private_spans:
            return None

        canon_path = canonicalize_target_path(target_path, self._workspace_root)
        if not canon_path:
            return None

        detailed = _expand_arg_views_detailed([written_content])
        views = [(t, obf) for t, obf, _rev in detailed]

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
        self._insert_private_span(synth_span)
        span_id_set.add(synth_span.span_id)
        self._persist_if_configured()
        return synth_span.model_copy(deep=True)

    def _is_strict_workspace_subdir(self, canon_fold: str) -> bool:
        if self._workspace_root is None:
            return False
        canon_stripped = canon_fold.rstrip("/")
        if (
            not canon_stripped
            or _WIN_DRIVE_ROOT_FOLD_RE.match(canon_stripped) is not None
            or _UNC_SHARE_ROOT_FOLD_RE.match(canon_stripped) is not None
        ):
            return False
        ws_canon = canonicalize_target_path(os.fspath(self._workspace_root), None)
        ws_stripped = ws_canon.casefold().rstrip("/")
        if (
            not ws_stripped
            or _WIN_DRIVE_ROOT_FOLD_RE.match(ws_stripped) is not None
            or _UNC_SHARE_ROOT_FOLD_RE.match(ws_stripped) is not None
        ):
            return False
        return canon_stripped != ws_stripped and canon_stripped.startswith(f"{ws_stripped}/")

    def _matches_any_tainted_path(self, raw_candidate: str) -> bool:
        cleaned = raw_candidate.strip().lstrip("@")
        if not cleaned or cleaned.startswith(("http://", "https://")):
            return False
        canon = canonicalize_target_path(cleaned, self._workspace_root)
        if not canon:
            return False
        canon_fold = canon.casefold()
        allow_dir_prefix = self._is_strict_workspace_subdir(canon_fold)
        dir_prefix = f"{canon_fold.rstrip('/')}/" if allow_dir_prefix else ""
        for tp in self._tainted_paths:
            tp_fold = tp.casefold()
            if canon_fold == tp_fold:
                return True
            if allow_dir_prefix and tp_fold.startswith(dir_prefix):
                return True
        return False

    def touches_tainted_local_write(self, call_or_path: CandidateCall | str | Path) -> bool:
        """Return `True` if a path or `CandidateCall` reads, uploads, or pushes a tainted local file."""
        if not self._tainted_paths:
            return False

        if isinstance(call_or_path, (str, Path)):
            return self._matches_any_tainted_path(os.fspath(call_or_path))

        tool_l = call_or_path.tool.strip().lower()
        srv_l = call_or_path.server.strip().lower()
        if (srv_l == "git" and tool_l in {"push", "git_push"}) or tool_l == "git_push":
            return True

        flat = flatten_call_arguments(call_or_path.arguments, max_args=None)
        for flat_k, val in flat.items():
            if not val:
                continue
            leaf_key = flat_k.rsplit(".", 1)[-1].split("[", 1)[0].lower()
            if leaf_key in _PATH_HINT_KEYS and self._matches_any_tainted_path(val):
                return True

            if _REPO_WIDE_PUBLISH_CMD_RE.search(val) is not None:
                return True

            for tok in _SHELL_TOKEN_SPLIT_RE.split(val):
                tok_clean = tok.strip().lstrip("@")
                if not tok_clean or tok_clean.startswith("-"):
                    continue
                if set(tok_clean) <= {".", "/", "\\"}:
                    continue
                if "/" in tok_clean or "\\" in tok_clean or "." in tok_clean:
                    if self._matches_any_tainted_path(tok_clean):
                        return True

        return False

    def record_outbound_call(self, destination_key: str, call: CandidateCall) -> str:
        """Record outbound call arguments in a 16-call / 8 KiB rolling window per destination (`C7`)."""
        dest = destination_key.strip() or call.server.strip() or "default"
        flat = flatten_call_arguments(call.arguments, max_args=None)
        bounded_entry = {
            k: _utf8_truncate_tail(v, MAX_OUTBOUND_BYTES_PER_DEST) for k, v in flat.items()
        }
        window = self._outbound_history.setdefault(dest, [])
        window.append(bounded_entry)
        if len(window) > MAX_OUTBOUND_CALLS_PER_DEST:
            del window[:-MAX_OUTBOUND_CALLS_PER_DEST]

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
    ) -> bool:
        """Fast-path `<0.1ms` bypass gate before invoking `PredictorProtocol`."""
        if is_self_tamper or touches_tainted_local_write:
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
