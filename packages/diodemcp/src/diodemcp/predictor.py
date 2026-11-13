"""PredictorProtocol and zero-ML Tier0DeterministicPredictor fallback."""

from __future__ import annotations

import base64
import binascii
from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import ipaddress
import math
import os
import re
from typing import Any, Protocol, runtime_checkable
import unicodedata
import urllib.parse

from diodemcp.constants import CRITICAL_THRESHOLD
from diodemcp.registry import expand_self_tamper_variants
from diodemcp.types import CandidateCall, CapabilityFlag, DecisionScores, TaintSpan

_SENSITIVE_KEY_WORDS = (
    "secret",
    "token",
    "key",
    "pass",
    "pwd",
    "auth",
    "cred",
    "private",
    "cookie",
    "session",
    "bearer",
    "dsn",
)

_KNOWN_SECRET_RE = re.compile(
    r"(?:"
    r"sk_(?:live|test)_[A-Za-z0-9]{8,}"
    r"|whsec_[A-Za-z0-9]{12,}"
    r"|gh[pousr]_[A-Za-z0-9]{12,}"
    r"|github_pat_[A-Za-z0-9_]{16,}"
    r"|AKIA[0-9A-Z]{12,}"
    r"|AIza[0-9A-Za-z_-]{20,}"
    r"|xox[baprs]-[0-9A-Za-z-]{12,}"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r")"
)
_KV_ASSIGN_RE = re.compile(
    r"""["']?([A-Za-z0-9_.-]{2,64})["']?\s*[:=]\s*["']?([^\s"'#;,{}[\]]{8,})["']?"""
)
_URI_PASSWORD_RE = re.compile(r"://[^:/@\s]+:([^@\s]{6,})@")
_HIGH_ENTROPY_TOKEN_RE = re.compile(r"[A-Za-z0-9+/_=-]{16,}")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_B64_CANDIDATE_RE = re.compile(r"[A-Za-z0-9+/_-]{12,}={0,2}")
_B32_CANDIDATE_RE = re.compile(r"[A-Za-z2-7]{12,}={0,6}")
_HEX_CANDIDATE_RE = re.compile(r"[0-9a-fA-F]{16,}")
_PCT_ALNUM_RE = re.compile(r"%[3-7][0-9a-fA-F]")

_CODE_LINE_HINT_RE = re.compile(
    r"(?:\b(?:def|class|return|import|from|async|await|function|const|let|SELECT|INSERT|UPDATE)\b|=>|:=|->)"
)

_DISCLOSURE_VERB_RE = re.compile(
    r"\b(?:"
    r"paste|pasting|post|posting|publish|publishing|share|sharing"
    r"|include|including|attach|attaching|upload|uploading|quote|quoting"
    r")\b",
    re.IGNORECASE,
)

_CLAUSE_SPLIT_RE = re.compile(
    r"(?:"
    r"[!?;:\n]+"
    r"|\.(?=\s|$)"
    r"|,\s*(?:and|then|but|while|so|or)\b"
    r"|\b(?:and\s+then|then|but|while|whereas|after|before)\b"
    r"|\band\s+(?=(?:please\s+|also\s+|now\s+)?(?:"
    r"paste|pasting|post|posting|publish|publishing|share|sharing"
    r"|include|including|attach|attaching|upload|uploading|quote|quoting"
    r"|create|creating|open|opening|fix|fixing|check|checking|read|reading"
    r"|inspect|inspecting|summarize|summarizing|run|running|test|testing"
    r"|review|reviewing|debug|debugging|write|writing|edit|editing"
    r"|update|updating|send|sending|report|reporting|file|filing|log|logging"
    r")\b)"
    r")",
    re.IGNORECASE,
)

_MUTATION_INTENT_VERB_RE = re.compile(
    r"\b(?:run|execute|drop|truncate|delete|remove|write|edit|update|modify|create|install|add|overwrite|push|reset)\b",
    re.IGNORECASE,
)

_SELF_PROTECTION_RE = re.compile(
    r"(?:"
    r"(?:^|[~/\"'\s\\;|&><=()])\.diodemcp(?:[/\\\"'\s;|&><=()]|$)"
    r"|(?:^|[/\\\"'\s;|&><=()])(?:engine\.sock|engine\.addr|admin\.secret)(?:[\"'\s;|&><=()]|$)"
    r"|(?:^|[~/\"'\s\\;|&><=()])\.claude\.json(?:[\"'\s;|&><=()]|$)"
    r"|(?:^|[~/\"'\s\\;|&><=()])\.claude[/\\]settings(?:\.[A-Za-z0-9_.-]+)?(?:[/\\\"'\s;|&><=()]|$)"
    r"|(?:^|[~/\"'\s\\;|&><=()])\.mcp\.json(?:[\"'\s;|&><=()]|$)"
    r"|(?:^|[~/\"'\s\\;|&><=()])\.cursor[/\\]mcp\.json(?:[\"'\s;|&><=()]|$)"
    r"|(?:^|[~/\"'\s\\;|&><=()])\.vscode[/\\]mcp\.json(?:[\"'\s;|&><=()]|$)"
    r"|(?:^|[/\\\"'\s;|&><=()])mcp_config\.json(?:[\"'\s;|&><=()]|$)"
    r"|\bdiodemcp(?:\.(?:exe|mjs|py|cli))?\s+(?:allow|approve|unwrap|model|trust|shutdown)\b"
    r")",
    re.IGNORECASE,
)

_SABOTAGE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:DROP\s+(?:TABLE|DATABASE|SCHEMA|INDEX)(?:\s+IF\s+EXISTS)?\s+[A-Za-z0-9_.\"`]+"
            r"|TRUNCATE\s+TABLE\s+[A-Za-z0-9_.\"`]+"
            r"|DELETE\s+FROM\s+[A-Za-z0-9_.\"`]+\s*(?:;|$))",
            re.IGNORECASE,
        ),
        "drop/truncate/delete sql",
    ),
    (
        re.compile(
            r"(?:curl|wget|base64|printf|echo)\b[^\n]{0,2048}\|\s*(?:sudo\s+)?(?:ba|z|da|k)?sh\b"
            r"|\b(?:ba|z)?sh\s+-c\s+[\"']?\$\((?:curl|wget)\b"
            r"|\beval\s+[\"']?\$\((?:curl|wget)\b",
            re.IGNORECASE,
        ),
        "pipe-to-shell execution",
    ),
    (
        re.compile(
            r"\b(?:rm\s+-[rRf]{2,}(?:\s+\S+)?"
            r"|git\s+push\s+(?:--force|-f)\b"
            r"|git\s+reset\s+--hard\b"
            r"|mkfs\.\w+"
            r"|dd\s+if=\S+)",
            re.IGNORECASE,
        ),
        "destructive command",
    ),
    (
        re.compile(
            r"(?:^|[/\\])\.git[/\\]hooks(?:[/\\][A-Za-z0-9_.-]+)?"
            r"|(?:^|[/\\])\.github[/\\]workflows(?:[/\\][A-Za-z0-9_.-]+)?"
            r"|(?:^|~|[/\\])\.ssh[/\\](?:authorized_keys|config|id_[A-Za-z0-9_]+)"
            r"|(?:^|~|[/\\])\.(?:bashrc|bash_profile|zshrc|profile|mcp\.json|cursorrules)(?:$|[/\\])"
            r"|(?:^|[/\\])\.(?:claude|cursor)[/\\][A-Za-z0-9_./\\-]*"
            r"|(?:^|[/\\])CLAUDE\.md$"
            r"|/etc/crontab\b",
            re.IGNORECASE,
        ),
        "persistence/config target",
    ),
)


@runtime_checkable
class PredictorProtocol(Protocol):
    """Unified synchronous predictor interface shared by Tier-0 and ORT Jev models."""

    def evaluate(
        self,
        intent: str | None,
        private_spans: list[TaintSpan],
        untrusted_spans: list[TaintSpan],
        call: CandidateCall,
    ) -> DecisionScores: ...


def _is_missing_intent(intent: str | None) -> bool:
    if intent is None:
        return True
    stripped = intent.strip()
    return stripped == "" or stripped == "<|missing_intent|>"


def _shannon_entropy(token: str) -> float:
    if not token:
        return 0.0
    length = len(token)
    counts = Counter(token)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _has_mixed_classes(token: str) -> bool:
    classes = (
        any(ch.islower() for ch in token),
        any(ch.isupper() for ch in token),
        any(ch.isdigit() for ch in token),
        any(not ch.isalnum() for ch in token),
    )
    return sum(classes) >= 3 or (any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token))


def _is_printable_utf8(data: bytes) -> str | None:
    if not data:
        return None
    try:
        decoded = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    if len(decoded) < 6:
        return None
    printable_count = sum(ch.isprintable() or ch in "\r\n\t" for ch in decoded)
    if printable_count / len(decoded) < 0.85:
        return None
    return decoded


def _collect_strings(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        if value:
            out.append(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and k:
                out.append(k)
            _collect_strings(v, out)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_strings(item, out)


_ROUTING_ARG_KEYS: frozenset[str] = frozenset(
    {
        "to",
        "cc",
        "bcc",
        "recipient",
        "recipients",
        "channel",
        "repo",
        "owner",
    }
)

_IBAN_RE = re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b")
_URL_EXTRACT_RE = re.compile(r"https?://[^\s\"'<>()\[\]{}]+", re.IGNORECASE)
_DOMAIN_EXTRACT_RE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b",
    re.IGNORECASE,
)

_MULTI_PART_PUBLIC_SUFFIXES: frozenset[str] = frozenset(
    {
        "co.uk",
        "org.uk",
        "ac.uk",
        "gov.uk",
        "net.uk",
        "com.au",
        "net.au",
        "org.au",
        "edu.au",
        "co.jp",
        "ne.jp",
        "or.jp",
        "co.nz",
        "co.kr",
        "com.br",
        "com.mx",
        "com.cn",
        "co.in",
        "com.sg",
        "com.hk",
        "co.za",
        "com.tr",
        "co.il",
    }
)

_SPAN_FEATURE_CACHE_MAX: int = 512


def normalize_nfkc_strip_cf(text: str) -> str:
    """Normalize NFKC compatibility characters and strip Unicode `Cf` format/zero-width chars."""
    if not text or text.isascii():
        return text
    nfkc = unicodedata.normalize("NFKC", text)
    if not any(unicodedata.category(ch) == "Cf" for ch in nfkc):
        return nfkc
    return "".join(ch for ch in nfkc if unicodedata.category(ch) != "Cf")


def extract_etld_plus_one(host_or_url: str) -> str | None:
    """Extract normalized eTLD+1 (or IP/localhost literal) from a URL or hostname safely."""
    raw = normalize_nfkc_strip_cf(host_or_url).strip()
    if not raw:
        return None
    host: str | None = None
    if "://" in raw:
        try:
            parsed = urllib.parse.urlsplit(raw)
        except ValueError:
            return None
        host = parsed.hostname
    else:
        candidate = raw.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        if "@" in candidate:
            candidate = candidate.rsplit("@", 1)[-1]
        if candidate.startswith("[") and "]" in candidate:
            candidate = candidate[1 : candidate.index("]")]
        elif ":" in candidate:
            candidate = candidate.split(":", 1)[0]
        host = candidate
    if not host:
        return None
    clean_host = host.strip().rstrip(".").lower()
    if not clean_host:
        return None
    try:
        ipaddress.ip_address(clean_host)
        return clean_host
    except ValueError:
        pass
    parts = [p for p in clean_host.split(".") if p]
    if len(parts) <= 2:
        return ".".join(parts)
    last_two = ".".join(parts[-2:])
    if last_two in _MULTI_PART_PUBLIC_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    return last_two


@dataclass(frozen=True)
class _StructuredDestinations:
    emails: frozenset[str]
    ibans: frozenset[str]
    urls: frozenset[str]
    etld1s: frozenset[str]

    @property
    def is_empty(self) -> bool:
        return not (self.emails or self.ibans or self.urls or self.etld1s)


def _extract_structured_destinations(text: str) -> _StructuredDestinations:
    norm = normalize_nfkc_strip_cf(text)
    if not norm:
        return _StructuredDestinations(
            emails=frozenset(), ibans=frozenset(), urls=frozenset(), etld1s=frozenset()
        )
    emails = frozenset(m.casefold() for m in _EMAIL_RE.findall(norm))
    ibans_set: set[str] = set()
    for m in _IBAN_RE.finditer(norm.upper()):
        cand = m.group(0)
        if sum(ch.isdigit() for ch in cand) >= 4:
            ibans_set.add(cand)
    urls_set: set[str] = set()
    etlds_set: set[str] = set()
    for match in _URL_EXTRACT_RE.finditer(norm):
        raw_url = match.group(0).rstrip(".,;:!?\"')")
        if not raw_url:
            continue
        try:
            parsed = urllib.parse.urlsplit(raw_url)
        except ValueError:
            continue
        if parsed.scheme and parsed.netloc:
            clean_url = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}"
            if parsed.query:
                clean_url = f"{clean_url}?{parsed.query}"
            urls_set.add(clean_url)
            et = extract_etld_plus_one(parsed.hostname or "")
            if et:
                etlds_set.add(et)
    return _StructuredDestinations(
        emails=emails,
        ibans=frozenset(ibans_set),
        urls=frozenset(urls_set),
        etld1s=frozenset(etlds_set),
    )


@dataclass(frozen=True)
class _SpanFeatures:
    norm_text: str
    secret_candidates: tuple[str, ...]
    nontrivial_lines: tuple[str, ...]
    nontrivial_lines_cf: tuple[str, ...]
    span_windows: tuple[str, ...]
    span_windows_cf: tuple[str, ...]
    span_emails: frozenset[str]
    structured_row_tokens: tuple[tuple[str, ...], ...]
    structured_destinations: _StructuredDestinations


_SPAN_FEATURE_CACHE: OrderedDict[tuple[str, int, str], _SpanFeatures] = OrderedDict()


def _expand_arg_views_detailed(raw_strings: list[str]) -> list[tuple[str, bool, bool]]:
    """Return `(view_text, is_obfuscated, is_reversed)` triples for raw, NFKC/Cf-clean, and decoded views."""
    views: list[tuple[str, bool, bool]] = []
    seen: set[tuple[str, bool, bool]] = set()

    def _add(text: str, obfuscated: bool, is_rev: bool) -> None:
        if not text:
            return
        key = (text, obfuscated, is_rev)
        if key not in seen:
            seen.add(key)
            views.append(key)

    base_inputs: list[tuple[str, bool]] = []
    for raw in raw_strings:
        _add(raw, False, False)
        _add(raw[::-1], True, True)
        base_inputs.append((raw, False))

        norm = normalize_nfkc_strip_cf(raw)
        if norm != raw:
            _add(norm, True, False)
            _add(norm[::-1], True, True)
            base_inputs.append((norm, True))

        for src_for_unq in (raw, norm) if norm != raw else (raw,):
            for unq in (
                urllib.parse.unquote_plus(src_for_unq),
                urllib.parse.unquote(src_for_unq),
            ):
                if unq != src_for_unq:
                    pct_obf = bool(_PCT_ALNUM_RE.search(src_for_unq))
                    _add(unq, pct_obf, False)
                    _add(unq[::-1], True, True)
                    base_inputs.append((unq, True))
                    unq_norm = normalize_nfkc_strip_cf(unq)
                    if unq_norm != unq:
                        _add(unq_norm, True, False)
                        _add(unq_norm[::-1], True, True)
                        base_inputs.append((unq_norm, True))

    for candidate_src, _ in base_inputs:
        for match in _B64_CANDIDATE_RE.finditer(candidate_src):
            tok = match.group(0).rstrip("=")
            padded = tok + ("=" * ((4 - len(tok) % 4) % 4))
            for decoder in (base64.b64decode, base64.urlsafe_b64decode):
                try:
                    raw_bytes = decoder(padded)
                except (binascii.Error, ValueError):
                    continue
                decoded = _is_printable_utf8(raw_bytes)
                if decoded is not None:
                    _add(decoded, True, False)
                    _add(decoded[::-1], True, True)

        for match in _B32_CANDIDATE_RE.finditer(candidate_src):
            tok = match.group(0).rstrip("=")
            padded = tok.upper() + ("=" * ((8 - len(tok) % 8) % 8))
            try:
                raw_bytes = base64.b32decode(padded, casefold=True)
            except (binascii.Error, ValueError):
                continue
            decoded = _is_printable_utf8(raw_bytes)
            if decoded is not None:
                _add(decoded, True, False)
                _add(decoded[::-1], True, True)

        for match in _HEX_CANDIDATE_RE.finditer(candidate_src):
            run = match.group(0)
            hex_slices = [run] if len(run) % 2 == 0 else [run[1:], run[:-1]]
            for hx in hex_slices:
                if len(hx) < 16:
                    continue
                try:
                    raw_bytes = bytes.fromhex(hx)
                except ValueError:
                    continue
                decoded = _is_printable_utf8(raw_bytes)
                if decoded is not None:
                    _add(decoded, True, False)
                    _add(decoded[::-1], True, True)

    return views


def _compute_span_features(span: TaintSpan) -> _SpanFeatures:
    norm_text = normalize_nfkc_strip_cf(span.text)
    secrets: list[str] = []
    seen_sec: set[str] = set()

    def _push_sec(val: str) -> None:
        cleaned = val.strip().strip("\"'")
        if len(cleaned) >= 8 and cleaned not in seen_sec:
            seen_sec.add(cleaned)
            secrets.append(cleaned)

    for match in _KNOWN_SECRET_RE.finditer(norm_text):
        _push_sec(match.group(0))
    for match in _URI_PASSWORD_RE.finditer(norm_text):
        _push_sec(match.group(1))
    for match in _KV_ASSIGN_RE.finditer(norm_text):
        key_lower = match.group(1).lower()
        val = match.group(2)
        if any(word in key_lower for word in _SENSITIVE_KEY_WORDS):
            val_l = val.lower()
            if val_l not in {
                "true",
                "false",
                "null",
                "none",
                "production",
                "development",
            } and not val_l.startswith(
                ("replace_with_", "example_only", "your_", "changeme", "<your")
            ):
                _push_sec(val)
    if "PRIVATE KEY-----" in norm_text:
        for line in norm_text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("-----") and len(stripped) >= 24:
                _push_sec(stripped)
    entropy_threshold = 3.55 if (CapabilityFlag.HIGH_SECRET in span.flags) else 4.15
    for match in _HIGH_ENTROPY_TOKEN_RE.finditer(norm_text):
        tok = match.group(0)
        if _has_mixed_classes(tok) and _shannon_entropy(tok) >= entropy_threshold:
            _push_sec(tok)

    nontrivial_lines: list[str] = []
    nontrivial_lines_cf: list[str] = []
    seen_lines: set[str] = set()
    for line in norm_text.splitlines():
        stripped = line.strip()
        if len(stripped) >= 15 and stripped not in seen_lines:
            seen_lines.add(stripped)
            nontrivial_lines.append(stripped)
            nontrivial_lines_cf.append(stripped.casefold())

    norm_span = " ".join(norm_text.split())
    span_windows: list[str] = []
    span_windows_cf: list[str] = []
    if len(norm_span) >= 48:
        step = max(12, len(norm_span) // 32)
        for start in range(0, len(norm_span) - 47, step):
            win = norm_span[start : start + 48]
            span_windows.append(win)
            span_windows_cf.append(win.casefold())

    span_emails = frozenset(m.casefold() for m in _EMAIL_RE.findall(norm_text))

    structured_row_tokens: list[tuple[str, ...]] = []
    for line in norm_text.splitlines():
        if "|" in line or "\t" in line or line.count(",") >= 2 or line.count("=") >= 2:
            cells = tuple(
                cell.strip().casefold()
                for cell in re.split(r"[|\t,;]+", line)
                if len(cell.strip()) >= 5
            )
            if len(cells) >= 2:
                structured_row_tokens.append(cells)

    structured_destinations = _extract_structured_destinations(norm_text)

    return _SpanFeatures(
        norm_text=norm_text,
        secret_candidates=tuple(secrets),
        nontrivial_lines=tuple(nontrivial_lines),
        nontrivial_lines_cf=tuple(nontrivial_lines_cf),
        span_windows=tuple(span_windows),
        span_windows_cf=tuple(span_windows_cf),
        span_emails=span_emails,
        structured_row_tokens=tuple(structured_row_tokens),
        structured_destinations=structured_destinations,
    )


def _get_span_features(span: TaintSpan) -> _SpanFeatures:
    text_hash = hashlib.sha256(span.text.encode("utf-8", errors="replace")).hexdigest()
    cache_key = (span.span_id, int(span.flags), text_hash)
    cached = _SPAN_FEATURE_CACHE.get(cache_key)
    if cached is not None:
        _SPAN_FEATURE_CACHE.move_to_end(cache_key)
        return cached
    feat = _compute_span_features(span)
    _SPAN_FEATURE_CACHE[cache_key] = feat
    while len(_SPAN_FEATURE_CACHE) > _SPAN_FEATURE_CACHE_MAX:
        _SPAN_FEATURE_CACHE.popitem(last=False)
    return feat


def _extract_secret_candidates(span: TaintSpan) -> list[str]:
    """Extract secret values, tokens, and high-entropy credentials from a private span."""
    return list(_get_span_features(span).secret_candidates)


def _secret_matches_views(
    secret: str,
    raw_strings: list[str],
    views: list[tuple[str, bool]],
) -> tuple[bool, bool]:
    """Return `(matched, is_obfuscated)` if `secret` appears raw or encoded."""
    for view_text, is_obf in views:
        if secret in view_text:
            return True, is_obf

    sec_bytes = secret.encode("utf-8")
    b64_no_pad = base64.b64encode(sec_bytes).decode("ascii").rstrip("=")
    b64_url_no_pad = base64.urlsafe_b64encode(sec_bytes).decode("ascii").rstrip("=")
    b32_no_pad = base64.b32encode(sec_bytes).decode("ascii").rstrip("=").lower()
    hex_lower = sec_bytes.hex()
    pct_lower = "".join(f"%{b:02x}" for b in sec_bytes)
    url_quoted = urllib.parse.quote(secret, safe="").lower()
    rev = secret[::-1]

    for raw in raw_strings:
        norm_raw = normalize_nfkc_strip_cf(raw)
        for src in (raw, norm_raw) if norm_raw != raw else (raw,):
            src_lower = src.lower()
            if (
                b64_no_pad in src
                or b64_url_no_pad in src
                or b32_no_pad in src_lower
                or hex_lower in src_lower
                or pct_lower in src_lower
                or (url_quoted != secret.lower() and url_quoted in src_lower)
                or rev in src
            ):
                return True, True

    return False, False


def _detect_code_or_data_copy(
    span: TaintSpan,
    views: list[tuple[str, bool]],
    *,
    is_routing_key: bool = False,
    feat: _SpanFeatures | None = None,
) -> tuple[float, bool]:
    """Return `(base_score, is_obfuscated)` for multi-line code, block, or DB/PII copies."""
    if not span.text.strip():
        return 0.0, False

    if feat is None:
        feat = _get_span_features(span)

    is_high_secret = CapabilityFlag.HIGH_SECRET in span.flags
    best_score = 0.0
    best_obf = False

    for view_text, is_obf in views:
        view_norm = normalize_nfkc_strip_cf(view_text)
        view_cf = view_norm.casefold()

        matched_lines = [
            ln
            for ln, ln_cf in zip(feat.nontrivial_lines, feat.nontrivial_lines_cf)
            if ln_cf in view_cf
        ]
        if len(matched_lines) >= 2:
            score = 0.92 if is_obf else 0.90
            if score > best_score:
                best_score, best_obf = score, is_obf
            continue

        norm_view_cf = " ".join(view_cf.split())
        if feat.span_windows_cf and any(win in norm_view_cf for win in feat.span_windows_cf):
            score = 0.90 if is_obf else 0.88
            if score > best_score:
                best_score, best_obf = score, is_obf
            continue

        if is_high_secret and len(matched_lines) == 1 and len(matched_lines[0]) >= 15:
            score = 0.92
            if score > best_score:
                best_score, best_obf = score, is_obf
            continue

        if len(matched_lines) == 1 and len(matched_lines[0]) >= 26:
            if _CODE_LINE_HINT_RE.search(matched_lines[0]):
                score = 0.88 if is_obf else 0.85
                if score > best_score:
                    best_score, best_obf = score, is_obf
                continue

        if feat.span_emails:
            matched_emails = [email for email in feat.span_emails if email in view_cf]
            if matched_emails:
                if (
                    is_routing_key
                    and len(matched_emails) == 1
                    and not is_high_secret
                    and not is_obf
                ):
                    score = 0.55
                else:
                    score = 0.88 if is_obf else 0.86
                if score > best_score:
                    best_score, best_obf = score, is_obf
                if score >= 0.85:
                    continue

        for row_cells in feat.structured_row_tokens:
            hits = sum(1 for cell in row_cells if cell in view_cf)
            if hits >= 2:
                score = 0.88 if is_obf else 0.85
                if score > best_score:
                    best_score, best_obf = score, is_obf
                break

    return best_score, best_obf


def _intent_explicitly_authorizes_copy(intent: str, span: TaintSpan) -> bool:
    """True only when an explicit disclosure verb and target source appear in the same clause."""
    if CapabilityFlag.HIGH_SECRET in span.flags:
        return False

    target_identifiers: list[str] = []
    label = span.source_label.strip()
    if label:
        label_lower = label.lower()
        basename_lower = os.path.basename(label_lower.replace("\\", "/"))
        if len(label_lower) >= 5:
            target_identifiers.append(label_lower)
        if len(basename_lower) >= 5 and basename_lower not in target_identifiers:
            target_identifiers.append(basename_lower)

    if not target_identifiers:
        return False

    for clause in _CLAUSE_SPLIT_RE.split(intent):
        if not clause or not _DISCLOSURE_VERB_RE.search(clause):
            continue
        clause_lower = clause.lower()
        if any(target in clause_lower for target in target_identifiers):
            return True

    return False


def _detect_arg_provenance_hijack(
    clean_intent: str,
    private_spans: list[TaintSpan],
    untrusted_spans: list[TaintSpan],
    arg_forward_views_map: dict[str, list[tuple[str, bool]]],
) -> tuple[float, str | None]:
    """Return `(0.68, arg_key)` when a sink arg destination came strictly from an untrusted span."""
    untrusted_emails: set[str] = set()
    untrusted_ibans: set[str] = set()
    untrusted_urls: set[str] = set()
    untrusted_etlds: set[str] = set()

    for span in untrusted_spans:
        dest = _get_span_features(span).structured_destinations
        untrusted_emails.update(dest.emails)
        untrusted_ibans.update(dest.ibans)
        untrusted_urls.update(dest.urls)
        untrusted_etlds.update(dest.etld1s)

    trusted_emails: set[str] = set()
    trusted_ibans: set[str] = set()
    trusted_urls: set[str] = set()
    trusted_etlds: set[str] = set()

    for span in private_spans:
        dest = _get_span_features(span).structured_destinations
        is_untrust = span.dual_trust or bool(span.flags & CapabilityFlag.READS_UNTRUSTED)
        if is_untrust:
            untrusted_emails.update(dest.emails)
            untrusted_ibans.update(dest.ibans)
            untrusted_urls.update(dest.urls)
            untrusted_etlds.update(dest.etld1s)
        else:
            trusted_emails.update(dest.emails)
            trusted_ibans.update(dest.ibans)
            trusted_urls.update(dest.urls)
            trusted_etlds.update(dest.etld1s)

    if not (untrusted_emails or untrusted_ibans or untrusted_urls or untrusted_etlds):
        return 0.0, None

    norm_intent = normalize_nfkc_strip_cf(clean_intent)
    intent_cf = norm_intent.casefold()
    intent_dest = _extract_structured_destinations(norm_intent)
    trusted_emails.update(intent_dest.emails)
    trusted_ibans.update(intent_dest.ibans)
    trusted_urls.update(intent_dest.urls)
    trusted_etlds.update(intent_dest.etld1s)
    for dom_match in _DOMAIN_EXTRACT_RE.finditer(norm_intent):
        et = extract_etld_plus_one(dom_match.group(0))
        if et:
            trusted_etlds.add(et)

    for arg_key, forward_views in arg_forward_views_map.items():
        for view_text, _is_obf in forward_views:
            arg_dest = _extract_structured_destinations(view_text)
            if arg_dest.is_empty:
                continue
            for em in arg_dest.emails:
                if em in untrusted_emails and em not in trusted_emails and em not in intent_cf:
                    return 0.68, arg_key
            for iban in arg_dest.ibans:
                if iban in untrusted_ibans and iban not in trusted_ibans and iban.casefold() not in intent_cf:
                    return 0.68, arg_key
            for url in arg_dest.urls:
                if url in untrusted_urls and url not in trusted_urls and url.casefold() not in intent_cf:
                    return 0.68, arg_key
            for et in arg_dest.etld1s:
                if et in untrusted_etlds and et not in trusted_etlds and et not in intent_cf:
                    return 0.68, arg_key

    return 0.0, None


class Tier0DeterministicPredictor:
    """Zero-ML deterministic fallback predictor covering T1-T7 exfiltration and sabotage."""

    def __init__(self, *, hard_secrets_only: bool = False) -> None:
        self.hard_secrets_only = hard_secrets_only

    def evaluate_with_reasons(
        self,
        intent: str | None,
        private_spans: list[TaintSpan],
        untrusted_spans: list[TaintSpan],
        call: CandidateCall,
        *,
        hard_secrets_only: bool | None = None,
    ) -> tuple[DecisionScores, tuple[str, ...]]:
        effective_hard_only = (
            self.hard_secrets_only if hard_secrets_only is None else hard_secrets_only
        )
        missing_intent = _is_missing_intent(intent)
        clean_intent = "" if missing_intent or intent is None else intent.strip()

        arg_raw_map: dict[str, list[str]] = {}
        arg_views_map: dict[str, list[tuple[str, bool]]] = {}
        arg_forward_views_map: dict[str, list[tuple[str, bool]]] = {}

        for arg_key, arg_val in call.arguments.items():
            raw_list: list[str] = []
            _collect_strings(arg_val, raw_list)
            detailed = _expand_arg_views_detailed(raw_list)
            arg_raw_map[arg_key] = raw_list
            arg_views_map[arg_key] = [(t, obf) for t, obf, _rev in detailed]
            arg_forward_views_map[arg_key] = [(t, obf) for t, obf, rev in detailed if not rev]

        best_p_enc = 0.0
        best_span_idx: int | None = None
        best_span_id: str | None = None
        best_enc_arg_key: str | None = None
        reasons: list[str] = []

        for span_idx, span in enumerate(private_spans):
            feat = _get_span_features(span)
            secret_candidates = feat.secret_candidates
            is_high_secret_span = CapabilityFlag.HIGH_SECRET in span.flags

            for arg_key, views in arg_views_map.items():
                if not views:
                    continue
                raw_strings = arg_raw_map[arg_key]

                secret_matched = False
                secret_obfuscated = False
                for sec in secret_candidates:
                    matched, is_obf = _secret_matches_views(sec, raw_strings, views)
                    if matched:
                        secret_matched = True
                        secret_obfuscated = is_obf
                        break

                if secret_matched:
                    score = 0.97 if secret_obfuscated else 0.95
                    if score > best_p_enc:
                        best_p_enc = score
                        best_span_idx = span_idx
                        best_span_id = span.span_id
                        best_enc_arg_key = arg_key
                    continue

                is_routing_key = arg_key.strip().lower() in _ROUTING_ARG_KEYS
                copy_score, copy_obf = _detect_code_or_data_copy(
                    span,
                    views,
                    is_routing_key=is_routing_key,
                    feat=feat,
                )
                if copy_score > 0.0:
                    if (
                        effective_hard_only
                        and not missing_intent
                        and not (is_high_secret_span or copy_obf)
                    ):
                        continue
                    if is_high_secret_span or copy_obf:
                        score = max(copy_score, 0.92)
                    elif is_routing_key and copy_score <= 0.55:
                        if not missing_intent and _intent_explicitly_authorizes_copy(
                            clean_intent, span
                        ):
                            score = 0.08
                        else:
                            score = 0.55
                    elif missing_intent:
                        score = max(copy_score, CRITICAL_THRESHOLD + 0.05)
                    elif _intent_explicitly_authorizes_copy(clean_intent, span):
                        score = 0.08
                    else:
                        score = copy_score

                    if score > best_p_enc:
                        best_p_enc = score
                        if score >= 0.20:
                            best_span_idx = span_idx
                            best_span_id = span.span_id
                            best_enc_arg_key = arg_key

        best_p_hijack = 0.0
        hijack_arg_key: str | None = None

        # C4: Unconditional self-protection regardless of taint or intent
        for arg_key, forward_views in arg_forward_views_map.items():
            matched_self = False
            for view_text, _is_obf in forward_views:
                for variant in expand_self_tamper_variants(view_text):
                    if _SELF_PROTECTION_RE.search(variant) is not None:
                        matched_self = True
                        break
                if matched_self:
                    if 0.98 > best_p_hijack:
                        best_p_hijack = 0.98
                        hijack_arg_key = arg_key
                    break

        has_untrusted_taint = bool(untrusted_spans) or any(
            s.dual_trust or bool(s.flags & CapabilityFlag.READS_UNTRUSTED)
            for s in private_spans
        )

        if has_untrusted_taint:
            intent_lower = " ".join(normalize_nfkc_strip_cf(clean_intent).lower().split())
            has_mutation_verb = bool(_MUTATION_INTENT_VERB_RE.search(clean_intent))
            for arg_key, forward_views in arg_forward_views_map.items():
                for view_text, is_obf in forward_views:
                    if effective_hard_only and not is_obf:
                        continue
                    for pattern, _label in _SABOTAGE_PATTERNS:
                        for match in pattern.finditer(view_text):
                            matched_fragment = " ".join(
                                match.group(0).strip().lstrip("/\\").rstrip(";").lower().split()
                            )
                            if (
                                not is_obf
                                and not missing_intent
                                and has_mutation_verb
                                and matched_fragment
                                and matched_fragment in intent_lower
                            ):
                                continue
                            if 0.92 > best_p_hijack:
                                best_p_hijack = 0.92
                                hijack_arg_key = arg_key

            if not effective_hard_only:
                prov_score, prov_arg_key = _detect_arg_provenance_hijack(
                    clean_intent,
                    private_spans,
                    untrusted_spans,
                    arg_forward_views_map,
                )
                if prov_score > 0.0:
                    reasons.append("arg_provenance_untrusted")
                    if prov_score > best_p_hijack:
                        best_p_hijack = prov_score
                        hijack_arg_key = prov_arg_key

        attributed_span_idx: int | None = None
        attributed_span_id: str | None = None
        attributed_arg_key: str | None = None
        if best_p_enc >= 0.20 and best_p_enc >= best_p_hijack:
            attributed_span_idx = best_span_idx
            attributed_span_id = best_span_id
            attributed_arg_key = best_enc_arg_key
        elif best_p_hijack >= 0.20:
            attributed_span_idx = None
            attributed_span_id = None
            attributed_arg_key = hijack_arg_key

        scores = DecisionScores(
            p_enc=round(best_p_enc, 4),
            p_sem=0.0,
            p_hijack=round(best_p_hijack, 4),
            attributed_span_idx=attributed_span_idx,
            attributed_span_id=attributed_span_id,
            attributed_arg_key=attributed_arg_key,
        )
        return scores, tuple(reasons)

    def evaluate(
        self,
        intent: str | None,
        private_spans: list[TaintSpan],
        untrusted_spans: list[TaintSpan],
        call: CandidateCall,
        *,
        hard_secrets_only: bool | None = None,
    ) -> DecisionScores:
        scores, _reasons = self.evaluate_with_reasons(
            intent,
            private_spans,
            untrusted_spans,
            call,
            hard_secrets_only=hard_secrets_only,
        )
        return scores
