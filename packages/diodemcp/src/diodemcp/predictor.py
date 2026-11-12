"""PredictorProtocol and zero-ML Tier0DeterministicPredictor fallback."""

from __future__ import annotations

import base64
import binascii
import math
import os
import re
import urllib.parse
from collections import Counter
from typing import Any, Protocol, runtime_checkable

from diodemcp.constants import CRITICAL_THRESHOLD
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
    r"""([A-Za-z0-9_.-]{2,64})\s*[:=]\s*["']?([^\s"'#;,]{8,})["']?"""
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


def _expand_arg_views_detailed(raw_strings: list[str]) -> list[tuple[str, bool, bool]]:
    """Return `(view_text, is_obfuscated, is_reversed)` triples for raw and decoded views."""
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

        for unq in (urllib.parse.unquote_plus(raw), urllib.parse.unquote(raw)):
            if unq != raw:
                pct_obf = bool(_PCT_ALNUM_RE.search(raw))
                _add(unq, pct_obf, False)
                _add(unq[::-1], True, True)
                base_inputs.append((unq, True))

    for candidate_src, _ in base_inputs:
        for match in _B64_CANDIDATE_RE.finditer(candidate_src):
            tok = match.group(0)
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


def _extract_secret_candidates(span: TaintSpan) -> list[str]:
    """Extract secret values, tokens, and high-entropy credentials from a private span."""
    secrets: list[str] = []
    seen: set[str] = set()

    def _push(val: str) -> None:
        cleaned = val.strip().strip("\"'")
        if len(cleaned) >= 8 and cleaned not in seen:
            seen.add(cleaned)
            secrets.append(cleaned)

    text = span.text
    for match in _KNOWN_SECRET_RE.finditer(text):
        _push(match.group(0))

    for match in _URI_PASSWORD_RE.finditer(text):
        _push(match.group(1))

    for match in _KV_ASSIGN_RE.finditer(text):
        key_lower = match.group(1).lower()
        val = match.group(2)
        if any(word in key_lower for word in _SENSITIVE_KEY_WORDS):
            if val.lower() not in {"true", "false", "null", "none", "production", "development"}:
                _push(val)

    if "PRIVATE KEY-----" in text:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("-----") and len(stripped) >= 24:
                _push(stripped)

    entropy_threshold = 3.55 if (CapabilityFlag.HIGH_SECRET in span.flags) else 4.15
    for match in _HIGH_ENTROPY_TOKEN_RE.finditer(text):
        tok = match.group(0)
        if _has_mixed_classes(tok) and _shannon_entropy(tok) >= entropy_threshold:
            _push(tok)

    return secrets


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
        raw_lower = raw.lower()
        if (
            b64_no_pad in raw
            or b64_url_no_pad in raw
            or b32_no_pad in raw_lower
            or hex_lower in raw_lower
            or pct_lower in raw_lower
            or (url_quoted != secret.lower() and url_quoted in raw_lower)
            or rev in raw
        ):
            return True, True

    return False, False


def _detect_code_or_data_copy(
    span: TaintSpan,
    views: list[tuple[str, bool]],
) -> tuple[float, bool]:
    """Return `(base_score, is_obfuscated)` for multi-line code, block, or DB/PII copies."""
    span_text = span.text
    if not span_text.strip():
        return 0.0, False

    nontrivial_lines: list[str] = []
    seen_lines: set[str] = set()
    for line in span_text.splitlines():
        stripped = line.strip()
        if len(stripped) >= 15 and stripped not in seen_lines:
            seen_lines.add(stripped)
            nontrivial_lines.append(stripped)

    norm_span = " ".join(span_text.split())
    span_windows: list[str] = []
    if len(norm_span) >= 48:
        step = max(12, len(norm_span) // 32)
        for start in range(0, len(norm_span) - 47, step):
            span_windows.append(norm_span[start : start + 48])

    span_emails = set(_EMAIL_RE.findall(span_text))

    structured_row_tokens: list[list[str]] = []
    for line in span_text.splitlines():
        if "|" in line or "\t" in line or line.count(",") >= 2 or line.count("=") >= 2:
            cells = [
                cell.strip()
                for cell in re.split(r"[|\t,;]+", line)
                if len(cell.strip()) >= 5
            ]
            if len(cells) >= 2:
                structured_row_tokens.append(cells)

    best_score = 0.0
    best_obf = False

    for view_text, is_obf in views:
        matched_lines = [ln for ln in nontrivial_lines if ln in view_text]
        if len(matched_lines) >= 2:
            score = 0.92 if is_obf else 0.90
            if score > best_score:
                best_score, best_obf = score, is_obf
            continue

        norm_view = " ".join(view_text.split())
        if span_windows and any(win in norm_view for win in span_windows):
            score = 0.90 if is_obf else 0.88
            if score > best_score:
                best_score, best_obf = score, is_obf
            continue

        if len(matched_lines) == 1 and len(matched_lines[0]) >= 26:
            if _CODE_LINE_HINT_RE.search(matched_lines[0]):
                score = 0.88 if is_obf else 0.85
                if score > best_score:
                    best_score, best_obf = score, is_obf
                continue

        if span_emails and any(email in view_text for email in span_emails):
            score = 0.88 if is_obf else 0.86
            if score > best_score:
                best_score, best_obf = score, is_obf
            continue

        for row_cells in structured_row_tokens:
            hits = sum(1 for cell in row_cells if cell in view_text)
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

    for email in _EMAIL_RE.findall(span.text):
        email_lower = email.lower()
        if email_lower not in target_identifiers:
            target_identifiers.append(email_lower)

    if not target_identifiers:
        return False

    for clause in _CLAUSE_SPLIT_RE.split(intent):
        if not clause or not _DISCLOSURE_VERB_RE.search(clause):
            continue
        clause_lower = clause.lower()
        if any(target in clause_lower for target in target_identifiers):
            return True

    return False


class Tier0DeterministicPredictor:
    """Zero-ML deterministic fallback predictor covering T1-T7 exfiltration and sabotage."""

    def evaluate(
        self,
        intent: str | None,
        private_spans: list[TaintSpan],
        untrusted_spans: list[TaintSpan],
        call: CandidateCall,
    ) -> DecisionScores:
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

        for span_idx, span in enumerate(private_spans):
            secret_candidates = _extract_secret_candidates(span)
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

                copy_score, copy_obf = _detect_code_or_data_copy(span, views)
                if copy_score > 0.0:
                    if is_high_secret_span or copy_obf:
                        score = max(copy_score, 0.92)
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
            for view_text, _is_obf in forward_views:
                if _SELF_PROTECTION_RE.search(view_text) is not None:
                    if 0.98 > best_p_hijack:
                        best_p_hijack = 0.98
                        hijack_arg_key = arg_key
                    break

        has_untrusted_taint = bool(untrusted_spans) or any(
            s.dual_trust or bool(s.flags & CapabilityFlag.READS_UNTRUSTED)
            for s in private_spans
        )

        if has_untrusted_taint:
            intent_lower = " ".join(clean_intent.lower().split())
            has_mutation_verb = bool(_MUTATION_INTENT_VERB_RE.search(clean_intent))
            for arg_key, forward_views in arg_forward_views_map.items():
                for view_text, is_obf in forward_views:
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

        return DecisionScores(
            p_enc=round(best_p_enc, 4),
            p_sem=0.0,
            p_hijack=round(best_p_hijack, 4),
            attributed_span_idx=attributed_span_idx,
            attributed_span_id=attributed_span_id,
            attributed_arg_key=attributed_arg_key,
        )
