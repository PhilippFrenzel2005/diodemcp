"""Surgical multi-argument call redactor and zero-leak audit preview masker (`I5`, `I13`)."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
import copy
import re
from typing import Any
import urllib.parse

from diodemcp.packer import flatten_call_arguments
from diodemcp.predictor import (
    _B32_CANDIDATE_RE,
    _B64_CANDIDATE_RE,
    _EMAIL_RE,
    _HEX_CANDIDATE_RE,
    _HIGH_ENTROPY_TOKEN_RE,
    _KV_ASSIGN_RE,
    _SENSITIVE_KEY_WORDS,
    _extract_secret_candidates,
    _has_mixed_classes,
    _is_printable_utf8,
    _shannon_entropy,
)
from diodemcp.types import CandidateCall, TaintSpan

REDACTED_CALL_PLACEHOLDER: str = "[REDACTED_BY_DIODEMCP]"
REDACTED_AUDIT_PLACEHOLDER: str = "[REDACTED]"

_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]{1,32} )?PRIVATE KEY-----[\s\S]*?(?:-----END (?:[A-Z0-9 ]{1,32} )?PRIVATE KEY-----|$)"
)
_EXTENDED_SECRET_RE = re.compile(
    r"(?:"
    r"sk_(?:live|test|synth)_[A-Za-z0-9_]{6,}"
    r"|rk_(?:live|test)_[A-Za-z0-9_]{6,}"
    r"|whsec_[A-Za-z0-9_]{10,}"
    r"|gh[pousr]_[A-Za-z0-9_]{12,}"
    r"|github_pat_[A-Za-z0-9_]{16,}"
    r"|AKIA[0-9A-Z]{12,}"
    r"|ASIA[0-9A-Z]{12,}"
    r"|AIza[0-9A-Za-z_-]{20,}"
    r"|xox[baprs]-[0-9A-Za-z-]{10,}"
    r"|glpat-[0-9A-Za-z_-]{14,}"
    r"|npm_[A-Za-z0-9]{16,}"
    r"|pypi-[A-Za-z0-9_-]{16,}"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r")"
)
_BEARER_TOKEN_RE = re.compile(r"(\bBearer\s+)([^\s\"',;&]{6,})", re.IGNORECASE)
_URI_CRED_RE = re.compile(r"(://[^:/@\s\"']+:)([^@\s\"']{4,})(@)")
_PCT_ENCODED_TOKEN_RE = re.compile(r"[^\s\"'&=?#]*%[0-9A-Fa-f]{2}[^\s\"'&=?#]*")


def _dedupe_spans(
    leaked_span: TaintSpan | None,
    private_spans: Sequence[TaintSpan],
) -> list[TaintSpan]:
    out: list[TaintSpan] = []
    seen_ids: set[tuple[str, str]] = set()
    for sp in (leaked_span, *private_spans):
        if sp is None:
            continue
        key = (sp.span_id, sp.text[:128])
        if key not in seen_ids:
            seen_ids.add(key)
            out.append(sp)
    return out


def _collect_span_needles(spans: Sequence[TaintSpan]) -> tuple[list[str], list[str], list[str]]:
    """Return `(case_sensitive_needles, case_insensitive_needles, raw_markers)` sorted longest-first."""
    exact_set: set[str] = set()
    ci_set: set[str] = set()
    raw_markers: set[str] = set()

    for span in spans:
        secrets = _extract_secret_candidates(span)
        emails = _EMAIL_RE.findall(span.text)
        for token in [*secrets, *emails]:
            if len(token) < 6:
                continue
            raw_markers.add(token)
            exact_set.add(token)
            exact_set.add(token[::-1])

            tok_bytes = token.encode("utf-8")
            b64_full = base64.b64encode(tok_bytes).decode("ascii")
            b64_strip = b64_full.rstrip("=")
            b64u_full = base64.urlsafe_b64encode(tok_bytes).decode("ascii")
            b64u_strip = b64u_full.rstrip("=")
            for b64_var in (b64_full, b64_strip, b64u_full, b64u_strip):
                if len(b64_var) >= 10:
                    exact_set.add(b64_var)

            b32_full = base64.b32encode(tok_bytes).decode("ascii")
            b32_strip = b32_full.rstrip("=")
            if len(b32_strip) >= 12:
                ci_set.add(b32_full)
                ci_set.add(b32_strip)

            hex_str = tok_bytes.hex()
            if len(hex_str) >= 16:
                ci_set.add(hex_str)

            pct_all = "".join(f"%{b:02X}" for b in tok_bytes)
            if len(pct_all) >= 18:
                ci_set.add(pct_all)

            url_q = urllib.parse.quote(token, safe="")
            if url_q != token and len(url_q) >= 8:
                ci_set.add(url_q)

            if len(token) >= 12:
                dotted = ".".join(token[i : i + 4] for i in range(0, len(token), 4))
                exact_set.add(dotted)

        for line in span.text.splitlines():
            stripped = line.strip()
            if len(stripped) >= 15:
                exact_set.add(stripped)
                raw_markers.add(stripped)

        norm_span = " ".join(span.text.split())
        if len(norm_span) >= 48:
            step = max(12, len(norm_span) // 32)
            for start in range(0, len(norm_span) - 47, step):
                win = norm_span[start : start + 48]
                exact_set.add(win)
                raw_markers.add(win)

    exact_sorted = sorted(exact_set, key=len, reverse=True)
    ci_sorted = sorted(ci_set, key=len, reverse=True)
    markers_sorted = sorted(raw_markers, key=len, reverse=True)
    return exact_sorted, ci_sorted, markers_sorted


def _decoded_text_is_sensitive(decoded: str, raw_markers: Sequence[str]) -> bool:
    if not decoded:
        return False
    for probe in (decoded, decoded[::-1]):
        if _PEM_BLOCK_RE.search(probe) or _EXTENDED_SECRET_RE.search(probe):
            return True
        if _URI_CRED_RE.search(probe) or _BEARER_TOKEN_RE.search(probe):
            return True
        for marker in raw_markers:
            if marker in probe or (len(marker) >= 12 and marker.lower() in probe.lower()):
                return True
    return False


def _scrub_encoded_tokens(
    text: str,
    raw_markers: Sequence[str],
    placeholder: str,
) -> str:
    """Replace percent-encoded, hex, Base64, or Base32 substrings whose decoded text is sensitive."""

    def _replace_pct(match: re.Match[str]) -> str:
        tok = match.group(0)
        if "REDACTED" in tok:
            return tok
        for unq in (urllib.parse.unquote(tok), urllib.parse.unquote_plus(tok)):
            if unq != tok and _decoded_text_is_sensitive(unq, raw_markers):
                return placeholder
        return tok

    out = _PCT_ENCODED_TOKEN_RE.sub(_replace_pct, text)

    def _replace_hex(match: re.Match[str]) -> str:
        run = match.group(0)
        slices = [run] if len(run) % 2 == 0 else [run[1:], run[:-1]]
        for hx in slices:
            if len(hx) < 16:
                continue
            try:
                raw_bytes = bytes.fromhex(hx)
            except ValueError:
                continue
            decoded = _is_printable_utf8(raw_bytes)
            if decoded is not None and _decoded_text_is_sensitive(decoded, raw_markers):
                return placeholder
        return run

    out = _HEX_CANDIDATE_RE.sub(_replace_hex, out)

    def _b64_token_is_sensitive(candidate: str) -> bool:
        padded = candidate + ("=" * ((4 - len(candidate) % 4) % 4))
        for decoder in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                raw_bytes = decoder(padded)
            except (binascii.Error, ValueError):
                continue
            decoded = _is_printable_utf8(raw_bytes)
            if decoded is not None and _decoded_text_is_sensitive(decoded, raw_markers):
                return True
        return False

    def _replace_b64(match: re.Match[str]) -> str:
        tok = match.group(0)
        if "REDACTED" in tok:
            return tok
        if _b64_token_is_sensitive(tok):
            return placeholder
        if "_" in tok or "-" in tok:
            for part in re.split(r"[_-]+", tok):
                if len(part) >= 12 and _b64_token_is_sensitive(part):
                    return placeholder
        return tok

    out = _B64_CANDIDATE_RE.sub(_replace_b64, out)

    def _replace_b32(match: re.Match[str]) -> str:
        tok = match.group(0)
        if "REDACTED" in tok:
            return tok
        stripped = tok.rstrip("=")
        if len(stripped) < 12:
            return tok
        padded = stripped.upper() + ("=" * ((8 - len(stripped) % 8) % 8))
        try:
            raw_bytes = base64.b32decode(padded, casefold=True)
        except (binascii.Error, ValueError):
            return tok
        decoded = _is_printable_utf8(raw_bytes)
        if decoded is not None and _decoded_text_is_sensitive(decoded, raw_markers):
            return placeholder
        return tok

    out = _B32_CANDIDATE_RE.sub(_replace_b32, out)
    return out


def _scrub_string(
    value: str,
    exact_needles: Sequence[str],
    ci_needles: Sequence[str],
    raw_markers: Sequence[str],
    placeholder: str,
    *,
    scrub_high_entropy: bool,
) -> str:
    if not value:
        return value

    out = _PEM_BLOCK_RE.sub(placeholder, value)
    out = _URI_CRED_RE.sub(rf"\1{placeholder}\3", out)
    out = _BEARER_TOKEN_RE.sub(rf"\1{placeholder}", out)
    out = _EXTENDED_SECRET_RE.sub(placeholder, out)

    for needle in exact_needles:
        if needle and needle in out and "REDACTED" not in needle:
            out = out.replace(needle, placeholder)

    for ci_needle in ci_needles:
        if ci_needle and "REDACTED" not in ci_needle and ci_needle.lower() in out.lower():
            out = re.sub(re.escape(ci_needle), placeholder, out, flags=re.IGNORECASE)

    out = _scrub_encoded_tokens(out, raw_markers, placeholder)

    def _replace_kv(match: re.Match[str]) -> str:
        full = match.group(0)
        key_name = match.group(1)
        val_part = match.group(2)
        if "REDACTED" in val_part:
            return full
        if any(w in key_name.lower() for w in _SENSITIVE_KEY_WORDS):
            if val_part.lower() not in {"true", "false", "null", "none"}:
                return full.replace(val_part, placeholder, 1)
        return full

    out = _KV_ASSIGN_RE.sub(_replace_kv, out)

    if scrub_high_entropy:
        def _replace_entropy(match: re.Match[str]) -> str:
            tok = match.group(0)
            if "REDACTED" in tok or "/" in tok:
                return tok
            if len(tok) >= 20 and _has_mixed_classes(tok) and _shannon_entropy(tok) >= 4.05:
                return placeholder
            return tok

        out = _HIGH_ENTROPY_TOKEN_RE.sub(_replace_entropy, out)

    return out


def _path_matches_attributed(leaf_path: str, attributed_arg_key: str | None) -> bool:
    if not attributed_arg_key:
        return False
    return (
        leaf_path == attributed_arg_key
        or leaf_path.startswith(f"{attributed_arg_key}.")
        or leaf_path.startswith(f"{attributed_arg_key}[")
    )


def redact_call_arguments(
    call: CandidateCall,
    attributed_arg_key: str | None,
    leaked_span: TaintSpan | None,
    *,
    private_spans: Sequence[TaintSpan] = (),
) -> dict[str, Any]:
    """Deep-copy `call.arguments` and surgically redact attributed & multi-arg leaks (`I5`)."""
    copied_args: dict[str, Any] = copy.deepcopy(call.arguments)
    if not copied_args:
        return copied_args

    all_spans = _dedupe_spans(leaked_span, private_spans)
    exact_needles, ci_needles, raw_markers = _collect_span_needles(all_spans)

    def _transform(prefix: str, node: Any) -> tuple[Any, bool]:
        if isinstance(node, dict):
            changed_any = False
            sub_before: dict[str, Any] = {}
            for k, v in list(node.items()):
                key_s = str(k)
                child_path = f"{prefix}.{key_s}" if prefix else key_s
                sub_before[key_s] = v
                new_v, child_changed = _transform(child_path, v)
                node[k] = new_v
                changed_any = changed_any or child_changed

            # If a container itself was named as attributed_arg_key and nothing inside matched a substring,
            # redact string leaves inside that attributed container fail-closed.
            if prefix and prefix == attributed_arg_key and not changed_any:
                for k, v in list(node.items()):
                    if isinstance(v, str):
                        node[k] = REDACTED_CALL_PLACEHOLDER
                        changed_any = True
                    elif key_s := str(k):
                        _ = sub_before.get(key_s)
            return node, changed_any

        if isinstance(node, list):
            changed_any = False
            for idx, item in enumerate(node):
                child_path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                new_item, child_changed = _transform(child_path, item)
                node[idx] = new_item
                changed_any = changed_any or child_changed
            if prefix and prefix == attributed_arg_key and not changed_any:
                for idx, item in enumerate(node):
                    if isinstance(item, str):
                        node[idx] = REDACTED_CALL_PLACEHOLDER
                        changed_any = True
            return node, changed_any

        if isinstance(node, str):
            is_attr = _path_matches_attributed(prefix, attributed_arg_key)
            scrubbed = _scrub_string(
                node,
                exact_needles,
                ci_needles,
                raw_markers,
                REDACTED_CALL_PLACEHOLDER,
                scrub_high_entropy=is_attr,
            )
            if scrubbed != node:
                return scrubbed, True
            if prefix == attributed_arg_key:
                return REDACTED_CALL_PLACEHOLDER, True
            return node, False

        if prefix == attributed_arg_key and attributed_arg_key is not None:
            return REDACTED_CALL_PLACEHOLDER, True
        return node, False

    _transform("", copied_args)
    return copied_args


def _truncate_preview_preserving_redaction(text: str, max_chars: int = 160) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Avoid slicing mid-"[REDACTED]" marker
    open_bracket = cut.rfind("[")
    if open_bracket != -1 and "]" not in cut[open_bracket:]:
        if text[open_bracket : open_bracket + len(REDACTED_AUDIT_PLACEHOLDER)] == REDACTED_AUDIT_PLACEHOLDER:
            cut = text[: open_bracket + len(REDACTED_AUDIT_PLACEHOLDER)]
    return f"{cut}...[truncated]"


def build_redacted_audit_preview(
    args: dict[str, Any],
    attributed_arg_key: str | None = None,
    leaked_span: TaintSpan | None = None,
    *,
    private_spans: Sequence[TaintSpan] = (),
    max_value_chars: int = 160,
) -> dict[str, str]:
    """Produce bounded `dict[str, str]` previews with zero raw or encoded secret leakage (`I13`)."""
    if not args:
        return {}

    all_spans = _dedupe_spans(leaked_span, private_spans)
    exact_needles, ci_needles, raw_markers = _collect_span_needles(all_spans)
    flat = flatten_call_arguments(args, max_args=None)

    preview: dict[str, str] = {}
    for flat_key, raw_val in flat.items():
        key_l = flat_key.rsplit(".", 1)[-1].split("[", 1)[0].lower()
        is_attr = _path_matches_attributed(flat_key, attributed_arg_key)

        scrubbed = _scrub_string(
            raw_val,
            exact_needles,
            ci_needles,
            raw_markers,
            REDACTED_AUDIT_PLACEHOLDER,
            scrub_high_entropy=True,
        )

        # Sensitive parameter names (e.g. password, api_key, secret, authorization) or attributed
        # paraphrased leaks with no narrower match mask their payload completely.
        if scrubbed == raw_val and raw_val:
            if is_attr:
                scrubbed = REDACTED_AUDIT_PLACEHOLDER
            elif any(w in key_l for w in _SENSITIVE_KEY_WORDS) and raw_val.lower() not in {
                "true",
                "false",
                "null",
                "none",
                "{}",
                "[]",
            }:
                scrubbed = REDACTED_AUDIT_PLACEHOLDER

        preview[flat_key] = _truncate_preview_preserving_redaction(scrubbed, max_value_chars)

    return preview
