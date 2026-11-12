"""Normative DiodeModelBundle v1 Jev window packer, safe content encoder, and 4D mask reference."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from diodemcp.bundle import BundleManifest
from diodemcp.constants import (
    DEFAULT_INFERENCE_BUCKET,
    MAX_ARGS,
    ONNX_INPUT_NAMES,
    REQUIRED_SPECIAL_TOKENS,
    TOKEN_ROLE_PAD,
    TOKEN_ROLE_QUERY,
    TOKEN_ROLE_STATE,
)
from diodemcp.predictor import _expand_arg_views_detailed
from diodemcp.registry import canonicalize_call_head
from diodemcp.types import CandidateCall, CapabilityFlag, TaintSpan

if TYPE_CHECKING:
    from diodemcp.ledger import LedgerSnapshot

_BASE_BRACKET_SPECIAL_RE = re.compile(r"\[(CLS|SEP|PAD)\]", flags=re.IGNORECASE)
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")
_STRUCTURAL_ID_CACHE: OrderedDict[int, tuple[object, frozenset[int]]] = OrderedDict()
_STRUCTURAL_ID_CACHE_MAX = 16

_QUERY_TOKEN_ORDER: tuple[str, ...] = (
    "<|q_enc|>",
    "<|q_sem|>",
    "<|q_hijack|>",
    "<|q_src|>",
    "<|q_arg|>",
)


def sanitize_content_delimiters(text: str) -> str:
    """Neutralize literal Jev `<|...|>` and `[CLS]/[SEP]/[PAD]` tokens inside raw content spans (`C2`)."""
    if not text:
        return ""
    escaped = text
    while True:
        updated = escaped.replace("<|", r"<\|")
        updated = _BASE_BRACKET_SPECIAL_RE.sub(r"[\\\1]", updated)
        if updated == escaped:
            return updated
        escaped = updated


# Alias matching Track A's internal helper name for cross-track parity
sanitize_untrusted_delimiters = sanitize_content_delimiters


def _resolve_structural_ids(
    tokenizer: Any,
    special_token_ids: Mapping[str, int] | Collection[int] | None = None,
) -> frozenset[int]:
    found: set[int] = set()
    if special_token_ids is not None:
        if isinstance(special_token_ids, Mapping):
            for val in special_token_ids.values():
                found.add(int(val))
        else:
            for val in special_token_ids:
                found.add(int(val))

    cache_key = id(tokenizer)
    cached = _STRUCTURAL_ID_CACHE.get(cache_key)
    if cached is not None and cached[0] is tokenizer:
        _STRUCTURAL_ID_CACHE.move_to_end(cache_key)
        tok_ids = cached[1]
    else:
        resolved_tok_ids: set[int] = set()
        if hasattr(tokenizer, "token_to_id"):
            for tok_str in REQUIRED_SPECIAL_TOKENS:
                tid = tokenizer.token_to_id(tok_str)
                if tid is not None:
                    resolved_tok_ids.add(int(tid))
        elif getattr(tokenizer, "is_fast", False) and hasattr(tokenizer, "backend_tokenizer"):
            backend = tokenizer.backend_tokenizer
            for tok_str in REQUIRED_SPECIAL_TOKENS:
                tid = backend.token_to_id(tok_str)
                if tid is not None:
                    resolved_tok_ids.add(int(tid))
        elif hasattr(tokenizer, "convert_tokens_to_ids"):
            unk_id = getattr(tokenizer, "unk_token_id", None)
            for tok_str in REQUIRED_SPECIAL_TOKENS:
                tid = tokenizer.convert_tokens_to_ids(tok_str)
                if isinstance(tid, int) and tid >= 0 and (unk_id is None or tid != unk_id):
                    resolved_tok_ids.add(tid)

        tok_ids = frozenset(resolved_tok_ids)
        _STRUCTURAL_ID_CACHE[cache_key] = (tokenizer, tok_ids)
        while len(_STRUCTURAL_ID_CACHE) > _STRUCTURAL_ID_CACHE_MAX:
            _STRUCTURAL_ID_CACHE.popitem(last=False)

    if not found:
        return tok_ids
    return frozenset(found | tok_ids)


def encode_content_safe(
    tokenizer: Any,
    text: str,
    special_token_ids: Mapping[str, int] | Collection[int] | None = None,
) -> list[int]:
    """Encode raw intent/private/untrusted/arg text with zero structural token injection (`C2`)."""
    if not text:
        return []
    sanitized = sanitize_content_delimiters(text)
    structural_ids = _resolve_structural_ids(tokenizer, special_token_ids)

    encoded = tokenizer.encode(sanitized, add_special_tokens=False)
    raw_ids = encoded.ids if hasattr(encoded, "ids") else encoded
    ids = [int(x) for x in raw_ids]

    if not structural_ids.isdisjoint(ids):
        raise RuntimeError("Structural special token leaked into content span")
    return ids


def flatten_call_arguments(
    arguments: Mapping[str, Any],
    max_args: int | None = MAX_ARGS,
) -> dict[str, str]:
    """Flatten nested tool-call arguments using deterministic sorted dotted/indexed keys (`C3`)."""
    if max_args is not None and max_args < 1:
        raise ValueError("max_args must be >= 1")
    out: OrderedDict[str, str] = OrderedDict()

    def _at_limit() -> bool:
        return max_args is not None and len(out) >= max_args

    def _visit(prefix: str, value: Any) -> None:
        if _at_limit():
            return
        if isinstance(value, Mapping):
            if not value:
                out[prefix or "args"] = "{}"
                return
            for sub_k, sub_v in sorted(value.items(), key=lambda kv: str(kv[0])):
                key_str = str(sub_k)
                child_prefix = f"{prefix}.{key_str}" if prefix else key_str
                _visit(child_prefix, sub_v)
                if _at_limit():
                    return
        elif isinstance(value, (list, tuple)):
            if not value:
                out[prefix or "args"] = "[]"
                return
            for idx, item in enumerate(value):
                child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                _visit(child_prefix, item)
                if _at_limit():
                    return
        elif isinstance(value, str):
            out[prefix or "value"] = value
        elif value is None or isinstance(value, (bool, int, float)):
            out[prefix or "value"] = json.dumps(value)
        else:
            out[prefix or "value"] = str(value)

    if arguments:
        _visit("", arguments)
    return dict(out)


def build_reference_4d_attn_mask(token_roles: np.ndarray) -> np.ndarray:
    """Build additive `float32[B,1,L,L]` mask from `token_roles` (`0=PAD, 1=STATE, 2=QUERY`).

    Rules (`R5`, `I8`):
    - `STATE` (`1`) attends to all `STATE` (`1`) tokens (`0.0`), blocked from `QUERY`/`PAD` (`-10000.0`).
    - `QUERY` (`2`) attends to all `STATE` (`1`) tokens + its own diagonal `i == j` (`0.0`),
      isolated from all other `QUERY` tokens (`i != j -> -10000.0`) and `PAD` (`-10000.0`).
    - `PAD` (`0`) has `0.0` on self-diagonal `i == j` and `-10000.0` elsewhere so `Softmax` on
      `PAD` rows never produces `NaN` in FP16/INT8.
    """
    roles = np.asarray(token_roles, dtype=np.int64)
    if roles.ndim == 1:
        roles = roles[None, :]
    if roles.ndim != 2:
        raise ValueError(f"token_roles must have rank 2 [B, L], got shape {roles.shape}")
    if np.any((roles < TOKEN_ROLE_PAD) | (roles > TOKEN_ROLE_QUERY)):
        raise ValueError("token_roles contains invalid role values outside {0, 1, 2}")

    seq_len = roles.shape[1]
    is_state_i = (roles == TOKEN_ROLE_STATE)[:, :, None]
    is_query_i = (roles == TOKEN_ROLE_QUERY)[:, :, None]
    is_state_j = (roles == TOKEN_ROLE_STATE)[:, None, :]
    diag = np.eye(seq_len, dtype=bool)[None, :, :]

    allowed = ((is_state_i | is_query_i) & is_state_j) | diag
    return np.where(allowed[:, None, :, :], np.float32(0.0), np.float32(-10000.0))


def format_span_content(span: TaintSpan) -> str:
    """Format a span's source label and text for normative `<|pchunk|>` / `<|uchunk|>` encoding."""
    label = span.source_label.strip()
    if label and span.text:
        return f"{label}\n{span.text}"
    return span.text or label


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


_SPAN_OVERLAP_CACHE: OrderedDict[tuple[str, str], tuple[frozenset[str], frozenset[str]]] = OrderedDict()
_SPAN_OVERLAP_CACHE_MAX = 512


def _get_span_ngrams_and_words(span: TaintSpan) -> tuple[frozenset[str], frozenset[str]]:
    sample = f"{span.source_label}\n{span.text[:8192]}".lower()
    sample_hash = hashlib.sha256(sample.encode("utf-8", errors="replace")).hexdigest()
    key = (span.span_id, sample_hash)
    cached = _SPAN_OVERLAP_CACHE.get(key)
    if cached is not None:
        _SPAN_OVERLAP_CACHE.move_to_end(key)
        return cached
    span_ngrams, span_words = _extract_ngrams_and_words([sample], max_chars=8192)
    val = (frozenset(span_ngrams), frozenset(span_words))
    _SPAN_OVERLAP_CACHE[key] = val
    while len(_SPAN_OVERLAP_CACHE) > _SPAN_OVERLAP_CACHE_MAX:
        _SPAN_OVERLAP_CACHE.popitem(last=False)
    return val


def _score_span_overlap(
    span: TaintSpan,
    call_ngrams: set[str],
    call_words: set[str],
) -> float:
    if not call_ngrams and not call_words:
        return 0.0
    span_ngrams, span_words = _get_span_ngrams_and_words(span)
    word_hits = len(span_words & call_words)
    ngram_hits = len(span_ngrams & call_ngrams)
    denom = max(1, min(len(call_ngrams), 256))
    return (word_hits * 4.0) + (ngram_hits / denom) * 10.0 + (ngram_hits * 0.05)


@dataclass(frozen=True)
class PackedJevInputs:
    """Packed numpy tensors and candidate attribution mappings for `OrtJevPredictor`."""

    input_ids: np.ndarray
    seg_ids: np.ndarray
    token_roles: np.ndarray
    query_positions: np.ndarray
    pchunk_positions: np.ndarray
    pchunk_valid_mask: np.ndarray
    arg_positions: np.ndarray
    arg_valid_mask: np.ndarray
    bucket_length: int
    unpadded_length: int
    none_position: int
    was_truncated: bool
    selected_span_indices: tuple[int, ...]
    selected_span_ids: tuple[str, ...]
    active_pchunk_count: int
    active_arg_keys: tuple[str, ...]

    def as_ort_feed(self) -> dict[str, np.ndarray]:
        """Return exact 8 `ONNX_INPUT_NAMES` tensors for `onnxruntime.InferenceSession.run`."""
        feed = {
            "input_ids": self.input_ids,
            "seg_ids": self.seg_ids,
            "token_roles": self.token_roles,
            "query_positions": self.query_positions,
            "pchunk_positions": self.pchunk_positions,
            "pchunk_valid_mask": self.pchunk_valid_mask,
            "arg_positions": self.arg_positions,
            "arg_valid_mask": self.arg_valid_mask,
        }
        if tuple(feed.keys()) != ONNX_INPUT_NAMES:
            raise RuntimeError("PackedJevInputs feed keys diverged from ONNX_INPUT_NAMES")
        return feed


class JevWindowPacker:
    """Normative `DiodeModelBundle v1` sequence packer with `512`-token pre-selection (`R5`, `R6`)."""

    def __init__(self, tokenizer: Any, manifest: BundleManifest) -> None:
        self.tokenizer = tokenizer
        self.manifest = manifest
        self._sp = dict(manifest.special_token_ids)
        self._structural_ids = frozenset(int(v) for v in self._sp.values())

    def _encode_safe(self, text: str) -> list[int]:
        return encode_content_safe(self.tokenizer, text, self._structural_ids)

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
        ranked_entries: list[tuple[int, TaintSpan, bool, float]],
        max_slots: int,
        token_budget: int,
    ) -> tuple[list[tuple[int, TaintSpan, list[int]]], bool]:
        """Select and tokenize spans within `max_slots` and `token_budget`, preserving `orig_idx` order."""
        if not ranked_entries:
            return [], False
        if token_budget < 2:
            return [], True

        truncated = len(ranked_entries) > max_slots
        candidates = ranked_entries[:max_slots]

        encoded_candidates: list[tuple[int, TaintSpan, bool, list[int]]] = []
        total_needed = 0
        for orig_idx, span, is_pinned, _score in candidates:
            c_ids = self._encode_safe(format_span_content(span))
            encoded_candidates.append((orig_idx, span, is_pinned, c_ids))
            total_needed += 1 + len(c_ids)

        if total_needed <= token_budget:
            chosen = [(orig_idx, span, c_ids) for orig_idx, span, _pin, c_ids in encoded_candidates]
            chosen.sort(key=lambda item: item[0])
            return chosen, truncated

        # Budget exceeded: give pinned HIGH_SECRET spans their full pinned share while capping unpinned spans
        truncated = True
        pinned_count = sum(1 for _, _, pin, _ in encoded_candidates if pin)
        pinned_per_span_cap = max(1, (token_budget // max(1, pinned_count)) - 1)
        unpinned_per_span_cap: int | None = None

        selected: list[tuple[int, TaintSpan, list[int]]] = []
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

    def pack(
        self,
        snapshot: LedgerSnapshot,
        call: CandidateCall,
        *,
        max_bucket: int = DEFAULT_INFERENCE_BUCKET,
        pad_to_max_bucket: bool = False,
    ) -> PackedJevInputs:
        """Pack `(snapshot, call)` into the normative `DiodeModelBundle v1` tensor layout."""
        sp = self._sp
        max_pchunks = self.manifest.max_pchunks
        max_args = self.manifest.max_args
        effective_cap = self._select_effective_cap(max_bucket)
        was_truncated = False

        # 15 mandatory structural tokens:
        # [CLS], <|intent|>|<|missing_intent|>, [SEP], <|priv|>, [SEP], <|untrust|>, [SEP],
        # <|call|>, [SEP], <|none|>, and 5 <|q_*|> tokens.
        fixed_overhead = 15
        avail = effective_cap - fixed_overhead
        if avail < 4:
            raise ValueError(f"Bucket {effective_cap} is too small for mandatory Jev skeleton")

        clean_intent = snapshot.intent.strip() if snapshot.intent is not None else ""
        has_intent = bool(clean_intent and clean_intent != "<|missing_intent|>")
        intent_ids: list[int] = []
        if has_intent:
            raw_intent_ids = self._encode_safe(clean_intent)
            intent_cap = min(max(1, avail - 1), max(8, avail // 5))
            intent_ids, cut_intent = _middle_elide_ids(raw_intent_ids, intent_cap)
            was_truncated = was_truncated or cut_intent
            avail -= len(intent_ids)

        canon_srv, canon_tool = canonicalize_call_head(call.server, call.tool)
        raw_call_ids = self._encode_safe(f"{canon_srv}.{canon_tool}")
        call_cap = min(avail, max(4, avail // 8))
        call_head_ids, cut_call = _middle_elide_ids(raw_call_ids, call_cap)
        was_truncated = was_truncated or cut_call
        avail -= len(call_head_ids)

        # Flatten arguments (deterministic sorted C3 keys); preserve overflow args in last slot if > max_args
        all_flat_args = flatten_call_arguments(call.arguments, max_args=None)
        flat_items = list(all_flat_args.items())
        if len(flat_items) > max_args:
            was_truncated = True
            head_items = flat_items[: max_args - 1]
            tail_key = flat_items[max_args - 1][0]
            merged_tail_val = "\n".join(
                f"{k}={v}" if idx > 0 else v
                for idx, (k, v) in enumerate(flat_items[max_args - 1 :])
            )
            flat_items = [*head_items, (tail_key, merged_tail_val)]

        has_spans = bool(snapshot.private_spans or snapshot.untrusted_spans)
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

        # Pre-rank private & untrusted spans by HIGH_SECRET pinning + 4-gram/word overlap with call args
        raw_arg_values = list(all_flat_args.values())
        expanded_arg_texts = [t for t, _obf, _rev in _expand_arg_views_detailed(raw_arg_values[:32])]
        call_ngrams, call_words = _extract_ngrams_and_words(expanded_arg_texts or raw_arg_values)

        ranked_priv: list[tuple[int, TaintSpan, bool, float]] = []
        for idx, span in enumerate(snapshot.private_spans):
            is_pinned = bool(span.flags & CapabilityFlag.HIGH_SECRET)
            score = _score_span_overlap(span, call_ngrams, call_words)
            ranked_priv.append((idx, span, is_pinned, score))
        ranked_priv.sort(key=lambda item: (0 if item[2] else 1, -item[3], item[0]))

        ranked_untrust: list[tuple[int, TaintSpan, bool, float]] = []
        for idx, span in enumerate(snapshot.untrusted_spans):
            score = _score_span_overlap(span, call_ngrams, call_words)
            ranked_untrust.append((idx, span, False, score))
        ranked_untrust.sort(key=lambda item: (-item[3], -item[0]))

        selected_priv: list[tuple[int, TaintSpan, list[int]]] = []
        selected_untrust: list[tuple[int, TaintSpan, list[int]]] = []

        if ranked_priv and not ranked_untrust:
            selected_priv, cut_p = self._fit_ranked_spans(ranked_priv, max_pchunks, avail)
            was_truncated = was_truncated or cut_p
        elif ranked_untrust and not ranked_priv:
            selected_untrust, cut_u = self._fit_ranked_spans(ranked_untrust, max_pchunks, avail)
            was_truncated = was_truncated or cut_u
        elif ranked_priv and ranked_untrust:
            # First split ~70% private / ~30% untrusted, donating slack across pools
            u_share = max(2, int(avail * 0.30)) if avail >= 6 else 0
            p_share = avail - u_share

            # Probe untrusted first if small so slack flows cleanly to private
            selected_untrust, cut_u = self._fit_ranked_spans(ranked_untrust, max_pchunks, u_share)
            u_used = sum(1 + len(u_ids) for _, _, u_ids in selected_untrust)
            p_budget = avail - u_used
            selected_priv, cut_p = self._fit_ranked_spans(ranked_priv, max_pchunks, p_budget)
            p_used = sum(1 + len(p_ids) for _, _, p_ids in selected_priv)

            # If untrusted was truncated and private left unused slack, re-run untrusted with full slack
            if cut_u and (avail - p_used) > u_used:
                selected_untrust, cut_u = self._fit_ranked_spans(
                    ranked_untrust, max_pchunks, avail - p_used
                )

            was_truncated = was_truncated or cut_p or cut_u

        # Assemble canonical sequence strictly by integer ID
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
        _emit(sp["[CLS]"], 0, TOKEN_ROLE_STATE)
        if has_intent:
            _emit(sp["<|intent|>"], 0, TOKEN_ROLE_STATE)
            _emit_many(intent_ids, 0, TOKEN_ROLE_STATE)
        else:
            _emit(sp["<|missing_intent|>"], 0, TOKEN_ROLE_STATE)
        _emit(sp["[SEP]"], 0, TOKEN_ROLE_STATE)

        # 4-6. <|priv|> (<|pchunk|> chunk_ids)* [SEP]
        _emit(sp["<|priv|>"], 1, TOKEN_ROLE_STATE)
        pchunk_pos_list: list[int] = []
        selected_span_indices: list[int] = []
        selected_span_ids: list[str] = []

        for orig_idx, span, c_ids in selected_priv:
            is_dual = bool(
                span.dual_trust
                or (
                    bool(span.flags & CapabilityFlag.READS_PRIVATE)
                    and bool(span.flags & CapabilityFlag.READS_UNTRUSTED)
                )
            )
            p_seg = 2 if is_dual else 1
            p_pos = _emit(sp["<|pchunk|>"], p_seg, TOKEN_ROLE_STATE)
            _emit_many(c_ids, p_seg, TOKEN_ROLE_STATE)
            pchunk_pos_list.append(p_pos)
            selected_span_indices.append(orig_idx)
            selected_span_ids.append(span.span_id)

        _emit(sp["[SEP]"], 1, TOKEN_ROLE_STATE)

        # 7-9. <|untrust|> (<|uchunk|> uchunk_ids)* [SEP]
        _emit(sp["<|untrust|>"], 3, TOKEN_ROLE_STATE)
        for _orig_idx, _span, u_ids in selected_untrust:
            _emit(sp["<|uchunk|>"], 3, TOKEN_ROLE_STATE)
            _emit_many(u_ids, 3, TOKEN_ROLE_STATE)
        _emit(sp["[SEP]"], 3, TOKEN_ROLE_STATE)

        # 10-12. <|call|> server.tool (<|arg|> flat_key=val_str)* [SEP]
        _emit(sp["<|call|>"], 4, TOKEN_ROLE_STATE)
        _emit_many(call_head_ids, 4, TOKEN_ROLE_STATE)
        arg_pos_list: list[int] = []
        active_arg_keys: list[str] = []
        for flat_key, a_ids in packed_args:
            a_pos = _emit(sp["<|arg|>"], 4, TOKEN_ROLE_STATE)
            _emit_many(a_ids, 4, TOKEN_ROLE_STATE)
            arg_pos_list.append(a_pos)
            active_arg_keys.append(flat_key)
        _emit(sp["[SEP]"], 4, TOKEN_ROLE_STATE)

        # 13. <|none|> at none_pos
        none_pos = _emit(sp["<|none|>"], 4, TOKEN_ROLE_STATE)

        # 14. 5 Query tokens (seg=5, role=2)
        q_positions: list[int] = []
        for q_tok in _QUERY_TOKEN_ORDER:
            q_positions.append(_emit(sp[q_tok], 5, TOKEN_ROLE_QUERY))

        unpadded_len = len(ids)
        if unpadded_len > effective_cap:
            raise RuntimeError(
                f"JevWindowPacker overflow: unpadded_len={unpadded_len} > effective_cap={effective_cap}"
            )

        # 15. Pad right to target bucket using [PAD] (seg=0, role=0)
        bucket_len = self._select_target_bucket(
            unpadded_len, effective_cap, pad_to_max_bucket=pad_to_max_bucket
        )
        pad_count = bucket_len - unpadded_len
        if pad_count > 0:
            ids.extend([sp["[PAD]"]] * pad_count)
            segs.extend([0] * pad_count)
            roles.extend([TOKEN_ROLE_PAD] * pad_count)

        # Construct candidate position & validity tensors
        pchunk_positions = np.full((1, max_pchunks + 1), none_pos, dtype=np.int64)
        pchunk_valid_mask = np.zeros((1, max_pchunks + 1), dtype=np.bool_)
        for i, pos in enumerate(pchunk_pos_list):
            pchunk_positions[0, i] = pos
            pchunk_valid_mask[0, i] = True
        pchunk_positions[0, max_pchunks] = none_pos
        pchunk_valid_mask[0, max_pchunks] = True

        arg_positions = np.full((1, max_args + 1), none_pos, dtype=np.int64)
        arg_valid_mask = np.zeros((1, max_args + 1), dtype=np.bool_)
        for j, pos in enumerate(arg_pos_list):
            arg_positions[0, j] = pos
            arg_valid_mask[0, j] = True
        arg_positions[0, max_args] = none_pos
        arg_valid_mask[0, max_args] = True

        return PackedJevInputs(
            input_ids=np.asarray([ids], dtype=np.int64),
            seg_ids=np.asarray([segs], dtype=np.int64),
            token_roles=np.asarray([roles], dtype=np.int64),
            query_positions=np.asarray([q_positions], dtype=np.int64),
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
            active_pchunk_count=len(selected_span_indices),
            active_arg_keys=tuple(active_arg_keys),
        )
