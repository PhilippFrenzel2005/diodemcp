"""14-token Jev tokenizer builder and C2-hardened safe content encoder."""

from __future__ import annotations

from collections import OrderedDict
import inspect
from pathlib import Path
import re

from tokenizers import AddedToken, Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from diodemodel.contract import (
    JEV_SPECIAL_TOKENS,
    REQUIRED_BASE_SPECIAL_TOKENS,
    REQUIRED_SPECIAL_TOKENS,
)


SUPPORTED_BACKBONES: dict[str, str] = {
    "base": "answerdotai/ModernBERT-base",
    "large": "answerdotai/ModernBERT-large",
    "nanojev": "Qwen/Qwen3-0.6B",
}

_BASE_BRACKET_SPECIAL_RE = re.compile(r"\[(CLS|SEP|PAD)\]", flags=re.IGNORECASE)
_STRUCTURAL_ID_CACHE: OrderedDict[int, tuple[object, frozenset[int]]] = OrderedDict()
_STRUCTURAL_ID_CACHE_MAX = 16


def _make_unsplit_special_token(token_str: str) -> AddedToken:
    return AddedToken(
        token_str,
        single_word=False,
        lstrip=False,
        rstrip=False,
        normalized=False,
        special=True,
    )


def _load_fast_tokenizer(resolved_id_or_path: str) -> PreTrainedTokenizerBase:
    candidate_path = Path(resolved_id_or_path)
    if candidate_path.is_file() and candidate_path.name == "tokenizer.json":
        return PreTrainedTokenizerFast(tokenizer_file=str(candidate_path))

    try:
        return AutoTokenizer.from_pretrained(resolved_id_or_path, use_fast=True)
    except Exception:
        if candidate_path.is_dir():
            if (candidate_path / "tokenizer_config.json").is_file():
                return PreTrainedTokenizerFast.from_pretrained(str(candidate_path))
            if (candidate_path / "tokenizer.json").is_file():
                return PreTrainedTokenizerFast(
                    tokenizer_file=str(candidate_path / "tokenizer.json")
                )
        raise


def _add_special_tokens_preserving_existing(
    tokenizer: PreTrainedTokenizerBase,
    special_dict: dict[str, AddedToken | list[AddedToken]],
) -> None:
    sig = inspect.signature(tokenizer.add_special_tokens)
    if "replace_extra_special_tokens" in sig.parameters:
        tokenizer.add_special_tokens(special_dict, replace_extra_special_tokens=False)
    elif "replace_additional_special_tokens" in sig.parameters:
        tokenizer.add_special_tokens(special_dict, replace_additional_special_tokens=False)
    else:
        existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        if "additional_special_tokens" in special_dict:
            merged = list(existing)
            existing_strs = {str(t) for t in existing}
            for item in special_dict["additional_special_tokens"]:  # type: ignore[union-attr]
                if str(item) not in existing_strs:
                    merged.append(item)
            special_dict = {**special_dict, "additional_special_tokens": merged}
        tokenizer.add_special_tokens(special_dict)


def build_jev_tokenizer(base_model_id: str, out_dir: Path) -> dict[str, int]:
    """Extend a ModernBERT or Qwen3 fast tokenizer with [CLS]/[SEP]/[PAD] + 14 Jev special tokens."""
    resolved = SUPPORTED_BACKBONES.get(str(base_model_id), str(base_model_id))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_fast_tokenizer(resolved)

    # Unconditionally bind cls_token, sep_token, and pad_token to [CLS], [SEP], and [PAD]
    # even when decoder backbones (e.g. Qwen3-0.6B) pre-populate pad_token = "<|endoftext|>".
    special_dict: dict[str, AddedToken | list[AddedToken]] = {
        "cls_token": _make_unsplit_special_token("[CLS]"),
        "sep_token": _make_unsplit_special_token("[SEP]"),
        "pad_token": _make_unsplit_special_token("[PAD]"),
        "additional_special_tokens": [
            _make_unsplit_special_token(jev_tok) for jev_tok in JEV_SPECIAL_TOKENS
        ],
    }
    _add_special_tokens_preserving_existing(tokenizer, special_dict)

    tokenizer.save_pretrained(str(out_dir))
    if not (out_dir / "tokenizer.json").is_file() and hasattr(tokenizer, "backend_tokenizer"):
        tokenizer.backend_tokenizer.save(str(out_dir / "tokenizer.json"))

    if not (out_dir / "tokenizer.json").is_file():
        raise RuntimeError(f"Failed to write fast tokenizer.json to {out_dir}")

    rust_tok = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    special_token_ids: dict[str, int] = {}
    for tok_str in (*REQUIRED_BASE_SPECIAL_TOKENS, *JEV_SPECIAL_TOKENS):
        tok_id = rust_tok.token_to_id(tok_str)
        if tok_id is None:
            raise RuntimeError(f"Required special token {tok_str!r} missing from saved tokenizer.json")
        encoded = rust_tok.encode(tok_str, add_special_tokens=False).ids
        if encoded != [tok_id]:
            raise RuntimeError(
                f"Special token {tok_str!r} split into {encoded} instead of [{tok_id}]"
            )
        special_token_ids[tok_str] = int(tok_id)

    if (
        tokenizer.cls_token_id != special_token_ids["[CLS]"]
        or tokenizer.sep_token_id != special_token_ids["[SEP]"]
        or tokenizer.pad_token_id != special_token_ids["[PAD]"]
    ):
        raise RuntimeError("Tokenizer [CLS]/[SEP]/[PAD] attribute IDs do not match tokenizer.json")

    return special_token_ids


def sanitize_untrusted_delimiters(text: str) -> str:
    """Neutralize literal Jev `<|...|>` and `[CLS]/[SEP]/[PAD]` tokens inside raw content spans."""
    if not text:
        return ""
    escaped = text
    while True:
        updated = escaped.replace("<|", r"<\|")
        updated = _BASE_BRACKET_SPECIAL_RE.sub(r"[\\\1]", updated)
        if updated == escaped:
            return updated
        escaped = updated


def _resolve_structural_ids(tokenizer: PreTrainedTokenizerBase | Tokenizer) -> frozenset[int]:
    cache_key = id(tokenizer)
    cached = _STRUCTURAL_ID_CACHE.get(cache_key)
    if cached is not None and cached[0] is tokenizer:
        _STRUCTURAL_ID_CACHE.move_to_end(cache_key)
        return cached[1]

    found: set[int] = set()
    if isinstance(tokenizer, Tokenizer):
        for tok_str in REQUIRED_SPECIAL_TOKENS:
            tid = tokenizer.token_to_id(tok_str)
            if tid is not None:
                found.add(int(tid))
    elif getattr(tokenizer, "is_fast", False) and hasattr(tokenizer, "backend_tokenizer"):
        backend = tokenizer.backend_tokenizer
        for tok_str in REQUIRED_SPECIAL_TOKENS:
            tid = backend.token_to_id(tok_str)
            if tid is not None:
                found.add(int(tid))
    else:
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for tok_str in REQUIRED_SPECIAL_TOKENS:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if isinstance(tid, int) and tid >= 0 and (unk_id is None or tid != unk_id):
                found.add(tid)

    frozen = frozenset(found)
    _STRUCTURAL_ID_CACHE[cache_key] = (tokenizer, frozen)
    while len(_STRUCTURAL_ID_CACHE) > _STRUCTURAL_ID_CACHE_MAX:
        _STRUCTURAL_ID_CACHE.popitem(last=False)
    return frozen


def encode_content_safe(
    tokenizer: PreTrainedTokenizerBase | Tokenizer,
    text: str,
) -> list[int]:
    """Encode raw intent/private/untrusted/arg text with zero structural special-token injection (C2)."""
    if not text:
        return []
    sanitized = sanitize_untrusted_delimiters(text)
    structural_ids = _resolve_structural_ids(tokenizer)

    if isinstance(tokenizer, Tokenizer):
        ids = [int(x) for x in tokenizer.encode(sanitized, add_special_tokens=False).ids]
    else:
        ids = [int(x) for x in tokenizer.encode(sanitized, add_special_tokens=False)]

    if not set(ids).isdisjoint(structural_ids):
        raise RuntimeError("Structural special token leaked into content span")
    return ids
