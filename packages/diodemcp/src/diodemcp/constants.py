"""Canonical constants for DiodeMCP Track B and DiodeModelBundle v1."""

from __future__ import annotations

from pathlib import Path
from typing import Final

DEFAULT_STATE_DIR_NAME: Final[str] = ".diodemcp"
DEFAULT_STATE_DIR: Final[Path] = Path.home() / DEFAULT_STATE_DIR_NAME

DEFAULT_HF_NAMESPACE: Final[str] = "diodemcp"
DEFAULT_MODEL_REPOS: Final[dict[str, str]] = {
    "base": "diodemcp/diodemcp-jev-base-int8",
    "large": "diodemcp/diodemcp-jev-large-int8",
}

MAX_LEDGER_SPAN_BYTES: Final[int] = 65536
DEFAULT_IDLE_TIMEOUT_SECONDS: Final[int] = 300

WARN_THRESHOLD: Final[float] = 0.20
CRITICAL_THRESHOLD: Final[float] = 0.80

BUNDLE_VERSION: Final[str] = "1.0"
REQUIRED_BUNDLE_FILES: Final[tuple[str, ...]] = (
    "manifest.json",
    "model_int8.onnx",
    "tokenizer.json",
    "calibration.json",
)

MAX_SEQ_LEN: Final[int] = 8192
LENGTH_BUCKETS: Final[tuple[int, ...]] = (256, 512, 1024, 2048, 4096, 8192)
MAX_PCHUNKS: Final[int] = 32
MAX_ARGS: Final[int] = 16

JEV_SPECIAL_TOKENS: Final[tuple[str, ...]] = (
    "<|intent|>",
    "<|missing_intent|>",
    "<|priv|>",
    "<|pchunk|>",
    "<|untrust|>",
    "<|uchunk|>",
    "<|call|>",
    "<|arg|>",
    "<|q_enc|>",
    "<|q_sem|>",
    "<|q_hijack|>",
    "<|q_src|>",
    "<|q_arg|>",
    "<|none|>",
)

REQUIRED_SPECIAL_TOKENS: Final[tuple[str, ...]] = (
    "[CLS]",
    "[SEP]",
    "[PAD]",
    *JEV_SPECIAL_TOKENS,
)

CANONICAL_SEGMENT_IDS: Final[dict[str, int]] = {
    "TRUSTED_INTENT": 0,
    "PRIVATE_DATA": 1,
    "DUAL_TRUST_DATA": 2,
    "UNTRUSTED_DATA": 3,
    "CANDIDATE_ACTION": 4,
    "DECISION_QUERY": 5,
}
