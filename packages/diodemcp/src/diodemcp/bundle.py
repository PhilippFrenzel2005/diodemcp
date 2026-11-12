"""DiodeModelBundle v1 manifest schema, tokenizer cross-check, and SHA256 validator."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from tokenizers import Tokenizer

from diodemcp.constants import (
    BUNDLE_VERSION,
    CANONICAL_SEGMENT_IDS,
    CRITICAL_THRESHOLD,
    LENGTH_BUCKETS,
    MAX_ARGS,
    MAX_PCHUNKS,
    MAX_SEQ_LEN,
    REQUIRED_BUNDLE_FILES,
    REQUIRED_SPECIAL_TOKENS,
    WARN_THRESHOLD,
)

_SHA256_LOWER_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_STREAM_CHUNK_BYTES = 65536


class BundleValidationError(ValueError):
    """Raised when a DiodeModelBundle v1 directory fails structural or SHA256 validation."""


class BundleManifest(BaseModel):
    """Frozen DiodeModelBundle v1 manifest contract."""

    bundle_version: Literal["1.0"] = BUNDLE_VERSION
    model_id: str = Field(min_length=1)
    sha256_model_onnx: str
    sha256_tokenizer_json: str
    sha256_calibration_json: str
    max_seq_len: int = Field(default=MAX_SEQ_LEN, ge=1)
    length_buckets: list[int] = Field(default_factory=lambda: list(LENGTH_BUCKETS))
    max_pchunks: int = Field(default=MAX_PCHUNKS, ge=1)
    max_args: int = Field(default=MAX_ARGS, ge=1)
    special_token_ids: dict[str, int]
    segment_ids: dict[str, int] = Field(default_factory=lambda: dict(CANONICAL_SEGMENT_IDS))
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "warn_min": WARN_THRESHOLD,
            "critical_min": CRITICAL_THRESHOLD,
        }
    )
    ece_val: float = Field(ge=0.0, le=1.0)

    @field_validator("sha256_model_onnx", "sha256_tokenizer_json", "sha256_calibration_json")
    @classmethod
    def _validate_sha256_hex(cls, value: str) -> str:
        if not _SHA256_LOWER_HEX_RE.match(value):
            raise ValueError("SHA256 digest must be a 64-character lowercase hex string")
        return value

    @field_validator("length_buckets")
    @classmethod
    def _validate_length_buckets(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("length_buckets must not be empty")
        if any(b <= 0 for b in value):
            raise ValueError("length_buckets must contain only positive integers")
        if any(value[i] >= value[i + 1] for i in range(len(value) - 1)):
            raise ValueError("length_buckets must be strictly ascending")
        return value

    @field_validator("special_token_ids")
    @classmethod
    def _validate_special_token_ids(cls, value: dict[str, int]) -> dict[str, int]:
        missing = [tok for tok in REQUIRED_SPECIAL_TOKENS if tok not in value]
        if missing:
            raise ValueError(f"Missing required special tokens: {missing}")

        for tok, token_id in value.items():
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError(f"Special token {tok!r} must map to a non-negative int")

        required_ids = [value[tok] for tok in REQUIRED_SPECIAL_TOKENS]
        if len(set(required_ids)) != len(REQUIRED_SPECIAL_TOKENS):
            raise ValueError("All 17 REQUIRED_SPECIAL_TOKENS must have distinct integer IDs")

        return value

    @field_validator("segment_ids")
    @classmethod
    def _validate_segment_ids(cls, value: dict[str, int]) -> dict[str, int]:
        if value != CANONICAL_SEGMENT_IDS:
            raise ValueError(
                f"segment_ids must match CANONICAL_SEGMENT_IDS ({CANONICAL_SEGMENT_IDS})"
            )
        return value

    @field_validator("thresholds")
    @classmethod
    def _validate_thresholds(cls, value: dict[str, float]) -> dict[str, float]:
        if "warn_min" not in value or "critical_min" not in value:
            raise ValueError("thresholds must include 'warn_min' and 'critical_min'")
        warn_min = value["warn_min"]
        critical_min = value["critical_min"]
        if not (0.05 <= warn_min <= 0.30):
            raise ValueError("warn_min must satisfy 0.05 <= warn_min <= 0.30")
        if not (0.50 <= critical_min <= 0.90):
            raise ValueError("critical_min must satisfy 0.50 <= critical_min <= 0.90")
        if not (warn_min < critical_min):
            raise ValueError("thresholds must satisfy warn_min < critical_min")
        return value

    @model_validator(mode="after")
    def _validate_max_seq_len_covers_buckets(self) -> BundleManifest:
        if self.max_seq_len < self.length_buckets[-1]:
            raise ValueError("max_seq_len must be >= the largest bucket in length_buckets")
        return self


def compute_file_sha256(file_path: Path, chunk_size: int = _STREAM_CHUNK_BYTES) -> str:
    """Compute lowercase SHA256 hex digest by streaming the file in chunks."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_tokenizer_special_tokens(
    tokenizer_path: Path, raw_tok: dict[str, Any], expected_ids: dict[str, int]
) -> None:
    if "model" in raw_tok:
        try:
            tok_obj = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:
            raise BundleValidationError(
                f"Invalid tokenizer.json in {tokenizer_path.parent}: {exc}"
            ) from exc

        for tok_str, expected_id in expected_ids.items():
            actual_id = tok_obj.token_to_id(tok_str)
            if actual_id != expected_id:
                raise BundleValidationError(
                    f"Tokenizer special token mismatch for {tok_str!r}: "
                    f"manifest expects {expected_id}, tokenizer has {actual_id}"
                )
        return

    if "added_tokens" in raw_tok or "vocab" in raw_tok:
        combined: dict[str, Any] = {}
        vocab = raw_tok.get("vocab")
        if isinstance(vocab, dict):
            combined.update(vocab)
        added = raw_tok.get("added_tokens")
        if isinstance(added, list):
            for entry in added:
                if isinstance(entry, dict) and "content" in entry and "id" in entry:
                    combined[str(entry["content"])] = entry["id"]

        for tok_str, expected_id in expected_ids.items():
            actual_id = combined.get(tok_str)
            if actual_id != expected_id:
                raise BundleValidationError(
                    f"Tokenizer special token mismatch for {tok_str!r}: "
                    f"manifest expects {expected_id}, tokenizer has {actual_id}"
                )


def validate_bundle_dir(bundle_dir: Path, verify_sha256: bool = True) -> BundleManifest:
    """Validate a DiodeModelBundle v1 directory, artifact SHA256s, and tokenizer token IDs."""
    root = Path(bundle_dir)
    if not root.exists() or not root.is_dir():
        raise BundleValidationError(f"Bundle directory does not exist: {root}")

    for filename in REQUIRED_BUNDLE_FILES:
        candidate = root / filename
        if not candidate.is_file():
            raise BundleValidationError(
                f"Bundle directory {root} is missing required file: {filename}"
            )

    manifest_path = root / "manifest.json"
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = BundleManifest.model_validate(raw_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise BundleValidationError(f"Invalid manifest.json in {root}: {exc}") from exc

    calibration_path = root / "calibration.json"
    try:
        raw_calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"Invalid calibration.json in {root}: {exc}") from exc
    if not isinstance(raw_calibration, dict):
        raise BundleValidationError(f"Invalid calibration.json in {root}: expected JSON object")

    tokenizer_path = root / "tokenizer.json"
    try:
        raw_tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"Invalid tokenizer.json in {root}: {exc}") from exc
    if not isinstance(raw_tokenizer, dict):
        raise BundleValidationError(f"Invalid tokenizer.json in {root}: expected JSON object")

    if verify_sha256:
        expected_hashes = (
            ("model_int8.onnx", manifest.sha256_model_onnx),
            ("tokenizer.json", manifest.sha256_tokenizer_json),
            ("calibration.json", manifest.sha256_calibration_json),
        )
        for filename, expected_sha in expected_hashes:
            artifact_path = root / filename
            try:
                actual_sha = compute_file_sha256(artifact_path)
            except OSError as exc:
                raise BundleValidationError(f"Failed reading {artifact_path}: {exc}") from exc
            if actual_sha != expected_sha:
                raise BundleValidationError(
                    f"SHA256 mismatch for {artifact_path}: expected {expected_sha}, "
                    f"got {actual_sha}"
                )

    _verify_tokenizer_special_tokens(tokenizer_path, raw_tokenizer, manifest.special_token_ids)
    return manifest
