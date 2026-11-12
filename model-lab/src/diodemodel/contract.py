"""Frozen DiodeModelBundle v1 contract, manifest schema, and ONNX tensor validator."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal

import onnx
from onnx import TensorProto
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from tokenizers import Tokenizer


JEV_SPECIAL_TOKENS: tuple[str, ...] = (
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

REQUIRED_BASE_SPECIAL_TOKENS: tuple[str, ...] = ("[CLS]", "[SEP]", "[PAD]")
REQUIRED_SPECIAL_TOKENS: tuple[str, ...] = REQUIRED_BASE_SPECIAL_TOKENS + JEV_SPECIAL_TOKENS
ALL_STRUCTURAL_SPECIAL_TOKENS: tuple[str, ...] = REQUIRED_SPECIAL_TOKENS

REQUIRED_BUNDLE_FILES: tuple[str, ...] = (
    "manifest.json",
    "model_int8.onnx",
    "tokenizer.json",
    "calibration.json",
)

CANONICAL_SEGMENT_IDS: dict[str, int] = {
    "TRUSTED_INTENT": 0,
    "PRIVATE_DATA": 1,
    "DUAL_TRUST_DATA": 2,
    "UNTRUSTED_DATA": 3,
    "CANDIDATE_ACTION": 4,
    "DECISION_QUERY": 5,
}

ROLE_PAD: int = 0
ROLE_STATE: int = 1
ROLE_QUERY: int = 2
TOKEN_ROLES: dict[str, int] = {
    "PAD": ROLE_PAD,
    "STATE": ROLE_STATE,
    "QUERY": ROLE_QUERY,
}

ONNX_OPSET_VERSION: int = 17
DEFAULT_LENGTH_BUCKETS: list[int] = [128, 256, 512, 1024, 2048, 4096]
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class BundleValidationError(ValueError):
    """Raised when a DiodeModelBundle v1 directory fails structural, SHA256, or ONNX validation."""


@dataclass(frozen=True)
class TensorSpec:
    """Expected ONNX graph input/output tensor signature."""

    name: str
    elem_type: int
    shape: tuple[int | None, ...]


ONNX_INPUT_SPECS: dict[str, TensorSpec] = {
    "input_ids": TensorSpec("input_ids", TensorProto.INT64, (None, None)),
    "seg_ids": TensorSpec("seg_ids", TensorProto.INT64, (None, None)),
    "token_roles": TensorSpec("token_roles", TensorProto.INT64, (None, None)),
    "query_positions": TensorSpec("query_positions", TensorProto.INT64, (None, 5)),
    "pchunk_positions": TensorSpec("pchunk_positions", TensorProto.INT64, (None, 33)),
    "pchunk_valid_mask": TensorSpec("pchunk_valid_mask", TensorProto.BOOL, (None, 33)),
    "arg_positions": TensorSpec("arg_positions", TensorProto.INT64, (None, 17)),
    "arg_valid_mask": TensorSpec("arg_valid_mask", TensorProto.BOOL, (None, 17)),
}

ONNX_OUTPUT_SPECS: dict[str, TensorSpec] = {
    "noul_probs": TensorSpec("noul_probs", TensorProto.FLOAT, (None, 3)),
    "src_choice_probs": TensorSpec("src_choice_probs", TensorProto.FLOAT, (None, 33)),
    "arg_choice_probs": TensorSpec("arg_choice_probs", TensorProto.FLOAT, (None, 17)),
}


def clamp_thresholds(thresholds: dict[str, float]) -> dict[str, float]:
    """Explicit export helper: clamp warn_min to [0.05, 0.30] and critical_min to [0.50, 0.90]."""
    if "warn_min" not in thresholds or "critical_min" not in thresholds:
        raise ValueError("thresholds must contain both 'warn_min' and 'critical_min'")
    warn_raw = float(thresholds["warn_min"])
    crit_raw = float(thresholds["critical_min"])
    if not (math.isfinite(warn_raw) and math.isfinite(crit_raw)):
        raise ValueError("thresholds must be finite floats")
    warn_clamped = min(max(warn_raw, 0.05), 0.30)
    crit_clamped = min(max(crit_raw, 0.50), 0.90)
    return {"warn_min": warn_clamped, "critical_min": crit_clamped}


class BundleManifest(BaseModel):
    """Manifest schema for DiodeModelBundle v1 (exact parity with Track B bundle.py)."""

    bundle_version: Literal["1.0"] = "1.0"
    model_id: str = Field(min_length=1)
    sha256_model_onnx: str
    sha256_tokenizer_json: str
    sha256_calibration_json: str
    max_seq_len: int = Field(default=8192, ge=1)
    length_buckets: list[int] = Field(default_factory=lambda: list(DEFAULT_LENGTH_BUCKETS))
    max_pchunks: int = 32
    max_args: int = 16
    special_token_ids: dict[str, int]
    segment_ids: dict[str, int] = Field(default_factory=lambda: dict(CANONICAL_SEGMENT_IDS))
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {"warn_min": 0.20, "critical_min": 0.80}
    )
    ece_val: float = Field(ge=0.0, le=1.0)

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model_id must be non-empty")
        return cleaned

    @field_validator("sha256_model_onnx", "sha256_tokenizer_json", "sha256_calibration_json")
    @classmethod
    def _validate_sha256_hex(cls, value: str) -> str:
        if not _HEX64_RE.fullmatch(value):
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

    @field_validator("max_pchunks")
    @classmethod
    def _validate_max_pchunks(cls, value: int) -> int:
        if value != 32:
            raise ValueError("DiodeModelBundle v1 requires max_pchunks == 32")
        return value

    @field_validator("max_args")
    @classmethod
    def _validate_max_args(cls, value: int) -> int:
        if value != 16:
            raise ValueError("DiodeModelBundle v1 requires max_args == 16")
        return value

    @field_validator("special_token_ids")
    @classmethod
    def _validate_special_token_ids(cls, value: dict[str, int]) -> dict[str, int]:
        missing = [tok for tok in REQUIRED_SPECIAL_TOKENS if tok not in value]
        if missing:
            raise ValueError(f"Missing required special tokens: {missing}")
        extra = [tok for tok in value if tok not in REQUIRED_SPECIAL_TOKENS]
        if extra:
            raise ValueError(f"Unexpected extra special tokens: {extra}")

        seen_ids: dict[int, str] = {}
        for tok in REQUIRED_SPECIAL_TOKENS:
            tok_id = value[tok]
            if isinstance(tok_id, bool) or not isinstance(tok_id, int) or tok_id < 0:
                raise ValueError(f"Special token {tok!r} must map to a non-negative int")
            if tok_id in seen_ids:
                raise ValueError(
                    f"All 17 REQUIRED_SPECIAL_TOKENS must have distinct integer IDs "
                    f"(duplicate {tok_id} on {seen_ids[tok_id]!r} and {tok!r})"
                )
            seen_ids[tok_id] = tok
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
        if isinstance(warn_min, bool) or isinstance(critical_min, bool):
            raise ValueError("thresholds must be numeric floats, not booleans")
        if not (math.isfinite(warn_min) and math.isfinite(critical_min)):
            raise ValueError("thresholds must be finite floats")
        if not (0.05 <= warn_min <= 0.30):
            raise ValueError("warn_min must satisfy 0.05 <= warn_min <= 0.30")
        if not (0.50 <= critical_min <= 0.90):
            raise ValueError("critical_min must satisfy 0.50 <= critical_min <= 0.90")
        if not (warn_min < critical_min):
            raise ValueError("thresholds must satisfy warn_min < critical_min")
        return {"warn_min": float(warn_min), "critical_min": float(critical_min)}

    @field_validator("ece_val")
    @classmethod
    def _validate_ece_val(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"ece_val must be in [0.0, 1.0], got {value!r}")
        return value

    @model_validator(mode="after")
    def _validate_max_seq_len_covers_buckets(self) -> BundleManifest:
        if self.max_seq_len < self.length_buckets[-1]:
            raise ValueError("max_seq_len must be >= the largest bucket in length_buckets")
        return self


def compute_sha256(file_path: Path) -> str:
    """Compute lowercase SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        while chunk := fh.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_value_info(value_info: onnx.ValueInfoProto, spec: TensorSpec, kind: str) -> None:
    if not value_info.type.HasField("tensor_type"):
        raise ValueError(f"ONNX {kind} {spec.name!r} is not a tensor")
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type != spec.elem_type:
        expected_name = TensorProto.DataType.Name(spec.elem_type)
        actual_name = TensorProto.DataType.Name(tensor_type.elem_type)
        raise ValueError(
            f"ONNX {kind} {spec.name!r} expected dtype {expected_name}, got {actual_name}"
        )
    if not tensor_type.HasField("shape"):
        raise ValueError(f"ONNX {kind} {spec.name!r} is missing tensor shape metadata")
    dims = tensor_type.shape.dim
    if len(dims) != len(spec.shape):
        raise ValueError(
            f"ONNX {kind} {spec.name!r} expected rank {len(spec.shape)}, got {len(dims)}"
        )
    for axis, (dim_proto, expected_dim) in enumerate(zip(dims, spec.shape, strict=True)):
        if expected_dim is not None:
            if not dim_proto.HasField("dim_value") or dim_proto.dim_value != expected_dim:
                actual_desc = (
                    dim_proto.dim_value
                    if dim_proto.HasField("dim_value")
                    else dim_proto.dim_param or "?"
                )
                raise ValueError(
                    f"ONNX {kind} {spec.name!r} axis {axis} expected static dim {expected_dim}, "
                    f"got {actual_desc}"
                )


def validate_onnx_contract(onnx_path: Path) -> None:
    """Validate that an ONNX graph matches the DiodeModelBundle v1 opset=17 tensor contract."""
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

    model = onnx.load(str(onnx_path), load_external_data=False)

    default_opsets = [
        imp.version for imp in model.opset_import if imp.domain in ("", "ai.onnx")
    ]
    if not default_opsets or default_opsets[0] != ONNX_OPSET_VERSION:
        raise ValueError(
            f"Expected ONNX ai.onnx opset == {ONNX_OPSET_VERSION}, got {default_opsets}"
        )

    initializer_names = {init.name for init in model.graph.initializer}
    graph_inputs = [inp for inp in model.graph.input if inp.name not in initializer_names]
    actual_input_names = tuple(inp.name for inp in graph_inputs)
    expected_input_names = tuple(ONNX_INPUT_SPECS.keys())
    if actual_input_names != expected_input_names:
        raise ValueError(
            f"ONNX inputs mismatch: expected {expected_input_names}, got {actual_input_names}"
        )

    for inp_proto in graph_inputs:
        _validate_value_info(inp_proto, ONNX_INPUT_SPECS[inp_proto.name], kind="input")

    actual_output_names = tuple(out.name for out in model.graph.output)
    expected_output_names = tuple(ONNX_OUTPUT_SPECS.keys())
    if actual_output_names != expected_output_names:
        raise ValueError(
            f"ONNX outputs mismatch: expected {expected_output_names}, got {actual_output_names}"
        )

    for out_proto in model.graph.output:
        _validate_value_info(out_proto, ONNX_OUTPUT_SPECS[out_proto.name], kind="output")


def _verify_tokenizer_special_tokens(
    tokenizer_path: Path,
    raw_tok: dict[str, Any],
    expected_ids: dict[str, int],
) -> None:
    if not any(k in raw_tok for k in ("model", "added_tokens", "vocab")):
        raise BundleValidationError(
            f"Invalid tokenizer.json in {tokenizer_path.parent}: "
            "must contain 'model', 'added_tokens', or 'vocab'"
        )

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
    """Validate a complete DiodeModelBundle v1 directory, SHA256s, tokenizer IDs, and ONNX graph."""
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
                actual_sha = compute_sha256(artifact_path)
            except OSError as exc:
                raise BundleValidationError(f"Failed reading {artifact_path}: {exc}") from exc
            if actual_sha != expected_sha:
                raise BundleValidationError(
                    f"SHA256 mismatch for {artifact_path}: expected {expected_sha}, "
                    f"got {actual_sha}"
                )

    _verify_tokenizer_special_tokens(tokenizer_path, raw_tokenizer, manifest.special_token_ids)

    onnx_path = root / "model_int8.onnx"
    try:
        validate_onnx_contract(onnx_path)
    except Exception as exc:
        raise BundleValidationError(f"Invalid model_int8.onnx in {root}: {exc}") from exc

    return manifest
