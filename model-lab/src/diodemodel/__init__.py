"""DiodeMCP Track A model-lab package."""

from diodemodel.contract import (
    ALL_STRUCTURAL_SPECIAL_TOKENS,
    BundleManifest,
    BundleValidationError,
    CANONICAL_SEGMENT_IDS,
    JEV_SPECIAL_TOKENS,
    ONNX_INPUT_SPECS,
    ONNX_OPSET_VERSION,
    ONNX_OUTPUT_SPECS,
    REQUIRED_SPECIAL_TOKENS,
    TOKEN_ROLES,
    clamp_thresholds,
    validate_bundle_dir,
    validate_onnx_contract,
)
from diodemodel.parse_seeds import (
    ALLOWED_SEED_LICENSES,
    SUPPORTED_CODE_AND_DOC_FORMATS,
    SeedRecord,
    filter_permissive_records,
    load_permissive_seeds,
    scrub_real_secret_patterns,
)
from diodemodel.tokenizer_builder import (
    SUPPORTED_BACKBONES,
    build_jev_tokenizer,
    encode_content_safe,
)

__all__ = [
    "ALLOWED_SEED_LICENSES",
    "ALL_STRUCTURAL_SPECIAL_TOKENS",
    "BundleManifest",
    "BundleValidationError",
    "CANONICAL_SEGMENT_IDS",
    "JEV_SPECIAL_TOKENS",
    "ONNX_INPUT_SPECS",
    "ONNX_OPSET_VERSION",
    "ONNX_OUTPUT_SPECS",
    "REQUIRED_SPECIAL_TOKENS",
    "SUPPORTED_BACKBONES",
    "SUPPORTED_CODE_AND_DOC_FORMATS",
    "SeedRecord",
    "TOKEN_ROLES",
    "build_jev_tokenizer",
    "clamp_thresholds",
    "encode_content_safe",
    "filter_permissive_records",
    "load_permissive_seeds",
    "scrub_real_secret_patterns",
    "validate_bundle_dir",
    "validate_onnx_contract",
]

__version__ = "0.1.0"
