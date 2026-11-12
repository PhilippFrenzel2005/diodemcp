"""DiodeMCP Track A model-lab package."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

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
from diodemodel.tokenizer_builder import (
    SUPPORTED_BACKBONES,
    build_jev_tokenizer,
    encode_content_safe,
)

if TYPE_CHECKING:
    from diodemodel.dataset_packer import (
        AuditReport,
        DatasetLeakageError,
        JevDatasetPacker,
        PackedJevRow,
        adapt_benchmark_holdout_seeds,
        audit_packed_splits,
        build_disjoint_dataset_splits,
        build_reference_4d_attn_mask,
        pack_dataset_to_parquet,
    )
    from diodemodel.mutators import (
        JevTrainingPair,
        SpanItem,
        apply_missing_intent_mask,
        flatten_call_arguments,
        generate_programmatic_pairs,
    )
    from diodemodel.parse_seeds import (
        ALLOWED_SEED_LICENSES,
        SUPPORTED_CODE_AND_DOC_FORMATS,
        SeedRecord,
        filter_permissive_records,
        load_permissive_seeds,
        scrub_real_secret_patterns,
    )
    from diodemodel.synth_vllm import (
        DEFAULT_SYNTH_MODEL_ID,
        iter_matched_vllm_pairs,
        load_synthesized_shards,
        scrub_pair_real_secrets,
        synthesize_matched_shards,
    )

_LAZY_EXPORTS: dict[str, str] = {
    "ALLOWED_SEED_LICENSES": "diodemodel.parse_seeds",
    "SUPPORTED_CODE_AND_DOC_FORMATS": "diodemodel.parse_seeds",
    "SeedRecord": "diodemodel.parse_seeds",
    "filter_permissive_records": "diodemodel.parse_seeds",
    "load_permissive_seeds": "diodemodel.parse_seeds",
    "scrub_real_secret_patterns": "diodemodel.parse_seeds",
    "JevTrainingPair": "diodemodel.mutators",
    "SpanItem": "diodemodel.mutators",
    "apply_missing_intent_mask": "diodemodel.mutators",
    "flatten_call_arguments": "diodemodel.mutators",
    "generate_programmatic_pairs": "diodemodel.mutators",
    "DEFAULT_SYNTH_MODEL_ID": "diodemodel.synth_vllm",
    "iter_matched_vllm_pairs": "diodemodel.synth_vllm",
    "load_synthesized_shards": "diodemodel.synth_vllm",
    "scrub_pair_real_secrets": "diodemodel.synth_vllm",
    "synthesize_matched_shards": "diodemodel.synth_vllm",
    "AuditReport": "diodemodel.dataset_packer",
    "DatasetLeakageError": "diodemodel.dataset_packer",
    "JevDatasetPacker": "diodemodel.dataset_packer",
    "PackedJevRow": "diodemodel.dataset_packer",
    "adapt_benchmark_holdout_seeds": "diodemodel.dataset_packer",
    "audit_packed_splits": "diodemodel.dataset_packer",
    "build_disjoint_dataset_splits": "diodemodel.dataset_packer",
    "build_reference_4d_attn_mask": "diodemodel.dataset_packer",
    "pack_dataset_to_parquet": "diodemodel.dataset_packer",
}


def __getattr__(name: str) -> Any:
    mod_name = _LAZY_EXPORTS.get(name)
    if mod_name is not None:
        mod = importlib.import_module(mod_name)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALLOWED_SEED_LICENSES",
    "ALL_STRUCTURAL_SPECIAL_TOKENS",
    "AuditReport",
    "BundleManifest",
    "BundleValidationError",
    "CANONICAL_SEGMENT_IDS",
    "DEFAULT_SYNTH_MODEL_ID",
    "DatasetLeakageError",
    "JEV_SPECIAL_TOKENS",
    "JevDatasetPacker",
    "JevTrainingPair",
    "ONNX_INPUT_SPECS",
    "ONNX_OPSET_VERSION",
    "ONNX_OUTPUT_SPECS",
    "PackedJevRow",
    "REQUIRED_SPECIAL_TOKENS",
    "SUPPORTED_BACKBONES",
    "SUPPORTED_CODE_AND_DOC_FORMATS",
    "SeedRecord",
    "SpanItem",
    "TOKEN_ROLES",
    "adapt_benchmark_holdout_seeds",
    "apply_missing_intent_mask",
    "audit_packed_splits",
    "build_disjoint_dataset_splits",
    "build_jev_tokenizer",
    "build_reference_4d_attn_mask",
    "clamp_thresholds",
    "encode_content_safe",
    "filter_permissive_records",
    "flatten_call_arguments",
    "generate_programmatic_pairs",
    "iter_matched_vllm_pairs",
    "load_permissive_seeds",
    "load_synthesized_shards",
    "pack_dataset_to_parquet",
    "scrub_pair_real_secrets",
    "scrub_real_secret_patterns",
    "synthesize_matched_shards",
    "validate_bundle_dir",
    "validate_onnx_contract",
]

__version__ = "0.1.0"
