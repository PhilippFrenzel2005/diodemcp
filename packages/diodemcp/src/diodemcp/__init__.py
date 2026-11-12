"""DiodeMCP Track B: Local taint-tracking firewall and Jev cross-encoder runtime."""

from __future__ import annotations

from diodemcp.bundle import BundleManifest, BundleValidationError, validate_bundle_dir
from diodemcp.ledger import (
    LedgerSnapshot,
    SessionLedger,
    compute_simhash64,
    compute_workspace_key,
)
from diodemcp.packer import (
    JevWindowPacker,
    PackedJevInputs,
    build_reference_4d_attn_mask,
    encode_content_safe,
    flatten_call_arguments,
    sanitize_content_delimiters,
)
from diodemcp.predictor import PredictorProtocol, Tier0DeterministicPredictor
from diodemcp.redactor import (
    REDACTED_AUDIT_PLACEHOLDER,
    REDACTED_CALL_PLACEHOLDER,
    build_redacted_audit_preview,
    redact_call_arguments,
)
from diodemcp.registry import CapabilityRegistry, canonicalize_target_path
from diodemcp.types import CandidateCall, CapabilityFlag, DecisionScores, RiskBand, TaintSpan

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BundleManifest",
    "BundleValidationError",
    "CandidateCall",
    "CapabilityFlag",
    "CapabilityRegistry",
    "DecisionScores",
    "JevWindowPacker",
    "LedgerSnapshot",
    "PackedJevInputs",
    "PredictorProtocol",
    "REDACTED_AUDIT_PLACEHOLDER",
    "REDACTED_CALL_PLACEHOLDER",
    "RiskBand",
    "SessionLedger",
    "TaintSpan",
    "Tier0DeterministicPredictor",
    "build_redacted_audit_preview",
    "build_reference_4d_attn_mask",
    "canonicalize_target_path",
    "compute_simhash64",
    "compute_workspace_key",
    "encode_content_safe",
    "flatten_call_arguments",
    "redact_call_arguments",
    "sanitize_content_delimiters",
    "validate_bundle_dir",
]
