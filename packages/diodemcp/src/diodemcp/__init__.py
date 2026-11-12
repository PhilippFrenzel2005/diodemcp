"""DiodeMCP Track B: Local taint-tracking firewall and Jev cross-encoder runtime."""

from __future__ import annotations

from diodemcp.bundle import BundleManifest, BundleValidationError, validate_bundle_dir
from diodemcp.engine import (
    AsyncEngineClient,
    DiodeEngineServer,
    EngineClient,
    write_active_bundle_config,
)
from diodemcp.hub import BundleDownloadError, MODEL_ALIASES, download_and_verify_bundle
from diodemcp.ledger import (
    LedgerSnapshot,
    SessionLedger,
    compute_simhash64,
    compute_workspace_key,
)
from diodemcp.ort_predictor import OrtJevPredictor
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
from diodemcp.transport import (
    TransportAuthError,
    TransportConfig,
    TransportError,
    TransportPermissionError,
    compute_popup_binding_sha256,
    default_transport,
    ensure_admin_secret,
    read_admin_secret,
)
from diodemcp.types import CandidateCall, CapabilityFlag, DecisionScores, RiskBand, TaintSpan

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AsyncEngineClient",
    "BundleDownloadError",
    "BundleManifest",
    "BundleValidationError",
    "CandidateCall",
    "CapabilityFlag",
    "CapabilityRegistry",
    "DecisionScores",
    "DiodeEngineServer",
    "EngineClient",
    "JevWindowPacker",
    "LedgerSnapshot",
    "MODEL_ALIASES",
    "OrtJevPredictor",
    "PackedJevInputs",
    "PredictorProtocol",
    "REDACTED_AUDIT_PLACEHOLDER",
    "REDACTED_CALL_PLACEHOLDER",
    "RiskBand",
    "SessionLedger",
    "TaintSpan",
    "Tier0DeterministicPredictor",
    "TransportAuthError",
    "TransportConfig",
    "TransportError",
    "TransportPermissionError",
    "build_redacted_audit_preview",
    "build_reference_4d_attn_mask",
    "canonicalize_target_path",
    "compute_popup_binding_sha256",
    "compute_simhash64",
    "compute_workspace_key",
    "default_transport",
    "download_and_verify_bundle",
    "encode_content_safe",
    "ensure_admin_secret",
    "flatten_call_arguments",
    "read_admin_secret",
    "redact_call_arguments",
    "sanitize_content_delimiters",
    "validate_bundle_dir",
    "write_active_bundle_config",
]
