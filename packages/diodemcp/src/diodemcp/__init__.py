"""DiodeMCP Track B: Local taint-tracking firewall and Jev cross-encoder runtime."""

from __future__ import annotations

from diodemcp.bundle import BundleManifest, BundleValidationError, validate_bundle_dir
from diodemcp.predictor import PredictorProtocol, Tier0DeterministicPredictor
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
    "PredictorProtocol",
    "RiskBand",
    "TaintSpan",
    "Tier0DeterministicPredictor",
    "canonicalize_target_path",
    "validate_bundle_dir",
]
