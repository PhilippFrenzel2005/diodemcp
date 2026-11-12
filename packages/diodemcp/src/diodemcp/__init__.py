"""DiodeMCP Track B: Local taint-tracking firewall and Jev cross-encoder runtime."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diodemcp.bundle import BundleManifest, BundleValidationError, validate_bundle_dir
    from diodemcp.cli import main as cli_main, require_interactive_tty
    from diodemcp.config_patcher import (
        PROTECTED_OAUTH_WARNING,
        load_shim_headers_for_server,
        read_wrap_manifest,
        unwrap_ide_configs,
        wrap_ide_configs,
    )
    from diodemcp.engine import (
        AsyncEngineClient,
        DiodeEngineServer,
        EngineClient,
        write_active_bundle_config,
    )
    from diodemcp.hooks import (
        emit_fail_closed_deny,
        extract_workspace_root,
        main_hook_cli,
        run_hook,
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
    from diodemcp.shim import (
        MAX_JSONRPC_LINE_BYTES,
        async_run_http_bridge_shim,
        async_run_stdio_shim,
        run_http_bridge_shim,
        run_stdio_shim,
        validate_streamable_http_url,
    )
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
    from diodemcp.types import (
        CandidateCall,
        CapabilityFlag,
        DecisionScores,
        RiskBand,
        TaintSpan,
    )

__version__ = "0.1.0"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AsyncEngineClient": ("diodemcp.engine", "AsyncEngineClient"),
    "BundleDownloadError": ("diodemcp.hub", "BundleDownloadError"),
    "BundleManifest": ("diodemcp.bundle", "BundleManifest"),
    "BundleValidationError": ("diodemcp.bundle", "BundleValidationError"),
    "CandidateCall": ("diodemcp.types", "CandidateCall"),
    "CapabilityFlag": ("diodemcp.types", "CapabilityFlag"),
    "CapabilityRegistry": ("diodemcp.registry", "CapabilityRegistry"),
    "DecisionScores": ("diodemcp.types", "DecisionScores"),
    "DiodeEngineServer": ("diodemcp.engine", "DiodeEngineServer"),
    "EngineClient": ("diodemcp.engine", "EngineClient"),
    "JevWindowPacker": ("diodemcp.packer", "JevWindowPacker"),
    "LedgerSnapshot": ("diodemcp.ledger", "LedgerSnapshot"),
    "MAX_JSONRPC_LINE_BYTES": ("diodemcp.shim", "MAX_JSONRPC_LINE_BYTES"),
    "MODEL_ALIASES": ("diodemcp.hub", "MODEL_ALIASES"),
    "OrtJevPredictor": ("diodemcp.ort_predictor", "OrtJevPredictor"),
    "PROTECTED_OAUTH_WARNING": ("diodemcp.config_patcher", "PROTECTED_OAUTH_WARNING"),
    "PackedJevInputs": ("diodemcp.packer", "PackedJevInputs"),
    "PredictorProtocol": ("diodemcp.predictor", "PredictorProtocol"),
    "REDACTED_AUDIT_PLACEHOLDER": ("diodemcp.redactor", "REDACTED_AUDIT_PLACEHOLDER"),
    "REDACTED_CALL_PLACEHOLDER": ("diodemcp.redactor", "REDACTED_CALL_PLACEHOLDER"),
    "RiskBand": ("diodemcp.types", "RiskBand"),
    "SessionLedger": ("diodemcp.ledger", "SessionLedger"),
    "TaintSpan": ("diodemcp.types", "TaintSpan"),
    "Tier0DeterministicPredictor": ("diodemcp.predictor", "Tier0DeterministicPredictor"),
    "TransportAuthError": ("diodemcp.transport", "TransportAuthError"),
    "TransportConfig": ("diodemcp.transport", "TransportConfig"),
    "TransportError": ("diodemcp.transport", "TransportError"),
    "TransportPermissionError": ("diodemcp.transport", "TransportPermissionError"),
    "async_run_http_bridge_shim": ("diodemcp.shim", "async_run_http_bridge_shim"),
    "async_run_stdio_shim": ("diodemcp.shim", "async_run_stdio_shim"),
    "build_redacted_audit_preview": ("diodemcp.redactor", "build_redacted_audit_preview"),
    "build_reference_4d_attn_mask": ("diodemcp.packer", "build_reference_4d_attn_mask"),
    "canonicalize_target_path": ("diodemcp.registry", "canonicalize_target_path"),
    "cli_main": ("diodemcp.cli", "main"),
    "compute_popup_binding_sha256": ("diodemcp.transport", "compute_popup_binding_sha256"),
    "compute_simhash64": ("diodemcp.ledger", "compute_simhash64"),
    "compute_workspace_key": ("diodemcp.ledger", "compute_workspace_key"),
    "default_transport": ("diodemcp.transport", "default_transport"),
    "download_and_verify_bundle": ("diodemcp.hub", "download_and_verify_bundle"),
    "emit_fail_closed_deny": ("diodemcp.hooks", "emit_fail_closed_deny"),
    "encode_content_safe": ("diodemcp.packer", "encode_content_safe"),
    "ensure_admin_secret": ("diodemcp.transport", "ensure_admin_secret"),
    "extract_workspace_root": ("diodemcp.hooks", "extract_workspace_root"),
    "flatten_call_arguments": ("diodemcp.packer", "flatten_call_arguments"),
    "load_shim_headers_for_server": ("diodemcp.config_patcher", "load_shim_headers_for_server"),
    "main_hook_cli": ("diodemcp.hooks", "main_hook_cli"),
    "read_admin_secret": ("diodemcp.transport", "read_admin_secret"),
    "read_wrap_manifest": ("diodemcp.config_patcher", "read_wrap_manifest"),
    "redact_call_arguments": ("diodemcp.redactor", "redact_call_arguments"),
    "require_interactive_tty": ("diodemcp.cli", "require_interactive_tty"),
    "run_hook": ("diodemcp.hooks", "run_hook"),
    "run_http_bridge_shim": ("diodemcp.shim", "run_http_bridge_shim"),
    "run_stdio_shim": ("diodemcp.shim", "run_stdio_shim"),
    "sanitize_content_delimiters": ("diodemcp.packer", "sanitize_content_delimiters"),
    "unwrap_ide_configs": ("diodemcp.config_patcher", "unwrap_ide_configs"),
    "validate_bundle_dir": ("diodemcp.bundle", "validate_bundle_dir"),
    "validate_streamable_http_url": ("diodemcp.shim", "validate_streamable_http_url"),
    "wrap_ide_configs": ("diodemcp.config_patcher", "wrap_ide_configs"),
    "write_active_bundle_config": ("diodemcp.engine", "write_active_bundle_config"),
}

__all__ = ["__version__", *sorted(_LAZY_EXPORTS.keys())]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'diodemcp' has no attribute {name!r}")
    mod_name, attr_name = target
    mod = importlib.import_module(mod_name)
    val = getattr(mod, attr_name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
