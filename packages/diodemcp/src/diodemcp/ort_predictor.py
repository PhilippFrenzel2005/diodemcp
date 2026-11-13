"""Zero-PyTorch ONNX Runtime Jev cross-encoder predictor with Tier-0 floor and bucket escalation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import threading
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from diodemcp.bundle import BundleManifest, BundleValidationError, validate_bundle_dir
from diodemcp.constants import (
    DEFAULT_INFERENCE_BUCKET,
    ONNX_INPUT_NAMES,
    ONNX_OUTPUT_NAMES,
)
from diodemcp.ledger import LedgerSnapshot
from diodemcp.packer import JevWindowPacker, PackedJevInputs
from diodemcp.predictor import Tier0DeterministicPredictor
from diodemcp.types import CandidateCall, DecisionScores, TaintSpan

_ESC_BORDERLINE_MIN: float = 0.12
_ESC_MAX_BUCKET: int = 2048
_PROB_EPSILON: float = 1e-5


def _extract_valid_prob_row(raw_tensor: Any, expected_cols: int) -> np.ndarray | None:
    """Return 1D finite probability row `[expected_cols]` in `[0.0, 1.0]` or `None` if invalid."""
    try:
        raw_arr = np.asarray(raw_tensor)
    except (TypeError, ValueError):
        return None
    if (
        raw_arr.dtype == np.bool_
        or raw_arr.ndim != 2
        or raw_arr.shape[0] < 1
        or raw_arr.shape[1] != expected_cols
    ):
        return None
    try:
        row = np.asarray(raw_arr[0], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(row)):
        return None
    if np.any(row < -_PROB_EPSILON) or np.any(row > 1.0 + _PROB_EPSILON):
        return None
    return np.clip(row, 0.0, 1.0)


def _align_head_prob(
    prob: float,
    head_warn: float,
    head_crit: float,
    manifest_warn: float,
    manifest_crit: float,
) -> float:
    """Piecewise-linearly align a neural head probability from `(head_warn, head_crit)` to `(manifest_warn, manifest_crit)`."""
    if head_warn == manifest_warn and head_crit == manifest_crit:
        return prob
    if prob <= 0.0:
        return 0.0
    if prob >= 1.0:
        return 1.0
    if prob < head_warn:
        val = (prob / head_warn) * manifest_warn
    elif prob < head_crit:
        val = manifest_warn + ((prob - head_warn) / (head_crit - head_warn)) * (
            manifest_crit - manifest_warn
        )
    else:
        val = manifest_crit + ((prob - head_crit) / (1.0 - head_crit)) * (
            1.0 - manifest_crit
        )
    return float(min(1.0, max(0.0, val)))


def _resolve_per_head_thresholds(
    calibration: Mapping[str, Any],
    manifest_warn: float,
    manifest_crit: float,
) -> dict[str, tuple[float, float]]:
    """Extract validated `(warn_min, critical_min)` per head from `calibration.json`."""
    resolved: dict[str, tuple[float, float]] = {
        "enc": (manifest_warn, manifest_crit),
        "sem": (manifest_warn, manifest_crit),
        "hijack": (manifest_warn, manifest_crit),
    }
    pht = calibration.get("per_head_thresholds")
    if not isinstance(pht, Mapping):
        return resolved

    diag = calibration.get("post_int8_val_diagnostics")
    zero_val_warn = (
        isinstance(diag, Mapping) and diag.get("overall_benign_fpr_at_warn") == 0.0
    )
    zero_val_crit = (
        isinstance(diag, Mapping) and diag.get("overall_benign_fpr_at_critical") == 0.0
    )

    for head in ("enc", "sem", "hijack"):
        entry = pht.get(head)
        if not isinstance(entry, Mapping):
            continue
        raw_w = entry.get("warn_min")
        raw_c = entry.get("critical_min")
        if (
            isinstance(raw_w, bool)
            or isinstance(raw_c, bool)
            or not isinstance(raw_w, (int, float))
            or not isinstance(raw_c, (int, float))
        ):
            continue
        w = float(raw_w)
        c = float(raw_c)
        if not (math.isfinite(w) and math.isfinite(c) and 0.0 < w < c < 1.0):
            continue
        eff_w = manifest_warn if zero_val_warn else w
        eff_c = manifest_crit if zero_val_crit else c
        if 0.0 < eff_w < eff_c < 1.0:
            resolved[head] = (eff_w, eff_c)

    return resolved



class OrtJevPredictor:
    """Zero-PyTorch CPU ONNX Runtime predictor implementing `PredictorProtocol` (`R5`, `R6`)."""

    def __init__(
        self,
        bundle_dir: Path | str,
        intra_op_threads: int = 4,
    ) -> None:
        if (
            isinstance(intra_op_threads, bool)
            or not isinstance(intra_op_threads, int)
            or intra_op_threads < 1
        ):
            raise ValueError("intra_op_threads must be a positive integer >= 1")

        self.bundle_dir: Path = Path(bundle_dir).expanduser()
        self.intra_op_threads: int = intra_op_threads
        self.manifest: BundleManifest = validate_bundle_dir(
            self.bundle_dir, verify_sha256=True
        )
        self.calibration: dict[str, Any] = json.loads(
            (self.bundle_dir / "calibration.json").read_text(encoding="utf-8")
        )
        self._manifest_warn_min: float = float(self.manifest.thresholds["warn_min"])
        self._manifest_critical_min: float = float(
            self.manifest.thresholds["critical_min"]
        )
        self.per_head_thresholds: dict[str, tuple[float, float]] = (
            _resolve_per_head_thresholds(
                self.calibration,
                self._manifest_warn_min,
                self._manifest_critical_min,
            )
        )
        self.tokenizer: Tokenizer = Tokenizer.from_file(
            str(self.bundle_dir / "tokenizer.json")
        )
        self.packer: JevWindowPacker = JevWindowPacker(self.tokenizer, self.manifest)
        self.tier0: Tier0DeterministicPredictor = Tier0DeterministicPredictor(
            hard_secrets_only=True
        )
        self._infer_lock = threading.Lock()

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_op_threads
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        onnx_path = self.bundle_dir / "model_int8.onnx"
        try:
            self._session = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise BundleValidationError(
                f"Failed initializing ONNX Runtime session for {onnx_path}: {exc}"
            ) from exc

        actual_inputs = tuple(inp.name for inp in self._session.get_inputs())
        if actual_inputs != ONNX_INPUT_NAMES:
            raise BundleValidationError(
                f"ONNX input contract mismatch in {onnx_path}: "
                f"expected {ONNX_INPUT_NAMES}, got {actual_inputs}"
            )

        actual_outputs = tuple(out.name for out in self._session.get_outputs())
        if actual_outputs != ONNX_OUTPUT_NAMES:
            raise BundleValidationError(
                f"ONNX output contract mismatch in {onnx_path}: "
                f"expected {ONNX_OUTPUT_NAMES}, got {actual_outputs}"
            )

    def _run_ort_pass(
        self, packed: PackedJevInputs
    ) -> tuple[float, float, float, np.ndarray | None, np.ndarray | None]:
        feed = packed.as_ort_feed()
        try:
            with self._infer_lock:
                raw_outputs = self._session.run(list(ONNX_OUTPUT_NAMES), feed)
        except Exception:
            return 1.0, 1.0, 1.0, None, None

        if len(raw_outputs) != len(ONNX_OUTPUT_NAMES):
            return 1.0, 1.0, 1.0, None, None

        noul_row = _extract_valid_prob_row(raw_outputs[0], 3)
        src_row = _extract_valid_prob_row(raw_outputs[1], self.manifest.max_pchunks + 1)
        arg_row = _extract_valid_prob_row(raw_outputs[2], self.manifest.max_args + 1)
        if noul_row is None or src_row is None or arg_row is None:
            return 1.0, 1.0, 1.0, None, None

        enc_w, enc_c = self.per_head_thresholds["enc"]
        sem_w, sem_c = self.per_head_thresholds["sem"]
        hij_w, hij_c = self.per_head_thresholds["hijack"]
        return (
            _align_head_prob(
                float(noul_row[0]),
                enc_w,
                enc_c,
                self._manifest_warn_min,
                self._manifest_critical_min,
            ),
            _align_head_prob(
                float(noul_row[1]),
                sem_w,
                sem_c,
                self._manifest_warn_min,
                self._manifest_critical_min,
            ),
            _align_head_prob(
                float(noul_row[2]),
                hij_w,
                hij_c,
                self._manifest_warn_min,
                self._manifest_critical_min,
            ),
            src_row,
            arg_row,
        )

    def _decode_model_attribution(
        self,
        packed_src: PackedJevInputs,
        packed_arg: PackedJevInputs,
        private_spans: Sequence[TaintSpan],
        src_row: np.ndarray | None,
        arg_row: np.ndarray | None,
    ) -> tuple[int | None, str | None, str | None]:
        model_span_idx: int | None = None
        model_span_id: str | None = None
        model_arg_key: str | None = None

        if src_row is not None and packed_src.active_pchunk_count > 0:
            active_src = src_row[: packed_src.active_pchunk_count]
            best_slot = int(np.argmax(active_src))
            none_src_prob = float(src_row[self.manifest.max_pchunks])
            if float(active_src[best_slot]) > none_src_prob:
                orig_idx = packed_src.selected_span_indices[best_slot]
                if 0 <= orig_idx < len(private_spans):
                    model_span_idx = orig_idx
                    model_span_id = packed_src.selected_span_ids[best_slot]

        if arg_row is not None and len(packed_arg.active_arg_keys) > 0:
            active_args = arg_row[: len(packed_arg.active_arg_keys)]
            best_arg_slot = int(np.argmax(active_args))
            none_arg_prob = float(arg_row[self.manifest.max_args])
            if float(active_args[best_arg_slot]) > none_arg_prob:
                model_arg_key = packed_arg.active_arg_keys[best_arg_slot]

        return model_span_idx, model_span_id, model_arg_key

    def _evaluate_core(
        self,
        intent: str | None,
        private_spans: list[TaintSpan],
        untrusted_spans: list[TaintSpan],
        snapshot: LedgerSnapshot,
        call: CandidateCall,
        *,
        tier0_result: DecisionScores | None = None,
    ) -> DecisionScores:
        # 1. Full-ledger deterministic Tier-0 floor (zero truncation bypass — C1 / R5)
        tier0_scores = (
            tier0_result
            if tier0_result is not None
            else self.tier0.evaluate(
                intent=intent,
                private_spans=private_spans,
                untrusted_spans=untrusted_spans,
                call=call,
            )
        )

        # 2. Default 512-token pre-selected Jev window inference
        packed = self.packer.pack(snapshot, call, max_bucket=DEFAULT_INFERENCE_BUCKET)
        packed_src = packed
        packed_arg = packed
        m_enc, m_sem, m_hijack, src_row, arg_row = self._run_ort_pass(packed)
        m_max = max(m_enc, m_sem, m_hijack)

        warn_min = float(self.manifest.thresholds["warn_min"])
        critical_min = float(self.manifest.thresholds["critical_min"])
        escalation_cap = min(
            _ESC_MAX_BUCKET,
            max(self.manifest.length_buckets),
            self.manifest.max_seq_len,
        )

        # 3. Borderline under truncation: escalate once up to max_bucket=2048 (per-head monotonic floor)
        if (
            packed.was_truncated
            and (_ESC_BORDERLINE_MIN <= m_max < critical_min)
            and escalation_cap > packed.bucket_length
        ):
            packed_esc = self.packer.pack(snapshot, call, max_bucket=escalation_cap)
            if (
                packed_esc.bucket_length > packed.bucket_length
                or packed_esc.unpadded_length > packed.unpadded_length
            ):
                e_enc, e_sem, e_hijack, e_src_row, e_arg_row = self._run_ort_pass(packed_esc)
                e_max = max(e_enc, e_sem, e_hijack)
                if max(e_enc, e_sem) >= max(m_enc, m_sem):
                    packed_src = packed_esc
                    src_row = e_src_row
                if e_max >= m_max:
                    packed_arg = packed_esc
                    arg_row = e_arg_row
                m_enc = max(m_enc, e_enc)
                m_sem = max(m_sem, e_sem)
                m_hijack = max(m_hijack, e_hijack)
                m_max = max(m_enc, m_sem, m_hijack)

        # 4. Combine Tier-0 floor with calibrated Jev probabilities
        p_enc = max(tier0_scores.p_enc, m_enc)
        p_sem = max(tier0_scores.p_sem, m_sem)
        p_hijack = max(tier0_scores.p_hijack, m_hijack)
        p_max = max(p_enc, p_sem, p_hijack)
        exfil_max = max(p_enc, p_sem)
        model_exfil_max = max(m_enc, m_sem)
        tier0_exfil_max = max(tier0_scores.p_enc, tier0_scores.p_sem)

        model_span_idx, model_span_id, model_arg_key = self._decode_model_attribution(
            packed_src, packed_arg, private_spans, src_row, arg_row
        )
        if model_exfil_max < warn_min:
            model_span_idx = None
            model_span_id = None
        if m_max < warn_min:
            model_arg_key = None

        attributed_span_idx: int | None = None
        attributed_span_id: str | None = None
        attributed_arg_key: str | None = None

        if p_max >= warn_min:
            if exfil_max >= warn_min:
                if (
                    tier0_exfil_max >= model_exfil_max
                    and tier0_scores.attributed_span_idx is not None
                ):
                    attributed_span_idx = tier0_scores.attributed_span_idx
                    attributed_span_id = tier0_scores.attributed_span_id
                elif model_span_idx is not None and model_exfil_max > tier0_exfil_max:
                    attributed_span_idx = model_span_idx
                    attributed_span_id = model_span_id
                else:
                    attributed_span_idx = tier0_scores.attributed_span_idx
                    attributed_span_id = tier0_scores.attributed_span_id

            if tier0_scores.p_max >= m_max and tier0_scores.attributed_arg_key is not None:
                attributed_arg_key = tier0_scores.attributed_arg_key
            elif model_arg_key is not None and m_max > tier0_scores.p_max:
                attributed_arg_key = model_arg_key
            else:
                attributed_arg_key = tier0_scores.attributed_arg_key

        return DecisionScores(
            p_enc=round(p_enc, 6),
            p_sem=round(p_sem, 6),
            p_hijack=round(p_hijack, 6),
            attributed_span_idx=attributed_span_idx,
            attributed_span_id=attributed_span_id,
            attributed_arg_key=attributed_arg_key,
        )

    def evaluate_snapshot(
        self,
        snapshot: LedgerSnapshot,
        call: CandidateCall,
        *,
        tier0_result: DecisionScores | None = None,
    ) -> DecisionScores:
        """Evaluate `(snapshot, call)` directly without re-copying `LedgerSnapshot`."""
        return self._evaluate_core(
            intent=snapshot.intent,
            private_spans=list(snapshot.private_spans),
            untrusted_spans=list(snapshot.untrusted_spans),
            snapshot=snapshot,
            call=call,
            tier0_result=tier0_result,
        )

    def evaluate(
        self,
        intent: str | None,
        private_spans: list[TaintSpan],
        untrusted_spans: list[TaintSpan],
        call: CandidateCall,
        *,
        tier0_result: DecisionScores | None = None,
    ) -> DecisionScores:
        """Evaluate `(intent, private_spans, untrusted_spans, call)` per `PredictorProtocol`."""
        snapshot = LedgerSnapshot(
            intent=intent,
            private_spans=tuple(private_spans),
            untrusted_spans=tuple(untrusted_spans),
        )
        return self._evaluate_core(
            intent=intent,
            private_spans=private_spans,
            untrusted_spans=untrusted_spans,
            snapshot=snapshot,
            call=call,
            tier0_result=tier0_result,
        )

    predict = evaluate
