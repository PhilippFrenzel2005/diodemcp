"""Zero-PyTorch ONNX Runtime Jev cross-encoder predictor with Tier-0 floor and bucket escalation."""

from __future__ import annotations

from collections.abc import Sequence
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


def _sanitize_probability_fail_closed(raw_value: Any) -> float:
    """Coerce `NaN`, `+/-inf`, or out-of-range model outputs fail-closed to `1.0` (`C5.3`)."""
    if isinstance(raw_value, bool):
        return 1.0
    try:
        val = float(raw_value)
    except (TypeError, ValueError):
        return 1.0
    if math.isnan(val) or math.isinf(val):
        return 1.0
    if val < -_PROB_EPSILON or val > 1.0 + _PROB_EPSILON:
        return 1.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _extract_valid_pointer_row(raw_tensor: Any, expected_cols: int) -> np.ndarray | None:
    """Return 1D finite probability row `[expected_cols]` or `None` if malformed/non-finite."""
    try:
        arr = np.asarray(raw_tensor, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != expected_cols:
        return None
    row = arr[0]
    if not np.all(np.isfinite(row)):
        return None
    if np.any(row < -1e-4) or np.any(row > 1.0 + 1e-4):
        return None
    return row


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

        self.bundle_dir: Path = Path(bundle_dir)
        self.intra_op_threads: int = intra_op_threads
        self.manifest: BundleManifest = validate_bundle_dir(
            self.bundle_dir, verify_sha256=True
        )
        self.calibration: dict[str, Any] = json.loads(
            (self.bundle_dir / "calibration.json").read_text(encoding="utf-8")
        )
        self.tokenizer: Tokenizer = Tokenizer.from_file(
            str(self.bundle_dir / "tokenizer.json")
        )
        self.packer: JevWindowPacker = JevWindowPacker(self.tokenizer, self.manifest)
        self.tier0: Tier0DeterministicPredictor = Tier0DeterministicPredictor()
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

        try:
            noul_arr = np.asarray(raw_outputs[0], dtype=np.float64)
        except (TypeError, ValueError):
            return 1.0, 1.0, 1.0, None, None

        if noul_arr.ndim != 2 or noul_arr.shape[0] < 1 or noul_arr.shape[1] != 3:
            return 1.0, 1.0, 1.0, None, None

        m_enc = _sanitize_probability_fail_closed(noul_arr[0, 0])
        m_sem = _sanitize_probability_fail_closed(noul_arr[0, 1])
        m_hijack = _sanitize_probability_fail_closed(noul_arr[0, 2])

        src_row = _extract_valid_pointer_row(raw_outputs[1], self.manifest.max_pchunks + 1)
        arg_row = _extract_valid_pointer_row(raw_outputs[2], self.manifest.max_args + 1)
        return m_enc, m_sem, m_hijack, src_row, arg_row

    def _decode_model_attribution(
        self,
        packed: PackedJevInputs,
        private_spans: Sequence[TaintSpan],
        src_row: np.ndarray | None,
        arg_row: np.ndarray | None,
    ) -> tuple[int | None, str | None, str | None]:
        model_span_idx: int | None = None
        model_span_id: str | None = None
        model_arg_key: str | None = None

        if src_row is not None and packed.active_pchunk_count > 0:
            active_src = src_row[: packed.active_pchunk_count]
            best_slot = int(np.argmax(active_src))
            none_src_prob = float(src_row[self.manifest.max_pchunks])
            if float(active_src[best_slot]) > none_src_prob:
                orig_idx = packed.selected_span_indices[best_slot]
                if 0 <= orig_idx < len(private_spans):
                    model_span_idx = orig_idx
                    model_span_id = packed.selected_span_ids[best_slot]

        if arg_row is not None and len(packed.active_arg_keys) > 0:
            active_args = arg_row[: len(packed.active_arg_keys)]
            best_arg_slot = int(np.argmax(active_args))
            none_arg_prob = float(arg_row[self.manifest.max_args])
            if float(active_args[best_arg_slot]) > none_arg_prob:
                model_arg_key = packed.active_arg_keys[best_arg_slot]

        return model_span_idx, model_span_id, model_arg_key

    def _evaluate_core(
        self,
        intent: str | None,
        private_spans: list[TaintSpan],
        untrusted_spans: list[TaintSpan],
        snapshot: LedgerSnapshot,
        call: CandidateCall,
    ) -> DecisionScores:
        # 1. Full-ledger deterministic Tier-0 floor (zero truncation bypass — C1 / R5)
        tier0_scores = self.tier0.evaluate(
            intent=intent,
            private_spans=private_spans,
            untrusted_spans=untrusted_spans,
            call=call,
        )

        # 2. Default 512-token pre-selected Jev window inference
        packed = self.packer.pack(snapshot, call, max_bucket=DEFAULT_INFERENCE_BUCKET)
        m_enc, m_sem, m_hijack, src_row, arg_row = self._run_ort_pass(packed)
        m_max = max(m_enc, m_sem, m_hijack)

        warn_min = float(self.manifest.thresholds["warn_min"])
        critical_min = float(self.manifest.thresholds["critical_min"])
        escalation_cap = min(
            _ESC_MAX_BUCKET,
            max(self.manifest.length_buckets),
            self.manifest.max_seq_len,
        )

        # 3. Borderline under truncation: escalate once up to max_bucket=2048 (monotonic floor)
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
                if e_max >= m_max:
                    packed = packed_esc
                    src_row = e_src_row
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

        model_span_idx, model_span_id, model_arg_key = self._decode_model_attribution(
            packed, private_spans, src_row, arg_row
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
                    tier0_scores.p_enc >= model_exfil_max
                    and tier0_scores.attributed_span_idx is not None
                ):
                    attributed_span_idx = tier0_scores.attributed_span_idx
                    attributed_span_id = tier0_scores.attributed_span_id
                elif model_span_idx is not None:
                    attributed_span_idx = model_span_idx
                    attributed_span_id = model_span_id
                else:
                    attributed_span_idx = tier0_scores.attributed_span_idx
                    attributed_span_id = tier0_scores.attributed_span_id

            if tier0_scores.p_max >= m_max and tier0_scores.attributed_arg_key is not None:
                attributed_arg_key = tier0_scores.attributed_arg_key
            elif model_arg_key is not None:
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
    ) -> DecisionScores:
        """Evaluate `(snapshot, call)` directly without re-copying `LedgerSnapshot`."""
        return self._evaluate_core(
            intent=snapshot.intent,
            private_spans=list(snapshot.private_spans),
            untrusted_spans=list(snapshot.untrusted_spans),
            snapshot=snapshot,
            call=call,
        )

    def evaluate(
        self,
        intent: str | None,
        private_spans: list[TaintSpan],
        untrusted_spans: list[TaintSpan],
        call: CandidateCall,
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
        )
