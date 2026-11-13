"""Length-bucketed 2-stage curriculum trainer, atomic Drive checkpoint resume, and evaluation suite.

Implements Rulings ``RA3`` and ``I11``:
* **Length-bucketed curriculum** (``512 -> 1024 -> 4096``) with dynamic bucket alignment/padding.
* **Stage 1 (Representation + Pointer Supervision):** optimises ``compute_jev_loss`` with active
  masked pointer cross-entropy over ``src_pointer`` (33 slots) and ``arg_pointer`` (17 slots) plus
  strictly proper ``BCE + Brier`` on the 3 ``noul_head`` queries (``<|q_enc|>``, ``<|q_sem|>``,
  ``<|q_hijack|>``).
* **Stage 2 (Strictly Proper Calibration + Hard-Negative Refinement):** applies ``3.0x`` sample
  weights on false positives on ``BENIGN_TWIN`` / ``Trifecta-Safe`` benign rows and ``3.0x`` sample
  weights on false negatives on ``T1`` / ``T2`` obfuscated leaks, followed by unweighted post-hoc
  temperature scaling (:func:`fit_temperature_scaling`) on ``val_calib``.
* **Atomic Google Drive checkpointing & exact mid-curriculum resume:** writes via a sibling ``.tmp``
  file + ``os.fsync`` + ``os.replace`` and restores ``epoch``, ``stage``, ``bucket_len``,
  ``global_step``, ``model_state_dict``, ``optimizer_state_dict``, ``scheduler_state_dict``,
  ``best_val_metric``, and ``rng_state`` bit-for-bit.
* **Evaluation suite (:func:`evaluate_jev_checkpoint`):** per-threat (``T1..T7``) recall, symmetric
  ``FlipJev`` (attack recall + benign twin DoS FPR), ``Trifecta-Safe`` FPR, pointer top-1 accuracy on
  active leaks, and per-head/overall ECE + Brier via :func:`temperature_calibration_report`.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Final

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import ModernBertConfig, Qwen3Config

from diodemodel.contract import (
    CANONICAL_SEGMENT_IDS,
    ONNX_INPUT_SPECS,
    ROLE_PAD,
    ROLE_QUERY,
    ROLE_STATE,
)
from diodemodel.dataset_packer import PackedJevRow
from diodemodel.losses_rlcd import (
    DEFAULT_IGNORE_INDEX,
    JevLossConfig,
    brier_loss,
    compute_jev_loss,
    fit_temperature_scaling,
    temperature_calibration_report,
)
from diodemodel.modeling_jev import (
    ARG_SLOTS,
     ModernBERTJevModel,
    NanoJevCausalModel,
    SRC_SLOTS,
    _JevModelBase,
)
from diodemodel.mutators import ALL_THREAT_CATEGORIES

CURRICULUM_BUCKETS: Final[tuple[int, ...]] = (512, 1024, 4096)
HARD_NEGATIVE_WEIGHT: Final[float] = 3.0
INPUT_TENSOR_NAMES: Final[tuple[str, ...]] = tuple(ONNX_INPUT_SPECS.keys())


@dataclass
class JevTrainerConfig:
    """Configuration for 2-stage length-bucketed curriculum training."""

    curriculum_buckets: tuple[int, ...] = CURRICULUM_BUCKETS
    batch_size: int = 8
    grad_accum_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    stage1_epochs: int = 1
    stage2_epochs: int = 1
    stage1_loss_config: JevLossConfig = field(
        default_factory=lambda: JevLossConfig(
            bce_weight=1.0,
            brier_weight=0.5,
            head_weights=(1.0, 1.0, 1.0),
            src_pointer_weight=1.0,
            arg_pointer_weight=1.0,
        )
    )
    stage2_loss_config: JevLossConfig = field(
        default_factory=lambda: JevLossConfig(
            bce_weight=1.0,
            brier_weight=0.5,
            head_weights=(1.0, 1.0, 1.0),
            src_pointer_weight=0.25,
            arg_pointer_weight=0.25,
        )
    )
    hard_negative_weight: float = HARD_NEGATIVE_WEIGHT
    fp_threshold: float = 0.5
    fn_threshold: float = 0.5
    tool_name_dropout: float = 0.30
    arg_key_dropout: float = 0.15
    head_lr_mult: float = 12.0
    warmup_ratio: float = 0.05
    special_token_ids: Mapping[str, int] | None = None
    tool_name_replacement_ids: Sequence[int] | None = None
    vocab_size: int | None = None
    pad_token_id: int | None = None
    save_every_steps: int | None = 500
    use_bf16: bool = False
    gradient_checkpointing: bool = False
    seed: int = 42
    max_steps: int | None = None
    max_steps_per_bucket: int | None = None
    checkpoint_dir: Path | str | None = None
    status_log_path: Path | str | None = None
    device: str | torch.device = "cpu"


CurriculumConfig = JevTrainerConfig


# ---------------------------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------------------------
def log_train_event(status_log_path: Path | str | None, event: str, **payload: Any) -> None:
    """Append a structured JSONL telemetry line and flush immediately for Drive resilience."""
    if status_log_path is None:
        return
    path = Path(status_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()


# ---------------------------------------------------------------------------------------------
# Dataset loading, dynamic bucket alignment & batch collation
# ---------------------------------------------------------------------------------------------
def _normalise_row_dict(raw: Mapping[str, Any] | PackedJevRow) -> dict[str, Any]:
    """Normalise a ``PackedJevRow`` or Parquet/in-memory dict so canonical and alias keys coexist."""
    if isinstance(raw, PackedJevRow):
        d: dict[str, Any] = {
            "pair_id": "packed_row",
            "seed_id": "packed_seed",
            "repo_or_source": "synthetic/packed",
            "group_key": "synthetic/packed",
            "split": "train",
            "category": "T1",
            "threat_category": "T1",
            "attack_family": "packed",
            "variant_kind": "packed",
            "flipjev_applied": False,
            "intent_masked": False,
            "is_trifecta_benign": max(raw.y_enc, raw.y_sem, raw.y_hijack) < 0.5,
            "bucket_length": int(raw.bucket_length),
            "unpadded_length": int(raw.unpadded_length),
            "none_position": int(raw.none_position),
            "was_truncated": bool(raw.was_truncated),
            "input_ids": [int(x) for x in raw.input_ids],
            "seg_ids": [int(x) for x in raw.seg_ids],
            "token_roles": [int(x) for x in raw.token_roles],
            "query_positions": [int(x) for x in raw.query_positions],
            "pchunk_positions": [int(x) for x in raw.pchunk_positions],
            "pchunk_valid_mask": [bool(x) for x in raw.pchunk_valid_mask],
            "arg_positions": [int(x) for x in raw.arg_positions],
            "arg_valid_mask": [bool(x) for x in raw.arg_valid_mask],
            "y_enc": float(raw.y_enc),
            "y_sem": float(raw.y_sem),
            "y_hijack": float(raw.y_hijack),
            "target_src_idx": int(raw.target_src_idx),
            "src_target": int(raw.target_src_idx),
            "target_arg_idx": int(raw.target_arg_idx),
            "arg_target": int(raw.target_arg_idx),
        }
        return d

    d = dict(raw)
    input_ids = [int(x) for x in d["input_ids"]]
    seg_ids = [int(x) for x in d["seg_ids"]]
    token_roles = [int(x) for x in d["token_roles"]]
    q_pos = [int(x) for x in d["query_positions"]]
    p_pos = [int(x) for x in d["pchunk_positions"]]
    p_val = [bool(x) for x in d["pchunk_valid_mask"]]
    a_pos = [int(x) for x in d["arg_positions"]]
    a_val = [bool(x) for x in d["arg_valid_mask"]]

    y_enc = float(d.get("y_enc", 0.0))
    y_sem = float(d.get("y_sem", 0.0))
    y_hijack = float(d.get("y_hijack", 0.0))

    src_t = int(d.get("target_src_idx", d.get("src_target", SRC_SLOTS - 1)))
    arg_t = int(d.get("target_arg_idx", d.get("arg_target", ARG_SLOTS - 1)))

    cat = str(d.get("category", d.get("threat_category", "T1")))
    threat_cat = str(d.get("threat_category", cat))
    attack_fam = str(d.get("attack_family", d.get("variant_kind", "programmatic")))
    variant_k = str(d.get("variant_kind", attack_fam))

    non_pad = [i for i, r in enumerate(token_roles) if r != ROLE_PAD]
    unpadded_len = int(d.get("unpadded_length", (non_pad[-1] + 1) if non_pad else len(input_ids)))
    none_pos = int(d.get("none_position", p_pos[SRC_SLOTS - 1] if len(p_pos) == SRC_SLOTS else max(0, unpadded_len - 6)))

    is_benign = max(y_enc, y_sem, y_hijack) < 0.5
    default_trifecta = is_benign and (
        "benign" in variant_k.lower()
        or "twin" in variant_k.lower()
        or "trifecta" in variant_k.lower()
        or any(p_val[:-1])
    )
    is_trifecta = bool(d.get("is_trifecta_benign", default_trifecta))
    flipjev = bool(d.get("flipjev_applied", "flipjev" in variant_k.lower() or "flipjev" in attack_fam.lower()))

    return {
        **d,
        "pair_id": str(d.get("pair_id", "row")),
        "seed_id": str(d.get("seed_id", "seed")),
        "repo_or_source": str(d.get("repo_or_source", "synthetic")),
        "group_key": str(d.get("group_key", "synthetic")),
        "split": str(d.get("split", "train")),
        "category": cat,
        "threat_category": threat_cat,
        "attack_family": attack_fam,
        "variant_kind": variant_k,
        "flipjev_applied": flipjev,
        "intent_masked": bool(d.get("intent_masked", False)),
        "is_trifecta_benign": is_trifecta,
        "bucket_length": int(d.get("bucket_length", len(input_ids))),
        "unpadded_length": unpadded_len,
        "none_position": none_pos,
        "was_truncated": bool(d.get("was_truncated", False)),
        "input_ids": input_ids,
        "seg_ids": seg_ids,
        "token_roles": token_roles,
        "query_positions": q_pos,
        "pchunk_positions": p_pos,
        "pchunk_valid_mask": p_val,
        "arg_positions": a_pos,
        "arg_valid_mask": a_val,
        "y_enc": y_enc,
        "y_sem": y_sem,
        "y_hijack": y_hijack,
        "target_src_idx": src_t,
        "src_target": src_t,
        "target_arg_idx": arg_t,
        "arg_target": arg_t,
    }


def load_packed_jev_dataset(
    source: Path | str | Sequence[Mapping[str, Any] | PackedJevRow],
    *,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Load packed Jev rows from a Parquet file, JSONL file, or in-memory sequence."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if path.suffix.lower() == ".parquet":
            tbl = pq.read_table(path)
            if max_rows is not None:
                tbl = tbl.slice(0, int(max_rows))
            raw_list = tbl.to_pylist()
        else:
            raw_list = []
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        raw_list.append(json.loads(line))
                        if max_rows is not None and len(raw_list) >= max_rows:
                            break
    else:
        raw_list = list(source[:max_rows] if max_rows is not None else source)

    return [_normalise_row_dict(r) for r in raw_list]


def align_row_to_bucket(
    row: Mapping[str, Any] | PackedJevRow,
    bucket_len: int,
    *,
    pad_token_id: int | None = None,
) -> dict[str, Any]:
    """Pad or clamp/align a packed Jev row to ``bucket_len`` while preserving all Jev invariants."""
    if bucket_len < 16:
        raise ValueError(f"bucket_len must be >= 16, got {bucket_len}")
    d = _normalise_row_dict(row)
    input_ids = list(d["input_ids"])
    seg_ids = list(d["seg_ids"])
    token_roles = list(d["token_roles"])
    q_pos = list(d["query_positions"])
    p_pos = list(d["pchunk_positions"])
    p_val = list(d["pchunk_valid_mask"])
    a_pos = list(d["arg_positions"])
    a_val = list(d["arg_valid_mask"])

    if pad_token_id is not None:
        pad_id = int(pad_token_id)
    else:
        pad_candidates = [input_ids[i] for i, r in enumerate(token_roles) if r == ROLE_PAD]
        pad_id = int(pad_candidates[0]) if pad_candidates else int(d.get("pad_token_id", 0))

    non_pad = [i for i, r in enumerate(token_roles) if r != ROLE_PAD]
    unpadded_len = (non_pad[-1] + 1) if non_pad else 0
    orig_none_pos = int(d["none_position"])
    was_truncated = bool(d["was_truncated"])

    y_enc = float(d["y_enc"])
    y_sem = float(d["y_sem"])
    y_hijack = float(d["y_hijack"])
    src_target = int(d["src_target"])
    arg_target = int(d["arg_target"])

    if unpadded_len <= bucket_len:
        ids_out = input_ids[:unpadded_len]
        segs_out = seg_ids[:unpadded_len]
        roles_out = token_roles[:unpadded_len]
        pad_needed = bucket_len - len(ids_out)
        if pad_needed > 0:
            ids_out.extend([pad_id] * pad_needed)
            segs_out.extend([0] * pad_needed)
            roles_out.extend([ROLE_PAD] * pad_needed)
        none_pos_out = min(max(0, orig_none_pos), bucket_len - 6)
        q_pos_out = [min(max(0, p), bucket_len - 1) for p in q_pos]
        p_pos_out = [min(max(0, p), bucket_len - 1) for p in p_pos]
        p_val_out = list(p_val)
        a_pos_out = [min(max(0, p), bucket_len - 1) for p in a_pos]
        a_val_out = list(a_val)
        p_pos_out[SRC_SLOTS - 1] = none_pos_out
        p_val_out[SRC_SLOTS - 1] = True
        a_pos_out[ARG_SLOTS - 1] = none_pos_out
        a_val_out[ARG_SLOTS - 1] = True
    else:
        # Clamp long sequence into bucket_len while preserving [SEP], <|none|>, and 5 query tokens at the end,
        # and preserving the closing [SEP] (seg_id == 1) of <|priv|> when clamping cuts inside seg_ids in {1, 2}.
        was_truncated = True
        prefix_len = bucket_len - 7
        sep_idx = min(max(0, orig_none_pos - 1), len(input_ids) - 1)
        none_idx = min(max(0, orig_none_pos), len(input_ids) - 1)
        q_indices = [min(max(0, p), len(input_ids) - 1) for p in q_pos]
        tail_indices = [sep_idx, none_idx, *q_indices]

        orig_priv_indices = [
            i
            for i in range(min(unpadded_len, len(token_roles)))
            if token_roles[i] == ROLE_STATE and seg_ids[i] in (1, 2)
        ]
        orig_priv_sep_idx = max(orig_priv_indices) if orig_priv_indices else -1

        ids_out = input_ids[:prefix_len] + [input_ids[idx] for idx in tail_indices]
        segs_out = seg_ids[:prefix_len] + [seg_ids[idx] for idx in tail_indices]
        roles_out = token_roles[:prefix_len] + [token_roles[idx] for idx in tail_indices]

        none_pos_out = bucket_len - 6
        segs_out[none_pos_out - 1] = CANONICAL_SEGMENT_IDS["CANDIDATE_ACTION"]
        roles_out[none_pos_out - 1] = ROLE_STATE
        segs_out[none_pos_out] = CANONICAL_SEGMENT_IDS["CANDIDATE_ACTION"]
        roles_out[none_pos_out] = ROLE_STATE

        q_pos_out = [bucket_len - 5 + k for k in range(5)]
        for qp in q_pos_out:
            segs_out[qp] = CANONICAL_SEGMENT_IDS["DECISION_QUERY"]
            roles_out[qp] = ROLE_QUERY

        surviving_priv = [
            i for i in range(prefix_len) if roles_out[i] == ROLE_STATE and segs_out[i] in (1, 2)
        ]
        if surviving_priv and orig_priv_sep_idx >= prefix_len:
            priv_sep_out = surviving_priv[-1]
            sep_tok_val = (
                input_ids[orig_priv_sep_idx]
                if 0 <= orig_priv_sep_idx < len(input_ids)
                else input_ids[sep_idx]
            )
            ids_out[priv_sep_out] = sep_tok_val
            segs_out[priv_sep_out] = CANONICAL_SEGMENT_IDS["PRIVATE_DATA"]
            roles_out[priv_sep_out] = ROLE_STATE
            priv_pchunk_limit = priv_sep_out
            arg_limit = priv_sep_out if priv_sep_out == prefix_len - 1 else prefix_len
        else:
            priv_pchunk_limit = (
                min(prefix_len, orig_priv_sep_idx) if orig_priv_sep_idx >= 0 else prefix_len
            )
            arg_limit = prefix_len

        p_pos_out = [none_pos_out] * SRC_SLOTS
        p_val_out = [False] * SRC_SLOTS
        for i in range(SRC_SLOTS - 1):
            if p_val[i] and 0 <= p_pos[i] < priv_pchunk_limit:
                p_pos_out[i] = int(p_pos[i])
                p_val_out[i] = True
        p_pos_out[SRC_SLOTS - 1] = none_pos_out
        p_val_out[SRC_SLOTS - 1] = True

        a_pos_out = [none_pos_out] * ARG_SLOTS
        a_val_out = [False] * ARG_SLOTS
        for j in range(ARG_SLOTS - 1):
            if a_val[j] and 0 <= a_pos[j] < arg_limit:
                a_pos_out[j] = int(a_pos[j])
                a_val_out[j] = True
        a_pos_out[ARG_SLOTS - 1] = none_pos_out
        a_val_out[ARG_SLOTS - 1] = True
        unpadded_len = bucket_len

    # Preserve binary labels y_enc, y_sem, y_hijack intact; if an attributed candidate pointer slot
    # was evicted by clamping (or is invalid), map it to DEFAULT_IGNORE_INDEX (-100) so pointer CE
    # ignores that row without falsely supervising <|none|> (32 / 16) on an attack row.
    if src_target != DEFAULT_IGNORE_INDEX:
        if not (0 <= src_target < SRC_SLOTS and p_val_out[src_target]):
            src_target = DEFAULT_IGNORE_INDEX
    if arg_target != DEFAULT_IGNORE_INDEX:
        if not (0 <= arg_target < ARG_SLOTS and a_val_out[arg_target]):
            arg_target = DEFAULT_IGNORE_INDEX

    return {
        **d,
        "bucket_length": bucket_len,
        "unpadded_length": min(unpadded_len, bucket_len),
        "none_position": none_pos_out,
        "was_truncated": was_truncated,
        "input_ids": ids_out,
        "seg_ids": segs_out,
        "token_roles": roles_out,
        "query_positions": q_pos_out,
        "pchunk_positions": p_pos_out,
        "pchunk_valid_mask": p_val_out,
        "arg_positions": a_pos_out,
        "arg_valid_mask": a_val_out,
        "y_enc": y_enc,
        "y_sem": y_sem,
        "y_hijack": y_hijack,
        "target_src_idx": src_target,
        "src_target": src_target,
        "target_arg_idx": arg_target,
        "arg_target": arg_target,
    }


def _apply_segment4_invariance_dropout(
    aligned_rows: list[dict[str, Any]],
    *,
    tool_name_dropout: float = 0.0,
    arg_key_dropout: float = 0.0,
    special_token_ids: Mapping[str, int] | None = None,
    tool_name_replacement_ids: Sequence[int] | None = None,
    vocab_size: int | None = None,
    pad_token_id: int | None = None,
    rng: random.Random | np.random.Generator | None = None,
) -> None:
    """Apply tool-name and optional arg-key invariance dropout strictly within ``seg_ids == 4``.

    Never modifies ``seg_ids == 3`` (``UNTRUSTED_DATA``), special structural tokens
    (``<|call|>``, ``<|arg|>``, ``[SEP]``, ``<|none|>``, query tokens), or emits special / ``[UNK]`` IDs.
    """
    p_tool = max(0.0, min(1.0, float(tool_name_dropout)))
    p_arg = max(0.0, min(1.0, float(arg_key_dropout)))
    if p_tool <= 0.0 and p_arg <= 0.0:
        return

    special_set: set[int] = (
        {int(v) for v in special_token_ids.values() if v is not None}
        if special_token_ids is not None
        else set()
    )
    excluded: set[int] = set(special_set) | {0, 1, 2, 3}
    if pad_token_id is not None:
        excluded.add(int(pad_token_id))
    else:
        for r in aligned_rows:
            if r.get("pad_token_id") is not None:
                excluded.add(int(r["pad_token_id"]))

    explicit_pool = [
        int(t) for t in (tool_name_replacement_ids or ()) if int(t) not in excluded and int(t) > 0
    ]

    low_ord = 200
    high_ord = 0
    small_pool: list[int] = []
    if not explicit_pool:
        high_specials = [v for v in special_set if v >= 300]
        if high_specials:
            vocab_base = min(high_specials)
            if vocab_size is not None:
                vocab_base = min(vocab_base, int(vocab_size))
        elif vocab_size is not None:
            vocab_base = int(vocab_size)
        else:
            max_seen = 0
            for r in aligned_rows:
                for t in r["input_ids"]:
                    ti = int(t)
                    if ti not in excluded and ti > max_seen:
                        max_seen = ti
            vocab_base = (max_seen + 1) if max_seen >= 300 else max(64, max_seen + 1)

        if vocab_base >= 400:
            low_ord = 200
            high_ord = max(low_ord + 1, vocab_base - 100)
        else:
            small_pool = [i for i in range(11, vocab_base) if i not in excluded]
            if not small_pool:
                small_pool = [i for i in range(4, vocab_base) if i not in excluded]

    if not explicit_pool and high_ord <= low_ord and not small_pool:
        return

    def _draw_float() -> float:
        if rng is None:
            return random.random()
        if isinstance(rng, np.random.Generator):
            return float(rng.random())
        return float(rng.random())

    def _draw_int(lo: int, hi_excl: int) -> int:
        if rng is None:
            return random.randrange(lo, hi_excl)
        if isinstance(rng, np.random.Generator):
            return int(rng.integers(lo, hi_excl))
        return int(rng.randrange(lo, hi_excl))

    def _sample_replacement() -> int | None:
        if explicit_pool:
            return explicit_pool[_draw_int(0, len(explicit_pool))]
        if high_ord > low_ord:
            for _ in range(16):
                cand = _draw_int(low_ord, high_ord)
                if cand not in excluded:
                    return cand
            return low_ord if low_ord not in excluded else None
        if small_pool:
            return small_pool[_draw_int(0, len(small_pool))]
        return None

    call_id = int(special_token_ids["<|call|>"]) if special_token_ids and "<|call|>" in special_token_ids else None
    arg_id = int(special_token_ids["<|arg|>"]) if special_token_ids and "<|arg|>" in special_token_ids else None
    sep_id = int(special_token_ids["[SEP]"]) if special_token_ids and "[SEP]" in special_token_ids else None

    seg_cand = CANONICAL_SEGMENT_IDS["CANDIDATE_ACTION"]

    for r in aligned_rows:
        ids = list(r["input_ids"])
        segs = r["seg_ids"]
        roles = r["token_roles"]
        none_pos = int(r["none_position"])
        a_pos_list = r["arg_positions"]
        a_val_list = r["arg_valid_mask"]

        seg4_state = [
            p
            for p in range(min(none_pos, len(ids)))
            if int(segs[p]) == seg_cand and int(roles[p]) == ROLE_STATE
        ]
        if not seg4_state:
            continue

        call_pos: int | None = None
        if call_id is not None:
            for p in seg4_state:
                if int(ids[p]) == call_id:
                    call_pos = p
                    break
        else:
            call_pos = seg4_state[0]

        arg_marker_set: set[int] = set()
        for j in range(max(0, len(a_pos_list) - 1)):
            if bool(a_val_list[j]):
                ap = int(a_pos_list[j])
                if 0 <= ap < none_pos and int(segs[ap]) == seg_cand:
                    arg_marker_set.add(ap)
        if arg_id is not None:
            for p in seg4_state:
                if int(ids[p]) == arg_id:
                    arg_marker_set.add(p)

        sep_pos: int | None = None
        if sep_id is not None:
            for p in seg4_state:
                if (call_pos is None or p > call_pos) and int(ids[p]) == sep_id:
                    sep_pos = p
                    break
        elif call_pos is not None and (none_pos - 1) > call_pos and int(segs[none_pos - 1]) == seg_cand:
            sep_pos = none_pos - 1

        # 1. Tool-name span replacement strictly inside (call_pos + 1 .. first_arg_or_sep_pos - 1)
        if p_tool > 0.0 and call_pos is not None and _draw_float() < p_tool:
            boundaries = [p for p in arg_marker_set if p > call_pos]
            boundaries.append(sep_pos if sep_pos is not None else none_pos)
            first_boundary = min(boundaries)
            for p in range(call_pos + 1, first_boundary):
                if int(segs[p]) == seg_cand and int(roles[p]) == ROLE_STATE and int(ids[p]) not in special_set:
                    rep = _sample_replacement()
                    if rep is not None:
                        ids[p] = rep

        # 2. Optional arg-key token replacement immediately after <|arg|> markers in seg_ids == 4
        if p_arg > 0.0 and arg_marker_set:
            end_limit = sep_pos if sep_pos is not None else none_pos
            for ap in sorted(arg_marker_set):
                key_pos = ap + 1
                if (
                    key_pos < end_limit
                    and key_pos not in arg_marker_set
                    and int(segs[key_pos]) == seg_cand
                    and int(roles[key_pos]) == ROLE_STATE
                    and int(ids[key_pos]) not in special_set
                    and _draw_float() < p_arg
                ):
                    rep = _sample_replacement()
                    if rep is not None:
                        ids[key_pos] = rep

        r["input_ids"] = ids


def collate_jev_batch(
    rows: Sequence[Mapping[str, Any] | PackedJevRow],
    *,
    bucket_len: int = 512,
    pad_token_id: int | None = None,
    device: str | torch.device = "cpu",
    tool_name_dropout: float = 0.0,
    arg_key_dropout: float = 0.0,
    special_token_ids: Mapping[str, int] | None = None,
    tool_name_replacement_ids: Sequence[int] | None = None,
    vocab_size: int | None = None,
    rng: random.Random | np.random.Generator | None = None,
) -> dict[str, Any]:
    """Collate and align a batch of rows to ``bucket_len`` on ``device``."""
    if not rows:
        raise ValueError("collate_jev_batch requires at least 1 row")
    dev = torch.device(device)
    aligned = [align_row_to_bucket(r, bucket_len, pad_token_id=pad_token_id) for r in rows]
    if tool_name_dropout > 0.0 or arg_key_dropout > 0.0:
        _apply_segment4_invariance_dropout(
            aligned,
            tool_name_dropout=tool_name_dropout,
            arg_key_dropout=arg_key_dropout,
            special_token_ids=special_token_ids,
            tool_name_replacement_ids=tool_name_replacement_ids,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            rng=rng,
        )

    input_ids = torch.tensor([r["input_ids"] for r in aligned], dtype=torch.int64, device=dev)
    seg_ids = torch.tensor([r["seg_ids"] for r in aligned], dtype=torch.int64, device=dev)
    token_roles = torch.tensor([r["token_roles"] for r in aligned], dtype=torch.int64, device=dev)
    query_positions = torch.tensor([r["query_positions"] for r in aligned], dtype=torch.int64, device=dev)
    pchunk_positions = torch.tensor([r["pchunk_positions"] for r in aligned], dtype=torch.int64, device=dev)
    pchunk_valid_mask = torch.tensor([r["pchunk_valid_mask"] for r in aligned], dtype=torch.bool, device=dev)
    arg_positions = torch.tensor([r["arg_positions"] for r in aligned], dtype=torch.int64, device=dev)
    arg_valid_mask = torch.tensor([r["arg_valid_mask"] for r in aligned], dtype=torch.bool, device=dev)

    y_enc = torch.tensor([r["y_enc"] for r in aligned], dtype=torch.float32, device=dev)
    y_sem = torch.tensor([r["y_sem"] for r in aligned], dtype=torch.float32, device=dev)
    y_hijack = torch.tensor([r["y_hijack"] for r in aligned], dtype=torch.float32, device=dev)
    noul_targets = torch.stack([y_enc, y_sem, y_hijack], dim=-1)

    src_target = torch.tensor([r["src_target"] for r in aligned], dtype=torch.int64, device=dev)
    arg_target = torch.tensor([r["arg_target"] for r in aligned], dtype=torch.int64, device=dev)

    categories = [str(r["threat_category"]) for r in aligned]
    variants = [str(r["variant_kind"]) for r in aligned]
    is_trifecta = torch.tensor([bool(r["is_trifecta_benign"]) for r in aligned], dtype=torch.bool, device=dev)
    flipjev = torch.tensor([bool(r["flipjev_applied"]) for r in aligned], dtype=torch.bool, device=dev)

    return {
        "input_ids": input_ids,
        "seg_ids": seg_ids,
        "token_roles": token_roles,
        "query_positions": query_positions,
        "pchunk_positions": pchunk_positions,
        "pchunk_valid_mask": pchunk_valid_mask,
        "arg_positions": arg_positions,
        "arg_valid_mask": arg_valid_mask,
        "noul_targets": noul_targets,
        "y_enc": y_enc,
        "y_sem": y_sem,
        "y_hijack": y_hijack,
        "target_src_idx": src_target,
        "src_target": src_target,
        "target_arg_idx": arg_target,
        "arg_target": arg_target,
        "threat_category": categories,
        "category": categories,
        "variant_kind": variants,
        "attack_family": variants,
        "is_trifecta_benign": is_trifecta,
        "flipjev_applied": flipjev,
        "bucket_len": int(bucket_len),
    }


build_collated_batch = collate_jev_batch


def make_synthetic_curriculum_dataset(
    num_rows: int = 16,
    *,
    vocab_size: int = 64,
    seed: int = 42,
    pad_token_id: int = 0,
) -> list[dict[str, Any]]:
    """Build deterministic synthetic packed rows spanning ``T1..T7``, ``FlipJev``, and ``BENIGN_TWIN``."""
    if num_rows < 1:
        raise ValueError("num_rows must be >= 1")
    rng = np.random.default_rng(seed)
    cls_id = 1
    sep_id = 2
    query_ids = (3, 4, 5, 6, 7)
    none_id = 8
    pchunk_id = 9
    arg_id = 10

    rows: list[dict[str, Any]] = []
    bucket_Pool = (128, 256, 512)

    for idx in range(num_rows):
        cat = ALL_THREAT_CATEGORIES[idx % len(ALL_THREAT_CATEGORIES)]
        bucket_len = bucket_Pool[idx % len(bucket_Pool)]
        n_state = int(rng.integers(36, min(96, bucket_len - 8)))
        none_pos = n_state - 1
        call_sep_pos = n_state - 2
        priv_sep_pos = max(12, n_state // 2)

        ids = [int(pad_token_id)] * bucket_len
        segs = [0] * bucket_len
        roles = [ROLE_PAD] * bucket_len

        for pos in range(n_state):
            ids[pos] = int(rng.integers(11, max(12, vocab_size)))
            roles[pos] = ROLE_STATE
            if pos < 4:
                segs[pos] = CANONICAL_SEGMENT_IDS["TRUSTED_INTENT"]
            elif pos <= priv_sep_pos:
                segs[pos] = (
                    CANONICAL_SEGMENT_IDS["DUAL_TRUST_DATA"]
                    if (cat == "T4" and pos > 8)
                    else CANONICAL_SEGMENT_IDS["PRIVATE_DATA"]
                )
            elif pos < priv_sep_pos + 6:
                segs[pos] = CANONICAL_SEGMENT_IDS["UNTRUSTED_DATA"]
            else:
                segs[pos] = CANONICAL_SEGMENT_IDS["CANDIDATE_ACTION"]

        ids[0] = cls_id
        ids[3] = sep_id
        ids[priv_sep_pos] = sep_id
        segs[priv_sep_pos] = CANONICAL_SEGMENT_IDS["PRIVATE_DATA"]
        ids[priv_sep_pos + 5] = sep_id
        segs[priv_sep_pos + 5] = CANONICAL_SEGMENT_IDS["UNTRUSTED_DATA"]
        ids[call_sep_pos] = sep_id
        segs[call_sep_pos] = CANONICAL_SEGMENT_IDS["CANDIDATE_ACTION"]
        ids[none_pos] = none_id
        segs[none_pos] = CANONICAL_SEGMENT_IDS["CANDIDATE_ACTION"]

        pchunk_slot_0 = 5
        pchunk_slot_1 = min(priv_sep_pos - 2, 9)
        ids[pchunk_slot_0] = pchunk_id
        ids[pchunk_slot_1] = pchunk_id

        arg_slot_0 = priv_sep_pos + 7
        arg_slot_1 = min(call_sep_pos - 2, priv_sep_pos + 11)
        ids[arg_slot_0] = arg_id
        ids[arg_slot_1] = arg_id

        q_positions: list[int] = []
        for k in range(5):
            qp = n_state + k
            q_positions.append(qp)
            ids[qp] = query_ids[k]
            segs[qp] = CANONICAL_SEGMENT_IDS["DECISION_QUERY"]
            roles[qp] = ROLE_QUERY

        p_positions = [none_pos] * SRC_SLOTS
        p_valid = [False] * SRC_SLOTS
        p_positions[0] = pchunk_slot_0
        p_positions[1] = pchunk_slot_1
        p_valid[0] = True
        p_valid[1] = True
        p_positions[SRC_SLOTS - 1] = none_pos
        p_valid[SRC_SLOTS - 1] = True

        a_positions = [none_pos] * ARG_SLOTS
        a_valid = [False] * ARG_SLOTS
        a_positions[0] = arg_slot_0
        a_positions[1] = arg_slot_1
        a_valid[0] = True
        a_valid[1] = True
        a_positions[ARG_SLOTS - 1] = none_pos
        a_valid[ARG_SLOTS - 1] = True

        # Ensure every 4th row is a BENIGN_TWIN / Trifecta-Safe row, and include symmetric FlipJev rows.
        is_benign = (idx % 4) == 3
        flipjev_applied = (idx % 3) == 0

        if is_benign:
            y_enc, y_sem, y_hijack = 0.0, 0.0, 0.0
            t_src = SRC_SLOTS - 1
            t_arg = ARG_SLOTS - 1
            variant = "flipjev_benign_dos_twin" if flipjev_applied else "BENIGN_TWIN"
            is_trifecta = True
        else:
            is_trifecta = False
            if cat == "T1":
                y_enc, y_sem, y_hijack = 1.0, 0.0, 0.0
                t_src = idx % 2
                t_arg = idx % 2
                variant = "obfuscated_b64_leak"
            elif cat == "T2":
                y_enc, y_sem, y_hijack = 0.0, 1.0, 0.0
                t_src = idx % 2
                t_arg = idx % 2
                variant = "obfuscated_code_leak"
            elif cat in {"T3", "T4", "T7"}:
                y_enc, y_sem, y_hijack = 0.0, 1.0, (1.0 if cat == "T4" else 0.0)
                t_src = idx % 2
                t_arg = idx % 2
                variant = f"prog_{cat.lower()}_leak"
            else:
                y_enc, y_sem, y_hijack = 0.0, 0.0, 1.0
                t_src = SRC_SLOTS - 1
                t_arg = idx % 2
                variant = f"prog_{cat.lower()}_hijack"
            if flipjev_applied:
                variant = f"{variant}_flipjev"

        rows.append(
            {
                "pair_id": f"synth_{idx:04d}_{cat.lower()}",
                "seed_id": f"seed_{idx:04d}",
                "repo_or_source": f"synthetic/{cat.lower()}_{idx // 2}",
                "group_key": f"synthetic/{cat.lower()}_{idx // 2}",
                "split": "train",
                "category": cat,
                "threat_category": cat,
                "attack_family": variant,
                "variant_kind": variant,
                "flipjev_applied": flipjev_applied,
                "intent_masked": (idx % 5) == 0,
                "is_trifecta_benign": is_trifecta,
                "bucket_length": bucket_len,
                "unpadded_length": n_state + 5,
                "none_position": none_pos,
                "was_truncated": False,
                "input_ids": ids,
                "seg_ids": segs,
                "token_roles": roles,
                "query_positions": q_positions,
                "pchunk_positions": p_positions,
                "pchunk_valid_mask": p_valid,
                "arg_positions": a_positions,
                "arg_valid_mask": a_valid,
                "y_enc": y_enc,
                "y_sem": y_sem,
                "y_hijack": y_hijack,
                "target_src_idx": t_src,
                "src_target": t_src,
                "target_arg_idx": t_arg,
                "arg_target": t_arg,
            }
        )
    return rows


# ---------------------------------------------------------------------------------------------
# Stage-2 Hard-Negative Reweighting
# ---------------------------------------------------------------------------------------------
def compute_hard_negative_weights(
    noul_logits: torch.Tensor,
    noul_targets: torch.Tensor,
    *,
    threat_categories: Sequence[str],
    variant_kinds: Sequence[str] | None = None,
    is_trifecta_benign: torch.Tensor | Sequence[bool] | None = None,
    hard_weight: float = HARD_NEGATIVE_WEIGHT,
    fp_threshold: float = 0.5,
    fn_threshold: float = 0.5,
) -> torch.Tensor:
    """Compute Stage-2 sample weights (`[B]`):

    * ``hard_weight`` (``3.0x``) on false positives on ``BENIGN_TWIN`` / ``Trifecta-Safe`` benign rows.
    * ``hard_weight`` (``3.0x``) on false negatives on ``T1`` / ``T2`` (and obfuscated) leak rows.
    * ``1.0x`` on all other rows.
    """
    probs = torch.sigmoid(noul_logits.detach().float())
    targets = noul_targets.detach().to(device=probs.device, dtype=torch.float32)
    batch_size = probs.shape[0]
    if len(threat_categories) != batch_size:
        raise ValueError(
            f"threat_categories length ({len(threat_categories)}) != batch_size ({batch_size})"
        )

    variants = (
        [str(v) for v in variant_kinds]
        if variant_kinds is not None
        else ["" for _ in range(batch_size)]
    )
    if is_trifecta_benign is None:
        trifecta_flags = [False] * batch_size
    elif isinstance(is_trifecta_benign, torch.Tensor):
        trifecta_flags = [bool(x) for x in is_trifecta_benign.detach().cpu().tolist()]
    else:
        trifecta_flags = [bool(x) for x in is_trifecta_benign]

    p_max = probs.amax(dim=-1).cpu().tolist()
    y_max = targets.amax(dim=-1).cpu().tolist()
    y_cpu = targets.cpu().tolist()
    p_cpu = probs.cpu().tolist()

    weights: list[float] = []
    for i in range(batch_size):
        cat_upper = str(threat_categories[i]).strip().upper()
        var_lower = variants[i].strip().lower()
        is_benign_row = y_max[i] < 0.5

        is_benign_twin_or_trifecta = is_benign_row and (
            trifecta_flags[i]
            or cat_upper in {"BENIGN_TWIN", "TRIFECTA_SAFE", "TRIFECTA-SAFE", "BENIGN_DEV", "BENIGN"}
            or "benign" in var_lower
            or "twin" in var_lower
            or "trifecta" in var_lower
        )

        if is_benign_twin_or_trifecta and p_max[i] >= fp_threshold:
            weights.append(float(hard_weight))
            continue

        is_attack_row = y_max[i] >= 0.5
        if is_attack_row:
            active_attack_probs = [
                p_cpu[i][h] for h in (0, 1, 2) if y_cpu[i][h] >= 0.5
            ]
            detected_prob = max(active_attack_probs) if active_attack_probs else p_max[i]
            if detected_prob < fn_threshold:
                weights.append(float(hard_weight))
                continue

        weights.append(1.0)

    return torch.tensor(weights, dtype=torch.float32, device=noul_logits.device)


# ---------------------------------------------------------------------------------------------
# Atomic Google Drive Checkpointing & RNG State Management
# ---------------------------------------------------------------------------------------------
def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(rng_state: Mapping[str, Any]) -> None:
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch_cpu" in rng_state:
        t_cpu = rng_state["torch_cpu"]
        if not isinstance(t_cpu, torch.ByteTensor):
            t_cpu = torch.as_tensor(t_cpu, dtype=torch.uint8).cpu()
        torch.set_rng_state(t_cpu)
    if "torch_cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def save_atomic_checkpoint(
    checkpoint_path: Path | str,
    *,
    model: _JevModelBase,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    stage: int,
    bucket_len: int,
    global_step: int,
    best_val_metric: float,
    extra_state: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write checkpoint via temporary sibling file + ``fsync`` + ``os.replace``."""
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")

    model_kind = "nanojev" if isinstance(model, NanoJevCausalModel) else "modernbert"
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "stage": int(stage),
        "bucket_len": int(bucket_len),
        "global_step": int(global_step),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else {},
        "best_val_metric": float(best_val_metric),
        "rng_state": _capture_rng_state(),
        "model_kind": model_kind,
        "span_pool_pointers": bool(getattr(model, "span_pool_pointers", model_kind == "nanojev")),
        "backbone_config": model.backbone.config.to_dict(),
    }
    if extra_state:
        payload.update(dict(extra_state))

    with tmp_path.open("wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, target)
    return target


def load_checkpoint(
    checkpoint_path: Path | str,
    *,
    model: _JevModelBase | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load an atomic Jev checkpoint and optionally restore model, optimizer, scheduler, and RNG."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload: dict[str, Any] = torch.load(str(path), map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict"):
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if restore_rng and "rng_state" in payload:
        _restore_rng_state(payload["rng_state"])
    return payload


def _instantiate_model_from_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> _JevModelBase:
    model_kind = str(payload.get("model_kind", "modernbert"))
    cfg_dict = dict(payload.get("backbone_config", {}))
    cfg_dict.pop("attn_implementation", None)
    cfg_dict.pop("_attn_implementation", None)
    span_pool = bool(payload.get("span_pool_pointers", model_kind == "nanojev"))
    if model_kind == "nanojev":
        cfg = Qwen3Config(**cfg_dict)
        model: _JevModelBase = NanoJevCausalModel.from_config(cfg, span_pool_pointers=span_pool)
    else:
        cfg = ModernBertConfig(**cfg_dict)
        model = ModernBERTJevModel.from_config(cfg, span_pool_pointers=span_pool)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(torch.device(device)).eval()


def _select_batch_bucket(
    batch_rows: Sequence[Mapping[str, Any] | PackedJevRow],
    candidate_buckets: Sequence[int] = CURRICULUM_BUCKETS,
) -> int:
    """Pick the smallest bucket in ``sorted(candidate_buckets)`` >= ``max(unpadded_length)`` in batch."""
    sorted_b = sorted(int(b) for b in candidate_buckets) if candidate_buckets else list(CURRICULUM_BUCKETS)
    max_unpadded = 0
    for r in batch_rows:
        if isinstance(r, PackedJevRow):
            u_len = int(r.unpadded_length)
        else:
            u_len = int(r.get("unpadded_length", len(r.get("input_ids", ()))))
        if u_len > max_unpadded:
            max_unpadded = u_len
    for b in sorted_b:
        if b >= max_unpadded:
            return b
    return sorted_b[-1]


def _get_unpadded_length(row: Mapping[str, Any] | PackedJevRow) -> int:
    if isinstance(row, PackedJevRow):
        return int(row.unpadded_length)
    return int(row.get("unpadded_length", len(row.get("input_ids", ()))))


def _select_eligible_bucket_indices(
    rows: Sequence[Mapping[str, Any] | PackedJevRow],
    bucket_len: int,
    *,
    prev_bucket_len: int = 0,
) -> list[int]:
    """Return non-truncating row indices for ``bucket_len``, preferring ``(prev_bucket_len, bucket_len]``.

    Never prefix-clamps ``>bucket_len`` rows whenever any ``<= bucket_len`` row exists, and falls
    back gracefully to ``<= bucket_len`` (or all rows on tiny custom fixtures) when needed.
    """
    if not rows:
        return []
    lengths = [_get_unpadded_length(r) for r in rows]
    non_trunc = [i for i, u in enumerate(lengths) if u <= int(bucket_len)]
    if not non_trunc:
        return list(range(len(rows)))
    if int(prev_bucket_len) > 0:
        band = [i for i in non_trunc if lengths[i] > int(prev_bucket_len)]
        if band:
            return band
    return non_trunc


def _perm_for_round(total_len: int, seed: int, wrap_round: int) -> list[int]:
    idx = list(range(total_len))
    random.Random(int(seed) + int(wrap_round) * 7919).shuffle(idx)
    return idx


def _select_unit_rows_with_cursor(
    rows: Sequence[Mapping[str, Any] | PackedJevRow],
    eligible_set: set[int],
    *,
    target_count: int,
    seed: int,
    cursor: int,
    wrap_round: int,
) -> tuple[list[Any], int, int]:
    """Select ``target_count`` distinct eligible rows using an advancing shuffled cursor."""
    n = len(rows)
    if n == 0 or not eligible_set or target_count <= 0:
        return [], cursor, wrap_round

    desired = min(int(target_count), len(eligible_set))
    selected_indices: list[int] = []
    seen_in_unit: set[int] = set()
    cur = int(cursor) % n
    rnd = int(wrap_round)
    perm = _perm_for_round(n, seed, rnd)

    while len(selected_indices) < desired:
        idx = perm[cur]
        if idx in eligible_set and idx not in seen_in_unit:
            seen_in_unit.add(idx)
            selected_indices.append(idx)
        cur += 1
        if cur >= n:
            cur = 0
            rnd += 1
            perm = _perm_for_round(n, seed, rnd)

    return [rows[i] for i in selected_indices], cur, rnd


def build_jev_optimizer_and_scheduler(
    model: _JevModelBase,
    cfg: JevTrainerConfig,
    total_planned_steps: int,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """Build AdamW with differential head LR (``head_lr_mult``) and zero decay on norm/bias/emb."""
    head_prefixes = ("noul_head.", "src_pointer.", "arg_pointer.", "seg_embedding.")
    head_decay: list[torch.nn.Parameter] = []
    head_no_decay: list[torch.nn.Parameter] = []
    backbone_decay: list[torch.nn.Parameter] = []
    backbone_no_decay: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = name.startswith(head_prefixes)
        lname = name.lower()
        is_no_decay = (
            param.ndim <= 1
            or lname.endswith(".bias")
            or "norm" in lname
            or "embedding" in lname
            or "embed" in lname
        )
        if is_head:
            (head_no_decay if is_no_decay else head_decay).append(param)
        else:
            (backbone_no_decay if is_no_decay else backbone_decay).append(param)

    base_lr = float(cfg.learning_rate)
    head_lr = base_lr * float(cfg.head_lr_mult)
    wd = float(cfg.weight_decay)
    param_groups: list[dict[str, Any]] = []
    if backbone_decay:
        param_groups.append({"params": backbone_decay, "lr": base_lr, "weight_decay": wd, "group_role": "backbone_decay"})
    if backbone_no_decay:
        param_groups.append({"params": backbone_no_decay, "lr": base_lr, "weight_decay": 0.0, "group_role": "backbone_no_decay"})
    if head_decay:
        param_groups.append({"params": head_decay, "lr": head_lr, "weight_decay": wd, "group_role": "head_decay"})
    if head_no_decay:
        param_groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0, "group_role": "head_no_decay"})
    if not param_groups:
        param_groups.append({"params": list(model.parameters()), "lr": base_lr, "weight_decay": wd})

    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)

    planned = max(1, int(total_planned_steps))
    warmup_ratio = max(0.0, float(cfg.warmup_ratio))
    warmup_steps = int(math.ceil(warmup_ratio * planned)) if (warmup_ratio > 0.0 and planned > 1) else 0
    warmup_steps = min(warmup_steps, max(0, planned - 1))

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, planned - warmup_steps)
        progress = min(1.0, float(step - warmup_steps + 1) / float(decay_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    return optimizer, scheduler


def _resolve_model_vocab_size(model: _JevModelBase, cfg: JevTrainerConfig) -> int | None:
    if cfg.vocab_size is not None:
        return int(cfg.vocab_size)
    backbone = getattr(model, "backbone", None)
    bcfg = getattr(backbone, "config", None)
    v = getattr(bcfg, "vocab_size", None)
    return int(v) if v is not None else None


# ---------------------------------------------------------------------------------------------
# Training loop (single bucket / epoch & full 2-stage length curriculum)
# ---------------------------------------------------------------------------------------------
def train_one_epoch(
    model: _JevModelBase,
    rows: Sequence[Mapping[str, Any] | PackedJevRow],
    optimizer: torch.optim.Optimizer,
    *,
    stage: int = 1,
    bucket_len: int = 512,
    config: JevTrainerConfig | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    start_batch_idx: int = 0,
    max_steps: int | None = None,
    status_log_path: Path | str | None = None,
    unit_idx: int = 0,
    checkpoint_dir: Path | str | None = None,
    best_val_metric: float = float("inf"),
    already_prepared: bool = False,
    prev_bucket_len: int = 0,
    extra_checkpoint_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one curriculum bucket pass in either Stage 1 or Stage 2."""
    cfg = config or JevTrainerConfig()
    dev = torch.device(cfg.device)
    model.to(dev)
    model.train()

    if already_prepared:
        unit_rows: Sequence[Mapping[str, Any] | PackedJevRow] = rows
    else:
        eligible_idx = _select_eligible_bucket_indices(
            rows, int(bucket_len), prev_bucket_len=int(prev_bucket_len)
        )
        random.Random(int(cfg.seed) + int(unit_idx) * 1009 + int(epoch) * 37).shuffle(eligible_idx)
        unit_rows = [rows[i] for i in eligible_idx]

    eff_vocab_size = _resolve_model_vocab_size(model, cfg)

    loss_cfg = cfg.stage1_loss_config if stage == 1 else cfg.stage2_loss_config
    batch_size = max(1, int(cfg.batch_size))
    grad_accum = max(1, int(cfg.grad_accum_steps))
    total_batches = (len(unit_rows) + batch_size - 1) // batch_size
    if cfg.max_steps_per_bucket is not None:
        total_batches = min(total_batches, int(cfg.max_steps_per_bucket) * grad_accum)

    eff_ckpt_dir = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else (Path(cfg.checkpoint_dir) if cfg.checkpoint_dir is not None else None)
    )
    save_every = int(cfg.save_every_steps) if (cfg.save_every_steps is not None and cfg.save_every_steps > 0) else None

    losses: list[float] = []
    steps_run = 0
    cur_step = int(global_step)
    last_batch_idx = start_batch_idx - 1
    saved_at_step: int | None = None

    optimizer.zero_grad(set_to_none=True)
    for b_idx in range(start_batch_idx, total_batches):
        if max_steps is not None and cur_step >= max_steps:
            break
        batch_slice = unit_rows[b_idx * batch_size : (b_idx + 1) * batch_size]
        if not batch_slice:
            continue
        batch = collate_jev_batch(
            batch_slice,
            bucket_len=bucket_len,
            pad_token_id=cfg.pad_token_id,
            device=dev,
            tool_name_dropout=float(cfg.tool_name_dropout),
            arg_key_dropout=float(cfg.arg_key_dropout),
            special_token_ids=cfg.special_token_ids,
            tool_name_replacement_ids=cfg.tool_name_replacement_ids,
            vocab_size=eff_vocab_size,
        )
        inputs = [batch[name] for name in INPUT_TENSOR_NAMES]

        if cfg.use_bf16:
            autocast_dev = "cuda" if dev.type == "cuda" else "cpu"
            with torch.autocast(autocast_dev, dtype=torch.bfloat16):
                noul_logits, src_logits, arg_logits = model.forward_logits(*inputs)
        else:
            noul_logits, src_logits, arg_logits = model.forward_logits(*inputs)

        sample_weights: torch.Tensor | None = None
        hard_neg_frac = 0.0
        if stage >= 2:
            sample_weights = compute_hard_negative_weights(
                noul_logits,
                batch["noul_targets"],
                threat_categories=batch["threat_category"],
                variant_kinds=batch["variant_kind"],
                is_trifecta_benign=batch["is_trifecta_benign"],
                hard_weight=cfg.hard_negative_weight,
                fp_threshold=cfg.fp_threshold,
                fn_threshold=cfg.fn_threshold,
            )
            hard_neg_frac = float((sample_weights > 1.0).float().mean().item())

        loss_dict = compute_jev_loss(
            noul_logits,
            src_logits,
            arg_logits,
            batch["noul_targets"],
            batch["src_target"],
            batch["arg_target"],
            batch["pchunk_valid_mask"],
            batch["arg_valid_mask"],
            config=loss_cfg,
            sample_weights=sample_weights,
        )
        raw_loss: torch.Tensor = loss_dict["loss"]
        (raw_loss / grad_accum).backward()

        is_accum_boundary = ((b_idx - start_batch_idx + 1) % grad_accum == 0) or (b_idx == total_batches - 1)
        if is_accum_boundary:
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            cur_step += 1
            steps_run += 1

            loss_val = float(raw_loss.detach().item())
            losses.append(loss_val)
            log_train_event(
                status_log_path or cfg.status_log_path,
                "train_step",
                stage=int(stage),
                epoch=int(epoch),
                bucket_len=int(bucket_len),
                global_step=cur_step,
                loss=loss_val,
                bce=float(loss_dict["bce"].detach().item()),
                brier=float(loss_dict["brier"].detach().item()),
                src_pointer_ce=float(loss_dict["src_pointer_ce"].detach().item()),
                arg_pointer_ce=float(loss_dict["arg_pointer_ce"].detach().item()),
                hard_negative_fraction=hard_neg_frac,
            )
            if eff_ckpt_dir is not None and save_every is not None and (cur_step % save_every == 0):
                bucket_done_now = b_idx >= total_batches - 1
                extra_periodic: dict[str, Any] = dict(extra_checkpoint_state or {})
                extra_periodic.update(
                    {
                        "next_unit_idx": (unit_idx + 1) if bucket_done_now else unit_idx,
                        "next_batch_idx": 0 if bucket_done_now else (b_idx + 1),
                    }
                )
                save_atomic_checkpoint(
                    eff_ckpt_dir / "checkpoint_latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    stage=stage,
                    bucket_len=bucket_len,
                    global_step=cur_step,
                    best_val_metric=best_val_metric,
                    extra_state=extra_periodic,
                )
                saved_at_step = cur_step
        last_batch_idx = b_idx

    completed_bucket = (last_batch_idx >= total_batches - 1) or (total_batches == 0)
    return {
        "global_step": cur_step,
        "steps_run": steps_run,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "last_batch_idx": last_batch_idx,
        "completed_bucket": completed_bucket,
        "saved_at_step": saved_at_step,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_curriculum_training(
    model: _JevModelBase,
    train_rows: Sequence[Mapping[str, Any] | PackedJevRow] | Path | str,
    *,
    val_rows: Sequence[Mapping[str, Any] | PackedJevRow] | Path | str | None = None,
    config: JevTrainerConfig | None = None,
    resume_from: Path | str | None = None,
) -> dict[str, Any]:
    """Run the 2-stage length-bucketed curriculum (`512 -> 1024 -> 4096`) with atomic Drive resume."""
    cfg = config or JevTrainerConfig()
    _seed_everything(cfg.seed)

    dev = torch.device(cfg.device)
    model.to(dev)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    loaded_train = load_packed_jev_dataset(train_rows)
    loaded_val = load_packed_jev_dataset(val_rows) if val_rows is not None else []

    ckpt_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir is not None else None
    status_log = (
        Path(cfg.status_log_path)
        if cfg.status_log_path is not None
        else (ckpt_dir / "live_train_status.jsonl" if ckpt_dir is not None else None)
    )

    # Build deterministic schedule of (unit_idx, stage, epoch, bucket_len)
    schedule: list[tuple[int, int, int]] = []
    for ep in range(cfg.stage1_epochs):
        for b_len in cfg.curriculum_buckets:
            schedule.append((1, ep, int(b_len)))
    for ep in range(cfg.stage2_epochs):
        for b_len in cfg.curriculum_buckets:
            schedule.append((2, ep, int(b_len)))

    batch_size = max(1, int(cfg.batch_size))
    grad_accum = max(1, int(cfg.grad_accum_steps))
    unique_buckets = sorted({int(b) for b in cfg.curriculum_buckets})
    prev_bucket_map = {
        b: (unique_buckets[i - 1] if i > 0 else 0) for i, b in enumerate(unique_buckets)
    }
    eligible_map: dict[int, set[int]] = {
        b: set(
            _select_eligible_bucket_indices(
                loaded_train, b, prev_bucket_len=prev_bucket_map[b]
            )
        )
        for b in unique_buckets
    }

    cap_rows = (
        int(cfg.max_steps_per_bucket) * grad_accum * batch_size
        if cfg.max_steps_per_bucket is not None
        else None
    )
    total_planned_steps = 0
    for _st, _ep, b_len in schedule:
        n_elig = len(eligible_map.get(int(b_len), ()))
        u_rows = min(n_elig, cap_rows) if cap_rows is not None else n_elig
        u_batches = (u_rows + batch_size - 1) // batch_size
        if cfg.max_steps_per_bucket is not None:
            u_batches = min(u_batches, int(cfg.max_steps_per_bucket) * grad_accum)
        if u_batches > 0:
            total_planned_steps += max(1, (u_batches + grad_accum - 1) // grad_accum)
    total_planned_steps = max(1, total_planned_steps)

    optimizer, scheduler = build_jev_optimizer_and_scheduler(
        model, cfg, total_planned_steps
    )

    global_step = 0
    best_val_metric = float("inf")
    start_unit_idx = 0
    start_batch_in_unit = 0
    resumed_from_step = 0
    cursor = 0
    wrap_round = 0
    calibration_report: dict[str, Any] | None = None

    resume_path: Path | None = None
    if resume_from is not None:
        cand = Path(resume_from)
        resume_path = (cand / "checkpoint_latest.pt") if cand.is_dir() else cand
    elif ckpt_dir is not None and (ckpt_dir / "checkpoint_latest.pt").is_file():
        resume_path = ckpt_dir / "checkpoint_latest.pt"

    if resume_path is not None and resume_path.is_file():
        ckpt = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=True,
            map_location=dev,
        )
        global_step = int(ckpt["global_step"])
        resumed_from_step = global_step
        best_val_metric = float(ckpt.get("best_val_metric", float("inf")))
        start_unit_idx = int(ckpt.get("next_unit_idx", 0))
        start_batch_in_unit = int(ckpt.get("next_batch_idx", 0))
        if start_batch_in_unit > 0:
            cursor = int(ckpt.get("unit_start_cursor", 0))
            wrap_round = int(ckpt.get("unit_start_wrap_round", 0))
        else:
            cursor = int(ckpt.get("next_cursor", 0))
            wrap_round = int(ckpt.get("next_wrap_round", 0))
        calibration_report = ckpt.get("calibration_report", None)
        log_train_event(
            status_log,
            "curriculum_resumed",
            checkpoint=str(resume_path),
            resumed_from_step=resumed_from_step,
            stage=int(ckpt["stage"]),
            epoch=int(ckpt["epoch"]),
            bucket_len=int(ckpt["bucket_len"]),
        )

    buckets_visited: list[int] = []
    stages_visited: list[int] = []
    step_losses: list[float] = []
    steps_executed_this_run = 0

    for unit_idx in range(start_unit_idx, len(schedule)):
        if cfg.max_steps is not None and global_step >= cfg.max_steps:
            break
        stage, epoch, bucket_len = schedule[unit_idx]
        b_start = start_batch_in_unit if unit_idx == start_unit_idx else 0
        elig_set = eligible_map.get(int(bucket_len), set())
        target_rows = len(elig_set) if cap_rows is None else min(len(elig_set), cap_rows)
        unit_start_cursor = cursor
        unit_start_wrap_round = wrap_round
        unit_rows, after_cursor, after_wrap_round = _select_unit_rows_with_cursor(
            loaded_train,
            elig_set,
            target_count=target_rows,
            seed=int(cfg.seed),
            cursor=cursor,
            wrap_round=wrap_round,
        )
        cursor = after_cursor
        wrap_round = after_wrap_round
        if not unit_rows:
            continue

        unit_ckpt_extra = {
            "unit_start_cursor": unit_start_cursor,
            "unit_start_wrap_round": unit_start_wrap_round,
            "next_cursor": cursor,
            "next_wrap_round": wrap_round,
        }
        res = train_one_epoch(
            model,
            unit_rows,
            optimizer,
            stage=stage,
            bucket_len=bucket_len,
            config=cfg,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            start_batch_idx=b_start,
            max_steps=cfg.max_steps,
            status_log_path=status_log,
            unit_idx=unit_idx,
            checkpoint_dir=ckpt_dir,
            best_val_metric=best_val_metric,
            already_prepared=True,
            prev_bucket_len=prev_bucket_map.get(int(bucket_len), 0),
            extra_checkpoint_state=unit_ckpt_extra,
        )
        global_step = int(res["global_step"])
        steps_executed_this_run += int(res["steps_run"])
        if res["steps_run"] > 0:
            buckets_visited.append(bucket_len)
            if stage not in stages_visited:
                stages_visited.append(stage)
            step_losses.append(float(res["mean_loss"]))

        if res["completed_bucket"]:
            next_unit_idx = unit_idx + 1
            next_batch_idx = 0
        else:
            next_unit_idx = unit_idx
            next_batch_idx = int(res["last_batch_idx"]) + 1

        if ckpt_dir is not None:
            latest_file = ckpt_dir / "checkpoint_latest.pt"
            stage_bucket_file = ckpt_dir / f"checkpoint_stage{stage}_bucket{bucket_len}.pt"
            extra = {
                "next_unit_idx": next_unit_idx,
                "next_batch_idx": next_batch_idx,
                "unit_start_cursor": unit_start_cursor,
                "unit_start_wrap_round": unit_start_wrap_round,
                "next_cursor": cursor,
                "next_wrap_round": wrap_round,
                "calibration_report": calibration_report,
            }
            if res.get("saved_at_step") != global_step:
                save_atomic_checkpoint(
                    latest_file,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    stage=stage,
                    bucket_len=bucket_len,
                    global_step=global_step,
                    best_val_metric=best_val_metric,
                    extra_state=extra,
                )
            save_atomic_checkpoint(
                stage_bucket_file,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                stage=stage,
                bucket_len=bucket_len,
                global_step=global_step,
                best_val_metric=best_val_metric,
                extra_state=extra,
            )

    # Post-hoc temperature scaling on unweighted val_rows when finishing Stage 2
    if loaded_val and calibration_report is None and (cfg.max_steps is None or global_step >= total_planned_steps):
        model.eval()
        val_logits_list: list[torch.Tensor] = []
        val_targets_list: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(loaded_val), batch_size):
                vb_slice = loaded_val[i : i + batch_size]
                calib_bucket = _select_batch_bucket(vb_slice, cfg.curriculum_buckets)
                vb = collate_jev_batch(
                    vb_slice,
                    bucket_len=calib_bucket,
                    pad_token_id=cfg.pad_token_id,
                    device=dev,
                )
                nl, _, _ = model.forward_logits(*[vb[name] for name in INPUT_TENSOR_NAMES])
                val_logits_list.append(nl.detach().cpu())
                val_targets_list.append(vb["noul_targets"].detach().cpu())
        val_logits = torch.cat(val_logits_list, dim=0)
        val_targets = torch.cat(val_targets_list, dim=0)
        fitted_temps = fit_temperature_scaling(val_logits, val_targets)
        model.set_logit_temperatures(fitted_temps)
        calibration_report = temperature_calibration_report(val_logits, val_targets, fitted_temps)
        best_val_metric = min(best_val_metric, float(calibration_report["ece_val"]))
        log_train_event(
            status_log,
            "temperature_calibrated",
            global_step=global_step,
            temperatures=[float(x) for x in fitted_temps.tolist()],
            ece_val=float(calibration_report["ece_val"]),
        )
        if ckpt_dir is not None and schedule:
            last_stage, last_epoch, last_bucket = schedule[-1]
            final_extra = {
                "next_unit_idx": len(schedule),
                "next_batch_idx": 0,
                "calibration_report": calibration_report,
            }
            save_atomic_checkpoint(
                ckpt_dir / "checkpoint_latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=last_epoch,
                stage=last_stage,
                bucket_len=last_bucket,
                global_step=global_step,
                best_val_metric=best_val_metric,
                extra_state=final_extra,
            )
            save_atomic_checkpoint(
                ckpt_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=last_epoch,
                stage=last_stage,
                bucket_len=last_bucket,
                global_step=global_step,
                best_val_metric=best_val_metric,
                extra_state=final_extra,
            )

    return {
        "global_step": global_step,
        "resumed_from_step": resumed_from_step,
        "steps_executed_this_run": steps_executed_this_run,
        "buckets_visited": buckets_visited,
        "stages_visited": stages_visited,
        "mean_loss": float(np.mean(step_losses)) if step_losses else 0.0,
        "best_val_metric": best_val_metric,
        "calibration_report": calibration_report,
    }


# ---------------------------------------------------------------------------------------------
# Evaluation Suite (`evaluate_jev_checkpoint`)
# ---------------------------------------------------------------------------------------------
def compute_auroc_and_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Compute exact pure-numpy AUROC (Mann-Whitney U with tie averaging) and PR-AUC (average precision)."""
    y = (np.asarray(y_true, dtype=np.float64) >= 0.5).astype(np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    n = int(len(y))
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0

    # AUROC with average ranks for ties
    order_asc = np.argsort(s, kind="mergesort")
    s_asc = s[order_asc]
    y_asc = y[order_asc]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and s_asc[j] == s_asc[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[i:j] = avg_rank
        i = j
    rank_sum_pos = float(ranks[y_asc == 1].sum())
    auroc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)

    # PR-AUC (step average precision grouped over unique score thresholds descending)
    order_desc = order_asc[::-1]
    s_desc = s[order_desc]
    y_desc = y[order_desc]
    tp = 0.0
    fp = 0.0
    prev_recall = 0.0
    ap = 0.0
    k = 0
    while k < n:
        end = k + 1
        while end < n and s_desc[end] == s_desc[k]:
            end += 1
        tp += float(y_desc[k:end].sum())
        fp += float((end - k) - int(y_desc[k:end].sum()))
        recall = tp / float(n_pos)
        precision = tp / max(1.0, tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        k = end

    return float(max(0.0, min(1.0, auroc))), float(max(0.0, min(1.0, ap)))


def evaluate_jev_checkpoint(
    model_or_checkpoint: _JevModelBase | Path | str,
    eval_rows: Sequence[Mapping[str, Any] | PackedJevRow] | Path | str,
    *,
    model: _JevModelBase | None = None,
    bucket_len: int | None = None,
    batch_size: int = 16,
    threshold: float = 0.5,
    warn_min: float = 0.20,
    critical_min: float = 0.80,
    pad_token_id: int | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Evaluate a Jev model or checkpoint across ``T1..T7`` recall, ``FlipJev``, Trifecta FPR,
    pointer top-1 accuracy on active leaks, and per-head/overall ECE + Brier."""
    dev = torch.device(device)
    if isinstance(model_or_checkpoint, (str, Path)):
        ckpt_payload = load_checkpoint(model_or_checkpoint, model=model, restore_rng=False, map_location=dev)
        eval_model = (
            model.to(dev).eval()
            if model is not None
            else _instantiate_model_from_checkpoint_payload(ckpt_payload, device=dev)
        )
    else:
        eval_model = model_or_checkpoint.to(dev).eval()

    rows = load_packed_jev_dataset(eval_rows)
    if not rows:
        raise ValueError("evaluate_jev_checkpoint requires at least 1 evaluation row")

    all_noul_logits: list[torch.Tensor] = []
    all_noul_probs: list[torch.Tensor] = []
    all_src_probs: list[torch.Tensor] = []
    all_arg_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_src_targets: list[torch.Tensor] = []
    all_arg_targets: list[torch.Tensor] = []
    all_categories: list[str] = []
    all_variants: list[str] = []
    all_trifecta: list[bool] = []
    all_flipjev: list[bool] = []

    with torch.no_grad():
        for start in range(0, len(rows), max(1, batch_size)):
            batch_slice = rows[start : start + max(1, batch_size)]
            eff_bucket = (
                int(bucket_len)
                if bucket_len is not None
                else _select_batch_bucket(batch_slice, CURRICULUM_BUCKETS)
            )
            batch = collate_jev_batch(
                batch_slice,
                bucket_len=eff_bucket,
                pad_token_id=pad_token_id,
                device=dev,
            )
            inputs = [batch[name] for name in INPUT_TENSOR_NAMES]
            noul_l, _, _ = eval_model.forward_logits(*inputs)
            noul_p, src_p, arg_p = eval_model(*inputs)

            all_noul_logits.append(noul_l.detach().cpu())
            all_noul_probs.append(noul_p.detach().cpu())
            all_src_probs.append(src_p.detach().cpu())
            all_arg_probs.append(arg_p.detach().cpu())
            all_targets.append(batch["noul_targets"].detach().cpu())
            all_src_targets.append(batch["src_target"].detach().cpu())
            all_arg_targets.append(batch["arg_target"].detach().cpu())
            all_categories.extend(batch["threat_category"])
            all_variants.extend(batch["variant_kind"])
            all_trifecta.extend(bool(x) for x in batch["is_trifecta_benign"].detach().cpu().tolist())
            all_flipjev.extend(bool(x) for x in batch["flipjev_applied"].detach().cpu().tolist())

    noul_logits = torch.cat(all_noul_logits, dim=0)
    noul_probs = torch.cat(all_noul_probs, dim=0)
    src_probs = torch.cat(all_src_probs, dim=0)
    arg_probs = torch.cat(all_arg_probs, dim=0)
    noul_targets = torch.cat(all_targets, dim=0)
    src_targets = torch.cat(all_src_targets, dim=0)
    arg_targets = torch.cat(all_arg_targets, dim=0)

    p_max = noul_probs.amax(dim=-1).numpy()
    y_max = noul_targets.amax(dim=-1).numpy()
    y_np = noul_targets.numpy()
    src_pred = src_probs.argmax(dim=-1).numpy()
    arg_pred = arg_probs.argmax(dim=-1).numpy()
    src_t_np = src_targets.numpy()
    arg_t_np = arg_targets.numpy()

    is_attack = y_max >= 0.5
    is_benign = ~is_attack
    detected = p_max >= float(threshold)

    # 1. Per-threat (T1..T7) recall
    per_threat_recall: dict[str, float] = {}
    per_threat_counts: dict[str, int] = {}
    for cat in ALL_THREAT_CATEGORIES:
        mask = np.array([c.upper() == cat for c in all_categories], dtype=bool) & is_attack
        cnt = int(mask.sum())
        per_threat_counts[cat] = cnt
        per_threat_recall[cat] = float(detected[mask].mean()) if cnt > 0 else 0.0

    attack_recall = float(detected[is_attack].mean()) if bool(is_attack.any()) else 0.0

    # 2. Symmetric FlipJev (attack recall + benign twin DoS FPR)
    flip_mask = np.array(all_flipjev, dtype=bool)
    flip_attack_mask = flip_mask & is_attack
    flip_benign_mask = flip_mask & is_benign
    flipjev_attack_recall = (
        float(detected[flip_attack_mask].mean()) if bool(flip_attack_mask.any()) else 0.0
    )
    flipjev_benign_dos_fpr = (
        float(detected[flip_benign_mask].mean()) if bool(flip_benign_mask.any()) else 0.0
    )
    if bool(flip_attack_mask.any()) and bool(flip_benign_mask.any()):
        flipjev_robust_accuracy = 0.5 * (flipjev_attack_recall + (1.0 - flipjev_benign_dos_fpr))
    elif bool(flip_attack_mask.any()):
        flipjev_robust_accuracy = flipjev_attack_recall
    elif bool(flip_benign_mask.any()):
        flipjev_robust_accuracy = 1.0 - flipjev_benign_dos_fpr
    else:
        flipjev_robust_accuracy = 0.0

    # 3. Trifecta-Safe / Benign Twin FPR
    trifecta_mask = np.array(all_trifecta, dtype=bool) & is_benign
    if not bool(trifecta_mask.any()):
        trifecta_mask = is_benign
    trifecta_fpr = float(detected[trifecta_mask].mean()) if bool(trifecta_mask.any()) else 0.0

    # 4. Pointer top-1 accuracy on active leaks
    active_leak_src = ((y_np[:, 0] >= 0.5) | (y_np[:, 1] >= 0.5)) & (src_t_np >= 0) & (src_t_np < SRC_SLOTS - 1)
    active_leak_arg = ((y_np[:, 0] >= 0.5) | (y_np[:, 1] >= 0.5)) & (arg_t_np >= 0) & (arg_t_np < ARG_SLOTS - 1)
    src_pointer_top1_acc = (
        float((src_pred[active_leak_src] == src_t_np[active_leak_src]).mean())
        if bool(active_leak_src.any())
        else 0.0
    )
    arg_pointer_top1_acc = (
        float((arg_pred[active_leak_arg] == arg_t_np[active_leak_arg]).mean())
        if bool(active_leak_arg.any())
        else 0.0
    )
    pointer_top1_acc = 0.5 * (src_pointer_top1_acc + arg_pointer_top1_acc)

    # 5. Per-head & overall ECE + Brier + AUROC / PR-AUC + band-threshold recall/FPR
    temps = eval_model.logit_temperatures.detach().cpu()
    calib_rep = temperature_calibration_report(noul_logits, noul_targets, temps)
    brier_per_head_t = ((noul_probs - noul_targets) ** 2).mean(dim=0)
    brier_per_head = [float(x) for x in brier_per_head_t.tolist()]
    brier_overall = float(brier_per_head_t.mean().item())

    auroc, pr_auc = compute_auroc_and_pr_auc(is_attack, p_max)
    p_np = noul_probs.numpy()
    auroc_per_head: list[float] = []
    pr_auc_per_head: list[float] = []
    for h in range(3):
        h_auc, h_ap = compute_auroc_and_pr_auc(y_np[:, h], p_np[:, h])
        auroc_per_head.append(h_auc)
        pr_auc_per_head.append(h_ap)

    det_warn = p_max >= float(warn_min)
    det_crit = p_max >= float(critical_min)
    recall_at_warn_min = float(det_warn[is_attack].mean()) if bool(is_attack.any()) else 0.0
    recall_at_critical_min = float(det_crit[is_attack].mean()) if bool(is_attack.any()) else 0.0
    fpr_at_warn_min = float(det_warn[is_benign].mean()) if bool(is_benign.any()) else 0.0
    fpr_at_critical_min = float(det_crit[is_benign].mean()) if bool(is_benign.any()) else 0.0

    return {
        "num_rows": len(rows),
        "per_threat_recall": per_threat_recall,
        "per_threat_counts": per_threat_counts,
        "attack_recall": attack_recall,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "auroc_per_head": auroc_per_head,
        "pr_auc_per_head": pr_auc_per_head,
        "recall_at_warn_min": recall_at_warn_min,
        "recall_at_critical_min": recall_at_critical_min,
        "fpr_at_warn_min": fpr_at_warn_min,
        "fpr_at_critical_min": fpr_at_critical_min,
        "flipjev_attack_recall": flipjev_attack_recall,
        "flipjev_benign_dos_fpr": flipjev_benign_dos_fpr,
        "flipjev_dos_fpr": flipjev_benign_dos_fpr,
        "flipjev_robust_accuracy": float(flipjev_robust_accuracy),
        "trifecta_fpr": trifecta_fpr,
        "trifecta_safe_fpr": trifecta_fpr,
        "benign_twin_fpr": trifecta_fpr,
        "src_pointer_top1_acc": src_pointer_top1_acc,
        "arg_pointer_top1_acc": arg_pointer_top1_acc,
        "pointer_top1_acc": pointer_top1_acc,
        "ece_per_head": list(calib_rep["ece_after"]),
        "ece_val": float(calib_rep["ece_val"]),
        "brier_per_head": brier_per_head,
        "brier": brier_overall,
        "calibration_report": calib_rep,
    }


# ---------------------------------------------------------------------------------------------
# CLI Entrypoint (`python -m diodemodel.train`)
# ---------------------------------------------------------------------------------------------
def _build_tiny_smoke_modernbert(vocab_size: int = 64, pad_token_id: int = 0) -> ModernBERTJevModel:
    cfg = ModernBertConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        local_attention=8,
        max_position_embeddings=4096,
        pad_token_id=pad_token_id,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    return ModernBERTJevModel.from_config(cfg)


def _build_tiny_smoke_nanojev(vocab_size: int = 64, pad_token_id: int = 0) -> NanoJevCausalModel:
    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=4096,
        pad_token_id=pad_token_id,
    )
    return NanoJevCausalModel.from_config(cfg)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m diodemodel.train",
        description="DiodeMCP Track A 2-stage length-bucketed curriculum trainer.",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=None,
        help="Run fast CPU smoke test for N steps on 16 synthetic rows across tiny ModernBERT & NanoJev.",
    )
    parser.add_argument(
        "--num-synthetic-rows",
        type=int,
        default=16,
        help="Number of synthetic rows to generate in smoke mode (default: 16).",
    )
    parser.add_argument(
        "--arch",
        choices=("both", "modernbert", "nanojev"),
        default="both",
        help="Architecture to run in smoke mode or pretrained training.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Hugging Face model ID or local path for pretrained training.",
    )
    parser.add_argument("--vocab-size", type=int, default=None, help="Extended Jev tokenizer vocab size.")
    parser.add_argument("--train-parquet", type=Path, default=None, help="Packed train Parquet path.")
    parser.add_argument("--val-parquet", type=Path, default=None, help="Packed val_calib Parquet path.")
    parser.add_argument("--eval-parquet", type=Path, default=None, help="Packed eval_holdout Parquet path.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for atomic checkpoints and live_train_status.jsonl.",
    )
    parser.add_argument(
        "--buckets",
        type=str,
        default="512,1024,4096",
        help="Comma-separated curriculum bucket lengths (default: 512,1024,4096).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate.")
    parser.add_argument("--stage1-epochs", type=int, default=1, help="Stage-1 epochs.")
    parser.add_argument("--stage2-epochs", type=int, default=1, help="Stage-2 epochs.")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 autocast.")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable backbone gradient checkpointing.",
    )
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint file or dir to resume.")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu or cuda).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    buckets = tuple(int(x.strip()) for x in args.buckets.split(",") if x.strip())
    dev = args.device or ("cuda" if torch.cuda.is_available() and args.smoke_steps is None else "cpu")

    if args.smoke_steps is not None:
        smoke_steps = max(1, int(args.smoke_steps))
        synth_rows = make_synthetic_curriculum_dataset(
            args.num_synthetic_rows, vocab_size=64, seed=args.seed, pad_token_id=0
        )
        tmp_ctx = tempfile.TemporaryDirectory(prefix="diodemodel_smoke_") if args.checkpoint_dir is None else None
        base_ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir is not None else Path(tmp_ctx.name)  # type: ignore[union-attr]
        base_ckpt_dir.mkdir(parents=True, exist_ok=True)
        shared_status_log = base_ckpt_dir / "live_train_status.jsonl"

        arches = ("modernbert", "nanojev") if args.arch == "both" else (args.arch,)
        t0 = time.perf_counter()
        try:
            for arch_name in arches:
                model: _JevModelBase = (
                    _build_tiny_smoke_modernbert()
                    if arch_name == "modernbert"
                    else _build_tiny_smoke_nanojev()
                )
                arch_ckpt_dir = base_ckpt_dir / arch_name
                cfg = JevTrainerConfig(
                    curriculum_buckets=buckets,
                    batch_size=min(8, max(2, len(synth_rows) // 2)),
                    learning_rate=1e-3,
                    stage1_epochs=1,
                    stage2_epochs=1,
                    max_steps=smoke_steps,
                    max_steps_per_bucket=1,
                    pad_token_id=0,
                    checkpoint_dir=arch_ckpt_dir,
                    status_log_path=shared_status_log,
                    device=dev,
                    seed=args.seed,
                )
                train_res = run_curriculum_training(
                    model, synth_rows, val_rows=synth_rows[:8], config=cfg
                )
                eval_res = evaluate_jev_checkpoint(
                    model, synth_rows[:8], bucket_len=512, batch_size=8, pad_token_id=0, device=dev
                )
                log_train_event(
                    shared_status_log,
                    "smoke_arch_completed",
                    arch=arch_name,
                    global_step=train_res["global_step"],
                    ece_val=eval_res["ece_val"],
                    attack_recall=eval_res["attack_recall"],
                )
            elapsed = time.perf_counter() - t0
            print(
                f"Smoke run OK: {len(synth_rows)} synthetic rows across {arches} "
                f"({smoke_steps} steps/model) in {elapsed:.2f}s"
            )
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()
        return 0

    if args.backbone is None or args.train_parquet is None:
        raise ValueError("--backbone and --train-parquet are required when not running --smoke-steps")

    is_nano = args.arch == "nanojev" or "qwen" in args.backbone.lower() or "nano" in args.backbone.lower()
    model_cls = NanoJevCausalModel if is_nano else ModernBERTJevModel
    model = model_cls.from_pretrained(args.backbone, vocab_size=args.vocab_size)

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir is not None else Path("./checkpoints")
    cfg = JevTrainerConfig(
        curriculum_buckets=buckets,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        use_bf16=bool(args.bf16),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        checkpoint_dir=ckpt_dir,
        status_log_path=ckpt_dir / "live_train_status.jsonl",
        device=dev,
        seed=args.seed,
    )
    run_curriculum_training(
        model,
        args.train_parquet,
        val_rows=args.val_parquet,
        config=cfg,
        resume_from=args.resume_from,
    )
    if args.eval_parquet is not None and args.eval_parquet.is_file():
        metrics = evaluate_jev_checkpoint(
            model,
            args.eval_parquet,
            bucket_len=buckets[-1],
            batch_size=args.batch_size,
            device=dev,
        )
        (ckpt_dir / "eval_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
    return 0


__all__ = [
    "CURRICULUM_BUCKETS",
    "CurriculumConfig",
    "HARD_NEGATIVE_WEIGHT",
    "JevTrainerConfig",
    "align_row_to_bucket",
    "build_collated_batch",
    "build_jev_optimizer_and_scheduler",
    "collate_jev_batch",
    "compute_auroc_and_pr_auc",
    "compute_hard_negative_weights",
    "evaluate_jev_checkpoint",
    "load_checkpoint",
    "load_packed_jev_dataset",
    "log_train_event",
    "main",
    "make_synthetic_curriculum_dataset",
    "run_curriculum_training",
    "save_atomic_checkpoint",
    "train_one_epoch",
]


if __name__ == "__main__":
    raise SystemExit(main())
