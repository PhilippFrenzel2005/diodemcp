"""Shared pytest fixtures for DiodeMCP Track B, including a zero-PyTorch opset-17 ONNX v1 bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from diodemcp.constants import (
    BUNDLE_VERSION,
    CANONICAL_SEGMENT_IDS,
    CRITICAL_THRESHOLD,
    LENGTH_BUCKETS,
    MAX_ARGS,
    MAX_PCHUNKS,
    MAX_SEQ_LEN,
    ONNX_INPUT_NAMES,
    ONNX_OUTPUT_NAMES,
    REQUIRED_SPECIAL_TOKENS,
    WARN_THRESHOLD,
)

# Semantic & domain vocabulary used by the synthetic opset-17 Jev cross-encoder
SYNTHETIC_DOMAIN_WORDS: tuple[str, ...] = (
    "[UNK]",
    # Benign developer & tool vocabulary
    "check",
    "fix",
    "login",
    "timeout",
    "bug",
    "unit",
    "tests",
    "status",
    "summary",
    "update",
    "read",
    "open",
    "issue",
    "github",
    "create_issue",
    "slack",
    "post_message",
    "fetch",
    "fetch_url",
    "postgres",
    "query",
    "bash",
    "run_command",
    "body",
    "text",
    "title",
    "channel",
    "url",
    "headers",
    "Authorization",
    "sql",
    "command",
    "repo",
    "acme",
    "monorepo",
    "payload",
    "notes",
    "config",
    "safe",
    "ok",
    "distractor",
    "filler",
    "module",
    "helper",
    "retry",
    "alpha",
    "beta",
    "gamma",
    "delta",
    # Concept 0: M&A / financial confidential data & zero-overlap paraphrases
    "confidential",
    "acquisition",
    "valuation",
    "quarterly",
    "churn",
    "upcoming",
    "buyout",
    "pricetag",
    "seasonal",
    "attrition",
    # Concept 1: Proprietary pricing / rebate & zero-overlap paraphrases
    "proprietary",
    "rebate",
    "multiplier",
    "enterprise",
    "threshold",
    "internal",
    "kickback",
    "coefficient",
    "corporate",
    "cutoff",
    # Concept 2: Weak borderline cue (produces ~0.35 WARN alone at 512)
    "roadmap",
    "milestone",
    "timeline",
    "deliverable",
    # Concept 3: Strong middle cue (pushes score >0.90 CRITICAL when 2048 reveals middle)
    "stealth",
    "patent",
    "blueprint",
    "unannounced",
    "invention",
    "architecture",
    # Concept 4: Semantic untrusted hijack / prompt injection & sensitive target cues
    "disregard",
    "instructions",
    "silently",
    "reroute",
    "attacker",
    "webhook",
    "forwarding",
    "endpoint",
    "permissions",
    # Concept 5: Explicit user intent authorization cues
    "share",
    "publish",
    "disclose",
)


def build_synthetic_tokenizer() -> tuple[Tokenizer, dict[str, int], dict[str, int]]:
    """Build a deterministic fast Rust tokenizer containing all 17 Jev special tokens."""
    special_ids = {tok: idx for idx, tok in enumerate(REQUIRED_SPECIAL_TOKENS)}
    vocab: dict[str, int] = dict(special_ids)
    next_id = len(vocab)
    for word in SYNTHETIC_DOMAIN_WORDS:
        if word not in vocab:
            vocab[word] = next_id
            next_id += 1

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.add_special_tokens(
        [
            AddedToken(
                s,
                single_word=False,
                lstrip=False,
                rstrip=False,
                normalized=False,
                special=True,
            )
            for s in REQUIRED_SPECIAL_TOKENS
        ]
    )
    return tok, special_ids, vocab


def build_synthetic_jev_onnx_bytes(
    vocab: dict[str, int],
    *,
    inject_noul_values: tuple[float, float, float] | None = None,
    inject_src_nan: bool = False,
    inject_arg_nan: bool = False,
) -> bytes:
    """Construct a ~50 KB `opset=17` ONNX v1 Jev cross-encoder with internal 4D `token_roles` mask."""
    vocab_size = max(vocab.values()) + 1
    d_model = 40
    num_heads = 4
    head_dim = 10  # num_heads * head_dim == d_model

    # Token embedding layout (d_model = 40):
    #   0..5   : semantic concepts c0..c5
    #   6..11  : segment one-hot (from seg_emb)
    #   12     : local anchor (<|pchunk|>, <|arg|>)
    #   13..17 : query markers (<|q_enc|>, <|q_sem|>, <|q_hijack|>, <|q_src|>, <|q_arg|>)
    #   18     : semantic cue salience (+2.0 on cue words, -2.0 on filler/structural)
    #   19     : <|none|> indicator (+1.0 on <|none|>)
    #   20..39 : attention & FFN write-back slots (initialized to 0.0)
    tok_emb = np.zeros((vocab_size, d_model), dtype=np.float32)
    tok_emb[:, 18] = -2.0

    concept_groups: tuple[tuple[int, tuple[str, ...]], ...] = (
        (
            0,
            (
                "confidential",
                "acquisition",
                "valuation",
                "quarterly",
                "churn",
                "upcoming",
                "buyout",
                "pricetag",
                "seasonal",
                "attrition",
            ),
        ),
        (
            1,
            (
                "proprietary",
                "rebate",
                "multiplier",
                "enterprise",
                "threshold",
                "internal",
                "kickback",
                "coefficient",
                "corporate",
                "cutoff",
            ),
        ),
        (2, ("roadmap", "milestone", "timeline", "deliverable")),
        (
            3,
            (
                "stealth",
                "patent",
                "blueprint",
                "unannounced",
                "invention",
                "architecture",
            ),
        ),
        (
            4,
            (
                "disregard",
                "instructions",
                "silently",
                "reroute",
                "attacker",
                "webhook",
                "forwarding",
                "endpoint",
                "permissions",
            ),
        ),
        (5, ("share", "publish", "disclose")),
    )
    for ch_idx, words in concept_groups:
        for w in words:
            wid = vocab[w]
            tok_emb[wid, ch_idx] = 1.0
            tok_emb[wid, 18] = 2.0

    for anchor_tok in ("<|pchunk|>", "<|arg|>"):
        tok_emb[vocab[anchor_tok], 12] = 3.0
    for q_offset, q_tok in enumerate(
        ("<|q_enc|>", "<|q_sem|>", "<|q_hijack|>", "<|q_src|>", "<|q_arg|>")
    ):
        tok_emb[vocab[q_tok], 13 + q_offset] = 3.0
    tok_emb[vocab["<|none|>"], 19] = 1.0

    seg_emb = np.zeros((6, d_model), dtype=np.float32)
    for s_idx in range(6):
        seg_emb[s_idx, 6 + s_idx] = 3.0

    # Multi-head projections W_q, W_k, W_v, W_o: [40, 40]
    w_q = np.zeros((d_model, d_model), dtype=np.float32)
    w_k = np.zeros((d_model, d_model), dtype=np.float32)
    w_v = np.zeros((d_model, d_model), dtype=np.float32)
    w_o = np.zeros((d_model, d_model), dtype=np.float32)

    # Head 0 (cols 0..9): Local window aggregator for <|pchunk|> and <|arg|>
    w_q[12, 0] = 2.0  # anchor query dim -> head0 dim 0
    w_k[18, 0] = 3.0  # salience -> head0 dim 0 (+18 on semantic cue, -18 on filler)
    for c in range(5):
        w_v[c, c] = 5.0
        w_o[c, 20 + c] = 1.0  # write local concepts to h1 dims 20..24

    # Head 1 (cols 10..19): Global Taint Reader
    #   dim 10: seg 1/2 selector for q_enc, q_sem, q_arg
    #   dim 11: salience selector for q_enc, q_sem, q_arg
    #   dim 12: seg 2/3 selector for q_hijack, q_arg
    #   dim 13: salience selector for q_hijack
    for q_dim in (13, 14, 17):  # q_enc, q_sem, q_arg
        w_q[q_dim, 10] = 1.5
        w_q[q_dim, 11] = 1.5
    w_k[6 + 1, 10] = 3.0  # seg 1 (PRIVATE_DATA)
    w_k[6 + 2, 10] = 3.0  # seg 2 (DUAL_TRUST_DATA)
    w_k[18, 11] = 2.0

    w_q[15, 12] = 1.5  # q_hijack
    w_q[15, 13] = 1.5
    w_q[17, 12] = 1.5  # q_arg also reads untrusted seg 3 for hijack arg attribution
    w_k[6 + 2, 12] = 3.0  # seg 2 (DUAL_TRUST_DATA)
    w_k[6 + 3, 12] = 3.0  # seg 3 (UNTRUSTED_DATA)
    w_k[18, 13] = 2.0

    for c in range(4):
        w_v[c, 10 + c] = 5.0
        w_o[10 + c, 25 + c] = 1.0  # write priv concepts c0..c3 to h1 dims 25..28
    w_v[4, 14] = 5.0
    w_o[14, 29] = 1.0  # write hijack concept c4 to h1 dim 29

    # Head 2 (cols 20..29): Global Candidate Action (seg 4) Reader
    for q_dim in (13, 14, 15, 16):  # q_enc, q_sem, q_hijack, q_src
        w_q[q_dim, 20] = 1.5
        w_q[q_dim, 21] = 1.5
    w_k[6 + 4, 20] = 3.0  # seg 4 (CANDIDATE_ACTION)
    w_k[18, 21] = 2.0
    for c in range(5):
        w_v[c, 20 + c] = 5.0
        w_o[20 + c, 30 + c] = 1.0  # write call concepts c0..c4 to h1 dims 30..34

    # Head 3 (cols 30..39): Global Trusted Intent (seg 0) Authorization Reader
    for q_dim in (13, 14, 15):  # q_enc, q_sem, q_hijack
        w_q[q_dim, 30] = 1.5
        w_q[q_dim, 31] = 1.5
    w_k[6 + 0, 30] = 3.0  # seg 0 (TRUSTED_INTENT)
    w_k[18, 31] = 2.0
    w_v[5, 30] = 5.0
    w_o[30, 35] = 1.0  # write auth concept c5 to h1 dim 35

    # Position-wise conjunctive FFN over clipped [0, 1] features:
    # Computes Relu(taint_c + call_c - 1.3) for c in 0..4 and passes auth_c through.
    d_ff = 16
    w_ff1 = np.zeros((d_model, d_ff), dtype=np.float32)
    b_ff1 = np.zeros((d_ff,), dtype=np.float32)
    w_ff2 = np.zeros((d_ff, d_model), dtype=np.float32)
    b_ff2 = np.zeros((d_model,), dtype=np.float32)

    for c in range(5):
        w_ff1[25 + c, c] = 1.0  # taint_c
        w_ff1[30 + c, c] = 1.0  # call_c
        b_ff1[c] = -1.3
    w_ff1[35, 5] = 1.0  # auth_c

    # Write conjunctive risk signals into h2 dims 36 (sem), 37 (hijack), 38 (enc)
    w_ff2[0, 36] = 8.5  # Concept 0 conjunction -> sem CRITICAL (~0.95)
    w_ff2[1, 36] = 8.5  # Concept 1 conjunction -> sem CRITICAL (~0.95)
    w_ff2[2, 36] = 3.4  # Concept 2 weak cue -> sem WARN (~0.35)
    w_ff2[3, 36] = 5.5  # Concept 3 strong middle cue -> sem CRITICAL (>0.95 with c2)
    w_ff2[5, 36] = -10.0  # Intent authorization suppresses sem

    w_ff2[4, 37] = 8.5  # Concept 4 conjunction -> hijack CRITICAL (~0.95)
    w_ff2[5, 37] = -10.0  # Intent authorization suppresses hijack

    # NOUL linear head: [3, 40] for (<|q_enc|>, <|q_sem|>, <|q_hijack|>)
    w_noul = np.zeros((3, d_model), dtype=np.float32)
    w_noul[0, 38] = 1.0  # p_enc
    w_noul[1, 36] = 1.0  # p_sem
    w_noul[2, 37] = 1.0  # p_hijack
    b_noul = np.array([-3.5, -3.0, -3.0], dtype=np.float32)
    temp_noul = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Pointer bilinear projection matrices [40, 40]
    # q_src (dims 30..34) matches pchunk local concepts (dims 20..24); <|none|> (dim 19) gets +1.5 baseline
    w_src_ptr = np.zeros((d_model, d_model), dtype=np.float32)
    for c in range(5):
        w_src_ptr[30 + c, 20 + c] = 8.0
    w_src_ptr[16, 19] = 1.5  # clipped q_src (1.0 at dim 16) * none (1.0 at dim 19) * 1.5 = +1.5

    # q_arg (dims 25..29) matches arg local concepts (dims 20..24); <|none|> (dim 19) gets +1.5 baseline
    w_arg_ptr = np.zeros((d_model, d_model), dtype=np.float32)
    for c in range(5):
        w_arg_ptr[25 + c, 20 + c] = 8.0
    w_arg_ptr[17, 19] = 1.5  # clipped q_arg (1.0 at dim 17) * none (1.0 at dim 19) * 1.5 = +1.5
    if inject_src_nan:
        w_src_ptr[16, 19] = np.float32(np.nan)
    if inject_arg_nan:
        w_arg_ptr[17, 19] = np.float32(np.nan)

    initializers = [
        numpy_helper.from_array(tok_emb, name="tok_emb"),
        numpy_helper.from_array(seg_emb, name="seg_emb"),
        numpy_helper.from_array(w_q, name="w_q"),
        numpy_helper.from_array(w_k, name="w_k"),
        numpy_helper.from_array(w_v, name="w_v"),
        numpy_helper.from_array(w_o, name="w_o"),
        numpy_helper.from_array(w_ff1, name="w_ff1"),
        numpy_helper.from_array(b_ff1, name="b_ff1"),
        numpy_helper.from_array(w_ff2, name="w_ff2"),
        numpy_helper.from_array(b_ff2, name="b_ff2"),
        numpy_helper.from_array(w_noul, name="w_noul"),
        numpy_helper.from_array(b_noul, name="b_noul"),
        numpy_helper.from_array(temp_noul, name="temp_noul"),
        numpy_helper.from_array(w_src_ptr, name="w_src_ptr"),
        numpy_helper.from_array(w_arg_ptr, name="w_arg_ptr"),
        # Scalar & shape constants
        numpy_helper.from_array(np.array(0, dtype=np.int64), name="c_i64_0"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), name="c_i64_1"),
        numpy_helper.from_array(np.array(2, dtype=np.int64), name="c_i64_2"),
        numpy_helper.from_array(
            np.array(vocab["<|pchunk|>"], dtype=np.int64), name="c_id_pchunk"
        ),
        numpy_helper.from_array(
            np.array(vocab["<|arg|>"], dtype=np.int64), name="c_id_arg"
        ),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="ax_0"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="ax_1"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), name="ax_2"),
        numpy_helper.from_array(np.array([num_heads, head_dim], dtype=np.int64), name="c_heads_dim"),
        numpy_helper.from_array(np.array([d_model], dtype=np.int64), name="c_d_model_1d"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="c_f32_0"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="c_f32_1"),
        numpy_helper.from_array(np.array(-10000.0, dtype=np.float32), name="c_f32_neg_mask"),
        numpy_helper.from_array(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 4, 1, 1),
            name="c_head0_sel",
        ),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="sl_0"),
        numpy_helper.from_array(np.array([3], dtype=np.int64), name="sl_3"),
        numpy_helper.from_array(np.array([4], dtype=np.int64), name="sl_4"),
        numpy_helper.from_array(np.array([5], dtype=np.int64), name="sl_5"),
    ]
    if inject_noul_values is not None:
        initializers.append(
            numpy_helper.from_array(
                np.array([inject_noul_values], dtype=np.float32),
                name="c_injected_noul",
            )
        )

    nodes: list[onnx.NodeProto] = [
        # 1. Embeddings
        helper.make_node("Gather", ["tok_emb", "input_ids"], ["x_tok"], axis=0),
        helper.make_node("Gather", ["seg_emb", "seg_ids"], ["x_seg"], axis=0),
        helper.make_node("Add", ["x_tok", "x_seg"], ["h0"]),
        # 2. Internal 4D additive mask from token_roles (0=PAD, 1=STATE, 2=QUERY)
        helper.make_node("Equal", ["token_roles", "c_i64_1"], ["is_state"]),
        helper.make_node("Equal", ["token_roles", "c_i64_2"], ["is_query"]),
        helper.make_node("Or", ["is_state", "is_query"], ["state_or_query"]),
        helper.make_node("Unsqueeze", ["state_or_query", "ax_2"], ["sq_i"]),
        helper.make_node("Unsqueeze", ["is_state", "ax_1"], ["st_j"]),
        helper.make_node("And", ["sq_i", "st_j"], ["cross_ok"]),
        helper.make_node("Shape", ["token_roles"], ["roles_shape"]),
        helper.make_node("Gather", ["roles_shape", "c_i64_1"], ["seq_len_scalar"], axis=0),
        helper.make_node("Range", ["c_i64_0", "seq_len_scalar", "c_i64_1"], ["arange_l"]),
        helper.make_node("Unsqueeze", ["arange_l", "ax_1"], ["idx_i"]),
        helper.make_node("Unsqueeze", ["arange_l", "ax_0"], ["idx_j"]),
        helper.make_node("Equal", ["idx_i", "idx_j"], ["diag_2d"]),
        helper.make_node("Unsqueeze", ["diag_2d", "ax_0"], ["diag_3d"]),
        helper.make_node("Or", ["cross_ok", "diag_3d"], ["allowed_3d"]),
        helper.make_node("Unsqueeze", ["allowed_3d", "ax_1"], ["allowed_4d"]),
        helper.make_node(
            "Where", ["allowed_4d", "c_f32_0", "c_f32_neg_mask"], ["attn_mask_4d"]
        ),
        # Head 0 exact chunk-scoped bias (j >= i, same seg_id, and same <|pchunk|>/<|arg|> span via CumSum)
        helper.make_node("Sub", ["idx_j", "idx_i"], ["rel_dist"]),
        helper.make_node("GreaterOrEqual", ["rel_dist", "c_i64_0"], ["dist_ge_0"]),
        helper.make_node("Unsqueeze", ["dist_ge_0", "ax_0"], ["in_win_3d"]),
        helper.make_node("Equal", ["input_ids", "c_id_pchunk"], ["is_pchunk_tok"]),
        helper.make_node("Equal", ["input_ids", "c_id_arg"], ["is_arg_tok"]),
        helper.make_node("Or", ["is_pchunk_tok", "is_arg_tok"], ["is_delim_tok"]),
        helper.make_node("Cast", ["is_delim_tok"], ["is_delim_i64"], to=TensorProto.INT64),
        helper.make_node("CumSum", ["is_delim_i64", "c_i64_1"], ["chunk_ids"]),
        helper.make_node("Unsqueeze", ["chunk_ids", "ax_2"], ["chunk_i"]),
        helper.make_node("Unsqueeze", ["chunk_ids", "ax_1"], ["chunk_j"]),
        helper.make_node("Equal", ["chunk_i", "chunk_j"], ["same_chunk_3d"]),
        helper.make_node("Unsqueeze", ["seg_ids", "ax_2"], ["seg_i"]),
        helper.make_node("Unsqueeze", ["seg_ids", "ax_1"], ["seg_j"]),
        helper.make_node("Equal", ["seg_i", "seg_j"], ["same_seg_3d"]),
        helper.make_node("And", ["same_seg_3d", "same_chunk_3d"], ["seg_and_chunk_3d"]),
        helper.make_node("And", ["in_win_3d", "seg_and_chunk_3d"], ["local_ok_3d"]),
        helper.make_node("Unsqueeze", ["local_ok_3d", "ax_1"], ["local_ok_4d"]),
        helper.make_node(
            "Where", ["local_ok_4d", "c_f32_0", "c_f32_neg_mask"], ["head0_raw_bias"]
        ),
        helper.make_node("Mul", ["head0_raw_bias", "c_head0_sel"], ["head0_bias_4d"]),
        # 3. Multi-Head Self-Attention
        helper.make_node("Gather", ["roles_shape", "ax_0"], ["b_dim_1d"], axis=0),
        helper.make_node("Gather", ["roles_shape", "ax_1"], ["l_dim_1d"], axis=0),
        helper.make_node(
            "Concat", ["b_dim_1d", "l_dim_1d", "c_heads_dim"], ["mha_shape"], axis=0
        ),
        helper.make_node(
            "Concat", ["b_dim_1d", "l_dim_1d", "c_d_model_1d"], ["flat_shape"], axis=0
        ),
        helper.make_node("MatMul", ["h0", "w_q"], ["q_lin"]),
        helper.make_node("MatMul", ["h0", "w_k"], ["k_lin"]),
        helper.make_node("MatMul", ["h0", "w_v"], ["v_lin"]),
        helper.make_node("Reshape", ["q_lin", "mha_shape"], ["q_4d"]),
        helper.make_node("Reshape", ["k_lin", "mha_shape"], ["k_4d"]),
        helper.make_node("Reshape", ["v_lin", "mha_shape"], ["v_4d"]),
        helper.make_node("Transpose", ["q_4d"], ["q_t"], perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", ["k_4d"], ["k_t"], perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["v_4d"], ["v_t"], perm=[0, 2, 1, 3]),
        helper.make_node("MatMul", ["q_t", "k_t"], ["qk_raw"]),
        helper.make_node("Add", ["qk_raw", "attn_mask_4d"], ["qk_masked"]),
        helper.make_node("Add", ["qk_masked", "head0_bias_4d"], ["qk_total"]),
        helper.make_node("Softmax", ["qk_total"], ["attn_probs"], axis=-1),
        helper.make_node("MatMul", ["attn_probs", "v_t"], ["ctx_4d"]),
        helper.make_node("Transpose", ["ctx_4d"], ["ctx_t"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["ctx_t", "flat_shape"], ["ctx_flat"]),
        helper.make_node("MatMul", ["ctx_flat", "w_o"], ["attn_out"]),
        helper.make_node("Add", ["h0", "attn_out"], ["h1_raw"]),
        # 4. Position-wise conjunctive FFN over clipped [0, 1] attention features
        helper.make_node("Clip", ["h1_raw", "c_f32_0", "c_f32_1"], ["h1_clip"]),
        helper.make_node("MatMul", ["h1_clip", "w_ff1"], ["ff_pre"]),
        helper.make_node("Add", ["ff_pre", "b_ff1"], ["ff_bias"]),
        helper.make_node("Relu", ["ff_bias"], ["ff_act"]),
        helper.make_node("MatMul", ["ff_act", "w_ff2"], ["ff_out"]),
        helper.make_node("Add", ["ff_out", "b_ff2"], ["ff_out_b"]),
        helper.make_node("Add", ["h1_raw", "ff_out_b"], ["h2"]),
        # Clip feature dims while preserving full dynamic range on FFN logits via h2 + h1_clip
        # Gather query_positions [B, 5], pchunk_positions [B, 33], arg_positions [B, 17]
        helper.make_node("Shape", ["query_positions"], ["q_pos_shape"]),
        helper.make_node(
            "Concat", ["q_pos_shape", "c_d_model_1d"], ["q_exp_shape"], axis=0
        ),
        helper.make_node("Unsqueeze", ["query_positions", "ax_2"], ["q_pos_3d"]),
        helper.make_node("Expand", ["q_pos_3d", "q_exp_shape"], ["q_idx_exp"]),
        helper.make_node("GatherElements", ["h2", "q_idx_exp"], ["q_h2"], axis=1),
        helper.make_node("GatherElements", ["h1_clip", "q_idx_exp"], ["q_h1c"], axis=1),
        # 5. NOUL Head: Slice first 3 queries (<|q_enc|>, <|q_sem|>, <|q_hijack|>)
        helper.make_node("Slice", ["q_h2", "sl_0", "sl_3", "ax_1"], ["q_risk"]),
        helper.make_node("Unsqueeze", ["w_noul", "ax_0"], ["w_noul_3d"]),
        helper.make_node("Mul", ["q_risk", "w_noul_3d"], ["risk_mul"]),
        helper.make_node(
            "ReduceSum", ["risk_mul", "ax_2"], ["risk_dot"], keepdims=0
        ),
        helper.make_node("Add", ["risk_dot", "b_noul"], ["risk_biased"]),
        helper.make_node("Div", ["risk_biased", "temp_noul"], ["risk_scaled"]),
    ]

    if inject_noul_values is None:
        nodes.append(helper.make_node("Sigmoid", ["risk_scaled"], ["noul_probs"]))
    else:
        # Multiply risk_scaled by 0.0 and add c_injected_noul to preserve input dependency & batch B
        nodes.extend(
            [
                helper.make_node("Mul", ["risk_scaled", "c_f32_0"], ["risk_zero"]),
                helper.make_node("Add", ["risk_zero", "c_injected_noul"], ["noul_probs"]),
            ]
        )

    nodes.extend(
        [
            # 6. Source Pointer Head (<|q_src|> at index 3 vs pchunk_positions [B, 33])
            helper.make_node("MatMul", ["q_h1c", "w_src_ptr"], ["q_all_src_proj"]),
            helper.make_node(
                "Slice", ["q_all_src_proj", "sl_3", "sl_4", "ax_1"], ["q_src_proj"]
            ),
            helper.make_node("Shape", ["pchunk_positions"], ["p_pos_shape"]),
            helper.make_node(
                "Concat", ["p_pos_shape", "c_d_model_1d"], ["p_exp_shape"], axis=0
            ),
            helper.make_node("Unsqueeze", ["pchunk_positions", "ax_2"], ["p_pos_3d"]),
            helper.make_node("Expand", ["p_pos_3d", "p_exp_shape"], ["p_idx_exp"]),
            helper.make_node("GatherElements", ["h1_clip", "p_idx_exp"], ["p_h1c"], axis=1),
            helper.make_node("Transpose", ["p_h1c"], ["p_h1c_t"], perm=[0, 2, 1]),
            helper.make_node("MatMul", ["q_src_proj", "p_h1c_t"], ["src_logits_3d"]),
            helper.make_node("Squeeze", ["src_logits_3d", "ax_1"], ["src_logits_2d"]),
            helper.make_node(
                "Where",
                ["pchunk_valid_mask", "src_logits_2d", "c_f32_neg_mask"],
                ["src_masked_logits"],
            ),
            helper.make_node("Softmax", ["src_masked_logits"], ["src_choice_probs"], axis=-1),
            # 7. Argument Pointer Head (<|q_arg|> at index 4 vs arg_positions [B, 17])
            helper.make_node("MatMul", ["q_h1c", "w_arg_ptr"], ["q_all_arg_proj"]),
            helper.make_node(
                "Slice", ["q_all_arg_proj", "sl_4", "sl_5", "ax_1"], ["q_arg_proj"]
            ),
            helper.make_node("Shape", ["arg_positions"], ["a_pos_shape"]),
            helper.make_node(
                "Concat", ["a_pos_shape", "c_d_model_1d"], ["a_exp_shape"], axis=0
            ),
            helper.make_node("Unsqueeze", ["arg_positions", "ax_2"], ["a_pos_3d"]),
            helper.make_node("Expand", ["a_pos_3d", "a_exp_shape"], ["a_idx_exp"]),
            helper.make_node("GatherElements", ["h1_clip", "a_idx_exp"], ["a_h1c"], axis=1),
            helper.make_node("Transpose", ["a_h1c"], ["a_h1c_t"], perm=[0, 2, 1]),
            helper.make_node("MatMul", ["q_arg_proj", "a_h1c_t"], ["arg_logits_3d"]),
            helper.make_node("Squeeze", ["arg_logits_3d", "ax_1"], ["arg_logits_2d"]),
            helper.make_node(
                "Where",
                ["arg_valid_mask", "arg_logits_2d", "c_f32_neg_mask"],
                ["arg_masked_logits"],
            ),
            helper.make_node("Softmax", ["arg_masked_logits"], ["arg_choice_probs"], axis=-1),
        ]
    )

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["B", "L"]),
        helper.make_tensor_value_info("seg_ids", TensorProto.INT64, ["B", "L"]),
        helper.make_tensor_value_info("token_roles", TensorProto.INT64, ["B", "L"]),
        helper.make_tensor_value_info("query_positions", TensorProto.INT64, ["B", 5]),
        helper.make_tensor_value_info(
            "pchunk_positions", TensorProto.INT64, ["B", MAX_PCHUNKS + 1]
        ),
        helper.make_tensor_value_info(
            "pchunk_valid_mask", TensorProto.BOOL, ["B", MAX_PCHUNKS + 1]
        ),
        helper.make_tensor_value_info(
            "arg_positions", TensorProto.INT64, ["B", MAX_ARGS + 1]
        ),
        helper.make_tensor_value_info(
            "arg_valid_mask", TensorProto.BOOL, ["B", MAX_ARGS + 1]
        ),
    ]
    outputs = [
        helper.make_tensor_value_info("noul_probs", TensorProto.FLOAT, ["B", 3]),
        helper.make_tensor_value_info(
            "src_choice_probs", TensorProto.FLOAT, ["B", MAX_PCHUNKS + 1]
        ),
        helper.make_tensor_value_info(
            "arg_choice_probs", TensorProto.FLOAT, ["B", MAX_ARGS + 1]
        ),
    ]

    if tuple(inp.name for inp in inputs) != ONNX_INPUT_NAMES:
        raise RuntimeError("Synthetic ONNX inputs diverged from ONNX_INPUT_NAMES")
    if tuple(out.name for out in outputs) != ONNX_OUTPUT_NAMES:
        raise RuntimeError("Synthetic ONNX outputs diverged from ONNX_OUTPUT_NAMES")

    graph = helper.make_graph(
        nodes=nodes,
        name="SyntheticDiodeJevV1",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="diodemcp-test-fixture",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


def write_synthetic_jev_bundle(
    bundle_dir: Path,
    *,
    model_id: str = "diodemcp-jev-base-int8",
    length_buckets: list[int] | None = None,
    thresholds: dict[str, float] | None = None,
    inject_noul_values: tuple[float, float, float] | None = None,
    inject_src_nan: bool = False,
    inject_arg_nan: bool = False,
) -> Path:
    """Materialize a complete, SHA256-verified DiodeModelBundle v1 directory on disk."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    tok, special_ids, vocab = build_synthetic_tokenizer()

    onnx_bytes = build_synthetic_jev_onnx_bytes(
        vocab,
        inject_noul_values=inject_noul_values,
        inject_src_nan=inject_src_nan,
        inject_arg_nan=inject_arg_nan,
    )
    tok_bytes = tok.to_str().encode("utf-8")
    calibration_payload: dict[str, Any] = {
        "method": "temperature_scaling",
        "temperatures": [1.0, 1.0, 1.0],
        "ece_val": 0.014,
        "num_bins": 15,
    }
    cal_bytes = json.dumps(calibration_payload, indent=2).encode("utf-8")

    (bundle_dir / "model_int8.onnx").write_bytes(onnx_bytes)
    (bundle_dir / "tokenizer.json").write_bytes(tok_bytes)
    (bundle_dir / "calibration.json").write_bytes(cal_bytes)

    manifest_payload: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "model_id": model_id,
        "sha256_model_onnx": hashlib.sha256(onnx_bytes).hexdigest(),
        "sha256_tokenizer_json": hashlib.sha256(tok_bytes).hexdigest(),
        "sha256_calibration_json": hashlib.sha256(cal_bytes).hexdigest(),
        "max_seq_len": MAX_SEQ_LEN,
        "length_buckets": length_buckets if length_buckets is not None else list(LENGTH_BUCKETS),
        "max_pchunks": MAX_PCHUNKS,
        "max_args": MAX_ARGS,
        "special_token_ids": special_ids,
        "segment_ids": dict(CANONICAL_SEGMENT_IDS),
        "thresholds": thresholds
        if thresholds is not None
        else {"warn_min": WARN_THRESHOLD, "critical_min": CRITICAL_THRESHOLD},
        "ece_val": 0.014,
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2), encoding="utf-8"
    )
    return bundle_dir


@pytest.fixture(scope="session")
def _cached_synthetic_bundle_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """Build the synthetic v1 bundle once per pytest session for fast per-test copies."""
    master_dir = tmp_path_factory.mktemp("master_jev_bundle")
    write_synthetic_jev_bundle(master_dir)
    return {
        name: (master_dir / name).read_bytes()
        for name in ("manifest.json", "model_int8.onnx", "tokenizer.json", "calibration.json")
    }


@pytest.fixture
def synthetic_jev_bundle_dir(
    tmp_path: Path,
    _cached_synthetic_bundle_files: dict[str, bytes],
) -> Path:
    """Return an isolated per-test DiodeModelBundle v1 directory backed by the opset-17 ONNX graph."""
    bundle_dir = tmp_path / "synthetic_jev_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for name, content in _cached_synthetic_bundle_files.items():
        (bundle_dir / name).write_bytes(content)
    return bundle_dir
