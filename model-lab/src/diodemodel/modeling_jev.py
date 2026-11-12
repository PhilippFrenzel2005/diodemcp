"""ModernBERT-Jev (base/large) and NanoJev (Qwen3-0.6B): `token_roles`-masked Jev cross-encoders.

Both models expose the exact 8-input `DiodeModelBundle v1` forward signature and return
``(noul_probs[B,3], src_choice_probs[B,33], arg_choice_probs[B,17])``; ``forward_logits`` returns the
uncalibrated, already pointer-masked logits for numerically stable training and post-hoc temperature
scaling.

Attention isolation (Rulings RA3 / R5 / C1 / I7 / I8) is built *inside* the graph from
``token_roles: int64[B, L]`` (``0=PAD, 1=STATE, 2=QUERY``) using plain ONNX-traceable tensor ops:

* STATE rows attend to STATE columns only (plus their own diagonal).
* QUERY rows attend to every STATE column plus their own diagonal -> zero cross-talk between queries.
* PAD rows attend only to themselves (diagonal ``0.0``) so no row is ever fully masked (no NaN in
  fp16/bf16/INT8); STATE/QUERY rows block PAD columns, so right-padding never changes a real token.

How the mask reaches the backbone (verified against ``transformers`` 5.x): both
``ModernBertModel.forward`` and ``Qwen3Model.forward`` accept ``attention_mask`` as a *dict keyed by layer
type* (``{"full_attention": ..., "sliding_attention": ...}``) and then skip their own mask creation
entirely (`if not isinstance(attention_mask_mapping := attention_mask, dict)`). The eager attention
kernel simply adds the 4D additive mask to the scores. We therefore need no monkey-patching, and the
causal handling of Qwen3 is *not* re-applied: our own mask already contains the causal lower triangle.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import torch
from torch import nn
from transformers import ModernBertModel, Qwen3Model

from diodemodel.contract import (
    CANONICAL_SEGMENT_IDS,
    ROLE_QUERY,
    ROLE_STATE,
)

MASK_NEG: Final[float] = -10000.0
NUM_SEGMENTS: Final[int] = len(CANONICAL_SEGMENT_IDS)  # 6
NUM_NOUL_HEADS: Final[int] = 3  # <|q_enc|>, <|q_sem|>, <|q_hijack|>
SRC_SLOTS: Final[int] = 33  # 32 <|pchunk|> + <|none|> (slot 32)
ARG_SLOTS: Final[int] = 17  # 16 <|arg|> + <|none|> (slot 16)
NUM_QUERY_TOKENS: Final[int] = 5  # enc, sem, hijack, src, arg
POINTER_NONE_SLOTS: Final[tuple[int, int]] = (SRC_SLOTS - 1, ARG_SLOTS - 1)

#: ONNX/trace scope names of the FP32-only heads (A6 uses them for `nodes_to_exclude`).
HEAD_SCOPE_NAMES: Final[tuple[str, ...]] = ("noul_head", "src_pointer", "arg_pointer")

#: Dynamic axes for `torch.onnx.export` (batch + sequence dynamic; slot dims are static by contract).
ONNX_DYNAMIC_AXES: Final[dict[str, dict[int, str]]] = {
    "input_ids": {0: "batch", 1: "seq"},
    "seg_ids": {0: "batch", 1: "seq"},
    "token_roles": {0: "batch", 1: "seq"},
    "query_positions": {0: "batch"},
    "pchunk_positions": {0: "batch"},
    "pchunk_valid_mask": {0: "batch"},
    "arg_positions": {0: "batch"},
    "arg_valid_mask": {0: "batch"},
    "noul_probs": {0: "batch"},
    "src_choice_probs": {0: "batch"},
    "arg_choice_probs": {0: "batch"},
}


def build_token_roles_attn_mask(
    token_roles: torch.Tensor,
    *,
    causal: bool = False,
    sliding_window: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the additive ``[B, 1, L, L]`` attention mask (``0.0`` open / ``-10000.0`` blocked).

    With ``causal=False, sliding_window=None`` this is numerically identical to
    ``dataset_packer.build_reference_4d_attn_mask`` / Track B ``packer.build_reference_4d_attn_mask``.

    * ``causal=True`` additionally keeps only ``j <= i`` (NanoJev decoder backbone). QUERY tokens sit at
      the end of the row, so they still read every preceding STATE token and only themselves.
    * ``sliding_window=w`` intersects **STATE rows only** with ``|i - j| <= w`` (ModernBERT local layers,
      ``w = config.sliding_window`` is the half-width, matching HF). QUERY rows keep their full STATE
      receptive field; PAD rows stay diagonal-only.

    Only ONNX-traceable tensor ops are used (no Python control flow on tensor values, no ``eye``/``arange``
    on traced sizes: positions come from a ``cumsum`` of ones so the graph is fully dynamic in ``L``).
    Role values outside ``{0, 1, 2}`` behave like PAD (diagonal only).
    """
    roles = token_roles
    is_state = roles == ROLE_STATE
    is_query = roles == ROLE_QUERY
    positions = torch.cumsum(torch.ones_like(roles), dim=1) - 1
    rel = positions.unsqueeze(2) - positions.unsqueeze(1)  # i - j
    diag = rel == 0
    allowed = ((is_state | is_query).unsqueeze(2) & is_state.unsqueeze(1)) | diag
    if causal:
        allowed = allowed & (rel >= 0)
    if sliding_window is not None:
        within = rel.abs() <= int(sliding_window)
        # Boolean logic only (ORT CPU has no `Where` kernel for bool tensors).
        allowed = allowed & (within | ~is_state.unsqueeze(2))
    open_val = torch.zeros((), dtype=dtype, device=roles.device)
    blocked_val = torch.full((), MASK_NEG, dtype=dtype, device=roles.device)
    return torch.where(allowed.unsqueeze(1), open_val, blocked_val)


class PointerHead(nn.Module):
    """Scaled dot-product pointer over gathered slot hidden states, masked *before* the softmax."""

    def __init__(self, hidden_size: int, pointer_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, pointer_dim)
        self.k_proj = nn.Linear(hidden_size, pointer_dim)
        self.scale = float(pointer_dim) ** -0.5
        for lin in (self.q_proj, self.k_proj):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)

    def forward(
        self, query_hidden: torch.Tensor, slot_hidden: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        q = self.q_proj(query_hidden)  # [B, d]
        k = self.k_proj(slot_hidden)  # [B, N, d]
        scores = torch.matmul(k, q.unsqueeze(-1)).squeeze(-1) * self.scale  # [B, N]
        # The last slot is `<|none|>` and is ALWAYS valid (guarantees no all-masked softmax row).
        valid = torch.cat([valid_mask[:, :-1], torch.ones_like(valid_mask[:, -1:])], dim=1)
        neg = torch.full((), MASK_NEG, dtype=scores.dtype, device=scores.device)
        return torch.where(valid, scores, neg)


class NoulHead(nn.Module):
    """Shared dense+GELU transform with a per-query linear read-out for enc / sem / hijack."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_weight = nn.Parameter(torch.empty(NUM_NOUL_HEADS, hidden_size))
        self.out_bias = nn.Parameter(torch.zeros(NUM_NOUL_HEADS))
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.out_weight, std=0.02)

    def forward(self, query_hidden: torch.Tensor) -> torch.Tensor:  # [B, 3, H] -> [B, 3]
        h = torch.nn.functional.gelu(self.dense(query_hidden))
        return (h * self.out_weight).sum(dim=-1) + self.out_bias


def pin_onnx_output_shapes(model: Any) -> Any:
    """Pin the contract's static output dims on a traced ONNX ``ModelProto`` (in place; also returned).

    ``torch.onnx.export`` (tracer, ``dynamo=False``) reports the static slot dimension of the three
    outputs as a symbolic ``*_dim_1`` even though ONNX shape inference proves it is 3 / 33 / 17, which
    makes ``contract.validate_onnx_contract`` reject the graph. Call this on the loaded export (A6 does
    so before quantisation / saving) to restore ``[batch, 3]``, ``[batch, 33]``, ``[batch, 17]``.
    """
    static = {
        "noul_probs": NUM_NOUL_HEADS,
        "src_choice_probs": SRC_SLOTS,
        "arg_choice_probs": ARG_SLOTS,
    }
    for out in model.graph.output:
        width = static.get(out.name)
        if width is None:
            continue
        dims = out.type.tensor_type.shape.dim
        while len(dims) > 0:
            dims.pop()
        dims.add().dim_param = "batch"
        dims.add().dim_value = width
    return model


def _gather_positions(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather ``hidden[b, positions[b, n]]`` -> ``[B, N, H]`` (positions clamped into ``[0, L-1]``)."""
    seq_len = hidden.shape[1]
    safe = positions.clamp(min=0, max=seq_len - 1)
    index = safe.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    return torch.gather(hidden, 1, index)


class _JevModelBase(nn.Module):
    """Shared Jev heads + segment embedding + temperature buffer around an eager HF backbone."""

    #: HF backbone class (set by subclasses).
    backbone_cls: type
    #: Whether STATE rows use a causal lower-triangular mask (decoder backbone).
    causal: bool = False

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_segments: int = NUM_SEGMENTS,
        pointer_dim: int | None = None,
    ) -> None:
        super().__init__()
        impl = getattr(backbone.config, "_attn_implementation", None)
        if impl != "eager":
            raise ValueError(
                "Jev models require attn_implementation='eager' (additive 4D token_roles mask + "
                f"deterministic opset-17 ONNX export); backbone uses {impl!r}"
            )
        hidden = int(backbone.config.hidden_size)
        self.backbone = backbone
        self.seg_embedding = nn.Embedding(num_segments, hidden)
        nn.init.normal_(self.seg_embedding.weight, std=0.02)
        dim = int(pointer_dim) if pointer_dim else hidden
        self.noul_head = NoulHead(hidden)
        self.src_pointer = PointerHead(hidden, dim)
        self.arg_pointer = PointerHead(hidden, dim)
        self.register_buffer("logit_temperatures", torch.ones(NUM_NOUL_HEADS, dtype=torch.float32))

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> _JevModelBase:
        """Build from a (tiny) config with random weights: no network, eager attention."""
        backbone = cls.backbone_cls._from_config(config, attn_implementation="eager")
        return cls(backbone, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        vocab_size: int | None = None,
        num_segments: int = NUM_SEGMENTS,
        pointer_dim: int | None = None,
        **backbone_kwargs: Any,
    ) -> _JevModelBase:
        """Load backbone weights (HF id or local dir) with eager attention; heads are freshly initialised.

        ``vocab_size`` (``len(jev_tokenizer)``) grows the embedding matrix for the 17 added special
        tokens (never shrinks it; Qwen3's 151936-row table already covers the new ids).
        """
        backbone_kwargs.pop("attn_implementation", None)
        backbone = cls.backbone_cls.from_pretrained(
            name_or_path, attn_implementation="eager", **backbone_kwargs
        )
        if vocab_size is not None and vocab_size > backbone.get_input_embeddings().num_embeddings:
            backbone.resize_token_embeddings(int(vocab_size))
        return cls(backbone, num_segments=num_segments, pointer_dim=pointer_dim)

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        self.backbone.gradient_checkpointing_enable(**kwargs)

    # ------------------------------------------------------------------ temperatures
    def set_logit_temperatures(self, temperatures: torch.Tensor) -> None:
        """Write per-head temperatures (enc, sem, hijack) into the persistent buffer."""
        t = torch.as_tensor(temperatures, dtype=torch.float32).detach().reshape(-1)
        if t.shape != (NUM_NOUL_HEADS,):
            raise ValueError(f"logit_temperatures must have shape (3,), got {tuple(t.shape)}")
        if not bool(torch.isfinite(t).all()) or bool((t <= 0).any()):
            raise ValueError(f"logit_temperatures must be finite and > 0, got {t.tolist()}")
        with torch.no_grad():
            self.logit_temperatures.copy_(t.to(self.logit_temperatures.device))

    # ------------------------------------------------------------------ masks / encoder
    def build_attention_masks(
        self, token_roles: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def embed(self, input_ids: torch.Tensor, seg_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_input_embeddings()(input_ids) + self.seg_embedding(seg_ids)

    def _run_backbone(
        self, embeds: torch.Tensor, masks: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.backbone(inputs_embeds=embeds, attention_mask=dict(masks)).last_hidden_state

    def encode(
        self, input_ids: torch.Tensor, seg_ids: torch.Tensor, token_roles: torch.Tensor
    ) -> torch.Tensor:
        """Return final hidden states ``[B, L, H]`` under the `token_roles` mask."""
        embeds = self.embed(input_ids, seg_ids)
        masks = self.build_attention_masks(token_roles, dtype=embeds.dtype)
        return self._run_backbone(embeds, masks)

    # ------------------------------------------------------------------ heads
    def forward_logits(
        self,
        input_ids: torch.Tensor,
        seg_ids: torch.Tensor,
        token_roles: torch.Tensor,
        query_positions: torch.Tensor,
        pchunk_positions: torch.Tensor,
        pchunk_valid_mask: torch.Tensor,
        arg_positions: torch.Tensor,
        arg_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uncalibrated ``(noul_logits[B,3], src_logits[B,33], arg_logits[B,17])`` (pointers pre-masked)."""
        hidden = self.encode(input_ids, seg_ids, token_roles)
        q = _gather_positions(hidden, query_positions)  # [B, 5, H]
        noul_logits = self.noul_head(q[:, 0:3])
        src_logits = self.src_pointer(
            q[:, 3], _gather_positions(hidden, pchunk_positions), pchunk_valid_mask
        )
        arg_logits = self.arg_pointer(
            q[:, 4], _gather_positions(hidden, arg_positions), arg_valid_mask
        )
        return noul_logits.float(), src_logits.float(), arg_logits.float()

    def forward(
        self,
        input_ids: torch.Tensor,
        seg_ids: torch.Tensor,
        token_roles: torch.Tensor,
        query_positions: torch.Tensor,
        pchunk_positions: torch.Tensor,
        pchunk_valid_mask: torch.Tensor,
        arg_positions: torch.Tensor,
        arg_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(noul_probs, src_choice_probs, arg_choice_probs)`` with temperatures baked in."""
        noul_logits, src_logits, arg_logits = self.forward_logits(
            input_ids,
            seg_ids,
            token_roles,
            query_positions,
            pchunk_positions,
            pchunk_valid_mask,
            arg_positions,
            arg_valid_mask,
        )
        noul_probs = torch.sigmoid(noul_logits / self.logit_temperatures)
        return (
            noul_probs,
            torch.softmax(src_logits, dim=-1),
            torch.softmax(arg_logits, dim=-1),
        )


class ModernBERTJevModel(_JevModelBase):
    """ModernBERT-base / -large backbone (bidirectional): global layers use the reference mask, local
    (sliding-window) layers intersect STATE rows with the window while QUERY rows read STATE fully."""

    backbone_cls = ModernBertModel
    causal = False

    def build_attention_masks(
        self, token_roles: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> dict[str, torch.Tensor]:
        cfg = self.backbone.config
        masks = {"full_attention": build_token_roles_attn_mask(token_roles, dtype=dtype)}
        if "sliding_attention" in cfg.layer_types:
            masks["sliding_attention"] = build_token_roles_attn_mask(
                token_roles, sliding_window=int(cfg.sliding_window), dtype=dtype
            )
        return masks


class NanoJevCausalModel(_JevModelBase):
    """Qwen3-0.6B decoder backbone (NanoJev): causal STATE; QUERY tokens at the end read every preceding
    STATE token plus themselves only. HF never re-applies causality because we pass a dict mask."""

    backbone_cls = Qwen3Model
    causal = True

    def build_attention_masks(
        self, token_roles: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> dict[str, torch.Tensor]:
        mask = build_token_roles_attn_mask(token_roles, causal=True, dtype=dtype)
        masks = {"full_attention": mask}
        if "sliding_attention" in getattr(self.backbone.config, "layer_types", ()):
            # Qwen3-0.6B ships without sliding layers; if a config enables them they share the mask
            # (a causal STATE+QUERY mask is already a superset-safe receptive field).
            masks["sliding_attention"] = mask
        return masks

    def _run_backbone(
        self, embeds: torch.Tensor, masks: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.backbone(
            inputs_embeds=embeds, attention_mask=dict(masks), use_cache=False
        ).last_hidden_state


__all__ = [
    "ARG_SLOTS",
    "HEAD_SCOPE_NAMES",
    "MASK_NEG",
    "ONNX_DYNAMIC_AXES",
    "SRC_SLOTS",
    "ModernBERTJevModel",
    "NanoJevCausalModel",
    "NoulHead",
    "PointerHead",
    "build_token_roles_attn_mask",
    "pin_onnx_output_shapes",
]
