"""Strictly proper scoring losses + post-hoc temperature scaling for the Jev models (Fixes `I11`).

Training objective (all strictly proper, no target shifting, no asymmetric reward weighting):

``loss = mean_h[ head_weight_h * ( bce_weight * BCEWithLogits_h + brier_weight * (sigmoid(z_h) - y_h)^2 ) ]
         + src_pointer_weight * CE(src pointer over valid 0..32 slots)
         + arg_pointer_weight * CE(arg pointer over valid 0..16 slots)``

* Targets ``y`` must be exactly ``{0.0, 1.0}``: any other value raises (no label smoothing, no
  "reward" targets). Cost asymmetry lives *only* in ``thresholds`` of the bundle manifest.
* Defaults ``bce_weight=1.0, brier_weight=0.5``: BCE supplies sharp, well-scaled gradients, while Brier is
  a bounded (<= 1) proper score that damps over-confident mistakes and directly optimises the
  calibration-relevant squared error; 0.5 keeps both terms on a comparable gradient scale
  (d BCE/dz = p - y vs d Brier/dz = 2(p - y)p(1 - p) <= 0.5|p - y|).
* Pointer cross-entropy is masked to the valid slots *before* the softmax. Supervision mirrors
  ``dataset_packer``: leak rows carry a real ``target_src_idx < 32``; non-leak rows carry the NONE slot
  (32); attack rows carry a real ``target_arg_idx < 16`` (or NONE=16 when the call has no arguments);
  benign rows carry NONE (16). Rows whose target equals ``ignore_index`` (default ``-100``) contribute
  nothing, so a trainer can disable pointer supervision per row (e.g. in Stage-1 or for mined rows).
* ``sample_weights`` is the hard-negative-mining hook. Weights bias the *training* distribution; always
  re-fit temperatures on an unweighted ``val_calib`` split with :func:`fit_temperature_scaling`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_IGNORE_INDEX: Final[int] = -100
DEFAULT_ECE_BINS: Final[int] = 15
TEMPERATURE_BOUNDS: Final[tuple[float, float]] = (0.1, 10.0)
_EPS_DEN: Final[float] = 1e-12


@dataclass(frozen=True)
class JevLossConfig:
    """Loss weights. Deliberately has no label-smoothing / pos_weight / target-shift knobs (I11)."""

    bce_weight: float = 1.0
    brier_weight: float = 0.5
    head_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # enc, sem, hijack
    src_pointer_weight: float = 1.0
    arg_pointer_weight: float = 1.0
    ignore_index: int = DEFAULT_IGNORE_INDEX


# ---------------------------------------------------------------------------------------------
# Elementary proper scores
# ---------------------------------------------------------------------------------------------
def _check_binary_targets(y: torch.Tensor) -> None:
    ok = (y == 0) | (y == 1)
    if not bool(ok.all()):
        bad = y[~ok].flatten()[:3].tolist()
        raise ValueError(
            "targets must be exactly binary {0, 1} for strictly proper scoring "
            f"(no smoothing / shifted targets); got {bad}"
        )


def brier_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Unreduced Brier score ``(sigmoid(z) - y)^2`` with the same shape as ``logits``."""
    return (torch.sigmoid(logits.float()) - targets.float()) ** 2


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None, dim: int = 0) -> torch.Tensor:
    if weights is None:
        return values.mean(dim=dim)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp(min=_EPS_DEN)


def _normalise_sample_weights(
    sample_weights: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    allow_per_head: bool = False,
    name: str = "sample_weights",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Validate finiteness, non-negativity, shape, and device of ``sample_weights``.

    Returns ``(per_head_weights[N, 3] | None, per_row_weights[N] | None)`` on ``(device, float32)``.
    When ``allow_per_head=False`` (pointer weights), only ``(N,)`` is accepted so 2D ``(N, 1)`` /
    ``(N, 3)`` tensors cannot silently broadcast to ``(N, N)``.
    """
    if sample_weights is None:
        return None, None
    w = torch.as_tensor(sample_weights).to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError(f"{name} must be finite and non-negative")
    if w.shape == (batch_size,):
        if allow_per_head:
            return w[:, None].expand(batch_size, 3), w
        return None, w
    if allow_per_head and w.shape == (batch_size, 3):
        return w, None
    expected = "(N,) or (N, 3)" if allow_per_head else "(N,)"
    raise ValueError(
        f"{name} must have shape {expected} with N={batch_size}; got {tuple(w.shape)}"
    )


def masked_pointer_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    sample_weights: torch.Tensor | None = None,
    validate_targets: bool = True,
) -> torch.Tensor:
    """Pointer CE over the valid slots only (masked before softmax); last slot is always valid.

    Rows with ``targets == ignore_index`` are excluded. Returns an exact (grad-connected) zero when every
    row is ignored. A non-ignored target on an invalid slot is a data bug and raises ``ValueError``.
    """
    logits = logits.float()
    _, row_w = _normalise_sample_weights(
        sample_weights,
        batch_size=logits.shape[0],
        device=logits.device,
        allow_per_head=False,
        name="sample_weights",
    )
    valid = torch.cat([valid_mask[:, :-1], torch.ones_like(valid_mask[:, -1:])], dim=1)
    active = targets != ignore_index
    if validate_targets and bool(active.any()):
        t_act = targets[active]
        n_slots = logits.shape[1]
        if bool(((t_act < 0) | (t_act >= n_slots)).any()):
            raise ValueError(f"pointer target outside [0, {n_slots - 1}] (invalid target)")
        picked = valid[active].gather(1, t_act.unsqueeze(1)).squeeze(1)
        if not bool(picked.all()):
            raise ValueError("pointer target refers to an invalid (masked) slot")
    masked = logits.masked_fill(~valid, float("-inf"))
    safe_targets = torch.where(active, targets, torch.zeros_like(targets))
    per_row = F.cross_entropy(masked, safe_targets, reduction="none")
    per_row = torch.where(active, per_row, torch.zeros_like(per_row))
    weights = active.to(per_row.dtype)
    if row_w is not None:
        weights = weights * row_w.to(per_row.dtype)
    return (per_row * weights).sum() / weights.sum().clamp(min=_EPS_DEN)


def compute_jev_loss(
    noul_logits: torch.Tensor,
    src_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    noul_targets: torch.Tensor,
    target_src_idx: torch.Tensor,
    target_arg_idx: torch.Tensor,
    pchunk_valid_mask: torch.Tensor,
    arg_valid_mask: torch.Tensor,
    *,
    config: JevLossConfig | None = None,
    sample_weights: torch.Tensor | None = None,
    pointer_sample_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute the strictly proper Jev loss. Returns ``loss`` plus reporting components.

    ``sample_weights``: ``[N]`` (all heads + pointers) or ``[N, 3]`` (noul heads only).
    ``pointer_sample_weights``: optional ``[N]`` override for the two pointer heads.
    """
    cfg = config or JevLossConfig()
    y = noul_targets.to(device=noul_logits.device, dtype=torch.float32)
    _check_binary_targets(y)
    n = noul_logits.shape[0]
    head_w, row_w = _normalise_sample_weights(
        sample_weights,
        batch_size=n,
        device=noul_logits.device,
        allow_per_head=True,
        name="sample_weights",
    )

    logits = noul_logits.float()
    bce_ph = _weighted_mean(F.binary_cross_entropy_with_logits(logits, y, reduction="none"), head_w)
    brier_ph = _weighted_mean(brier_loss(logits, y), head_w)
    hw = torch.tensor(cfg.head_weights, dtype=bce_ph.dtype, device=bce_ph.device)
    noul_term = (hw * (cfg.bce_weight * bce_ph + cfg.brier_weight * brier_ph)).mean()

    ptr_w = row_w
    if pointer_sample_weights is not None:
        _, ptr_w = _normalise_sample_weights(
            pointer_sample_weights,
            batch_size=n,
            device=src_logits.device,
            allow_per_head=False,
            name="pointer_sample_weights",
        )
    src_ce = masked_pointer_cross_entropy(
        src_logits, target_src_idx, pchunk_valid_mask, ignore_index=cfg.ignore_index, sample_weights=ptr_w
    )
    arg_ce = masked_pointer_cross_entropy(
        arg_logits, target_arg_idx, arg_valid_mask, ignore_index=cfg.ignore_index, sample_weights=ptr_w
    )
    loss = noul_term + cfg.src_pointer_weight * src_ce + cfg.arg_pointer_weight * arg_ce
    return {
        "loss": loss,
        "bce": bce_ph.mean(),
        "brier": brier_ph.mean(),
        "bce_per_head": bce_ph,
        "brier_per_head": brier_ph,
        "src_pointer_ce": src_ce,
        "arg_pointer_ce": arg_ce,
    }


# ---------------------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------------------
def _to_float64_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().double().numpy()
    return np.asarray(x, dtype=np.float64)


def expected_calibration_error(
    probs: Any, targets: Any, num_bins: int = DEFAULT_ECE_BINS
) -> float:
    """Equal-width-bin ECE of binary positive-class probabilities (1-D)."""
    p = _to_float64_numpy(probs)
    y = _to_float64_numpy(targets)
    if p.ndim != 1 or p.shape != y.shape:
        raise ValueError(f"expected 1-D probs/targets of equal shape, got {p.shape} / {y.shape}")
    if p.size == 0:
        return 0.0
    bins = np.minimum((p * num_bins).astype(np.int64), num_bins - 1)
    bins = np.maximum(bins, 0)
    ece = 0.0
    for b in range(num_bins):
        sel = bins == b
        if sel.any():
            ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(ece)


def _nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    z = logits / temperature
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_temperature_scaling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
    max_iter: int = 200,
) -> torch.Tensor:
    """Per-head post-hoc temperature scaling by L-BFGS on NLL (``sigmoid(z / T)``), ``T = exp(u)``.

    ``logits`` / ``targets``: ``[N, H]`` (H = 3: enc, sem, hijack). Returns ``float32[H]`` ready for
    ``model.set_logit_temperatures`` / ``calibration.json``. A head whose targets are single-class has
    no finite NLL optimum (``T -> 0``) so it keeps ``T = 1.0``; a fit that fails to beat ``T = 1`` also
    returns ``1.0``. Results are clamped to ``bounds``.
    """
    z_all = logits.detach().cpu().double()
    y_all = targets.detach().cpu().double()
    if z_all.ndim != 2 or z_all.shape != y_all.shape:
        raise ValueError(f"logits/targets must be [N, H] with equal shape, got {tuple(z_all.shape)} / {tuple(y_all.shape)}")
    _check_binary_targets(y_all)
    lo, hi = bounds
    temps: list[float] = []
    for h in range(z_all.shape[1]):
        z, y = z_all[:, h], y_all[:, h]
        if z.numel() == 0 or float(y.min()) == float(y.max()) or not bool(torch.isfinite(z).all()):
            temps.append(1.0)
            continue
        u = torch.zeros((), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([u], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure(u=u, z=z, y=y) -> torch.Tensor:
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(z / torch.exp(u), y)
            loss.backward()
            return loss

        opt.step(closure)
        t = float(torch.exp(u.detach()))
        t = min(max(t, lo), hi) if math.isfinite(t) else 1.0
        zn, yn = z.numpy(), y.numpy()
        temps.append(t if _nll(zn, yn, t) <= _nll(zn, yn, 1.0) else 1.0)
    return torch.tensor(temps, dtype=torch.float32)


def temperature_calibration_report(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperatures: torch.Tensor,
    num_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, Any]:
    """ECE/NLL before (T=1) and after ``temperatures`` per head, plus a ``calibration.json`` payload.

    ``ece_val`` is the macro mean over heads of the *after* ECE (the manifest's single ``ece_val``).
    """
    z_t = logits.detach().cpu().double()
    t_t = temperatures.detach().cpu().double()
    z = z_t.numpy()
    y = targets.detach().cpu().double().numpy()
    t = t_t.numpy()
    ece_b, ece_a, nll_b, nll_a = [], [], [], []
    for h in range(z.shape[1]):
        p_before = torch.sigmoid(z_t[:, h]).numpy()
        p_after = torch.sigmoid(z_t[:, h] / t_t[h]).numpy()
        ece_b.append(expected_calibration_error(p_before, y[:, h], num_bins))
        ece_a.append(expected_calibration_error(p_after, y[:, h], num_bins))
        nll_b.append(_nll(z[:, h], y[:, h], 1.0))
        nll_a.append(_nll(z[:, h], y[:, h], float(t[h])))
    ece_val = float(np.mean(ece_a))
    return {
        "ece_before": ece_b,
        "ece_after": ece_a,
        "nll_before": nll_b,
        "nll_after": nll_a,
        "ece_val": ece_val,
        "calibration": {
            "method": "temperature_scaling",
            "temperatures": [float(x) for x in temperatures.detach().cpu().tolist()],
            "ece_val": ece_val,
            "num_bins": int(num_bins),
        },
    }


def probs_to_logits(
    probs: Any, temperatures: Any | None = None, eps: float = 1e-7
) -> torch.Tensor:
    """Recover *raw* logits from exported probabilities: ``z = T * logit(p)`` (float64 internally).

    Used by A6 to refit temperatures on ORT-INT8 outputs whose ``T`` is already baked in. float32 probs
    saturate near 0/1 (resolution ~6e-8), so |z| beyond ~12 is not recoverable; such saturated samples
    carry ~zero NLL so the fit is unaffected. Probabilities are clamped to ``[eps, 1 - eps]``.
    """
    p = torch.as_tensor(probs).double().clamp(eps, 1.0 - eps)
    z = torch.log(p) - torch.log1p(-p)
    if temperatures is not None:
        z = z * torch.as_tensor(temperatures).double()
    return z.float()


__all__ = [
    "JevLossConfig",
    "brier_loss",
    "compute_jev_loss",
    "expected_calibration_error",
    "fit_temperature_scaling",
    "masked_pointer_cross_entropy",
    "probs_to_logits",
    "temperature_calibration_report",
]
