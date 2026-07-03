"""
Thin helpers around the vendored SCT `SpectralLinear` so the toy actually
exercises the library (not a re-implementation).

Energy target η -> rank r -> truncated dense weight W_r (for use as a forward
override). Reconstruction uses SCT's own factor convention:

    SpectralLinear.from_linear stores  U=(in×r), V=(out×r), s=(r,)
    forward y = (x@U)*s @ V.T  ==>  W_r (out×in) = V diag(s) U.T
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectral_compact_training import SpectralLinear


def rank_for_energy(W: torch.Tensor, energy: float) -> tuple[int, float]:
    """Smallest r with cumulative singular energy ≥ energy.
    Returns (rank, actual_discarded_energy_fraction)."""
    s = torch.linalg.svdvals(W.float().cpu())
    e = s ** 2
    cum = torch.cumsum(e, 0) / e.sum()
    r = int(torch.searchsorted(cum, torch.tensor(float(energy))).item()) + 1
    r = max(1, min(r, W.shape[0], W.shape[1]))
    discarded = 1.0 - (e[:r].sum() / e.sum()).item()
    return r, discarded


def sct_truncate(W: torch.Tensor, energy: float):
    """Return (W_r, rank, params, discarded_fraction) for dense weight W (out×in)
    truncated to the given energy target, via SCT's SpectralLinear."""
    out_f, in_f = W.shape
    r, discarded = rank_for_energy(W, energy)
    lin = nn.Linear(in_f, out_f, bias=False)
    lin.weight.data = W.detach().float().cpu()
    layer = SpectralLinear.from_linear(lin, rank=r)
    W_r = (layer.V @ torch.diag(layer.s) @ layer.U.T).to(W.device, W.dtype)
    params = layer.param_count()  # r*(in+out+1)
    return W_r, r, params, discarded


def discarded_energy(W: torch.Tensor, r: int) -> float:
    """Σ_{i>r} σ_i^2  (absolute discarded spectral energy) — the Eckart–Young
    truncation error ||W - W_r||_F^2."""
    s = torch.linalg.svdvals(W.float().cpu())
    return (s[r:] ** 2).sum().item()
