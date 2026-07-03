"""
Unified linear-attention + linear-MLP toy model (Part B.1).
================================================================================

Architecture (all linear, no softmax, no nonlinearity):

    Q = X Wq^T ,  K = X Wk^T ,  V = X Wv^T          (d_in -> d_head projections)
    A = (Q K^T) / d_head                            (linear-attention scores, T×T)
    C = A V                                          (context, T×d_head)
    h = C Wm^T                                       (linear MLP,  d_head -> d_hidden)
    O = h Wo^T                                       (readout,     d_hidden -> d_out)
    loss = mean_t || O_t - Y_t ||^2                  (regression MSE)

WHY THIS SHAPE — the property we exploit
----------------------------------------
Holding every weight matrix *except one* fixed, the map  W_i -> O  is LINEAR in
W_i (attention is bilinear in (Wq,Wk) but linear in each separately; the rest of
the chain is linear). Therefore, for a perturbation ΔW_i of a single matrix:

    O(W_i + ΔW_i) - O(W_i)  =  ΔO  is EXACTLY linear in ΔW_i, and
    L(W_i + ΔW_i) - L(W_i)  =  E||ΔO||^2  +  2 E[(O - Y)·ΔO]
                               └ quadratic ┘   └ linear (→0 at the optimum) ┘

So near a trained optimum (grad ≈ 0 ⇒ the linear term vanishes because
E[(O−Y)·ΔO] is the projection of the residual onto the perturbation direction,
which the normal equations kill), the downstream loss is EXACTLY quadratic in
ΔW_i. That is precisely A.1's "loss is locally quadratic in the weight
perturbation", here checkable as an equality rather than a hand-wave.

`quadratic_vs_measured_deltaL` below returns *both* numbers so we can watch the
linear (cross) term shrink to zero as training converges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn


LAYER_NAMES = ["q_proj", "k_proj", "v_proj", "mlp", "readout"]


def spectrum_weight(out_features: int, in_features: int, decay: float,
                    seed: int, device, dtype) -> torch.Tensor:
    """Random weight W (out×in) with a *prescribed geometric* singular spectrum
    σ_i = decay**i (then renormalised to unit spectral norm). A decaying spectrum
    is what makes energy-based rank truncation meaningful (fast decay ⇒ a few
    ranks capture most energy ⇒ cheap SCT compression — the A.1 story)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = min(out_features, in_features)
    U = torch.linalg.qr(torch.randn(out_features, k, generator=g))[0]
    V = torch.linalg.qr(torch.randn(in_features, k, generator=g))[0]
    s = decay ** torch.arange(k, dtype=torch.float64)
    s = (s / s[0]).to(torch.float32)
    W = (U * s) @ V.T
    return W.to(device=device, dtype=dtype)


@dataclass
class ToyConfig:
    d_in: int = 64
    d_head: int = 64          # = TurboQuant head_dim (codebooks exist for 64)
    d_hidden: int = 64
    d_out: int = 64
    T: int = 16
    attn_scale: float = None  # default 1/d_head, set in __post_init__
    spectrum_decay: float = 0.90  # teacher singular-value decay per matrix

    def __post_init__(self):
        if self.attn_scale is None:
            self.attn_scale = 1.0 / self.d_head


class LinearAttnToy(nn.Module):
    """The student (and, when `teacher=True`, the fixed teacher)."""

    def __init__(self, cfg: ToyConfig, teacher: bool = False, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_in, cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_in, cfg.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_in, cfg.d_head, bias=False)
        self.mlp = nn.Linear(cfg.d_head, cfg.d_hidden, bias=False)
        self.readout = nn.Linear(cfg.d_hidden, cfg.d_out, bias=False)

        if teacher:
            # Fixed teacher: prescribed decaying spectrum per matrix so that
            # (a) the target is well defined and (b) the trained student inherits
            # a decaying spectrum SCT can compress.
            base = 100 + seed
            self.q_proj.weight.data = spectrum_weight(cfg.d_head, cfg.d_in, cfg.spectrum_decay, base + 0, "cpu", torch.float32)
            self.k_proj.weight.data = spectrum_weight(cfg.d_head, cfg.d_in, cfg.spectrum_decay, base + 1, "cpu", torch.float32)
            self.v_proj.weight.data = spectrum_weight(cfg.d_head, cfg.d_in, cfg.spectrum_decay, base + 2, "cpu", torch.float32)
            self.mlp.weight.data = spectrum_weight(cfg.d_hidden, cfg.d_head, cfg.spectrum_decay, base + 3, "cpu", torch.float32)
            self.readout.weight.data = spectrum_weight(cfg.d_out, cfg.d_hidden, cfg.spectrum_decay, base + 4, "cpu", torch.float32)

    # ---- forward with optional per-layer weight overrides -------------------
    def _w(self, name: str, overrides: Optional[Dict[str, torch.Tensor]]):
        if overrides is not None and name in overrides:
            return overrides[name]
        return getattr(self, name).weight

    def forward(self, X: torch.Tensor,
                overrides: Optional[Dict[str, torch.Tensor]] = None,
                return_activations: bool = False):
        """X: (B,T,d_in) -> O: (B,T,d_out).

        `overrides` maps a layer name to a replacement weight tensor (used to
        inject SCT-truncated weights without mutating the module). This is the
        single code path used by every arm, so measured/predicted numbers are
        always computed against the identical forward.
        """
        Wq = self._w("q_proj", overrides)
        Wk = self._w("k_proj", overrides)
        Wv = self._w("v_proj", overrides)
        Wm = self._w("mlp", overrides)
        Wo = self._w("readout", overrides)

        Q = X @ Wq.T
        K = X @ Wk.T
        V = X @ Wv.T
        A = (Q @ K.transpose(-2, -1)) * self.cfg.attn_scale
        C = A @ V
        h = C @ Wm.T
        O = h @ Wo.T
        if return_activations:
            return O, {"Q": Q, "K": K, "V": V, "A": A, "C": C, "h": h}
        return O


def mse_loss(O: torch.Tensor, Y: torch.Tensor) -> float:
    """Mean squared error per token (summed over output dims, averaged over B,T)."""
    return ((O - Y) ** 2).sum(dim=-1).mean().item()


@torch.no_grad()
def population_loss(model: LinearAttnToy, X: torch.Tensor, Y: torch.Tensor,
                    overrides=None) -> float:
    O = model(X, overrides=overrides)
    return mse_loss(O, Y)


@torch.no_grad()
def quadratic_vs_measured_deltaL(model: LinearAttnToy, X: torch.Tensor,
                                 Y: torch.Tensor, layer: str,
                                 W_replacement: torch.Tensor,
                                 baseline_O: torch.Tensor,
                                 baseline_loss: float):
    """Return (measured ΔL, pure-quadratic ΔL, linear/cross-term) for replacing
    a *single* layer's weight with `W_replacement` (others frozen).

        measured   = L(perturbed) - L(baseline)                 [the truth]
        quadratic  = E|| O_perturbed - O_baseline ||^2          [the A.1 model]
        cross      = measured - quadratic  (= 2 E[(O-Y)·ΔO])    [→0 at optimum]

    If |cross| << quadratic, the downstream loss is (locally) quadratic in ΔW.
    """
    O_pert = model(X, overrides={layer: W_replacement})
    measured = mse_loss(O_pert, Y) - baseline_loss
    dO = O_pert - baseline_O
    quadratic = (dO ** 2).sum(dim=-1).mean().item()
    cross = measured - quadratic
    return measured, quadratic, cross


def train_student(cfg: ToyConfig, data, teacher: LinearAttnToy, steps: int = 6000,
                  batch: int = 256, lr: float = 8e-3, device=None, log=print):
    """Train a dense student to (near) the population optimum with online
    Gaussian batches + cosine LR decay, so the student sits at a genuine loss
    minimum before we SCT-factor / TurboQuant-quantize it.

    Driving the loss to ~1e-8 matters: the A.1 local-quadratic equality only
    holds once the residual (O−Y) is tiny compared with the truncation-induced
    ΔO, so that the linear cross-term 2E[(O−Y)·ΔO] is negligible. Returns the
    trained student."""
    device = device or torch.device("cpu")
    student = LinearAttnToy(cfg, teacher=False, seed=7).to(device)
    teacher = teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    log(f"[train] steps={steps} batch={batch} lr={lr} (cosine) device={device}")
    for step in range(steps):
        X = data.sample(batch).to(device)
        with torch.no_grad():
            Y = teacher(X)
        O = student(X)
        loss = ((O - Y) ** 2).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            log(f"[train]  step {step:5d}  loss={loss.item():.6e}")
    return student
