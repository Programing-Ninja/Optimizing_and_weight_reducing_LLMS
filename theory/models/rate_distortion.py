"""
Part A deliverable — joint SCT × TurboQuant rate-distortion solver.
================================================================================
Given constants MEASURED by the toy (Part B):

    ΔL(η, b) ≈ α·(1−η)  +  β_p·2^(−p·b)          (SCT bias + TurboQuant variance)
                └ bias ┘        └ variance ┘

and a memory model M(η, b) = weight_bytes(η) + kv_bytes(b), this module:

  1. evaluates the ΔL(η, b) distortion surface and the memory surface,
  2. solves the budget-constrained problem  min ΔL  s.t.  M ≤ M_budget  (the
     rate-distortion problem of A.3), and
  3. reports the KKT "equal marginal loss per byte" split and the resulting
     regime structure (tight budgets lean on one method, loose on the other,
     with a crossover budget = the joint optimum the project hunts for).

The exponent p is left as a knob because the toy found the *finite-rate* value
p≈1.8 rather than the asymptotic high-rate p=2 — using the measured p makes the
predicted optimum honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class ByteModel:
    """Maps compression settings to bytes.

    Weights: `weight_bytes_fn(eta)` returns total weight bytes at energy η
    (built from the measured SCT energy→params curve). `dense_weight_bytes` is
    the fp16 dense reference.

    KV cache: per cached token we store keys at `b` bits/coord (TurboQuantProd:
    b−1 MSE + 1 QJL) and values at `b` bits/coord (TurboQuantMSE), across
    `d_head` coords and `n_layers_kv` full-attention layers, over context
    length `L`. fp16 reference is 16 bits/coord for K and V.
    """
    weight_bytes_fn: Callable[[float], float]
    dense_weight_bytes: float
    d_head: int = 64
    L: int = 4096          # context length (tokens cached)
    n_layers_kv: int = 1   # full-attention layers whose KV is compressed

    def kv_bytes(self, b: float) -> float:
        # (K bits + V bits) = 2*b per coord ; /8 -> bytes
        return self.L * self.n_layers_kv * self.d_head * (2.0 * b) / 8.0

    def dense_kv_bytes(self) -> float:
        return self.L * self.n_layers_kv * self.d_head * (2.0 * 16.0) / 8.0

    def weight_bytes(self, eta: float) -> float:
        return float(self.weight_bytes_fn(eta))

    def total(self, eta: float, b: float) -> float:
        return self.weight_bytes(eta) + self.kv_bytes(b)

    def dense_total(self) -> float:
        return self.dense_weight_bytes + self.dense_kv_bytes()


@dataclass
class RateDistortion:
    beta_p: float                     # TQ variance coeff: ΔL_tq ≈ β_p·2^(−p·b)
    p: float                          # measured effective exponent (≈1.8; theory 2)
    bytes: ByteModel
    sct_dL_fn: Optional[Callable[[float], float]] = None  # MEASURED ΔL_sct(η) — preferred
    alpha: Optional[float] = None     # parametric fallback ΔL_sct≈α(1−η) (local slope only)

    # ---- distortion ---------------------------------------------------------
    def dL_sct(self, eta: float) -> float:
        """SCT distortion. Uses the MEASURED distortion–rate curve when available
        (the honest choice: the toy showed ΔL_sct is the curvature-weighted
        quadratic, NOT α(1−η) — the latter fits at only R²≈0.37). Falls back to
        the α(1−η) local linearisation if no curve is supplied."""
        if self.sct_dL_fn is not None:
            return float(self.sct_dL_fn(eta))
        return self.alpha * (1.0 - eta)

    def dL_tq(self, b: float) -> float:
        return self.beta_p * 2.0 ** (-self.p * b)

    def dL(self, eta: float, b: float) -> float:
        return self.dL_sct(eta) + self.dL_tq(b)

    # ---- KKT marginal loss reduction per byte -------------------------------
    def marginal_weight(self, eta: float, deta: float = 1e-3) -> float:
        """|dΔL/dη| / |d(weight bytes)/dη|  — loss reduced per weight byte spent.
        Uses the measured distortion curve's local derivative (not a constant α)."""
        e1 = min(eta + deta, 0.999999)
        dloss = -(self.dL_sct(e1) - self.dL_sct(eta)) / (e1 - eta)  # -dΔL/dη ≥ 0
        dbytes = (self.bytes.weight_bytes(e1) - self.bytes.weight_bytes(eta)) / (e1 - eta)
        return abs(dloss) / (abs(dbytes) + 1e-30)

    def marginal_kv(self, b: float) -> float:
        """|dΔL/db| / |d(KV bytes)/db|  — loss reduced per KV byte spent."""
        dloss = self.beta_p * self.p * math.log(2) * 2.0 ** (-self.p * b)
        dbytes = self.bytes.L * self.bytes.n_layers_kv * self.bytes.d_head * 2.0 / 8.0
        return dloss / (dbytes + 1e-30)

    # ---- budget-constrained optimum -----------------------------------------
    def optimum(self, M_budget: float, eta_grid=None, b_grid=None):
        """min ΔL s.t. M(η,b) ≤ M_budget. Returns dict with η*, b*, ΔL*, bytes."""
        eta_grid = eta_grid if eta_grid is not None else np.linspace(0.30, 0.9999, 400)
        b_grid = b_grid if b_grid is not None else np.linspace(1.0, 8.0, 400)
        best = None
        for eta in eta_grid:
            wb = self.bytes.weight_bytes(eta)
            if wb > M_budget:
                continue
            # remaining budget for KV -> max feasible b
            kv_avail = M_budget - wb
            b_max = kv_avail / (self.bytes.L * self.bytes.n_layers_kv * self.bytes.d_head * 2.0 / 8.0)
            feas_b = b_grid[b_grid <= b_max]
            if feas_b.size == 0:
                continue
            for b in feas_b:
                loss = self.dL(eta, b)
                if best is None or loss < best["dL"]:
                    best = {"eta": float(eta), "b": float(b), "dL": float(loss),
                            "weight_bytes": float(wb), "kv_bytes": float(self.bytes.kv_bytes(b)),
                            "total_bytes": float(wb + self.bytes.kv_bytes(b))}
        return best

    def optimum_trace(self, budget_fractions=None):
        """Optimal (η*, b*) across a sweep of budgets (as fraction of dense).
        Reveals the regime structure / crossover budget."""
        dense = self.bytes.dense_total()
        fracs = budget_fractions if budget_fractions is not None else np.linspace(0.05, 0.95, 25)
        trace = []
        for f in fracs:
            opt = self.optimum(f * dense)
            if opt is not None:
                opt["budget_frac"] = float(f)
                opt["marg_weight"] = self.marginal_weight(opt["eta"])
                opt["marg_kv"] = self.marginal_kv(opt["b"])
                trace.append(opt)
        return trace


def build_weight_bytes_fn(energies, total_bytes):
    """Monotone-interpolate the measured SCT (energy → total weight bytes) curve
    into a callable weight_bytes(eta). Clamps outside the measured range."""
    e = np.asarray(energies, float)
    tb = np.asarray(total_bytes, float)
    order = np.argsort(e)
    e, tb = e[order], tb[order]

    def fn(eta):
        return float(np.interp(np.clip(eta, e[0], e[-1]), e, tb))
    return fn


def build_sct_dL_fn(energies, measured_dL):
    """Interpolate the MEASURED SCT distortion–rate curve (energy → ΔL) into a
    callable ΔL_sct(η). This replaces the α(1−η) model, which the toy rejected
    (the true relationship is the curvature-weighted quadratic, concave in 1−η).
    Clamped and monotone-sorted; ΔL is non-increasing in η."""
    e = np.asarray(energies, float)
    y = np.asarray(measured_dL, float)
    order = np.argsort(e)
    e, y = e[order], y[order]

    def fn(eta):
        return float(np.interp(np.clip(eta, e[0], e[-1]), e, y))
    return fn
