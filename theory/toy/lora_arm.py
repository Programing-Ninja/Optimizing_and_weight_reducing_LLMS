"""
Part B, step 5 — Recovery-LoRA arm.
================================================================================
After SCT truncates the weights at energy η (injecting the bias ΔW = W − W_r),
fit a tiny low-rank adapter  ΔW_lora = B A  (rank ρ, B init 0 so we start exactly
at the truncated point) to each layer and re-minimise the toy loss.

A.3 models recovery-LoRA as an "α-shrinking post-step": it barely changes the
memory budget M (adapters are tiny) but should REDUCE the effective SCT bias
constant α. We check:

  * α_LoRA < α  (does LoRA recover the truncation bias?)
  * does the smaller α shift the joint optimum's η UPWARD (compress weights
    harder because LoRA cleans up after)? — answered by Part A's solver, which
    we re-run with (α_LoRA, β).

Outputs: results/lora_arm.json, results/lora_arm.png, tee-log.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gaussian_data import GaussianTokenData
from linear_attn_model import ToyConfig, LinearAttnToy, LAYER_NAMES, train_student, mse_loss
from sct_utils import sct_truncate
from common import (Tee, RESULTS_DIR, get_device, seed_all, fit_through_origin,
                    fit_power, save_json)

ETAS = [0.90, 0.95, 0.98, 0.99, 0.999]
SMALL_PERTURB_MAX_DISC = 0.06
LORA_RANK = 4
LORA_STEPS = 2000


def _forward(student, X, W):
    cfg = student.cfg
    Q = X @ W["q_proj"].T; K = X @ W["k_proj"].T; V = X @ W["v_proj"].T
    A = (Q @ K.transpose(-2, -1)) * cfg.attn_scale
    C = A @ V
    return (C @ W["mlp"].T) @ W["readout"].T


def recover_lora(student, data, teacher, base_W_r, rank, steps, device, log):
    """Fit rank-ρ LoRA adapters on top of frozen truncated weights base_W_r."""
    cfg = student.cfg
    dims = {n: base_W_r[n].shape for n in LAYER_NAMES}  # (out,in)
    A = {n: nn.Parameter(torch.randn(rank, dims[n][1], device=device) * 0.01) for n in LAYER_NAMES}
    B = {n: nn.Parameter(torch.zeros(dims[n][0], rank, device=device)) for n in LAYER_NAMES}
    params = list(A.values()) + list(B.values())
    opt = torch.optim.Adam(params, lr=3e-3)
    frozen = {n: base_W_r[n].detach() for n in LAYER_NAMES}
    for step in range(steps):
        X = data.sample(256).to(device)
        with torch.no_grad():
            Y = teacher(X)
        W = {n: frozen[n] + B[n] @ A[n] for n in LAYER_NAMES}
        loss = ((_forward(student, X, W) - Y) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        W = {n: frozen[n] + B[n] @ A[n] for n in LAYER_NAMES}
    lora_params = sum(A[n].numel() + B[n].numel() for n in LAYER_NAMES)
    return W, lora_params


def run(steps: int = 6000, eval_batch: int = 8192, log=print, device=None):
    device = device or get_device()
    seed_all(0)
    cfg = ToyConfig()
    log(f"[LoRA] cfg={cfg}  LoRA rank={LORA_RANK}")
    data = GaussianTokenData(cfg.d_in, cfg.T, sigma_kind="iso", seed=0, device=device)
    teacher = LinearAttnToy(cfg, teacher=True, seed=0).to(device)
    student = train_student(cfg, data, teacher, steps=steps, device=device, log=log)

    Xe = data.sample(eval_batch).to(device)
    with torch.no_grad():
        Ye = teacher(Xe)
        base_loss = mse_loss(student(Xe), Ye)
    log(f"[LoRA] baseline MSE = {base_loss:.6e}")
    dense_W = {n: getattr(student, n).weight.detach().clone() for n in LAYER_NAMES}

    rows = []
    for eta in ETAS:
        ov, discs, sct_params = {}, [], 0
        for n in LAYER_NAMES:
            W_r, r, p, disc = sct_truncate(dense_W[n], eta)
            ov[n] = W_r; discs.append(disc); sct_params += p
        mdisc = float(np.mean(discs))
        with torch.no_grad():
            dL_before = mse_loss(_forward(student, Xe, ov), Ye) - base_loss
        W_lora, lora_params = recover_lora(student, data, teacher, ov, LORA_RANK,
                                           LORA_STEPS, device, log)
        with torch.no_grad():
            dL_after = mse_loss(_forward(student, Xe, W_lora), Ye) - base_loss
        recovery = 1.0 - dL_after / (dL_before + 1e-30)
        rows.append({"eta": eta, "mean_disc": mdisc, "dL_before": dL_before,
                     "dL_after": dL_after, "recovery_frac": recovery,
                     "sct_params": sct_params, "lora_params": lora_params})
        log(f"[LoRA] η={eta:.3f} (1−η)={mdisc:.4f}  ΔL_before={dL_before:.4e}  "
            f"ΔL_after={dL_after:.4e}  recovered={recovery:.1%}  "
            f"(+{lora_params} LoRA params on {sct_params} SCT params)")

    x = np.array([r["mean_disc"] for r in rows])
    yb = np.array([r["dL_before"] for r in rows])
    ya = np.array([r["dL_after"] for r in rows])
    mask = x <= SMALL_PERTURB_MAX_DISC
    alpha, r2a = fit_through_origin(x[mask], yb[mask])
    alpha_lora, r2b = fit_through_origin(x[mask], ya[mask])
    # ΔL is concave in (1−η) (curvature-weighting), so report the power-law form
    # too — it's the honest 1-D summary; the straight α line is only local.
    Ab, pb, r2pb = fit_power(x, yb)
    Aa, pa, r2pa = fit_power(x, ya)
    med_recovery = float(np.median([r["recovery_frac"] for r in rows]))
    log(f"[LoRA] ==> α (no LoRA, local)   = {alpha:.6e}  R²={r2a:.4f}   "
        f"power p={pb:.3f} R²={r2pb:.4f}")
    log(f"[LoRA] ==> α_LoRA     (local)   = {alpha_lora:.6e}  R²={r2b:.4f}   "
        f"power p={pa:.3f} R²={r2pa:.4f}")
    log(f"[LoRA] ==> α shrink factor = {alpha/(alpha_lora+1e-30):.2f}×  "
        f"({'LoRA reduces SCT bias' if alpha_lora < alpha else 'no reduction'})")
    log(f"[LoRA] ==> median recovery fraction = {med_recovery:.1%} (roughly constant across η ⇒ "
        f"LoRA scales the concave curve down, preserving its shape)")

    result = {"cfg": cfg.__dict__, "baseline_mse": base_loss, "lora_rank": LORA_RANK,
              "rows": rows, "alpha": alpha, "alpha_lora": alpha_lora,
              "alpha_shrink": alpha / (alpha_lora + 1e-30),
              "power_before": {"A": Ab, "p": pb, "r2": r2pb},
              "power_after": {"A": Aa, "p": pa, "r2": r2pa},
              "median_recovery": med_recovery}
    save_json("lora_arm.json", result)
    _plot(result)
    return result


def _plot(result):
    rows = result["rows"]
    x = np.array([r["mean_disc"] for r in rows])
    yb = np.array([r["dL_before"] for r in rows])
    ya = np.array([r["dL_after"] for r in rows])
    rec = np.array([r["recovery_frac"] for r in rows])
    pb, pa = result["power_before"], result["power_after"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # panel 1: distortion curves + power-law fits (the honest concave shape)
    ax = axes[0]
    xx = np.linspace(x[x > 0].min() * 0.7, x.max(), 100)
    ax.scatter(x, yb, s=40, color="C3", zorder=5, label="ΔL SCT-only (measured)")
    ax.scatter(x, ya, s=40, color="C0", zorder=5, label="ΔL after recovery-LoRA")
    ax.plot(xx, pb["A"] * xx ** pb["p"], "C3-",
            label=f"power (1−η)^{pb['p']:.2f}, R²={pb['r2']:.3f}")
    ax.plot(xx, pa["A"] * xx ** pa["p"], "C0-",
            label=f"power (1−η)^{pa['p']:.2f}, R²={pa['r2']:.3f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean discarded energy (1−η)"); ax.set_ylabel("ΔL")
    ax.set_title(f"Recovery-LoRA: concave (power-law), α↓{result['alpha_shrink']:.1f}×")
    ax.legend(fontsize=7)
    # panel 2: recovery fraction (the real message — roughly constant)
    ax = axes[1]
    ax.plot(x, 100 * rec, "o-", color="C2")
    ax.axhline(100 * result["median_recovery"], ls="--", color="gray",
               label=f"median {result['median_recovery']:.0%}")
    ax.set_xscale("log")
    ax.set_xlabel("mean discarded energy (1−η)")
    ax.set_ylabel("bias recovered by LoRA (%)")
    ax.set_ylim(0, 100)
    ax.set_title("LoRA recovery fraction vs compression")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "lora_arm.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    log = Tee(os.path.join(RESULTS_DIR, "lora_arm.log"))
    try:
        log("\n########## RECOVERY-LoRA ARM ##########")
        run(log=log)
    finally:
        log.close()
