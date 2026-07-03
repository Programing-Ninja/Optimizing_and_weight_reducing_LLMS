"""
Part B, step 2 — SCT arm.
================================================================================
Factor the trained toy weights with SCT at a sweep of energy targets η and check:

  (1) LOCAL-QUADRATIC EQUALITY (the A.1 assumption, made checkable):
      for each layer truncated *alone*, measured ΔL ≈ pure-quadratic E||ΔO||^2,
      i.e. the linear/cross term is negligible ⇒ loss really is quadratic in ΔW.

  (2) ECKART–YOUNG LOSS LAW:
      per-layer quadratic ΔL is a curvature-weighted image of the discarded
      spectral energy Σ_{i>r}σ_i^2. Under isotropic Σ_x on the readout layer,
      curvature ∝ I ⇒ ΔL == discarded energy exactly (the cleanest check).

  (3) ESTIMATE α:
      truncate ALL layers together at η; slope of measured ΔL vs discarded
      energy fraction (1−η_actual) is the constant α used by Part A.

Outputs: results/sct_arm.json, results/sct_arm.png, tee-log results/sct_arm.log
"""

from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gaussian_data import GaussianTokenData
from linear_attn_model import (ToyConfig, LinearAttnToy, LAYER_NAMES,
                               train_student, population_loss,
                               quadratic_vs_measured_deltaL, mse_loss)
from sct_utils import sct_truncate, discarded_energy
from common import (Tee, RESULTS_DIR, get_device, seed_all, fit_through_origin,
                    fit_power, save_json)


# Sweep spans aggressive (η=0.30) to near-lossless (η=0.999) so the measured
# distortion–rate curve the solver consumes isn't truncated at the data edge.
ENERGIES = [0.30, 0.40, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
SMALL_PERTURB_MAX_DISC = 0.06  # α is the LOCAL slope in the valid-linearisation regime


def eckart_young_control(d=64, n=200000, decay=0.9, device=None):
    """Cleanest possible Eckart–Young check, isolated from the attention pipeline.

    A single linear layer y = W x with ISOTROPIC Gaussian x. Then
        ΔL = E‖(W−W_r)x‖² = ‖W−W_r‖_F² = Σ_{i>r} σ_i²   (EXACTLY, Eckart–Young).
    Returns list of (rank, measured ΔL, discarded energy) — they must coincide.
    """
    import torch as _t
    from sct_utils import sct_truncate, discarded_energy
    from linear_attn_model import spectrum_weight
    device = device or get_device()
    W = spectrum_weight(d, d, decay, seed=321, device=device, dtype=_t.float32)
    X = _t.randn(n, d, device=device)  # isotropic
    with _t.no_grad():
        Y = X @ W.T
        rows = []
        for eta in (0.8, 0.9, 0.95, 0.99, 0.999):
            W_r, r, _, _ = sct_truncate(W, eta)
            dl = ((X @ W_r.T - Y) ** 2).sum(-1).mean().item()
            rows.append((r, dl, discarded_energy(W, r)))
    return rows


def run(sigma_kind: str = "iso", steps: int = 3000, eval_batch: int = 4096,
        log=print, device=None):
    device = device or get_device()
    seed_all(0)
    cfg = ToyConfig()
    log(f"[SCT] cfg={cfg}")
    log(f"[SCT] Σ_x kind = {sigma_kind}")

    data = GaussianTokenData(cfg.d_in, cfg.T, sigma_kind=sigma_kind, decay=0.85,
                             seed=0, device=device)
    teacher = LinearAttnToy(cfg, teacher=True, seed=0).to(device)
    student = train_student(cfg, data, teacher, steps=steps, device=device, log=log)

    # Large fixed eval set = Monte-Carlo stand-in for the population.
    Xe = data.sample(eval_batch).to(device)
    with torch.no_grad():
        Ye = teacher(Xe)
        base_O = student(Xe)
    base_loss = mse_loss(base_O, Ye)
    log(f"[SCT] baseline population MSE = {base_loss:.6e}")

    dense_W = {n: getattr(student, n).weight.detach().clone() for n in LAYER_NAMES}
    layer_dims = {n: tuple(dense_W[n].shape) for n in LAYER_NAMES}  # (out,in)

    # ---- (1)+(2) per-layer sweep: quadratic equality + Eckart–Young ----------
    per_layer = {n: {"energy": [], "rank": [], "discarded_frac": [],
                     "measured": [], "quadratic": [], "cross": [],
                     "discarded_energy_abs": []} for n in LAYER_NAMES}
    for n in LAYER_NAMES:
        W = dense_W[n]
        tot_energy = (torch.linalg.svdvals(W.float().cpu()) ** 2).sum().item()
        for eta in ENERGIES:
            W_r, r, params, disc_frac = sct_truncate(W, eta)
            measured, quad, cross = quadratic_vs_measured_deltaL(
                student, Xe, Ye, n, W_r, base_O, base_loss)
            per_layer[n]["energy"].append(eta)
            per_layer[n]["rank"].append(r)
            per_layer[n]["discarded_frac"].append(disc_frac)
            per_layer[n]["measured"].append(measured)
            per_layer[n]["quadratic"].append(quad)
            per_layer[n]["cross"].append(cross)
            per_layer[n]["discarded_energy_abs"].append(discarded_energy(W, r))
        # cross-term magnitude relative to quadratic (equality diagnostic)
        q = np.array(per_layer[n]["quadratic"])
        c = np.abs(np.array(per_layer[n]["cross"]))
        rel = float(np.median(c[q > 0] / q[q > 0])) if np.any(q > 0) else float("nan")
        log(f"[SCT] layer {n:8s} dims={layer_dims[n]}  "
            f"median |cross|/quadratic = {rel:.3e}  (small ⇒ loss locally quadratic in ΔW)")

    # ---- (3) all-layers-together sweep -> α -----------------------------------
    allL = {"energy": [], "mean_discarded_frac": [], "measured": [],
            "total_params": [], "total_bytes_fp16": []}
    dense_bytes = sum(o * i for (o, i) in layer_dims.values()) * 2
    for eta in ENERGIES:
        overrides, discs, params = {}, [], 0
        for n in LAYER_NAMES:
            W_r, r, p, disc = sct_truncate(dense_W[n], eta)
            overrides[n] = W_r
            discs.append(disc)
            params += p
        loss_eta = population_loss(student, Xe, Ye, overrides=overrides)
        allL["energy"].append(eta)
        allL["mean_discarded_frac"].append(float(np.mean(discs)))
        allL["measured"].append(loss_eta - base_loss)
        allL["total_params"].append(params)
        allL["total_bytes_fp16"].append(params * 2)

    x = np.array(allL["mean_discarded_frac"])
    y = np.array(allL["measured"])
    # α = LOCAL marginal slope in the small-perturbation (valid-linearisation)
    # regime. Over the full range ΔL is CONCAVE in (1−η): the last-discarded,
    # small-σ modes carry disproportionately high loss-curvature, so a single
    # global line is a poor model (this is the curvature-weighting A.1 warns of).
    mask = x <= SMALL_PERTURB_MAX_DISC
    alpha, r2 = fit_through_origin(x[mask], y[mask])
    alpha_global, r2_global = fit_through_origin(x, y)
    # The scalar α(1−η) model is REJECTED globally; a power law A(1−η)^p is the
    # better 1-D summary, but the EXACT model is the curvature-weighted quadratic
    # (panel 1). Report all three so the model-selection is explicit.
    A_pow, p_pow, r2_pow = fit_power(x, y)
    log(f"[SCT] model selection for ΔL vs (1−η), all-layers:")
    log(f"[SCT]   linear α(1−η) GLOBAL : α={alpha_global:.4e}  R²={r2_global:.4f}  (REJECTED)")
    log(f"[SCT]   linear α(1−η) LOCAL  : α={alpha:.4e}  R²={r2:.4f}  (valid only for discarded≤{SMALL_PERTURB_MAX_DISC})")
    log(f"[SCT]   power  A(1−η)^p      : A={A_pow:.4e} p={p_pow:.3f}  R²={r2_pow:.4f}  (better 1-D summary; concave)")
    log(f"[SCT]   EXACT model = curvature-weighted quadratic tr(ΔW H ΔW) — see panel 1 / known-H check")

    # ---- across-layer coupling: is Σ per-layer quad = all-layers ΔL? ----------
    sumq = np.zeros(len(ENERGIES))
    for n in LAYER_NAMES:
        sumq += np.array(per_layer[n]["quadratic"])
    allm = np.array(allL["measured"])
    ratio = allm / (sumq + 1e-30)
    log(f"[SCT] across-layer coupling: measured / Σ(per-layer quad) ranges "
        f"{ratio.min():.2f}–{ratio.max():.2f} (｟1 ⇒ SUB-additive across layers at "
        f"aggressive η; →1 as η→1)")

    # ---- known-H check on the READOUT layer -----------------------------------
    # Readout is the last layer: ΔO = h ΔW^T exactly, so quadratic ΔL must equal
    # tr(ΔW G_h ΔW^T) with G_h = E[hᵀh] the *known* input-Gram curvature.
    with torch.no_grad():
        _, acts = student(Xe, return_activations=True)
        H_flat = acts["h"].reshape(-1, cfg.d_hidden)
        G_h = (H_flat.T @ H_flat) / H_flat.shape[0]
    known_h_rows = []
    W_ro = dense_W["readout"]
    for eta in ENERGIES:
        W_r, r, _, _ = sct_truncate(W_ro, eta)
        dW = (W_ro - W_r)
        pred = torch.trace(dW @ G_h @ dW.T).item()
        q = per_layer["readout"]["quadratic"][ENERGIES.index(eta)]
        known_h_rows.append((eta, q, pred))
    kh = np.array([(q, p) for (_, q, p) in known_h_rows])
    kh_relerr = float(np.median(np.abs(kh[:, 0] - kh[:, 1]) / (kh[:, 0] + 1e-30)))
    log(f"[SCT] known-H readout check: median |quad − tr(ΔW G_h ΔWᵀ)|/quad = {kh_relerr:.3e}")
    log(f"[SCT]     (≈0 ⇒ loss is exactly the quadratic form with H = input Gram)")

    # ---- standalone Eckart–Young control (ΔL == discarded energy) -------------
    ey_rows = eckart_young_control(device=device)
    ey_relerr = float(np.median([abs(dl - de) / de for (_, dl, de) in ey_rows]))
    log(f"[SCT] Eckart–Young control (single linear layer, iso input):")
    for (r, dl, de) in ey_rows:
        log(f"[SCT]     rank={r:3d}  ΔL={dl:.6e}  discarded-energy={de:.6e}  "
            f"rel.err={abs(dl-de)/de:.2e}")
    log(f"[SCT]     median rel.err = {ey_relerr:.3e}  (≈0 ⇒ ΔL == Σ_{{i>r}}σ_i² exactly)")

    result = {
        "sigma_kind": sigma_kind, "cfg": cfg.__dict__,
        "baseline_mse": base_loss,
        "alpha": alpha, "alpha_r2": r2,
        "alpha_global": alpha_global, "alpha_global_r2": r2_global,
        "power_A": A_pow, "power_p": p_pow, "power_r2": r2_pow,
        "across_layer_ratio_min": float(ratio.min()),
        "across_layer_ratio_max": float(ratio.max()),
        "energies": ENERGIES, "per_layer": per_layer, "all_layers": allL,
        "layer_dims": layer_dims, "dense_bytes_fp16": dense_bytes,
        "known_h_readout_relerr": kh_relerr,
        "eckart_young_control": [{"rank": r, "deltaL": dl, "discarded_energy": de}
                                 for (r, dl, de) in ey_rows],
        "eckart_young_relerr": ey_relerr,
    }
    save_json(f"sct_arm_{sigma_kind}.json", result)
    _plot(result, sigma_kind)
    return result


def _plot(result, sigma_kind):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    # panel 1: measured vs quadratic per layer (equality)
    ax = axes[0]
    for n in LAYER_NAMES:
        m = np.array(result["per_layer"][n]["measured"])
        q = np.array(result["per_layer"][n]["quadratic"])
        ax.scatter(q, m, s=18, label=n)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    ax.set_xlabel("pure-quadratic ΔL  E‖ΔO‖²")
    ax.set_ylabel("measured ΔL")
    ax.set_title("(1) local-quadratic equality")
    ax.legend(fontsize=7)
    # panel 2: per-layer ΔL vs discarded energy
    ax = axes[1]
    for n in LAYER_NAMES:
        de = np.array(result["per_layer"][n]["discarded_energy_abs"])
        q = np.array(result["per_layer"][n]["quadratic"])
        ax.plot(de, q, "o-", ms=4, label=n)
    ax.set_xlabel("discarded energy  Σ_{i>r} σ_i²")
    ax.set_ylabel("quadratic ΔL")
    ax.set_title("(2) Eckart–Young loss law")
    ax.legend(fontsize=7)
    # panel 3: all-layers distortion–rate curve + model comparison
    ax = axes[2]
    x = np.array(result["all_layers"]["mean_discarded_frac"])
    y = np.array(result["all_layers"]["measured"])
    ax.scatter(x, y, s=30, color="C3", zorder=5, label="measured (all layers)")
    xx = np.linspace(x[x > 0].min() * 0.5, x.max(), 100)
    # power-law fit (good) vs linear α(1−η) (poor, global) — on log axes
    ax.plot(xx, result["power_A"] * xx ** result["power_p"], "C0-",
            label=f"power A(1−η)^{result['power_p']:.2f}, R²={result['power_r2']:.3f}")
    ax.plot(xx, result["alpha_global"] * xx, "k--",
            label=f"linear α(1−η), R²={result['alpha_global_r2']:.2f} (rejected)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean discarded energy fraction (1−η)")
    ax.set_ylabel("ΔL")
    ax.set_title("(3) SCT distortion–rate: concave, not linear")
    ax.legend(fontsize=7)
    fig.suptitle(f"SCT arm — Σ_x = {sigma_kind}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"sct_arm_{sigma_kind}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    log = Tee(os.path.join(RESULTS_DIR, "sct_arm.log"))
    try:
        for kind in ("iso", "aniso"):
            log(f"\n########## SCT ARM  (Σ_x={kind}) ##########")
            run(sigma_kind=kind, log=log)
    finally:
        log.close()
