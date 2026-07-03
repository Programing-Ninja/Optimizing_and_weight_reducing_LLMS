"""
Part B, step 4 — Combined arm (the additivity test).
================================================================================
Apply SCT (weight truncation at energy η) AND TurboQuant (KV quantization at b
bits) at once over an (η, b) grid, and test the central A.3 claim:

    ΔL(η, b)  ≈  ΔL_sct(η)  +  ΔL_tq(b)          (bias + variance, ADDITIVE)

IMPORTANT: the additive baseline uses the MEASURED single-method marginals —
ΔL_sct(η) is the toy loss with SCT-only, ΔL_tq(b) the loss with TQ-only — NOT the
α(1−η) / β·2^(−2b) parametric fits. This is deliberate: testing against measured
marginals isolates the *interaction* (are the two methods additive?) from the
question of whether either method's 1-D model is correct. So this arm is
unaffected by the SCT-curve / p≈1.83 corrections — it never used them.

We measure ΔL on the grid and quantify the CROSS TERM
κ = measured − (ΔL_sct + ΔL_tq). If |κ| ≪ ΔL, weight compression and
KV quantization are independent to first order and the joint optimum can be
predicted from the two cheap single-method sweeps. If κ is large, the methods
COUPLE and the optimum can't be found by tuning each alone — that itself is the
headline finding (Part E, question 1).

Outputs: results/combined_arm.json, results/combined_arm.png, tee-log.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gaussian_data import GaussianTokenData
from linear_attn_model import ToyConfig, LinearAttnToy, LAYER_NAMES, train_student, mse_loss
from sct_utils import sct_truncate
from common import Tee, RESULTS_DIR, get_device, seed_all, save_json

from turboquant.quantizer import TurboQuantMSE, TurboQuantProd

ETAS = [0.90, 0.95, 0.98, 0.99, 0.999]
BITS = [2, 3, 4]


@torch.no_grad()
def forward_combined(student, X, overrides, key_quant, val_quant):
    """Toy forward with SCT-truncated weights (`overrides`) AND TurboQuant KV.
    Passing key_quant=None (val_quant=None) disables KV quantization."""
    def W(name):
        return overrides[name] if overrides and name in overrides else getattr(student, name).weight
    cfg = student.cfg
    Q = X @ W("q_proj").T
    K = X @ W("k_proj").T
    V = X @ W("v_proj").T
    if key_quant is not None:
        A = key_quant.attention_score(Q, key_quant.quantize(K)) * cfg.attn_scale
        V = val_quant.dequantize(val_quant.quantize(V))
    else:
        A = (Q @ K.transpose(-2, -1)) * cfg.attn_scale
    C = A @ V
    h = C @ W("mlp").T
    return h @ W("readout").T


def run(steps: int = 6000, eval_batch: int = 8192, log=print, device=None):
    device = device or get_device()
    seed_all(0)
    cfg = ToyConfig()
    log(f"[COMB] cfg={cfg}")
    data = GaussianTokenData(cfg.d_in, cfg.T, sigma_kind="iso", seed=0, device=device)
    teacher = LinearAttnToy(cfg, teacher=True, seed=0).to(device)
    student = train_student(cfg, data, teacher, steps=steps, device=device, log=log)

    Xe = data.sample(eval_batch).to(device)
    with torch.no_grad():
        Ye = teacher(Xe)
        base_loss = mse_loss(student(Xe), Ye)
    log(f"[COMB] baseline MSE = {base_loss:.6e}")
    dense_W = {n: getattr(student, n).weight.detach().clone() for n in LAYER_NAMES}

    def sct_overrides(eta):
        ov, discs = {}, []
        for n in LAYER_NAMES:
            W_r, r, p, disc = sct_truncate(dense_W[n], eta)
            ov[n] = W_r
            discs.append(disc)
        return ov, float(np.mean(discs))

    # SCT-only marginal (per η)
    sct_only = {}
    for eta in ETAS:
        ov, mdisc = sct_overrides(eta)
        dL = mse_loss(forward_combined(student, Xe, ov, None, None), Ye) - base_loss
        sct_only[eta] = {"mean_disc": mdisc, "dL": dL}
        log(f"[COMB] SCT-only  η={eta:.3f}  (1−η)={mdisc:.4f}  ΔL={dL:.4e}")

    # TQ-only marginal (per b)
    tq_only = {}
    for b in BITS:
        kq = TurboQuantProd(dim=cfg.d_head, bits=b, device=device)
        vq = TurboQuantMSE(dim=cfg.d_head, bits=b, device=device)
        dL = mse_loss(forward_combined(student, Xe, None, kq, vq), Ye) - base_loss
        tq_only[b] = {"rate": 2.0 ** (-2 * b), "dL": dL}
        log(f"[COMB] TQ-only   b={b}  2^(-2b)={2**(-2*b):.4e}  ΔL={dL:.4e}")

    # Combined grid + cross term
    log("[COMB] --- combined (η,b) grid: measured vs additive prediction ---")
    grid = []
    cross_rel = []
    for eta in ETAS:
        ov, mdisc = sct_overrides(eta)
        for b in BITS:
            kq = TurboQuantProd(dim=cfg.d_head, bits=b, device=device)
            vq = TurboQuantMSE(dim=cfg.d_head, bits=b, device=device)
            measured = mse_loss(forward_combined(student, Xe, ov, kq, vq), Ye) - base_loss
            additive = sct_only[eta]["dL"] + tq_only[b]["dL"]
            cross = measured - additive
            rel = abs(cross) / (abs(measured) + 1e-30)
            cross_rel.append(rel)
            grid.append({"eta": eta, "bits": b, "mean_disc": mdisc,
                         "rate": 2.0 ** (-2 * b), "measured": measured,
                         "additive": additive, "cross": cross, "cross_rel": rel})
            log(f"[COMB]  η={eta:.3f} b={b}  measured={measured:.4e}  "
                f"additive={additive:.4e}  cross={cross:+.3e}  |cross|/meas={rel:.3f}")

    med_cross = float(np.median(cross_rel))
    max_cross = float(np.max(cross_rel))
    # sign of the coupling at the most-aggressive corner (smallest η, smallest b)
    corner = min(grid, key=lambda g: (g["eta"], g["bits"]))
    coupling_sign = "sub-additive (errors partially cancel)" if corner["cross"] < 0 \
        else "super-additive (errors compound)"
    if med_cross < 0.10 and max_cross < 0.15:
        verdict = "ADDITIVE (bias+variance independent to 1st order across the grid)"
    elif med_cross < 0.10:
        verdict = (f"ADDITIVE in the operating regime (median {med_cross:.1%}), but "
                   f"COUPLED at aggressive joint compression (up to {max_cross:.1%}); "
                   f"coupling is {coupling_sign}")
    else:
        verdict = (f"COUPLED (median {med_cross:.1%}) — joint sweep needed; "
                   f"coupling is {coupling_sign}")
    log(f"[COMB] ==> median |cross|/measured = {med_cross:.4f}, max = {max_cross:.4f}")
    log(f"[COMB] ==> corner (η={corner['eta']}, b={corner['bits']}) cross = "
        f"{corner['cross']:+.3e} ⇒ {coupling_sign}")
    log(f"[COMB] ==> ADDITIVITY VERDICT: {verdict}")

    # empirical minimum-ΔL cell at a couple of illustrative byte budgets
    result = {
        "cfg": cfg.__dict__, "baseline_mse": base_loss,
        "etas": ETAS, "bits": BITS,
        "sct_only": {str(k): v for k, v in sct_only.items()},
        "tq_only": {str(k): v for k, v in tq_only.items()},
        "grid": grid, "median_cross_rel": med_cross, "max_cross_rel": max_cross,
        "additivity_verdict": verdict,
    }
    save_json("combined_arm.json", result)
    _plot(result)
    return result


def _plot(result):
    etas = result["etas"]; bits = result["bits"]
    M = np.full((len(etas), len(bits)), np.nan)
    A = np.full_like(M, np.nan)
    Cr = np.full_like(M, np.nan)
    for g in result["grid"]:
        i = etas.index(g["eta"]); j = bits.index(g["bits"])
        M[i, j] = g["measured"]; A[i, j] = g["additive"]; Cr[i, j] = g["cross_rel"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    # measured vs additive scatter
    ax = axes[0]
    ax.scatter(A.ravel(), M.ravel(), s=30)
    lim = np.nanmax([A.max(), M.max()])
    ax.plot([0, lim], [0, lim], "k--", label="additive = measured")
    ax.set_xlabel("additive baseline: measured (SCT-only + TQ-only)")
    ax.set_ylabel("measured ΔL (combined)"); ax.set_title("additivity of bias+variance")
    ax.legend(fontsize=8)
    # cross-term heatmap
    ax = axes[1]
    im = ax.imshow(Cr, aspect="auto", origin="lower", cmap="magma")
    ax.set_xticks(range(len(bits))); ax.set_xticklabels(bits)
    ax.set_yticks(range(len(etas))); ax.set_yticklabels(etas)
    ax.set_xlabel("KV bits b"); ax.set_ylabel("SCT energy η")
    ax.set_title(f"|cross|/measured (median {result['median_cross_rel']:.3f})")
    fig.colorbar(im, ax=ax)
    # measured ΔL heatmap
    ax = axes[2]
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(bits))); ax.set_xticklabels(bits)
    ax.set_yticks(range(len(etas))); ax.set_yticklabels(etas)
    ax.set_xlabel("KV bits b"); ax.set_ylabel("SCT energy η")
    ax.set_title("measured ΔL(η,b)")
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"Combined arm — {result['additivity_verdict']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "combined_arm.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    log = Tee(os.path.join(RESULTS_DIR, "combined_arm.log"))
    try:
        log("\n########## COMBINED ARM ##########")
        run(log=log)
    finally:
        log.close()
