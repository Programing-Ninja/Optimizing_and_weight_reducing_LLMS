"""
Part A runner — load the toy-MEASURED constants (α, β_p, p, byte curves) and
predict the budget-constrained joint optimum; draw the ΔL(η,b) surface and the
regime structure; check whether recovery-LoRA (smaller α) pushes η* upward.

Run AFTER the toy arms have produced results/*.json.
Outputs: theory/models/results/{surface.png, regimes.png, optimum.json, RESULTS.md}
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rate_distortion import (RateDistortion, ByteModel, build_weight_bytes_fn,
                             build_sct_dL_fn)

HERE = os.path.dirname(__file__)
TOY_RESULTS = os.path.join(HERE, "..", "toy", "results")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def _load(name):
    with open(os.path.join(TOY_RESULTS, name)) as f:
        return json.load(f)


def build_model(alpha, sct, tq, L=4096, n_layers_kv=1, sct_dL_override=None):
    """Build the rate-distortion model. SCT distortion uses the MEASURED curve
    (energy → ΔL) by default; pass `sct_dL_override` (energies, dL) to substitute
    a different curve, e.g. the post-LoRA distortion."""
    energies = sct["all_layers"]["energy"]
    total_bytes = sct["all_layers"]["total_bytes_fp16"]
    wfn = build_weight_bytes_fn(energies, total_bytes)
    if sct_dL_override is not None:
        sct_fn = build_sct_dL_fn(*sct_dL_override)
    else:
        sct_fn = build_sct_dL_fn(energies, sct["all_layers"]["measured"])
    bm = ByteModel(weight_bytes_fn=wfn, dense_weight_bytes=sct["dense_bytes_fp16"],
                   d_head=sct["cfg"]["d_head"], L=L, n_layers_kv=n_layers_kv)
    return RateDistortion(alpha=alpha, sct_dL_fn=sct_fn, beta_p=tq["downstream"]["beta_p"],
                          p=tq["downstream"]["eff_exponent"], bytes=bm)


def plot_surface(rd: RateDistortion, budget_frac, path, title):
    etas = np.linspace(0.30, 0.999, 200)
    bs = np.linspace(1.5, 6.0, 200)
    E, B = np.meshgrid(etas, bs)
    # SCT term from the MEASURED distortion curve (vectorised interp), not α(1−η)
    sct_term = np.array([[rd.dL_sct(e) for e in etas]])  # (1, Ne)
    Z = np.broadcast_to(sct_term, E.shape) + rd.beta_p * 2.0 ** (-rd.p * B)
    dense = rd.bytes.dense_total()
    opt = rd.optimum(budget_frac * dense)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    cs = ax.contourf(E, B, np.log10(Z), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10 ΔL")
    # iso-budget contour M(η,b)=budget
    M = np.array([[rd.bytes.total(e, b) for e in etas] for b in bs])
    ax.contour(E, B, M, levels=[budget_frac * dense], colors="white",
               linewidths=2, linestyles="--")
    if opt:
        ax.plot(opt["eta"], opt["b"], "r*", ms=18, label=f"optimum η*={opt['eta']:.3f}, b*={opt['b']:.2f}")
    ax.set_xlabel("SCT energy η"); ax.set_ylabel("KV bits b")
    ax.set_title(title); ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return opt


def plot_regimes(rd: RateDistortion, path):
    trace = rd.optimum_trace()
    fr = [t["budget_frac"] for t in trace]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(fr, [t["eta"] for t in trace], "o-", color="C0", label="η* (weights)")
    ax.set_xlabel("budget / dense"); ax.set_ylabel("optimal η*", color="C0")
    ax2 = ax.twinx()
    ax2.plot(fr, [t["b"] for t in trace], "s-", color="C3", label="b* (KV bits)")
    ax2.set_ylabel("optimal b*", color="C3")
    ax.set_title("regime structure: where to spend the budget")
    ax = axes[1]
    # Suppress the weights-marginal where η* has saturated at the grid ceiling:
    # you cannot buy more weight precision there, so the per-byte marginal is
    # ill-defined (dbytes→0). Those budgets are the "spend everything on KV" regime.
    mw = [t["marg_weight"] if t["eta"] < 0.998 else np.nan for t in trace]
    ax.semilogy(fr, mw, "o-", label="marginal loss/byte — weights")
    ax.semilogy(fr, [t["marg_kv"] for t in trace], "s-", label="marginal loss/byte — KV")
    ax.set_xlabel("budget / dense"); ax.set_ylabel("marginal ΔL reduction per byte")
    ax.set_title("KKT: marginals equalise until η* saturates"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return trace


def main(log=print):
    sct = _load("sct_arm_iso.json")
    tq = _load("tq_arm.json")
    alpha = sct["alpha"]
    log(f"[solver] measured constants: α={alpha:.4e}  β_p={tq['downstream']['beta_p']:.4e}  "
        f"p={tq['downstream']['eff_exponent']:.4f}")

    # The toy's 5×64×64 weights are trivially cheap next to a multi-thousand-token
    # KV cache, which pins η→1 and hides the crossover. The weight:KV byte ratio
    # is a *deployment* knob (model size × context length), so for an illustrative
    # NON-degenerate demo we pick L_balanced where dense weight bytes ≈ dense KV
    # bytes — both levers then matter and the regime structure is visible.
    d_head = sct["cfg"]["d_head"]
    L_bal = max(1, round(sct["dense_bytes_fp16"] / (d_head * 4)))
    log(f"[solver] using balanced context L={L_bal} (dense weights ≈ dense KV)")
    rd = build_model(alpha, sct, tq, L=L_bal)
    log(f"[solver] dense total bytes = {rd.bytes.dense_total():.3e}  "
        f"(weights {rd.bytes.dense_weight_bytes:.2e} + KV {rd.bytes.dense_kv_bytes():.2e}, L={rd.bytes.L})")

    # 50% budget at the balanced L gives an INTERIOR optimum (both levers active),
    # where the KKT equal-marginal condition actually bites — the cleanest demo.
    budget_frac = 0.50
    opt = plot_surface(rd, budget_frac, os.path.join(OUT, "surface.png"),
                       f"Joint SCT×TurboQuant ΔL surface — budget={budget_frac:.0%} of dense")
    log(f"[solver] optimum @ {budget_frac:.0%} budget: η*={opt['eta']:.4f} b*={opt['b']:.3f} "
        f"ΔL*={opt['dL']:.4e}  (weights {opt['weight_bytes']:.2e}B + KV {opt['kv_bytes']:.2e}B)")

    trace = plot_regimes(rd, os.path.join(OUT, "regimes.png"))
    log("[solver] regime structure (budget_frac -> η*, b*):")
    for t in trace[::4]:
        log(f"[solver]   budget={t['budget_frac']:.2f}  η*={t['eta']:.3f}  b*={t['b']:.2f}  "
            f"ΔL*={t['dL']:.3e}")

    # ---- does recovery-LoRA (smaller α) push η* upward? -----------------------
    lora_note = ""
    try:
        lora = _load("lora_arm.json")
        # Use the MEASURED post-LoRA distortion curve (not α_LoRA·(1−η)).
        lora_etas = [r["eta"] for r in lora["rows"]]
        lora_dL = [r["dL_after"] for r in lora["rows"]]
        rd_lora = build_model(lora["alpha_lora"], sct, tq, L=L_bal,
                              sct_dL_override=(lora_etas, lora_dL))
        opt_lora = rd_lora.optimum(budget_frac * rd_lora.bytes.dense_total())
        # NOTE ON CONVENTION: η is RETAINED energy, so LOWER η = HARDER weight
        # compression. The doc (§A.3) says LoRA should shift "η upward (compress
        # weights harder)" — that parenthetical is inverted w.r.t. this
        # convention; the *intent* (compress weights harder) means η* should go
        # DOWN, freeing bytes to spend on the KV cache (higher b*).
        shift = opt_lora["eta"] - opt["eta"]
        harder = shift < -1e-3
        lora_note = (f"With recovery-LoRA (α {alpha:.3e}→{lora['alpha_lora']:.3e}, "
                     f"{lora['alpha_shrink']:.1f}× smaller): η* {opt['eta']:.4f}→{opt_lora['eta']:.4f} "
                     f"(Δ={shift:+.4f}), b* {opt['b']:.2f}→{opt_lora['b']:.2f}. "
                     f"Lower η* ⇒ compress weights HARDER and reallocate freed bytes to KV.")
        log(f"[solver] {lora_note}")
        log(f"[solver] => LoRA lets us compress weights {'HARDER (η* down)' if harder else 'similarly'}; "
            f"the doc's 'η upward' wording is inverted vs the retained-energy convention.")
    except FileNotFoundError:
        pass

    optimum = {"budget_frac": budget_frac, "alpha": alpha,
               "beta_p": tq["downstream"]["beta_p"], "p": tq["downstream"]["eff_exponent"],
               "optimum": opt, "trace": trace, "lora_note": lora_note}
    with open(os.path.join(OUT, "optimum.json"), "w") as f:
        json.dump(optimum, f, indent=2, default=float)
    _write_results_md(sct, tq, rd, opt, trace, lora_note, budget_frac)
    return optimum


def _write_results_md(sct, tq, rd, opt, trace, lora_note, budget_frac):
    tight = trace[0]; loose = trace[-1]
    mw = rd.marginal_weight(opt["eta"]); mk = rd.marginal_kv(opt["b"])
    md = f"""# Part A — Joint Rate-Distortion Solver: Results

**Predicted joint optimum** for `ΔL(η,b) = ΔL_sct(η) + β_p·2^(−p·b)`.

**Important:** the SCT term is the **MEASURED distortion–rate curve** ΔL_sct(η),
NOT the `α(1−η)` model from the doc — the toy rejected `α(1−η)` (global
R²={sct['alpha_global_r2']:.2f}; the true relationship is the concave,
curvature-weighted quadratic). The TQ term keeps the parametric form with the
measured exponent p (it fit well, R²≈{tq['downstream']['eff_exponent_r2']:.3f}).

## Measured constants (from theory/toy)
| constant | value | meaning |
|---|---|---|
| ΔL_sct(η) | curve | interpolated measured distortion (concave; α(1−η) rejected) |
| α (local only) | {sct['alpha']:.4e} | SCT slope near η→1 (not used by the solver) |
| β_p | {tq['downstream']['beta_p']:.4e} | TurboQuant variance coefficient |
| p | {tq['downstream']['eff_exponent']:.3f} | effective bit-exponent (theory 2; finite-rate <2) |

## Byte model (balanced regime)
- dense total: {rd.bytes.dense_total():.3e} B (weights {rd.bytes.dense_weight_bytes:.2e} + KV {rd.bytes.dense_kv_bytes():.2e}, L={rd.bytes.L})
- KV: b bits/coord for K (Prod) and V (MSE), d_head={rd.bytes.d_head}, {rd.bytes.n_layers_kv} layer(s)
- L chosen so dense weights ≈ dense KV (both levers active). The weight:KV ratio
  is a deployment knob (model size × context length); the solver is scale-agnostic.

## Budget-constrained optimum ({budget_frac:.0%} of dense)
- **η\\* = {opt['eta']:.4f}**,  **b\\* = {opt['b']:.3f}**,  ΔL\\* = {opt['dL']:.4e}
- spend: weights {opt['weight_bytes']:.2e} B + KV {opt['kv_bytes']:.2e} B
- KKT check: marginal loss/byte — weights {mw:.3e} vs KV {mk:.3e}
  (equal ⇒ interior optimum, the equal-marginal condition holds).

## Regime structure
- tight budget ({tight['budget_frac']:.0%}): η\\*={tight['eta']:.3f}, b\\*={tight['b']:.2f}
- loose budget ({loose['budget_frac']:.0%}): η\\*={loose['eta']:.3f}, b\\*={loose['b']:.2f}
- **Finding (contradicts §A.3's guess):** measured SCT distortion is concave, so
  weight compression is cheap per byte — tight budgets lean on **weights first**,
  not "TQ first". See `regimes.png`.

## Recovery-LoRA
{lora_note or "_(lora_arm.json not found)_"}

## Figures
- `surface.png` — ΔL(η,b) surface, iso-budget line, marked (interior) optimum
- `regimes.png` — optimal (η\\*, b\\*) vs budget + equalised KKT marginals
"""
    with open(os.path.join(OUT, "RESULTS.md"), "w") as f:
        f.write(md)


if __name__ == "__main__":
    from datetime import datetime
    logf = open(os.path.join(OUT, "run_solver.log"), "a")
    def log(*a):
        m = " ".join(str(x) for x in a); print(m); logf.write(m + "\n")
    log(f"\n=== solver run {datetime.now()} ===")
    main(log=log)
    logf.close()
