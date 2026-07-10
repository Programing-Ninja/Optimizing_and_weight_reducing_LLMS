"""
theory_validate.py — validate the theory-track models against a MEASURED LLM sweep.

The toy (theory/toy/) established four claims and a solver. This module re-tests
each one on real sweep output (results_*/pareto_results.json from run_pareto.py),
using ΔL := ln(ppl / ppl_dense) — the mean-NLL gap, the LLM analogue of the toy's
ΔMSE — as the distortion measure:

  1. SCT distortion–rate shape: the toy REJECTED α(1−η) (ΔL is concave in 1−η,
     power exponent p_w<1). Fit both a line and a power law A(1−η)^p_w to the
     measured curve and compare R².
  2. TurboQuant law ΔL ≈ β_p·2^(−p·b): fit (β_p, p) from the KV-bits arm at dense
     weights; the toy measured p≈1.83 (asymptotic theory says 2).
  3. Additivity: cross(η,b) = ΔL_joint − ΔL_sct(η) − ΔL_tq(b), relative to
     ΔL_joint. The toy found mostly-additive with SUB-additive (negative)
     coupling at aggressive joint compression.
  4. Recovery-LoRA: recovery fraction 1 − ΔL_lora/ΔL_nolora per η (toy: ~84%,
     roughly constant in η).
  5. Solver: rebuild the Part A rate–distortion model from the MEASURED curves
     (reusing theory/models/rate_distortion.py verbatim) and check its predicted
     (η*, b*) per byte budget against the best measured sweep point under the
     same budget.

b convention: theory's `b` is bits/coordinate for BOTH K and V (ByteModel charges
2b per coord). The sweep quantizes K at key_bits and V at value_bits, so we map
each KV config to b_eff = (key_bits + value_bits) / 2.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "theory", "models"))

from rate_distortion import (RateDistortion, ByteModel,           # noqa: E402
                             build_weight_bytes_fn, build_sct_dL_fn)

TOY_RESULTS = os.path.join(ROOT, "theory", "toy", "results")


# ─────────────────────────────────────────────────────────────────────────────
#  LOADING + DISTORTION
# ─────────────────────────────────────────────────────────────────────────────

def load_sweep(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    base_ppl = data["baseline_raw"]["perplexity"]
    for pt in data["points"]:
        pt["dL"] = math.log(pt["raw"]["perplexity"]) - math.log(base_ppl)
        pt["b_eff"] = (None if pt["key_bits"] is None
                       else 0.5 * (pt["key_bits"] + pt["value_bits"]))
    return data


def _pts(data, *, lora=None, sct=None, kv=None):
    """Filter sweep points. sct/kv: True=compressed, False=dense/fp16, None=any."""
    out = []
    for p in data["points"]:
        if lora is not None and p["lora"] != lora:
            continue
        if sct is not None and (p["energy"] is not None) != sct:
            continue
        if kv is not None and (p["key_bits"] is not None) != kv:
            continue
        out.append(p)
    return out


def load_toy():
    """Toy-measured constants for side-by-side comparison (None if absent)."""
    toy = {}
    for key, fname in [("sct", "sct_arm_iso.json"), ("tq", "tq_arm.json"),
                       ("combined", "combined_arm.json"), ("lora", "lora_arm.json")]:
        p = os.path.join(TOY_RESULTS, fname)
        toy[key] = json.load(open(p)) if os.path.exists(p) else None
    return toy


# ─────────────────────────────────────────────────────────────────────────────
#  FITS
# ─────────────────────────────────────────────────────────────────────────────

def _r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def fit_linear_origin(x, y):
    """y ≈ a·x through the origin (the α(1−η) model). Returns (a, R²)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    a = float((x * y).sum() / max((x * x).sum(), 1e-30))
    return a, _r2(y, a * x)


def fit_power(x, y):
    """y ≈ A·x^p via log-log least squares (positive points only).
    Returns (A, p, R² in log space, R² in linear space)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (x > 0) & (y > 0)
    if m.sum() < 2:
        return None
    lx, ly = np.log(x[m]), np.log(y[m])
    p, lnA = np.polyfit(lx, ly, 1)
    A = float(np.exp(lnA))
    return A, float(p), _r2(ly, p * lx + lnA), _r2(y[m], A * x[m] ** p)


def fit_tq_exponent(b, dL):
    """ln ΔL = ln β_p − p·ln2·b  →  (β_p, p, R² in log space)."""
    b, dL = np.asarray(b, float), np.asarray(dL, float)
    m = dL > 0
    if m.sum() < 2:
        return None
    slope, intercept = np.polyfit(b[m], np.log(dL[m]), 1)
    p = -slope / math.log(2.0)
    beta_p = float(np.exp(intercept))
    return beta_p, float(p), _r2(np.log(dL[m]), slope * b[m] + intercept)


# ─────────────────────────────────────────────────────────────────────────────
#  ARMS
# ─────────────────────────────────────────────────────────────────────────────

def sct_arm(data):
    """Measured SCT distortion–rate curve (KV=fp16, no LoRA) + shape fits."""
    pts = sorted(_pts(data, lora=False, sct=True, kv=False), key=lambda p: p["energy"])
    etas = [p["energy"] for p in pts]
    dLs = [p["dL"] for p in pts]
    wbytes = [p["weight_bytes"] for p in pts]
    dense = _pts(data, lora=False, sct=False, kv=False)[0]

    x = [1.0 - e for e in etas]
    lin = fit_linear_origin(x, dLs)
    pow_ = fit_power(x, dLs)
    return {
        "etas": etas, "dL": dLs, "weight_bytes": wbytes,
        "dense_weight_bytes": dense["weight_bytes"],
        "alpha_global": lin[0], "alpha_global_r2": lin[1],
        "power": None if pow_ is None else
            {"A": pow_[0], "p": pow_[1], "r2_log": pow_[2], "r2_lin": pow_[3]},
        "concave": None if pow_ is None else bool(pow_[1] < 1.0),
    }


def tq_arm(data):
    """Measured TQ distortion (dense weights, no LoRA) + exponent fit."""
    pts = sorted(_pts(data, lora=False, sct=False, kv=True), key=lambda p: p["b_eff"])
    bs = [p["b_eff"] for p in pts]
    dLs = [p["dL"] for p in pts]
    kv_bytes = [p["kv_bytes"] for p in pts]
    fp16 = _pts(data, lora=False, sct=False, kv=False)[0]
    fit = fit_tq_exponent(bs, dLs)
    return {
        "b_eff": bs, "dL": dLs, "kv_bytes": kv_bytes,
        "kv_bytes_fp16": fp16["kv_bytes"],
        "fit": None if fit is None else
            {"beta_p": fit[0], "p": fit[1], "r2_log": fit[2]},
    }


def additivity_arm(data, sct, tq):
    """cross(η,b) = ΔL_joint − ΔL_sct(η) − ΔL_tq(b), relative to ΔL_joint."""
    sct_fn = build_sct_dL_fn(sct["etas"], sct["dL"])
    b_arr, dl_arr = np.asarray(tq["b_eff"], float), np.asarray(tq["dL"], float)
    order = np.argsort(b_arr)
    b_arr, dl_arr = b_arr[order], dl_arr[order]

    def tq_fn(b):
        return float(np.interp(b, b_arr, dl_arr))

    rows = []
    for p in _pts(data, lora=False, sct=True, kv=True):
        additive = sct_fn(p["energy"]) + tq_fn(p["b_eff"])
        cross = p["dL"] - additive
        rows.append({
            "energy": p["energy"], "b_eff": p["b_eff"],
            "dL_joint": p["dL"], "dL_additive": additive,
            "cross": cross,
            "cross_rel": cross / p["dL"] if abs(p["dL"]) > 1e-9 else 0.0,
        })
    rels = [r["cross_rel"] for r in rows]
    return {
        "rows": rows,
        "median_cross_rel": float(np.median(rels)) if rels else None,
        "max_abs_cross_rel": float(np.max(np.abs(rels))) if rels else None,
        "sign": ("sub-additive (negative)" if rels and np.median(rels) < 0
                 else "super-additive (positive)" if rels else "n/a"),
    }


def lora_arm(data, sct):
    """Recovery fraction per η at KV=fp16, plus the post-LoRA distortion curve."""
    on = {p["energy"]: p for p in _pts(data, lora=True, sct=True, kv=False)}
    rows, etas_l, dl_l = [], [], []
    for eta, dL_off in zip(sct["etas"], sct["dL"]):
        p_on = on.get(eta)
        if p_on is None:
            continue
        rec = 1.0 - p_on["dL"] / dL_off if abs(dL_off) > 1e-9 else 0.0
        rows.append({"energy": eta, "dL_off": dL_off, "dL_on": p_on["dL"],
                     "recovery": rec})
        etas_l.append(eta)
        dl_l.append(max(p_on["dL"], 0.0))  # LoRA can overshoot the dense baseline
    recs = [r["recovery"] for r in rows]
    return {
        "rows": rows,
        "median_recovery": float(np.median(recs)) if recs else None,
        "etas": etas_l, "dL": dl_l,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SOLVER VALIDATION (Part A on measured LLM curves)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMByteModel(ByteModel):
    """ByteModel whose KV cost slope is FITTED from the measured compressed KV
    bytes (kv_bytes ≈ c0 + c1·b_eff — TurboQuant stores per-token norms/scales,
    hence the intercept). We fold the intercept c0 into weight_bytes so the
    parent's linear-in-b machinery (optimum's b_max, marginal_kv) stays exact,
    and override the fp16 KV reference with the measured value."""
    kv_intercept: float = 0.0
    measured_dense_kv: float = 0.0

    def weight_bytes(self, eta: float) -> float:
        return super().weight_bytes(eta) + self.kv_intercept

    def dense_kv_bytes(self) -> float:
        return self.measured_dense_kv


def build_llm_rd(sct, tq, sct_curve_override=None):
    """RateDistortion over the MEASURED 70B/8B curves.

    sct_curve_override=(etas, dL) substitutes e.g. the post-LoRA curve.
    """
    # measured KV byte line: kv_bytes = c0 + c1·b_eff
    c1, c0 = np.polyfit(tq["b_eff"], tq["kv_bytes"], 1)
    # parent kv_bytes(b) = L·n_layers_kv·d_head·2b/8 → set the product to match c1
    bm = LLMByteModel(
        weight_bytes_fn=build_weight_bytes_fn(sct["etas"] + [1.0],
                                              sct["weight_bytes"] + [sct["dense_weight_bytes"]]),
        dense_weight_bytes=sct["dense_weight_bytes"],
        d_head=1, n_layers_kv=1, L=float(c1) * 8.0 / 2.0,
        kv_intercept=float(max(c0, 0.0)),
        measured_dense_kv=tq["kv_bytes_fp16"],
    )
    if sct_curve_override is not None:
        sct_fn = build_sct_dL_fn(*sct_curve_override)
    else:
        sct_fn = build_sct_dL_fn(sct["etas"] + [1.0], sct["dL"] + [0.0])
    fit = tq["fit"]
    return RateDistortion(sct_dL_fn=sct_fn, beta_p=fit["beta_p"], p=fit["p"], bytes=bm)


def solver_arm(data, sct, tq, lora, budget_fracs=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """Predicted (η*, b*) per budget vs the best MEASURED point under the same
    budget. Budgets are fractions of the measured dense total (weights + fp16 KV
    at the sweep's kv_ref_tokens context)."""
    rd = build_llm_rd(sct, tq)
    dense_total = rd.bytes.dense_total()

    eta_lo = min(sct["etas"])
    eta_grid = np.linspace(eta_lo, 0.9999, 300)
    b_grid = np.linspace(1.0, 8.0, 300)

    rows = []
    measured = _pts(data, lora=False)
    for frac in budget_fracs:
        budget = frac * dense_total
        pred = rd.optimum(budget, eta_grid=eta_grid, b_grid=b_grid)
        under = [p for p in measured if p["total_bytes"] <= budget]
        best = min(under, key=lambda p: p["dL"]) if under else None
        rows.append({
            "budget_frac": frac,
            "predicted": pred,
            "measured_best": None if best is None else {
                "label": best["label"], "energy": best["energy"],
                "b_eff": best["b_eff"], "dL": best["dL"],
                "total_bytes": best["total_bytes"], "utility": best.get("utility"),
            },
        })

    # LoRA re-solve: does the LLM reproduce the toy's "compress weights harder
    # once LoRA recovers the bias" shift (toy: η* 0.84 → 0.30)? Use the tightest
    # FEASIBLE budget (the solver refuses to extrapolate below the measured η range).
    lora_shift = None
    feasible = [r for r in rows if r["predicted"] is not None]
    if lora["etas"] and len(lora["etas"]) >= 2 and tq["fit"] is not None and feasible:
        rd_l = build_llm_rd(sct, tq, sct_curve_override=(lora["etas"] + [1.0],
                                                         lora["dL"] + [0.0]))
        frac = feasible[0]["budget_frac"]
        o0 = feasible[0]["predicted"]
        o1 = rd_l.optimum(frac * dense_total, eta_grid=eta_grid, b_grid=b_grid)
        if o0 and o1:
            lora_shift = {"budget_frac": frac, "eta_no_lora": o0["eta"],
                          "eta_with_lora": o1["eta"], "b_no_lora": o0["b"],
                          "b_with_lora": o1["b"]}
    return {"dense_total_bytes": dense_total, "rows": rows, "lora_shift": lora_shift}


# ─────────────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, spec=".3g"):
    return "n/a" if v is None else format(v, spec)


def write_report(out_md, model_name, sct, tq, add, lora, sol, toy):
    t_sct = toy.get("sct") or {}
    t_tq = ((toy.get("tq") or {}).get("downstream") or {})
    t_comb = toy.get("combined") or {}
    t_lora = toy.get("lora") or {}

    lines = [
        f"# Theory validation at LLM scale — {model_name}",
        "",
        "ΔL := ln(ppl / ppl_dense) (mean-NLL gap), measured by `run_pareto.py` with",
        "every forward routed through the TurboQuant cache. Each claim below was",
        "first established on the analytic toy (`theory/RESULTS.md`); this report",
        "re-tests it on the real model.",
        "",
        "## 1. SCT distortion–rate shape (toy: concave, α(1−η) REJECTED)",
        "",
        "| fit | LLM measured | toy |",
        "|---|---|---|",
        f"| linear α(1−η) R² | {_fmt(sct['alpha_global_r2'])} | {_fmt(t_sct.get('alpha_global_r2'))} |",
        f"| power exponent p_w | {_fmt((sct['power'] or {}).get('p'))} | {_fmt(t_sct.get('power_p'))} |",
        f"| power R² (log) | {_fmt((sct['power'] or {}).get('r2_log'))} | {_fmt(t_sct.get('power_r2'))} |",
        f"| concave (p_w<1)? | {sct['concave']} | True |",
        "",
        "## 2. TurboQuant ΔL ≈ β_p·2^(−p·b) (toy: p≈1.83, asymptote 2)",
        "",
        "| quantity | LLM measured | toy |",
        "|---|---|---|",
        f"| effective exponent p | {_fmt((tq['fit'] or {}).get('p'))} | {_fmt(t_tq.get('eff_exponent'))} |",
        f"| β_p | {_fmt((tq['fit'] or {}).get('beta_p'))} | {_fmt(t_tq.get('beta_p'))} |",
        f"| fit R² (log) | {_fmt((tq['fit'] or {}).get('r2_log'))} | — |",
        "",
        "b_eff = (key_bits + value_bits)/2; the sweep's KV grid is coarse (2–4 bits),",
        "so p is a 3–4-point fit — treat as an order check, not a precision estimate.",
        "",
        "## 3. Additivity of the two error sources (toy: sub-additive, up to −46%)",
        "",
        f"- median cross/joint = **{_fmt(add['median_cross_rel'])}**"
        f" (toy {_fmt(t_comb.get('median_cross_rel'))})",
        f"- max |cross|/joint = **{_fmt(add['max_abs_cross_rel'])}**"
        f" (toy {_fmt(t_comb.get('max_cross_rel'))})",
        f"- sign: **{add['sign']}** (toy: sub-additive — weight compression makes",
        "  the KV cache cheaper to quantize)",
        "",
        "## 4. Recovery-LoRA (toy: recovers ~84% of the SCT bias, ~constant in η)",
        "",
        "| η | ΔL no-LoRA | ΔL LoRA | recovered |",
        "|---|---|---|---|",
    ]
    for r in lora["rows"]:
        lines.append(f"| {r['energy']:g} | {r['dL_off']:.4f} | {r['dL_on']:.4f} "
                     f"| {r['recovery']:.0%} |")
    lines += [
        "",
        f"Median recovery **{_fmt(lora['median_recovery'])}**"
        f" (toy {_fmt(t_lora.get('median_recovery'))}).",
        "",
        "## 5. Solver check — predicted (η*, b*) vs best measured point per budget",
        "",
        "Budget = fraction of the measured dense total (weights + fp16 KV at the",
        "sweep context). Predicted optima come from `theory/models/rate_distortion.py`",
        "fed ONLY the measured curves above; 'measured best' is the lowest-ΔL sweep",
        "point whose bytes fit the budget (grid is coarse — agreement means the",
        "prediction lands in the same cell, not the same decimal). 'infeasible'",
        "means the budget is below what the measured η range can reach — the solver",
        "does not extrapolate; sweep lower energies to probe those budgets.",
        "",
        "| budget | predicted η*, b* | predicted ΔL | measured best (η, b_eff) | measured ΔL |",
        "|---|---|---|---|---|",
    ]
    for row in sol["rows"]:
        pr, mb = row["predicted"], row["measured_best"]
        pr_s = "infeasible" if pr is None else f"{pr['eta']:.3f}, {pr['b']:.2f}"
        pr_d = "—" if pr is None else f"{pr['dL']:.4f}"
        mb_s = "none under budget" if mb is None else \
            f"{mb['label']} ({'1.0' if mb['energy'] is None else format(mb['energy'], 'g')}, " \
            f"{'fp16' if mb['b_eff'] is None else format(mb['b_eff'], 'g')})"
        mb_d = "—" if mb is None else f"{mb['dL']:.4f}"
        lines.append(f"| {row['budget_frac']:.0%} | {pr_s} | {pr_d} | {mb_s} | {mb_d} |")

    if sol["lora_shift"]:
        s = sol["lora_shift"]
        lines += [
            "",
            f"With the post-LoRA curve at a 50% budget the solver moves η* "
            f"{s['eta_no_lora']:.3f} → **{s['eta_with_lora']:.3f}** and b* "
            f"{s['b_no_lora']:.2f} → **{s['b_with_lora']:.2f}** "
            "(toy: LoRA lets the solver compress weights harder, η* 0.84→0.30).",
        ]
    lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))


def plot_validation(out_png, sct, tq, add, sol, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (1) SCT shape
    ax = axes[0][0]
    x = np.array([1.0 - e for e in sct["etas"]])
    y = np.array(sct["dL"])
    ax.plot(x, y, "o-", label="measured ΔL")
    xs = np.linspace(max(x.min(), 1e-4), x.max(), 100)
    ax.plot(xs, sct["alpha_global"] * xs, "--",
            label=f"α(1−η), R²={sct['alpha_global_r2']:.2f}")
    if sct["power"]:
        pw = sct["power"]
        ax.plot(xs, pw["A"] * xs ** pw["p"], ":",
                label=f"A(1−η)^{pw['p']:.2f}, logR²={pw['r2_log']:.2f}")
    ax.set_xlabel("1 − η (discarded energy)")
    ax.set_ylabel("ΔL = ln ppl − ln ppl₀")
    ax.set_title("SCT distortion–rate (KV=fp16, no LoRA)")
    ax.legend(fontsize=8)

    # (2) TQ exponent
    ax = axes[0][1]
    b = np.array(tq["b_eff"])
    dl = np.array(tq["dL"])
    m = dl > 0
    ax.semilogy(b[m], dl[m], "o", label="measured")
    if tq["fit"]:
        f = tq["fit"]
        bs = np.linspace(b.min(), b.max(), 50)
        ax.semilogy(bs, f["beta_p"] * 2.0 ** (-f["p"] * bs), "--",
                    label=f"β·2^(−{f['p']:.2f}b), R²={f['r2_log']:.2f}")
    ax.set_xlabel("b_eff (bits/coord)")
    ax.set_ylabel("ΔL")
    ax.set_title("TurboQuant distortion vs bits (dense weights)")
    ax.legend(fontsize=8)

    # (3) additivity
    ax = axes[1][0]
    if add["rows"]:
        pred = [r["dL_additive"] for r in add["rows"]]
        meas = [r["dL_joint"] for r in add["rows"]]
        ax.scatter(pred, meas, c=[r["energy"] for r in add["rows"]], cmap="viridis")
        lim = [0, max(max(pred), max(meas)) * 1.05]
        ax.plot(lim, lim, "k--", lw=1, label="additive (y=x)")
        ax.set_xlabel("ΔL_sct(η) + ΔL_tq(b)  (additive prediction)")
        ax.set_ylabel("ΔL_joint (measured)")
        ax.legend(fontsize=8)
    ax.set_title(f"Additivity — median cross {_fmt(add['median_cross_rel'], '.1%')}, "
                 f"{add['sign']}")

    # (4) solver overlay in (bytes, ΔL)
    ax = axes[1][1]
    pts = _pts(data, lora=False)
    ax.scatter([p["total_bytes"] / 2**30 for p in pts], [p["dL"] for p in pts],
               s=18, alpha=0.6, label="measured sweep points")
    px, py = [], []
    for row in sol["rows"]:
        if row["predicted"]:
            px.append(row["predicted"]["total_bytes"] / 2**30)
            py.append(row["predicted"]["dL"])
    ax.plot(px, py, "r*-", ms=12, label="predicted optimum per budget")
    ax.set_xlabel("total bytes (GiB, weights + KV@ref context)")
    ax.set_ylabel("ΔL")
    ax.set_title("Part A solver on measured curves vs sweep")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def run(results_json: str, out_dir: str | None = None) -> dict:
    out_dir = out_dir or os.path.dirname(os.path.abspath(results_json))
    data = load_sweep(results_json)
    model_name = data["config"]["model"]
    toy = load_toy()

    sct = sct_arm(data)
    tq = tq_arm(data)
    add = additivity_arm(data, sct, tq)
    lora = lora_arm(data, sct)
    sol = solver_arm(data, sct, tq, lora) if tq["fit"] else \
        {"dense_total_bytes": None, "rows": [], "lora_shift": None}

    out_md = os.path.join(out_dir, "theory_validation.md")
    out_png = os.path.join(out_dir, "theory_validation.png")
    out_json = os.path.join(out_dir, "theory_validation.json")
    write_report(out_md, model_name, sct, tq, add, lora, sol, toy)
    plot_validation(out_png, sct, tq, add, sol, data)
    with open(out_json, "w") as f:
        json.dump({"model": model_name, "sct": sct, "tq": tq, "additivity": add,
                   "lora": lora, "solver": sol}, f, indent=2, default=float)
    print(f"  theory validation -> {out_md}\n                    -> {out_png}")
    return {"sct": sct, "tq": tq, "additivity": add, "lora": lora, "solver": sol}
