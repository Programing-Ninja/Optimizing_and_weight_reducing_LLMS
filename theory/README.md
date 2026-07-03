# Theory Track — Analytic Error Models + Toy Validation

Implements the **Part A** (rate-distortion solver) and **Part B** (toy
validation) deliverables of [`THEORETICAL_ANALYSIS.md`](THEORETICAL_ANALYSIS.md).
Everything runs on a laptop/single GPU in ~1 minute — no HPC needed.

## TL;DR — run it all

```bash
conda activate ml_lab
python theory/run_experiments.py      # runs Part B arms + Part A solver
```

Outputs land in `theory/toy/results/` and `theory/models/results/`, and a
consolidated summary in [`theory/RESULTS.md`](RESULTS.md).

## What it validates (and the surprises)

| Doc claim | Verdict from the toy |
|---|---|
| **A.1** loss is locally quadratic in ΔW | **Confirmed as an equality** (cross-term ~1e-4; known-H readout check to 7e-8; Eckart–Young control exact). |
| **A.1 model** ΔL ≈ α(1−η) | **REJECTED** — ΔL is concave in (1−η) (linear R²=0.23 iso, **−2.6** aniso). The exact model is the curvature-weighted quadratic `tr(ΔW H ΔW)`; a power law `(1−η)^0.6` is a rough 1-D summary. **The solver uses the measured curve, not α(1−η).** Truncating all layers at once is also *sub-additive across layers* (measured is 0.2–1.0× the sum of per-layer quadratics). |
| **A.2** TurboQuant = unbiased variance ~2^(−2b) | **Confirmed** (bias²/var ~1e-5; the error histogram is a wide bell centred on 0), but the **effective exponent is p≈1.83, not 2** (finite-rate Lloyd–Max). |
| **A.3** SCT bias + TQ variance are additive | **Mostly additive** (median cross 5%) but **sub-additive coupling up to 46%** at aggressive joint compression — compressing weights first makes the KV cache cheaper to quantize. |
| Recovery-LoRA shrinks α | **Confirmed: ~84% of the bias recovered, 5.4× smaller local α**; concave shape preserved. In the solver LoRA then compresses weights harder (η\* 0.84→0.30) and reallocates bytes to KV. |
| **A.3 guess** "tight budgets lean on TQ first" | **CONTRADICTED** — measured SCT distortion is concave (cheap per byte), so tight budgets lean on **weight compression first**. |

> **Terminology flag (raised per workspace rule 6):** §A.3 says recovery-LoRA
> should push "η upward (compress weights harder)". Since η is *retained* energy,
> harder compression means η **down**. The solver confirms the *intent* (η*
> 0.708→0.640) — the doc's "upward" wording is inverted vs the convention.

## Layout

```
theory/
  THEORETICAL_ANALYSIS.md   the planning/reference doc (Parts A–E)
  run_experiments.py        orchestrator: all arms + solver + master RESULTS.md
  RESULTS.md                consolidated results (generated)
  toy/                      Part B — toy validation
    gaussian_data.py          Gaussian token generator (iso / anisotropic Σ_x)
    linear_attn_model.py      linear-attn + linear-MLP toy; the local-quadratic property
    sct_utils.py              energy→rank→truncated weight via vendored SpectralLinear
    common.py                 tee-logging, fits, seeding
    sct_arm.py                step 2 — SCT bias, α, Eckart–Young, known-H
    tq_arm.py                 step 3 — TurboQuant unbiasedness, variance law, β, p
    combined_arm.py           step 4 — additivity / cross-term test
    lora_arm.py               step 5 — recovery-LoRA α-shrink
    results/                  per-arm JSON + PNG + .log
  models/                   Part A — rate-distortion solver
    rate_distortion.py        RateDistortion + ByteModel; KKT optimum
    run_solver.py             load measured (α,β,p) → surface, regimes, optimum
    results/                  surface.png, regimes.png, optimum.json, RESULTS.md
```

## Design note — why the toy makes A.1 an *equality*

The toy loss is **exactly quadratic in each weight matrix when the others are
held fixed** (attention is bilinear in (Wq,Wk) but linear in each separately; the
rest of the chain is linear). So for a single-matrix perturbation ΔW near the
trained optimum, `L(W+ΔW)−L(W) = E‖ΔO‖² + 2E[(O−Y)·ΔO]`, and the linear
cross-term vanishes as the residual → 0. That turns A.1's `≈` into a checkable
`=`. See `linear_attn_model.quadratic_vs_measured_deltaL`.

## Run individual arms

```bash
cd theory/toy
python sct_arm.py        # SCT (iso + aniso)
python tq_arm.py         # TurboQuant
python combined_arm.py   # additivity
python lora_arm.py       # recovery-LoRA
cd ../models && python run_solver.py
```

Requires the sibling `SCT/` and `turboquant/` submodules on the path
(`run_experiments.py` wires them automatically; standalone runs need
`PYTHONPATH=SCT:turboquant:theory/toy:theory/models`).
