# Part A — Joint Rate-Distortion Solver: Results

**Predicted joint optimum** for `ΔL(η,b) = ΔL_sct(η) + β_p·2^(−p·b)`.

**Important:** the SCT term is the **MEASURED distortion–rate curve** ΔL_sct(η),
NOT the `α(1−η)` model from the doc — the toy rejected `α(1−η)` (global
R²=0.23; the true relationship is the concave,
curvature-weighted quadratic). The TQ term keeps the parametric form with the
measured exponent p (it fit well, R²≈1.000).

## Measured constants (from theory/toy)
| constant | value | meaning |
|---|---|---|
| ΔL_sct(η) | curve | interpolated measured distortion (concave; α(1−η) rejected) |
| α (local only) | 7.8261e-04 | SCT slope near η→1 (not used by the solver) |
| β_p | 2.8382e-03 | TurboQuant variance coefficient |
| p | 1.834 | effective bit-exponent (theory 2; finite-rate <2) |

## Byte model (balanced regime)
- dense total: 8.192e+04 B (weights 4.10e+04 + KV 4.10e+04, L=160)
- KV: b bits/coord for K (Prod) and V (MSE), d_head=64, 1 layer(s)
- L chosen so dense weights ≈ dense KV (both levers active). The weight:KV ratio
  is a deployment knob (model size × context length); the solver is scale-agnostic.

## Budget-constrained optimum (50% of dense)
- **η\* = 0.8385**,  **b\* = 5.351**,  ΔL\* = 6.4240e-05
- spend: weights 2.72e+04 B + KV 1.37e+04 B
- KKT check: marginal loss/byte — weights 1.571e-09 vs KV 1.565e-09
  (equal ⇒ interior optimum, the equal-marginal condition holds).

## Regime structure
- tight budget (12%): η\*=0.300, b\*=1.47
- loose budget (95%): η\*=1.000, b\*=8.00
- **Finding (contradicts §A.3's guess):** measured SCT distortion is concave, so
  weight compression is cheap per byte — tight budgets lean on **weights first**,
  not "TQ first". See `regimes.png`.

## Recovery-LoRA
With recovery-LoRA (α 7.826e-04→1.456e-04, 5.4× smaller): η* 0.8385→0.3000 (Δ=-0.5385), b* 5.35→8.00. Lower η* ⇒ compress weights HARDER and reallocate freed bytes to KV.

## Figures
- `surface.png` — ΔL(η,b) surface, iso-budget line, marked (interior) optimum
- `regimes.png` — optimal (η\*, b\*) vs budget + equalised KKT marginals
