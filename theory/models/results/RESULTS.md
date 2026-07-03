# Part A — Joint Rate-Distortion Solver: Results

**Predicted joint optimum surface** for `ΔL(η,b) = α(1−η) + β_p·2^(−p·b)`,
using constants measured by the Part B toy.

## Measured constants (from theory/toy)
| constant | value | meaning |
|---|---|---|
| α | 7.8261e-04 | SCT bias slope (local, small-perturbation) |
| β_p | 2.8382e-03 | TurboQuant variance coefficient |
| p | 1.834 | effective bit-exponent (theory 2; finite-rate <2) |

## Byte model
- dense total: 8.192e+04 B (weights 4.10e+04 + KV 4.10e+04, L=160)
- KV: b bits/coord for K (Prod) and V (MSE), d_head=64, 1 layer(s)

## Budget-constrained optimum (35% of dense)
- **η\* = 0.7080**,  **b\* = 3.579**,  ΔL\* = 2.5853e-04
- spend: weights 1.95e+04 B + KV 9.16e+03 B

## Regime structure (the crossover the project hunts for)
- tight budget (20%): η\*=0.500, b\*=1.86
- loose budget (95%): η\*=1.000, b\*=8.00
- At the optimum the KKT marginals (loss reduction per byte) equalise across the
  two methods — see `regimes.png` right panel.

## Recovery-LoRA
With recovery-LoRA (α 7.826e-04→1.456e-04, 5.4× smaller): η* 0.7080→0.6403 (Δ=-0.0677), b* 3.58→4.61. Lower η* ⇒ compress weights HARDER and reallocate freed bytes to KV.

## Figures
- `surface.png` — ΔL(η,b) surface, iso-budget line, marked optimum
- `regimes.png` — optimal (η\*, b\*) vs budget + equalised KKT marginals
