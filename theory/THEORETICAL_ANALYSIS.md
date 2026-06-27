# Theoretical Analysis: The Combined Geometry of Weight Compression and KV Quantization

**Status:** planning / reference doc (written 2026-06-27, while HPC access is unavailable)
**Owner:** Harshvardhan
**Companion code (to be built):** `theory/toy/` (toy linear model), `theory/models/` (analytic SCT/TurboQuant error models)

---

## 0. Why this doc exists

The campus HPC is not reachable right now, so the large-model Pareto sweep
(`run_pareto.py`, Llama-3.1-8B) cannot run. That sweep is **empirical** — it measures
utility vs. compression but does not explain *why* the frontier has the shape it has,
nor does it predict where the optimum should sit before we spend A100-hours finding it.

This document defines the **theory track** we can do entirely on a laptop:

1. Write down closed-form (or near-closed-form) error models for **SCT** (weight
   compression) and **TurboQuant** (KV quantization) individually.
2. Validate those models on a deliberately *tiny, analytically tractable* system — a
   1–2 layer linear-attention + linear-MLP network on Gaussian data — where we can
   compute the true error and check the model is right.
3. Compose the two models into a **single distortion–rate objective** and predict the
   *optimal joint operating point* (how much to compress weights vs. how much to
   quantize the cache for a fixed memory budget).
4. Use those predictions to (a) place the HPC sweep's grid intelligently and (b)
   sanity-check the empirical frontier when it does run.

The project's north star: **find the optimal extent of weight compression and
quantization jointly**, with minimal utility loss. This doc is the math half of that.

---

## Part A — Analytic error models

### A.1 SCT as spectral low-rank truncation (a *bias* term)

SCT stores each weight as `W = U diag(s) V^T` with `s ∈ R^r`. Picking the rank `r`
via an **energy** target `η ∈ (0,1]` means: keep the smallest `r` such that

```
    Σ_{i=1}^{r} σ_i²  ≥  η · Σ_{i=1}^{n} σ_i²
```

where `σ_1 ≥ σ_2 ≥ …` are the singular values of the (pretrained) dense `W`.

The truncation error is exactly the discarded spectral energy:

```
    ||W − W_r||_F²  =  Σ_{i>r} σ_i²  =  (1 − η) · ||W||_F²       (Eckart–Young)
```

So **SCT injects a deterministic, structured bias** into every compressed layer. Its
size is controlled by `η` and the *spectral decay* of the layer:

- A fast-decaying spectrum (typical of large attention/MLP projections) ⇒ high `η`
  costs few ranks ⇒ cheap compression. This is exactly why CLAUDE.md notes SCT works
  best at hidden dim ≥ 2048.
- A flat spectrum ⇒ truncation is expensive ⇒ little compression for the same `η`.

**Predicted utility curve (the hypothesis to test):** for a downstream loss that is
locally quadratic in the weight perturbation `ΔW = W − W_r`, the excess loss should be

```
    ΔL_SCT(η)  ≈  ½ · tr(ΔW^T H ΔW)  ∝  Σ_{i>r(η)} σ_i² · (curvature factor)
```

i.e. utility loss grows like the **discarded energy `(1−η)`**, modulated by how much the
network's loss curvature `H` aligns with the discarded singular directions. The toy
experiment (Part B) is designed so `H` is known, turning the `≈` into an `=`.

Knobs the model exposes: `η` (energy) → `r` (rank) → bytes `r(m+n+1)` vs. dense `mn`.

### A.2 TurboQuant as unbiased inner-product noise (a *variance* term)

TurboQuant compresses KV-cache entries with random rotation Π + Lloyd–Max scalar
quantization at `b−1` bits + a 1-bit QJL sign residual, and — crucially — the combined
**inner-product estimator is unbiased** (CLAUDE.md, TurboQuantProd flow). So unlike SCT,
TurboQuant contributes **zero bias but nonzero variance** to the attention logits.

Model each quantized key/value coordinate as `x̃ = x + ε`, with `E[ε]=0` and per-coordinate
quantization variance `D(b)`. For a `b`-bit scalar quantizer on a source with variance
`τ²`, high-rate quantization theory gives the standard distortion–rate law:

```
    D(b)  ≈  c · τ² · 2^(−2b)               (c = quantizer shape constant)
```

For a Lloyd–Max quantizer matched to the (rotated, ≈ Beta/Gaussian-like) source, `c` is
the known optimal constant; the random rotation Π is what makes the per-coordinate source
distribution stable so a single codebook applies. The attention score is an inner product
`q·k`, so the **score error variance** scales like

```
    Var[ q·k̃ − q·k ]  ≈  ||q||² · D(b)  ∝  2^(−2b)
```

and the QJL sign bit further corrects the residual norm term (the `||r||·√(π/2)/d` term in
the estimator), tightening the constant without changing the `2^(−2b)` rate.

**Predicted utility curve:** because the estimator is unbiased, the leading effect on a
quadratic downstream loss is a **variance term that decays geometrically in bits**:

```
    ΔL_TQ(b)  ≈  ½ · (sensitivity) · 2^(−2b)
```

This is the key qualitative contrast with SCT: **SCT error is bias and falls ~linearly in
retained-energy; TurboQuant error is variance and falls ~exponentially in bits.** That
asymmetry is what makes the *joint* optimum nontrivial and worth solving.

Knobs the model exposes: `b` (bits/coordinate for keys; value bit-width 2 vs 4) → cache
bytes per token ∝ `b` vs. fp16's 16.

### A.3 The combined objective — rate–distortion for joint compression

Put a single memory budget on the model: weight bytes + KV bytes ≤ `M`. Treat the two
distortions as approximately **independent and additive** to first order (bias from SCT,
variance from TQ act on different operators — static weights vs. dynamic cache — so their
cross term is second-order; **this independence is itself a claim the toy model must
check**). Then total excess loss:

```
    ΔL(η, b)  ≈  α · (1 − η)        +        β · 2^(−2b)
                  └ SCT bias ┘                └ TQ variance ┘
```

with `α`, `β` task/architecture constants we *estimate from the single-method sweeps*
(Part C, steps 1–2). Memory cost:

```
    M(η, b)  ≈  γ · r(η)            +        δ · b           (+ constants)
```

**Optimal allocation.** Minimizing `ΔL` s.t. `M ≤ M_budget` is a rate–distortion problem;
the Lagrangian/KKT stationarity condition says the **marginal loss reduction per byte must
equalize across the two methods**:

```
        ∂ΔL/∂(weight bytes)   =   ∂ΔL/∂(KV bytes)
```

Spending the next byte on whichever method currently has the steeper loss-vs-byte slope.
Because TQ's distortion is exponential in bits while SCT's is roughly linear in energy,
the prediction is a regime structure:

- **Tight budgets:** lean on TQ first (cheap exponential wins per bit) until its curve
  flattens, then spend on weight rank.
- **Loose budgets:** the opposite — push `b` to 4-bit early (diminishing returns), invest
  remaining budget in higher SCT energy.
- A **crossover budget** where the two marginal slopes meet = the joint optimum the
  project is hunting for.

Add recovery-LoRA as a third lever: it does not change `M` much (adapters are tiny) but it
**reduces `α`** (it folds into attention projections and recovers the bias SCT introduced —
see `iteration.txt`). So in the model, recovery-LoRA is an `α`-shrinking post-step, and the
interesting question is whether it shifts the optimal `η` *upward* (compress weights harder
because LoRA cleans up after it).

**Deliverable of Part A:** a small Python module `theory/models/` that, given measured
`(α, β, γ, δ)`, draws the predicted `ΔL(η,b)` surface and marks the budget-constrained
optimum — to overlay on the empirical Pareto frontier later.

---

## Part B — Toy validation experiment (runs on a laptop)

**Goal:** a system small enough that the *true* loss is computable in closed form, so we
can confirm A.1–A.3 are quantitatively right (not just directionally) before trusting them
at 8B scale.

### B.1 Setup

- **Data:** inputs `x ∈ R^d` drawn i.i.d. Gaussian `N(0, Σ_x)` (start `Σ_x = I`, then a
  controlled anisotropic `Σ_x` to exercise spectral decay). Teacher target `y = f*(x)`
  generated by a fixed random *linear* teacher (optionally with a known nonlinearity later).
- **Model:** 1–2 layer network with **linear attention** (Katharopoulos et al. 2020 —
  attention without softmax is exactly linear, hence analytically tractable) feeding a
  **linear MLP**. With linear attention + linear MLP and Gaussian data, the end-to-end map
  is linear and the population loss is a quadratic form ⇒ `H` (the curvature in A.1/A.2) is
  known in closed form.
- **Task:** regression MSE (clean, quadratic loss — the cleanest place to see bias vs.
  variance). A classification variant can come later.

### B.2 What we measure

1. **SCT arm:** factor the trained weight matrices with SCT at a sweep of energies
   `η ∈ {0.5 … 0.999}`. Plot measured ΔMSE vs. `(1−η)` and vs. discarded energy
   `Σ_{i>r} σ_i²`. **Check:** does it match `½ tr(ΔW^T H ΔW)`? Does the slope give `α`?
2. **TurboQuant arm:** run the (already-vendored) TurboQuant estimator on the toy's
   K/V at `b ∈ {1,2,3,4}` bits. Plot measured ΔMSE vs. `2^(−2b)`. **Check:** straight line
   through the origin (unbiased!) with slope `β`? Confirm `E[ε]≈0` empirically.
3. **Combined arm:** apply both at once over the `(η, b)` grid. **Check:** does
   `ΔL ≈ α(1−η) + β2^(−2b)` hold, or is there a measurable cross term? Map the actual
   minimum-loss point at a fixed budget and compare to the A.3 prediction.
4. **Recovery-LoRA arm:** fit a tiny LoRA after SCT; confirm it lowers `α` and check
   whether the joint optimum's `η` moves up.

### B.3 Why this is worth doing

If the toy confirms the additive bias+variance model, we get a *predictive* tool: feed in
`(α,β)` cheaply estimated from a couple of small real-model runs and the model tells us
where to put the A100 grid — instead of brute-forcing a 6×5×2 sweep blind. If the toy
*refutes* additivity (large cross term), that itself is the headline finding: weight
compression and KV quantization interact, and the optimum can't be found by tuning each
alone.

---

## Part C — Analysis roadmap

| Step | What | Where it runs | Output |
|------|------|---------------|--------|
| 1 | Build `theory/toy/` model + Gaussian data generator | laptop | trained toy weights, known `H` |
| 2 | SCT energy sweep on toy → estimate `α`, verify Eckart–Young loss law | laptop | `alpha`, fit plot |
| 3 | TurboQuant bit sweep on toy → estimate `β`, verify unbiased `2^(−2b)` law | laptop | `beta`, fit plot |
| 4 | Combined `(η,b)` grid on toy → test additivity, locate empirical optimum | laptop | cross-term magnitude |
| 5 | Recovery-LoRA on toy → measure `α`-shrink and optimum shift | laptop | `α_LoRA` |
| 6 | `theory/models/` rate–distortion solver → predicted joint optimum surface | laptop | prediction figure |
| 7 | When HPC returns: estimate `(α,β)` from 2–3 small Llama points, predict frontier, **then** run `run_pareto.py` and overlay prediction vs. measured | HPC (A100) | validated/falsified model |

---

## Part D — References

Grouped by role. Items marked **(verify)** are ones to confirm exact title/venue before
citing in a writeup; the rest are well established.

### Weight low-rank / spectral compression
- Eckart, Young (1936). *The approximation of one matrix by another of lower rank.*
  Psychometrika. — the optimality of truncated SVD; basis for A.1.
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Zhao et al. (2024). *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
  Projection.* arXiv:2403.03507. — low-rank in the *gradient*, complementary framing to SCT.
- Aghajanyan et al. (2020). *Intrinsic Dimensionality Explains the Effectiveness of
  Language Model Fine-Tuning.* arXiv:2012.13255. — why low rank suffices.
- Absil, Mahony, Sepulchre (2008). *Optimization Algorithms on Matrix Manifolds.* —
  Stiefel manifold retraction theory behind SCT's `retract_all`/QR step.

### KV-cache quantization
- **(verify)** TurboQuant (2025). *Online vector quantization with near-optimal distortion
  rate.* — the vendored method; primary reference for A.2. Confirm authors/arXiv id.
- Zandieh et al. (2024). *QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with
  Zero Overhead.* arXiv:2406.03482. — the QJL sign-residual TurboQuant uses.
- Hooper et al. (2024). *KVQuant: Towards 10 Million Context Length LLM Inference with KV
  Cache Quantization.* arXiv:2401.18079.
- Liu et al. (2024). *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.*
  arXiv:2402.02750. — the 2-bit reference point (cf. CLAUDE.md cos_sim≈0.94).

### Quantization & rate–distortion theory
- Lloyd (1982, orig. 1957) and Max (1960). *Least squares quantization in PCM* / *Quantizing
  for minimum distortion.* — the Lloyd–Max optimal scalar quantizer in `codebook.py`.
- Cover, Thomas. *Elements of Information Theory.* — rate–distortion, the `2^(−2b)` law,
  the additive-distortion + Lagrangian framework of A.3.
- Dettmers, Zettlemoyer (2023). *The case for 4-bit precision: k-bit Inference Scaling
  Laws.* arXiv:2212.09720. — empirical bits-vs-quality scaling; sanity check for `β` regime.
- Johnson, Lindenstrauss (1984); Achlioptas (2003), *Database-friendly random projections.*
  — JL lemma underpinning the random rotation Π / QJL projection.

### Joint low-rank + quantization (closest prior art to *our* question)
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314.
- Li et al. (2023). *LoftQ: LoRA-Fine-Tuning-Aware Quantization for LLMs.* arXiv:2310.08659.
- Guo et al. (2023). *LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition.*
  arXiv:2311.12023. — explicitly decomposes `W ≈ Q + L` (quantized + low-rank); the most
  direct precedent for "how much rank vs. how many bits," though it targets *weights*, not
  the *KV cache* — our weight(SCT)×cache(TQ) split is the novel axis.

### Structured pruning (the third compression lever in scope)
- Ma et al. (2023). *LLM-Pruner: On the Structural Pruning of Large Language Models.*
  arXiv:2305.11627.
- Ashkboos et al. (2024). *SliceGPT: Compress Large Language Models by Deleting Rows and
  Columns.* arXiv:2401.15024. — itself spectral, so it interacts directly with SCT.
- Xia et al. (2023). *Sheared LLaMA: Accelerating LLM Pre-training via Structured Pruning.*
  arXiv:2310.06694.

### Architecture / scaling background
- Katharopoulos et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers
  with Linear Attention.* arXiv:2006.16236. — the linear-attention used in the toy model.
- Kaplan et al. (2020), *Scaling Laws for Neural Language Models* (arXiv:2001.08361);
  Hoffmann et al. (2022), *Chinchilla* (arXiv:2203.15556). — utility-vs-resource framing.

---

## Part E — Open questions to resolve as we go

1. **Independence of bias and variance (A.3).** Is the cross term really second-order, or
   does compressing weights change the cache statistics enough to couple them? Toy step 4
   decides this.
2. **Constant `c` / `β` for the *rotated* source.** TurboQuant's rotation is meant to make
   the source distribution codebook-friendly; we should confirm the empirical `2^(−2b)`
   slope matches the Lloyd–Max-on-Beta constant the codebooks were generated for.
3. **Does recovery-LoRA move the optimal `η`?** If yes, the joint sweep must include LoRA in
   the inner loop, not as a fixed post-step.
4. **Where do values (2 vs 4-bit) sit vs. keys?** TurboQuant treats them differently; the
   model currently lumps them into one `b`. May need separate `b_K`, `b_V`.

---

*Next action when resumed:* scaffold `theory/toy/` (data + linear-attention model + sweep
harness) and `theory/models/` (the `α(1−η)+β2^{−2b}` solver), then run steps 1–4 locally.
