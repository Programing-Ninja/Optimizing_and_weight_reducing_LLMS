# Optimizing_and_weight_reducing_LLMS

Here I try to find ways to get better weight and KV cache optimization, starting off with finding the optimal frontier by combining SCT and TurboQuant.

## Sub-projects

- `SCT/` — Spectral Compact Training: replaces nn.Linear with U diag(s) V^T factors; gradients flow through spectral factors with Stiefel retraction.
- `turboquant/` — TurboQuant KV cache compression: random rotation + Lloyd-Max quantization for keys + group quantization for values.

## Experiments

### SCT x TurboQuant Joint Pareto Frontier

**File:** `experiments/sct_tq_pareto.py`

Sweeps SCT energy thresholds (None=dense, 0.99, 0.95, 0.90, 0.80) x TurboQuant KV
configurations (fp16-full, k=4/v=4, k=3/v=4, k=3/v=2, k=2/v=2) on SmolLM2-135M.

Measures perplexity, weight bytes, compressed KV bytes, peak RSS, and tokens/sec
for each combination, then computes the Pareto frontier.

**Run (full sweep):**
```bash
# From sct_tq/ directory:
experiments/run.sh experiments/sct_tq_pareto.py --max-tokens 768
```

**Quick smoke test (2 energies x 2 TQ configs, ~30 seconds on CPU):**
```bash
experiments/run.sh experiments/sct_tq_pareto.py --quick --max-tokens 128
```

**With optional SCT finetune (to recover quality after truncation):**
```bash
experiments/run.sh experiments/sct_tq_pareto.py --max-tokens 512 --finetune-steps 200 --energies 0.95 0.90
```

**Outputs:**
- `experiments/sct_tq_pareto_results.json` — all config results
- `experiments/sct_tq_pareto.png` — scatter plot: perplexity vs total bytes, Pareto frontier highlighted

**Key findings (SmolLM2-135M, 128 tokens):**
- TurboQuant (k=3, v=4) gives 3.77x KV byte reduction (2949KB -> 783KB) at a perplexity cost of 3.6x (22.76 -> 81.82 on 128 tokens with full compression).
- SCT weight compression is negligible or negative at 135M scale (hidden_dim=576 < 2048 threshold). See CLAUDE.md for details.
- Without finetune, SCT severely degrades perplexity — use `--finetune-steps` for quality recovery.
- The Cache subclass approach (TurboQuantLayer subclassing DynamicLayer from transformers 5.x) works successfully.

**Environment:**
```bash
# Venv is at /scratch/DA24B039/sct_tq/.venv
# Python 3.13.12 from /opt/miniconda3/bin/python
# PyTorch 2.12.0+cpu, transformers 5.12.0
# run.sh sets LD_LIBRARY_PATH=/opt/miniconda3/lib (precautionary for CXXABI)
```

---

### SCT Per-Component LR Finetune (reusable function)

**File:** `experiments/sct_finetune.py`

Exposes `finetune_sct(model, tokenizer, *, sct_lr, lr_ratio, unfreeze, steps, ...)` — the
per-component learning rate recipe proven in `sct_per_component_lr.ipynb`. Two param groups:
- Group B (high lr = sct_lr): U, s, V factors of every SpectralLinear module
- Group A (low lr = sct_lr / lr_ratio): norm layers + (optionally) q/k/v/o_proj

Returns `{final_loss, final_ppl, steps, time_sec, sct_params, dense_params, loss_curve}`.

**Quick manual test:**
```bash
.venv/bin/python experiments/sct_finetune.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --energy 0.95 --sct-lr 5e-4 --lr-ratio 25 --steps 120
```

---

### SCT HP-Tuning Harness

**File:** `experiments/sct_hp_tune.py`

Grid-searches (energy, sct_lr, lr_ratio, unfreeze) for the best per-component LR recipe.
For each config: loads fresh model, applies SCT at given energy, calls `finetune_sct()`,
evaluates held-out perplexity on wikitext-2. Prints table sorted by ppl (best first).
Saves all results + best recipe to `sct_hp_tune_{model_tag}.json`.

**Default grid (small, intended for 135M):**
```bash
.venv/bin/python experiments/sct_hp_tune.py \
  --energies 0.95 \
  --sct-lrs 1e-3 5e-4 1e-4 \
  --lr-ratios 25 10 \
  --unfreeze norms+attn norms_only \
  --steps 300 --max-eval-tokens 512
```

**Smoke validation (single config, ~3 min on 135M CPU):**
```bash
.venv/bin/python experiments/sct_hp_tune.py \
  --energies 0.95 --sct-lrs 5e-4 --lr-ratios 25 \
  --unfreeze norms+attn --steps 120 --max-eval-tokens 256
# Result: post-SCT ppl 6318 -> post-finetune ppl 144 (44x improvement, 120 steps)
```

**Probe mode (timing only — no full finetune):**
```bash
.venv/bin/python experiments/sct_hp_tune.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --energies 0.95 --sct-lrs 5e-4 --lr-ratios 25 --unfreeze norms+attn \
  --probe
# 135M: 591 MB load, ~1.48s/step -> 300 steps in ~7.4 min
# Use with 1.7B to get step budget before committing to long run
```

**Outputs:**
- `experiments/sct_hp_tune_{model_tag}.json` — all grid results + best recipe
