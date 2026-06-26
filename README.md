# Optimizing_and_weight_reducing_LLMS

Here I try to find ways to get better weight and KV cache optimization, starting off with finding the optimal frontier by combining SCT and TurboQuant.

## Sub-projects

- `SCT/` — Spectral Compact Training: replaces nn.Linear with U diag(s) V^T factors; gradients flow through spectral factors with Stiefel retraction.
- `turboquant/` — TurboQuant KV cache compression: random rotation + Lloyd-Max quantization for keys + group quantization for values.

## Experiments

### Llama-3.1-8B Utility-vs-Compression Pareto Pipeline (A100)

**Files:** `run_pareto.py` + the `pipeline/` package · **Launcher:** `run.sh`

The scaled-up pipeline. Where the SmolLM2-135M experiment used perplexity only and
hid SCT's weight savings (hidden=576 < 2048), this runs **base Llama-3.1-8B**
(hidden=4096, where SCT actually compresses) and replaces "accuracy" with a real
**utility metric**.

**Utility metric** (`pipeline/utility.py`) — four benchmarks, each normalized to the
dense fp16 baseline (baseline ≈ 1.0), weighted (equal by default):

| Component | Benchmark | Direction |
|-----------|-----------|-----------|
| `s_ppl`   | wikitext-2 perplexity | lower better → `baseline_ppl/ppl` |
| `s_hs`    | HellaSwag acc_norm    | higher better |
| `s_mmlu`  | MMLU 4-way acc        | higher better |
| `s_tqa`   | TruthfulQA MC2        | higher better |

`U = Σ wᵢ·sᵢ ∈ [0,1]`. All four are run on the **same** compressed+quantized model,
forced through the TurboQuant cache (`use_cache=True`, eager attention) so KV
quantization genuinely affects the loglikelihood scores — a plain forward would run
`use_cache=False` and make quantization look free.

**Sweep axes:** SCT energy `{dense, 0.99, 0.97, 0.95, 0.90, 0.85}` × KV bits
`{fp16, 4×4, 3×4, 3×2, 2×2}` × recovery-LoRA `{off, on}` (toggle).

**Recovery-LoRA** (`pipeline/recovery.py`) — `peft` LoRA on the attention projections
(`q/k/v/o_proj`, which stay `nn.Linear` after SCT compresses only the MLP), trained on
an alpaca slice then `merge_and_unload`-ed → **zero extra storage**. The off/on toggle
exposes the recovery gain.

**Outputs** (`results_llama8b/`): `pareto_results.json`,
`pareto_utility_vs_compression.png` (frontier), and
`heatmap_energy_kv_{nolora,lora}.png` — the heatmap answers the two-way question (how
weight-compression energy shifts the tolerable quantization level, and vice-versa).

**Run:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e SCT && pip install -e turboquant
cp .env.example .env          # paste HF_TOKEN (request gated Llama-3.1-8B access first)

./run.sh quick                # smoke test FIRST (tiny subsets, 1-2 points)
./run.sh full                 # full energy x KV x LoRA grid on the A100
```

Subset sizes, LoRA steps, and utility weights are all CLI flags
(`python run_pareto.py --help`).

---

### SCT x TurboQuant Joint Pareto Frontier (SmolLM2-135M, CPU — earlier work)

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

**Plot the results:**
```bash
experiments/run.sh experiments/plot_hp_tune.py
# -> experiments/sct_hp_tune_SmolLM2-135M.png
```

#### Results — SmolLM2-135M sweep (8 configs, 300 steps each)

![SCT HP-tuning sweep](experiments/sct_hp_tune_SmolLM2-135M.png)

| energy | sct_lr | unfreeze    | params | post-FT ppl |
|:------:|:------:|:------------|:------:|:-----------:|
| 0.95   | 5e-4   | norms+attn  | 0.94x  | **67.7**    |
| 0.90   | 5e-4   | norms_only  | 1.11x  | 78.9        |
| 0.90   | 5e-4   | norms+attn  | 1.11x  | 79.5        |
| 0.95   | 5e-4   | norms_only  | 0.94x  | 160.2       |
| 0.95   | 1e-3   | norms+attn  | 0.94x  | 166.1       |
| 0.95   | 1e-3   | norms_only  | 0.94x  | 170.2       |
| 0.90   | 1e-3   | norms+attn  | 1.11x  | 692.8       |
| 0.90   | 1e-3   | norms_only  | 1.11x  | 697.9       |

**Observations:**
- **Best recipe: energy 0.95, sct_lr 5e-4, lr_ratio 25, unfreeze norms+attn → ppl 67.7.**
  This is the only sub-70 config and beats the next best by ~15%.
- **sct_lr is the dominant knob.** Dropping sct_lr from 1e-3 to 5e-4 helps everywhere;
  at energy 0.90 it is the difference between recovery (≈79 ppl) and collapse (≈690 ppl).
  1e-3 is too aggressive for the spectral factors here.
- **Higher energy (0.95) needs the right LR.** With sct_lr 5e-4 it wins outright (67.7),
  but with 1e-3 it lands at ~166–170 — still far better than energy-0.90/1e-3's ~690s.
- **Unfreezing attention helps only at low LR / high energy.** norms+attn is best in the
  winning row, but at energy 0.90 it gives no real gain over norms_only, and pairing it
  with sct_lr 1e-3 produces the worst result.
- **Compression caveat persists at 135M.** energy 0.95 keeps params *below* dense (0.94x)
  while energy 0.90 actually *inflates* them to 1.11x — at hidden_dim=576 the spectral
  factors are large relative to the dense weights (see CLAUDE.md), so the best-quality
  config also happens to be the only compressive one here. Real weight savings need ≥1.7B.
- Even the best post-finetune ppl (67.7) is well above dense (1.81); 300 steps on 500
  samples is a recovery probe, not a full retrain. The recipe ranking is the takeaway.
