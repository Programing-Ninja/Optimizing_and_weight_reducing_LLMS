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
