# HPC Setup — SCT × TurboQuant Utility-vs-Compression Pareto (Llama-3.1-8B)

First-time setup on a fresh HPC node with one **NVIDIA A100 40GB**. Run the blocks in
order. Lines in `<ANGLE_BRACKETS>` are placeholders — adjust to your cluster.

Every `./run.sh` invocation tees a clean log to `logs/latest.log` (plus a timestamped
copy). If anything breaks, that file is the thing to paste back for debugging.

---

## 0. Get the code onto the cluster
`SCT/` and `turboquant/` are **git submodules** — clone with `--recurse-submodules`
or they'll be empty and the `pip install -e` step will fail.
```bash
git clone --recurse-submodules \
    https://github.com/Programing-Ninja/Optimizing_and_weight_reducing_LLMS.git
cd Optimizing_and_weight_reducing_LLMS

# If you cloned WITHOUT --recurse-submodules, pull them now:
git submodule update --init --recursive
```

## 1. Grab a GPU + load toolchain modules
```bash
# --- interactive A100 session (SLURM example; adjust partition/account) ---
srun --partition=<GPU_PARTITION> --gres=gpu:a100:1 --cpus-per-task=8 \
     --mem=64G --time=08:00:00 --pty bash

# --- modules (names vary by cluster; `module avail` to discover) ---
module purge
module load <cuda/12.x>
module load <python/3.11>      # or use a conda base
nvidia-smi                     # confirm an A100 is visible
```

## 2. Put HF + pip caches on SCRATCH (home quotas are small; the 8B model is ~16GB)
```bash
export SCRATCH="${SCRATCH:-$HOME/scratch}"          # adjust to your scratch path
export HF_HOME="$SCRATCH/hf_cache"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"
# (optional) persist these by appending the 3 exports to ~/.bashrc
```

## 3. Create the virtual environment (first time only)
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
```

## 4. Install PyTorch (CUDA build matching the module loaded in step 1)
```bash
# Pick the wheel matching your CUDA. For CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('cuda ok:', torch.cuda.is_available(), torch.version.cuda)"
```

## 5. Install pipeline deps + the two vendored packages (editable)
```bash
pip install -r requirements.txt
pip install -e SCT
pip install -e turboquant
```

## 6. Secrets — Hugging Face token for the gated model
```bash
# Llama-3.1-8B is gated: accept the license at
#   https://huggingface.co/meta-llama/Llama-3.1-8B   (once, in a browser)
cp -n .env.example .env        # if .env doesn't exist yet
# edit .env and paste your token into HF_TOKEN=...   (READ-scope token)
nano .env
chmod 600 .env
```

## 7. Smoke tests — ALWAYS in this order
```bash
# (a) integration only: model load + SCT + TQ-cache forward, NO datasets/training.
#     Catches transformers/cache/OOM issues without touching the network.
./run.sh dry

# (b) tiny end-to-end: tiny subsets, 1-2 grid points, downloads the eval datasets.
./run.sh quick
```
Both write to `logs/latest.log`. Only proceed if `./run.sh dry` prints
`DRY RUN PASSED ✓`.

## 8. The real run
```bash
./run.sh full
# outputs -> results_llama8b/{pareto_results.json,
#                              pareto_utility_vs_compression.png,
#                              heatmap_energy_kv_nolora.png,
#                              heatmap_energy_kv_lora.png}
```

### Optional: run `full` as a batch job (so it survives logout)
```bash
cat > pareto.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=sct-tq-pareto
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
module purge; module load <cuda/12.x> <python/3.11>
source .venv/bin/activate
export HF_HOME="$SCRATCH/hf_cache"
./run.sh full
EOF
sbatch pareto.sbatch
```

---

## Knobs (see `python run_pareto.py --help`)
| Flag | Default | Purpose |
|------|---------|---------|
| `--energies` | `0.99 0.97 0.95 0.90 0.85` | SCT energy levels (dense always added) |
| `--kv-configs` | `none,4x4,3x4,3x2,2x2` | TurboQuant (key×value) bits per config |
| `--lora` | `both` | recovery-LoRA: `off` / `on` / `both` (toggle) |
| `--hellaswag --mmlu --truthfulqa` | `400 400 200` | per-benchmark subset sizes |
| `--ppl-tokens` | `2048` | wikitext-2 perplexity token budget |
| `--lora-steps` | `200` | recovery-LoRA training steps per energy |
| `--weights` | equal | `ppl,hellaswag,mmlu,truthfulqa` utility weights |

## If a run fails
1. Open `logs/latest.log` — it has the exact command and full stderr.
2. Re-run `./run.sh dry` to confirm whether it's a model/cache issue (dry fails) or a
   dataset/training issue (dry passes, quick fails).
3. OOM on 40GB → lower `--ppl-tokens`, eval subsets, or `--lora-steps`; ensure only
   one run is using the GPU (`nvidia-smi`).
