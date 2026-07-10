#!/usr/bin/env bash
# ============================================================================
#  run.sh — launch the SCT x TurboQuant utility-vs-compression Pareto sweep.
#  Target: single NVIDIA A100 40GB.  Run from the repo root.
#
#  Every run is tee'd to a CLEAN, timestamped log under logs/ AND to
#  logs/latest.log — paste that file here if anything breaks.
#
#  One-time setup: see HPC_SETUP.md (fresh-environment command list).
#
#  Usage:
#     ./run.sh dry       # integration smoke test (no datasets/training) — DO THIS FIRST
#     ./run.sh quick     # tiny end-to-end smoke (small subsets, 1-2 points)
#     ./run.sh full      # the full energy x KV x LoRA grid (8B)
#     ./run.sh dry70b    # 70B integration smoke on GPU 1 (GPU 0 is occupied)
#     ./run.sh quick70b  # 70B tiny end-to-end smoke
#     ./run.sh full70b   # 70B full grid (smaller eval subsets; expect days)
#     ./run.sh theory    # theory-validation stage on the newest sweep results
#     ./run.sh <args...> # passed straight to run_pareto.py
#
#  GPU selection: the 70b modes default to GPU 1 (the free A100 80GB). When both
#  GPUs are free, append `--gpus 0,1` to split the dense model across them.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs

# 70B knobs shared by the three 70b modes. --max-gpu-mem 72 leaves headroom for
# eager-attention activations + LoRA optimizer state on the 80GB card.
SEVENTY_B=(--model meta-llama/Llama-3.1-70B --gpus 1 --big-model on
           --max-gpu-mem 72 --lora-batch 1 --lora-grad-checkpoint)

# Load secrets (.env is gitignored). HF_TOKEN is required for the gated Llama
# models (accept the 8B AND 70B licenses on HuggingFace — they are separate).
if [ -f .env ]; then set -a; source .env; set +a; fi

MODE="${1:-dry}"; shift || true

if [ "$MODE" != "theory" ]; then   # theory stage is offline — no token needed
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is empty. Add it to .env (request access to" >&2
    echo "       meta-llama/Llama-3.1-8B / -70B on HuggingFace first)." >&2
    exit 1
  fi
  export HF_TOKEN
  huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true
fi
LOG="logs/pareto_${MODE}_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" logs/latest.log

case "$MODE" in
  dry)   CMD=(python -u run_pareto.py --dry-run "$@") ;;
  quick) CMD=(python -u run_pareto.py --quick "$@") ;;
  full)  CMD=(python -u run_pareto.py
            --energies 0.99 0.97 0.95 0.90 0.85
            --kv-configs none,4x4,3x4,3x2,2x2
            --lora both "$@") ;;
  dry70b)   CMD=(python -u run_pareto.py "${SEVENTY_B[@]}" --dry-run "$@") ;;
  quick70b) CMD=(python -u run_pareto.py "${SEVENTY_B[@]}" --quick "$@") ;;
  # full70b: same grid, smaller eval subsets + shorter LoRA — the dense-baseline
  # points run partially CPU-offloaded and dominate wall time.
  full70b)  CMD=(python -u run_pareto.py "${SEVENTY_B[@]}"
            --energies 0.99 0.97 0.95 0.90 0.85
            --kv-configs none,4x4,3x4,3x2,2x2
            --lora both
            --ppl-tokens 1024 --hellaswag 200 --mmlu 200 --truthfulqa 100
            --lora-steps 100 "$@") ;;
  theory)   CMD=(python -u run_theory_validation.py "$@") ;;
  *)     CMD=(python -u run_pareto.py "$MODE" "$@") ;;
esac

echo "=== $(date -u +%FT%TZ) | mode=$MODE | log=$LOG ===" | tee "$LOG"
echo "=== cmd: ${CMD[*]} ===" | tee -a "$LOG"
# Capture everything (stdout+stderr) into the log AND the console.
"${CMD[@]}" 2>&1 | tee -a "$LOG"
status="${PIPESTATUS[0]}"
echo "=== exit $status | $(date -u +%FT%TZ) ===" | tee -a "$LOG"
exit "$status"
