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
#     ./run.sh full      # the full energy x KV x LoRA grid
#     ./run.sh <args...> # passed straight to run_pareto.py
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs results_llama8b

# Load secrets (.env is gitignored). HF_TOKEN is required for gated Llama-3.1-8B.
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is empty. Add it to .env (request access to" >&2
  echo "       meta-llama/Llama-3.1-8B on HuggingFace first)." >&2
  exit 1
fi
export HF_TOKEN
huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true

MODE="${1:-dry}"; shift || true
LOG="logs/pareto_${MODE}_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" logs/latest.log

case "$MODE" in
  dry)   CMD=(python -u run_pareto.py --dry-run "$@") ;;
  quick) CMD=(python -u run_pareto.py --quick "$@") ;;
  full)  CMD=(python -u run_pareto.py
            --energies 0.99 0.97 0.95 0.90 0.85
            --kv-configs none,4x4,3x4,3x2,2x2
            --lora both "$@") ;;
  *)     CMD=(python -u run_pareto.py "$MODE" "$@") ;;
esac

echo "=== $(date -u +%FT%TZ) | mode=$MODE | log=$LOG ===" | tee "$LOG"
echo "=== cmd: ${CMD[*]} ===" | tee -a "$LOG"
# Capture everything (stdout+stderr) into the log AND the console.
"${CMD[@]}" 2>&1 | tee -a "$LOG"
status="${PIPESTATUS[0]}"
echo "=== exit $status | $(date -u +%FT%TZ) ===" | tee -a "$LOG"
exit "$status"
