"""
utility.py — aggregate the four raw metrics into a single utility score U in [0,1],
normalized against the DENSE fp16 baseline so the baseline scores ~1.0.

  s_ppl  = min(baseline_ppl / ppl, 1.0)      # perplexity: lower is better -> ratio
  s_hs   = acc_norm  / baseline_hs
  s_mmlu = acc       / baseline_mmlu
  s_tqa  = mc2       / baseline_tqa
  U      = w_ppl*s_ppl + w_hs*s_hs + w_mmlu*s_mmlu + w_tqa*s_tqa   (weights sum to 1)

Weights are configurable; defaults are equal (0.25 each). Raw + normalized + U are
all stored so nothing is hidden behind the scalar.
"""

from __future__ import annotations

DEFAULT_WEIGHTS = {"ppl": 0.25, "hellaswag": 0.25, "mmlu": 0.25, "truthfulqa": 0.25}


def normalized_scores(raw: dict, baseline: dict) -> dict:
    """Per-component scores normalized to the baseline (baseline -> ~1.0)."""
    eps = 1e-9
    s_ppl = min(baseline["perplexity"] / max(raw["perplexity"], eps), 1.0)
    s_hs = raw["hellaswag"] / max(baseline["hellaswag"], eps)
    s_mmlu = raw["mmlu"] / max(baseline["mmlu"], eps)
    s_tqa = raw["truthfulqa"] / max(baseline["truthfulqa"], eps)
    return {"ppl": s_ppl, "hellaswag": s_hs, "mmlu": s_mmlu, "truthfulqa": s_tqa}


def aggregate_utility(raw: dict, baseline: dict, weights: dict | None = None) -> dict:
    """Return {scores, utility} given raw metrics and the dense baseline."""
    w = weights or DEFAULT_WEIGHTS
    s = normalized_scores(raw, baseline)
    total_w = sum(w.values()) or 1.0
    U = (w["ppl"] * s["ppl"] + w["hellaswag"] * s["hellaswag"]
         + w["mmlu"] * s["mmlu"] + w["truthfulqa"] * s["truthfulqa"]) / total_w
    return {"scores": s, "utility": U}
