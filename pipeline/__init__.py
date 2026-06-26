"""
SCT x TurboQuant Utility-vs-Compression Pareto pipeline (Llama-3.1-8B, A100).

Modules
-------
sct_apply    : apply Spectral Compact Training (weight-side compression) to Llama MLP
tq_cache     : TurboQuant KV-cache (inference-side quantization) as a transformers Cache
recovery     : recovery-LoRA (peft) to recover utility lost to compression
eval_tasks   : perplexity + HellaSwag + MMLU + TruthfulQA, forced through the TQ cache
utility      : normalize the four metrics vs a dense baseline -> aggregate utility U
compression  : weight + KV byte accounting -> compression ratio
pareto       : Pareto frontier + scatter / heatmap plots

The orchestrator is ../run_pareto.py.
"""

__all__ = [
    "sct_apply",
    "tq_cache",
    "recovery",
    "eval_tasks",
    "utility",
    "compression",
    "pareto",
]
