"""
compression.py — byte accounting for the x-axis (compression ratio).

weight_bytes : sum over all model parameters of numel * dtype_size. After SCT the MLP
               SpectralLinear factors (U, s, V) replace dense W, so this automatically
               reflects weight compression. Recovery-LoRA params, if merged into the
               model, are counted here too; if kept as separate adapters they are added
               via `extra_param_bytes`.
kv_bytes     : peak TurboQuant compressed KV bytes for the eval workload, or the fp16
               baseline 2*L*Hkv*hd*T*2 when KV is not quantized.
ratio        : dense_fp16_total_bytes / total_bytes  (higher = more compressed).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def model_weight_bytes(model: nn.Module) -> int:
    """Total parameter bytes at each parameter's stored dtype."""
    total = 0
    seen = set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel() * p.element_size()
    return total


def dense_fp16_weight_bytes(model: nn.Module) -> int:
    """What the same parameter *shapes* would cost dense in fp16 — but SpectralLinear
    factors are compressed, so we reconstruct the dense-equivalent MLP cost.

    For the baseline we instead call this on the unmodified dense model; for SCT models
    use `dense_equivalent_weight_bytes` below.
    """
    return sum(p.numel() for p in model.parameters()) * 2


def total_bytes(weight_bytes: int, kv_bytes: int) -> int:
    return weight_bytes + kv_bytes


def compression_ratio(baseline_total_bytes: int, this_total_bytes: int) -> float:
    return baseline_total_bytes / max(this_total_bytes, 1)
