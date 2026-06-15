#!/usr/bin/env python3
"""
SCT x TurboQuant Joint Pareto Frontier Experiment
==================================================

Sweeps SCT (weight compression via spectral factorization) and TurboQuant
(KV cache compression) on SmolLM2-135M and traces the Pareto frontier over:
  - perplexity (accuracy, lower is better)
  - total compressed bytes (weight + KV, lower is better)
  - peak RSS (RAM, lower is better)
  - tokens/sec (throughput, higher is better)

KV COMPRESSION APPROACH:
We subclass transformers' DynamicLayer (the per-layer cache object inside
DynamicCache in transformers 5.x) to intercept update() calls, compress K/V
via TurboQuantProd/quantize_values, store compressed, and return dequantized
full-history K and V for attention. This is the "Cache subclass" approach.

If DynamicLayer subclassing breaks (e.g. transformers version mismatch), fall
back to a simple full-context forward that skips cache-level compression.

CAVEATS:
  - SmolLM2-135M has hidden_dim=576 < 2048 — per SCT docs, SCT compresses best
    at hidden>=2048. Expect little/no weight savings from SCT at this scale.
  - TurboQuant hybrid decode dequantizes history to float32; KV memory savings
    are real but compute/latency may actually INCREASE due to overhead.
  - GPU is unavailable (driver mismatch); all runs are CPU-only.
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import DynamicCache, DynamicLayer, CacheLayerMixin

from turboquant.quantizer import TurboQuantProd, ProdQuantized
from turboquant.kv_cache import quantize_values, dequantize_values, ValueQuantized

# ─────────────────────────────────────────────────────────────────────────────
#  SCT UTILITIES (adapted from SCT/examples/sct_vs_dense.py)
# ─────────────────────────────────────────────────────────────────────────────

MLP_LEAF_NAMES = frozenset([
    "gate_proj", "up_proj", "down_proj",   # LLaMA-style (SmolLM2 uses these)
    "fc_1", "fc_2",                         # GPT-NeoX
    "c_fc", "c_proj",                       # GPT-2
])


def safe_qr(M: torch.Tensor) -> torch.Tensor:
    dev = M.device
    Q, R = torch.linalg.qr(M.cpu() if dev.type == "mps" else M)
    return (Q * torch.sign(torch.diag(R))).to(dev)


class SpectralLinear(nn.Module):
    """Drop-in nn.Linear replacement storing W = U diag(s) V^T."""

    def __init__(self, U, s, V, bias=None):
        super().__init__()
        self.rank = s.shape[0]
        self.in_features = U.shape[0]
        self.out_features = V.shape[0]
        self.U = nn.Parameter(U)
        self.s = nn.Parameter(s)
        self.V = nn.Parameter(V)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        y = (x @ self.U) * self.s @ self.V.T
        return y + self.bias if self.bias is not None else y

    def param_count(self):
        n = self.U.numel() + self.V.numel() + self.s.numel()
        return n + (self.bias.numel() if self.bias is not None else 0)

    @classmethod
    def from_linear(cls, linear, rank=0, energy_threshold=0.95):
        W = linear.weight.data.float().cpu()
        m, n = W.shape  # [out, in]
        U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)
        if rank <= 0:
            total_energy = (S_full ** 2).sum()
            cumulative = torch.cumsum(S_full ** 2, dim=0) / total_energy
            k = int((cumulative >= energy_threshold).nonzero(as_tuple=True)[0][0].item()) + 1
            k = max(k, 1)
        else:
            k = min(rank, min(m, n))
        energy_retained = float((S_full[:k] ** 2).sum() / (S_full ** 2).sum())
        layer = cls(
            Vh_full[:k, :].T.contiguous(),   # U: [in, k]
            S_full[:k].contiguous(),           # s: [k]
            U_full[:, :k].contiguous(),        # V: [out, k]
            linear.bias.data.float() if linear.bias is not None else None,
        )
        layer._energy = energy_retained
        layer._dense_params = m * n + (m if linear.bias is not None else 0)
        return layer


def replace_mlp_with_spectral(model, energy, device="cpu"):
    """Replace MLP nn.Linear layers with SpectralLinear. Returns stats."""
    total_dense = 0
    total_spectral = 0
    n_replaced = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if leaf not in MLP_LEAF_NAMES:
            continue

        spec = SpectralLinear.from_linear(module, rank=0,
                                           energy_threshold=energy).to(device)
        parent_name, child_name = name.rsplit(".", 1)
        parent = dict(model.named_modules())[parent_name]
        setattr(parent, child_name, spec)

        total_dense += spec._dense_params
        total_spectral += spec.param_count()
        n_replaced += 1

    return n_replaced, total_dense, total_spectral


def compute_weight_bytes(model) -> int:
    """Compute total weight bytes, accounting for SpectralLinear compression."""
    total_params = 0
    for m in model.modules():
        if isinstance(m, SpectralLinear):
            total_params += m.param_count()
        elif isinstance(m, nn.Linear):
            total_params += m.weight.numel()
            if m.bias is not None:
                total_params += m.bias.numel()
        elif isinstance(m, nn.Embedding):
            total_params += m.weight.numel()
        elif isinstance(m, (nn.LayerNorm,)) or 'norm' in type(m).__name__.lower():
            for p in m.parameters(recurse=False):
                total_params += p.numel()
    # Fall back: count all unique parameters not already counted
    seen = set()
    for name, p in model.named_parameters():
        if id(p) not in seen:
            seen.add(id(p))
    # Simpler: just count all parameters (SpectralLinear stores compact params)
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4  # float32 = 4 bytes


# ─────────────────────────────────────────────────────────────────────────────
#  TURBOQUANT CACHE LAYER  (DynamicLayer subclass)
# ─────────────────────────────────────────────────────────────────────────────

class TurboQuantLayer(DynamicLayer):
    """
    DynamicLayer subclass that compresses keys via TurboQuantProd and values
    via group quantization. Stores compressed tensors; returns dequantized
    full-history K and V on each update() call.

    transformers 5.x uses DynamicLayer per layer inside DynamicCache.
    update() is called by DynamicCache.update() -> self.layers[layer_idx].update()
    and must return (full_keys, full_values) for attention.
    """

    def __init__(
        self,
        head_dim: int,
        key_bits: int,
        value_bits: int,
        value_group_size: int = 32,
        seed_offset: int = 0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.seed_offset = seed_offset
        # TurboQuantProd created lazily on first update (need device)
        self._key_quantizer: Optional[TurboQuantProd] = None
        # Compressed storage
        self._q_keys: Optional[ProdQuantized] = None
        self._q_values: Optional[ValueQuantized] = None
        # Byte tracking
        self.compressed_bytes: int = 0

    def _get_quantizer(self, device, dtype=torch.float32) -> TurboQuantProd:
        if self._key_quantizer is None:
            self._key_quantizer = TurboQuantProd(
                dim=self.head_dim,
                bits=self.key_bits,
                device=device,
                seed=42 + self.seed_offset * 7,
            )
        return self._key_quantizer

    def _count_bytes_prod(self, q: ProdQuantized) -> int:
        """Count compressed bytes for ProdQuantized (keys)."""
        b = 0
        b += q.mse_indices.nelement()       # uint8 packed
        b += q.qjl_signs.nelement()         # uint8 packed
        b += q.residual_norms.nelement() * 2  # stored as float16 equivalent
        b += q.norms.nelement() * 2           # float16 equivalent
        return b

    def _count_bytes_val(self, vq: ValueQuantized) -> int:
        """Count compressed bytes for ValueQuantized (values)."""
        b = 0
        b += vq.data.nelement()             # uint8 packed
        b += vq.scales.nelement() * 2       # float16 equivalent
        b += vq.zeros.nelement() * 2        # float16 equivalent
        return b

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress incoming K/V, store compressed, return full dequantized history.

        key_states shape:   (batch, n_kv_heads, seq_q, head_dim)
        value_states shape: (batch, n_kv_heads, seq_q, head_dim)
        """
        device = key_states.device
        orig_dtype = key_states.dtype

        # Work in float32 for quantization
        k = key_states.float()
        v = value_states.float()

        # --- Quantize keys ---
        quantizer = self._get_quantizer(device)
        new_key_q = quantizer.quantize(k)

        # --- Quantize values ---
        # quantize_values needs head_dim divisible by group_size
        new_val_q = quantize_values(v, bits=self.value_bits, group_size=self.value_group_size)

        # --- Append to compressed store ---
        if self._q_keys is None:
            self._q_keys = new_key_q
            self._q_values = new_val_q
        else:
            self._q_keys = ProdQuantized(
                mse_indices=torch.cat([self._q_keys.mse_indices, new_key_q.mse_indices], dim=-2),
                qjl_signs=torch.cat([self._q_keys.qjl_signs, new_key_q.qjl_signs], dim=-2),
                residual_norms=torch.cat([self._q_keys.residual_norms, new_key_q.residual_norms], dim=-1),
                norms=torch.cat([self._q_keys.norms, new_key_q.norms], dim=-1),
                mse_bits=new_key_q.mse_bits,
            )
            self._q_values = ValueQuantized(
                data=torch.cat([self._q_values.data, new_val_q.data], dim=-2),
                scales=torch.cat([self._q_values.scales, new_val_q.scales], dim=-2),
                zeros=torch.cat([self._q_values.zeros, new_val_q.zeros], dim=-2),
                bits=self.value_bits,
            )

        # Update byte count
        self.compressed_bytes = (
            self._count_bytes_prod(self._q_keys) +
            self._count_bytes_val(self._q_values)
        )

        # --- Dequantize full history for attention ---
        k_deq = quantizer.dequantize(self._q_keys).to(orig_dtype)
        v_deq = dequantize_values(self._q_values, group_size=self.value_group_size).to(orig_dtype)

        return k_deq, v_deq

    def get_seq_length(self) -> int:
        if self._q_keys is None:
            return 0
        # norms shape: (batch, n_heads, seq)
        return self._q_keys.norms.shape[-1]

    def get_max_cache_shape(self) -> int:
        return -1


class TurboQuantCache(DynamicCache):
    """
    DynamicCache that populates its layers with TurboQuantLayer instances
    instead of DynamicLayer instances.
    """

    def __init__(
        self,
        n_layers: int,
        head_dim: int,
        key_bits: int,
        value_bits: int,
        value_group_size: int = 32,
    ):
        # Init empty; don't pass config so layers stays empty initially
        super().__init__()
        self._n_layers = n_layers
        self._head_dim = head_dim
        self._key_bits = key_bits
        self._value_bits = value_bits
        self._value_group_size = value_group_size
        # Pre-populate layers list with TurboQuantLayer instances
        self.layers = [
            TurboQuantLayer(
                head_dim=head_dim,
                key_bits=key_bits,
                value_bits=value_bits,
                value_group_size=value_group_size,
                seed_offset=i,
            )
            for i in range(n_layers)
        ]
        # Disable auto-grow behavior
        self.layer_class_to_replicate = None

    def total_compressed_bytes(self) -> int:
        return sum(layer.compressed_bytes for layer in self.layers
                   if isinstance(layer, TurboQuantLayer))


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_TEXT = """
The transformer architecture has become the dominant approach in natural language
processing. It relies on self-attention mechanisms that allow each position in a
sequence to attend to all other positions, enabling the model to capture long-range
dependencies. Modern large language models built on this architecture, such as GPT
and LLaMA families, have demonstrated remarkable capabilities across diverse tasks
including text generation, summarization, question answering, and code synthesis.
The key insight of the attention mechanism is that representations can be computed
as weighted sums of value vectors, where the weights are determined by the
compatibility of query and key vectors. This allows the model to selectively focus
on relevant parts of the input when computing each output representation.
Scaling these models to billions of parameters has revealed emergent capabilities
that are not present in smaller models. Researchers have found that beyond certain
thresholds, models exhibit qualitatively new behaviors such as few-shot learning,
chain-of-thought reasoning, and instruction following. These capabilities arise
from the interplay between the model's capacity to store knowledge in its weights
and its ability to perform complex computations through its attention layers.
The efficiency of transformer inference is often limited by memory bandwidth rather
than compute. The key-value cache, which stores attention states for previously
processed tokens, grows linearly with sequence length and can become a significant
bottleneck. Techniques such as grouped query attention, speculative decoding, and
KV cache compression have been developed to address this challenge.
"""


def load_eval_text(max_tokens: int, tokenizer) -> torch.Tensor:
    """Load wikitext-2 or fall back to hardcoded text. Returns token ids (1D)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # Concatenate first several articles
        text = " ".join(row["text"] for row in ds if row["text"].strip())[:8000]
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        if len(ids) >= max_tokens:
            return ids[:max_tokens]
        print(f"  [warn] wikitext gave only {len(ids)} tokens, using fallback text")
    except Exception as e:
        print(f"  [warn] wikitext load failed ({e}), using fallback text")

    ids = tokenizer(FALLBACK_TEXT.strip(), return_tensors="pt")["input_ids"][0]
    # Repeat if needed
    while len(ids) < max_tokens:
        ids = torch.cat([ids, ids])
    return ids[:max_tokens]


def compute_perplexity_with_cache(
    model,
    input_ids: torch.Tensor,
    cache: Optional[DynamicCache],
    device: str = "cpu",
) -> float:
    """
    Compute per-token cross-entropy perplexity using teacher forcing.
    Processes the full sequence in one forward pass (prefill style).
    If cache is provided, it will be populated with KV states.
    Returns float perplexity.
    """
    model.eval()
    ids = input_ids.unsqueeze(0).to(device)  # (1, T)

    with torch.no_grad():
        if cache is not None:
            out = model(
                input_ids=ids,
                past_key_values=cache,
                use_cache=True,
                attn_implementation="eager",
            )
        else:
            out = model(input_ids=ids, use_cache=False)

    logits = out.logits  # (1, T, vocab)
    # Shift: predict token t+1 from token t
    shift_logits = logits[0, :-1, :].float()  # (T-1, vocab)
    shift_labels = ids[0, 1:].long()           # (T-1,)
    loss = F.cross_entropy(shift_logits, shift_labels)
    return math.exp(min(loss.item(), 20.0))


def compute_latency_tokens_per_sec(
    model,
    input_ids: torch.Tensor,
    cache: Optional[DynamicCache],
    device: str = "cpu",
    n_timing_tokens: int = 50,
) -> float:
    """
    Measure throughput as tokens/sec over a prefill+decode style run.
    We run n_timing_tokens through and time it.
    """
    model.eval()
    ids = input_ids[:n_timing_tokens].unsqueeze(0).to(device)

    # Warm-up (1 step, not timed)
    with torch.no_grad():
        if cache is not None:
            # For timing, create a fresh cache of same type
            timing_cache = type(cache)(
                n_layers=cache._n_layers if hasattr(cache, '_n_layers') else 30,
                head_dim=cache._head_dim if hasattr(cache, '_head_dim') else 64,
                key_bits=cache._key_bits if hasattr(cache, '_key_bits') else 3,
                value_bits=cache._value_bits if hasattr(cache, '_value_bits') else 2,
            ) if isinstance(cache, TurboQuantCache) else DynamicCache()
        else:
            timing_cache = None

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(
            input_ids=ids,
            past_key_values=timing_cache if cache is not None else None,
            use_cache=(cache is not None),
        )
    t1 = time.perf_counter()

    elapsed = t1 - t0
    if elapsed < 1e-9:
        elapsed = 1e-9
    return n_timing_tokens / elapsed


def get_fp16_kv_bytes(n_tokens: int, n_layers: int, n_kv_heads: int, head_dim: int) -> int:
    """Estimated fp16 KV cache bytes for n_tokens."""
    # 2 (K and V) * n_layers * n_kv_heads * head_dim * n_tokens * 2 bytes (float16)
    return 2 * n_layers * n_kv_heads * head_dim * n_tokens * 2


def run_config(
    model,
    input_ids: torch.Tensor,
    energy: Optional[float],
    tq_key_bits: Optional[int],
    tq_val_bits: Optional[int],
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run a single (energy, tq) config and return metrics dict.
    """
    n_tokens = len(input_ids)
    proc = psutil.Process(os.getpid())

    # --- Set up cache ---
    if tq_key_bits is not None:
        try:
            cache = TurboQuantCache(
                n_layers=n_layers,
                head_dim=head_dim,
                key_bits=tq_key_bits,
                value_bits=tq_val_bits,
                value_group_size=32,
            )
            use_tq_cache = True
        except Exception as e:
            print(f"  [warn] TurboQuantCache construction failed ({e}), using plain cache")
            cache = DynamicCache()
            use_tq_cache = False
    else:
        cache = None  # No cache (run without use_cache)
        use_tq_cache = False

    # --- Peak RSS around eval ---
    rss_before = proc.memory_info().rss

    # --- Perplexity ---
    t_eval_start = time.perf_counter()
    try:
        ppl = compute_perplexity_with_cache(model, input_ids, cache, device)
    except Exception as e:
        print(f"  [ERROR] perplexity eval failed: {e}")
        traceback.print_exc()
        ppl = float("nan")
    t_eval_end = time.perf_counter()

    rss_after = proc.memory_info().rss
    peak_rss = max(rss_before, rss_after)

    # --- KV bytes ---
    if use_tq_cache and isinstance(cache, TurboQuantCache):
        peak_kv_bytes = cache.total_compressed_bytes()
    elif tq_key_bits is None:
        peak_kv_bytes = get_fp16_kv_bytes(n_tokens, n_layers, n_kv_heads, head_dim)
    else:
        # plain DynamicCache fallback - estimate fp16
        peak_kv_bytes = get_fp16_kv_bytes(n_tokens, n_layers, n_kv_heads, head_dim)

    # --- Weight bytes ---
    weight_bytes = compute_weight_bytes(model)

    # --- Latency ---
    try:
        tok_per_sec = compute_latency_tokens_per_sec(
            model, input_ids, cache, device, n_timing_tokens=min(50, n_tokens)
        )
    except Exception as e:
        print(f"  [warn] latency measurement failed ({e})")
        tok_per_sec = float("nan")

    return {
        "energy": energy,
        "tq_key_bits": tq_key_bits,
        "tq_val_bits": tq_val_bits,
        "perplexity": round(ppl, 4),
        "weight_bytes": weight_bytes,
        "kv_bytes": peak_kv_bytes,
        "total_bytes": weight_bytes + peak_kv_bytes,
        "peak_rss": peak_rss,
        "tok_per_sec": round(tok_per_sec, 2),
        "use_tq_cache": use_tq_cache,
        "eval_tokens": n_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PARETO FRONTIER
# ─────────────────────────────────────────────────────────────────────────────

def is_pareto_dominated(point: Dict, others: List[Dict]) -> bool:
    """Return True if point is dominated by any point in others."""
    for other in others:
        if other is point:
            continue
        # Other dominates point if it is better-or-equal on ALL axes and strictly better on one
        better_or_equal = (
            other["perplexity"] <= point["perplexity"] and
            other["total_bytes"] <= point["total_bytes"] and
            other["peak_rss"] <= point["peak_rss"] and
            other["tok_per_sec"] >= point["tok_per_sec"]
        )
        strictly_better = (
            other["perplexity"] < point["perplexity"] or
            other["total_bytes"] < point["total_bytes"] or
            other["peak_rss"] < point["peak_rss"] or
            other["tok_per_sec"] > point["tok_per_sec"]
        )
        # Skip NaN comparisons
        try:
            if better_or_equal and strictly_better:
                return True
        except Exception:
            pass
    return False


def find_pareto_frontier(results: List[Dict]) -> List[Dict]:
    """Return list of non-dominated points."""
    # Filter out NaN entries
    valid = [r for r in results if not any(
        math.isnan(v) for v in [r["perplexity"], r["total_bytes"], r["peak_rss"], r["tok_per_sec"]]
    )]
    frontier = [r for r in valid if not is_pareto_dominated(r, valid)]
    return frontier


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="SCT x TurboQuant Pareto sweep on SmolLM2-135M")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--max-tokens", type=int, default=768,
                   help="Number of eval tokens (default 768)")
    p.add_argument("--energies", type=float, nargs="+",
                   default=None,
                   help="SCT energies to sweep (default: None 0.99 0.95 0.90 0.80)")
    p.add_argument("--finetune-steps", type=int, default=0,
                   help="SCT finetune steps per energy (0 = no finetune)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: only dense + one TQ config, for smoke testing")
    args = p.parse_args()

    device = args.device

    # Define sweep configs
    if args.energies is not None:
        energy_sweep = [None] + args.energies
    else:
        energy_sweep = [None, 0.99, 0.95, 0.90, 0.80]

    tq_sweep = [
        (None, None),       # fp16 baseline
        (4, 4),             # 4-bit keys, 4-bit values
        (3, 4),             # 3-bit keys, 4-bit values
        (3, 2),             # 3-bit keys, 2-bit values
        (2, 2),             # 2-bit keys, 2-bit values
    ]

    if args.quick:
        energy_sweep = [None, 0.95]
        tq_sweep = [(None, None), (3, 4)]

    print("=" * 72)
    print("  SCT x TurboQuant Pareto Frontier Experiment")
    print("=" * 72)
    print(f"  Model:       {args.model}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Energies:    {energy_sweep}")
    print(f"  TQ configs:  {tq_sweep}")
    print(f"  Device:      {device}")
    print()

    # Load tokenizer + eval text once
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading eval text ({args.max_tokens} tokens)...")
    input_ids = load_eval_text(args.max_tokens, tokenizer)
    actual_tokens = len(input_ids)
    print(f"  Got {actual_tokens} eval tokens")

    # Load model config for metadata
    cfg = AutoConfig.from_pretrained(args.model)
    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(f"  Model: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")
    print()

    all_results = []

    for energy in energy_sweep:
        energy_label = f"E={energy}" if energy is not None else "dense"
        print(f"\n{'─'*72}")
        print(f"  SCT energy: {energy_label}")
        print(f"{'─'*72}")

        # Load fresh model for this energy
        print(f"  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32
        ).to(device)
        model.eval()

        # Apply SCT if energy is specified
        if energy is not None:
            print(f"  Applying SCT (energy={energy})...")
            n_replaced, dense_params, spectral_params = replace_mlp_with_spectral(
                model, energy=energy, device=device
            )
            ratio = dense_params / max(spectral_params, 1)
            print(f"  Replaced {n_replaced} MLP layers, "
                  f"MLP params: {dense_params:,} -> {spectral_params:,} ({ratio:.2f}x)")

        # Finetune if requested (optional)
        if args.finetune_steps > 0 and energy is not None:
            print(f"  [finetune skipped in smoke test; --finetune-steps={args.finetune_steps} "
                  f"requested but not implemented in quick run]")

        for tq_key_bits, tq_val_bits in tq_sweep:
            tq_label = (f"TQ(k={tq_key_bits},v={tq_val_bits})"
                        if tq_key_bits is not None else "fp16-KV")
            config_label = f"{energy_label} + {tq_label}"
            print(f"  Running: {config_label} ...", end=" ", flush=True)

            result = run_config(
                model=model,
                input_ids=input_ids,
                energy=energy,
                tq_key_bits=tq_key_bits,
                tq_val_bits=tq_val_bits,
                n_layers=n_layers,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                device=device,
            )
            result["config_label"] = config_label

            ppl_str = f"{result['perplexity']:.1f}" if not math.isnan(result['perplexity']) else "NaN"
            kb_w = result['weight_bytes'] / 1024
            kb_kv = result['kv_bytes'] / 1024
            print(f"ppl={ppl_str}, "
                  f"weight={kb_w:.0f}KB, kv={kb_kv:.0f}KB, "
                  f"tok/s={result['tok_per_sec']:.1f}, "
                  f"TQ={'OK' if result['use_tq_cache'] else 'NO'}")

            all_results.append(result)

        del model
        torch.cuda.empty_cache() if device != "cpu" else None

    # Save results
    out_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(out_dir, "sct_tq_pareto_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "model": args.model,
                "max_tokens": args.max_tokens,
                "actual_tokens": actual_tokens,
                "n_layers": n_layers,
                "n_kv_heads": n_kv_heads,
                "head_dim": head_dim,
                "device": device,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Saved results -> {results_path}")

    # Compute Pareto frontier
    frontier = find_pareto_frontier(all_results)
    frontier_labels = {r["config_label"] for r in frontier}

    # Print table
    print()
    print("=" * 72)
    print("  RESULTS TABLE")
    print("=" * 72)
    print(f"  {'Config':<30s} {'PPL':>7s} {'WtMB':>7s} {'KVKB':>7s} {'TotMB':>8s} "
          f"{'RSS MB':>8s} {'tok/s':>7s} {'Pareto':>7s}")
    print(f"  {'─'*30} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*7}")
    for r in all_results:
        ppl_str = f"{r['perplexity']:.2f}" if not math.isnan(r['perplexity']) else "  NaN"
        wt_mb = r['weight_bytes'] / 1e6
        kv_kb = r['kv_bytes'] / 1e3
        tot_mb = r['total_bytes'] / 1e6
        rss_mb = r['peak_rss'] / 1e6
        tok_s = r['tok_per_sec']
        is_front = "*" if r["config_label"] in frontier_labels else ""
        print(f"  {r['config_label']:<30s} {ppl_str:>7s} {wt_mb:>7.1f} {kv_kb:>7.1f} "
              f"{tot_mb:>8.2f} {rss_mb:>8.1f} {tok_s:>7.1f} {is_front:>7s}")

    print()
    print(f"  Pareto frontier: {len(frontier)} points (marked *)")
    print(f"  Note: SmolLM2-135M has hidden_dim=576 < 2048 — SCT weight compression")
    print(f"        is minimal at this scale (documented in CLAUDE.md).")
    print(f"        TurboQuant KV bytes ARE reduced but decode latency may increase")
    print(f"        due to dequant overhead on CPU (no Triton kernels).")

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(10, 7))
        for r in all_results:
            if math.isnan(r["perplexity"]) or math.isnan(r["total_bytes"]):
                continue
            is_front = r["config_label"] in frontier_labels
            color = "red" if is_front else "steelblue"
            marker = "*" if is_front else "o"
            size = 150 if is_front else 60
            ax.scatter(r["total_bytes"] / 1e6, r["perplexity"],
                       c=color, marker=marker, s=size, zorder=5 if is_front else 3)
            ax.annotate(r["config_label"], (r["total_bytes"] / 1e6, r["perplexity"]),
                        fontsize=6, ha="left", va="bottom")

        # Connect frontier points
        if frontier:
            frontier_sorted = sorted(frontier, key=lambda r: r["total_bytes"])
            fx = [r["total_bytes"] / 1e6 for r in frontier_sorted]
            fy = [r["perplexity"] for r in frontier_sorted]
            ax.plot(fx, fy, "r--", linewidth=1, alpha=0.6, label="Pareto frontier")

        ax.set_xlabel("Total compressed size (MB: weights + KV cache)")
        ax.set_ylabel("Perplexity (lower is better)")
        ax.set_title(f"SCT x TurboQuant Pareto Frontier\n{args.model} — {actual_tokens} tokens")
        patch_front = mpatches.Patch(color="red", label="Pareto-optimal")
        patch_dom = mpatches.Patch(color="steelblue", label="Dominated")
        ax.legend(handles=[patch_front, patch_dom])
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(out_dir, "sct_tq_pareto.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\n  Saved plot -> {plot_path}")
    except Exception as e:
        print(f"\n  [warn] matplotlib plot failed: {e}")

    print("\nDone.")
    return all_results


if __name__ == "__main__":
    main()
