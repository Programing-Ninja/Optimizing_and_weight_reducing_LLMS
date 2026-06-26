"""
tq_cache.py — TurboQuant KV cache as a transformers Cache (DynamicLayer subclass).

WHY a Cache subclass: the four utility benchmarks are loglikelihood tasks (a single
forward, no autoregressive decode). To make TurboQuant's quantization actually affect
quality we must route attention through compressed->dequantized K/V *during the
forward*. transformers calls cache.layers[i].update(k, v) inside attention; by
subclassing DynamicLayer we intercept that call, quantize keys (TurboQuantProd) and
values (group quant), store the compressed tensors (for byte accounting), and return
the dequantized full history so attention sees the lossy K/V. Run the model with
attn_implementation="eager" and use_cache=True so this path is exercised in prefill.

Real turboquant primitives are used: TurboQuantProd, ProdQuantized, quantize_values,
dequantize_values, ValueQuantized.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from turboquant.quantizer import TurboQuantProd, ProdQuantized
from turboquant.kv_cache import quantize_values, dequantize_values, ValueQuantized


class TurboQuantLayer(DynamicLayer):
    """Per-layer cache that compresses K (TurboQuantProd) and V (group quant)."""

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
        self._key_quantizer: Optional[TurboQuantProd] = None
        self._q_keys: Optional[ProdQuantized] = None
        self._q_values: Optional[ValueQuantized] = None
        self.compressed_bytes: int = 0

    def _get_quantizer(self, device) -> TurboQuantProd:
        if self._key_quantizer is None:
            self._key_quantizer = TurboQuantProd(
                dim=self.head_dim,
                bits=self.key_bits,
                device=device,
                seed=42 + self.seed_offset * 7,
            )
        return self._key_quantizer

    @staticmethod
    def _count_bytes_prod(q: ProdQuantized) -> int:
        b = q.mse_indices.nelement()            # bit-packed uint8
        b += q.qjl_signs.nelement()             # bit-packed uint8
        b += q.residual_norms.nelement() * 2    # float16 storage equiv
        b += q.norms.nelement() * 2             # float16 storage equiv
        return b

    @staticmethod
    def _count_bytes_val(vq: ValueQuantized) -> int:
        b = vq.data.nelement()                  # bit-packed uint8
        b += vq.scales.nelement() * 2           # float16 storage equiv
        b += vq.zeros.nelement() * 2            # float16 storage equiv
        return b

    def update(self, key_states, value_states, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress incoming K/V, append to store, return dequantized full history."""
        orig_dtype = key_states.dtype
        device = key_states.device
        k = key_states.float()
        v = value_states.float()

        quantizer = self._get_quantizer(device)
        new_key_q = quantizer.quantize(k)
        new_val_q = quantize_values(v, bits=self.value_bits, group_size=self.value_group_size)

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

        self.compressed_bytes = (
            self._count_bytes_prod(self._q_keys) + self._count_bytes_val(self._q_values)
        )

        k_deq = quantizer.dequantize(self._q_keys).to(orig_dtype)
        v_deq = dequantize_values(self._q_values, group_size=self.value_group_size).to(orig_dtype)
        return k_deq, v_deq

    def get_seq_length(self, *args, **kwargs) -> int:
        return 0 if self._q_keys is None else self._q_keys.norms.shape[-1]

    def get_max_cache_shape(self, *args, **kwargs) -> int:
        return -1


class TurboQuantCache(DynamicCache):
    """DynamicCache populated with TurboQuantLayer instances (one per model layer)."""

    def __init__(self, n_layers: int, head_dim: int, key_bits: int, value_bits: int,
                 value_group_size: int = 32):
        super().__init__()
        self._n_layers = n_layers
        self._head_dim = head_dim
        self._key_bits = key_bits
        self._value_bits = value_bits
        self._value_group_size = value_group_size
        self.layers = [
            TurboQuantLayer(head_dim, key_bits, value_bits, value_group_size, seed_offset=i)
            for i in range(n_layers)
        ]
        self.layer_class_to_replicate = None  # disable auto-grow

    def total_compressed_bytes(self) -> int:
        return sum(l.compressed_bytes for l in self.layers if isinstance(l, TurboQuantLayer))


def fp16_kv_bytes(n_tokens: int, n_layers: int, n_kv_heads: int, head_dim: int) -> int:
    """Baseline fp16 KV-cache bytes: 2 (K+V) * L * Hkv * hd * T * 2 bytes."""
    return 2 * n_layers * n_kv_heads * head_dim * n_tokens * 2
