# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Rules

1. Always commit and push previous work to GitHub before starting a new prompt. Remote: `https://github.com/Programing-Ninja/Optimizing_and_weight_reducing_LLMS.git`
2. Append every prompt (yours or the user's) to `prompts.txt` for tracing project growth.
3. Before implementing anything, write a plan in `path.txt`, then implement it.
4. After each experiment or attempt, document what was tried, what worked, and what didn't in `iteration.txt`.
5. Update the relevant `README.md` whenever a major feature or result is added.
6. If something is counterintuitive or there is a clearly better approach, raise it explicitly.

## Model Selection

- Documentation tasks → Haiku (lightweight, fast)
- Planning and architecture decisions → Opus (heavyweight reasoning)
- Implementation → Sonnet or Haiku depending on complexity

## Repository Structure

This workspace contains two independent sub-projects, each with its own git repo:

```
sct_tq/
  SCT/           — Spectral Compact Training (training method)
  turboquant/    — TurboQuant KV cache compression (inference method)
  experiments/   — Scratch space for cross-project experiments
```

---

## SCT — Spectral Compact Training

**What it does:** Replaces `nn.Linear` with `SpectralLinear`, storing weights as `W = U diag(s) V^T`. Gradients flow through the three small factors (never through a dense `m×n` matrix). After each optimizer step, U and V are retracted to the Stiefel manifold via QR.

### Install

```bash
cd SCT
pip install -e .
# For examples:
pip install -e ".[examples]"
```

### Key commands

```bash
# 70B architecture memory validation (fits on 8GB RAM, CPU)
python examples/sct_steamdeck.py

# SmolLM2 fine-tuning vs dense head-to-head
python examples/sct_vs_dense.py --energy 0.95 --steps 400

# SmolLM2 fine-tuning with SCT
python examples/sct_smollm2.py --energy 0.95 --steps 400
```

### Core API

```python
from spectral_compact_training import SpectralLinear, retract_all

# From scratch
layer = SpectralLinear(in_features=4096, out_features=11008, rank=32)

# From pretrained dense layer
layer = SpectralLinear.from_linear(dense_layer, rank=32)

# After every optimizer.step():
retract_all(model)
```

### Architecture

- `spectral_compact_training/spectral_layer.py` — the entire method: `SpectralLinear`, `safe_qr`, `retract_all`
- `spectral_compact_training/__init__.py` — re-exports the above
- `examples/` — standalone scripts and Colab notebooks for all published experiments

**`safe_qr`** falls back to CPU for MPS/AMD backends that have QR driver bugs — don't remove this.

**`from_linear` SVD convention:** `W = U_svd @ diag(S) @ Vh_svd`, mapped to the forward convention `y = x @ U * s @ V.T` via `U_ours = Vh[:k].T`, `V_ours = U_svd[:, :k]`.

### Key constraints

- Rank is clamped to `min(in_features, out_features)` — never exceeds the layer's true rank.
- SCT compresses best at hidden dim ≥ 2048 (1.7B+ models). Below this, ranks near the full dimension give little compression.
- Stiefel retraction (`O(mk²)` per layer) grows expensive at high rank — at 70B scale it's ~40–50% of total step time.

---

## TurboQuant — KV Cache Compression

**What it does:** Compresses KV cache entries online during LLM inference using random rotation + Lloyd-Max scalar quantization (b−1 bits for keys) + QJL residual sign bits (1 bit) + group quantization for values (2-bit or 4-bit). The combined inner-product estimator is unbiased.

### Install

```bash
cd turboquant
pip install -e .
# With vLLM integration:
pip install -e ".[vllm]"
# With Triton kernels:
pip install -e ".[triton]"
```

### Key commands

```bash
# Paper theorem validation (CPU, no GPU needed) — 9 tests
python validate_paper.py

# Adversarial audit of all claims
python audit_claims.py

# Modular architecture tests (19 tests)
python -m pytest test_modular.py -v

# Core quantizer tests (7 tests)
python -m pytest test_turboquant.py -v

# A/B benchmark: baseline vLLM vs TurboQuant (requires GPU + Qwen3.5-27B-AWQ)
CUDA_VISIBLE_DEVICES=0,1,4,6 python proof.py
```

### Module map

| File | Role |
|------|------|
| `turboquant/codebook.py` | Lloyd-Max optimal scalar quantizer for Beta distribution |
| `turboquant/codebooks/` | Pre-generated codebook JSON files (d=64/128/576, bits 1–4) |
| `turboquant/rotation.py` | Random orthogonal rotation (Π) and QJL projection matrix (S) |
| `turboquant/quantizer.py` | `TurboQuantMSE` (Alg. 1) and `TurboQuantProd` (Alg. 2) |
| `turboquant/kv_cache.py` | KV cache manager with value bit-packing |
| `turboquant/capture.py` | `RingBuffer`, `KVCaptureEngine` — modular KV hooks |
| `turboquant/store.py` | `CompressedKVStore` — quantize + append + flat cache |
| `turboquant/score.py` | `compute_hybrid_attention` from compressed keys |
| `turboquant/integration/vllm.py` | vLLM adapter: monkey-patch, `free_kv_cache`, hybrid decode |
| `turboquant/triton_kernels.py` | 3 fused Triton kernels for decode attention |
| `turboquant/vllm_attn_backend.py` | Thin shim delegating to `integration/vllm.py` |

### Key constraints

- Only compresses **full-attention layers** — linear-attention (Mamba, GQA-linear, etc.) is not handled.
- 2-bit value quantization gives cos_sim ≈ 0.94; use 4-bit values (cos_sim ≈ 0.997) for quality-sensitive workloads.
- Hybrid decode dequantizes all compressed history to float32 — memory savings are real but compute savings are not yet realized.
- vLLM integration is a monkey-patch on vLLM 0.18.0; newer vLLM versions may break the hook points.
- Codebook files are loaded at module import via `get_codebook` — ensure `turboquant/codebooks/` is present in any deployment.

### TurboQuantProd quantization flow

1. `TurboQuantMSE` quantizes `x` at `(b−1)` bits → `x̃`
2. Residual `r = x − x̃` projected via QJL matrix `S`: `sign(S·r)` → 1 bit/dim
3. Inner-product estimate: `<q, x̃> + ||r|| · √(π/2)/d · <S^T·signs, q>`
