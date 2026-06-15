#!/usr/bin/env python3
"""
sct_finetune.py — Reusable per-component-LR finetune for SCT models.

Exposes:
    finetune_sct(model, tokenizer, *, sct_lr, lr_ratio, unfreeze, steps,
                 batch_size, max_seq_len, max_samples, device, log_every, seed)
    -> dict

Per-component learning rate recipe (from sct_per_component_lr.ipynb):
  Group A (dense, low lr = sct_lr / lr_ratio):
    - unfreeze=="norms_only"  -> *norm* params only
    - unfreeze=="norms+attn"  -> q_proj/k_proj/v_proj/o_proj + *norm* params
  Group B (SCT factors, high lr = sct_lr):
    - m.U, m.s, m.V for every SpectralLinear module
  Everything else: frozen (requires_grad=False)

Cosine schedule with short warmup, grad clip 1.0, retract_all after every step.
Data: tatsu-lab/alpaca formatted as in sct_vs_dense.py.
"""

import math
import os
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running from repo root without install
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if os.path.isdir(os.path.join(_ROOT, "SCT")):
    sys.path.insert(0, os.path.join(_ROOT, "SCT"))

from spectral_compact_training import SpectralLinear, retract_all

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False


# ─────────────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────────────

def _format_alpaca(ex):
    if ex.get("input", "").strip():
        return (f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}")
    return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"


def _prepare_data(tokenizer, max_seq_len: int, max_samples: int, seed: int):
    """Load tatsu-lab/alpaca, format, tokenize. Returns (input_ids, attn_mask, labels)."""
    if not _HAS_DATASETS:
        raise ImportError("datasets library required: pip install datasets")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
    texts = [_format_alpaca(ex) for ex in ds]
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return enc["input_ids"], enc["attention_mask"], labels


# ─────────────────────────────────────────────────────────────────────────────
#  PARAM GROUP BUILDER
# ─────────────────────────────────────────────────────────────────────────────

_ATTN_LEAF_NAMES = frozenset(["q_proj", "k_proj", "v_proj", "o_proj"])
_NORM_KEYWORDS = ("layernorm", "ln_", "norm", "rmsnorm")


def _build_param_groups(model, sct_lr: float, dense_lr: float, unfreeze: str):
    """
    Freeze all params then identify group A (dense, low lr) and group B (SCT, high lr).

    Returns (param_groups_list, sct_param_count, dense_param_count).
    """
    # First freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Collect SpectralLinear factor params (group B)
    sct_ids = set()
    sct_params = []
    for m in model.modules():
        if isinstance(m, SpectralLinear):
            for factor in (m.U, m.s, m.V):
                if id(factor) not in sct_ids:
                    sct_ids.add(id(factor))
                    factor.requires_grad = True
                    sct_params.append(factor)
            # bias stays frozen (tiny, not a spectral factor)

    # Collect dense group A params
    dense_ids = set()
    dense_params = []

    def _add(p):
        if id(p) not in sct_ids and id(p) not in dense_ids:
            dense_ids.add(id(p))
            p.requires_grad = True
            dense_params.append(p)

    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        is_norm = any(k in name.lower() for k in _NORM_KEYWORDS)
        is_attn = leaf in _ATTN_LEAF_NAMES

        if is_norm:
            for p in module.parameters(recurse=False):
                _add(p)
        elif unfreeze == "norms+attn" and is_attn and isinstance(module, nn.Linear):
            for p in module.parameters(recurse=False):
                _add(p)

    param_groups = []
    if sct_params:
        param_groups.append({"params": sct_params, "lr": sct_lr})
    if dense_params:
        param_groups.append({"params": dense_params, "lr": dense_lr})

    return param_groups, len(sct_params), len(dense_params)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN FINETUNE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def finetune_sct(
    model,
    tokenizer,
    *,
    sct_lr: float,
    lr_ratio: float,
    unfreeze: str,          # "norms_only" or "norms+attn"
    steps: int,
    batch_size: int = 4,
    max_seq_len: int = 128,
    max_samples: int = 500,
    device: str = "cpu",
    log_every: int = 50,
    seed: int = 42,
) -> dict:
    """
    Fine-tune an SCT model (already has SpectralLinear layers) with
    per-component learning rates.

    Returns dict with keys:
        final_loss, final_ppl, steps, time_sec,
        sct_params, dense_params, loss_curve (list of per-step losses)
    """
    assert unfreeze in ("norms_only", "norms+attn"), \
        f"unfreeze must be 'norms_only' or 'norms+attn', got {unfreeze!r}"

    torch.manual_seed(seed)
    dense_lr = sct_lr / lr_ratio

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"  [finetune] Loading data (max_samples={max_samples}, seq_len={max_seq_len})...")
    input_ids, attn_mask, labels = _prepare_data(
        tokenizer, max_seq_len, max_samples, seed
    )
    print(f"  [finetune] {input_ids.shape[0]} samples loaded")

    # ── Param groups ──────────────────────────────────────────────────────
    param_groups, n_sct, n_dense = _build_param_groups(
        model, sct_lr=sct_lr, dense_lr=dense_lr, unfreeze=unfreeze
    )
    if not param_groups:
        raise RuntimeError("No trainable parameters found! "
                           "Ensure SpectralLinear layers are present in model.")

    n_sct_params = sum(p.numel() for g in param_groups if g["lr"] == sct_lr
                       for p in g["params"])
    n_dense_params = sum(p.numel() for g in param_groups if g["lr"] == dense_lr
                         for p in g["params"])
    trainable_all = [p for g in param_groups for p in g["params"]]

    print(f"  [finetune] sct_lr={sct_lr:.2e}, dense_lr={dense_lr:.2e} "
          f"(ratio={lr_ratio:.0f}x), unfreeze={unfreeze}")
    print(f"  [finetune] SCT factor params: {n_sct_params:,} | "
          f"Dense group params: {n_dense_params:,}")

    # ── Optimizer + Schedule ──────────────────────────────────────────────
    opt = torch.optim.AdamW(param_groups, weight_decay=0.01)
    warmup = min(20, steps // 5)

    def lr_fn(step):
        if step < warmup:
            return step / max(warmup, 1)
        return 0.5 * (1 + math.cos(
            math.pi * (step - warmup) / max(steps - warmup, 1)
        ))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    # ── Training loop ─────────────────────────────────────────────────────
    model.to(device).train()
    n = input_ids.shape[0]
    bs = batch_size
    loss_curve = []
    step = 0
    t0 = time.time()

    for _epoch in range(9999):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            if step >= steps:
                break
            idx = perm[i : i + bs]
            xb = input_ids[idx].to(device)
            mb = attn_mask[idx].to(device)
            yb = labels[idx].to(device)

            logits = model(input_ids=xb, attention_mask=mb).logits
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                yb[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_all, 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()

            # Stiefel retraction — critical after every step
            retract_all(model)

            loss_val = loss.item()
            loss_curve.append(round(loss_val, 5))
            step += 1

            if step % log_every == 0 or step == 1 or step == steps:
                recent = loss_curve[-log_every:]
                avg = sum(recent) / len(recent)
                elapsed = time.time() - t0
                print(f"  [finetune] step {step:4d}/{steps} | "
                      f"loss {avg:.4f} | ppl {math.exp(min(avg, 20)):.1f} | "
                      f"{elapsed:.1f}s")
        if step >= steps:
            break

    elapsed = time.time() - t0
    tail = loss_curve[-20:] if len(loss_curve) >= 20 else loss_curve
    final_loss = sum(tail) / len(tail)
    final_ppl = math.exp(min(final_loss, 20.0))

    print(f"  [finetune] Done: {step} steps in {elapsed:.1f}s | "
          f"final_loss={final_loss:.4f} | final_ppl={final_ppl:.2f}")

    return {
        "final_loss": round(final_loss, 5),
        "final_ppl": round(final_ppl, 3),
        "steps": step,
        "time_sec": round(elapsed, 2),
        "sct_params": n_sct_params,
        "dense_params": n_dense_params,
        "loss_curve": loss_curve,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Quick CLI for manual testing
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # inline replace_mlp_with_spectral for standalone use
    MLP_LEAF_NAMES = frozenset([
        "gate_proj", "up_proj", "down_proj",
        "fc_1", "fc_2", "c_fc", "c_proj",
    ])

    def _adaptive_rank(W, energy):
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        total = (S ** 2).sum()
        cumulative = torch.cumsum(S ** 2, dim=0) / total
        k = int((cumulative >= energy).nonzero(as_tuple=True)[0][0].item()) + 1
        return max(k, 1)

    def replace_mlp_with_spectral(model, energy, device="cpu"):
        total_dense = 0
        total_spectral = 0
        n_replaced = 0
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            leaf = name.rsplit(".", 1)[-1] if "." in name else name
            if leaf not in MLP_LEAF_NAMES:
                continue
            W = module.weight.data.float().cpu()
            m, n = W.shape
            k = min(_adaptive_rank(W, energy), m, n)
            spec = SpectralLinear.from_linear(module, rank=k).to(device)
            parent_name, child_name = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, spec)
            total_dense += m * n
            total_spectral += spec.param_count()
            n_replaced += 1
        return n_replaced, total_dense, total_spectral

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--energy", type=float, default=0.95)
    p.add_argument("--sct-lr", type=float, default=5e-4)
    p.add_argument("--lr-ratio", type=float, default=25)
    p.add_argument("--unfreeze", default="norms+attn",
                   choices=["norms_only", "norms+attn"])
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    print(f"Applying SCT energy={args.energy}...")
    n, d, s = replace_mlp_with_spectral(model, args.energy, args.device)
    print(f"  {n} MLP layers replaced, {d:,} dense -> {s:,} spectral")

    result = finetune_sct(
        model, tok,
        sct_lr=args.sct_lr,
        lr_ratio=args.lr_ratio,
        unfreeze=args.unfreeze,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("\nResult:", result)
