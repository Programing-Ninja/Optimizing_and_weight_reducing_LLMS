"""
recovery.py — recovery-LoRA: recover utility lost to SCT/TQ compression by training
low-rank adapters on the attention projections (which stay nn.Linear after SCT
compresses only the MLP), then merging them back into the base weights.

We use HuggingFace `peft`. The SCT SpectralLinear MLP factors and all other base
weights are frozen; only the LoRA adapters train. After training we `merge_and_unload`
so the adapter folds into the existing q/k/v/o_proj weights — adding ZERO storage
(the recovered model has the same byte footprint as the un-recovered SCT model).

Default data is a small tatsu-lab/alpaca slice (matches the repo's prior finetune).
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F


def _prepare_alpaca(tokenizer, max_seq_len: int, max_samples: int, seed: int):
    from datasets import load_dataset

    def fmt(ex):
        if ex.get("input", "").strip():
            return (f"### Instruction:\n{ex['instruction']}\n\n"
                    f"### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}")
        return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
    texts = [fmt(ex) for ex in ds]
    enc = tokenizer(texts, truncation=True, max_length=max_seq_len,
                    padding="max_length", return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return enc["input_ids"], enc["attention_mask"], labels


def train_recovery_lora(
    model,
    tokenizer,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    lr: float = 1e-4,
    steps: int = 200,
    batch_size: int = 4,
    max_seq_len: int = 256,
    max_samples: int = 1000,
    device: str = "cuda",
    seed: int = 42,
    log_every: int = 50,
    merge: bool = True,
):
    """Train a recovery-LoRA on `model` (in place). Returns (model, info dict).

    If merge=True, the adapter is folded into the base weights and a plain model is
    returned (zero extra storage). If merge=False, the peft-wrapped model is returned.
    """
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(target_modules), bias="none", task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.to(device).train()

    input_ids, attn_mask, labels = _prepare_alpaca(tokenizer, max_seq_len, max_samples, seed)
    n = input_ids.shape[0]

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    warmup = min(20, steps // 5)

    def lr_fn(step):
        if step < warmup:
            return step / max(warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(steps - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    loss_curve, step, t0 = [], 0, time.time()
    while step < steps:
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            if step >= steps:
                break
            idx = perm[i:i + batch_size]
            xb = input_ids[idx].to(device)
            mb = attn_mask[idx].to(device)
            yb = labels[idx].to(device)
            logits = peft_model(input_ids=xb, attention_mask=mb).logits
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                yb[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()
            loss_curve.append(round(float(loss.item()), 5))
            step += 1
            if step % log_every == 0 or step == 1 or step == steps:
                recent = loss_curve[-log_every:]
                avg = sum(recent) / len(recent)
                print(f"  [recovery-LoRA] step {step:4d}/{steps} | loss {avg:.4f} | "
                      f"ppl {math.exp(min(avg, 20)):.1f} | {time.time()-t0:.1f}s")

    info = {
        "rank": rank, "alpha": alpha, "steps": step,
        "time_sec": round(time.time() - t0, 2),
        "final_loss": round(sum(loss_curve[-20:]) / min(len(loss_curve), 20), 5),
        "trainable_params": sum(p.numel() for p in trainable),
        "target_modules": list(target_modules),
    }

    peft_model.eval()
    if merge:
        merged = peft_model.merge_and_unload()  # fold adapter into base weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return merged, info
    return peft_model, info
