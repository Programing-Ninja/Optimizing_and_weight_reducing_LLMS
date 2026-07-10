"""
eval_tasks.py — the four utility benchmarks, all on the SAME (possibly compressed +
quantized) model, with every forward routed through an optional TurboQuant cache so
KV quantization actually affects the scores.

Tasks (all loglikelihood-based, subset-able for fast sweeps):
  - perplexity : wikitext-2-raw-v1  (lower is better)
  - HellaSwag  : token-length-normalized acc_norm over 4 endings
  - MMLU       : zero-shot 4-way (Answer: A/B/C/D) accuracy
  - TruthfulQA : MC2 (normalized probability mass on true answers)

A `cache_factory` callable builds a FRESH TurboQuantCache per forward (None = dense
fp16 baseline, run with use_cache=False). Default batch size is 1 for robustness;
these are short sequences and the subset sizes are small.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
#  CACHE FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_cache_factory(model, key_bits: Optional[int], value_bits: Optional[int],
                       value_group_size: int = 32):
    """Return a zero-arg factory producing a fresh cache per forward.

    key_bits None -> returns None (dense fp16 baseline, no KV quantization).
    """
    if key_bits is None:
        return lambda: None

    from .tq_cache import TurboQuantCache  # local import to avoid hard dep when unused

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    # Shared across every cache this factory produces (one eval point = many forwards).
    # Keyed by (layer seed_offset, device) inside TurboQuantLayer so per-layer device
    # placement under multi-GPU dispatch is still inferred correctly on first use.
    quantizer_cache: dict = {}

    def factory():
        return TurboQuantCache(
            n_layers=n_layers,
            head_dim=head_dim,
            key_bits=key_bits,
            value_bits=value_bits,
            value_group_size=value_group_size,
            quantizer_cache=quantizer_cache,
        )

    return factory


# ─────────────────────────────────────────────────────────────────────────────
#  LOW-LEVEL SCORING
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _continuation_logprob(model, ctx_ids: torch.Tensor, cont_ids: torch.Tensor,
                          cache_factory: Callable, device: str):
    """Sum log p(continuation | context). Returns (sum_logprob, n_cont_tokens)."""
    cont_len = cont_ids.shape[-1]
    input_ids = torch.cat([ctx_ids, cont_ids], dim=-1).unsqueeze(0).to(device)  # (1, T)
    cache = cache_factory()
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=cache is not None)
    logits = out.logits[0].float()  # (T, V)
    # token at position (ctx_len + i) is predicted by logits[ctx_len + i - 1]
    sliced = logits[-cont_len - 1:-1, :]            # (cont_len, V)
    logprobs = F.log_softmax(sliced, dim=-1)
    token_lp = logprobs.gather(-1, cont_ids.to(device).view(-1, 1)).squeeze(-1)
    return float(token_lp.sum().item()), cont_len


def _mc_argmax(model, tokenizer, context: str, choices_text, cache_factory, device,
               length_normalize: bool = True):
    """Return index of the highest-scoring continuation among choices_text."""
    ctx_ids = tokenizer(context, return_tensors="pt")["input_ids"][0]
    scores = []
    for ch in choices_text:
        cont_ids = tokenizer(ch, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if cont_ids.numel() == 0:
            cont_ids = tokenizer(" ", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        lp, n = _continuation_logprob(model, ctx_ids, cont_ids, cache_factory, device)
        scores.append(lp / n if length_normalize else lp)
    return int(max(range(len(scores)), key=lambda i: scores[i]))


# ─────────────────────────────────────────────────────────────────────────────
#  PERPLEXITY  (wikitext-2)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_perplexity(model, tokenizer, cache_factory, device, max_tokens=2048,
                    window=1024) -> float:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0][:max_tokens]

    total_nll, total_tok = 0.0, 0
    for start in range(0, len(ids) - 1, window):
        chunk = ids[start:start + window + 1]
        if chunk.numel() < 2:
            break
        inp = chunk[:-1].unsqueeze(0).to(device)
        tgt = chunk[1:].to(device)
        cache = cache_factory()
        out = model(input_ids=inp, past_key_values=cache, use_cache=cache is not None)
        logits = out.logits[0].float()
        nll = F.cross_entropy(logits, tgt, reduction="sum")
        total_nll += float(nll.item())
        total_tok += tgt.numel()
    return math.exp(min(total_nll / max(total_tok, 1), 20.0))


# ─────────────────────────────────────────────────────────────────────────────
#  HELLASWAG
# ─────────────────────────────────────────────────────────────────────────────

def _hellaswag_ctx(row) -> str:
    ctx = (row["ctx_a"] + " " + row["ctx_b"].capitalize()).strip() if row.get("ctx_b") else row["ctx_a"]
    label = row.get("activity_label", "")
    return (f"{label}: {ctx}" if label else ctx)


@torch.no_grad()
def eval_hellaswag(model, tokenizer, cache_factory, device, limit=400, seed=42) -> float:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:limit]

    correct = 0
    for n, i in enumerate(idxs):
        row = ds[i]
        ctx = _hellaswag_ctx(row)
        endings = [" " + e.strip() for e in row["endings"]]
        gold = int(row["label"])
        pred = _mc_argmax(model, tokenizer, ctx, endings, cache_factory, device, True)
        correct += int(pred == gold)
    return correct / max(len(idxs), 1)


# ─────────────────────────────────────────────────────────────────────────────
#  MMLU  (zero-shot, 4-way letter)
# ─────────────────────────────────────────────────────────────────────────────

_LETTERS = ["A", "B", "C", "D"]


def _mmlu_prompt(row) -> str:
    q = row["question"].strip()
    lines = [q]
    for letter, choice in zip(_LETTERS, row["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


@torch.no_grad()
def eval_mmlu(model, tokenizer, cache_factory, device, limit=400, seed=42) -> float:
    ds = load_dataset("cais/mmlu", "all", split="test")
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:limit]

    correct = 0
    for i in idxs:
        row = ds[i]
        prompt = _mmlu_prompt(row)
        choices_text = [f" {l}" for l in _LETTERS]
        gold = int(row["answer"])
        pred = _mc_argmax(model, tokenizer, prompt, choices_text, cache_factory, device,
                          length_normalize=False)
        correct += int(pred == gold)
    return correct / max(len(idxs), 1)


# ─────────────────────────────────────────────────────────────────────────────
#  TRUTHFULQA  (MC2)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_truthfulqa_mc2(model, tokenizer, cache_factory, device, limit=200, seed=42) -> float:
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:limit]

    scores = []
    for i in idxs:
        row = ds[i]
        question = f"Q: {row['question'].strip()}\nA:"
        choices = row["mc2_targets"]["choices"]
        labels = row["mc2_targets"]["labels"]  # 1 = true, 0 = false

        ctx_ids = tokenizer(question, return_tensors="pt")["input_ids"][0]
        logprobs = []
        for ch in choices:
            cont = tokenizer(" " + ch.strip(), return_tensors="pt",
                             add_special_tokens=False)["input_ids"][0]
            if cont.numel() == 0:
                cont = tokenizer(" ", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            lp, _ = _continuation_logprob(model, ctx_ids, cont, cache_factory, device)
            logprobs.append(lp)
        lp_t = torch.tensor(logprobs)
        probs = torch.softmax(lp_t, dim=0)
        true_mass = float(probs[torch.tensor(labels, dtype=torch.bool)].sum().item())
        scores.append(true_mass)
    return sum(scores) / max(len(scores), 1)


# ─────────────────────────────────────────────────────────────────────────────
#  DRIVER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(model, tokenizer, cache_factory, device, limits: dict) -> dict:
    """Run the four benchmarks. `limits` keys: ppl_tokens, hellaswag, mmlu, truthfulqa."""
    model.eval()
    out = {}
    out["perplexity"] = eval_perplexity(
        model, tokenizer, cache_factory, device, max_tokens=limits.get("ppl_tokens", 2048))
    out["hellaswag"] = eval_hellaswag(
        model, tokenizer, cache_factory, device, limit=limits.get("hellaswag", 400))
    out["mmlu"] = eval_mmlu(
        model, tokenizer, cache_factory, device, limit=limits.get("mmlu", 400))
    out["truthfulqa"] = eval_truthfulqa_mc2(
        model, tokenizer, cache_factory, device, limit=limits.get("truthfulqa", 200))
    return out
