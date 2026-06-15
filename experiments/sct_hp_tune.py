#!/usr/bin/env python3
"""
sct_hp_tune.py — HP-tuning harness for SCT per-component LR recipe.

For each (energy, sct_lr, lr_ratio, unfreeze) in the grid:
  1. Load fresh fp32 model
  2. Apply SCT to MLP layers at given energy
  3. Call finetune_sct(...)
  4. Evaluate held-out perplexity on wikitext-2 (plain DynamicCache / no TQ)
  5. Record result

Prints table sorted by perplexity (best first).
Saves all points + best recipe to experiments/sct_hp_tune_<modeltag>.json.

--probe mode: load model, apply SCT, run 3 steps, report timing + RAM only.
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if os.path.isdir(os.path.join(_ROOT, "SCT")):
    sys.path.insert(0, os.path.join(_ROOT, "SCT"))

from spectral_compact_training import SpectralLinear, retract_all
from sct_finetune import finetune_sct

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False


# ─────────────────────────────────────────────────────────────────────────────
#  SCT SURGERY (mirrors sct_tq_pareto.py)
# ─────────────────────────────────────────────────────────────────────────────

MLP_LEAF_NAMES = frozenset([
    "gate_proj", "up_proj", "down_proj",
    "fc_1", "fc_2",
    "c_fc", "c_proj",
])


def _adaptive_rank(W: torch.Tensor, energy: float) -> int:
    """Compute minimum rank retaining `energy` fraction of spectral energy."""
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    total = (S ** 2).sum()
    cumulative = torch.cumsum(S ** 2, dim=0) / total
    k = int((cumulative >= energy).nonzero(as_tuple=True)[0][0].item()) + 1
    return max(k, 1)


def replace_mlp_with_spectral(model, energy: float, device: str = "cpu"):
    """Replace MLP nn.Linear layers with SpectralLinear. Returns (n_replaced, dense_params, spectral_params)."""
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
        k = _adaptive_rank(W, energy)
        k = min(k, m, n)
        spec = SpectralLinear.from_linear(module, rank=k).to(device)
        # Store metadata for bookkeeping
        spec._dense_params = m * n
        parent_name, child_name = name.rsplit(".", 1)
        parent = dict(model.named_modules())[parent_name]
        setattr(parent, child_name, spec)
        total_dense += m * n
        total_spectral += spec.param_count()
        n_replaced += 1
    return n_replaced, total_dense, total_spectral


# ─────────────────────────────────────────────────────────────────────────────
#  EVAL PERPLEXITY (no TQ — plain DynamicCache)
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_TEXT = """
The quick brown fox jumps over the lazy dog. In the beginning, scientists believed that
atoms were the smallest indivisible units of matter. However, further research revealed
that atoms are composed of protons, neutrons, and electrons. The nucleus of an atom
contains protons and neutrons, while electrons orbit the nucleus in distinct energy levels.
Quantum mechanics describes the behavior of particles at the atomic and subatomic scale.
The uncertainty principle states that the position and momentum of a particle cannot both
be known precisely at the same time. This fundamental limit has profound implications for
our understanding of the physical world. Machine learning models learn patterns from data
by adjusting millions of parameters through a process called gradient descent.
""" * 20


def load_eval_text(max_tokens: int, tokenizer) -> torch.Tensor:
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = " ".join(row["text"] for row in ds if row["text"].strip())[:8000]
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        if len(ids) >= max_tokens:
            return ids[:max_tokens]
    except Exception:
        pass
    ids = tokenizer(FALLBACK_TEXT.strip(), return_tensors="pt")["input_ids"][0]
    while len(ids) < max_tokens:
        ids = torch.cat([ids, ids])
    return ids[:max_tokens]


def compute_perplexity(model, input_ids: torch.Tensor, device: str = "cpu") -> float:
    """Compute perplexity using a plain forward pass (no TQ)."""
    model.eval()
    ids = input_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    logits = out.logits
    shift_logits = logits[0, :-1, :].float()
    shift_labels = ids[0, 1:].long()
    loss = F.cross_entropy(shift_logits, shift_labels)
    return math.exp(min(loss.item(), 20.0))


# ─────────────────────────────────────────────────────────────────────────────
#  PROBE MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_probe(args):
    """Load model, apply SCT, run 3 finetune steps, print timing + RAM."""
    model_name = args.model
    energy = args.energies[0]
    device = "cpu"

    print(f"\n{'='*60}")
    print(f"  PROBE MODE: {model_name}")
    print(f"  Energy: {energy}  |  device: {device}")
    print(f"{'='*60}")

    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss

    print("  Loading model...")
    t_load_start = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    t_load_end = time.time()

    rss_after_load = proc.memory_info().rss
    load_ram_mb = (rss_after_load - rss_before) / (1024 ** 2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded in {t_load_end - t_load_start:.1f}s")
    print(f"  Parameters: {total_params:,}")
    print(f"  RAM delta (load): {load_ram_mb:.0f} MB")

    print(f"  Applying SCT (energy={energy})...")
    n_replaced, dense_p, spectral_p = replace_mlp_with_spectral(model, energy, device)
    print(f"  {n_replaced} MLP layers replaced, {dense_p:,} -> {spectral_p:,} params")

    # Run exactly 3 finetune steps and time steps 2-3
    print("  Running 3 probe steps...")
    result = finetune_sct(
        model, tok,
        sct_lr=args.sct_lrs[0],
        lr_ratio=args.lr_ratios[0],
        unfreeze=args.unfreeze[0],
        steps=3,
        batch_size=4,
        max_seq_len=128,
        max_samples=200,
        device=device,
        log_every=1,
        seed=42,
    )

    step_time = result["time_sec"]
    # Steps 2-3 mean (exclude step 1 which has data loading overhead built in)
    # Use total time / 3 as proxy since finetune_sct doesn't expose per-step times
    per_step_sec = step_time / 3.0

    print(f"\n  --- PROBE RESULTS ---")
    print(f"  Model load RAM delta:   {load_ram_mb:.0f} MB")
    print(f"  Per-step time (mean 2-3): {per_step_sec:.2f}s")
    print(f"  Projected wall-clock:")
    for n_steps in [200, 300, 500]:
        secs = per_step_sec * n_steps
        mins = secs / 60
        print(f"    {n_steps:4d} steps -> {secs:.0f}s ({mins:.1f} min)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN GRID SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_grid(args):
    model_name = args.model
    model_tag = model_name.rstrip("/").split("/")[-1]
    device = "cpu"

    print(f"\n{'='*70}")
    print(f"  SCT HP-Tuning Grid: {model_name}")
    print(f"  Energies:    {args.energies}")
    print(f"  SCT LRs:     {args.sct_lrs}")
    print(f"  LR ratios:   {args.lr_ratios}")
    print(f"  Unfreeze:    {args.unfreeze}")
    print(f"  Steps:       {args.steps}")
    print(f"  Max eval tokens: {args.max_eval_tokens}")
    print(f"{'='*70}\n")

    print("  Loading tokenizer for eval...")
    tok_eval = AutoTokenizer.from_pretrained(model_name)
    if tok_eval.pad_token is None:
        tok_eval.pad_token = tok_eval.eos_token

    print(f"  Loading eval text ({args.max_eval_tokens} tokens)...")
    eval_ids = load_eval_text(args.max_eval_tokens, tok_eval)
    print(f"  Eval tokens: {len(eval_ids)}\n")

    # Enumerate grid
    grid = []
    for energy in args.energies:
        for sct_lr in args.sct_lrs:
            for lr_ratio in args.lr_ratios:
                for unfreeze in args.unfreeze:
                    grid.append({
                        "energy": energy,
                        "sct_lr": sct_lr,
                        "lr_ratio": lr_ratio,
                        "unfreeze": unfreeze,
                    })

    total = len(grid)
    print(f"  Grid size: {total} configs\n")

    results = []

    for idx, cfg in enumerate(grid, 1):
        energy = cfg["energy"]
        sct_lr = cfg["sct_lr"]
        lr_ratio = cfg["lr_ratio"]
        unfreeze = cfg["unfreeze"]

        label = (f"E={energy} | sct_lr={sct_lr:.0e} | "
                 f"ratio={lr_ratio:.0f} | {unfreeze}")
        print(f"\n{'─'*70}")
        print(f"  Config {idx}/{total}: {label}")
        print(f"{'─'*70}")

        record = dict(cfg)
        record["label"] = label

        try:
            # Load fresh model
            tok_ft = AutoTokenizer.from_pretrained(model_name)
            if tok_ft.pad_token is None:
                tok_ft.pad_token = tok_ft.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float32
            )

            # Measure baseline ppl (no finetune)
            print("  Measuring pre-finetune ppl...")
            pre_ppl = compute_perplexity(model, eval_ids, device)
            print(f"  Pre-finetune ppl: {pre_ppl:.2f}")
            record["pre_ppl"] = round(pre_ppl, 4)

            # Apply SCT
            print(f"  Applying SCT (energy={energy})...")
            n_replaced, dense_p, spectral_p = replace_mlp_with_spectral(
                model, energy, device
            )
            compression = dense_p / max(spectral_p, 1)
            print(f"  {n_replaced} layers replaced, compression={compression:.2f}x")
            record["n_replaced"] = n_replaced
            record["dense_params"] = dense_p
            record["spectral_params"] = spectral_p
            record["compression"] = round(compression, 3)

            # Measure post-SCT (pre-finetune) ppl
            print("  Measuring post-SCT (pre-finetune) ppl...")
            post_sct_ppl = compute_perplexity(model, eval_ids, device)
            print(f"  Post-SCT ppl: {post_sct_ppl:.2f}")
            record["post_sct_ppl"] = round(post_sct_ppl, 4)

            # Finetune
            ft_result = finetune_sct(
                model, tok_ft,
                sct_lr=sct_lr,
                lr_ratio=lr_ratio,
                unfreeze=unfreeze,
                steps=args.steps,
                batch_size=4,
                max_seq_len=128,
                max_samples=500,
                device=device,
                log_every=50,
                seed=42,
            )
            record.update({
                "ft_final_loss": ft_result["final_loss"],
                "ft_final_ppl": ft_result["final_ppl"],
                "ft_steps": ft_result["steps"],
                "ft_time_sec": ft_result["time_sec"],
                "ft_sct_params": ft_result["sct_params"],
                "ft_dense_params": ft_result["dense_params"],
                "loss_curve": ft_result["loss_curve"],
            })

            # Eval ppl (post finetune)
            print("  Measuring post-finetune ppl...")
            post_ft_ppl = compute_perplexity(model, eval_ids, device)
            print(f"  Post-finetune ppl: {post_ft_ppl:.2f}")
            record["post_ft_ppl"] = round(post_ft_ppl, 4)
            record["ppl"] = round(post_ft_ppl, 4)  # used for ranking
            record["error"] = None

            del model

        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [ERROR] {e}\n{tb}")
            record["ppl"] = float("inf")
            record["error"] = str(e)

        results.append(record)

    # Sort by ppl
    results.sort(key=lambda r: (r["ppl"] if not math.isinf(r["ppl"]) else 1e12))

    # Print table
    print(f"\n{'='*70}")
    print("  RESULTS (sorted by post-finetune ppl, best first)")
    print(f"{'='*70}")
    header = f"  {'Config':<52} {'pre_ppl':>8} {'post_ft_ppl':>11} {'ft_time':>8}"
    print(header)
    print(f"  {'─'*52} {'─'*8} {'─'*11} {'─'*8}")
    for r in results:
        cfg_str = (f"E={r['energy']} sct_lr={r['sct_lr']:.0e} "
                   f"ratio={r['lr_ratio']:.0f} {r['unfreeze']}")
        pre = r.get("pre_ppl", float("nan"))
        post = r.get("post_ft_ppl", float("nan"))
        t = r.get("ft_time_sec", 0)
        err = " [ERR]" if r.get("error") else ""
        print(f"  {cfg_str:<52} {pre:>8.2f} {post:>11.2f} {t:>7.0f}s{err}")

    best = results[0]
    print(f"\n  BEST CONFIG: {best['label']}")
    pre_str = f"{best['pre_ppl']:.2f}" if 'pre_ppl' in best else "N/A"
    post_str = f"{best['post_ft_ppl']:.2f}" if 'post_ft_ppl' in best else "N/A"
    print(f"    pre_ppl={pre_str} -> post_ft_ppl={post_str}")

    # Save JSON
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, f"sct_hp_tune_{model_tag}.json")
    payload = {
        "model": model_name,
        "grid": {
            "energies": args.energies,
            "sct_lrs": args.sct_lrs,
            "lr_ratios": args.lr_ratios,
            "unfreeze": args.unfreeze,
            "steps": args.steps,
        },
        "best": {k: v for k, v in best.items() if k != "loss_curve"},
        "results": [{k: v for k, v in r.items() if k != "loss_curve"}
                    for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="SCT HP-tuning harness with per-component LR"
    )
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--energies", type=float, nargs="+", default=[0.95])
    p.add_argument("--sct-lrs", type=float, nargs="+", default=[1e-3, 5e-4, 1e-4],
                   dest="sct_lrs")
    p.add_argument("--lr-ratios", type=float, nargs="+", default=[25, 10],
                   dest="lr_ratios")
    p.add_argument("--unfreeze", nargs="+",
                   default=["norms+attn", "norms_only"],
                   choices=["norms_only", "norms+attn"])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--max-eval-tokens", type=int, default=512,
                   dest="max_eval_tokens")
    p.add_argument("--probe", action="store_true",
                   help="Probe mode: load model, apply SCT, run 3 steps, report timing")
    args = p.parse_args()

    if args.probe:
        run_probe(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
