#!/usr/bin/env python3
"""
run_pareto.py — SCT x TurboQuant utility-vs-compression Pareto sweep
(Llama-3.1-8B on one A100 40/80GB; Llama-3.1-70B via --big-model + --gpus).

Sweeps:
  - SCT energy E  (weight-side compression; None = dense baseline)
  - TurboQuant KV bits (key_bits, value_bits)  (inference-side; None = fp16 KV)
  - recovery-LoRA toggle {off, on}  (on = retrain LoRA on the SCT model, then merge)

For each point it computes the four utility benchmarks (forced through the TQ cache),
aggregates them into U normalized vs the dense baseline, accounts the compressed bytes,
and traces the Pareto frontier + an energy x KV-bits utility heatmap.

LoRA scope note: recovery-LoRA training never involves the KV cache (the cache is an
inference-time object), so a LoRA is trained ONCE per SCT energy and reused across all
KV configs — there is no meaningful "per-KV-point" LoRA. The off/on toggle is what
exposes the recovery gain.

One model lives in memory at a time.

70B path (--big-model, single A100 80GB while GPU 0 is occupied: --gpus 1):
  load on CPU -> SCT with the SVD on the GPU -> accelerate dispatch_model with
  a per-GPU max_memory cap. Compressed points usually fit wholly on the GPU;
  the dense baseline runs partially CPU-offloaded (slow but correct).
"""

import argparse
import gc
import json
import os
import sys
import time

# CUDA_VISIBLE_DEVICES must be pinned BEFORE torch initializes CUDA.
from pipeline.big_model import pin_gpus_from_argv
pin_gpus_from_argv()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.sct_apply import apply_sct
from pipeline.tq_cache import TurboQuantLayer, fp16_kv_bytes
from pipeline.eval_tasks import make_cache_factory, evaluate_all
from pipeline.utility import aggregate_utility, DEFAULT_WEIGHTS
from pipeline.compression import model_weight_bytes, compression_ratio
from pipeline.recovery import train_recovery_lora
from pipeline.big_model import load_model, dispatch_big, input_device
from pipeline import pareto

HERE = os.path.dirname(os.path.abspath(__file__))


def results_dir_for(model_name: str) -> str:
    tag = model_name.rstrip("/").split("/")[-1].replace(".", "").lower()
    # keep the historical dir name for the 8B so old results stay in place
    if tag == "llama-31-8b":
        return os.path.join(HERE, "results_llama8b")
    return os.path.join(HERE, f"results_{tag}")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def is_big(args) -> bool:
    if args.big_model == "on":
        return True
    if args.big_model == "off":
        return False
    return "70b" in args.model.lower()  # auto


def build_point_model(args, dtype, energy):
    """Load the base model and apply SCT at `energy`, handling both paths.

    Small path: load -> .to(cuda) -> SCT on-GPU (unchanged 8B behaviour).
    Big path:   load on CPU -> SCT with the SVD on the GPU (shrinks the model
                BEFORE placement) -> dispatch across visible GPUs + CPU.

    Returns (model, tok, sct_stats, device) — `device` is where inputs go.
    """
    big = is_big(args)
    model, tok = load_model(args.model, dtype, big, device=args.device)
    svd_dev = "cuda:0" if (big and torch.cuda.is_available()) else None
    sct_stats = apply_sct(model, energy, svd_method=args.svd, svd_device=svd_dev)
    if big:
        model, _ = dispatch_big(model, max_gpu_mem_gib=args.max_gpu_mem,
                                offload_dir=args.offload_dir)
    model.eval()
    return model, tok, sct_stats, input_device(model)


@torch.no_grad()
def measure_kv_bytes(model, key_bits, value_bits, ref_tokens, group_size=32):
    """Real compressed KV bytes at a reference context length, summed over layers."""
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    if key_bits is None:
        return fp16_kv_bytes(ref_tokens, n_layers, n_kv_heads, head_dim)

    dev = next(model.parameters()).device
    layer = TurboQuantLayer(head_dim, key_bits, value_bits, group_size, seed_offset=0)
    k = torch.randn(1, n_kv_heads, ref_tokens, head_dim, device=dev)
    v = torch.randn(1, n_kv_heads, ref_tokens, head_dim, device=dev)
    layer.update(k, v)
    return layer.compressed_bytes * n_layers


def free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dry_run(args, dtype):
    """Smoke test: exercise model load + SCT + TQ-cache forward + KV accounting +
    peft availability, with NO dataset downloads and NO training. Isolates the
    model/cache integration from dataset/network issues on a fresh HPC node."""
    print("=" * 78)
    print("  DRY RUN — model + SCT + TurboQuant cache integration (no datasets/training)")
    print("=" * 78)
    ok = True

    energy = (args.energies[0] if args.energies else 0.95)
    print(f"[1-2/6] loading {args.model} (dtype={args.dtype}, "
          f"big={is_big(args)}) + SCT at energy={energy} ...")
    model, tok, stats, dev = build_point_model(args, dtype, energy)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"      OK: {nparams/1e9:.2f}B params, "
          f"hidden={model.config.hidden_size}, layers={model.config.num_hidden_layers}, "
          f"inputs->{dev}")
    print(f"      SCT: MLP {stats['mlp_ratio']:.2f}x, mean rank "
          f"{sum(stats['ranks'])/max(len(stats['ranks']),1):.0f}")

    prompt = "The quick brown fox jumps over the lazy dog. In a few words,"
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(dev)

    print("[3/6] forward through TurboQuantCache (k=3,v=2) ...")
    try:
        cache = make_cache_factory(model, 3, 2)()
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        print(f"      OK: logits {tuple(out.logits.shape)}, "
              f"compressed_bytes={cache.total_compressed_bytes():,}")
    except Exception as e:
        ok = False
        print(f"      FAIL: {type(e).__name__}: {e}")

    print("[4/6] dense forward (no cache) ...")
    try:
        out2 = model(input_ids=ids, use_cache=False)
        print(f"      OK: logits {tuple(out2.logits.shape)}")
    except Exception as e:
        ok = False
        print(f"      FAIL: {type(e).__name__}: {e}")

    print(f"[5/6] KV byte accounting @ {args.kv_ref_tokens} tokens ...")
    try:
        kv_fp16 = measure_kv_bytes(model, None, None, args.kv_ref_tokens)
        kv_q = measure_kv_bytes(model, 3, 2, args.kv_ref_tokens)
        print(f"      OK: fp16={kv_fp16/1e6:.1f}MB  q(3x2)={kv_q/1e6:.1f}MB  "
              f"({kv_fp16/max(kv_q,1):.1f}x)")
    except Exception as e:
        ok = False
        print(f"      FAIL: {type(e).__name__}: {e}")

    print("[6/6] recovery-LoRA backend (peft) availability ...")
    try:
        import peft
        print(f"      OK: peft {peft.__version__}")
    except Exception as e:
        ok = False
        print(f"      FAIL: peft not importable: {e}")

    free(model)
    print("=" * 78)
    print("  DRY RUN PASSED ✓" if ok else "  DRY RUN FAILED ✗ — fix the FAIL above before a real run")
    print("=" * 78)
    return 0 if ok else 1


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    # --- big-model / multi-GPU controls (70B path) ---
    p.add_argument("--gpus", default=None,
                   help="comma GPU ids to expose (sets CUDA_VISIBLE_DEVICES before "
                        "torch import; e.g. --gpus 1 on the shared node)")
    p.add_argument("--big-model", choices=["auto", "on", "off"], default="auto",
                   help="CPU-load + SCT-on-GPU-SVD + accelerate dispatch. "
                        "'auto' turns on when the model name contains '70B'")
    p.add_argument("--max-gpu-mem", type=float, default=72.0,
                   help="GiB cap per visible GPU for dispatch (headroom for "
                        "activations/optimizer; 72 of 80 GiB default)")
    p.add_argument("--offload-dir", default=None,
                   help="disk offload dir if even CPU RAM is insufficient")
    p.add_argument("--svd", choices=["auto", "full", "lowrank"], default="auto",
                   help="SCT factorization: full SVD or adaptive randomized "
                        "(auto = randomized for 70B-class layers)")
    p.add_argument("--energies", type=float, nargs="+", default=[0.99, 0.97, 0.95, 0.90, 0.85],
                   help="SCT energies (dense baseline is always added)")
    p.add_argument("--kv-configs", default="none,4x4,3x4,3x2,2x2",
                   help="comma list; each 'KxV' bits or 'none' for fp16")
    p.add_argument("--lora", choices=["off", "on", "both"], default="both",
                   help="recovery-LoRA: off, on, or both (toggle)")
    p.add_argument("--kv-ref-tokens", type=int, default=1024,
                   help="reference context length for KV byte accounting")
    # eval subset sizes (the 'subset (fast sweep)' decision)
    p.add_argument("--ppl-tokens", type=int, default=2048)
    p.add_argument("--hellaswag", type=int, default=400)
    p.add_argument("--mmlu", type=int, default=400)
    p.add_argument("--truthfulqa", type=int, default=200)
    # recovery-LoRA hyperparams
    p.add_argument("--lora-steps", type=int, default=200)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--lora-samples", type=int, default=1000)
    p.add_argument("--lora-batch", type=int, default=4,
                   help="LoRA batch size (use 1-2 on the 70B)")
    p.add_argument("--lora-grad-checkpoint", action="store_true",
                   help="gradient checkpointing during recovery-LoRA (70B)")
    # utility weights
    p.add_argument("--weights", default=None,
                   help="comma 'ppl,hellaswag,mmlu,truthfulqa' (default equal 0.25 each)")
    p.add_argument("--quick", action="store_true",
                   help="smoke mode: 1 energy x 2 kv x lora-off, tiny subsets")
    p.add_argument("--dry-run", action="store_true",
                   help="integration smoke test only: model+SCT+TQ cache forward, "
                        "no datasets, no training. Run this FIRST on a new node.")
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    if args.dry_run:
        raise SystemExit(dry_run(args, dtype))

    results_dir = results_dir_for(args.model)
    os.makedirs(results_dir, exist_ok=True)

    # --- parse sweep axes ---
    def parse_kv(tok):
        if tok.lower() == "none":
            return (None, None)
        a, b = tok.lower().split("x")
        return (int(a), int(b))

    kv_configs = [parse_kv(t) for t in args.kv_configs.split(",")]
    energies = [None] + list(args.energies)        # None = dense baseline
    lora_modes = {"off": [False], "on": [True], "both": [False, True]}[args.lora]

    weights = DEFAULT_WEIGHTS
    if args.weights:
        w = [float(x) for x in args.weights.split(",")]
        weights = {"ppl": w[0], "hellaswag": w[1], "mmlu": w[2], "truthfulqa": w[3]}

    limits = {"ppl_tokens": args.ppl_tokens, "hellaswag": args.hellaswag,
              "mmlu": args.mmlu, "truthfulqa": args.truthfulqa}

    if args.quick:
        energies = [None, 0.95]
        kv_configs = [(None, None), (3, 2)]
        lora_modes = [False]
        limits = {"ppl_tokens": 256, "hellaswag": 20, "mmlu": 20, "truthfulqa": 10}
        args.lora_steps = 10

    print("=" * 78)
    print(f"  SCT x TurboQuant — Utility vs Compression Pareto ({args.model})")
    print("=" * 78)
    print(f"  model={args.model} device={args.device} dtype={args.dtype} "
          f"big={is_big(args)} gpus={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"  energies={energies}")
    print(f"  kv_configs={kv_configs}")
    print(f"  lora_modes={lora_modes}  subset_limits={limits}")
    print()

    all_points = []
    baseline_raw = None     # dense, fp16 KV, no LoRA -> normalization reference
    baseline_bytes = None
    t_start = time.time()

    for energy in energies:
        for lora_on in lora_modes:
            if energy is None and lora_on:
                continue  # recovery-LoRA on the dense model is not a "recovery" point

            tag = ("dense" if energy is None else f"E{energy:g}") + ("+LoRA" if lora_on else "")
            print(f"\n{'─'*78}\n  BUILD: {tag}\n{'─'*78}")

            model, tok, sct_stats, dev = build_point_model(args, dtype, energy)

            lora_info = None
            if lora_on:
                model, lora_info = train_recovery_lora(
                    model, tok, rank=args.lora_rank, lr=args.lora_lr,
                    steps=args.lora_steps, max_samples=args.lora_samples,
                    batch_size=args.lora_batch, device=dev, merge=True,
                    grad_checkpoint=args.lora_grad_checkpoint)
                model.eval()

            weight_bytes = model_weight_bytes(model)

            for key_bits, val_bits in kv_configs:
                kv_label = "fp16" if key_bits is None else f"{key_bits}x{val_bits}"
                label = f"{tag} / KV={kv_label}"
                print(f"  EVAL: {label} ...", flush=True)

                cache_factory = make_cache_factory(model, key_bits, val_bits)
                t0 = time.time()
                raw = evaluate_all(model, tok, cache_factory, dev, limits)
                kv_bytes = measure_kv_bytes(model, key_bits, val_bits, args.kv_ref_tokens)
                total_bytes = weight_bytes + kv_bytes

                # establish baseline at (dense, fp16, no-LoRA)
                if energy is None and key_bits is None and not lora_on:
                    baseline_raw = dict(raw)
                    baseline_bytes = total_bytes

                point = {
                    "label": label, "energy": energy, "lora": lora_on,
                    "key_bits": key_bits, "value_bits": val_bits, "kv_label": kv_label,
                    "raw": raw, "weight_bytes": weight_bytes, "kv_bytes": kv_bytes,
                    "total_bytes": total_bytes, "sct": sct_stats, "lora_info": lora_info,
                    "eval_sec": round(time.time() - t0, 1),
                }
                all_points.append(point)
                print(f"    ppl={raw['perplexity']:.2f} hs={raw['hellaswag']:.3f} "
                      f"mmlu={raw['mmlu']:.3f} tqa={raw['truthfulqa']:.3f} | "
                      f"wt={weight_bytes/1e9:.2f}GB kv={kv_bytes/1e6:.1f}MB | "
                      f"{point['eval_sec']:.0f}s")

            free(model)

    # --- normalize to baseline + utility ---
    assert baseline_raw is not None, "baseline (dense, fp16, no-LoRA) was not evaluated"
    for pt in all_points:
        agg = aggregate_utility(pt["raw"], baseline_raw, weights)
        pt["scores"] = agg["scores"]
        pt["utility"] = agg["utility"]
        pt["compression_ratio"] = compression_ratio(baseline_bytes, pt["total_bytes"])

    # --- save + plot ---
    results_path = os.path.join(results_dir, "pareto_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": {"model": args.model, "dtype": args.dtype, "weights": weights,
                       "limits": limits, "kv_ref_tokens": args.kv_ref_tokens,
                       "lora": {"steps": args.lora_steps, "rank": args.lora_rank,
                                "lr": args.lora_lr, "samples": args.lora_samples}},
            "baseline_raw": baseline_raw, "baseline_bytes": baseline_bytes,
            "points": all_points, "wall_sec": round(time.time() - t_start, 1),
        }, f, indent=2)
    print(f"\n  Saved -> {results_path}")

    front = pareto.pareto_frontier(all_points)
    pareto.plot_pareto(all_points, os.path.join(results_dir, "pareto_utility_vs_compression.png"),
                       f"Utility vs Compression — {args.model}")
    for lora_flag in set(p["lora"] for p in all_points):
        suffix = "lora" if lora_flag else "nolora"
        pareto.plot_energy_kv_heatmap(
            all_points, os.path.join(results_dir, f"heatmap_energy_kv_{suffix}.png"),
            f"Utility over SCT energy x KV bits ({'with' if lora_flag else 'no'} recovery-LoRA)",
            lora=lora_flag)

    # --- table ---
    print("\n" + "=" * 78)
    print(f"  {'Config':<28s} {'U':>6s} {'ratio':>6s} {'ppl':>7s} {'hs':>5s} "
          f"{'mmlu':>5s} {'tqa':>5s} {'front':>5s}")
    front_ids = {id(p) for p in front}
    for pt in sorted(all_points, key=lambda r: -r["utility"]):
        mark = "*" if id(pt) in front_ids else ""
        print(f"  {pt['label']:<28s} {pt['utility']:>6.3f} {pt['compression_ratio']:>6.2f} "
              f"{pt['raw']['perplexity']:>7.2f} {pt['raw']['hellaswag']:>5.2f} "
              f"{pt['raw']['mmlu']:>5.2f} {pt['raw']['truthfulqa']:>5.2f} {mark:>5s}")
    print(f"\n  Pareto frontier: {len(front)} points (*)  |  wall {time.time()-t_start:.0f}s")
    print("  Plots + JSON ->", results_dir)


if __name__ == "__main__":
    main()
