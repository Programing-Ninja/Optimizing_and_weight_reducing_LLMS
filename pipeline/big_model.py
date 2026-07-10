"""
big_model.py — loading + device placement for models too big for one GPU.

The 70B path: Llama-3.1-70B is ~140GB in bf16 and does not fit on a single
A100 80GB. The strategy is:

  1. load the checkpoint on CPU (low_cpu_mem_usage streams shards, so peak CPU
     RAM ~= model size, not 2x),
  2. apply SCT while the model is on CPU, running each layer's SVD on the GPU
     (pipeline/sct_apply.py `svd_device`) — this SHRINKS the model before we
     decide placement, so more (often all) of it fits on the GPU,
  3. `dispatch_big()` — accelerate `infer_auto_device_map` + `dispatch_model`
     with a per-GPU `max_memory` cap; decoder layers that don't fit stay on CPU
     and are streamed through accelerate's AlignDevicesHook at forward time.

GPU pinning: CUDA_VISIBLE_DEVICES must be set BEFORE torch initializes CUDA, so
`pin_gpus_from_argv()` is called at the very top of run_pareto.py, before any
`import torch`. On the shared HPC node use `--gpus 1` (GPU 0 is in use); when
both A100s are free, `--gpus 0,1` splits the dense model across them.
"""

from __future__ import annotations

import os
import sys


def pin_gpus_from_argv(argv=None) -> str | None:
    """Scan argv for `--gpus <ids>` and set CUDA_VISIBLE_DEVICES accordingly.

    MUST run before torch is imported. Returns the pinned string or None.
    An explicit CUDA_VISIBLE_DEVICES already in the environment wins only if
    --gpus is absent.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    gpus = None
    for i, a in enumerate(argv):
        if a == "--gpus" and i + 1 < len(argv):
            gpus = argv[i + 1]
        elif a.startswith("--gpus="):
            gpus = a.split("=", 1)[1]
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    return gpus


def is_dispatched(model) -> bool:
    """True if the model has an accelerate device map (never .to() such a model)."""
    return bool(getattr(model, "hf_device_map", None))


def input_device(model) -> str:
    """Device where input_ids must land: the embedding layer's device."""
    try:
        emb = model.get_input_embeddings()
        dev = emb.weight.device
        if dev.type != "meta":
            return str(dev)
    except Exception:
        pass
    return str(next(model.parameters()).device)


def load_model(model_name: str, dtype, big: bool, device: str = "cuda"):
    """Load tokenizer + model.

    big=False: legacy path — load and move wholly to `device` (8B on A100).
    big=True : load on CPU only (placement happens later via dispatch_big()).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if not big:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, attn_implementation="eager")
        model.to(device)
        model.eval()
        return model, tok

    print(f"  [big] loading {model_name} on CPU (low_cpu_mem_usage) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="eager",
        low_cpu_mem_usage=True, device_map=None)
    model.eval()
    return model, tok


def dispatch_big(model, max_gpu_mem_gib: float = 72.0, cpu_mem_gib: float | None = None,
                 offload_dir: str | None = None, verbose: bool = True):
    """Place the (possibly SCT-compressed) model across visible GPUs + CPU.

    max_gpu_mem_gib caps EACH visible GPU (leave headroom for activations, the
    fp32 attention math of eager mode, and LoRA optimizer state — 72 of 80 GiB
    is a sane default). Layers that don't fit stay on CPU and are streamed at
    forward time. Returns (model, device_map).
    """
    import torch
    from accelerate import dispatch_model, infer_auto_device_map

    if not torch.cuda.is_available():
        if verbose:
            print("  [big] no CUDA visible — model stays on CPU")
        return model, {"": "cpu"}

    n_gpu = torch.cuda.device_count()
    max_memory = {i: f"{max_gpu_mem_gib:.0f}GiB" for i in range(n_gpu)}
    max_memory["cpu"] = f"{cpu_mem_gib:.0f}GiB" if cpu_mem_gib else "512GiB"

    # Never split a decoder layer across devices.
    no_split = getattr(model, "_no_split_modules", None) or ["LlamaDecoderLayer"]

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split,
        dtype=next(model.parameters()).dtype)

    kwargs = {}
    if any(str(v) == "disk" for v in device_map.values()):
        kwargs["offload_dir"] = offload_dir or "offload_tmp"
        os.makedirs(kwargs["offload_dir"], exist_ok=True)

    model = dispatch_model(model, device_map=device_map, **kwargs)

    if verbose:
        placements = {}
        for dev in device_map.values():
            placements[str(dev)] = placements.get(str(dev), 0) + 1
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"  [big] dispatched: {placements} modules | weights "
              f"{param_bytes/2**30:.1f} GiB | max_memory={max_memory}", flush=True)
        if "cpu" in placements or "disk" in placements:
            print("  [big] NOTE: part of the model is CPU/disk-offloaded — "
                  "forwards will be PCIe-bound and slow. This is expected for "
                  "the dense 70B baseline; SCT points should fit on-GPU.", flush=True)
    return model, device_map
