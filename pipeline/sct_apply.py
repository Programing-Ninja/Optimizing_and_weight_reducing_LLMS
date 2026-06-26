"""
sct_apply.py — apply Spectral Compact Training to a Llama model's MLP layers.

SCT replaces each MLP nn.Linear (W: [out, in]) with a SpectralLinear factored as
W = U diag(s) V^T, keeping only the top-k singular directions where k is chosen so
the retained spectral *energy* (sum of top-k squared singular values / total) meets
a threshold. Lower energy -> smaller k -> more weight compression -> more accuracy
loss (to be recovered by recovery-LoRA and/or measured directly).

The packaged SpectralLinear.from_linear() runs SVD on CPU, which is far too slow for
an 8B model. We replicate its exact factor convention but run the SVD on the weight's
own device (the GPU), in float32, then cast factors back to the model dtype.

Convention (must match SpectralLinear in spectral_compact_training/spectral_layer.py):
    SVD:   W = U_full @ diag(S) @ Vh_full          (W is [out, in])
    Ours:  y = (x @ U) * s @ V.T
    =>     U = Vh[:k].T   (shape [in, k])
           V = U_full[:, :k]  (shape [out, k])
           s = S[:k]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectral_compact_training import SpectralLinear  # real package API

# Llama / LLaMA-family MLP projection leaf names.
MLP_LEAF_NAMES = frozenset(["gate_proj", "up_proj", "down_proj"])


def energy_to_rank(singular_values: torch.Tensor, energy: float) -> int:
    """Smallest k whose cumulative squared-singular-value energy >= `energy`."""
    s2 = singular_values.double() ** 2
    total = s2.sum()
    if total <= 0:
        return 1
    cumulative = torch.cumsum(s2, dim=0) / total
    idx = (cumulative >= energy).nonzero(as_tuple=True)[0]
    k = int(idx[0].item()) + 1 if len(idx) else len(singular_values)
    return max(k, 1)


@torch.no_grad()
def _spectral_from_linear_on_device(linear: nn.Linear, energy: float):
    """Factor a dense nn.Linear into a SpectralLinear on the weight's device.

    Returns (spectral_layer, k, energy_retained, dense_params).
    """
    W = linear.weight.data
    dev, model_dtype = W.device, W.dtype
    m, n = W.shape  # [out, in]

    Wf = W.float()  # SVD in float32 for stability
    U_full, S_full, Vh_full = torch.linalg.svd(Wf, full_matrices=False)

    k = min(energy_to_rank(S_full, energy), m, n)
    energy_retained = float((S_full[:k] ** 2).sum() / (S_full ** 2).sum())

    # Build SpectralLinear without re-running SVD (bypass __init__'s random init).
    layer = SpectralLinear.__new__(SpectralLinear)
    nn.Module.__init__(layer)
    layer.in_features = n
    layer.out_features = m
    layer.rank = k
    layer.U = nn.Parameter(Vh_full[:k, :].T.contiguous().to(model_dtype))
    layer.V = nn.Parameter(U_full[:, :k].contiguous().to(model_dtype))
    layer.s = nn.Parameter(S_full[:k].contiguous().to(model_dtype))
    layer = layer.to(dev)

    if linear.bias is not None:
        # Llama proj layers are bias-free; keep a fallback just in case.
        layer.bias = nn.Parameter(linear.bias.data.clone())

    dense_params = m * n + (m if linear.bias is not None else 0)
    return layer, k, energy_retained, dense_params


@torch.no_grad()
def apply_sct(model: nn.Module, energy: float | None, verbose: bool = True) -> dict:
    """Replace MLP nn.Linear layers with SpectralLinear at the given energy.

    energy=None is a no-op (returns the dense baseline stats). Returns a dict with
    per-run compression statistics.
    """
    if energy is None:
        dense_mlp = 0
        for name, mod in model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if isinstance(mod, nn.Linear) and leaf in MLP_LEAF_NAMES:
                dense_mlp += mod.weight.numel()
        return {
            "energy": None,
            "n_replaced": 0,
            "mlp_dense_params": dense_mlp,
            "mlp_spectral_params": dense_mlp,
            "mlp_ratio": 1.0,
            "ranks": [],
            "energy_retained_mean": 1.0,
        }

    targets = []
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(mod, nn.Linear) and leaf in MLP_LEAF_NAMES:
            targets.append(name)

    total_dense, total_spectral, ranks, retained = 0, 0, [], []
    name_to_module = dict(model.named_modules())

    for name in targets:
        module = name_to_module[name]
        spec, k, er, dense_params = _spectral_from_linear_on_device(module, energy)
        parent_name, child_name = name.rsplit(".", 1)
        setattr(name_to_module[parent_name], child_name, spec)

        total_dense += dense_params
        total_spectral += spec.param_count()
        ranks.append(k)
        retained.append(er)

    ratio = total_dense / max(total_spectral, 1)
    if verbose:
        print(f"  [SCT] energy={energy}: replaced {len(targets)} MLP linears | "
              f"MLP params {total_dense:,} -> {total_spectral:,} ({ratio:.2f}x) | "
              f"mean rank {sum(ranks)/max(len(ranks),1):.0f}")

    # Free the SVD work buffers.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "energy": energy,
        "n_replaced": len(targets),
        "mlp_dense_params": total_dense,
        "mlp_spectral_params": total_spectral,
        "mlp_ratio": ratio,
        "ranks": ranks,
        "energy_retained_mean": sum(retained) / max(len(retained), 1),
    }
