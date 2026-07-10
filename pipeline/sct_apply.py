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


def energy_to_rank_with_total(singular_values: torch.Tensor, energy: float,
                              total_energy: float) -> int:
    """Like energy_to_rank, but with the exact total Σσ² = ‖W‖_F² supplied
    (needed when only the top of the spectrum was computed via svd_lowrank)."""
    s2 = singular_values.double() ** 2
    if total_energy <= 0:
        return 1
    cumulative = torch.cumsum(s2, dim=0) / total_energy
    idx = (cumulative >= energy).nonzero(as_tuple=True)[0]
    k = int(idx[0].item()) + 1 if len(idx) else len(singular_values)
    return max(k, 1)


@torch.no_grad()
def _adaptive_lowrank_svd(Wf: torch.Tensor, energy: float, q0: int = 256, niter: int = 4):
    """Randomized top-q SVD, growing the sketch size q until the retained
    spectral energy meets `energy`. The energy DENOMINATOR is exact —
    Σσ_i² = ‖W‖_F² — so the threshold is honest even though only the top of the
    spectrum is computed. Needed at 70B scale: a full fp32 SVD of one 8192×28672
    matrix takes minutes on an A100, ×240 matrices ×6 energies = days;
    svd_lowrank at the ranks the sweep actually uses takes seconds.

    Returns (U, S, Vh, total_energy) with >= the rank needed for `energy`
    (unless that exceeds min(m,n), where it saturates)."""
    m, n = Wf.shape
    min_dim = min(m, n)
    total = float((Wf.double() ** 2).sum())
    q = min(q0, min_dim)
    while True:
        U, S, V = torch.svd_lowrank(Wf, q=q, niter=niter)
        retained = float((S.double() ** 2).sum()) / max(total, 1e-30)
        if retained >= energy or q >= min_dim:
            return U, S, V.T, total
        q = min(q * 2, min_dim)  # sketch too small for the target — double it


@torch.no_grad()
def _spectral_from_linear_on_device(linear: nn.Linear, energy: float,
                                    svd_method: str = "auto",
                                    svd_device: str | None = None,
                                    lowrank_threshold: int = 4096):
    """Factor a dense nn.Linear into a SpectralLinear, keeping the factors on the
    weight's ORIGINAL device (a CPU-resident 70B stays on CPU) while running the
    SVD itself on `svd_device` (the GPU) when given.

    svd_method: "full" = exact torch.linalg.svd; "lowrank" = adaptive randomized;
    "auto" = lowrank when min(m,n) >= lowrank_threshold (i.e. 70B-class layers).

    Returns (spectral_layer, k, energy_retained, dense_params).
    """
    W = linear.weight.data
    dev, model_dtype = W.device, W.dtype
    m, n = W.shape  # [out, in]

    Wf = W.float()  # SVD in float32 for stability
    if svd_device is not None and str(Wf.device) != str(svd_device):
        Wf = Wf.to(svd_device)

    use_lowrank = (svd_method == "lowrank" or
                   (svd_method == "auto" and min(m, n) >= lowrank_threshold))
    if use_lowrank:
        U_full, S_full, Vh_full, total_energy = _adaptive_lowrank_svd(Wf, energy)
        k = min(energy_to_rank_with_total(S_full, energy, total_energy), m, n)
        energy_retained = float((S_full[:k].double() ** 2).sum() / max(total_energy, 1e-30))
    else:
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
def apply_sct(model: nn.Module, energy: float | None, verbose: bool = True,
              svd_method: str = "auto", svd_device: str | None = None,
              log_every: int = 24) -> dict:
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

    import time as _time
    t0 = _time.time()
    for i, name in enumerate(targets):
        module = name_to_module[name]
        spec, k, er, dense_params = _spectral_from_linear_on_device(
            module, energy, svd_method=svd_method, svd_device=svd_device)
        parent_name, child_name = name.rsplit(".", 1)
        setattr(name_to_module[parent_name], child_name, spec)
        # Drop the dense weight immediately — at 70B, keeping both dense and
        # factored copies alive across 240 layers would blow CPU RAM.
        module.weight = None
        del module

        total_dense += dense_params
        total_spectral += spec.param_count()
        ranks.append(k)
        retained.append(er)
        if verbose and log_every and (i + 1) % log_every == 0:
            rate = (i + 1) / max(_time.time() - t0, 1e-9)
            print(f"  [SCT] {i+1}/{len(targets)} layers factored "
                  f"({rate:.1f}/s, ETA {(len(targets)-i-1)/rate:.0f}s)", flush=True)

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
