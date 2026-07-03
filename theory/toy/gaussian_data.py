"""
Gaussian token data + teacher for the toy validation experiment (Part B.1).
================================================================================

We deliberately keep the data *analytically tractable*:

  * Tokens x ∈ R^{d_in} are drawn i.i.d. Gaussian  x ~ N(0, Σ_x).
      - `iso`  : Σ_x = I               (flat input spectrum)
      - `aniso`: Σ_x = diag(λ_i)       (controlled spectral decay λ_i = ρ^i)
    The anisotropic case is what lets us exercise the *curvature alignment*
    term in A.1 (the loss cares about discarded directions weighted by H, not
    just by raw singular energy).

  * A single sequence is a matrix X ∈ R^{T×d_in} of T such tokens.

  * The teacher is a *fixed* linear-attention network of the SAME architecture
    as the student (see linear_attn_model.py). Using a same-architecture teacher
    makes the regression *realizable*: the population-optimal student weights are
    the teacher weights, so after training the student sits (near) a true loss
    minimum — exactly the regime where A.1's "loss is locally quadratic in ΔW"
    assumption (grad ≈ 0, quadratic curvature dominates) holds by construction.

Everything here is plain, seedable, and CPU/GPU agnostic.
"""

from __future__ import annotations

import torch


def make_input_covariance(d_in: int, kind: str = "iso", decay: float = 0.9,
                          device=None, dtype=torch.float32) -> torch.Tensor:
    """Return Σ_x (d_in × d_in), and its matrix square-root is taken by caller.

    kind='iso'   -> identity (flat spectrum).
    kind='aniso' -> diag(decay**i), i=0..d_in-1 (geometric spectral decay).
    """
    if kind == "iso":
        return torch.eye(d_in, device=device, dtype=dtype)
    elif kind == "aniso":
        i = torch.arange(d_in, device=device, dtype=dtype)
        lam = decay ** i
        lam = lam / lam.mean()  # normalise so tr(Σ_x)=d_in (comparable scale to iso)
        return torch.diag(lam)
    raise ValueError(f"unknown Σ_x kind: {kind!r}")


class GaussianTokenData:
    """Streaming generator of Gaussian token sequences X ∈ R^{T×d_in}."""

    def __init__(self, d_in: int, T: int, sigma_kind: str = "iso",
                 decay: float = 0.9, seed: int = 0, device=None,
                 dtype=torch.float32):
        self.d_in = d_in
        self.T = T
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

        Sigma = make_input_covariance(d_in, sigma_kind, decay, device="cpu", dtype=dtype)
        self.Sigma_x = Sigma.to(self.device)
        # symmetric sqrt so that  x = L z,  z~N(0,I)  gives Cov(x)=Σ_x
        evals, evecs = torch.linalg.eigh(Sigma)
        self.L = (evecs @ torch.diag(evals.clamp_min(0).sqrt()) @ evecs.T).to(self.device)

    def sample(self, batch: int) -> torch.Tensor:
        """Return X of shape (batch, T, d_in) with rows ~ N(0, Σ_x)."""
        z = torch.randn(batch, self.T, self.d_in, generator=self.gen, dtype=self.dtype)
        z = z.to(self.device)
        return z @ self.L.T  # apply Σ_x^{1/2}


if __name__ == "__main__":
    # quick self-check: empirical covariance should match Σ_x
    d = 16
    data = GaussianTokenData(d_in=d, T=8, sigma_kind="aniso", decay=0.8, seed=1)
    X = data.sample(20000).reshape(-1, d)
    emp = (X.T @ X) / X.shape[0]
    err = (emp - data.Sigma_x).abs().max().item()
    print(f"[gaussian_data] max|emp Σ_x - Σ_x| = {err:.4f}  (should be small)")
