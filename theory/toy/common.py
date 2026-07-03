"""Shared utilities for the toy arms: tee-logging, fits, seeding, device."""

from __future__ import annotations

import os
import sys
import time
import json
import numpy as np
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class Tee:
    """Logger that writes timestamped lines to stdout AND a file."""

    def __init__(self, path: str):
        self.f = open(path, "a")
        self.f.write(f"\n\n{'='*78}\nRUN @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*78}\n")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def fit_through_origin(x, y):
    """Least-squares slope for y ≈ k·x (no intercept). Returns (slope, R2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    k = float((x * y).sum() / (x * x).sum())
    ss_res = float(((y - k * x) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return k, r2


def fit_affine(x, y):
    """Least-squares y ≈ a·x + b. Returns (a, b, R2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = a * x + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), r2


def fit_power(x, y):
    """Least-squares y ≈ A·x^p (log-log linear). Returns (A, p, R2) where R2 is
    measured in LINEAR space (so it's comparable to the through-origin fit)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = (x > 0) & (y > 0)
    p, logA = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    A = float(np.exp(logA))
    pred = A * x[m] ** p
    ss_res = float(((y[m] - pred) ** 2).sum())
    ss_tot = float(((y[m] - y[m].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return A, float(p), r2


def save_json(name: str, obj: dict):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return path
