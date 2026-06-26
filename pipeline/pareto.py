"""
pareto.py — Pareto frontier over (utility, compression ratio) plus two figures:
  1. scatter of utility vs compression ratio, frontier highlighted, LoRA off/on
     drawn distinctly so the recovery gain is visible.
  2. heatmap of utility over the (SCT energy x KV-bits) grid — this is what answers
     the two-way question (how energy changes the tolerable quant level & vice-versa).
"""

from __future__ import annotations

import math
from typing import List, Dict


def pareto_frontier(points: List[Dict]) -> List[Dict]:
    """Non-dominated points maximizing utility AND compression_ratio."""
    valid = [p for p in points
             if not (math.isnan(p.get("utility", float("nan")))
                     or math.isnan(p.get("compression_ratio", float("nan"))))]
    front = []
    for p in valid:
        dominated = False
        for q in valid:
            if q is p:
                continue
            if (q["utility"] >= p["utility"] and q["compression_ratio"] >= p["compression_ratio"]
                    and (q["utility"] > p["utility"] or q["compression_ratio"] > p["compression_ratio"])):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda r: r["compression_ratio"])


def plot_pareto(points: List[Dict], out_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    front = pareto_frontier(points)
    front_ids = {id(p) for p in front}

    fig, ax = plt.subplots(figsize=(10, 7))
    for p in points:
        if math.isnan(p.get("utility", float("nan"))):
            continue
        on_front = id(p) in front_ids
        lora = p.get("lora", False)
        color = "crimson" if on_front else ("seagreen" if lora else "steelblue")
        marker = "*" if on_front else ("^" if lora else "o")
        size = 220 if on_front else 70
        ax.scatter(p["compression_ratio"], p["utility"], c=color, marker=marker,
                   s=size, edgecolors="k", linewidths=0.4, zorder=5 if on_front else 3)
        ax.annotate(p.get("label", ""), (p["compression_ratio"], p["utility"]),
                    fontsize=6, ha="left", va="bottom", alpha=0.8)

    if front:
        fx = [p["compression_ratio"] for p in front]
        fy = [p["utility"] for p in front]
        ax.plot(fx, fy, "r--", lw=1.2, alpha=0.7, label="Pareto frontier")

    ax.set_xlabel("Compression ratio  (dense-fp16 bytes / total bytes,  higher = smaller)")
    ax.set_ylabel("Utility U  (1.0 = dense baseline)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_energy_kv_heatmap(points: List[Dict], out_path: str, title: str, lora: bool = False):
    """Heatmap of utility over energy (rows) x KV-config (cols), filtered by LoRA flag."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pts = [p for p in points if p.get("lora", False) == lora]
    energies = sorted({p["energy"] if p["energy"] is not None else 1.0 for p in pts}, reverse=True)
    kvs = sorted({p["kv_label"] for p in pts})

    grid = np.full((len(energies), len(kvs)), np.nan)
    for p in pts:
        e = p["energy"] if p["energy"] is not None else 1.0
        r = energies.index(e)
        c = kvs.index(p["kv_label"])
        grid[r, c] = p["utility"]

    fig, ax = plt.subplots(figsize=(1.4 * len(kvs) + 3, 1.0 * len(energies) + 2))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(kvs)), kvs, rotation=30, ha="right")
    ax.set_yticks(range(len(energies)), [f"{e:g}" for e in energies])
    ax.set_xlabel("KV quantization (key_bits, value_bits)")
    ax.set_ylabel("SCT energy retained")
    ax.set_title(title)
    for r in range(len(energies)):
        for c in range(len(kvs)):
            if not np.isnan(grid[r, c]):
                ax.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center",
                        color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="Utility U")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
