#!/usr/bin/env python3
"""Render the SCT hyperparameter-tuning sweep results as a single figure.

Reads the JSON written by sct_hp_tune.py and produces a sorted horizontal bar
chart of post-finetune perplexity for every config, with the best config
highlighted and each bar annotated with its compression ratio.

Usage:
    experiments/run.sh experiments/plot_hp_tune.py \
        [results.json] [out.png]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "sct_hp_tune_SmolLM2-135M.json"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else in_path.with_suffix(".png")

    data = json.loads(in_path.read_text())
    results = [r for r in data["results"] if r.get("error") is None]
    # Best = lowest post-finetune perplexity.
    results.sort(key=lambda r: r["post_ft_ppl"], reverse=True)
    best_label = data["best"]["label"]

    labels = [r["label"] for r in results]
    ppls = [r["post_ft_ppl"] for r in results]
    energies = [r["energy"] for r in results]
    comps = [r["compression"] for r in results]

    # Color by energy threshold; emphasize the best config.
    palette = {0.95: "#2a7ae2", 0.9: "#9bbce6"}
    colors = [palette.get(e, "#bbbbbb") for e in energies]
    for i, lab in enumerate(labels):
        if lab == best_label:
            colors[i] = "#e2562a"

    fig, ax = plt.subplots(figsize=(11, 6))
    ypos = range(len(labels))
    bars = ax.barh(list(ypos), ppls, color=colors, edgecolor="#333", linewidth=0.5)

    for bar, ppl, comp in zip(bars, ppls, comps):
        ax.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"ppl {ppl:.1f}  |  {comp:.2f}x params",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Post-finetune perplexity (log scale, lower is better)")
    ax.set_xlim(right=max(ppls) * 1.9)
    ax.set_title(
        f"SCT HP-tuning sweep — {data['model']}\n"
        f"300 finetune steps/config · best = {best_label} (ppl {data['best']['post_ft_ppl']:.1f})",
        fontsize=11,
    )

    legend = [
        plt.Rectangle((0, 0), 1, 1, color="#e2562a"),
        plt.Rectangle((0, 0), 1, 1, color="#2a7ae2"),
        plt.Rectangle((0, 0), 1, 1, color="#9bbce6"),
    ]
    ax.legend(legend, ["best config", "energy 0.95", "energy 0.90"], loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
