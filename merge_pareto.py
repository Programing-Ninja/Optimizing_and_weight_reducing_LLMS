#!/usr/bin/env python3
"""
merge_pareto.py — append new points from a standalone run_pareto.py output into
the canonical results_llama8b/pareto_results.json, recomputing utility and
compression_ratio against the CANONICAL file's own baseline (not the new run's,
which may not have one when it was launched with --lora on alone).

Usage:
  python merge_pareto.py --new <new_points.json> --relabel-suffix "(steps=1600)"

`--new` may be a full run_pareto.py output (uses its "points" list) or a bare
list of point dicts. Points are matched into the canonical grid by label; if
--relabel-suffix is given it's inserted into "<tag>+LoRA<suffix> / KV=<kv>"
before merging, so points that would otherwise collide with an existing label
(same energy/lora/kv but a different LoRA step budget) stay distinct.
"""
import argparse
import json
import os

from pipeline.utility import aggregate_utility
from pipeline.compression import compression_ratio

HERE = os.path.dirname(os.path.abspath(__file__))
CANON = os.path.join(HERE, "results_llama8b", "pareto_results.json")


def relabel(label: str, suffix: str) -> str:
    # "E0.85+LoRA / KV=fp16" -> "E0.85+LoRA(steps=1600) / KV=fp16"
    lora_marker = "+LoRA"
    if lora_marker in label and suffix:
        return label.replace(lora_marker, lora_marker + suffix, 1)
    return label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new", required=True, help="path to the new run's json")
    p.add_argument("--relabel-suffix", default="", help="e.g. '(steps=1600)'")
    p.add_argument("--out", default=CANON)
    args = p.parse_args()

    with open(CANON) as f:
        canon = json.load(f)
    with open(args.new) as f:
        new_data = json.load(f)

    baseline_raw = canon["baseline_raw"]
    baseline_bytes = canon["baseline_bytes"]
    weights = canon["config"]["weights"]

    new_points = new_data["points"] if isinstance(new_data, dict) else new_data
    existing_labels = {pt["label"] for pt in canon["points"]}

    added = []
    for pt in new_points:
        pt = dict(pt)
        pt["label"] = relabel(pt["label"], args.relabel_suffix)
        if pt["label"] in existing_labels:
            raise SystemExit(f"label collision, refusing to merge: {pt['label']}")
        agg = aggregate_utility(pt["raw"], baseline_raw, weights)
        pt["scores"] = agg["scores"]
        pt["utility"] = agg["utility"]
        pt["compression_ratio"] = compression_ratio(baseline_bytes, pt["total_bytes"])
        canon["points"].append(pt)
        existing_labels.add(pt["label"])
        added.append(pt["label"])

    canon["wall_sec"] = canon.get("wall_sec", 0) + new_data.get("wall_sec", 0) if isinstance(new_data, dict) else canon.get("wall_sec", 0)
    canon["complete"] = True

    with open(args.out, "w") as f:
        json.dump(canon, f, indent=2)

    print(f"merged {len(added)} point(s) -> {args.out} (grid now {len(canon['points'])} points)")
    for lbl in added:
        print("  +", lbl)


if __name__ == "__main__":
    main()
