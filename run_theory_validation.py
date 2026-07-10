#!/usr/bin/env python3
"""
run_theory_validation.py — validate the theory-track models against a measured
LLM sweep (the LLM-scale analogue of theory/run_experiments.py).

    python run_theory_validation.py results_llama-31-70b/pareto_results.json
    python run_theory_validation.py                 # newest results_*/pareto_results.json

Consumes pareto_results.json (produced by run_pareto.py), re-tests the four toy
claims (SCT concavity, TQ exponent p, additivity, LoRA recovery) at LLM scale,
and runs the Part A rate–distortion solver on the MEASURED curves, comparing its
predicted (η*, b*) with the best measured sweep point per byte budget.

Outputs (next to the input JSON): theory_validation.{md,png,json}
"""

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def newest_results() -> str | None:
    cands = glob.glob(os.path.join(HERE, "results_*", "pareto_results.json"))
    return max(cands, key=os.path.getmtime) if cands else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="?", default=None,
                   help="path to pareto_results.json (default: newest results_*/)")
    p.add_argument("--out-dir", default=None,
                   help="output dir (default: alongside the input JSON)")
    args = p.parse_args()

    path = args.results or newest_results()
    if path is None or not os.path.exists(path):
        sys.exit("No pareto_results.json found — run ./run.sh full (or full70b) first.")

    print(f"Validating theory against: {path}")
    from pipeline.theory_validate import run
    run(path, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
