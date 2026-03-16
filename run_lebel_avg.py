#!/usr/bin/env python3
"""
Run the story-averaged LeBel 2023 benchmark for one model and one layer.

Usage (on FarmShare, with conda env bbscore active and env vars set):

  python run_lebel_avg.py --model llama3_8b_base --layer 0
  python run_lebel_avg.py --model llama3_8b_instruct --layer 16

Results are saved to $RESULTS_PATH/results/ (or $SCIKIT_LEARN_DATA/results/)
as {model}_{layer_name}_LeBel2023Benchmark.pkl.

Requires: HF_TOKEN (and HF_HOME if needed), SCIKIT_LEARN_DATA or RESULTS_PATH
pointing to data root (e.g. ~/scratch/bbscore_data).
"""

import argparse
import sys
import os

# Paths: run from bbscore_public (script_dir) or from parent repo
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_script_dir)
for p in (_script_dir, _parent):
    if p and p not in sys.path:
        sys.path.insert(0, p)

# Layer name format expected by Llama-3 hook (model.layers.N)
def layer_name_for_index(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}"


def main():
    parser = argparse.ArgumentParser(
        description="Run LeBel2023Benchmark (story-averaged) for one layer."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3_8b_base",
        help="Model identifier (e.g. llama3_8b_base, llama3_8b_instruct)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index (0..30; 31 excluded)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="UTS01",
        help="Subject ID",
    )
    parser.add_argument(
        "--n_cv_folds",
        type=int,
        default=5,
        help="Number of CV folds (story-level)",
    )
    args = parser.parse_args()

    if args.layer > 30:
        print("Warning: layer 31 is typically excluded; use 0..30.", file=sys.stderr)

    layer_name = layer_name_for_index(args.layer)

    # Import directly from file to avoid triggering benchmarks/__init__.py
    # (which imports NSD/BBS and requires the full bbscore environment to be loaded)
    import importlib.util
    _bench_path = os.path.join(_script_dir, "benchmarks", "LeBel2023", "LeBel2023Benchmark.py")
    _spec = importlib.util.spec_from_file_location("LeBel2023Benchmark", _bench_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    LeBel2023Benchmark = _mod.LeBel2023Benchmark

    bench = LeBel2023Benchmark(
        model_identifier=args.model,
        layer_name=layer_name,
        subject_id=args.subject,
        n_cv_folds=args.n_cv_folds,
    )
    bench.run()


if __name__ == "__main__":
    main()
