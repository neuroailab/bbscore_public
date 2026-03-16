#!/usr/bin/env python3
"""
Run the TR-level LeBel 2023 benchmark for one model and one layer.
Uses sparse random projection (1024 or 512 dims) to reduce memory.
After git pull, the pipeline uses thresholded voxels (~26k for UTS01).

Usage (on FarmShare, with conda env bbscore active):

  python run_lebel_tr.py --model llama3_8b_base --layer 16
  python run_lebel_tr.py --model llama3_8b_base --layer 16 --projection-dim 512

Requires: HF_TOKEN, SCIKIT_LEARN_DATA, and enough memory (e.g. 96G).
Delete any existing ridge checkpoint for this layer if switching to/from projection.
"""

import argparse
import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_script_dir)
for p in (_script_dir, _parent):
    if p and p not in sys.path:
        sys.path.insert(0, p)


def main():
    parser = argparse.ArgumentParser(
        description="Run LeBel2023 TR-level benchmark with sparse projection."
    )
    parser.add_argument("--model", type=str, default="llama3_8b_base")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--subject", type=str, default="UTS01")
    parser.add_argument("--projection-dim", type=int, default=1024,
                        help="1024 or 512; use 512 if 1024 still OOMs")
    args = parser.parse_args()

    layer_name = f"model.layers.{args.layer}"

    import importlib.util
    _bench_path = os.path.join(
        _script_dir, "benchmarks", "LeBel2023", "LeBel2023TRBenchmark.py"
    )
    _spec = importlib.util.spec_from_file_location(
        "LeBel2023TRBenchmark", _bench_path
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    LeBel2023TRBenchmark = _mod.LeBel2023TRBenchmark

    bench = LeBel2023TRBenchmark(
        model_identifier=args.model,
        layer_name=layer_name,
        subject_id=args.subject,
        random_projection="sparse",
        projection_dim=args.projection_dim,
    )
    bench.run()


if __name__ == "__main__":
    main()
