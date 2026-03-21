"""
Run early_vs_late_learning.py for all Experiment 1 subject pairs.

Subjects 1-20, excluding 3, 4, 16 (incomplete BOLD data).
Iterates over all unique pairs, saves per-pair JSON results,
then aggregates into a single heatmap-ready JSON file.
"""

import subprocess
import sys
import os
import json
import itertools
import time

EXCLUDED = {3, 4, 16}
SUBJECTS = [f"sub-Exp1s{i:02d}" for i in range(1, 21) if i not in EXCLUDED]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REGION = "hippocampus"

pairs = list(itertools.combinations(SUBJECTS, 2))
total = len(pairs)

print(f"Subjects ({len(SUBJECTS)}): {', '.join(SUBJECTS)}")
print(f"Total pairs: {total}\n")

failed = []
for idx, (sub_a, sub_b) in enumerate(pairs, 1):
    print(f"\n{'#'*60}")
    print(f"  Pair {idx}/{total}: {sub_a}  x  {sub_b}")
    print(f"{'#'*60}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "early_vs_late_learning.py",
         "--subject-a", sub_a, "--subject-b", sub_b],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  *** FAILED: {sub_a} x {sub_b} (exit code {result.returncode}) ***")
        failed.append((sub_a, sub_b))
    else:
        print(f"\n  Completed in {elapsed:.1f}s")

# ── Aggregate per-pair JSONs into a heatmap-ready summary ──
print(f"\n{'='*60}")
print("Aggregating results for heatmap...")

heatmap = {
    "subjects": SUBJECTS,
    "region": REGION,
    "a_to_b_change": {},
    "b_to_a_change": {},
    "a_to_b_early": {},
    "a_to_b_late": {},
    "b_to_a_early": {},
    "b_to_a_late": {},
}

for sub_a, sub_b in pairs:
    pair_file = os.path.join(
        RESULTS_DIR, f"early_vs_late_{sub_a}_{sub_b}_{REGION}.json"
    )
    if not os.path.exists(pair_file):
        continue

    with open(pair_file) as f:
        r = json.load(f)

    key = f"{sub_a}__{sub_b}"
    heatmap["a_to_b_change"][key] = r["a_to_b"]["change"]
    heatmap["b_to_a_change"][key] = r["b_to_a"]["change"]
    heatmap["a_to_b_early"][key] = r["a_to_b"]["early"]["median_r"]
    heatmap["a_to_b_late"][key] = r["a_to_b"]["late"]["median_r"]
    heatmap["b_to_a_early"][key] = r["b_to_a"]["early"]["median_r"]
    heatmap["b_to_a_late"][key] = r["b_to_a"]["late"]["median_r"]

heatmap_file = os.path.join(RESULTS_DIR, "exp1_heatmap.json")
with open(heatmap_file, "w") as f:
    json.dump(heatmap, f, indent=2)

print(f"Heatmap data saved to {heatmap_file}")
print(f"\n{'='*60}")
print(f"ALL DONE: {total - len(failed)}/{total} pairs succeeded")
if failed:
    print(f"\nFailed pairs ({len(failed)}):")
    for a, b in failed:
        print(f"  {a}  x  {b}")
print(f"{'='*60}")
