import json
from pathlib import Path

results_dir = Path("results")
files = sorted(results_dir.glob("early_vs_late_*_hippocampus.json"))

records = []
for f in files:
    with open(f) as fh:
        records.append(json.load(fh))

header = (
    f"{'Subject A':<14} {'Subject B':<14} "
    f"{'A→B Early':>10} {'A→B Late':>10} {'A→B Δ':>10}   "
    f"{'B→A Early':>10} {'B→A Late':>10} {'B→A Δ':>10}"
)
sep = "-" * len(header)

print(sep)
print("Cross-Subject Hippocampal Ridge Regression — Route Learning")
print("Values are median Pearson r across CV folds")
print(sep)
print(header)
print(sep)

for r in records:
    a2b_e = r["a_to_b"]["early"]["median_r"]
    a2b_l = r["a_to_b"]["late"]["median_r"]
    a2b_d = a2b_l - a2b_e
    b2a_e = r["b_to_a"]["early"]["median_r"]
    b2a_l = r["b_to_a"]["late"]["median_r"]
    b2a_d = b2a_l - b2a_e

    sa = r["subject_a"].replace("sub-Exp1", "")
    sb = r["subject_b"].replace("sub-Exp1", "")

    print(
        f"{sa:<14} {sb:<14} "
        f"{a2b_e:>10.4f} {a2b_l:>10.4f} {a2b_d:>+10.4f}   "
        f"{b2a_e:>10.4f} {b2a_l:>10.4f} {b2a_d:>+10.4f}"
    )

print(sep)

# Summary statistics
a2b_early_vals = [r["a_to_b"]["early"]["median_r"] for r in records]
a2b_late_vals = [r["a_to_b"]["late"]["median_r"] for r in records]
b2a_early_vals = [r["b_to_a"]["early"]["median_r"] for r in records]
b2a_late_vals = [r["b_to_a"]["late"]["median_r"] for r in records]
a2b_change_vals = [l - e for l, e in zip(a2b_late_vals, a2b_early_vals)]
b2a_change_vals = [l - e for l, e in zip(b2a_late_vals, b2a_early_vals)]

import numpy as np

def stats(vals, label):
    v = np.array(vals)
    print(
        f"  {label:<20}  "
        f"mean={v.mean():+.4f}  median={np.median(v):+.4f}  "
        f"std={v.std():.4f}  min={v.min():+.4f}  max={v.max():+.4f}"
    )

print("\nSummary Statistics:")
print("-" * 90)
stats(a2b_early_vals, "A→B Early")
stats(a2b_late_vals, "A→B Late")
stats(a2b_change_vals, "A→B Change")
print()
stats(b2a_early_vals, "B→A Early")
stats(b2a_late_vals, "B→A Late")
stats(b2a_change_vals, "B→A Change")
print("-" * 90)

n_a2b_increase = sum(1 for d in a2b_change_vals if d > 0)
n_b2a_increase = sum(1 for d in b2a_change_vals if d > 0)
n = len(records)
print(f"\nA→B: {n_a2b_increase}/{n} pairs increased ({100*n_a2b_increase/n:.1f}%)")
print(f"B→A: {n_b2a_increase}/{n} pairs increased ({100*n_b2a_increase/n:.1f}%)")
