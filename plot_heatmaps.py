import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

results_dir = Path("results")
files = sorted(results_dir.glob("early_vs_late_*_hippocampus.json"))

records = []
for f in files:
    with open(f) as fh:
        records.append(json.load(fh))

subjects = sorted({r["subject_a"] for r in records} | {r["subject_b"] for r in records})
sub_idx = {s: i for i, s in enumerate(subjects)}
n = len(subjects)

labels = [s.replace("sub-Exp1s", "s") for s in subjects]

a2b_early = np.full((n, n), np.nan)
a2b_late = np.full((n, n), np.nan)
b2a_early = np.full((n, n), np.nan)
b2a_late = np.full((n, n), np.nan)

for r in records:
    i = sub_idx[r["subject_a"]]
    j = sub_idx[r["subject_b"]]
    a2b_early[i, j] = r["a_to_b"]["early"]["median_r"]
    a2b_late[i, j] = r["a_to_b"]["late"]["median_r"]
    b2a_early[i, j] = r["b_to_a"]["early"]["median_r"]
    b2a_late[i, j] = r["b_to_a"]["late"]["median_r"]

def combine_triangles(upper, lower):
    """Upper triangle from `upper`, lower triangle from `lower`, diagonal NaN."""
    mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = upper[i, j]   # A→B
            mat[j, i] = lower[i, j]   # B→A (transposed into lower tri)
    return mat

combined_early = combine_triangles(a2b_early, b2a_early)
combined_late = combine_triangles(a2b_late, b2a_late)

all_vals = np.concatenate([
    combined_early[~np.isnan(combined_early)],
    combined_late[~np.isnan(combined_late)],
])
vmin, vmax = all_vals.min(), all_vals.max()

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
panels = [
    (combined_early, "Early Learning (runs 1–7)"),
    (combined_late,  "Late Learning (runs 8–14)"),
]

for ax, (mat, title) in zip(axes, panels):
    masked = np.ma.array(mat, mask=np.isnan(mat))
    im = ax.imshow(masked, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Subject")
    ax.plot([-.5, n-.5], [-.5, n-.5], color="gray", lw=0.8, ls="--")

fig.subplots_adjust(right=0.88, wspace=0.30)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label="Median Pearson r")
fig.suptitle(
    "Cross-Subject Hippocampal Ridge Regression — Route Learning",
    fontsize=13, fontweight="bold", y=1.0,
)
fig.savefig("heatmap_early_vs_late.png", dpi=200, bbox_inches="tight")
print("Saved heatmap_early_vs_late.png")

# --- Change heatmap (late minus early), same combined layout ---
a2b_change = a2b_late - a2b_early
b2a_change = b2a_late - b2a_early
combined_change = combine_triangles(a2b_change, b2a_change)

all_change = combined_change[~np.isnan(combined_change)]
clim = max(abs(all_change.min()), abs(all_change.max()))

fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
masked_change = np.ma.array(combined_change, mask=np.isnan(combined_change))
im2 = ax2.imshow(masked_change, cmap="RdBu_r", vmin=-clim, vmax=clim, aspect="equal")
ax2.set_xticks(range(n))
ax2.set_yticks(range(n))
ax2.set_xticklabels(labels, rotation=90, fontsize=7)
ax2.set_yticklabels(labels, fontsize=7)
ax2.set_xlabel("Subject")
ax2.set_ylabel("Subject")
ax2.plot([-.5, n-.5], [-.5, n-.5], color="gray", lw=0.8, ls="--")

fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04,
              label="Δ Median Pearson r (Late − Early)")
fig2.suptitle(
    "Change in Cross-Subject Prediction with Learning",
    fontsize=13, fontweight="bold", y=1.0,
)
fig2.savefig("heatmap_change.png", dpi=200, bbox_inches="tight")
print("Saved heatmap_change.png")

plt.show()
