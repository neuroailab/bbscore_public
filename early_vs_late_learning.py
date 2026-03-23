"""
Early vs. Late Learning Cross-Subject Hippocampal Prediction

This script tests whether hippocampal differentiation (Chanales et al.)
follows a shared template across subjects or is idiosyncratic.

Approach:
  - Split 14 runs into first half (runs 1-7) and second half (runs 8-14)
  - For each half, train ridge regression: Subject A → Subject B
  - Use leave-one-run-out cross-validation within each half
  - Compare prediction accuracy across halves

Interpretation:
  - 2nd half > 1st half: Differentiation follows a shared template
    (both brains reorganize the same way)
  - 2nd half < 1st half: Differentiation is idiosyncratic
    (each brain pushes routes apart in its own direction)
  - No difference: Shared signal is mostly sensory, not learning-related

Usage:
    python early_vs_late_learning.py --subject-a sub-Exp1s01 --subject-b sub-Exp1s02
"""

import os
import sys
import glob
import json
import argparse
import warnings
import numpy as np
import nibabel as nib
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr

from nilearn.maskers import NiftiMasker
from nilearn.image import resample_to_img
from nilearn import datasets as ni_datasets


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

BIDS_DIR = os.environ.get(
    "ROUTE_LEARNING_DATA",
    "/home/out",
)

TASK = "routelearning"
# fMRIPrep MNI outputs (must match filenames under <subject>/func/)
MNI_SPACE = "MNI152NLin2009cAsym"

REGION_CONFIGS = {
    "hippocampus": ["Left Hippocampus", "Right Hippocampus"],
    "left_hippocampus": ["Left Hippocampus"],
    "right_hippocampus": ["Right Hippocampus"],
}

ALPHA_OPTIONS = [1e-4, 1e-2, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]


# ──────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────

def get_hippocampal_mask(region, reference_img):
    """Create hippocampal mask from Harvard-Oxford atlas."""
    labels = REGION_CONFIGS[region]
    atlas = ni_datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
    atlas_maps = atlas["maps"]
    if isinstance(atlas_maps, nib.Nifti1Image):
        atlas_img = atlas_maps
    else:
        atlas_img = nib.load(atlas_maps)

    target_indices = []
    for label_name in labels:
        for i, atlas_label in enumerate(atlas["labels"]):
            if atlas_label == label_name:
                target_indices.append(i)
                break

    atlas_data = atlas_img.get_fdata()
    mask_data = np.isin(atlas_data, target_indices).astype(np.float32)
    mask_atlas = nib.Nifti1Image(mask_data, atlas_img.affine)
    mask_resampled = resample_to_img(mask_atlas, reference_img, interpolation="nearest")
    mask_binary = (mask_resampled.get_fdata() > 0.5).astype(np.int8)
    return nib.Nifti1Image(mask_binary, mask_resampled.affine)


def find_bold_files(subject, bids_dir=None):
    """Find all BOLD files for a subject, sorted by run (raw BIDS or fMRIPrep preproc)."""
    root = bids_dir if bids_dir is not None else BIDS_DIR
    func_dir = os.path.join(root, subject, "func")
    pattern_raw = os.path.join(func_dir, f"{subject}_task-{TASK}_run-*_bold.nii.gz")
    files = sorted(glob.glob(pattern_raw))
    if files:
        return files
    pattern_fp = os.path.join(
        func_dir,
        f"{subject}_task-{TASK}_run-*_space-{MNI_SPACE}_desc-preproc_bold.nii.gz",
    )
    files = sorted(glob.glob(pattern_fp))
    if files:
        return files
    raise FileNotFoundError(
        f"No BOLD files for {subject} task-{TASK} in {func_dir} "
        f"(tried raw *_bold.nii.gz and fMRIPrep *_desc-preproc_bold.nii.gz)"
    )


def extract_run_timeseries(bold_file, mask_img):
    """Extract hippocampal voxel timeseries for a single run."""
    masker = NiftiMasker(
        mask_img=mask_img,
        standardize=False,  # Don't z-score per run; we standardize at training level
        detrend=True,
        high_pass=0.008,
        t_r=1.5,  # From the paper: TR = 1.5s
    )
    bold_img = nib.load(bold_file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        voxel_ts = masker.fit_transform(bold_img)
    return voxel_ts


def extract_all_runs(subject, mask_img, bids_dir=None):
    """
    Extract hippocampal timeseries for each run separately.

    Returns:
        List of arrays, each (n_TRs_in_run, n_voxels).
    """
    bold_files = find_bold_files(subject, bids_dir=bids_dir)
    print(f"  {subject}: {len(bold_files)} runs found")

    run_data = []
    for i, bf in enumerate(bold_files):
        ts = extract_run_timeseries(bf, mask_img)
        run_data.append(ts)
        print(f"    Run {i+1}: {ts.shape[0]} TRs, {ts.shape[1]} voxels")

    return run_data


def zscore_columns(data):
    """Z-score each column (voxel), handling constant columns gracefully."""
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True, ddof=1)
    std[std == 0] = 1.0  # Avoid division by zero; constant columns become 0
    return (data - mean) / std


def pearson_per_voxel(y_true, y_pred):
    """Compute Pearson r for each voxel (column), handling constant inputs."""
    n_voxels = y_true.shape[1]
    correlations = np.zeros(n_voxels)
    for v in range(n_voxels):
        true_v = y_true[:, v]
        pred_v = y_pred[:, v]
        # Skip voxels with zero variance (constant signal)
        if np.std(true_v) < 1e-10 or np.std(pred_v) < 1e-10:
            correlations[v] = 0.0
            continue
        r, _ = pearsonr(true_v, pred_v)
        correlations[v] = r if np.isfinite(r) else 0.0
    return correlations


def run_ridge_leave_one_run_out(source_runs, target_runs):
    """
    Leave-one-run-out ridge regression across a set of runs.

    For each held-out run, train on all other runs, predict the held-out
    run, and compute per-voxel Pearson correlation.

    Args:
        source_runs: List of arrays (n_TRs, n_source_voxels), one per run.
        target_runs: List of arrays (n_TRs, n_target_voxels), one per run.

    Returns:
        median_r: Median Pearson r across voxels and folds.
        all_fold_medians: List of median-r values, one per fold.
    """
    n_runs = len(source_runs)
    assert len(target_runs) == n_runs

    # Align TR counts per run (trim to shorter if needed)
    for i in range(n_runs):
        min_trs = min(source_runs[i].shape[0], target_runs[i].shape[0])
        source_runs[i] = source_runs[i][:min_trs]
        target_runs[i] = target_runs[i][:min_trs]

    folds = []

    for held_out in range(n_runs):
        # Concatenate training runs
        train_source = np.concatenate(
            [source_runs[i] for i in range(n_runs) if i != held_out], axis=0
        )
        train_target = np.concatenate(
            [target_runs[i] for i in range(n_runs) if i != held_out], axis=0
        )
        test_source = source_runs[held_out]
        test_target = target_runs[held_out]

        # Z-score using training statistics
        src_mean = np.mean(train_source, axis=0, keepdims=True)
        src_std = np.std(train_source, axis=0, keepdims=True, ddof=1)
        src_std[src_std == 0] = 1.0
        train_source = (train_source - src_mean) / src_std
        test_source = (test_source - src_mean) / src_std

        tgt_mean = np.mean(train_target, axis=0, keepdims=True)
        tgt_std = np.std(train_target, axis=0, keepdims=True, ddof=1)
        tgt_std[tgt_std == 0] = 1.0
        train_target = (train_target - tgt_mean) / tgt_std
        test_target = (test_target - tgt_mean) / tgt_std

        # Fit ridge regression
        model = RidgeCV(alphas=ALPHA_OPTIONS, store_cv_results=False)
        model.fit(train_source, train_target)

        # Predict and score
        predictions = model.predict(test_source)
        voxel_r = pearson_per_voxel(test_target, predictions)

        fold_info = {
            "fold": held_out + 1,
            "r_median": float(np.median(voxel_r)),
            "r_mean": float(np.mean(voxel_r)),
            "r_std": float(np.std(voxel_r)),
            "r_min": float(np.min(voxel_r)),
            "r_max": float(np.max(voxel_r)),
            "r_q25": float(np.percentile(voxel_r, 25)),
            "r_q75": float(np.percentile(voxel_r, 75)),
        }
        folds.append(fold_info)

        print(f"      Fold {held_out+1}/{n_runs}: median r = {fold_info['r_median']:.4f}")

    fold_medians = [f["r_median"] for f in folds]
    return {
        "median_r": float(np.mean(fold_medians)),
        "mean_r": float(np.mean([f["r_mean"] for f in folds])),
        "folds": folds,
    }


# ──────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────

def run_experiment(subject_a, subject_b, region="hippocampus", bids_dir=None):
    """
    Run the early vs. late learning cross-subject comparison.

    Splits 14 runs into:
      - First half: runs 1-7 (early learning)
      - Second half: runs 8-14 (late learning, post-differentiation)

    For each half, runs leave-one-run-out ridge regression in both
    directions (A→B and B→A).

    Args:
        bids_dir: BIDS dataset root (default: module BIDS_DIR / ROUTE_LEARNING_DATA).
    """
    data_root = bids_dir if bids_dir is not None else BIDS_DIR
    print(f"\n{'='*60}")
    print(f"Early vs. Late Learning Cross-Subject Prediction")
    print(f"  Subject A: {subject_a}")
    print(f"  Subject B: {subject_b}")
    print(f"  Region: {region}")
    print(f"  Data: {data_root}")
    print(f"{'='*60}\n")

    # Create mask from first subject's first run
    first_bold = find_bold_files(subject_a, bids_dir=data_root)[0]
    ref_img = nib.load(first_bold)
    print(f"Creating {region} mask...")
    mask_img = get_hippocampal_mask(region, ref_img)
    n_voxels = int(mask_img.get_fdata().sum())
    print(f"  Mask: {n_voxels} voxels\n")

    # Extract per-run timeseries for both subjects
    print("Extracting per-run hippocampal timeseries...")
    runs_a = extract_all_runs(subject_a, mask_img, bids_dir=data_root)
    runs_b = extract_all_runs(subject_b, mask_img, bids_dir=data_root)

    # Ensure both subjects have 14 runs
    n_runs = min(len(runs_a), len(runs_b))
    if n_runs < 14:
        print(f"\nWarning: Only {n_runs} runs available (expected 14)")
    runs_a = runs_a[:n_runs]
    runs_b = runs_b[:n_runs]

    # Split into halves
    mid = n_runs // 2
    first_half_a = [runs_a[i] for i in range(mid)]
    second_half_a = [runs_a[i] for i in range(mid, n_runs)]
    first_half_b = [runs_b[i] for i in range(mid)]
    second_half_b = [runs_b[i] for i in range(mid, n_runs)]

    print(f"\nFirst half: runs 1-{mid} ({len(first_half_a)} runs)")
    print(f"Second half: runs {mid+1}-{n_runs} ({len(second_half_a)} runs)")

    # ── Direction A → B ───────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Direction: {subject_a} → {subject_b}")
    print(f"{'─'*50}")

    print(f"\n  First half (early learning):")
    early_ab = run_ridge_leave_one_run_out(
        [r.copy() for r in first_half_a],
        [r.copy() for r in first_half_b],
    )

    print(f"\n  Second half (late learning):")
    late_ab = run_ridge_leave_one_run_out(
        [r.copy() for r in second_half_a],
        [r.copy() for r in second_half_b],
    )

    # ── Direction B → A ───────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Direction: {subject_b} → {subject_a}")
    print(f"{'─'*50}")

    print(f"\n  First half (early learning):")
    early_ba = run_ridge_leave_one_run_out(
        [r.copy() for r in first_half_b],
        [r.copy() for r in first_half_a],
    )

    print(f"\n  Second half (late learning):")
    late_ba = run_ridge_leave_one_run_out(
        [r.copy() for r in second_half_b],
        [r.copy() for r in second_half_a],
    )

    # ── Summary ───────────────────────────────────────────────
    diff_ab = late_ab["median_r"] - early_ab["median_r"]
    diff_ba = late_ba["median_r"] - early_ba["median_r"]
    direction_ab = "INCREASE" if diff_ab > 0 else "DECREASE"
    direction_ba = "INCREASE" if diff_ba > 0 else "DECREASE"

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {subject_a} → {subject_b}:")
    print(f"    First half  (early learning): median r = {early_ab['median_r']:.4f}")
    print(f"    Second half (late learning):  median r = {late_ab['median_r']:.4f}")
    print(f"    Change: {diff_ab:+.4f} ({direction_ab})")

    print(f"\n  {subject_b} → {subject_a}:")
    print(f"    First half  (early learning): median r = {early_ba['median_r']:.4f}")
    print(f"    Second half (late learning):  median r = {late_ba['median_r']:.4f}")
    print(f"    Change: {diff_ba:+.4f} ({direction_ba})")

    print(f"\n{'─'*50}")
    print(f"INTERPRETATION:")
    if diff_ab > 0 and diff_ba > 0:
        print(f"  Both directions show INCREASED prediction in late learning.")
        print(f"  → Hippocampal differentiation likely follows a SHARED TEMPLATE")
        print(f"    across subjects. Both brains reorganize in the same way.")
    elif diff_ab < 0 and diff_ba < 0:
        print(f"  Both directions show DECREASED prediction in late learning.")
        print(f"  → Hippocampal differentiation is likely IDIOSYNCRATIC.")
        print(f"    Each brain pushes overlapping routes apart differently.")
    else:
        print(f"  Mixed results across directions.")
        print(f"  → Asymmetry may reflect differences in signal quality")
        print(f"    or learning trajectories between subjects.")
    print(f"{'─'*50}\n")

    # ── Save detailed results ─────────────────────────────────
    results = {
        "subject_a": subject_a,
        "subject_b": subject_b,
        "region": region,
        "n_runs": int(n_runs),
        "a_to_b": {
            "early": early_ab,
            "late": late_ab,
            "change": float(diff_ab),
        },
        "b_to_a": {
            "early": early_ba,
            "late": late_ba,
            "change": float(diff_ba),
        },
    }

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(
        results_dir,
        f"early_vs_late_{subject_a}_{subject_b}_{region}.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"Results saved to {results_file}")

    return results


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Early vs. late learning cross-subject hippocampal prediction"
    )
    parser.add_argument(
        "--subject-a", type=str, default="sub-Exp1s01",
        help="BIDS subject ID for source subject"
    )
    parser.add_argument(
        "--subject-b", type=str, default="sub-Exp1s02",
        help="BIDS subject ID for target subject"
    )
    parser.add_argument(
        "--region", type=str, default="hippocampus",
        choices=["hippocampus", "left_hippocampus", "right_hippocampus"],
        help="Hippocampal region to analyze"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to BIDS dataset (overrides ROUTE_LEARNING_DATA env var)"
    )

    args = parser.parse_args()
    run_experiment(
        args.subject_a,
        args.subject_b,
        args.region,
        bids_dir=args.data_dir,
    )