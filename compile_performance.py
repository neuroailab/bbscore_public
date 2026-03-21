"""
Compile behavioral performance across scanning runs for Experiment 1
and plot alongside cross-subject hippocampal prediction results.

Behavioral measures:
  - In-scanner catch trial accuracy per run (destination and direction tests)
  - Post-scan picture test accuracy by segment (overlapping vs non-overlapping)

Cross-subject prediction:
  - Early (runs 1-7) vs late (runs 8-14) ridge regression median Pearson r
    from the early_vs_late_learning.py results
"""

import os
import csv
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

BIDS_DIR = os.environ.get(
    "ROUTE_LEARNING_DATA",
    os.path.join(os.getcwd(), "route-learning"),
)

EXP1_SUBJECTS = [f"sub-Exp1s{i:02d}" for i in range(1, 21)]
N_RUNS = 14


# ──────────────────────────────────────────────────────────────
# 1. Compile in-scanner catch trial performance per run
# ──────────────────────────────────────────────────────────────

def parse_catch_trials(subject):
    """
    Extract catch trial accuracy for each run.

    Returns dict mapping run number (1-14) to list of
    (test_type, test_correct) tuples.
    """
    func_dir = os.path.join(BIDS_DIR, subject, "func")
    runs = {}
    for run_num in range(1, N_RUNS + 1):
        events_file = os.path.join(
            func_dir,
            f"{subject}_task-routelearning_run-{run_num:02d}_events.tsv",
        )
        if not os.path.exists(events_file):
            continue
        trials = []
        with open(events_file) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["trial_type"] == "catch_trial":
                    test_type = row["test_type"]
                    correct = int(row["test_correct"])
                    trials.append((test_type, correct))
        runs[run_num] = trials
    return runs


def compile_catch_trial_data():
    """Compile catch trial accuracy across all subjects and runs."""
    subject_data = {}
    for sub in EXP1_SUBJECTS:
        runs = parse_catch_trials(sub)
        if len(runs) == N_RUNS:
            subject_data[sub] = runs
    return subject_data


def compute_accuracy_curves(subject_data):
    """
    Compute per-run accuracy curves, separated by test type.

    Returns arrays of shape (n_subjects, 14) for:
      - all catch trials combined
      - destination trials only
      - direction trials only
    """
    subjects = sorted(subject_data.keys())
    n_sub = len(subjects)

    all_acc = np.full((n_sub, N_RUNS), np.nan)
    dest_acc = np.full((n_sub, N_RUNS), np.nan)
    dir_acc = np.full((n_sub, N_RUNS), np.nan)

    for si, sub in enumerate(subjects):
        for run_num in range(1, N_RUNS + 1):
            trials = subject_data[sub].get(run_num, [])
            if not trials:
                continue

            all_correct = [t[1] for t in trials]
            dest_correct = [t[1] for t in trials if t[0] == "destination"]
            dir_correct = [t[1] for t in trials if t[0] == "direction"]

            all_acc[si, run_num - 1] = np.mean(all_correct)
            if dest_correct:
                dest_acc[si, run_num - 1] = np.mean(dest_correct)
            if dir_correct:
                dir_acc[si, run_num - 1] = np.mean(dir_correct)

    return subjects, all_acc, dest_acc, dir_acc


# ──────────────────────────────────────────────────────────────
# 2. Compile post-scan picture test performance
# ──────────────────────────────────────────────────────────────

def parse_picture_test(subject):
    """
    Parse the post-scan picture test for a subject.

    Derives competitor responses: an incorrect response that matches another
    route's destination (the data labels all non-target as "other", but since
    subjects always respond with one of the 4 route destinations, every
    non-target response on segment 1 is a competitor error).
    """
    beh_file = os.path.join(
        BIDS_DIR, subject, "beh",
        f"{subject}_task-picturetest_events.tsv",
    )
    if not os.path.exists(beh_file):
        return None

    rows = []
    with open(beh_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    all_destinations = {row["destination"] for row in rows}

    seg1_trials = []
    seg2_trials = []
    seg1_counts = {"target": 0, "competitor": 0, "other": 0}
    seg2_counts = {"target": 0, "competitor": 0, "other": 0}

    for row in rows:
        segment = int(row["segment"])
        hit = int(row["hits"])
        response = row["response"]
        destination = row["destination"]

        counts = seg1_counts if segment == 1 else seg2_counts
        trial_list = seg1_trials if segment == 1 else seg2_trials
        trial_list.append(hit)

        if response == destination:
            counts["target"] += 1
        elif response in all_destinations:
            counts["competitor"] += 1
        else:
            counts["other"] += 1

    all_trials = seg1_trials + seg2_trials

    def fracs(counts, n):
        return {k: counts[k] / n if n else 0 for k in counts}

    seg1_fracs = fracs(seg1_counts, len(seg1_trials))
    seg2_fracs = fracs(seg2_counts, len(seg2_trials))

    return {
        "overall_acc": np.mean(all_trials) if all_trials else np.nan,
        "seg1_acc": np.mean(seg1_trials) if seg1_trials else np.nan,
        "seg2_acc": np.mean(seg2_trials) if seg2_trials else np.nan,
        "seg1_n": len(seg1_trials),
        "seg2_n": len(seg2_trials),
        "seg1_target_frac": seg1_fracs["target"],
        "seg1_competitor_frac": seg1_fracs["competitor"],
        "seg1_other_frac": seg1_fracs["other"],
        "seg2_target_frac": seg2_fracs["target"],
        "seg2_competitor_frac": seg2_fracs["competitor"],
        "seg2_other_frac": seg2_fracs["other"],
    }


def compile_picture_test_data():
    """Compile picture test data across all subjects."""
    data = {}
    for sub in EXP1_SUBJECTS:
        result = parse_picture_test(sub)
        if result is not None:
            data[sub] = result
    return data


# ──────────────────────────────────────────────────────────────
# 3. Load cross-subject prediction results
# ──────────────────────────────────────────────────────────────

def load_prediction_results():
    """Load early vs late cross-subject prediction results."""
    results_dir = Path("results")
    files = sorted(results_dir.glob("early_vs_late_*_hippocampus.json"))

    records = []
    for f in files:
        with open(f) as fh:
            records.append(json.load(fh))

    early_vals = []
    late_vals = []
    for r in records:
        early_vals.append(r["a_to_b"]["early"]["median_r"])
        early_vals.append(r["b_to_a"]["early"]["median_r"])
        late_vals.append(r["a_to_b"]["late"]["median_r"])
        late_vals.append(r["b_to_a"]["late"]["median_r"])

    # Per-fold detail: collect per-run fold accuracies for a finer-grained view
    early_folds_by_run = defaultdict(list)
    late_folds_by_run = defaultdict(list)
    for r in records:
        for direction in ["a_to_b", "b_to_a"]:
            for fold in r[direction]["early"]["folds"]:
                early_folds_by_run[fold["fold"]].append(fold["r_median"])
            for fold in r[direction]["late"]["folds"]:
                late_folds_by_run[fold["fold"]].append(fold["r_median"])

    return {
        "n_pairs": len(records),
        "early_mean": np.mean(early_vals),
        "early_sem": np.std(early_vals) / np.sqrt(len(early_vals)),
        "late_mean": np.mean(late_vals),
        "late_sem": np.std(late_vals) / np.sqrt(len(late_vals)),
        "early_vals": early_vals,
        "late_vals": late_vals,
        "early_folds_by_run": dict(early_folds_by_run),
        "late_folds_by_run": dict(late_folds_by_run),
    }


# ──────────────────────────────────────────────────────────────
# 4. Plotting
# ──────────────────────────────────────────────────────────────

def plot_combined(all_acc, dest_acc, dir_acc, pic_test_data, pred_results, subjects):
    """Create a combined figure with behavioral + prediction results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    runs = np.arange(1, N_RUNS + 1)
    colors = {"all": "#2196F3", "dest": "#E91E63", "dir": "#4CAF50"}

    # ── Panel A: Catch trial accuracy learning curve ──────────
    ax = axes[0]
    for label, data, color in [
        ("All catch trials", all_acc, colors["all"]),
        ("Destination only", dest_acc, colors["dest"]),
        ("Direction only", dir_acc, colors["dir"]),
    ]:
        mean = np.nanmean(data, axis=0)
        sem = np.nanstd(data, axis=0) / np.sqrt(np.sum(~np.isnan(data), axis=0))
        ax.plot(runs, mean, "o-", color=color, label=label, markersize=4, linewidth=1.5)
        ax.fill_between(runs, mean - sem, mean + sem, alpha=0.15, color=color)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(0.25, color="gray", linestyle=":", alpha=0.4, linewidth=0.8, label="Chance (destination)")
    ax.axvline(7.5, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax.text(3.5, 0.05, "Early\n(runs 1–7)", ha="center", fontsize=8, alpha=0.5)
    ax.text(11, 0.05, "Late\n(runs 8–14)", ha="center", fontsize=8, alpha=0.5)
    ax.set_xlabel("Scanning Run")
    ax.set_ylabel("Accuracy")
    ax.set_title("A. In-Scanner Catch Trial Accuracy", fontweight="bold")
    ax.set_xticks(runs)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7, loc="lower right")

    # ── Panel B: Per-fold prediction across runs ──────────────
    ax = axes[1]
    early_folds = pred_results["early_folds_by_run"]
    late_folds = pred_results["late_folds_by_run"]

    early_fold_nums = sorted(early_folds.keys())
    late_fold_nums = sorted(late_folds.keys())

    early_means = [np.mean(early_folds[k]) for k in early_fold_nums]
    early_sems = [np.std(early_folds[k]) / np.sqrt(len(early_folds[k])) for k in early_fold_nums]
    late_means = [np.mean(late_folds[k]) for k in late_fold_nums]
    late_sems = [np.std(late_folds[k]) / np.sqrt(len(late_folds[k])) for k in late_fold_nums]

    early_x = np.array(early_fold_nums)
    late_x = np.array(late_fold_nums) + 7

    ax.errorbar(early_x, early_means, yerr=early_sems, fmt="o-",
                color="#1565C0", label="Early half folds", markersize=5, linewidth=1.5, capsize=3)
    ax.errorbar(late_x, late_means, yerr=late_sems, fmt="s-",
                color="#C62828", label="Late half folds", markersize=5, linewidth=1.5, capsize=3)
    ax.axvline(7.5, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Run (fold held out)")
    ax.set_ylabel("Median Pearson r")
    ax.set_title("B. Cross-Subject Prediction by Fold", fontweight="bold")
    ax.set_xticks(range(1, 15))
    ax.legend(fontsize=7)

    fig.suptitle(
        "Route Learning Experiment 1: Behavioral Performance & Cross-Subject Prediction",
        fontsize=14, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    fig.savefig("performance_vs_prediction.png", dpi=200, bbox_inches="tight")
    print("Saved performance_vs_prediction.png")


# ──────────────────────────────────────────────────────────────
# 5. Print summary table
# ──────────────────────────────────────────────────────────────

def print_summary(subjects, all_acc, dest_acc, pic_test_data, pred_results):
    sep = "=" * 80
    print(f"\n{sep}")
    print("EXPERIMENT 1 BEHAVIORAL PERFORMANCE SUMMARY")
    print(sep)

    print(f"\nSubjects with complete data: {len(subjects)}")
    print(f"Subjects: {', '.join(s.replace('sub-Exp1s', 's') for s in subjects)}")

    print(f"\n{'─' * 80}")
    print("IN-SCANNER CATCH TRIAL ACCURACY BY RUN")
    print(f"{'─' * 80}")
    print(f"{'Run':<6}", end="")
    print(f"{'All (mean±sem)':<20} {'Dest (mean±sem)':<20}")
    print("-" * 46)
    for run in range(N_RUNS):
        all_m = np.nanmean(all_acc[:, run])
        all_s = np.nanstd(all_acc[:, run]) / np.sqrt(np.sum(~np.isnan(all_acc[:, run])))
        dest_m = np.nanmean(dest_acc[:, run])
        dest_s = np.nanstd(dest_acc[:, run]) / np.sqrt(np.sum(~np.isnan(dest_acc[:, run])))
        print(f"{run+1:<6}{all_m:.3f} ± {all_s:.3f}       {dest_m:.3f} ± {dest_s:.3f}")

    early_all = np.nanmean(all_acc[:, :7], axis=1)
    late_all = np.nanmean(all_acc[:, 7:], axis=1)
    print(f"\nEarly half mean: {np.mean(early_all):.3f} ± {np.std(early_all)/np.sqrt(len(early_all)):.3f}")
    print(f"Late half mean:  {np.mean(late_all):.3f} ± {np.std(late_all)/np.sqrt(len(late_all)):.3f}")
    print(f"Change:          {np.mean(late_all) - np.mean(early_all):+.3f}")

    print(f"\n{'─' * 80}")
    print("POST-SCAN PICTURE TEST")
    print(f"{'─' * 80}")
    pred_subjects = set(subjects)
    seg1_vals = [pic_test_data[s]["seg1_acc"] for s in sorted(pic_test_data) if s in pred_subjects]
    seg2_vals = [pic_test_data[s]["seg2_acc"] for s in sorted(pic_test_data) if s in pred_subjects]
    overall_vals = [pic_test_data[s]["overall_acc"] for s in sorted(pic_test_data) if s in pred_subjects]
    print(f"  Overall accuracy:  {np.mean(overall_vals):.3f} ± {np.std(overall_vals)/np.sqrt(len(overall_vals)):.3f}")
    print(f"  Segment 1 (overlap): {np.mean(seg1_vals):.3f} ± {np.std(seg1_vals)/np.sqrt(len(seg1_vals)):.3f}")
    print(f"  Segment 2 (unique):  {np.mean(seg2_vals):.3f} ± {np.std(seg2_vals)/np.sqrt(len(seg2_vals)):.3f}")

    print(f"\n{'─' * 80}")
    print("CROSS-SUBJECT HIPPOCAMPAL PREDICTION (from early_vs_late_learning.py)")
    print(f"{'─' * 80}")
    print(f"  {pred_results['n_pairs']} subject pairs, both directions")
    print(f"  Early (runs 1–7):  median r = {pred_results['early_mean']:.4f} ± {pred_results['early_sem']:.4f}")
    print(f"  Late (runs 8–14):  median r = {pred_results['late_mean']:.4f} ± {pred_results['late_sem']:.4f}")
    diff = pred_results['late_mean'] - pred_results['early_mean']
    print(f"  Change:            Δ = {diff:+.4f}")
    print(sep)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Compiling behavioral performance for Experiment 1...")

    catch_data = compile_catch_trial_data()
    subjects, all_acc, dest_acc, dir_acc = compute_accuracy_curves(catch_data)
    print(f"  {len(subjects)} subjects with complete catch trial data")

    pic_test_data = compile_picture_test_data()
    print(f"  {len(pic_test_data)} subjects with picture test data")

    pred_results = load_prediction_results()
    print(f"  {pred_results['n_pairs']} subject pair results loaded")

    print_summary(subjects, all_acc, dest_acc, pic_test_data, pred_results)
    plot_combined(all_acc, dest_acc, dir_acc, pic_test_data, pred_results, subjects)
