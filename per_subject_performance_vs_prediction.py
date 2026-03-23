"""
Per-subject comparison: catch trial performance vs change in
cross-subject hippocampal prediction (late - early learning).

For each subject, computes:
  - Catch trial accuracy (overall, early half, late half, and the change)
  - Mean Δ in cross-subject prediction across all pairs involving that subject
Then plots the relationship.
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import stats

BIDS_DIR = os.environ.get(
    "ROUTE_LEARNING_DATA",
    os.path.join(os.getcwd(), "route-learning"),
)
N_RUNS = 14


# ──────────────────────────────────────────────────────────────
# 1. Per-subject catch trial performance
# ──────────────────────────────────────────────────────────────

def get_subject_catch_performance(subject):
    """Return per-run catch trial accuracy for a subject."""
    func_dir = os.path.join(BIDS_DIR, subject, "func")
    run_accuracies = {}
    for run_num in range(1, N_RUNS + 1):
        events_file = os.path.join(
            func_dir,
            f"{subject}_task-routelearning_run-{run_num:02d}_events.tsv",
        )
        if not os.path.exists(events_file):
            return None
        trials = []
        with open(events_file) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["trial_type"] == "catch_trial":
                    trials.append(int(row["test_correct"]))
        run_accuracies[run_num] = np.mean(trials) if trials else np.nan
    return run_accuracies


# ──────────────────────────────────────────────────────────────
# 2. Per-subject cross-subject prediction change
# ──────────────────────────────────────────────────────────────

def load_per_subject_prediction_change():
    """
    For each subject, compute the mean Δ(late - early) in cross-subject
    prediction across all pairs involving that subject.

    Returns dict: subject -> {
        'mean_delta': mean change across all pairs,
        'deltas': list of individual pair deltas,
        'n_pairs': number of pairs,
        'mean_delta_as_source': mean Δ when this subject is the source,
        'mean_delta_as_target': mean Δ when this subject is the target,
    }
    """
    results_dir = Path("results")
    files = sorted(results_dir.glob("early_vs_late_*_hippocampus.json"))

    subject_deltas = defaultdict(list)
    subject_deltas_as_source = defaultdict(list)
    subject_deltas_as_target = defaultdict(list)

    for f in files:
        with open(f) as fh:
            r = json.load(fh)

        sub_a = r["subject_a"]
        sub_b = r["subject_b"]
        delta_ab = r["a_to_b"]["late"]["median_r"] - r["a_to_b"]["early"]["median_r"]
        delta_ba = r["b_to_a"]["late"]["median_r"] - r["b_to_a"]["early"]["median_r"]

        # A→B: A is source, B is target
        subject_deltas[sub_a].append(delta_ab)
        subject_deltas[sub_a].append(delta_ba)
        subject_deltas[sub_b].append(delta_ab)
        subject_deltas[sub_b].append(delta_ba)

        subject_deltas_as_source[sub_a].append(delta_ab)
        subject_deltas_as_source[sub_b].append(delta_ba)
        subject_deltas_as_target[sub_b].append(delta_ab)
        subject_deltas_as_target[sub_a].append(delta_ba)

    result = {}
    for sub in subject_deltas:
        result[sub] = {
            "mean_delta": np.mean(subject_deltas[sub]),
            "deltas": subject_deltas[sub],
            "n_pairs": len(subject_deltas[sub]) // 2,
            "mean_delta_as_source": np.mean(subject_deltas_as_source[sub]),
            "mean_delta_as_target": np.mean(subject_deltas_as_target[sub]),
        }
    return result


# ──────────────────────────────────────────────────────────────
# 3. Combine and plot
# ──────────────────────────────────────────────────────────────

def main():
    pred_changes = load_per_subject_prediction_change()
    subjects = sorted(pred_changes.keys())

    # Gather behavioral data for subjects with prediction results
    sub_labels = []
    catch_overall = []
    catch_early = []
    catch_late = []
    catch_delta = []
    pred_delta = []
    pred_delta_source = []
    pred_delta_target = []

    for sub in subjects:
        perf = get_subject_catch_performance(sub)
        if perf is None:
            continue

        early_runs = [perf[r] for r in range(1, 8) if not np.isnan(perf.get(r, np.nan))]
        late_runs = [perf[r] for r in range(8, 15) if not np.isnan(perf.get(r, np.nan))]
        all_runs = [perf[r] for r in range(1, 15) if not np.isnan(perf.get(r, np.nan))]

        sub_labels.append(sub.replace("sub-Exp1s", "s"))
        catch_overall.append(np.mean(all_runs))
        catch_early.append(np.mean(early_runs))
        catch_late.append(np.mean(late_runs))
        catch_delta.append(np.mean(late_runs) - np.mean(early_runs))
        pred_delta.append(pred_changes[sub]["mean_delta"])
        pred_delta_source.append(pred_changes[sub]["mean_delta_as_source"])
        pred_delta_target.append(pred_changes[sub]["mean_delta_as_target"])

    catch_overall = np.array(catch_overall)
    catch_early = np.array(catch_early)
    catch_late = np.array(catch_late)
    catch_delta = np.array(catch_delta)
    pred_delta = np.array(pred_delta)
    pred_delta_source = np.array(pred_delta_source)
    pred_delta_target = np.array(pred_delta_target)

    # ── Print table ──────────────────────────────────────────
    print(f"\n{'Subject':<10} {'Catch':>8} {'Catch':>8} {'Catch':>8} {'Catch':>8}   "
          f"{'Pred':>8} {'Pred':>8} {'Pred':>8}")
    print(f"{'':>10} {'Overall':>8} {'Early':>8} {'Late':>8} {'Δ':>8}   "
          f"{'Δ(all)':>8} {'Δ(src)':>8} {'Δ(tgt)':>8}")
    print("-" * 85)
    for i, sub in enumerate(sub_labels):
        print(f"{sub:<10} {catch_overall[i]:>8.3f} {catch_early[i]:>8.3f} "
              f"{catch_late[i]:>8.3f} {catch_delta[i]:>+8.3f}   "
              f"{pred_delta[i]:>+8.4f} {pred_delta_source[i]:>+8.4f} "
              f"{pred_delta_target[i]:>+8.4f}")
    print("-" * 85)

    # ── Correlations ─────────────────────────────────────────
    print("\nCorrelations:")
    pairs = [
        ("Catch overall  vs  Pred Δ(all)", catch_overall, pred_delta),
        ("Catch Δ        vs  Pred Δ(all)", catch_delta, pred_delta),
        ("Catch early    vs  Pred Δ(all)", catch_early, pred_delta),
        ("Catch late     vs  Pred Δ(all)", catch_late, pred_delta),
        ("Catch overall  vs  Pred Δ(src)", catch_overall, pred_delta_source),
        ("Catch overall  vs  Pred Δ(tgt)", catch_overall, pred_delta_target),
    ]
    for label, x, y in pairs:
        r, p = stats.pearsonr(x, y)
        rho, p_rho = stats.spearmanr(x, y)
        print(f"  {label}:  Pearson r={r:+.3f} (p={p:.3f})  Spearman ρ={rho:+.3f} (p={p_rho:.3f})")

    # ── Plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    def scatter_with_fit(ax, x, y, xlabel, ylabel, title):
        ax.scatter(x, y, s=50, alpha=0.7, edgecolors="white", linewidth=0.5, zorder=3)
        for i, lbl in enumerate(sub_labels):
            ax.annotate(lbl, (x[i], y[i]), fontsize=6, alpha=0.6,
                        xytext=(4, 4), textcoords="offset points")
        r, p = stats.pearsonr(x, y)
        m, b = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_fit, m * x_fit + b, "--", color="red", alpha=0.5, linewidth=1.5)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        sig = "*" if p < 0.05 else ""
        ax.set_title(f"{title}\nr = {r:+.3f}, p = {p:.3f}{sig}", fontweight="bold", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    scatter_with_fit(
        axes[0, 0], catch_overall, pred_delta,
        "Overall Catch Trial Accuracy",
        "Mean Δ Prediction (Late − Early)",
        "A. Overall Accuracy vs Prediction Change",
    )

    scatter_with_fit(
        axes[0, 1], catch_delta, pred_delta,
        "Δ Catch Trial Accuracy (Late − Early)",
        "Mean Δ Prediction (Late − Early)",
        "B. Behavioral Improvement vs Prediction Change",
    )

    scatter_with_fit(
        axes[1, 0], catch_overall, pred_delta_source,
        "Overall Catch Trial Accuracy",
        "Mean Δ Prediction (as source)",
        "C. Accuracy vs Prediction Change (as source)",
    )

    scatter_with_fit(
        axes[1, 1], catch_overall, pred_delta_target,
        "Overall Catch Trial Accuracy",
        "Mean Δ Prediction (as target)",
        "D. Accuracy vs Prediction Change (as target)",
    )

    fig.suptitle(
        "Per-Subject: Catch Trial Performance vs Cross-Subject Prediction Change",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig("subject_performance_vs_prediction.png", dpi=200, bbox_inches="tight")
    print("\nSaved subject_performance_vs_prediction.png")


if __name__ == "__main__":
    main()
