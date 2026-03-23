"""
Route Learning Benchmarks for BBScore.

These benchmarks are intentionally implemented as a mostly self-contained module
so that RouteLearning logic (trial alignment, leave-one-run-out CV, warnings)
does not require modifying BBScore core code.

Each benchmark loads two RouteLearning assemblies (source + target), aligns
trial rows by (run_number, route_id, rep_in_run), then runs ridge regression
with *leave-one-run-out* cross-validation (GroupKFold).
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from metrics.ridge import RidgeMetric
from data.RouteLearning import (
    # Experiment 1: bilateral hippocampus
    RouteLearningExp1s01Hippo,
    RouteLearningExp1s02Hippo,
    RouteLearningExp1s03Hippo,
    RouteLearningExp1s04Hippo,
    RouteLearningExp1s05Hippo,
    # Experiment 1: left hippocampus
    RouteLearningExp1s01LeftHippo,
    RouteLearningExp1s02LeftHippo,
    # Experiment 1: right hippocampus
    RouteLearningExp1s01RightHippo,
    RouteLearningExp1s02RightHippo,
    # Experiment 2: bilateral hippocampus
    RouteLearningExp2s01Hippo,
    RouteLearningExp2s02Hippo,
    RouteLearningExp2s03Hippo,
    RouteLearningConfigurable,
)
from benchmarks import BENCHMARK_REGISTRY


def _pearson_safe(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size < 2:
        return float("nan")
    if np.var(y_true) < eps or np.var(y_pred) < eps:
        return 1.0 if np.allclose(y_true, y_pred, atol=eps) else 0.0
    r, _ = pearsonr(y_true, y_pred)
    return float((2 * r) / (1 + r) if r != 1 else 1.0)


class RouteLearningLOROBenchmark:
    """
    Minimal benchmark runner compatible with run.py:
    - add_metric(name)
    - run() -> {'metrics': {...}, 'ceiling': ...}
    """

    def __init__(
        self,
        source_assembly_class,
        target_assembly_class,
        debug: bool = False,
        warn_min_trials: int = 80,
        warn_min_trials_per_run: int = 4,
    ):
        self.source_assembly_class = source_assembly_class
        self.target_assembly_class = target_assembly_class
        self.debug = debug
        self.metrics = {}
        self.metric_params = {}
        self.warn_min_trials = int(warn_min_trials)
        self.warn_min_trials_per_run = int(warn_min_trials_per_run)

    def add_metric(self, name, metric_params=None):
        # We currently only support ridge-like regression metrics here.
        self.metrics[name] = name
        if metric_params:
            self.metric_params[name] = metric_params

    def _align_by_keys(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        groups: np.ndarray,
        X_keys: np.ndarray,
        Y_keys: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xk = [tuple(map(int, k)) for k in X_keys.tolist()]
        yk = [tuple(map(int, k)) for k in Y_keys.tolist()]
        x_map = {k: i for i, k in enumerate(xk)}
        y_map = {k: i for i, k in enumerate(yk)}
        common = sorted(set(x_map).intersection(set(y_map)))
        if not common:
            raise RuntimeError("No overlapping trial keys between source and target.")
        x_idx = np.array([x_map[k] for k in common], dtype=int)
        y_idx = np.array([y_map[k] for k in common], dtype=int)
        X_al = X[x_idx]
        Y_al = Y[y_idx]
        g_al = np.asarray(groups)[y_idx]  # groups are tied to target keys
        return X_al, Y_al, g_al

    def _warn_if_too_few_trials(self, groups: np.ndarray, n_trials: int):
        unique, counts = np.unique(groups, return_counts=True)
        min_per_run = int(counts.min()) if counts.size else 0
        if n_trials < self.warn_min_trials or min_per_run < self.warn_min_trials_per_run:
            print(
                "Warning: relatively few aligned trials for leave-one-run-out ridge.\n"
                f"  n_trials={n_trials}, n_runs={unique.size}, min_trials_per_run={min_per_run}\n"
                f"  recommended: n_trials>={self.warn_min_trials} and min_trials_per_run>={self.warn_min_trials_per_run}\n"
                "  Results may be noisy/unstable."
            )

    def _summarize_raw_scores(self, raw_pearson: np.ndarray, raw_r2: np.ndarray, ceiling: np.ndarray) -> dict:
        metric = RidgeMetric(ceiling=ceiling)
        out = {}
        for key, raw in [("pearson", raw_pearson), ("r2", raw_r2)]:
            ceiled = metric.apply_ceiling(raw)
            out[f"raw_{key}"] = raw
            out[f"ceiled_{key}"] = ceiled
            out[f"median_unceiled_{key}"] = np.median(raw, axis=1)
            out[f"median_ceiled_{key}"] = np.median(ceiled, axis=1)
            out[f"final_{key}"] = float(np.mean(out[f"median_ceiled_{key}"]))
            out[f"final_unceiled_{key}"] = float(np.mean(out[f"median_unceiled_{key}"]))
        return out

    def _run_ridge_loro(self, X: np.ndarray, Y: np.ndarray, groups: np.ndarray, ceiling: np.ndarray, return_per_run: bool = False):
        # Use a RidgeCV per fold to pick alpha, then score on held-out run.
        alphas = [
            1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
            0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10
        ]

        scoring_pearson = []
        scoring_r2 = []
        heldout_runs = []

        gkf = GroupKFold(n_splits=len(np.unique(groups)))
        for train_idx, test_idx in gkf.split(X, Y, groups=groups):
            Xtr, Xte = X[train_idx], X[test_idx]
            Ytr, Yte = Y[train_idx], Y[test_idx]

            model = RidgeCV(alphas=alphas, store_cv_results=False)
            model.fit(Xtr, Ytr)
            Yhat = model.predict(Xte)

            # per-target scores
            pear = np.array([_pearson_safe(Yte[:, i], Yhat[:, i]) for i in range(Yte.shape[1])])
            r2 = np.array([r2_score(Yte[:, i], Yhat[:, i]) for i in range(Yte.shape[1])])
            scoring_pearson.append(pear)
            scoring_r2.append(r2)
            heldout_runs.append(int(np.unique(groups[test_idx])[0]))

        # Stack folds: (n_folds, n_targets)
        raw_pearson = np.stack(scoring_pearson, axis=0)
        raw_r2 = np.stack(scoring_r2, axis=0)
        out = self._summarize_raw_scores(raw_pearson, raw_r2, ceiling)
        if return_per_run:
            out["heldout_runs"] = np.array(heldout_runs, dtype=int)
        return out

    def _run_ridge_fixed_train_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        ceiling: np.ndarray,
    ) -> dict:
        alphas = [
            1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
            0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10
        ]
        Xtr, Ytr = X[train_mask], Y[train_mask]
        Xte, Yte = X[test_mask], Y[test_mask]
        if Xtr.shape[0] < 2 or Xte.shape[0] < 2:
            raise RuntimeError("Not enough trials in train/test split for ridge.")

        model = RidgeCV(alphas=alphas, store_cv_results=False)
        model.fit(Xtr, Ytr)
        Yhat = model.predict(Xte)
        raw_pearson = np.array([_pearson_safe(Yte[:, i], Yhat[:, i]) for i in range(Yte.shape[1])])[None, :]
        raw_r2 = np.array([r2_score(Yte[:, i], Yhat[:, i]) for i in range(Yte.shape[1])])[None, :]
        return self._summarize_raw_scores(raw_pearson, raw_r2, ceiling)

    def run(self):
        src = self.source_assembly_class(debug=self.debug)
        tgt = self.target_assembly_class(debug=self.debug)

        # RouteLearning assembly returns (X, ceiling, groups, keys)
        X, _, _, X_keys = src.get_assembly(train=True)
        Y, ceiling, groups, Y_keys = tgt.get_assembly(train=True)

        X_al, Y_al, groups_al = self._align_by_keys(X, Y, groups, X_keys, Y_keys)
        self._warn_if_too_few_trials(groups_al, X_al.shape[0])

        results = {}
        for metric_name in (self.metrics.keys() or ["ridge"]):
            if metric_name != "ridge":
                print(f"Warning: RouteLearningLOROBenchmark currently supports only 'ridge'. Ignoring '{metric_name}'.")
                continue
            results["ridge"] = self._run_ridge_loro(X_al, Y_al, groups_al, ceiling)

        return {"metrics": results.get("ridge", {}), "ceiling": ceiling}


# ──────────────────────────────────────────────────────────────
# Experiment 1: Bilateral hippocampus, subject pairs
# ──────────────────────────────────────────────────────────────

class RouteLearningExp1s01toExp1s02Hippo(RouteLearningLOROBenchmark):
    """Exp1 sub01 → sub02, bilateral hippocampus (ridge: brain→brain)."""
    def __init__(self, debug: bool = False):
        super().__init__(RouteLearningExp1s01Hippo, RouteLearningExp1s02Hippo, debug=debug)


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s02Hippo"] = (
    RouteLearningExp1s01toExp1s02Hippo
)


# ──────────────────────────────────────────────────────────────
# Learning analyses for Exp1s01 → Exp1s02 (ideas 1–3)
# ──────────────────────────────────────────────────────────────

class RouteLearningExp1s01toExp1s02Hippo_LearningCurve(RouteLearningLOROBenchmark):
    """
    Idea 1: per-run learning curve.
    Returns per-heldout-run scores in addition to overall aggregates.
    """
    def __init__(self, debug: bool = False):
        super().__init__(RouteLearningExp1s01Hippo, RouteLearningExp1s02Hippo, debug=debug)

    def run(self):
        src = self.source_assembly_class(debug=self.debug)
        tgt = self.target_assembly_class(debug=self.debug)
        X, _, _, X_keys = src.get_assembly(train=True)
        Y, ceiling, groups, Y_keys = tgt.get_assembly(train=True)
        X_al, Y_al, groups_al = self._align_by_keys(X, Y, groups, X_keys, Y_keys)
        self._warn_if_too_few_trials(groups_al, X_al.shape[0])

        out = self._run_ridge_loro(X_al, Y_al, groups_al, ceiling, return_per_run=True)
        # Also expose per-run final scores (median over targets) for convenience.
        heldout = out["heldout_runs"]
        per_run = {}
        for i, r in enumerate(heldout.tolist()):
            per_run[int(r)] = {
                "final_pearson": float(np.median(out["ceiled_pearson"][i])),
                "final_r2": float(np.median(out["ceiled_r2"][i])),
            }
        out["per_run"] = per_run
        return {"metrics": out, "ceiling": ceiling}


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s02Hippo_LearningCurve"] = (
    RouteLearningExp1s01toExp1s02Hippo_LearningCurve
)


class RouteLearningExp1s01toExp1s02Hippo_EarlyVsLateLORO(RouteLearningLOROBenchmark):
    """
    Idea 2: early vs late learning comparison, each evaluated with LORO restricted to that phase.
    Early runs: 1–7, Late runs: 8–14.
    """
    def __init__(self, debug: bool = False):
        super().__init__(RouteLearningExp1s01Hippo, RouteLearningExp1s02Hippo, debug=debug)

    def run(self):
        src = self.source_assembly_class(debug=self.debug)
        tgt = self.target_assembly_class(debug=self.debug)
        X, _, _, X_keys = src.get_assembly(train=True)
        Y, ceiling, groups, Y_keys = tgt.get_assembly(train=True)
        X_al, Y_al, groups_al = self._align_by_keys(X, Y, groups, X_keys, Y_keys)

        early_mask = (groups_al >= 1) & (groups_al <= 7)
        late_mask = (groups_al >= 8) & (groups_al <= 14)

        res = {}
        if np.unique(groups_al[early_mask]).size >= 2:
            self._warn_if_too_few_trials(groups_al[early_mask], int(early_mask.sum()))
            res["early"] = self._run_ridge_loro(X_al[early_mask], Y_al[early_mask], groups_al[early_mask], ceiling, return_per_run=True)
        else:
            res["early"] = {"error": "Not enough early runs after alignment to run LORO."}

        if np.unique(groups_al[late_mask]).size >= 2:
            self._warn_if_too_few_trials(groups_al[late_mask], int(late_mask.sum()))
            res["late"] = self._run_ridge_loro(X_al[late_mask], Y_al[late_mask], groups_al[late_mask], ceiling, return_per_run=True)
        else:
            res["late"] = {"error": "Not enough late runs after alignment to run LORO."}

        return {"metrics": res, "ceiling": ceiling}


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s02Hippo_EarlyVsLateLORO"] = (
    RouteLearningExp1s01toExp1s02Hippo_EarlyVsLateLORO
)


class RouteLearningExp1s01toExp1s02Hippo_EarlyLateTransfer(RouteLearningLOROBenchmark):
    """
    Idea 3: train on early runs, test on late runs (and reverse).
    Uses a fixed train/test split (no CV) to assess representational stability/transfer.
    """
    def __init__(self, debug: bool = False):
        super().__init__(RouteLearningExp1s01Hippo, RouteLearningExp1s02Hippo, debug=debug)

    def run(self):
        src = self.source_assembly_class(debug=self.debug)
        tgt = self.target_assembly_class(debug=self.debug)
        X, _, _, X_keys = src.get_assembly(train=True)
        Y, ceiling, groups, Y_keys = tgt.get_assembly(train=True)
        X_al, Y_al, groups_al = self._align_by_keys(X, Y, groups, X_keys, Y_keys)

        early = (groups_al >= 1) & (groups_al <= 7)
        late = (groups_al >= 8) & (groups_al <= 14)

        self._warn_if_too_few_trials(groups_al, X_al.shape[0])

        res = {}
        try:
            res["train_early_test_late"] = self._run_ridge_fixed_train_test(X_al, Y_al, train_mask=early, test_mask=late, ceiling=ceiling)
        except Exception as e:
            res["train_early_test_late"] = {"error": str(e)}
        try:
            res["train_late_test_early"] = self._run_ridge_fixed_train_test(X_al, Y_al, train_mask=late, test_mask=early, ceiling=ceiling)
        except Exception as e:
            res["train_late_test_early"] = {"error": str(e)}

        return {"metrics": res, "ceiling": ceiling}


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s02Hippo_EarlyLateTransfer"] = (
    RouteLearningExp1s01toExp1s02Hippo_EarlyLateTransfer
)


class RouteLearningExp1s02toExp1s01Hippo(RouteLearningLOROBenchmark):
    """Exp1 sub02 → sub01, bilateral hippocampus (reverse direction)."""
    def __init__(self, debug: bool = False):
        super().__init__(RouteLearningExp1s02Hippo, RouteLearningExp1s01Hippo, debug=debug)


BENCHMARK_REGISTRY["RouteLearningExp1s02toExp1s01Hippo"] = (
    RouteLearningExp1s02toExp1s01Hippo
)


class RouteLearningExp1s01toExp1s03Hippo(RouteLearningLOROBenchmark):
    """Exp1 sub01 → sub03, bilateral hippocampus."""
    def __init__(self, debug: bool = False):
        super().__init__(RouteLearningExp1s01Hippo, RouteLearningExp1s03Hippo, debug=debug)


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s03Hippo"] = (
    RouteLearningExp1s01toExp1s03Hippo
)


class RouteLearningExp1s01toExp1s04Hippo(RouteLearningLOROBenchmark):
    """Exp1 sub01 → sub04, bilateral hippocampus."""
    def __init__(self, debug: bool = False):
        super().__init__(RouteLearningExp1s01Hippo, RouteLearningExp1s04Hippo, debug=debug)


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s04Hippo"] = (
    RouteLearningExp1s01toExp1s04Hippo
)


class RouteLearningExp1s01toExp1s05Hippo(RouteLearningLOROBenchmark):
    """Exp1 sub01 → sub05, bilateral hippocampus."""
    def __init__(self, debug: bool = False):
        super().__init__(RouteLearningExp1s01Hippo, RouteLearningExp1s05Hippo, debug=debug)


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s05Hippo"] = (
    RouteLearningExp1s01toExp1s05Hippo
)


# ──────────────────────────────────────────────────────────────
# Experiment 1: Left hippocampus
# ──────────────────────────────────────────────────────────────

class RouteLearningExp1s01toExp1s02LeftHippo(RouteLearningLOROBenchmark):
    """Exp1 sub01 → sub02, left hippocampus only."""
    def __init__(self, debug: bool = False):
        super().__init__(RouteLearningExp1s01LeftHippo, RouteLearningExp1s02LeftHippo, debug=debug)


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s02LeftHippo"] = (
    RouteLearningExp1s01toExp1s02LeftHippo
)


# ──────────────────────────────────────────────────────────────
# Experiment 1: Right hippocampus
# ──────────────────────────────────────────────────────────────

class RouteLearningExp1s01toExp1s02RightHippo(RouteLearningLOROBenchmark):
    """Exp1 sub01 → sub02, right hippocampus only."""
    def __init__(self, debug: bool = False):
        super().__init__(RouteLearningExp1s01RightHippo, RouteLearningExp1s02RightHippo, debug=debug)


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s02RightHippo"] = (
    RouteLearningExp1s01toExp1s02RightHippo
)


# ──────────────────────────────────────────────────────────────
# Experiment 2: Bilateral hippocampus
# ──────────────────────────────────────────────────────────────

class RouteLearningExp2s01toExp2s02Hippo(RouteLearningLOROBenchmark):
    """Exp2 sub01 → sub02, bilateral hippocampus."""
    def __init__(self, debug: bool = False):
        super().__init__(RouteLearningExp2s01Hippo, RouteLearningExp2s02Hippo, debug=debug)


BENCHMARK_REGISTRY["RouteLearningExp2s01toExp2s02Hippo"] = (
    RouteLearningExp2s01toExp2s02Hippo
)


class RouteLearningExp2s02toExp2s01Hippo(RouteLearningLOROBenchmark):
    """Exp2 sub02 → sub01, bilateral hippocampus (reverse)."""
    def __init__(self, debug: bool = False):
        super().__init__(RouteLearningExp2s02Hippo, RouteLearningExp2s01Hippo, debug=debug)


BENCHMARK_REGISTRY["RouteLearningExp2s02toExp2s01Hippo"] = (
    RouteLearningExp2s02toExp2s01Hippo
)


class RouteLearningExp2s01toExp2s03Hippo(RouteLearningLOROBenchmark):
    """Exp2 sub01 → sub03, bilateral hippocampus."""
    def __init__(self, debug: bool = False):
        super().__init__(RouteLearningExp2s01Hippo, RouteLearningExp2s03Hippo, debug=debug)


BENCHMARK_REGISTRY["RouteLearningExp2s01toExp2s03Hippo"] = (
    RouteLearningExp2s01toExp2s03Hippo
)


# ──────────────────────────────────────────────────────────────
# Auto-registration for all subject→subject mappings (Option A)
# ──────────────────────────────────────────────────────────────

def _make_fixed_subject_assembly(subject_label: str, region: str):
    """
    Create a lightweight assembly subclass bound to a specific subject + region.
    subject_label: e.g. 'Exp1s01' (without 'sub-')
    region: 'hippocampus' | 'left_hippocampus' | 'right_hippocampus'
    """
    subject_id = f"sub-{subject_label}"

    class _Asm(RouteLearningConfigurable):
        def __init__(self, **kwargs):
            # Preserve debug and any overrides passed by benchmark runner
            super().__init__(subject=subject_id, region=region, **kwargs)

    # Give it a stable name for debugging/logging
    suffix = (
        "Hippo" if region == "hippocampus"
        else ("LeftHippo" if region == "left_hippocampus" else "RightHippo")
    )
    _Asm.__name__ = f"RouteLearning{subject_label}{suffix}"
    _Asm.__qualname__ = _Asm.__name__
    return _Asm


def _make_mapping_benchmark_class(src_asm, tgt_asm, bench_name: str):
    """
    Create a RouteLearningLOROBenchmark subclass for a specific mapping.
    """
    class _Bench(RouteLearningLOROBenchmark):
        def __init__(self, debug: bool = False):
            super().__init__(src_asm, tgt_asm, debug=debug)

    _Bench.__name__ = bench_name
    _Bench.__qualname__ = bench_name
    return _Bench


def _make_learningcurve_benchmark_class(src_asm, tgt_asm, bench_name: str):
    class _Bench(RouteLearningLOROBenchmark):
        def __init__(self, debug: bool = False):
            super().__init__(src_asm, tgt_asm, debug=debug)

        def run(self):
            src = self.source_assembly_class(debug=self.debug)
            tgt = self.target_assembly_class(debug=self.debug)
            X, _, _, X_keys = src.get_assembly(train=True)
            Y, ceiling, groups, Y_keys = tgt.get_assembly(train=True)
            X_al, Y_al, groups_al = self._align_by_keys(X, Y, groups, X_keys, Y_keys)
            self._warn_if_too_few_trials(groups_al, X_al.shape[0])

            out = self._run_ridge_loro(X_al, Y_al, groups_al, ceiling, return_per_run=True)
            heldout = out["heldout_runs"]
            per_run = {}
            for i, r in enumerate(heldout.tolist()):
                per_run[int(r)] = {
                    "final_pearson": float(np.median(out["ceiled_pearson"][i])),
                    "final_r2": float(np.median(out["ceiled_r2"][i])),
                }
            out["per_run"] = per_run
            return {"metrics": out, "ceiling": ceiling}

    _Bench.__name__ = bench_name
    _Bench.__qualname__ = bench_name
    return _Bench


def _make_earlyvslate_loro_benchmark_class(src_asm, tgt_asm, bench_name: str):
    class _Bench(RouteLearningLOROBenchmark):
        def __init__(self, debug: bool = False):
            super().__init__(src_asm, tgt_asm, debug=debug)

        def run(self):
            src = self.source_assembly_class(debug=self.debug)
            tgt = self.target_assembly_class(debug=self.debug)
            X, _, _, X_keys = src.get_assembly(train=True)
            Y, ceiling, groups, Y_keys = tgt.get_assembly(train=True)
            X_al, Y_al, groups_al = self._align_by_keys(X, Y, groups, X_keys, Y_keys)

            early_mask = (groups_al >= 1) & (groups_al <= 7)
            late_mask = (groups_al >= 8) & (groups_al <= 14)

            res = {}
            if np.unique(groups_al[early_mask]).size >= 2:
                self._warn_if_too_few_trials(groups_al[early_mask], int(early_mask.sum()))
                res["early"] = self._run_ridge_loro(
                    X_al[early_mask], Y_al[early_mask], groups_al[early_mask], ceiling, return_per_run=True
                )
            else:
                res["early"] = {"error": "Not enough early runs after alignment to run LORO."}

            if np.unique(groups_al[late_mask]).size >= 2:
                self._warn_if_too_few_trials(groups_al[late_mask], int(late_mask.sum()))
                res["late"] = self._run_ridge_loro(
                    X_al[late_mask], Y_al[late_mask], groups_al[late_mask], ceiling, return_per_run=True
                )
            else:
                res["late"] = {"error": "Not enough late runs after alignment to run LORO."}

            return {"metrics": res, "ceiling": ceiling}

    _Bench.__name__ = bench_name
    _Bench.__qualname__ = bench_name
    return _Bench


def _make_earlylate_transfer_benchmark_class(src_asm, tgt_asm, bench_name: str):
    class _Bench(RouteLearningLOROBenchmark):
        def __init__(self, debug: bool = False):
            super().__init__(src_asm, tgt_asm, debug=debug)

        def run(self):
            src = self.source_assembly_class(debug=self.debug)
            tgt = self.target_assembly_class(debug=self.debug)
            X, _, _, X_keys = src.get_assembly(train=True)
            Y, ceiling, groups, Y_keys = tgt.get_assembly(train=True)
            X_al, Y_al, groups_al = self._align_by_keys(X, Y, groups, X_keys, Y_keys)

            early = (groups_al >= 1) & (groups_al <= 7)
            late = (groups_al >= 8) & (groups_al <= 14)

            self._warn_if_too_few_trials(groups_al, X_al.shape[0])

            res = {}
            try:
                res["train_early_test_late"] = self._run_ridge_fixed_train_test(
                    X_al, Y_al, train_mask=early, test_mask=late, ceiling=ceiling
                )
            except Exception as e:
                res["train_early_test_late"] = {"error": str(e)}
            try:
                res["train_late_test_early"] = self._run_ridge_fixed_train_test(
                    X_al, Y_al, train_mask=late, test_mask=early, ceiling=ceiling
                )
            except Exception as e:
                res["train_late_test_early"] = {"error": str(e)}

            return {"metrics": res, "ceiling": ceiling}

    _Bench.__name__ = bench_name
    _Bench.__qualname__ = bench_name
    return _Bench


def _register_all_routelearning_mappings():
    # Subjects in ds000217 as used by data/RouteLearning.py
    exp1 = [f"Exp1s{i:02d}" for i in range(1, 21)]
    exp2 = [f"Exp2s{i:02d}" for i in range(1, 22)]
    all_subjects = exp1 + exp2

    region_specs = [
        ("hippocampus", "Hippo"),
        ("left_hippocampus", "LeftHippo"),
        ("right_hippocampus", "RightHippo"),
    ]

    # Precreate assembly classes so we don't build them repeatedly
    assemblies = {}
    for subj in all_subjects:
        for region, suffix in region_specs:
            assemblies[(subj, suffix)] = _make_fixed_subject_assembly(subj, region)

    # Register every ordered pair (src != tgt) for each region
    for (region, suffix) in region_specs:
        for src in all_subjects:
            for tgt in all_subjects:
                if src == tgt:
                    continue
                src_asm = assemblies[(src, suffix)]
                tgt_asm = assemblies[(tgt, suffix)]

                # Overall
                name = f"RouteLearning{src}to{tgt}{suffix}"
                if name not in BENCHMARK_REGISTRY:
                    BENCHMARK_REGISTRY[name] = _make_mapping_benchmark_class(src_asm, tgt_asm, name)

                # Idea 1: learning curve
                name_lc = f"RouteLearning{src}to{tgt}{suffix}_LearningCurve"
                if name_lc not in BENCHMARK_REGISTRY:
                    BENCHMARK_REGISTRY[name_lc] = _make_learningcurve_benchmark_class(src_asm, tgt_asm, name_lc)

                # Idea 2: early vs late LORO
                name_el = f"RouteLearning{src}to{tgt}{suffix}_EarlyVsLateLORO"
                if name_el not in BENCHMARK_REGISTRY:
                    BENCHMARK_REGISTRY[name_el] = _make_earlyvslate_loro_benchmark_class(src_asm, tgt_asm, name_el)

                # Idea 3: early/late transfer
                name_tr = f"RouteLearning{src}to{tgt}{suffix}_EarlyLateTransfer"
                if name_tr not in BENCHMARK_REGISTRY:
                    BENCHMARK_REGISTRY[name_tr] = _make_earlylate_transfer_benchmark_class(src_asm, tgt_asm, name_tr)


# Execute registration on import
_register_all_routelearning_mappings()