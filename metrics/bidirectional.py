from typing import List, Optional, Dict, Union
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score

from .base import BaseMetric
from .utils import pearson_correlation_scorer


class BidirectionalMappingMetric(BaseMetric):
    def __init__(
        self,
        alpha_options: List[float] = [
            1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
            0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10
        ],
        ceiling: Optional[float] = None,
    ):
        super().__init__(ceiling)
        self.alpha_options = alpha_options
        self.score_type = score_type

    def gridsearch_ridge(self, X: np.ndarray, y: np.ndarray) -> float:
        ridge_cv = RidgeCV(alphas=self.alpha_options,
                           store_cv_values=True, alpha_per_target=True)
        ridge_cv.fit(X, y)
        return ridge_cv.alpha_

    def compute_raw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:

        scoring_funcs = {
            "pearson": lambda y_true, y_pred: np.array([pearson_correlation_scorer(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]),
            "r2": lambda y_true, y_pred: np.array([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]),
        }

        best_alpha_st = self.gridsearch_ridge(source, target)
        best_alpha_ts = self.gridsearch_ridge(target, source)

        if test_source is not None and test_target is not None:
            model_st = RidgeCV(alphas=self.alpha_options, store_cv_results=True,
                               alpha_per_target=True).fit(source, target)
            preds_st = model_st.predict(test_source)

            model_ts = RidgeCV(alphas=self.alpha_options, store_cv_results=True,
                               alpha_per_target=True).fit(target, source)
            preds_ts = model_ts.predict(test_target)

            scores = {}
            for scoring_func in scoring_funcs:
                score_st = scoring_funcs[scoring_func](test_target, preds_st)
                score_ts = scoring_funcs[scoring_func](test_source, preds_ts)
                scores[scoring_func] = np.array([[score_st], [score_ts]])

            return scores

        else:
            def bidirectional_model_factory(alpha_st, alpha_ts):
                def model_factory():
                    return {
                        'st': RidgeCV(alphas=self.alpha_options, store_cv_results=True, alpha_per_target=True),
                        'ts': RidgeCV(alphas=self.alpha_options, store_cv_results=True, alpha_per_target=True)
                    }
                return model_factory

            def bidirectional_scoring_func(y_true_st, y_pred_st, y_true_ts, y_pred_ts, chosen_scoring_func):
                st_score = chosen_scoring_func(y_true_st, y_pred_st)
                ts_score = chosen_scoring_func(y_true_ts, y_pred_ts)
                return np.array([[score_st], [score_ts]])

            def run_bidirectional_kfold(model_factory, X_s, y_s, X_t, y_t, n_splits=10, random_state=42, stratify_on=None):
                if stratify_on is not None:
                    if len(stratify_on) != X.shape[0]:
                        raise ValueError(
                            "Length of stratify_on must match X and y for StratifiedKFold.")
                    print(f"Using Stratified K-Fold with {n_splits} splits.")
                    kf = StratifiedKFold(
                        n_splits=n_splits, shuffle=True, random_state=random_state)
                    # Pass stratification target
                    split_iterator = kf.split(X, stratify_on)
                else:
                    print(f"Using Standard K-Fold with {n_splits} splits.")
                    kf = KFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
                    split_iterator = kf.split(X)  # Standard split

                scores_r2, scores_pearson = [], []

                for train_idx, val_idx in tqdm(split_iterator, total=n_splits, desc="Folds"):
                    X_train_s, X_val_s = X_s[train_idx], X_s[val_idx]
                    y_train_s, y_val_s = y_s[train_idx], y_s[val_idx]
                    X_train_t, X_val_t = X_t[train_idx], X_t[val_idx]
                    y_train_t, y_val_t = y_t[train_idx], y_t[val_idx]

                    model = model_factory()
                    model['st'].fit(X_train_s, y_train_s)
                    model['ts'].fit(X_train_t, y_train_t)
                    fold_preds_st = model['st'].predict(X_val_s)
                    fold_preds_ts = model['ts'].predict(X_val_t)

                    scores_r2.append(bidirectional_scoring_func(
                        y_val_s, fold_preds_st, y_val_t, fold_preds_ts, scoring_funcs['r2']))
                    scores_pearson.append(bidirectional_scoring_func(
                        y_val_s, fold_preds_st, y_val_t, fold_preds_ts, scoring_funcs['pearson']))
                return {'pearson': np.array(scores_pearson), 'r2': np.array(scores_r2)}

            model_factory = bidirectional_model_factory(
                best_alpha_st, best_alpha_ts)
            return run_bidirectional_kfold(model_factory, source, target, target, source, stratify_on)

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, float]]:

        raw_scores = self.compute_raw(
            source, target, test_source, test_target, stratify_on)
        if not isinstance(raw_scores, dict):
            return raw_scores  # Early return (for RSA, etc.)

        processed_scores = {}
        for key, value in raw_scores.items():
            ceiled_scores = self.apply_ceiling(value)
            ceiled_median_scores = (
                np.median(ceiled_scores, axis=1)
                if ceiled_scores.ndim > 1
                else ceiled_scores
            )
            unceiled_median_scores = (
                np.median(value, axis=1)
                if value.ndim > 1
                else value
            )
            final_ceiled_score = np.mean(ceiled_median_scores)
            final_unceiled_score = np.mean(unceiled_median_scores)

            if key not in ['preds', 'targets']:
                processed_scores.update({
                    f"raw_{key}": value,
                    f"ceiled_{key}": ceiled_scores,
                    f"median_unceiled_{key}": unceiled_median_scores,
                    f"median_ceiled_{key}": ceiled_median_scores,
                    f"final_{key}": final_ceiled_score,
                    f"final_unceiled_{key}": final_unceiled_score,
                })
            else:
                processed_scores.update({
                    f"raw_{key}": value,
                })

        return processed_scores
