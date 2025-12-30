from typing import Optional, Dict, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from .base import BaseMetric
from .utils import run_kfold_cv


class OneToOneMappingMetric(BaseMetric):
    def __init__(self, score_type: str = "pearson", ceiling: Optional[float] = None):
        super().__init__(ceiling)
        self.score_type = score_type

    def compute_raw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:

        def one_to_one_score(X, Y) -> float:
            d1 = X.shape[1]
            d2 = Y.shape[1]
            d = min(d1, d2)
            corr_matrix = np.zeros((d, d))
            for i in range(d):
                for j in range(d):
                    r, _ = pearsonr(X[:, i], Y[:, j])
                    corr_matrix[i, j] = r
            row_ind, col_ind = linear_sum_assignment(-corr_matrix)
            matched_corrs = corr_matrix[row_ind, col_ind]
            return np.mean(matched_corrs)

        if test_source is not None and test_target is not None:
            return {self.score_type: np.array([one_to_one_score(test_source, test_target)])}

        def dummy_model_factory():
            return None

        scoring_funcs = {self.score_type: one_to_one_score}
        return run_kfold_cv(dummy_model_factory, source, target, scoring_funcs, stratify_on=stratify_on)

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
