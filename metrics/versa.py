from typing import List, Optional, Dict, Union
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.spatial.distance import pdist

from .base import BaseMetric
from .utils import run_kfold_cv, pearsonr


class VeRSAMetric(BaseMetric):
    def __init__(self, alpha_options: List[float] = [0.1, 1.0, 10.0],  ceiling: Optional[float] = None, score_type: str = "pearson"):
        super().__init__(ceiling)
        self.alpha_options = alpha_options
        self.score_type = score_type

    def compute_raw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:

        def compute_rdm(X):
            return pdist(X, metric="correlation")

        def compute_versar_score(X, Y) -> float:
            rdm_pred = compute_rdm(X)
            rdm_test = compute_rdm(Y)
            return pearsonr(rdm_pred, rdm_test)[0]

        if test_source is not None and test_target is not None:
            ridge_cv = RidgeCV(alphas=self.alpha_options, cv=KFold(
                n_splits=5, shuffle=True, random_state=42))
            ridge_cv.fit(source, target)
            preds_test = ridge_cv.predict(test_source)
            return {self.score_type: np.array([compute_versar_score(preds_test, test_target)])}

        def model_factory():
            ridge_cv = RidgeCV(alphas=self.alpha_options, cv=KFold(
                n_splits=5, shuffle=True, random_state=42))
            return ridge_cv

        scoring_funcs = {self.score_type: compute_versar_score}
        return run_kfold_cv(model_factory, source, target, scoring_funcs, stratify_on=stratify_on)

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
