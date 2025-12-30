from typing import Optional, Dict
import numpy as np
from scipy.stats import pearsonr

from .base import BaseMetric
from .utils import run_kfold_cv


class SemiMatchingMetric(BaseMetric):
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

        def semi_match_score(X, Y) -> float:
            d1 = X.shape[1]
            d2 = Y.shape[1]
            corr_matrix = np.zeros((d1, d2))
            for i in range(d1):
                for j in range(d2):
                    r, _ = pearsonr(X[:, i], Y[:, j])
                    corr_matrix[i, j] = r
            row_max = corr_matrix.max(axis=1).mean()
            col_max = corr_matrix.max(axis=0).mean()
            return 0.5 * (row_max + col_max)

        if test_source is not None and test_target is not None:
            return {self.score_type: np.array([semi_match_score(test_source, test_target)])}

        def dummy_model_factory():
            return None

        scoring_funcs = {self.score_type: semi_match_score}
        return run_kfold_cv(dummy_model_factory, source, target, scoring_funcs, stratify_on=stratify_on)
