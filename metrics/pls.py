from typing import List, Optional, Dict, Union
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, make_scorer

from .base import BaseMetric
from .utils import run_kfold_cv, pearson_correlation_scorer


class PLSMetric(BaseMetric):
    def __init__(
        self,
        n_components_options: List[int] = [
            5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        ceiling: Optional[float] = None,
    ):
        super().__init__(ceiling)
        self.n_components_options = n_components_options

    def compute_raw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        # Define a scorer that computes the average Pearson correlation across output dimensions.
        def avg_pearson(y_true, y_pred):
            return np.mean([
                pearson_correlation_scorer(y_true[:, i], y_pred[:, i])
                for i in range(y_true.shape[1])
            ])

        scorer = make_scorer(avg_pearson, greater_is_better=True)

        # Create a single train/validation split (90/10)
        indices = np.arange(source.shape[0])
        train_indices, val_indices = train_test_split(
            indices, test_size=0.1, random_state=42)
        cv_split = [(train_indices, val_indices)]

        param_grid = {'n_components': self.n_components_options}
        grid_search = GridSearchCV(
            estimator=PLSRegression(),
            param_grid=param_grid,
            scoring=scorer,
            cv=cv_split,   # Use our custom 90/10 split
            verbose=3
        )
        grid_search.fit(source, target)
        best_n = grid_search.best_params_['n_components']

        # Define scoring functions for final evaluation.
        scoring_funcs = {
            "pearson": lambda y_true, y_pred: np.array([
                pearson_correlation_scorer(y_true[:, i], y_pred[:, i])
                for i in range(y_true.shape[1])
            ]),
            "r2": lambda y_true, y_pred: np.array([
                r2_score(y_true[:, i], y_pred[:, i])
                for i in range(y_true.shape[1])
            ]),
        }

        # Use the best n_components found in the grid search.
        def model_factory(): return PLSRegression(n_components=best_n)

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
