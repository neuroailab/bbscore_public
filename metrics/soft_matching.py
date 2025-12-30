import ot
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict, Union
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from .base import BaseMetric
from .utils import run_eval, run_kfold_cv, pearson_correlation_scorer


class SoftMatching():
    def __init__(self, correlation=False, max_iter=100000, reg=None):
        self.fitted = False
        self._correlation = correlation
        self.max_iter = max_iter
        self.reg = reg

    def fit(self, X, Y):
        if self._correlation:
            self._x_mean = X.mean(axis=0)
            self._y_mean = Y.mean(axis=0)
            self._x_norm = np.linalg.norm(X, axis=0)
            self._y_norm = np.linalg.norm(Y, axis=0)

        # compute the OT matrix
        if self._correlation:
            self.M = ot.dist(X.T, Y.T, metric="correlation")
            self.M = np.nan_to_num(self.M, nan=1)

            # clamp distances that blow up due to numerical issues
            # (such as columns with zero variance)
            self.M = np.clip(self.M, 0, 2)

            self.correlations = 1 - self.M
        else:
            # This is the squared euclidean distance matrix
            self.M = ot.dist(X.T, Y.T)

        # Constraint that the rows sum to 1/N_x and cols to 1/N_y
        self.Nx = self.M.shape[0]
        self.Ny = self.M.shape[1]
        a = np.ones(self.Nx)/self.Nx
        b = np.ones(self.Ny)/self.Ny

        # P is the optimal soft permutation
        if self.reg is not None:
            self.P = ot.sinkhorn(a, b, self.M, reg=self.reg,
                                 numItermax=self.max_iter)
        else:
            self.P = ot.emd(a, b, self.M, numItermax=self.max_iter)

        self.fitted = True

        return self

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return r2_score(Y, Y_pred, multioutput='raw_values')

    def predict(self, X):
        assert self.fitted, "Cannot make predictions if not fitted."
        if self._correlation:
            X_ = np.divide(X - self._x_mean, self._x_norm,
                           where=self._x_norm != 0)
        else:
            X_ = X
        if self._correlation:
            predicted_Y = X_ @ (self.P * self.correlations) * self.Ny
            predicted_Y = predicted_Y * self._y_norm + self._y_mean
        else:
            predicted_Y = X_ @ self.P * self.Ny
        return predicted_Y


class SoftMatchingEstimator(BaseEstimator):
    """
    A wrapper around SoftMatching()
    """

    def __init__(self, reg=None, correlation=False, max_iter=100000):
        self.reg = reg
        self.correlation = correlation
        self.max_iter = max_iter
        self.model = None

    def fit(self, X, Y):
        self.model = SoftMatching(
            correlation=self.correlation,
            max_iter=self.max_iter,
            reg=self.reg
        )
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class SoftMatchingMetric(BaseMetric):
    def __init__(
        self,
        reg_options: List[float] = [
            1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4
        ],
        correlation: bool = True,
        max_iter: int = 100000,
        ceiling: Optional[float] = None,
    ):
        super().__init__(ceiling)
        self.reg_options = reg_options
        self.correlation = correlation
        self.max_iter = max_iter

    def compute_raw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:

        if source.ndim > 2:
            N = source.shape[0]
            source = source.reshape(N, -1)
            if test_source is not None:
                N_test = test_source.shape[0]
                test_source = test_source.reshape(N_test, -1)

        indices = np.arange(source.shape[0])
        train_idx, val_idx = train_test_split(
            indices, test_size=0.1, random_state=42
        )

        X_train, X_val = source[train_idx], source[val_idx]
        Y_train, Y_val = target[train_idx], target[val_idx]

        best_reg = None
        best_score = -np.inf

        print("Tuning regularization parameter for soft matching...")
        for reg in self.reg_options:
            model = SoftMatchingEstimator(
                reg=reg,
                correlation=self.correlation,
                max_iter=self.max_iter,
            )
            model.fit(X_train, Y_train)
            pred = model.predict(X_val)

            score = np.mean([
                pearson_correlation_scorer(Y_val[:, i], pred[:, i])
                for i in range(Y_val.shape[1])
            ])

            print(f"reg: {reg}, pearson's r: {score:.4f}")

            if score > best_score:
                best_score = score
                best_reg = reg

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

        def model_factory():
            return SoftMatchingEstimator(
                reg=best_reg,
                correlation=self.correlation,
                max_iter=self.max_iter,
            )

        if test_source is None:
            return run_kfold_cv(model_factory, source, target, scoring_funcs,
                                stratify_on=stratify_on)
        return run_eval(model_factory, source, target, test_source,
                        test_target, scoring_funcs)

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, float]]:

        raw_scores = self.compute_raw(
            source, target, test_source, test_target, stratify_on
        )

        processed_scores = {}
        for key, value in raw_scores.items():
            ceiled_scores = self.apply_ceiling(value)
            ceiled_median_scores = (
                np.median(ceiled_scores, axis=1)
                if ceiled_scores.ndim > 1 else ceiled_scores
            )
            unceiled_median_scores = (
                np.median(value, axis=1)
                if value.ndim > 1 else value
            )

            processed_scores.update({
                f"raw_{key}": value,
                f"ceiled_{key}": ceiled_scores,
                f"median_unceiled_{key}": unceiled_median_scores,
                f"median_ceiled_{key}": ceiled_median_scores,
                f"final_{key}": float(np.mean(ceiled_median_scores)),
                f"final_unceiled_{key}": float(np.mean(unceiled_median_scores)),
            })

        return processed_scores
