from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Callable, Union
import numpy as np

from .utils import run_kfold_cv, pearson_correlation_scorer


class BaseMetric(ABC):
    """Base class for all metrics."""

    # Allow ceiling to be array
    def __init__(self, ceiling: Optional[np.ndarray] = None):
        self.ceiling = ceiling
        if self.ceiling is not None:
            # Ensure ceiling is a numpy array for element-wise operations
            self.ceiling = np.asarray(self.ceiling)
            # Replace zeros and very small values in ceiling to avoid division by zero
            self.ceiling[self.ceiling <= 1e-6] = 1e-6

    @abstractmethod
    def compute_raw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Union[Dict[str, np.ndarray], float]:
        pass

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, float]]:
        pass

    def apply_ceiling(self, scores: np.ndarray) -> np.ndarray:
        if self.ceiling is not None:
            return scores / self.ceiling
        return scores
