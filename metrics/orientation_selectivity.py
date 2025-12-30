from typing import List, Optional, Dict, Callable, Union

from .base import BaseMetric

import numpy as np
import xarray as xr


class OrientationSelectivity(BaseMetric):
    def __init__(
        self,
        ceiling: Optional[float] = None,
    ):
        super().__init__(ceiling)

    def compute_raw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:

        features = np.array(source)
        labels = np.array(target)

        responses = xr.DataArray(
            data=self.flatten(features),
            coords={
                "angles": ("image_idx", labels[:, 0]),
                "sfs": ("image_idx", labels[:, 1]),
                "phases": ("image_idx", labels[:, 2]),
                "colors": ("image_idx", labels[:, 3]),
            },
            dims=["image_idx", "unit_idx"],
        )

        cv_results = self._get_circular_variance(responses)
        preference_results = self._get_preference(responses)

        return cv_results | preference_results

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
        return raw_scores

    def flatten(self, x: np.ndarray):
        """
        Flatten matrix along all dims but first
        """
        return x.reshape((len(x), -1))

    def ringach_norm(self, otc):
        n_angles = 8

        # the angles we use evenly span 0 to pi, but do not wrap
        angles = np.linspace(0, np.pi, n_angles + 1)[:-1]

        # compute "R"
        numerator = np.sum(
            otc * np.exp(angles * 2 * 1j), axis=1
        )

        denominator = np.sum(otc, axis=1)
        R = numerator / denominator

        # compute circular variance
        CV = 1 - np.abs(R)

        return CV

    def _get_circular_variance(self, responses):
        otc = responses.groupby("angles").mean().T
        x = np.array(otc)

        # shift tuning curves to have min=0 to counter the effect
        # of layer normalization in vision transformers
        x = x + np.abs(np.min(x, axis=1, keepdims=True))

        cv = self.ringach_norm(x)

        per_selective_thresh_mean = np.mean(cv < np.nanmean(cv)) * 100
        per_selective_thresh_06 = np.mean(cv < 0.6) * 100

        data = {
            "cv": cv.tolist(),
            "mean_cv": np.nanmean(cv).item(),
            "median_cv": np.nanmedian(cv).item(),
            "per_selective_thresh_mean": per_selective_thresh_mean.item(),
            "per_selective_thresh_06": per_selective_thresh_06.item(),
        }

        return data

    def _get_preference(self, responses):
        angles = np.linspace(0, 180, 9)[:-1]

        raw_ori_pref = responses.groupby("angles").mean().argmax(axis=0).data
        orientation_preferences = np.array(
            [angles[x] for x in raw_ori_pref]
        )

        considered_orientations = [0, 45, 90, 135, 180]

        counts = [
            np.sum(orientation_preferences == 0),
            np.sum(orientation_preferences == 45),
            np.sum(orientation_preferences == 90),
            np.sum(orientation_preferences == 135),
            np.sum(orientation_preferences == 0),  # 0 and 180 are same
        ]

        total = sum(counts)
        percentages = [(100 * c / total).item() for c in counts]
        cardinality_index = (counts[0] + counts[2] + counts[4]) / sum(counts)

        data = {
            "orientations": considered_orientations,
            "preference_percent": percentages,
            "cardinality_index": cardinality_index.item(),
        }

        return data
