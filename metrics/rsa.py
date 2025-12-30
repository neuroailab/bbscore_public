from .base import BaseMetric

import numpy as np
from typing import Optional, Dict, Union

import rsatoolbox
from rsatoolbox.data import Dataset, TemporalDataset
from rsatoolbox.rdm.calc import calc_rdm, calc_rdm_movie
from rsatoolbox.rdm.compare import compare

# Comparison methods to report
_COMPARISON_METHODS = ['spearman', 'corr', 'kendall', 'cosine']


class RSAMetric(BaseMetric):
    """
    Standard RSA with NaN‐mask unification:
      - source shape: (n_conditions, n_model_features)
      - target shape: (n_conditions, n_brain_channels)
    Reports multiple RDM comparison metrics, ensuring no mismatched NaNs.
    """

    def compute_raw(self, source: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        assert source.ndim == 2, f"source must be 2D, got {source.shape}"
        assert target.ndim == 2, f"target must be 2D, got {target.shape}"
        assert source.shape[0] == target.shape[0], "number of conditions must match"

        # 1) build RDMs
        n_cond = source.shape[0]
        ds_src = Dataset(measurements=source, obs_descriptors={
                         'conds': list(range(n_cond))})
        ds_tgt = Dataset(measurements=target, obs_descriptors={
                         'conds': list(range(n_cond))})
        rdm_src = calc_rdm(ds_src, method='correlation', descriptor='conds')
        rdm_tgt = calc_rdm(ds_tgt, method='correlation', descriptor='conds')

        # 2) extract vectors
        vec_src = rdm_src.get_vectors()
        vec_tgt = rdm_tgt.get_vectors()

        # 3) unify NaN‐mask so both have NaNs in exactly the same positions
        nan_mask = np.isnan(vec_src) | np.isnan(vec_tgt)
        vec_src[nan_mask] = np.nan
        vec_tgt[nan_mask] = np.nan

        # 4) compute each comparison, catching any remaining errors
        scores: Dict[str, float] = {}
        for method in _COMPARISON_METHODS:
            try:
                scores[method] = float(
                    compare(vec_src, vec_tgt, method=method))
            except ValueError:
                # if it still fails (e.g. too few valid entries), return NaN
                scores[method] = np.nan

        return scores

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        src = test_source if test_source is not None else source
        if src.ndim == 3:
            N = src.shape[0]
            src = src.reshape(N, -1)
        tgt = test_target if test_target is not None else target
        rsa_scores = self.compute_raw(src, tgt)
        return {'rsa_scores': rsa_scores}


class TemporalRSAMetric(BaseMetric):
    """
    Static model vs. temporal brain RSA:
      - source shape: (n_conditions, n_model_features)
      - target shape: (n_conditions, n_brain_channels, n_timepoints)
    Reports time-resolved comparisons across multiple metrics.
    """

    def compute_raw(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        n_cond, _, n_time = tgt.shape
        ds_model = Dataset(measurements=src, obs_descriptors={
                           'conds': list(range(n_cond))})
        ds_brain = TemporalDataset(
            measurements=tgt,
            obs_descriptors={'conds': list(range(n_cond))},
            time_descriptors={'time': list(range(n_time))}
        )
        rdm_model = calc_rdm(
            ds_model, method='correlation', descriptor='conds')
        vec_model = rdm_model.get_vectors()
        rdm_brain = calc_rdm_movie(
            ds_brain,
            method='correlation',
            descriptor='conds',
            time_descriptor='time'
        )
        vecs_brain = rdm_brain.get_vectors()
        scores = {m: np.zeros(n_time) for m in _COMPARISON_METHODS}
        for t in range(n_time):
            for m in _COMPARISON_METHODS:
                scores[m][t] = compare(vec_model, vecs_brain[t], method=m)
        return scores

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        src = test_source if test_source is not None else source
        tgt = test_target if test_target is not None else target

        # Handle source dimensionality
        if src.ndim != 2:
            original_shape = src.shape
            src = src.reshape(src.shape[0], -1)
            print(
                f"Reshaped source from {src.ndim}D {original_shape} to 2D {src.shape}")

        assert tgt.ndim == 3, f"target must be 3D, got {tgt.shape}"
        assert src.shape[0] == tgt.shape[0], "number of conditions must match"
        scores = self.compute_raw(src, tgt)
        return {'rsa_scores': scores}


class RepetitionRSAMetric(BaseMetric):
    """
    Crossnobis RSA for repeated measurements:
      - source shape: (n_conditions, n_model_features)
      - target shape: (n_conditions, n_reps, n_brain_channels)
    Reports multiple comparison metrics against crossnobis brain RDM.
    """

    def compute_raw(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        n_cond, n_rep, _ = tgt.shape
        ds_model = Dataset(measurements=src, obs_descriptors={
                           'conds': list(range(n_cond))})
        rdm_model = calc_rdm(
            ds_model, method='correlation', descriptor='conds')
        vec_model = rdm_model.get_vectors()
        meas = tgt.reshape(n_cond * n_rep, -1)
        conds = np.repeat(np.arange(n_cond), n_rep).tolist()
        runs = np.tile(np.arange(n_rep), n_cond).tolist()
        ds_brain = Dataset(
            measurements=meas,
            obs_descriptors={'conds': conds, 'runs': runs}
        )
        rdm_brain = calc_rdm(
            ds_brain,
            method='crossnobis',
            descriptor='conds',
            cv_descriptor='runs'
        )
        vec_brain = rdm_brain.get_vectors()
        return {m: float(compare(vec_model, vec_brain, method=m)) for m in _COMPARISON_METHODS}

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        src = test_source if test_source is not None else source
        tgt = test_target if test_target is not None else target
        assert src.ndim == 2, f"source must be 2D, got {src.shape}"
        assert tgt.ndim == 3, f"target must be 3D, got {tgt.shape}"
        n_cond, n_rep, _ = tgt.shape
        assert src.shape[0] == n_cond, "number of conditions must match"
        scores = self.compute_raw(src, tgt)
        return {'rsa_scores': scores}


class DynamicRSAMetric(BaseMetric):
    """
    Time-generalization RSA for dynamic model & brain:
      - source shape: (n_conditions, n_time_model, n_model_features)
      - target shape: (n_conditions, n_time_brain, n_brain_channels)
    Supports lock-step or cross-temporal generalization, reporting multiple metrics.
    """

    def compute_raw(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
    ) -> Dict[str, Union[np.ndarray, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]]:
        # model RDM movie
        ds_model = TemporalDataset(
            measurements=src,
            obs_descriptors={'conds': list(range(n_cond_s))},
            time_descriptors={'time': list(range(t_s))}
        )
        rdm_m = calc_rdm_movie(
            ds_model, method='correlation', descriptor='conds', time_descriptor='time')
        vecs_m = rdm_m.get_vectors()
        # brain RDM movie
        ds_brain = TemporalDataset(
            measurements=tgt,
            obs_descriptors={'conds': list(range(n_cond_t))},
            time_descriptors={'time': list(range(t_t))}
        )
        rdm_b = calc_rdm_movie(
            ds_brain, method='correlation', descriptor='conds', time_descriptor='time')
        vecs_b = rdm_b.get_vectors()
        # choose lock-step or generalization based on stratify_on
        # repurpose stratify_on as flag for gen
        generalization = bool(stratify_on)
        if not generalization:
            assert t_s == t_t, "time dims must match for lock-step RSA"
            scores = {m: np.zeros(t_s) for m in _COMPARISON_METHODS}
            for i in range(t_s):
                for m in _COMPARISON_METHODS:
                    scores[m][i] = compare(vecs_m[i], vecs_b[i], method=m)
            return {'times': np.arange(t_s), 'metrics': scores}
        gen = {m: np.zeros((t_s, t_t)) for m in _COMPARISON_METHODS}
        for i in range(t_s):
            for j in range(t_t):
                for m in _COMPARISON_METHODS:
                    gen[m][i, j] = compare(vecs_m[i], vecs_b[j], method=m)
        return {'times_model': np.arange(t_s), 'times_brain': np.arange(t_t), 'gen_matrices': gen}

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        src = test_source if test_source is not None else source
        tgt = test_target if test_target is not None else target
        assert src.ndim == 3, f"source must be 3D, got {src.shape}"
        assert tgt.ndim == 3, f"target must be 3D, got {tgt.shape}"
        n_cond_s, t_s, _ = src.shape
        n_cond_t, t_t, _ = tgt.shape
        assert n_cond_s == n_cond_t, "conditions must match"
        scores = self.compute_raw(src, tgt)
        return {'rsa_scores': scores}


class TemporalRepetitionRSAMetric(BaseMetric):
    """
    Crossnobis time-resolved RSA:
      - source shape : (n_conditions, n_model_features)
      - target shape : (n_conditions, n_reps, n_brain_channels, n_timepoints)
    Reports time-resolved RDM similarity across multiple metrics using cross-validated Mahalanobis.
    """

    def compute_raw(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        # model RDM
        ds_model = Dataset(measurements=src, obs_descriptors={
                           'conds': list(range(n_cond))})
        rdm_model = calc_rdm(
            ds_model, method='correlation', descriptor='conds')
        vec_model = rdm_model.get_vectors()
        # flatten reps into runs and build temporal dataset
        meas = tgt.reshape(n_cond * n_rep, n_chan, n_time)
        conds = np.repeat(np.arange(n_cond), n_rep).tolist()
        runs = np.tile(np.arange(n_rep), n_cond).tolist()
        ds_brain = TemporalDataset(
            measurements=meas,
            obs_descriptors={'conds': conds, 'runs': runs},
            time_descriptors={'time': list(range(n_time))}
        )
        rdm_brain_movie = calc_rdm_movie(
            ds_brain,
            method='crossnobis',
            descriptor='conds',
            cv_descriptor='runs',
            time_descriptor='time'
        )
        vecs_b = rdm_brain_movie.get_vectors()
        scores = {m: np.zeros(n_time) for m in _COMPARISON_METHODS}
        for t in range(n_time):
            for m in _COMPARISON_METHODS:
                scores[m][t] = compare(vec_model, vecs_b[t], method=m)
        return {'times': np.arange(n_time), 'metrics': scores}

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        src = test_source if test_source is not None else source
        tgt = test_target if test_target is not None else target
        assert src.ndim == 2, f"source must be 2D, got {src.shape}"
        assert tgt.ndim == 4, f"target must be 4D, got {tgt.shape}"
        n_cond, n_rep, n_chan, n_time = tgt.shape
        assert src.shape[0] == n_cond, "number of conditions must match"
        scores = self.compute_raw(src, tgt)
        return {'rsa_scores': scores}
