"""
Route Learning Dataset Assembly for BBScore.

Loads hippocampal fMRI data from the OpenNeuro ds000217 "Route Learning" dataset
(Kuhl, Favila, Oza & Chanales) and packages it for brain-to-brain comparison
using the BBScore benchmarking framework.

Design based on TVSDAssembly single-monkey classes (TVSDAssemblyMonkeyFV1, etc.),
adapted for human fMRI comparison in the hippocampus.

Dataset: https://openneuro.org/datasets/ds000217/versions/1.0.0
    - 41 subjects across 2 experiments (~20 each)
    - Subjects scanned while learning overlapping routes on NYU campus
    - 14 rounds of route learning per subject
    - BIDS-formatted fMRI data (Siemens Allegra 3T)

Usage (brain-to-brain via AssemblyBenchmarkScorer):
    # These are used as source_assembly_class and target_assembly_class
    # in the benchmark definitions (see RouteLearningBenchmark.py).
    source = RouteLearningExp1s01Hippocampus()
    target = RouteLearningExp1s02Hippocampus()
    train_data, ceiling = source.get_assembly(train=True)
"""

import os
import glob
import json
import numpy as np

from typing import Optional, Callable, Union, List, Tuple, Dict, Any
from sklearn.datasets import get_data_home

from data.base import BaseDataset

try:
    import nibabel as nib
    from nilearn.maskers import NiftiMasker
    from nilearn.image import resample_to_img
    from nilearn import datasets as ni_datasets
except ImportError:
    raise ImportError(
        "nibabel and nilearn are required for the Route Learning dataset. "
        "Install them with: pip install nibabel nilearn"
    )


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

OPENNEURO_S3_BUCKET = "openneuro.org"
DATASET_ID = "ds000217"

# Local path to the BIDS dataset (relative to repo root, or set absolute).
# Override per-instance via root_dir= kwarg or by setting the
# ROUTE_LEARNING_DATA environment variable.
LOCAL_BIDS_DIR = os.environ.get(
    "ROUTE_LEARNING_DATA",
    os.path.join(os.getcwd(), "route-learning"),
)

EXPERIMENT_1_SUBJECTS = [f"sub-Exp1s{i:02d}" for i in range(1, 21)]
EXPERIMENT_2_SUBJECTS = [f"sub-Exp2s{i:02d}" for i in range(1, 22)]
ALL_SUBJECTS = EXPERIMENT_1_SUBJECTS + EXPERIMENT_2_SUBJECTS

TASK_ROUTE = "routelearning"

REGION_CONFIGS = {
    "hippocampus": {
        "labels": ["Left Hippocampus", "Right Hippocampus"],
        "description": "Bilateral hippocampus",
    },
    "left_hippocampus": {
        "labels": ["Left Hippocampus"],
        "description": "Left hippocampus only",
    },
    "right_hippocampus": {
        "labels": ["Right Hippocampus"],
        "description": "Right hippocampus only",
    },
}

DEFAULT_CEILING_THRESHOLD = -1000.0
DEFAULT_MNI_SPACE = "MNI152NLin2009cAsym"
DEFAULT_FMRIPREP_DERIV_DIRNAME = os.path.join("derivatives", "fmriprep")


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _safe_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _zscore_nan_safe(x: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    return (np.nan_to_num(x, nan=0.0) - np.nan_to_num(mean, nan=0.0)) / std


# ──────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────

def compute_fmri_tsnr_ceiling(data: np.ndarray) -> np.ndarray:
    """
    Compute per-voxel temporal SNR as a ceiling/reliability estimate.

    tSNR = mean / std across time, mapped to [0, 1] via tsnr/(tsnr+1).
    Serves the same role as ncsnr in NSD or split-half consistency in TVSD.

    Args:
        data: (n_timepoints, n_voxels) fMRI timeseries.

    Returns:
        ceiling: (n_voxels,) reliability values in [0, 1].
    """
    mean_signal = np.mean(data, axis=0)
    std_signal = np.std(data, axis=0, ddof=1)
    std_signal[std_signal == 0] = np.inf
    tsnr = mean_signal / std_signal
    ceiling = tsnr / (tsnr + 1.0)
    return ceiling.astype(np.float64)


def get_hippocampal_mask(
    region: str,
    reference_img: "nib.Nifti1Image",
) -> "nib.Nifti1Image":
    """
    Create a binary hippocampal mask from the Harvard-Oxford subcortical atlas,
    resampled to match the reference fMRI image.

    Args:
        region: 'hippocampus', 'left_hippocampus', or 'right_hippocampus'.
        reference_img: NIfTI image defining the target space.

    Returns:
        Binary NIfTI mask in the reference image space.
    """
    if region not in REGION_CONFIGS:
        raise ValueError(
            f"Unknown region '{region}'. Choose from: {list(REGION_CONFIGS.keys())}"
        )

    config = REGION_CONFIGS[region]
    atlas = ni_datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
    atlas_maps = atlas["maps"]
    # Newer nilearn returns a Nifti1Image directly; older versions return a path
    if isinstance(atlas_maps, nib.Nifti1Image):
        atlas_img = atlas_maps
    else:
        atlas_img = nib.load(atlas_maps)
    atlas_labels = atlas["labels"]

    target_indices = []
    for label_name in config["labels"]:
        for i, atlas_label in enumerate(atlas_labels):
            if atlas_label == label_name:
                target_indices.append(i)
                break

    if not target_indices:
        raise ValueError(
            f"Could not find atlas labels {config['labels']} "
            f"in Harvard-Oxford atlas."
        )

    atlas_data = atlas_img.get_fdata()
    mask_data = np.isin(atlas_data, target_indices).astype(np.float32)
    mask_atlas_space = nib.Nifti1Image(mask_data, atlas_img.affine)

    mask_resampled = resample_to_img(
        mask_atlas_space, reference_img, interpolation="nearest"
    )
    mask_binary_data = (mask_resampled.get_fdata() > 0.5).astype(np.int8)
    return nib.Nifti1Image(mask_binary_data, mask_resampled.affine)


# ──────────────────────────────────────────────────────────────
# Single-Subject Assembly (analogous to TVSDAssemblyMonkeyFV1)
# ──────────────────────────────────────────────────────────────

class RouteLearningSubjectAssembly(BaseDataset):
    """
    Data assembly for ONE subject's hippocampal fMRI from ds000217.

    Mirrors TVSDAssemblyMonkeyFV1 / TVSDAssemblyMonkeyNV1: loads data for a
    single subject and returns (data, ceiling) via get_assembly().

    Two instances are wired into AssemblyBenchmarkScorer as
    source_assembly_class and target_assembly_class for brain-to-brain
    ridge regression.

    Args:
        root_dir: Root directory for data storage.
        subject: BIDS subject ID (e.g. 'sub-Exp1s01').
        region: Hippocampal region to extract.
        task: BIDS task label.
        overwrite: Force re-download.
        ceiling_threshold: Min ceiling for voxel inclusion.
        standardize: Z-score each voxel timeseries.
        confound_strategy: Confounds subset to use. When loading from fMRIPrep,
            choose from: 'motion6', 'motion24', 'compcor', 'motion24+compcor',
            or None.
        use_fmriprep: If True, load fMRIPrep derivatives (recommended) instead
            of raw BIDS NIfTI. Required for atlas masking in MNI space.
        fmriprep_dir: Directory containing fMRIPrep derivatives. Defaults to
            <root_dir>/derivatives/fmriprep.
        mni_space: fMRIPrep space label to load (default: MNI152NLin2009cAsym).
        high_pass_hz: High-pass cutoff (Hz) applied inside NiftiMasker.
        censor: If True, add spike regressors for high-motion/outlier volumes.
        fd_threshold_mm: Framewise displacement threshold for censoring.
        dvars_threshold: DVARS threshold for censoring (uses fMRIPrep column if present).
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        subject: str = "sub-Exp1s01",
        region: str = "hippocampus",
        task: str = TASK_ROUTE,
        overwrite: bool = False,
        ceiling_threshold: float = DEFAULT_CEILING_THRESHOLD,
        standardize: bool = True,
        confound_strategy: Optional[str] = "motion24+compcor",
        use_fmriprep: bool = False,
        fmriprep_dir: Optional[str] = None,
        mni_space: str = DEFAULT_MNI_SPACE,
        high_pass_hz: float = 0.008,
        censor: bool = True,
        fd_threshold_mm: float = 0.5,
        dvars_threshold: float = 1.5,
        debug: bool = False,
        trial_lag_trs: int = 3,
        trial_window_trs: int = 16,
    ):
        super().__init__(root_dir)
        self.debug = debug
        self.subject = subject
        self.region = region
        self.task = task
        self.overwrite = overwrite
        self.ceiling_threshold = ceiling_threshold
        self.standardize = standardize
        self.confound_strategy = confound_strategy
        self.use_fmriprep = use_fmriprep
        self.mni_space = mni_space
        self.high_pass_hz = high_pass_hz
        self.censor = censor
        self.fd_threshold_mm = fd_threshold_mm
        self.dvars_threshold = dvars_threshold
        self.trial_lag_trs = int(trial_lag_trs)
        self.trial_window_trs = int(trial_window_trs)

        # Default fMRIPrep derivatives path
        if fmriprep_dir is None:
            # root_dir is resolved in BaseDataset; use that resolved value
            fmriprep_dir = os.path.join(self.root_dir, DEFAULT_FMRIPREP_DERIV_DIRNAME)
        self.fmriprep_dir = fmriprep_dir

        if subject not in ALL_SUBJECTS:
            raise ValueError(
                f"Unknown subject '{subject}'. Must be one of: {ALL_SUBJECTS}"
            )

        self.train_data = None
        self.train_ceiling = None
        self.test_data = None
        self.test_ceiling = None
        self._mask_img = None
        self._tr_seconds: Optional[float] = None
        self._trial_groups: Optional[np.ndarray] = None
        self._trial_keys: Optional[np.ndarray] = None

    def _check_files_exists(self, *paths):
        return all(os.path.exists(path) for path in paths)

    # ── Local data access ────────────────────────────────────

    def _get_subject_dir(self) -> str:
        """Return path to this subject's BIDS directory.

        The dataset is expected at root_dir/<subject>/, following
        standard BIDS layout:
            root_dir/
            ├── sub-Exp1s01/
            │   ├── anat/
            │   └── func/
            ├── sub-Exp1s02/
            ...
        """
        return os.path.join(self.root_dir, self.subject)

    def _verify_local_data(self):
        """Verify that the local BIDS data exists for this subject."""
        subj_dir = self._get_subject_dir()
        func_dir = os.path.join(subj_dir, "func")

        if not os.path.isdir(subj_dir):
            raise FileNotFoundError(
                f"Subject directory not found: {subj_dir}\n"
                f"Expected the route-learning BIDS dataset at: {self.root_dir}\n"
                f"Make sure the dataset is downloaded and the path is correct."
            )

        if not os.path.isdir(func_dir):
            raise FileNotFoundError(
                f"Functional data directory not found: {func_dir}\n"
                f"Expected BIDS func/ folder inside {subj_dir}"
            )

        print(f"Subject {self.subject} found at {subj_dir}")

    # ── fMRI Processing ───────────────────────────────────────

    def _get_tr(self) -> float:
        """
        Return TR (seconds) from BIDS task JSON.

        Defaults to 1.5s for ds000217 if metadata not found.
        """
        if self._tr_seconds is not None:
            return self._tr_seconds
        # Prefer task-level metadata shipped with this dataset
        task_json = os.path.join(self.root_dir, f"task-{self.task}_bold.json")
        tr = 1.5
        if os.path.exists(task_json):
            meta = _read_json(task_json)
            tr = _safe_float(meta.get("RepetitionTime", tr), tr)
        self._tr_seconds = float(tr)
        return self._tr_seconds

    def _parse_run_number(self, path: str) -> int:
        """
        Extract run number from filenames like ..._run-01_...
        Returns 1-indexed run integer.
        """
        import re
        m = re.search(r"_run-(\d+)_", os.path.basename(path))
        if not m:
            raise ValueError(f"Could not parse run number from: {path}")
        return int(m.group(1))

    def _events_path_for_run(self, run_number: int) -> str:
        func_dir = os.path.join(self._get_subject_dir(), "func")
        return os.path.join(
            func_dir,
            f"{self.subject}_task-{self.task}_run-{run_number:02d}_events.tsv",
        )

    def _load_study_trials(self, events_tsv: str) -> List[Dict[str, Any]]:
        """
        Load and return ordered study trials for one run.

        Each returned dict contains onset/duration (seconds), route_id, rep_in_run.
        """
        import pandas as pd
        if not os.path.exists(events_tsv):
            raise FileNotFoundError(f"Events file not found: {events_tsv}")
        df = pd.read_csv(events_tsv, sep="\t")
        df = df[df["trial_type"] == "study_trial"].copy()
        if df.empty:
            return []
        df = df.sort_values("onset", kind="mergesort")

        rep_counter: Dict[int, int] = {}
        trials: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            route_id = int(row["route_number"])
            rep_counter[route_id] = rep_counter.get(route_id, 0) + 1
            trials.append({
                "onset": float(row["onset"]),
                "duration": float(row["duration"]),
                "route_id": route_id,
                "rep_in_run": int(rep_counter[route_id]),
            })
        return trials

    def _build_trial_matrix_for_run(
        self,
        run_number: int,
        voxel_ts: np.ndarray,
        trials: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
        """
        Build (keys, X_run) for one run.

        keys are (run, route_id, rep_in_run). Each row is W*V concatenated values.
        """
        tr = self._get_tr()
        L = self.trial_lag_trs
        W = self.trial_window_trs
        V = voxel_ts.shape[1]

        rows: List[np.ndarray] = []
        keys: List[Tuple[int, int, int]] = []
        for t in trials:
            start_tr = int(np.floor(t["onset"] / tr)) + L
            end_tr = start_tr + W
            if start_tr < 0 or end_tr > voxel_ts.shape[0]:
                continue
            window = voxel_ts[start_tr:end_tr, :]  # (W, V)
            rows.append(window.reshape(W * V).astype(np.float64, copy=False))
            keys.append((run_number, int(t["route_id"]), int(t["rep_in_run"])))

        if not rows:
            return [], np.zeros((0, W * V), dtype=np.float64)
        return keys, np.stack(rows, axis=0)

    def _find_raw_bold_files(self) -> List[str]:
        """Find raw BIDS BOLD files for this subject/task, sorted by run."""
        func_dir = os.path.join(self._get_subject_dir(), "func")
        pattern = os.path.join(
            func_dir,
            f"{self.subject}_task-{self.task}_run-*_bold.nii.gz"
        )
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No BOLD files for {self.subject} task-{self.task} in {func_dir}"
            )
        return files

    def _find_fmriprep_run_records(self) -> List[Dict[str, str]]:
        """
        Find fMRIPrep preprocessed BOLD runs in MNI space with confounds.

        Returns:
            list of dicts with keys: 'bold', 'confounds'
        """
        # Typical layout: <fmriprep_dir>/<sub>/func/<sub>_task-..._run-.._space-..._desc-preproc_bold.nii.gz
        subj = self.subject
        space = self.mni_space

        bold_patterns = [
            os.path.join(self.fmriprep_dir, subj, "func",
                         f"{subj}_task-{self.task}_run-*_space-{space}_desc-preproc_bold.nii.gz"),
            # Be robust to nested session folders if present
            os.path.join(self.fmriprep_dir, subj, "**", "func",
                         f"{subj}_task-{self.task}_run-*_space-{space}_desc-preproc_bold.nii.gz"),
        ]

        bold_files: List[str] = []
        for pat in bold_patterns:
            bold_files.extend(glob.glob(pat, recursive=True))
        bold_files = sorted(set(bold_files))

        if not bold_files:
            raise FileNotFoundError(
                f"No fMRIPrep preprocessed BOLD files found for {subj} task-{self.task} "
                f"in space '{space}'. Looked in: {self.fmriprep_dir}\n"
                f"Expected pattern like: {subj}_task-{self.task}_run-*_space-{space}_desc-preproc_bold.nii.gz"
            )

        records: List[Dict[str, str]] = []
        for bold in bold_files:
            base = bold.replace("_desc-preproc_bold.nii.gz", "")
            conf = f"{base}_desc-confounds_timeseries.tsv"
            if not os.path.exists(conf):
                # Some fMRIPrep versions may omit 'desc-' in older outputs; check fallback
                conf_fallback = f"{base}_confounds_timeseries.tsv"
                conf = conf_fallback if os.path.exists(conf_fallback) else conf
            records.append({"bold": bold, "confounds": conf})

        # Sort by run number if present in filename (lexicographic generally works for _run-01_)
        return records

    def _load_fmriprep_confounds_df(self, confounds_file: str):
        import pandas as pd
        if not os.path.exists(confounds_file):
            return None
        return pd.read_csv(confounds_file, sep="\t")

    def _select_confounds(self, df) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Build confounds matrix (and censoring spikes) from fMRIPrep confounds.

        Returns:
            confounds_matrix: (T, K) or None
            info: dict with censoring summary
        """
        if self.confound_strategy is None:
            return None, {"n_censored": 0, "pct_censored": 0.0}

        if df is None or df.shape[0] == 0:
            return None, {"n_censored": 0, "pct_censored": 0.0}

        cols: List[str] = []
        strat = self.confound_strategy
        if strat in ("motion6", "motion24", "motion24+compcor"):
            motion_base = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
            cols.extend([c for c in motion_base if c in df.columns])
            if strat in ("motion24", "motion24+compcor"):
                # fMRIPrep typically provides derivatives and squares as separate columns
                # e.g., trans_x_derivative1, trans_x_power2, trans_x_derivative1_power2
                extras = []
                for c in motion_base:
                    for suf in ("_derivative1", "_power2", "_derivative1_power2"):
                        name = f"{c}{suf}"
                        if name in df.columns:
                            extras.append(name)
                cols.extend(extras)
        if strat in ("compcor", "motion24+compcor"):
            compcor_cols = [c for c in df.columns if c.startswith("a_comp_cor")]
            cols.extend(compcor_cols[:10])  # common default; adjust if desired
        if strat not in ("motion6", "motion24", "compcor", "motion24+compcor"):
            raise ValueError(
                f"Unknown confound strategy: {self.confound_strategy}. "
                "Use one of: motion6, motion24, compcor, motion24+compcor, or None."
            )

        cols = [c for c in cols if c in df.columns]
        if not cols and not self.censor:
            return None, {"n_censored": 0, "pct_censored": 0.0}

        X = df[cols].to_numpy(dtype=float) if cols else None
        if X is not None:
            X = np.nan_to_num(X, nan=0.0)

        # Censoring via spike regressors (keeps timepoints; avoids misalignment across subjects)
        n_tp = int(df.shape[0])
        censored_idx: List[int] = []
        if self.censor:
            fd = None
            if "framewise_displacement" in df.columns:
                fd = np.nan_to_num(df["framewise_displacement"].to_numpy(dtype=float), nan=0.0)
            dvars = None
            # fMRIPrep column is often "dvars" or "std_dvars" depending on version
            for cand in ("dvars", "std_dvars"):
                if cand in df.columns:
                    dvars = np.nan_to_num(df[cand].to_numpy(dtype=float), nan=0.0)
                    break
            if fd is not None:
                censored_idx.extend(list(np.where(fd > self.fd_threshold_mm)[0]))
            if dvars is not None:
                censored_idx.extend(list(np.where(dvars > self.dvars_threshold)[0]))
            censored_idx = sorted(set(int(i) for i in censored_idx if 0 <= int(i) < n_tp))

        spike = None
        if censored_idx:
            spike = np.zeros((n_tp, len(censored_idx)), dtype=float)
            for j, i in enumerate(censored_idx):
                spike[i, j] = 1.0

        if spike is not None:
            if X is None:
                X = spike
            else:
                X = np.concatenate([X, spike], axis=1)

        info = {
            "n_censored": len(censored_idx),
            "pct_censored": (100.0 * len(censored_idx) / max(1, n_tp)),
            "n_confounds": 0 if X is None else int(X.shape[1]),
        }
        return X, info

    def _extract_timeseries(self, run_records: List[Dict[str, str]]) -> np.ndarray:
        """
        Extract hippocampal voxel timeseries across runs.

        Returns:
            (total_TRs, n_voxels) array.
        """
        ref_img = nib.load(run_records[0]["bold"])

        if self._mask_img is None:
            if self.use_fmriprep:
                print(
                    f"Building Harvard-Oxford hippocampus mask in '{self.mni_space}' "
                    "space (resampled to BOLD grid)."
                )
            self._mask_img = get_hippocampal_mask(self.region, ref_img)
            n_voxels = int(self._mask_img.get_fdata().sum())
            print(
                f"Hippocampal mask ({self.region}): {n_voxels} voxels "
                f"in {ref_img.shape[:3]} space"
            )

        tr = self._get_tr()
        masker = NiftiMasker(
            mask_img=self._mask_img,
            standardize=self.standardize,
            detrend=True,
            high_pass=self.high_pass_hz,
            t_r=tr,
        )

        all_data = []
        for rec in run_records:
            bold_file = rec["bold"]
            confounds_file = rec.get("confounds", "")
            bold_img = nib.load(bold_file)

            confounds = None
            censor_info = {"n_censored": 0, "pct_censored": 0.0, "n_confounds": 0}
            if self.use_fmriprep:
                df = self._load_fmriprep_confounds_df(confounds_file)
                confounds, censor_info = self._select_confounds(df)
            try:
                voxel_ts = masker.fit_transform(bold_img, confounds=confounds)
                all_data.append(voxel_ts)
                if self.debug:
                    print(
                        f"  {os.path.basename(bold_file)}: "
                        f"ts={voxel_ts.shape}, confounds={censor_info.get('n_confounds', 0)}, "
                        f"censored={censor_info.get('n_censored', 0)} "
                        f"({censor_info.get('pct_censored', 0.0):.1f}%)"
                    )
            except Exception as e:
                print(f"Warning: Failed to process {bold_file}: {e}")
                continue

        if not all_data:
            raise RuntimeError(
                f"No data could be extracted for {self.subject}"
            )

        return np.concatenate(all_data, axis=0)

    def _extract_trial_matrix(self, run_records: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract trial-aligned spatiotemporal features across runs.

        Row key: (run_number, route_id, rep_in_run). Rows are sorted by this key
        to make cross-subject alignment stable.

        Returns:
            X: (n_trials, W*V)
            groups: (n_trials,) run_number labels for leave-one-run-out GroupKFold
            keys: (n_trials, 3) int array with columns [run_number, route_id, rep_in_run]
        """
        ref_img = nib.load(run_records[0]["bold"])
        if self._mask_img is None:
            self._mask_img = get_hippocampal_mask(self.region, ref_img)

        tr = self._get_tr()
        masker = NiftiMasker(
            mask_img=self._mask_img,
            standardize=self.standardize,
            detrend=True,
            high_pass=self.high_pass_hz,
            t_r=tr,
        )

        all_keys: List[Tuple[int, int, int]] = []
        all_rows: List[np.ndarray] = []
        all_groups: List[int] = []

        for rec in run_records:
            bold_file = rec["bold"]
            run_num = self._parse_run_number(bold_file)
            events_tsv = self._events_path_for_run(run_num)
            trials = self._load_study_trials(events_tsv)
            if not trials:
                continue

            bold_img = nib.load(bold_file)
            confounds = None
            if self.use_fmriprep:
                df = self._load_fmriprep_confounds_df(rec.get("confounds", ""))
                confounds, _ = self._select_confounds(df)

            voxel_ts = masker.fit_transform(bold_img, confounds=confounds)  # (T, V)
            keys, X_run = self._build_trial_matrix_for_run(run_num, voxel_ts, trials)
            if X_run.shape[0] == 0:
                continue

            all_keys.extend(keys)
            all_rows.append(X_run)
            all_groups.extend([run_num] * X_run.shape[0])

        if not all_rows:
            raise RuntimeError(f"No trial matrices could be extracted for {self.subject}")

        X = np.concatenate(all_rows, axis=0)
        groups = np.asarray(all_groups, dtype=int)

        # Sort deterministically by key (run, route, rep)
        key_arr = np.array(all_keys, dtype=int)  # (N, 3)
        order = np.lexsort((key_arr[:, 2], key_arr[:, 1], key_arr[:, 0]))
        X = X[order]
        groups = groups[order]
        key_arr = key_arr[order]
        return X, groups, key_arr

    # ── Data Preparation ──────────────────────────────────────

    def prepare_data(self, train: bool = True):
        """
        Download and prepare train/test data.

        Split: first N-1 runs = train, last run = test (mirrors TVSD).
        """
        self._verify_local_data()
        if self.use_fmriprep:
            run_records = self._find_fmriprep_run_records()
        else:
            run_files = self._find_raw_bold_files()
            run_records = [{"bold": p, "confounds": ""} for p in run_files]
        n_runs = len(run_records)

        print(f"Extracting hippocampal data for {self.subject}...")
        print(f"  Using all {n_runs} runs (LORO CV by run)")

        X_trials, groups, keys = self._extract_trial_matrix(run_records)

        # Reliability ceiling: compute per-voxel tSNR on concatenated timeseries,
        # then repeat per TR slice in the concatenated trial representation.
        ts_all = self._extract_timeseries(run_records)
        voxel_ceiling = compute_fmri_tsnr_ceiling(ts_all)  # (V,)
        W = self.trial_window_trs
        trial_ceiling = np.tile(voxel_ceiling, W)  # (W*V,)

        mask = trial_ceiling >= self.ceiling_threshold
        if not np.any(mask):
            print("Warning: No voxels pass ceiling threshold. Using all.")
            mask = np.ones(trial_ceiling.shape, dtype=bool)

        self.train_data = X_trials[:, mask]
        self.train_ceiling = trial_ceiling[mask]
        self.test_data = None
        self.test_ceiling = None
        self._trial_groups = groups
        self._trial_keys = keys

        print(
            f"  {self.subject} ready: X={self.train_data.shape}, groups={np.unique(groups).size} runs"
        )

    def get_assembly(self, train: bool = True):
        """
        Return (data, ceiling) for the requested split.

        Matches the interface expected by AssemblyBenchmarkScorer.

        Returns:
            (data, ceiling) where:
              data: (n_timepoints, n_voxels) hippocampal timeseries
              ceiling: (n_voxels,) per-voxel reliability
        """
        if self.train_data is None:
            self.prepare_data(train)

        if train:
            # Return groups so ridge can do leave-one-run-out GroupKFold.
            return self.train_data, self.train_ceiling, self._trial_groups, self._trial_keys
        return self.test_data, self.test_ceiling

    def __len__(self):
        if self.train_data is not None:
            return self.train_data.shape[0]
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError("Use get_assembly() instead.")


# ──────────────────────────────────────────────────────────────
# Per-subject convenience subclasses
# (mirrors TVSDAssemblyMonkeyFV1, TVSDAssemblyMonkeyNV1, etc.)
# ──────────────────────────────────────────────────────────────

def _make_subject_class(subject_id, region, docstring):
    """Factory for per-subject assembly subclasses."""
    class _Asm(RouteLearningSubjectAssembly):
        __doc__ = docstring
        def __init__(self, **kwargs):
            # Default root_dir is ./route-learning (local BIDS dataset)
            root = kwargs.pop('root_dir', LOCAL_BIDS_DIR)
            super().__init__(
                root_dir=root, subject=subject_id, region=region, **kwargs
            )
    _Asm.__name__ = (
        f"RouteLearning{subject_id.replace('-', '')}"
        f"{'Hippo' if region == 'hippocampus' else region.replace('_', '').title()}"
    )
    _Asm.__qualname__ = _Asm.__name__
    return _Asm


# ── Experiment 1: Bilateral hippocampus ───────────────────────

RouteLearningExp1s01Hippo = _make_subject_class(
    "sub-Exp1s01", "hippocampus", "Exp1 sub01, bilateral hippocampus.")

RouteLearningExp1s02Hippo = _make_subject_class(
    "sub-Exp1s02", "hippocampus", "Exp1 sub02, bilateral hippocampus.")

RouteLearningExp1s03Hippo = _make_subject_class(
    "sub-Exp1s03", "hippocampus", "Exp1 sub03, bilateral hippocampus.")

RouteLearningExp1s04Hippo = _make_subject_class(
    "sub-Exp1s04", "hippocampus", "Exp1 sub04, bilateral hippocampus.")

RouteLearningExp1s05Hippo = _make_subject_class(
    "sub-Exp1s05", "hippocampus", "Exp1 sub05, bilateral hippocampus.")

# ── Experiment 1: Left hippocampus ────────────────────────────

RouteLearningExp1s01LeftHippo = _make_subject_class(
    "sub-Exp1s01", "left_hippocampus", "Exp1 sub01, left hippocampus.")

RouteLearningExp1s02LeftHippo = _make_subject_class(
    "sub-Exp1s02", "left_hippocampus", "Exp1 sub02, left hippocampus.")

# ── Experiment 1: Right hippocampus ───────────────────────────

RouteLearningExp1s01RightHippo = _make_subject_class(
    "sub-Exp1s01", "right_hippocampus", "Exp1 sub01, right hippocampus.")

RouteLearningExp1s02RightHippo = _make_subject_class(
    "sub-Exp1s02", "right_hippocampus", "Exp1 sub02, right hippocampus.")

# ── Experiment 2: Bilateral hippocampus ───────────────────────

RouteLearningExp2s01Hippo = _make_subject_class(
    "sub-Exp2s01", "hippocampus", "Exp2 sub01, bilateral hippocampus.")

RouteLearningExp2s02Hippo = _make_subject_class(
    "sub-Exp2s02", "hippocampus", "Exp2 sub02, bilateral hippocampus.")

RouteLearningExp2s03Hippo = _make_subject_class(
    "sub-Exp2s03", "hippocampus", "Exp2 sub03, bilateral hippocampus.")


# ── Generic configurable assembly ─────────────────────────────

class RouteLearningConfigurable(RouteLearningSubjectAssembly):
    """
    Fully configurable single-subject assembly for any subject/region combo.

        assembly = RouteLearningConfigurable(
            subject='sub-Exp1s15', region='left_hippocampus'
        )
    """
    def __init__(self, subject: str, region: str = "hippocampus", **kwargs):
        root = kwargs.pop('root_dir', LOCAL_BIDS_DIR)
        super().__init__(root_dir=root, subject=subject, region=region, **kwargs)