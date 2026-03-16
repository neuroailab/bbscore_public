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
import numpy as np

from typing import Optional, Callable, Union, List, Tuple
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
        confound_strategy: 'motion6', 'motion24', 'compcor', or None.
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
        confound_strategy: Optional[str] = "motion6",
    ):
        super().__init__(root_dir)
        self.subject = subject
        self.region = region
        self.task = task
        self.overwrite = overwrite
        self.ceiling_threshold = ceiling_threshold
        self.standardize = standardize
        self.confound_strategy = confound_strategy

        if subject not in ALL_SUBJECTS:
            raise ValueError(
                f"Unknown subject '{subject}'. Must be one of: {ALL_SUBJECTS}"
            )

        self.train_data = None
        self.train_ceiling = None
        self.test_data = None
        self.test_ceiling = None
        self._mask_img = None

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

    def _find_bold_files(self) -> List[str]:
        """Find all BOLD files for this subject/task, sorted by run."""
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

    def _load_confounds(self, bold_file: str) -> Optional[np.ndarray]:
        """Load motion/confound regressors for a BOLD file."""
        if self.confound_strategy is None:
            return None

        base = bold_file.replace("_bold.nii.gz", "")
        candidates = [
            f"{base}_desc-confounds_timeseries.tsv",
            f"{base}_confounds.tsv",
        ]

        confound_file = None
        for c in candidates:
            if os.path.exists(c):
                confound_file = c
                break

        if confound_file is None:
            return None

        import pandas as pd
        df = pd.read_csv(confound_file, sep="\t")

        if self.confound_strategy == "motion6":
            cols = [c for c in df.columns
                    if c in ["trans_x", "trans_y", "trans_z",
                             "rot_x", "rot_y", "rot_z",
                             "X", "Y", "Z", "RotX", "RotY", "RotZ"]]
        elif self.confound_strategy == "motion24":
            cols = [c for c in df.columns
                    if any(c.startswith(p) for p in
                           ["trans_", "rot_", "X", "Y", "Z", "Rot"])]
        elif self.confound_strategy == "compcor":
            cols = [c for c in df.columns
                    if c.startswith("a_comp_cor")][:5]
        else:
            raise ValueError(
                f"Unknown confound strategy: {self.confound_strategy}"
            )

        if not cols:
            return None

        confound_matrix = df[cols].values
        return np.nan_to_num(confound_matrix, nan=0.0)

    def _extract_timeseries(self, run_files: List[str]) -> np.ndarray:
        """
        Extract hippocampal voxel timeseries across runs.

        Returns:
            (total_TRs, n_voxels) array.
        """
        ref_img = nib.load(run_files[0])

        if self._mask_img is None:
            self._mask_img = get_hippocampal_mask(self.region, ref_img)
            n_voxels = int(self._mask_img.get_fdata().sum())
            print(
                f"Hippocampal mask ({self.region}): {n_voxels} voxels "
                f"in {ref_img.shape[:3]} space"
            )

        masker = NiftiMasker(
            mask_img=self._mask_img,
            standardize=self.standardize,
            detrend=True,
            high_pass=0.008,
            t_r=2.0,
        )

        all_data = []
        for bold_file in run_files:
            bold_img = nib.load(bold_file)
            confounds = self._load_confounds(bold_file)
            try:
                voxel_ts = masker.fit_transform(bold_img, confounds=confounds)
                all_data.append(voxel_ts)
            except Exception as e:
                print(f"Warning: Failed to process {bold_file}: {e}")
                continue

        if not all_data:
            raise RuntimeError(
                f"No data could be extracted for {self.subject}"
            )

        return np.concatenate(all_data, axis=0)

    # ── Data Preparation ──────────────────────────────────────

    def prepare_data(self, train: bool = True):
        """
        Download and prepare train/test data.

        Split: first N-1 runs = train, last run = test (mirrors TVSD).
        """
        self._verify_local_data()
        run_files = self._find_bold_files()
        n_runs = len(run_files)

        if n_runs < 2:
            print(
                f"Warning: Only {n_runs} run(s) for {self.subject}. "
                "Using all data for both train and test."
            )
            train_files = run_files
            test_files = run_files
        else:
            train_files = run_files[:-1]
            test_files = run_files[-1:]

        print(f"Extracting hippocampal data for {self.subject}...")
        print(f"  Train: {len(train_files)} runs, Test: {len(test_files)} runs")

        train_ts = self._extract_timeseries(train_files)
        test_ts = self._extract_timeseries(test_files)

        train_ceiling = compute_fmri_tsnr_ceiling(train_ts)
        test_ceiling = compute_fmri_tsnr_ceiling(test_ts)

        mask = test_ceiling >= self.ceiling_threshold
        if not np.any(mask):
            print("Warning: No voxels pass ceiling threshold. Using all.")
            mask = np.ones(test_ceiling.shape, dtype=bool)

        self.train_data = train_ts[:, mask]
        self.test_data = test_ts[:, mask]
        self.train_ceiling = train_ceiling[mask]
        self.test_ceiling = test_ceiling[mask]

        print(
            f"  {self.subject} ready: train={self.train_data.shape}, "
            f"test={self.test_data.shape}"
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
            return self.train_data, self.train_ceiling
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