"""
Story-averaged LeBel 2023 benchmark.

One embedding per story (mean over TRs), one fMRI vector per story (mean over TRs).
Ridge with story-level GroupKFold. Saves LeBel2023Benchmark.pkl with
final_pearson and final_r2 for compatibility with the analysis notebook.
"""

import os
import pickle
import datetime
import numpy as np
import torch
from typing import Union, List, Optional

from sklearn.datasets import get_data_home
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA

from data.LeBel2023 import (
    LeBel2023TRStimulusSet, LeBel2023TRAssembly,
    LeBel2023FreeSurferLabels, RegionMapper,
)
from models import get_model_class_and_id


class LeBel2023Benchmark:
    """
    Story-averaged alignment benchmark for LeBel et al. (2023).

    Pipeline:
    1. Load TR-level stimuli and fMRI (same as TR benchmark).
    2. For each story: extract TR-level features, average over TRs → one vector;
       average fMRI over TRs → one vector.
    3. X = (n_stories, D), y = (n_stories, n_voxels), groups = story index.
    4. Ridge regression with GroupKFold (story-level CV).
    5. Save results as LeBel2023Benchmark.pkl with final_pearson, final_r2.
    """

    def __init__(
        self,
        model_identifier: str,
        layer_name: Union[str, List[str]],
        subject_id: str = 'UTS01',
        tr_duration: float = 2.0,
        hrf_delay: int = 2,
        n_cv_folds: int = 5,
        batch_size: Union[int, List[int]] = None,
        debug: bool = False,
        fast_mode: bool = True,
        max_tokens: int = 2048,
    ):
        if batch_size is None:
            batch_size = [4]
        self.debug = debug
        self.subject_id = subject_id
        self.tr_duration = tr_duration
        self.hrf_delay = hrf_delay
        self.n_cv_folds = n_cv_folds
        self.model_identifier = model_identifier

        if isinstance(batch_size, list):
            self.batch_size = batch_size[0]
        else:
            self.batch_size = batch_size

        self.layer_name = layer_name
        self.layer_names = (
            layer_name if isinstance(layer_name, list) else [layer_name]
        )

        self.model_class, self.model_id_mapping = get_model_class_and_id(
            model_identifier)
        self.model_instance = self.model_class()
        self.model = self.model_instance.get_model(self.model_id_mapping)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval()
        self.model.to(self.device)

        self.features = {l: [] for l in self.layer_names}
        self._register_hooks()

        data_home = get_data_home()
        results_base = os.environ.get('RESULTS_PATH', data_home)
        self.results_dir = os.path.join(results_base, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        self.ridge_voxel_chunk_size = 2000
        self.fast_mode = fast_mode
        self.max_tokens = max_tokens

    def _register_hooks(self):
        def hook_fn_factory(layer_id):
            def hook_fn(module, input, output):
                self.features[layer_id].append(output)
            return hook_fn

        for l_name in self.layer_names:
            found = False
            for name, module in self.model.named_modules():
                if name == l_name:
                    if isinstance(module, torch.nn.ModuleList):
                        module[-1].register_forward_hook(
                            hook_fn_factory(l_name))
                    else:
                        module.register_forward_hook(
                            hook_fn_factory(l_name))
                    found = True
                    break
            if not found:
                raise ValueError(f"Layer {l_name} not found in model.")

    def _extract_tr_feature(self, text: str) -> Optional[np.ndarray]:
        if not text.strip():
            return None
        for l in self.layer_names:
            self.features[l] = []
        input_ids = self.model_instance.preprocess_fn(text)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            _ = self.model(input_ids)
        layer_features = {}
        for l_name in self.layer_names:
            if not self.features[l_name]:
                raise ValueError(f"No features captured for layer {l_name}")
            feat = self.features[l_name][0]
            if isinstance(feat, tuple):
                feat = feat[0]
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)
            processed = self.model_instance.postprocess_fn(feat)
            if isinstance(processed, torch.Tensor):
                processed = processed.cpu().numpy()
            layer_features[l_name] = processed.squeeze()
        if len(self.layer_names) == 1:
            return layer_features[self.layer_names[0]]
        return np.concatenate(
            [layer_features[l] for l in self.layer_names], axis=-1)

    def _extract_story_features(
        self, stimulus_set: LeBel2023TRStimulusSet, story_idx: int
    ) -> np.ndarray:
        cumulative_texts, n_trs = stimulus_set.get_tr_texts(story_idx)
        if n_trs == 0:
            return np.array([])

        if self.fast_mode:
            # One forward pass on the full story text (last cumulative context).
            # Truncate to max_tokens tokens if needed to fit context window.
            full_text = cumulative_texts[-1]
            for l in self.layer_names:
                self.features[l] = []
            input_ids = self.model_instance.preprocess_fn(full_text)
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            # Truncate to last max_tokens tokens
            if input_ids.shape[1] > self.max_tokens:
                input_ids = input_ids[:, -self.max_tokens:]
            input_ids = input_ids.to(self.device)
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                _ = self.model(input_ids)
            feat_dim = None
            layer_vecs = {}
            for l_name in self.layer_names:
                if not self.features[l_name]:
                    return np.array([])
                feat = self.features[l_name][0]
                if isinstance(feat, tuple):
                    feat = feat[0]
                # Mean-pool over all token positions: (1, seq_len, D) → (D,)
                # This bypasses postprocess_fn which is designed for short TR texts
                # and returns near-zero values for long full-story texts.
                if feat.dim() == 3:
                    vec = feat.mean(dim=1).squeeze(0).cpu().float().numpy()
                elif feat.dim() == 2:
                    vec = feat.mean(dim=0).cpu().float().numpy()
                else:
                    vec = feat.squeeze().cpu().float().numpy()
                layer_vecs[l_name] = vec
                feat_dim = vec.shape[-1] if vec.ndim > 0 else 1
            if feat_dim is None:
                return np.array([])
            if len(self.layer_names) == 1:
                vec = layer_vecs[self.layer_names[0]]
            else:
                vec = np.concatenate([layer_vecs[l] for l in self.layer_names], axis=-1)
            # Return as (1, D) so the rest of run() can average over "TRs"
            return vec.reshape(1, -1).astype(np.float32)

        # Slow mode: one forward pass per TR (original behaviour)
        features_list = []
        for tr_idx in range(n_trs):
            feat = self._extract_tr_feature(cumulative_texts[tr_idx])
            features_list.append(feat)
        feat_dim = None
        for f in features_list:
            if f is not None:
                feat_dim = f.shape[-1]
                break
        if feat_dim is None:
            return np.array([])
        result = np.zeros((n_trs, feat_dim), dtype=np.float32)
        for i, f in enumerate(features_list):
            if f is not None:
                result[i] = f
        return result

    @staticmethod
    def _safe_pearson(a, b):
        """Pearson r; returns 0 if either input has zero variance."""
        a = a - a.mean()
        b = b - b.mean()
        na = np.sqrt((a * a).sum())
        nb = np.sqrt((b * b).sum())
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _safe_r2(y_true, y_pred):
        """R² clipped to [-1, 1] to prevent overflow reporting."""
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        if ss_tot < 1e-12:
            return 0.0
        ss_res = ((y_true - y_pred) ** 2).sum()
        return float(np.clip(1.0 - ss_res / ss_tot, -1.0, 1.0))

    def _run_group_kfold(self, X, y, groups):
        n_unique_groups = len(np.unique(groups))
        n_splits = min(self.n_cv_folds, n_unique_groups)
        n_voxels = y.shape[1]
        if n_splits < 2:
            raise ValueError(
                f"Need at least 2 story groups for CV, got {n_unique_groups}")
        gkf = GroupKFold(n_splits=n_splits)
        # Use a moderate fixed alpha — avoids RidgeCV LOO instability with
        # many outputs. Cross-validate alpha with a simple grid instead.
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        chunk = self.ridge_voxel_chunk_size
        pearson_scores = []
        r2_scores = []
        _diag_done = False
        for fold_idx, (train_idx, val_idx) in enumerate(
                gkf.split(X, y, groups)):
            print(f"  Fold {fold_idx + 1}/{n_splits}: "
                  f"train={len(train_idx)}, val={len(val_idx)}")
            X_train, X_val = X[train_idx], X[val_idx]
            fold_pearson = np.zeros(n_voxels, dtype=np.float64)
            fold_r2 = np.zeros(n_voxels, dtype=np.float64)
            for start in range(0, n_voxels, chunk):
                end = min(start + chunk, n_voxels)
                y_train_chunk = y[train_idx, start:end]
                y_val_chunk = y[val_idx, start:end]
                model = RidgeCV(alphas=alphas)
                model.fit(X_train, y_train_chunk)
                preds = model.predict(X_val)
                # Diagnostics on first fold/chunk to detect blown-up predictions
                if not _diag_done and fold_idx == 0:
                    _diag_done = True
                    print(f"  [diag fold1] chosen alpha={model.alpha_:.4g}, "
                          f"y_val range=[{y_val_chunk[:, 0].min():.4f}, "
                          f"{y_val_chunk[:, 0].max():.4f}], "
                          f"pred range=[{preds[:, 0].min():.4f}, "
                          f"{preds[:, 0].max():.4f}]")
                for i in range(end - start):
                    fold_pearson[start + i] = self._safe_pearson(
                        y_val_chunk[:, i], preds[:, i])
                    fold_r2[start + i] = self._safe_r2(
                        y_val_chunk[:, i], preds[:, i])
            pearson_scores.append(fold_pearson)
            r2_scores.append(fold_r2)
        return {
            'pearson': np.array(pearson_scores),
            'r2': np.array(r2_scores),
        }

    def run(self):
        mode = "fast (1 pass/story)" if self.fast_mode else "slow (1 pass/TR)"
        print(f"Extraction mode: {mode}")
        print("Loading TR-level stimuli...")
        stimulus_set = LeBel2023TRStimulusSet(tr_duration=self.tr_duration)

        print(f"Loading fMRI assembly for {self.subject_id}...")
        assembly = LeBel2023TRAssembly(subjects=[self.subject_id])
        story_fmri, ncsnr = assembly.get_assembly(
            story_names=stimulus_set.story_names)

        all_feat_avg = []
        all_fmri_avg = []
        group_ids = []

        for story_idx, story_name in enumerate(stimulus_set.story_names):
            if story_name not in story_fmri:
                print(f"Warning: No fMRI for '{story_name}', skipping.")
                continue
            print(f"Processing story {story_idx + 1}/{len(stimulus_set.story_names)}: {story_name}")

            story_features = self._extract_story_features(
                stimulus_set, story_idx)
            fmri_data = story_fmri[story_name]

            if story_features.size == 0:
                print(f"  Skipping: no features extracted.")
                continue

            if self.fast_mode:
                # Single embedding per story; average all fMRI TRs directly
                feat_avg = story_features[0]
                fmri_avg = np.mean(fmri_data, axis=0)
            else:
                n_trs_feat = story_features.shape[0]
                n_trs_fmri = fmri_data.shape[0]
                n_trs = min(n_trs_feat, n_trs_fmri)
                if n_trs < self.hrf_delay + 1:
                    print(f"  Skipping: too few TRs ({n_trs})")
                    continue
                story_features = story_features[:n_trs]
                fmri_data = fmri_data[:n_trs]
                shifted_features = story_features[:n_trs - self.hrf_delay]
                shifted_fmri = fmri_data[self.hrf_delay:]
                feat_avg = np.mean(shifted_features, axis=0)
                fmri_avg = np.mean(shifted_fmri, axis=0)

            all_feat_avg.append(feat_avg)
            all_fmri_avg.append(fmri_avg)
            group_ids.append(story_idx)

        if not all_feat_avg:
            raise ValueError("No data after alignment.")

        X = np.array(all_feat_avg, dtype=np.float64)
        y = np.array(all_fmri_avg, dtype=np.float64)
        groups = np.array(group_ids)

        # Diagnostics
        print(f"Story-averaged: stories={X.shape[0]}, "
              f"feature dim={X.shape[1]}, voxels={y.shape[1]}")
        print(f"X stats: mean={X.mean():.4f}, std={X.std():.4f}, "
              f"min={X.min():.4f}, max={X.max():.4f}, "
              f"nan={np.isnan(X).sum()}, inf={np.isinf(X).sum()}")

        # Z-score normalize features (critical for ridge stability)
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True)
        X_std[X_std < 1e-10] = 1.0
        X = (X - X_mean) / X_std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"X normalized: mean={X.mean():.4f}, std={X.std():.4f}")

        # PCA to reduce features: with 81 stories and 4096 features, ridge picks huge alpha
        # → all weights→0 → constant predictions → pearson=0. PCA to 50 PCs fixes this.
        n_components = min(50, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X = pca.fit_transform(X)
        print(f"PCA: {n_components} components, explained variance={pca.explained_variance_ratio_.sum():.3f}")
        # Re-normalise PC scores to unit variance so ridge alpha scale is predictable
        X_pc_std = X.std(axis=0, keepdims=True)
        X_pc_std[X_pc_std < 1e-10] = 1.0
        X = X / X_pc_std
        print(f"X_pca normalised: mean={X.mean():.4f}, std={X.std():.4f}, min={X.min():.4f}, max={X.max():.4f}")

        # Z-score normalize fMRI per voxel (critical: raw BOLD can vary wildly across voxels)
        print(f"Y stats: mean={y.mean():.4f}, std={y.std():.4f}, "
              f"min={y.min():.4f}, max={y.max():.4f}, "
              f"nan={np.isnan(y).sum()}, inf={np.isinf(y).sum()}")
        y_mean = y.mean(axis=0, keepdims=True)
        y_std = y.std(axis=0, keepdims=True)
        y_std[y_std < 1e-8] = 1.0  # avoid division by zero for constant voxels
        y = (y - y_mean) / y_std
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"Y normalized: mean={y.mean():.4f}, std={y.std():.4f}")

        fold_scores = self._run_group_kfold(X, y, groups)
        median_pearson = np.median(fold_scores['pearson'], axis=0)
        median_r2 = np.median(fold_scores['r2'], axis=0)

        # Try to apply FreeSurfer language mask (anatomical language voxels only)
        lang_mask = None
        try:
            fs_labels = LeBel2023FreeSurferLabels()
            label_dir = fs_labels.ensure_downloaded(self.subject_id)
            region_mapper = RegionMapper(label_dir)
            anat_lang_idx = region_mapper.get_language_indices()
            n_voxels = y.shape[1]
            lang_mask = np.zeros(n_voxels, dtype=bool)
            valid_idx = anat_lang_idx[anat_lang_idx < n_voxels]
            lang_mask[valid_idx] = True
            print(f"Language mask: {lang_mask.sum()}/{n_voxels} voxels")
        except Exception as e:
            print(f"FreeSurfer language mask failed ({e}); using top-10% voxels by Pearson.")

        if lang_mask is not None and lang_mask.sum() > 0:
            final_pearson = float(np.median(median_pearson[lang_mask]))
            final_r2 = float(np.median(median_r2[lang_mask]))
        else:
            # Fallback: top 10% voxels by median Pearson
            thresh = np.percentile(median_pearson, 90)
            top_mask = median_pearson >= thresh
            final_pearson = float(np.median(median_pearson[top_mask]))
            final_r2 = float(np.median(median_r2[top_mask]))

        print(f"Ridge (averaged): final_pearson={final_pearson:.4f}, final_r2={final_r2:.4f}")
        import json
        print("RESULT_JSON:" + json.dumps({
            "model": self.model_identifier,
            "layer": self.layer_name,
            "final_pearson": final_pearson,
            "final_r2": final_r2,
        }))

        results = {
            'ridge': {
                'final_pearson': final_pearson,
                'final_r2': final_r2,
                'n_stories': int(X.shape[0]),
                'n_lang_voxels': int(lang_mask.sum()) if lang_mask is not None else None,
            },
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'hrf_delay': self.hrf_delay,
            'tr_duration': self.tr_duration,
        }

        layer_str = (self.layer_name if isinstance(self.layer_name, str)
                     else "_".join(self.layer_names))
        benchmark_name = self.__class__.__name__
        results_file = os.path.join(
            self.results_dir,
            f"{self.model_identifier}_{layer_str}_{benchmark_name}.pkl"
        )
        merged = {"metrics": results, "ceiling": ncsnr}
        try:
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, 'wb') as f:
                pickle.dump(merged, f)
            print(f"Results saved to {results_file}")
        except OSError as e:
            # Disk quota exceeded or other write error — results are still printed above as RESULT_JSON
            print(f"WARNING: Could not save results file ({e}). Use RESULT_JSON line above to recover values.")
            # Try saving to /tmp as fallback
            tmp_file = os.path.join("/tmp", os.path.basename(results_file))
            try:
                with open(tmp_file, 'wb') as f:
                    pickle.dump(merged, f)
                print(f"Results saved to fallback path: {tmp_file}")
            except Exception as e2:
                print(f"WARNING: Fallback save also failed ({e2}). Rely on RESULT_JSON line.")
        return merged
