from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import xarray as xr
import os
import torch
from torch.utils.data import DataLoader
from sklearn.datasets import get_data_home

# Metric imports
from sklearn.metrics import accuracy_score, cohen_kappa_score
from brainio.assemblies import walk_coords

# Local imports
from metrics.base_online import OnlineMetric
from metrics.online_mappers import OnlineTransformerClassifier
from extractor_wrapper_online import OnlineFeatureExtractor

# ---------------------------------------------------------------------
# Helper Functions & Metric Calculators (Unchanged)
# ---------------------------------------------------------------------


def ensure_physion_file(filename: str, bucket_path: str = "physion_human"):
    """
    Download from public GCS bucket using anonymous client.
    """
    from google.cloud import storage

    data_home = get_data_home()
    target_dir = os.path.join(data_home, "PhysionBehavioral")
    os.makedirs(target_dir, exist_ok=True)

    local_path = os.path.join(target_dir, filename)

    if not os.path.exists(local_path):
        print(f"Downloading {filename} from GCS to {local_path}...")
        try:
            bucket_name = "bbscore_datasets"
            blob_name = f"{bucket_path}/{filename}"

            client = storage.Client.create_anonymous_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            return None

    return local_path


class ModelHumanAccuracyChebyshev:
    def __call__(self, model_data, human_data):
        return self.get_score(model_data, human_data)

    def extract_filename(self, file_name):
        # Robust extraction: handles bytes, full paths, and extensions
        if isinstance(file_name, bytes):
            file_name = file_name.decode('utf-8')
        file_name = str(file_name)
        file_name = os.path.basename(file_name)  # Remove directory path
        file_name = file_name.split('.')[0]       # Remove extension
        if '_img' in file_name:
            file_name = file_name.split('_img')[0]
        return file_name

    def label_to_grid_position(self, label, grid_size=16):
        grid_y = label // grid_size
        grid_x = label % grid_size
        return grid_y, grid_x

    def chebyshev_distance(self, label1, label2, grid_size=16):
        y1, x1 = self.label_to_grid_position(label1, grid_size)
        y2, x2 = self.label_to_grid_position(label2, grid_size)
        return max(abs(y1 - y2), abs(x1 - x2))

    def euclidean_distance(self, label1, label2, grid_size=16):
        y1, x1 = self.label_to_grid_position(label1, grid_size)
        y2, x2 = self.label_to_grid_position(label2, grid_size)
        return ((y1 - y2) ** 2 + (x1 - x2) ** 2) ** 0.5

    def get_score(self, MD, HD):
        # --- 1. Convert Model Data (MD) to DataFrame ---
        if not isinstance(MD, pd.DataFrame):
            if hasattr(MD, 'to_dataframe'):
                MD = MD.to_dataframe().reset_index()
            else:
                # Fallback for dictionaries or non-xarray assemblies
                try:
                    MD_ = {}
                    for coord, dims, values in walk_coords(MD):
                        MD_[coord] = np.array(values)
                    MD = pd.DataFrame(MD_)
                except Exception:
                    MD = pd.DataFrame(MD)

        # --- 2. Convert Human Data (HD) to DataFrame ---
        # FIX: Use to_dataframe() explicitly to avoid brainio walk_coords crash on Datasets
        if not isinstance(HD, pd.DataFrame):
            if hasattr(HD, 'to_dataframe'):
                HD = HD.to_dataframe().reset_index()
            else:
                # Fallback only if strictly necessary
                HD_ = {}
                for coord, dims, values in walk_coords(HD):
                    HD_[coord] = np.array(values)
                HD = pd.DataFrame(HD_)

        # --- 3. Clean Filenames ---
        if 'stimulus_id' in MD.columns:
            MD['stimulus_id'] = MD['stimulus_id'].apply(self.extract_filename)

        if 'stimulus_id' in HD.columns:
            HD['stimulus_id'] = HD['stimulus_id'].apply(self.extract_filename)

        # --- 4. Compute Metrics ---
        # Basic GT Accuracy (Model against its own ground truth labels)
        accuracy_gt = accuracy_score(
            MD['contacts'].values, MD['choice'].values)

        chebyshev_gt = [
            self.chebyshev_distance(gt, model_choice)
            for gt, model_choice in zip(MD['contacts'].values, MD['choice'].values)
        ]
        chebyshev_accuracy = np.mean([d <= 1 for d in chebyshev_gt])
        mean_chebyshev_gt = np.mean(chebyshev_gt)

        euclidean_gt = [
            self.euclidean_distance(gt, model_choice)
            for gt, model_choice in zip(MD['contacts'].values, MD['choice'].values)
        ]
        euclidean_accuracy = np.mean([d <= 1 for d in euclidean_gt])
        mean_euclidean_gt = np.mean(euclidean_gt)

        # INTERSECTION: Only keep stimuli present in both
        joint_stim_names = np.intersect1d(HD['stimulus_id'], MD['stimulus_id'])

        if len(joint_stim_names) == 0:
            print("Warning: No intersection found between Model and Human stimuli.")
            return {'center': 0.0, 'accuracy_model_gt': accuracy_gt}

        mask_MD = np.isin(MD['stimulus_id'], joint_stim_names)
        mask_HD = np.isin(HD['stimulus_id'], joint_stim_names)

        subset_MD = MD[mask_MD].groupby('stimulus_id').first().reset_index()
        subset_HD = HD[mask_HD].groupby('stimulus_id').first().reset_index()

        human_responses = subset_HD['choice'].values
        model_responses = subset_MD['choice'].values
        model_gt = subset_MD['contacts'].values

        accuracy = accuracy_score(human_responses, model_responses)

        chebyshev_dists = [
            self.chebyshev_distance(human_resp, model_resp)
            for human_resp, model_resp in zip(human_responses, model_responses)
        ]
        mean_chebyshev_distance = np.mean(chebyshev_dists)

        accuracy_hum = accuracy_score(human_responses, model_gt)

        chebyshev_hum = [
            self.chebyshev_distance(human_resp, model_resp)
            for human_resp, model_resp in zip(human_responses, model_gt)
        ]
        mean_chebyshev_hum = np.mean(chebyshev_hum)

        euclidean_dists = [
            self.euclidean_distance(human_resp, model_resp)
            for human_resp, model_resp in zip(human_responses, model_responses)
        ]
        mean_euclidean_distance = np.mean(euclidean_dists)

        euclidean_hum = [
            self.euclidean_distance(human_resp, model_resp)
            for human_resp, model_resp in zip(human_responses, model_gt)
        ]
        mean_euclidean_hum = np.mean(euclidean_hum)

        score = {}
        score['center'] = accuracy
        score['accuracy_model_human'] = accuracy
        score['accuracy_model_gt'] = accuracy_gt
        score['accuracy_human_gt'] = accuracy_hum
        score['accuracy_cheby_model_gt'] = chebyshev_accuracy
        score['accuracy_euclidean_model_gt'] = euclidean_accuracy
        score['chebyshev_distance_model_human'] = mean_chebyshev_distance
        score['chebyshev_distance_model_gt'] = mean_chebyshev_gt
        score['chebyshev_distance_human_gt'] = mean_chebyshev_hum
        score['euclidean_distance_model_human'] = mean_euclidean_distance
        score['euclidean_distance_model_gt'] = mean_euclidean_gt
        score['euclidean_distance_human_gt'] = mean_euclidean_hum
        return score


class ModelHumanCohenK:
    def __call__(self, model_data, human_data):
        return self.get_score(model_data, human_data)

    def extract_filename(self, file_name):
        file_name = str(file_name)
        file_name = file_name.split('.')[0]
        if '_img' in file_name:
            return file_name.split('_img')[0]
        return file_name

    def get_score(self, MD, HD):
        if not isinstance(MD, pd.DataFrame):
            if hasattr(MD, "to_dataframe"):
                MD = MD.to_dataframe().reset_index()
            else:
                _MD_temp = {}
                for coord, dims, values in walk_coords(MD):
                    _MD_temp[coord] = np.array(values)
                MD = pd.DataFrame(_MD_temp)

        if not isinstance(HD, pd.DataFrame):
            if hasattr(HD, "to_dataframe"):
                HD = HD.to_dataframe().reset_index()
            else:
                _HD_temp = {}
                for coord, dims, values in walk_coords(HD):
                    _HD_temp[coord] = np.array(values)
                HD = pd.DataFrame(_HD_temp)

        if 'scenario' not in MD.columns and 'scenario' in HD.columns:
            stim_to_scen = dict(
                zip(
                    HD['stimulus_id'].apply(self.extract_filename),
                    HD['scenario'],
                )
            )
            MD['clean_stim'] = MD['stimulus_id'].apply(self.extract_filename)
            MD['scenario'] = MD['clean_stim'].map(stim_to_scen)
            MD = MD.dropna(subset=['scenario'])

        if MD.empty:
            return {'center': 0.0, 'error': 0.0}

        scenarios = sorted(set(MD['scenario'].values))
        cohen_ks, accu = [], []

        for scenario in scenarios:
            _MD = MD[MD['scenario'] == scenario].copy()
            _HD = HD[HD['scenario'] == scenario].copy()
            _MD['stimulus_id'] = _MD['stimulus_id'].apply(
                self.extract_filename)

            if 'label' in _MD.columns:
                accu.append(accuracy_score(_MD['label'], _MD['choice']))
            else:
                accu.append(np.nan)

            _MD = _MD.sort_values('stimulus_id')

            gameIDs = set(_HD['gameID'])
            measures_for_model = []

            for gameID in gameIDs:
                _HD_game = _HD[_HD['gameID'] == gameID].copy()
                _HD_game['stimulus_id'] = _HD_game['stimulus_id'].apply(
                    self.extract_filename)
                _HD_game = _HD_game.sort_values('stimulus_id')

                joint_stim_names = np.intersect1d(
                    _HD_game['stimulus_id'], _MD['stimulus_id'])
                subset_MD = _MD[_MD['stimulus_id'].isin(joint_stim_names)]
                subset_HD = _HD_game[_HD_game['stimulus_id'].isin(
                    joint_stim_names)]

                subset_HD_unique = subset_HD.groupby(
                    'stimulus_id').first().reset_index()
                subset_MD_unique = subset_MD.groupby(
                    'stimulus_id').first().reset_index()

                if len(subset_HD_unique) > 0 and len(subset_MD_unique) > 0:
                    human_responses = subset_HD_unique['responseBool'].values
                    model_responses = subset_MD_unique['choice'].values
                    measure = cohen_kappa_score(
                        model_responses, human_responses)
                    measures_for_model.append(measure)
            if measures_for_model:
                cohen_ks.append(np.median(measures_for_model))
            else:
                cohen_ks.append(0.0)

        cohen_ks = np.array(cohen_ks)
        center = np.mean(cohen_ks) if len(cohen_ks) > 0 else 0.0
        error = np.std(cohen_ks) if len(cohen_ks) > 0 else 0.0

        score = {
            'center': center,
            'error': error,
            'scenario': scenarios,
            'scenario_accuracy': accu,
            'raw_values': cohen_ks,
        }

        if 'label' in MD.columns:
            values = np.mean(MD['choice'] == MD['label'])
            score['accuracy'] = values

        return score

# ---------------------------------------------------------------------
# Base Placement Class
# ---------------------------------------------------------------------


class OnlinePhysionPlacement(OnlineTransformerClassifier):
    """
    Base class for Physion Placement metrics.
    Requires a concrete `human_data_filename`.
    """

    def __init__(
        self,
        num_classes: int = 256,
        input_feature_dim: int = 512,  # Adjusted default, user should specify
        human_data_filename: Optional[str] = None,
        embed_dim: int = 252,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        lr_options: Optional[List[float]] = None,
        wd_options: Optional[List[float]] = None,
        n_epochs: int = 50,
        patience: int = 10,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ceiling: Optional[np.ndarray] = None,
        scheduler_type: str = "cosine",
    ):
        super().__init__(
            num_classes=num_classes,
            input_feature_dim=input_feature_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            lr_options=[1e-3, 1e-4],  # lr_options,
            wd_options=[0, 1e-2],  # wd_options,
            n_epochs=n_epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            ceiling=ceiling,
            scheduler_type=scheduler_type,
        )
        self.human_data_filename = human_data_filename

    def _load_human_data(self):
        if not self.human_data_filename:
            print("Error: No human data filename provided for Physion Placement.")
            return None

        fpath = ensure_physion_file(self.human_data_filename)
        if fpath:
            try:
                return xr.load_dataset(fpath, engine="h5netcdf")
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
        return None

    def compute_raw(
        self,
        extractor: OnlineFeatureExtractor,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        results = super().compute_raw(extractor, train_dataloader,
                                      val_dataloader, test_dataloader)

        if 'preds' in results and 'gt' in results and 'stimulus' in results:
            human_data = self._load_human_data()

            if human_data is not None:
                md_df = pd.DataFrame({
                    'stimulus_id': results['stimulus'],
                    'choice': results['preds'],
                    'contacts': results['gt'].flatten()
                })

                metric = ModelHumanAccuracyChebyshev()
                score_dict = metric(md_df, human_data)

                for k, v in score_dict.items():
                    if k == 'center':
                        continue
                    if isinstance(v, (int, float, np.number)):
                        results[f"physion_placement_{k}"] = float(v)
                    elif isinstance(v, np.ndarray) and v.size == 1:
                        results[f"physion_placement_{k}"] = float(v)

                results['physion_placement_score'] = score_dict['center']
            else:
                print(
                    f"Skipping Placement scoring: {self.human_data_filename} unavailable.")

        return results

# ---------------------------------------------------------------------
# Concrete Placement Classes
# ---------------------------------------------------------------------


class OnlinePhysionPlacementDetection(OnlinePhysionPlacement):
    def __init__(self, **kwargs):
        # Enforce the correct filename, allow other kwargs to pass through
        kwargs['human_data_filename'] = "PhysionHumanPlacementDetection2024.nc"
        super().__init__(**kwargs)


class OnlinePhysionPlacementPrediction(OnlinePhysionPlacement):
    def __init__(self, **kwargs):
        kwargs['human_data_filename'] = "PhysionHumanPlacementPrediction2024.nc"
        super().__init__(**kwargs)


# ---------------------------------------------------------------------
# Base Contact Class
# ---------------------------------------------------------------------

class OnlinePhysionContact(OnlineTransformerClassifier):
    """
    Base class for Physion Contact metrics.
    Requires a concrete `human_data_filename`.
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_feature_dim: int = 512,
        human_data_filename: Optional[str] = None,
        embed_dim: int = 252,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        lr_options: Optional[List[float]] = None,
        wd_options: Optional[List[float]] = None,
        n_epochs: int = 50,
        patience: int = 10,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ceiling: Optional[np.ndarray] = None,
        scheduler_type: str = "cosine",
    ):
        super().__init__(
            num_classes=num_classes,
            input_feature_dim=input_feature_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            lr_options=lr_options,
            wd_options=wd_options,
            n_epochs=n_epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            ceiling=ceiling,
            scheduler_type=scheduler_type,
        )
        self.human_data_filename = human_data_filename

    def _load_human_data(self):
        if not self.human_data_filename:
            print("Error: No human data filename provided for Physion Contact.")
            return None

        fpath = ensure_physion_file(self.human_data_filename)
        if fpath:
            try:
                return xr.load_dataset(fpath, engine="h5netcdf")
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
        return None

    def compute_raw(
        self,
        extractor: OnlineFeatureExtractor,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        results = super().compute_raw(extractor, train_dataloader,
                                      val_dataloader, test_dataloader)

        if 'preds' in results and 'gt' in results and 'stimulus' in results:
            human_data = self._load_human_data()

            if human_data is not None:
                stimulus = np.array(results['stimulus']).ravel()
                preds = np.array(results['preds'])
                gt = np.array(results['gt']).ravel()

                md_df = pd.DataFrame({
                    'stimulus_id': stimulus,
                    'choice':      preds,
                    'label':       gt,
                })

                metric = ModelHumanCohenK()
                score_dict = metric(md_df, human_data)

                results['physion_contact_cohen_kappa'] = score_dict['center']

                if 'error' in score_dict:
                    results['physion_contact_error'] = float(
                        score_dict['error'])
                if 'accuracy' in score_dict:
                    results['physion_contact_model_accuracy'] = float(
                        score_dict['accuracy'])
                if 'scenario' in score_dict:
                    results['physion_contact_scenarios'] = score_dict['scenario']
                if 'scenario_accuracy' in score_dict:
                    results['physion_contact_scenario_accuracy'] = [
                        float(a) if a is not None and not pd.isna(a) else None
                        for a in score_dict['scenario_accuracy']
                    ]
                if 'raw_values' in score_dict:
                    results['physion_contact_scenario_kappa'] = [
                        float(k) if k is not None and not pd.isna(k) else None
                        for k in score_dict['raw_values']
                    ]
            else:
                print(
                    f"Skipping Contact scoring: {self.human_data_filename} unavailable.")
        return results

# ---------------------------------------------------------------------
# Concrete Contact Classes
# ---------------------------------------------------------------------


class OnlinePhysionContactDetection(OnlinePhysionContact):
    def __init__(self, **kwargs):
        kwargs['human_data_filename'] = "PhysionHumanContactDetection2024.nc"
        super().__init__(**kwargs)


class OnlinePhysionContactPrediction(OnlinePhysionContact):
    def __init__(self, **kwargs):
        kwargs['human_data_filename'] = "PhysionHumanContactPrediction2024.nc"
        super().__init__(**kwargs)
