import os

import cv2
import math
import numpy as np
import torch

from google.cloud import storage
from google.auth.credentials import AnonymousCredentials
from PIL import Image
from sklearn.datasets import get_data_home
from torchvision.transforms import v2 as T2

from .mcvd_model import MCVD_Wrapper

import torch


class MCVD:
    """Loads pre-trained FitVid (Masked Autoencoder) models."""

    def __init__(self):
        """Initializes the FitVid loader."""
        self.model_mappings = {
            "MCVD-EGO": ("gs://stanford_neuroai_models/mcvd/mcvd_ego4d.pt", "gs://stanford_neuroai_models/mcvd/config_ego4d.yml"),
            "MCVD-PHYS": ("gs://stanford_neuroai_models/mcvd/mcvd_physion.pt", "gs://stanford_neuroai_models/mcvd/config_physion.yml"),
        }

        # Default parameters
        self.image_size = (64, 64)
        self.processor = T2.Compose([
            # 1) resize every frame to `self.image_size`
            T2.Resize(self.image_size, antialias=True),

            # 2) convert uint8 [0,255] tensor to float32 [0.0,1.0]
            #    (v2.ToTensor is deprecated in favor of ToImage+ToDtype)
            T2.ToImage(),                             # HWC→CHW, wraps in Image TVTensor
            # scales [0,255]→[0.0,1.0]
            T2.ToDtype(torch.float32, scale=True),
        ])
        self.fps = 25
        # Static flag
        self.static = False
        self.sequence_mode = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for FitVid.

        Args:
            input_data: Video file path (str), list of frames, or tensor.
            apply_masking (bool): Apply random masking (for pre-training).
            mask_ratio (float): Ratio of patches to mask if apply_masking.

        Returns:
            dict: Preprocessed input for the model.

        Raises:
            ValueError: If the processor is not initialized or input is invalid.
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first.")

        frames = []

        if isinstance(input_data, list) or isinstance(input_data, Image.Image):
            if isinstance(input_data, Image.Image):
                input_data = [input_data]
            # list of PIL.Image or numpy arrays
            arrs = []
            for frame in input_data:
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                # now numpy H×W×C
                tensor = torch.from_numpy(frame).permute(2, 0, 1)  # C,H,W
                arrs.append(tensor)
            frames = torch.stack(arrs, dim=0)  # (T, C, H, W)
        else:
            raise ValueError(
                "Input must be a filepath, list of frames, or tensor")

        # 2) Apply the entire v2 pipeline in one go
        #    works on (T, C, H, W) natively, no Python loop
        frames = self.processor(frames)

        # 3) FPS resampling
        if fps is not None and self.fps is not None and fps != self.fps and frames.shape[0] > 1:
            orig_T = frames.shape[0]
            duration = orig_T / float(fps)
            new_T = int(round(duration * float(self.fps)))
            idx = torch.linspace(0, orig_T - 1, new_T).long()
            frames = frames[idx]

        return frames[::2]

    def get_model(self, identifier):
        """
        Loads a FitVid model based on the identifier.

        Args:
            identifier (str): Identifier for the FitVid variant.

        Returns:
            model: The loaded FitVid model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, (model_url, config_url) in self.model_mappings.items():
            if identifier.startswith(prefix):
                # Define weights directory and ensure it exists.
                weights_dir = os.path.join(
                    get_data_home(), 'weights', self.__class__.__name__)
                os.makedirs(weights_dir, exist_ok=True)

                # Determine local file name from the URL.
                file_name = model_url.split('/')[-1]
                model_path = os.path.join(weights_dir, file_name)

                # Download the model weights if they are not already present locally.
                if not os.path.exists(model_path):
                    print(
                        f"Downloading model weights from {model_url} to {model_path} ...")
                    if model_url.startswith("gs://"):
                        # Parse the GCS URL to get the bucket name and blob name
                        parts = model_url[5:].split('/', 1)
                        if len(parts) != 2:
                            raise ValueError("Invalid GCS URL format.")
                        bucket_name, blob_name = parts
                        client = storage.Client(
                            credentials=AnonymousCredentials())
                        bucket = client.bucket(bucket_name)
                        blob = bucket.blob(blob_name)
                        blob.download_to_filename(model_path)
                    else:
                        raise ValueError(f"Unsupported model URL: {model_url}")

                config_path = os.path.join(
                    weights_dir, config_url.split('/')[-1])
                if not os.path.exists(config_path):
                    print(
                        f"Downloading model config from {config_url} to {config_path} ...")
                    if config_url.startswith("gs://"):
                        parts = config_url[5:].split('/', 1)
                        if len(parts) != 2:
                            raise ValueError("Invalid GCS URL format.")
                        bucket_name, blob_name = parts
                        client = storage.Client(
                            credentials=AnonymousCredentials())
                        bucket = client.bucket(bucket_name)
                        blob = bucket.blob(blob_name)
                        blob.download_to_filename(config_path)
                    else:
                        raise ValueError(f"Unsupported model URL: {model_url}")

                # Instantiate the model using the local weights file
                self.model = MCVD_Wrapper(model_path, config_path)
                # self.model = torch.compile(self.model.half())
                return self.model
        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses FitVid model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from FitVid model
                as a numpy array. Expected shape:
                (batch_size, seq_len, feature_dim) or (seq_len, feature_dim)

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1),
                where N is batch size (or 1 if single sample).
        """
        features_np = features_np[:, ::101]
        batch_size, T = features_np.shape[0], features_np.shape[1]
        flattened_features = features_np.reshape(batch_size, T, -1)

        return flattened_features
