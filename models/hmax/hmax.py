import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from google.cloud import storage
from google.auth.credentials import AnonymousCredentials
from PIL import Image
from sklearn.datasets import get_data_home
from torchvision import transforms

from .hmax_model import hmax

from torchvision.transforms import v2


class HMAX:
    """Loads pre-trained ConvNeXt models."""

    def __init__(self):
        """Initializes the ConvNeXt loader."""
        self.model_mappings = {
            "HMAX": "gs://stanford_neuroai_models/hmax/universal_patch_set.mat",
        }

        # Current processor
        self.processor = v2.Compose([
            v2.ToImage(),  # PIL / ndarray -> (C, H, W) tensor, uint8
            v2.Resize((64, 64), antialias=True),
            v2.ToDtype(torch.float32, scale=True),  # [0, 1] float32
            v2.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
        ])

        self.static = True

    def _to_pil_rgb(self, x):
        """Convert supported input types to RGB PIL.Image."""
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        elif isinstance(x, np.ndarray):
            # ensure uint8
            return Image.fromarray(x.astype(np.uint8)).convert("RGB")
        elif isinstance(x, str) and os.path.isfile(x):
            return Image.open(x).convert("RGB")
        else:
            raise ValueError(
                "Input must be a PIL Image, file path, or numpy array"
            )

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for HMAX using torchvision.transforms.v2.

        Args:
            input_data: PIL Image, file path (str), numpy array, or list of these.

        Returns:
            torch.Tensor: (N, C, 256, 256) preprocessed tensor.
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first."
            )

        # Normalize to a list of frames
        if isinstance(input_data, (list, tuple)):
            frames = input_data
        else:
            frames = [input_data]

        # Convert all frames to RGB PIL and apply transforms
        tensors = []
        for frame in frames:
            pil_img = self._to_pil_rgb(frame)
            tensors.append(self.processor(pil_img))

        # (N, C, 256, 256)
        return torch.stack(tensors, dim=0)

    def get_model(self, identifier):
        """
        Loads a HMAX model based on the identifier.

        Args:
            identifier (str):  Identifier for the ConvNeXt variant.

        Returns:
            model: The loaded ConvNeXt model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_params in self.model_mappings.items():
            # Define weights directory and ensure it exists.
            weights_dir = os.path.join(
                get_data_home(), 'weights', self.__class__.__name__)
            os.makedirs(weights_dir, exist_ok=True)

            # Determine local file name from the URL.
            file_name = model_params.split('/')[-1]
            model_path = os.path.join(weights_dir, file_name)

            # Download the model weights if they are not already present locally.
            if not os.path.exists(model_path):
                print(
                    f"Downloading model weights from {model_params} to {model_path} ...")
                if model_params.startswith("gs://"):
                    # Parse the GCS URL to get the bucket name and blob name
                    parts = model_params[5:].split('/', 1)
                    if len(parts) != 2:
                        raise ValueError("Invalid GCS URL format.")
                    bucket_name, blob_name = parts
                    client = storage.Client(
                        credentials=AnonymousCredentials())
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    blob.download_to_filename(model_path)
                else:
                    raise ValueError(
                        f"Unsupported model URL: {model_params}")
            model = hmax(model_path)
            return model

        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses ResNet model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from ResNet model
                as a numpy array. Expected shape:
                (batch_size, seq_len, feature_dim) or (seq_len, feature_dim)

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1),
                where N is batch size (or 1 if single sample).
        """
        batch_size, T = features_np.shape[0], features_np.shape[1]
        flattened_features = features_np.reshape(batch_size, T, -1)

        return flattened_features
