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

from .alexnet_barcode import AlexNetBarcode


class AlexNet:
    """Loads pre-trained ConvNeXt models."""

    def __init__(self):
        """Initializes the ConvNeXt loader."""
        self.model_mappings = {
            "AlexNet-Untrained": ('pytorch/vision:v0.10.0', 'alexnet', False),
            "AlexNet-ImageNet": ('pytorch/vision:v0.10.0', 'alexnet', True),
            "AlexNet-Barcode": ("gs://stanford_neuroai_models/alexnet/barcode.pt",),
        }

        # Current processor
        self.processor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        self.static = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for ConvNeXt.

        Args:
            input_data: PIL Image, file path (str), or numpy array.

        Returns:
            dict: Preprocessed input for the model.

        Raises:
            ValueError: If processor is not initialized or input is invalid
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first.")

        # Handle different input types
        if isinstance(input_data, str) and os.path.isfile(input_data):
            img = [self.processor(Image.open(input_data).convert("RGB"))]
        elif isinstance(input_data, np.ndarray):
            img = [self.processor(Image.fromarray(
                np.uint8(input_data)).convert("RGB"))]
        elif isinstance(input_data, Image.Image):
            img = [self.processor(input_data.convert("RGB"))]
        elif isinstance(input_data, list):
            img = [self.processor(i.convert("RGB")) for i in input_data]
        else:
            raise ValueError(
                "Input must be a PIL Image, file path, or numpy array")

        # Process with ConvNeXt processor
        return torch.stack(img)

    def get_model(self, identifier):
        """
        Loads a ConvNeXt model based on the identifier.

        Args:
            identifier (str):  Identifier for the ConvNeXt variant.

        Returns:
            model: The loaded ConvNeXt model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_params in self.model_mappings.items():
            if identifier == prefix and not 'Barcode' in identifier:
                model = torch.hub.load(
                    model_params[0], model_params[1], pretrained=model_params[2])
                return model
            elif identifier == prefix and 'Barcode' in identifier:
                # Define weights directory and ensure it exists.
                weights_dir = os.path.join(
                    get_data_home(), 'weights', self.__class__.__name__)
                os.makedirs(weights_dir, exist_ok=True)

                # Determine local file name from the URL.
                file_name = model_params[0].split('/')[-1]
                model_path = os.path.join(weights_dir, file_name)

                # Download the model weights if they are not already present locally.
                if not os.path.exists(model_path):
                    print(
                        f"Downloading model weights from {model_params[0]} to {model_path} ...")
                    if model_params[0].startswith("gs://"):
                        # Parse the GCS URL to get the bucket name and blob name
                        parts = model_params[0][5:].split('/', 1)
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
                            f"Unsupported model URL: {model_params[0]}")
                model = AlexNetBarcode()
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
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
