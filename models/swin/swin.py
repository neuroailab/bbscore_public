import os

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, SwinModel


class Swin:
    """Loads pre-trained Swin Transformer models."""

    def __init__(self):
        """Initializes the Swin Transformer loader."""
        self.model_mappings = {
            "SWIN-TINY": "microsoft/swin-tiny-patch4-window7-224",
            "SWIN-SMALL": "microsoft/swin-small-patch4-window7-224",
            "SWIN-BASE": "microsoft/swin-base-patch4-window7-224",
            "SWIN-LARGE": "microsoft/swin-large-patch4-window7-224",
        }

        # Current processor
        self.processor = None
        self.static = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for Swin Transformer.

        Args:
            input_data: PIL Image, file path (str), or numpy array.

        Returns:
            dict: Preprocessed input for the model.

        Raises:
            ValueError: If processor is not initialized or input is invalid.
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first.")

        # Handle different input types
        if isinstance(input_data, str) and os.path.isfile(input_data):
            img = Image.open(input_data).convert("RGB")
        elif isinstance(input_data, np.ndarray):
            img = Image.fromarray(np.uint8(input_data)).convert("RGB")
        elif isinstance(input_data, Image.Image):
            img = input_data.convert("RGB")
        else:
            raise ValueError(
                "Input must be a PIL Image, file path, or numpy array")

        # Process with Swin processor
        return self.processor(img, return_tensors="pt").pixel_values

    def get_model(self, identifier):
        """
        Loads a Swin Transformer model based on the identifier.

        Args:
            identifier (str):  Identifier for the Swin variant.

        Returns:
            model: The loaded Swin Transformer model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                model = SwinModel.from_pretrained(model_name)
                return model
        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses SWIN model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from IJEPA model
                as a numpy array.

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1).
        """
        batch_size = features_np.shape[0]
        flattened_features = features_np.reshape(batch_size, -1)

        return flattened_features
