import os
from typing import Dict

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SAM:
    """Loads pre-trained Segment Anything Model (SAM) models."""

    def __init__(self):
        """Initializes the SAM loader."""
        self.model_mappings = {
            "SAM-BASE": "facebook/sam-vit-base",
            "SAM-LARGE": "facebook/sam-vit-large",
            "SAM-HUGE": "facebook/sam-vit-huge",
        }

        # Current processor
        self.processor = None
        self.static = True  # SAM default image size

    def preprocess_fn(self, input_data, fps=None, prompts=None):
        """
        Preprocesses input data for SAM.

        Args:
            input_data: PIL Image, file path (str), or numpy array.
            prompts: Dict of prompt inputs (points, boxes, masks).

        Returns:
            dict: Preprocessed input for the model.

        Raises:
            ValueError: If the processor is not initialized or input is invalid.
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

        # Process image with SAM processor
        inputs = self.processor(img, return_tensors="pt").pixel_values

        return inputs

    def get_model(self, identifier):
        """
        Loads a SAM model based on the identifier.

        Args:
            identifier (str):  Identifier for the SAM variant.

        Returns:
            model: The loaded SAM model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                self.processor = AutoProcessor.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                return model
        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses SAM model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from IJEPA model
                as a numpy array.

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1).
        """
        batch_size = features_np.shape[0]
        flattened_features = features_np.reshape(batch_size, -1)

        return flattened_features
