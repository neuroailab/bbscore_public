import os

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import ConvNextImageProcessor, ConvNextForImageClassification


class ConvNeXt:
    """Loads pre-trained ConvNeXt models."""

    def __init__(self):
        """Initializes the ConvNeXt loader."""
        self.model_mappings = {
            "CONVNEXT-TINY": "facebook/convnext-tiny-224",
            "CONVNEXT-SMALL": "facebook/convnext-small-224",
            "CONVNEXT-BASE": "facebook/convnext-base-224",
            "CONVNEXT-LARGE": "facebook/convnext-large-224",
            "CONVNEXT-XLARGE": "facebook/convnext-xlarge-224",
        }

        # Current processor
        self.processor = None

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
            img = Image.open(input_data).convert("RGB")
        elif isinstance(input_data, np.ndarray):
            img = Image.fromarray(np.uint8(input_data)).convert("RGB")
        elif isinstance(input_data, Image.Image):
            img = input_data.convert("RGB")
        elif isinstance(input_data, list):
            img = [i.convert("RGB") for i in input_data]
        else:
            raise ValueError(
                "Input must be a PIL Image, file path, or numpy array")

        # Process with ConvNeXt processor
        return self.processor(img, return_tensors="pt").pixel_values

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
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                self.processor = ConvNextImageProcessor.from_pretrained(
                    model_name)
                model = ConvNextForImageClassification.from_pretrained(
                    model_name)
                return model
        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )
