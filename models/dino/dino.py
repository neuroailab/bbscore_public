import os

import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


class DINO:
    """Loads pre-trained DINO self-supervised vision models."""

    def __init__(self):
        """Initializes the DINO loader."""
        self.model_mappings = {
            "DINO-VITB8": "facebook/dino-vitb8",
            "DINO-VITB16": "facebook/dino-vitb16",
            "DINO-VITS8": "facebook/dino-vits8",
            "DINO-VITS16": "facebook/dino-vits16",
        }

        # DINO params
        self.static = True
        self.processor = None

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for DINO.

        Args:
            input_data: PIL Image, file path (str), or numpy array.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
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

        return self.processor(images=img, return_tensors="pt").pixel_values

    def get_model(self, identifier):
        """
        Loads a DINO model based on the identifier.

        Args:
            identifier (str): Identifier for the DINO variant.

        Returns:
            model: The loaded DINOv2 model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                self.processor = ViTImageProcessor.from_pretrained(model_name)
                model = ViTModel.from_pretrained(model_name)
                return model  # Return the processor, not a string
        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses DINO model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from DINO model
                as a numpy array.

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1).
        """
        batch_size = features_np.shape[0]
        flattened_features = features_np.reshape(batch_size, -1)

        return flattened_features
