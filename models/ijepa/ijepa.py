import os

import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel


class IJEPA:
    """Loads pre-trained Image Joint-Embedding Predictive Architecture (I-JEPA) models."""

    def __init__(self):
        """Initializes the I-JEPA loader."""
        self.model_mappings = {
            "IJEPA-HUGE14-1K": "facebook/ijepa_vith14_1k",
            "IJEPA-HUGE14-22K": "facebook/ijepa_vith14_22k",
            "IJEPA-HUGE16-1K": "facebook/ijepa_vith16_1k",
            "IJEPA-GIANT": "facebook/ijepa_vitg16_22k"
        }

        # Base transformation pipeline
        self.static = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for I-JEPA.

        Args:
            input_data: PIL Image, file path (str), or numpy array.
            apply_masking (bool): Apply random masking (for training).
            mask_ratio (float): Ratio of patches to mask if apply_masking.

        Returns:
            torch.Tensor or tuple: Preprocessed input tensor(s).

        Raises:
            ValueError: If input is invalid
        """
        # Handle different input types
        if isinstance(input_data, str) and os.path.isfile(input_data):
            img = Image.open(input_data).convert("RGB")
        elif isinstance(input_data, np.ndarray):
            img = Image.fromarray(np.uint8(input_data)).convert("RGB")
        elif isinstance(input_data, Image.Image):
            img = input_data.convert("RGB")
        elif isinstance(input_data, Image.Image):
            img = [i.convert("RGB") for i in input_data]
        else:
            raise ValueError(
                "Input must be a PIL Image, file path, or numpy array")

        return self.processor(images=img, return_tensors="pt").pixel_values

    def get_model(self, identifier):
        """
        Loads a IJEPA model based on the identifier.

        Args:
            identifier (str): Identifier for the IJEPA variant.

        Returns:
            model: The loaded IJEPA model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                self.processor = AutoProcessor.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                return model  # Return the processor, not a string
        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses IJEPA model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from IJEPA model
                as a numpy array.

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1).
        """
        batch_size = features_np.shape[0]
        flattened_features = features_np.reshape(batch_size, -1)

        return flattened_features
