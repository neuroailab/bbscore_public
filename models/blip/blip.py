import os

import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor


class BLIP:
    """Loads pre-trained BLIP models (vision encoder only, by default)."""

    def __init__(self):
        """Initializes the BLIP loader."""
        self.model_mappings = {
            "BLIP-BASE": "Salesforce/blip-image-captioning-base",
            "BLIP-LARGE": "Salesforce/blip-image-captioning-large",
            "BLIP-VQA": "Salesforce/blip-vqa-base",
            "BLIP-ITM": "Salesforce/blip-itm-base-coco",
        }

        # Current processor
        self.processor = None
        self.image_size = 384  # Default BLIP image size

        # Static flag
        self.static = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses image data for BLIP.

        Args:
            input_data: PIL Image, file path (str), or numpy array.

        Returns:
            dict: Preprocessed input for the model.

        Raises:
            ValueError: If processor not initialized, or input invalid.
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first.")

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

        if img.width != self.image_size or img.height != self.image_size:
            img = img.resize((self.image_size, self.image_size))

        return self.processor(images=img, return_tensors="pt").pixel_values

    def get_model(self, identifier, vision_only=True):
        """
        Loads a BLIP model or its vision encoder.

        Args:
            identifier (str): Identifier for the BLIP variant.
            vision_only (bool): Return only the vision encoder if True.

        Returns:
            model: The loaded BLIP model or vision encoder.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                self.processor = AutoProcessor.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)

                if vision_only and hasattr(model, "vision_model"):
                    return model.vision_model

                return model

        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses VideoMAE model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from VideoMAE model
                as a numpy array. Expected shape:
                (batch_size, seq_len, feature_dim) or (seq_len, feature_dim)

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1),
                where N is batch size (or 1 if single sample).
        """
        batch_size = features_np.shape[0]
        flattened_features = features_np.reshape(batch_size, -1)

        return flattened_features
