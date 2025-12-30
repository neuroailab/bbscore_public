import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision import transforms


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class ResNet:
    """Loads pre-trained ResNet models (vision encoder only, by default)."""

    def __init__(self):
        """Initializes the ResNet loader."""
        self.model_mappings = {
            "ResNet18": "microsoft/resnet-18",
            "ResNet34": "microsoft/resnet-34",
            "ResNet50": "microsoft/resnet-50",
            "ResNet50-DINO": "dino_resnet50",
        }

        # Current processor
        self.processor = None

        # Static flag
        self.static = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses image data for ResNet.

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

        return self.processor(img, return_tensors="pt").pixel_values

    def get_model(self, identifier):
        """
        Loads a ResNet model or its vision encoder.

        Args:
            identifier (str): Identifier for the ResNet variant.
            vision_only (bool): Return only the vision encoder if True.

        Returns:
            model: The loaded ResNet model or vision encoder.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            print(prefix, identifier)
            if identifier == prefix:
                if 'DINO' in identifier:
                    model = torch.hub.load(
                        'facebookresearch/dino:main', model_name)
                    self.processor = AutoImageProcessor.from_pretrained(
                        "microsoft/resnet-50")
                else:
                    self.processor = AutoImageProcessor.from_pretrained(
                        model_name, use_fast=True)
                    model = AutoModelForImageClassification.from_pretrained(
                        model_name)
                    model = torch.compile(model)
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
        batch_size = features_np.shape[0]
        flattened_features = features_np.reshape(batch_size, -1)

        return flattened_features
