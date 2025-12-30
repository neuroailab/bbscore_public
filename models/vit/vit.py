import os

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, ViTForImageClassification

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class VisionTransformer:
    """Loads pre-trained Vision Transformer (ViT) models."""

    def __init__(self):
        """Initializes the ViT loader."""
        self.model_mappings = {
            "VIT-BASE": "google/vit-base-patch16-224",
            "VIT-LARGE": "google/vit-large-patch16-224",
            "VIT-HUGE": "google/vit-huge-patch14-224-in21k",
        }

        # Transformation pipeline
        self.processor = None

        # Static flag
        self.static = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for ViT.

        Args:
            input_data: PIL Image, file path (str), or numpy array.

        Returns:
            torch.Tensor: Preprocessed input tensor.

        Raises:
            ValueError: If the input type is invalid.
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

        tensor = self.processor(img, return_tensors="pt").pixel_values
        return tensor.half()

    def get_model(self, identifier):
        """
        Loads a ViT model based on the identifier.

        Args:
            identifier (str): Identifier for the ViT variant.

        Returns:
            ViTForImageClassification: The loaded model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                model = ViTForImageClassification.from_pretrained(
                    model_name, torch_dtype=torch.float16)
                self.processor = ViTImageProcessor.from_pretrained(
                    model_name, use_fast=True)
                model = torch.compile(model)
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
        batch_size, T = features_np.shape[0], features_np.shape[1]
        flattened_features = features_np.reshape(batch_size, T, -1)

        return flattened_features
