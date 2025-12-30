import os

import numpy as np
import torch
from PIL import Image
from robustbench.utils import load_model
from torchvision import transforms


class ROBUST:
    """Loads pre-trained models (vision encoder only, by default)."""

    def __init__(self):
        """Initializes the loader."""
        self.model_mappings = {
            "Robust-Swin-L": {"model": "Xu2024MIMIR_Swin-L",
                              "data": "imagenet",
                              "threat": "Linf",
                              "preprocess": "BicubicRes256Crop224",
                              },
            "ConvNeXtV2-L + Swin-L": {
                "model": "Bai2024MixedNUTS",
                         "data": "imagenet",
                         "threat": "Linf",
                         "preprocess": "BicubicRes256Crop224",
            },
            "RaWideResNet-101-2": {
                "model": "Peng2023Robust",
                "data": "imagenet",
                         "threat": "Linf",
                         "preprocess": "Res256Crop224",
            },
            "ViT-B-ConvStem": {
                "model": "Singh2023Revisiting_ViT-B-ConvStem",
                "data": "imagenet",
                         "threat": "Linf",
                         "preprocess": "BicubicRes256Crop224",
            },
            "Robust-ResNet-50": {
                "model": "Salman2020Do_R50",
                "data": "imagenet",
                         "threat": "Linf",
                         "preprocess": "Res256Crop224",
            },
            'DeiT-B': {
                "model": "Tian2022Deeper_DeiT-B",
                "data": "imagenet",
                "threat": "corruptions",
                         "preprocess": "Res256Crop224",
            },
            'NoisyMix-ResNet-50': {
                "model": "Erichson2022NoisyMix_new",
                "data": "imagenet",
                "threat": "corruptions",
                         "preprocess": "Res256Crop224",
            }
        }

        # Current processor
        self.processor = {
            'Res256Crop224':
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ]),
            'Crop288':
            transforms.Compose([transforms.CenterCrop(288),
                                transforms.ToTensor()]),
            None:
            transforms.Compose([transforms.ToTensor()]),
            'Res224':
            transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor()
            ]),
            'BicubicRes256Crop224':
            transforms.Compose([
                transforms.Resize(
                    256,
                    interpolation=transforms.InterpolationMode("bicubic")),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        }

        # Static flag
        self.static = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses image data for model.

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
            img = self.processor(Image.open(input_data).convert("RGB"))
        elif isinstance(input_data, np.ndarray):
            img = self.processor(Image.fromarray(
                np.uint8(input_data)).convert("RGB"))
        elif isinstance(input_data, Image.Image):
            img = self.processor(input_data.convert("RGB"))
        elif isinstance(input_data, list):
            img = [self.processor(i.convert("RGB")) for i in input_data]
            img = torch.stack(img)
        else:
            raise ValueError(
                "Input must be a PIL Image, file path, or numpy array")

        return img

    def get_model(self, identifier):
        """
        Loads a model or its vision encoder.

        Args:
            identifier (str): Identifier for the model variant.
            vision_only (bool): Return only the vision encoder if True.

        Returns:
            model: The loaded model or vision encoder.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_params in self.model_mappings.items():
            if identifier == prefix:
                model = load_model(
                    model_name=model_params['model'], dataset=model_params['data'], threat_model=model_params['threat'])
                self.processor = self.processor[model_params['preprocess']]
                return model

        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from model
                as a numpy array. Expected shape:
                (batch_size, seq_len, feature_dim) or (seq_len, feature_dim)

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1),
                where N is batch size (or 1 if single sample).
        """
        batch_size = features_np.shape[0]
        flattened_features = features_np.reshape(batch_size, -1)

        return flattened_features
