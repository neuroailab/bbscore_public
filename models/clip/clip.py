import os

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPVisionModel, CLIPModel

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class CLIP:
    """Loads pre-trained CLIP models (vision encoder only, by default)."""

    def __init__(self):
        """Initializes the CLIP loader."""
        self.model_mappings = {
            "CLIP-VIT-B-32": "openai/clip-vit-base-patch32",
            "CLIP-VIT-B-16": "openai/clip-vit-base-patch16",
            "CLIP-VIT-L-14": "openai/clip-vit-large-patch14",
            "CLIP-VIT-L-14-336": "openai/clip-vit-large-patch14-336",
        }

        # Current processor and model
        self.processor = None
        self.input_size = 224  # Default size

        # Static flag
        self.static = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses image data for CLIP.

        Args:
            input_data: PIL Image, file path (str), or numpy array.

        Returns:
            dict: Preprocessed input for the model.
        Raises:
            ValueError: If processor is not initialized or input is invalid
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized.  Call get_model() first.")

        if isinstance(input_data, str) and os.path.isfile(input_data):
            img = Image.open(input_data).convert("RGB")
            if img.width != self.input_size or img.height != self.input_size:
                img = img.resize((self.input_size, self.input_size))
        elif isinstance(input_data, np.ndarray):
            img = Image.fromarray(np.uint8(input_data)).convert("RGB")
            if img.width != self.input_size or img.height != self.input_size:
                img = img.resize((self.input_size, self.input_size))
        elif isinstance(input_data, Image.Image):
            img = input_data.convert("RGB")
            if img.width != self.input_size or img.height != self.input_size:
                img = img.resize((self.input_size, self.input_size))
        elif isinstance(input_data, list):
            img = []
            for i in input_data:
                img += [i.convert("RGB").resize((self.input_size,
                                                 self.input_size))]
        else:
            raise ValueError(
                "Input must be a PIL Image, file path, or numpy array")

        return self.processor(images=img, return_tensors="pt").pixel_values.half()

    def get_model(self, identifier, vision_only=True):
        """
        Loads a CLIP model or its vision encoder.

        Args:
            identifier (str): Identifier string for the CLIP variant.
            vision_only (bool): Load only the vision encoder if True.

        Returns:
            model: The loaded CLIP model or vision encoder.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                # Update input size for CLIP-VIT-L-14-336
                if identifier.startswith("CLIP-VIT-L-14-336"):
                    self.input_size = 336

                # Load only vision model if requested
                if vision_only:
                    model = CLIPVisionModel.from_pretrained(
                        model_name, torch_dtype=torch.float16)
                else:
                    model = CLIPModel.from_pretrained(
                        model_name, torch_dtype=torch.float16)

                self.processor = CLIPProcessor.from_pretrained(
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
