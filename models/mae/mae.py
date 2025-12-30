from typing import Dict

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTMAEForPreTraining

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class MAE:
    """
    A class for loading pre-trained ViT MAE models from Hugging Face and
    providing a default preprocessing pipeline for image inputs.
    """

    def __init__(self):
        """
        Initialize the MAE loader and define a default preprocessing function.
        """
        self.model_mappings: Dict[str, str] = {
            "MAE-BASE": "facebook/vit-mae-base",
            "MAE-LARGE": "facebook/vit-mae-large",
        }

        # Default preprocessing function
        self.processor = None

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

        return self.processor(images=img, return_tensors="pt").pixel_values.half()

    def get_model(self, identifier: str) -> ViTMAEForPreTraining:
        """
        Load a ViT MAE model based on the identifier.

        Args:
            identifier (str): String identifier that starts with either
                'MAE-BASE' or 'MAE-LARGE'.

        Returns:
            ViTMAEForPreTraining: The loaded model with mask_ratio set to 0.0.

        Raises:
            ValueError: If the identifier doesn't match any known model types.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                # Instantiate the model with mask_ratio set to 0.0
                self.processor = AutoImageProcessor.from_pretrained(
                    model_name, use_fast=True)
                model = ViTMAEForPreTraining.from_pretrained(
                    model_name, mask_ratio=0.0, torch_dtype=torch.float16)
                model = torch.compile(model)
                return model

        raise ValueError(
            f"Unknown model identifier: {identifier}. Available prefixes: "
            f"{', '.join(self.model_mappings.keys())}"
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
