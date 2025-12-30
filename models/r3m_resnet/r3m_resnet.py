import os

import numpy as np
import torch
from PIL import Image
from r3m import load_r3m
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class R3M:
    """Loads pre-trained R3M models (vision encoder only, by default)."""

    def __init__(self):
        """Initializes the R3M loader."""
        self.model_mappings = {
            "R3M-ResNet50": "resnet50",
        }

        # Current processor
        self.processor = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            # FloatTensor [0..1]
                                             transforms.ToTensor(),
                                             transforms.Lambda(lambda t: t * 255)])

        # Static flag
        self.static = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses image data for R3M.

        Supports a single PIL Image or a list of PIL Images,
        and returns a batched Tensor ready for the model.

        Returns:
            dict: {"pixel_values": Tensor[B,3,224,224]}
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first.")

        # --- 1) Normalize input to a list of RGB PIL Images ---
        if isinstance(input_data, Image.Image):
            imgs = [input_data.convert("RGB")]
        elif isinstance(input_data, list):
            imgs = [im.convert("RGB") for im in input_data]
        else:
            raise ValueError("Input must be a PIL Image or list of PIL Images")

        # --- 2) ToTensor (0..1) and stack into B×3×H×W, then scale to 0..255 ---
        tensors = []
        for im in imgs:
            t = pil_to_tensor(im)           # uint8 C×H×W in [0..255]
            t = t.to(torch.float32)         # float32 C×H×W in [0..255]
            tensors.append(t)
        batch = torch.stack(tensors, dim=0)

        # --- 3) Resize shorter side to 256 with bilinear interpolate ---
        #     F.interpolate expects a 4D tensor.
        #     If your original images had different sizes, it will resize each in the batch.
        batch = F.interpolate(batch,
                              size=256,               # shorter edge → 256, keeps aspect ratio by default
                              mode='bilinear',
                              align_corners=False)

        # --- 4) Center-crop to 224×224 by tensor slicing ---
        _, _, H, W = batch.shape
        top = (H - 224) // 2
        left = (W - 224) // 2
        batch = batch[:, :, top:top+224, left:left+224]

        return batch

    def get_model(self, identifier):
        """
        Loads a R3M model or its vision encoder.

        Args:
            identifier (str): Identifier for the R3M variant.
            vision_only (bool): Return only the vision encoder if True.

        Returns:
            model: The loaded R3M model or vision encoder.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                model = load_r3m(model_name)
                model = torch.compile(model.half())
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
