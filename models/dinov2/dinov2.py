import os

import cv2
import numpy as np
import torch

from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode, pil_to_tensor


class DINOv2:
    """Loads pre-trained DINOv2 self-supervised vision models."""

    def __init__(self):
        """Initializes the DINOv2 loader."""
        self.model_mappings = {
            "DINOV2-BASE": "facebook/dinov2-base",
            "DINOV2-LARGE": "facebook/dinov2-large",
            "DINOV2-GIANT": "facebook/dinov2-giant",
            "DINOV2-SUBLARGE": "facebook/dinov2-large",
        }

        self.processor = None

        # Static flag
        self.static = True

    def preprocess_fn_(self, input_data, fps=None):
        """
        Preprocesses input data for DINOv2.

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
            img = np.stack([np.array(i) for i in img])
        else:
            raise ValueError(
                "Input must be a PIL Image, file path, or numpy array")
        import time
        t = time.time()
        r = self.processor(images=img, return_tensors="pt").pixel_values
        print(time.time() - t)
        return self.processor(images=img, return_tensors="pt").pixel_values

    def _to_numpy_rgb_uint8(self, x):
        if isinstance(x, np.ndarray):
            arr = x
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if arr.ndim == 2:  # gray → rgb
                arr = np.stack([arr]*3, axis=-1)
            return arr
        elif isinstance(x, Image.Image):
            return np.asarray(x.convert("RGB"))
        else:
            raise ValueError("Frame must be np.ndarray or PIL.Image")

    def preprocess_fn(self, frames, device="cpu"):
        """
        frames: list of PIL.Image or HxWx3 uint8 numpy arrays
        returns: torch.Tensor of shape (B, 3, 224, 224), normalized
        """
        if self.processor is None:
            raise RuntimeError(
                "Call get_model(...) first so we can read mean/std.")

        # 1) Normalize inputs → one batched uint8 tensor on CPU
        if not isinstance(frames, list):
            frames = [frames]
        np_frames = [self._to_numpy_rgb_uint8(
            f) for f in frames]  # list of HWC uint8
        x = torch.from_numpy(np.stack(np_frames, axis=0))  # (B, H, W, C) uint8
        x = x.permute(0, 3, 1, 2).contiguous()

        # 2) Build a v2 transform pipeline (vectorized, no Python loop)
        mean = self.processor.image_mean   # e.g., [0.485, 0.456, 0.406]
        std = self.processor.image_std    # e.g., [0.229, 0.224, 0.225]

        transform = T.Compose([
            # 0..1
            T.ConvertImageDtype(torch.float32),
            T.Resize(size=256, interpolation=InterpolationMode.BICUBIC,
                     antialias=True),
            T.CenterCrop(224),
            T.Normalize(mean=mean, std=std),
        ])

        # 3) Optionally move to GPU *before* the heavy ops
        # x = x.to("cuda", non_blocking=True)

        x = transform(x)  # (B, 3, 224, 224)
        return x

    def get_model(self, identifier):
        """
        Loads a DINOv2 model based on the identifier.

        Args:
            identifier (str): Identifier for the DINOv2 variant.

        Returns:
            model: The loaded DINOv2 model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                self.processor = AutoImageProcessor.from_pretrained(
                    model_name, use_fast=True)
                model = AutoModel.from_pretrained(
                    model_name, torch_dtype=torch.float16)
                model = torch.compile(model)
                return model  # Return the processor, not a string
        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses DINOv2 model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from DINOv2 model
                as a numpy array.

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1).
        """
        batch_size, T = features_np.shape[0], features_np.shape[1]
        flattened_features = features_np.reshape(batch_size, T, -1)

        return flattened_features
