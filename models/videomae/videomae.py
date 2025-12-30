import os

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, VideoMAEModel, VideoMAEImageProcessor


class VideoMAE:
    """Loads pre-trained VideoMAE (Masked Autoencoder) models."""

    def __init__(self):
        """Initializes the VideoMAE loader."""
        self.model_mappings = {
            "VIDEOMAE-BASE": "MCG-NJU/videomae-base",
            "VIDEOMAE-LARGE": "MCG-NJU/videomae-large",
        }

        # VideoMAE normalization values
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Default parameters
        self.required_frames = 16  # Default for VideoMAE
        self.image_size = 224
        self.fps = 6.25
        # Current processor
        self.processor = None

        # Static flag
        self.static = False

    def preprocess_fn(self, input_data, orig_fps=None):
        """
        Fast preprocessing + still call VideoMAE processor at the end.

        Args:
            input_data: list of PIL.Image or np.ndarray, or a single PIL.Image.
            orig_fps: if provided, resample from this fps to self.fps.
        Returns:
            Tensor of shape (T, C, H, W) ready for your model.
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first.")

        # 1) Normalize to list of np.ndarray in HWC uint8
        if isinstance(input_data, Image.Image):
            frames = [np.array(input_data.convert("RGB"), dtype=np.uint8)]
        elif isinstance(input_data, list):
            frames = []
            for f in input_data:
                if isinstance(f, Image.Image):
                    frames.append(np.array(f.convert("RGB"), dtype=np.uint8))
                elif isinstance(f, np.ndarray):
                    # assume already HWC uint8
                    frames.append(f)
                else:
                    raise ValueError(
                        "List elements must be PIL.Image or np.ndarray")
        else:
            raise ValueError("Input must be a PIL.Image or list of frames")

        # 2) Temporal resampling if needed
        if orig_fps is not None and orig_fps != self.fps:
            L = len(frames)
            new_len = max(int(round(L / orig_fps * self.fps)), 1)
            idx = np.linspace(0, L - 1, new_len, dtype=int)
            frames = [frames[i] for i in idx]

        # 3) Sample or pad to exactly required_frames
        L = len(frames)
        if L > self.required_frames:
            idx = np.linspace(0, L - 1, self.required_frames, dtype=int)
            frames = [frames[i] for i in idx]
        elif L < self.required_frames:
            frames += [frames[-1]] * (self.required_frames - L)

        # 4) Vectorized resize with OpenCV
        T = self.required_frames
        H, W = self.image_size, self.image_size
        resized = []
        for f in frames:
            # ensure HWC uint8
            if not isinstance(f, np.ndarray):
                f = np.array(f.convert("RGB"), dtype=np.uint8)
            resized.append(cv2.resize(
                f, (W, H), interpolation=cv2.INTER_LINEAR))

        # 5) Now hand off to the HF processor exactly as before:
        # It accepts a list of HWC uint8 arrays (or PIL images) and
        # will do its own to-tensor + normalization + masking logic.
        inputs = self.processor(
            images=resized,
            return_tensors="pt"
        ).pixel_values  # shape (1, T, C, H, W)
        # 6) remove batch dim if you want just (T, C, H, W)
        return inputs.squeeze(0).half()

    def get_model(self, identifier):
        """
        Loads a VideoMAE model based on the identifier.

        Args:
            identifier (str): Identifier for the VideoMAE variant.

        Returns:
            model: The loaded VideoMAE model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                self.processor = VideoMAEImageProcessor.from_pretrained(
                    model_name, use_fast=True)
                model = VideoMAEModel.from_pretrained(
                    model_name, attn_implementation="sdpa", torch_dtype=torch.float16)

                # Update required frame count if specified in config
                if hasattr(model.config, "num_frames"):
                    self.required_frames = model.config.num_frames

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
        batch_size = features_np.shape[0]
        flattened_features = features_np.reshape(
            batch_size, self.required_frames, -1)

        return flattened_features
