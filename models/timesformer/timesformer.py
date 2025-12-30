import os
from typing import List, Dict, Tuple, Optional, Union, Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    TimesformerModel,
)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class TimeSformer:
    """
    A class for loading pre-trained TimeSformer models for video understanding.
    """

    def __init__(self):
        """
        Initialize the TimeSformer loader and define a default preprocessing
        function.
        """
        self.model_mappings = {
            "TIMESFORMER-BASE": "facebook/timesformer-base-finetuned-k400",
            "TIMESFORMER-HR": "facebook/timesformer-hr-finetuned-k400",
            "TIMESFORMER-L": "facebook/timesformer-large-finetuned-k400",
        }

        # TimeSformer normalization values
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]

        # Default parameters
        self.required_frames = 8  # Standard for TimeSformer
        self.image_size = 224
        self.fps = 32

        # Current processor
        self.processor = None

        # Static flag
        self.static = False

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocess input data for TimeSformer models.

        Args:
            input_data: Can be a video file path, list of frames, or tensor.

        Returns:
            dict: Preprocessed input for the model.
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first.")

        frames = []

        # Handle different input types
        if isinstance(input_data, str) and os.path.isfile(input_data):
            # Load video from file path
            cap = cv2.VideoCapture(input_data)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample frames uniformly to match required frame count
            indices = np.linspace(
                0, frame_count - 1, self.required_frames, dtype=np.int32
            )

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_img = Image.fromarray(frame)

                    # Resize if needed
                    if (
                        frame_img.width != self.image_size
                        or frame_img.height != self.image_size
                    ):
                        frame_img = frame_img.resize(
                            (self.image_size, self.image_size))

                    frames.append(frame_img)
                else:
                    # If frame reading failed, create a black frame
                    frames.append(
                        Image.new("RGB", (self.image_size,
                                  self.image_size), color=0)
                    )

            cap.release()

        elif isinstance(input_data, list) or isinstance(input_data, Image.Image):
            if isinstance(input_data, Image.Image):
                input_data = [input_data]

            # List of frames (PIL Images or numpy arrays)
            input_length = len(input_data)

            # If input has more frames than required, sample frames
            if fps is not None:
                target_fps = getattr(self, "fps", None)
                if target_fps is not None and target_fps != fps and input_length > 1:
                    duration_sec = input_length / float(fps)
                    new_count = int(round(duration_sec * float(target_fps)))
                    # pick evenly spaced frames
                    indices = np.linspace(
                        0, input_length - 1, num=new_count).astype(int)
                    input_data = [input_data[i] for i in indices]

            input_length = len(input_data)
            if input_length > self.required_frames:
                indices = np.linspace(
                    0, input_length - 1, self.required_frames, dtype=np.int32
                )
                frames_to_process = [input_data[i] for i in indices]
            # If input has exactly the required number of frames, use all
            elif input_length == self.required_frames:
                frames_to_process = input_data
            # If input has fewer frames than required, repeat the last frame
            else:
                frames_to_process = input_data + [input_data[-1]] * (
                    self.required_frames - input_length
                )

            for frame in frames_to_process:
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(np.uint8(frame)).convert("RGB")

                # Resize if needed
                if frame.width != self.image_size or frame.height != self.image_size:
                    frame = frame.resize((self.image_size, self.image_size))

                frames.append(frame)
        else:
            raise ValueError(
                "Input must be a video file path or list of frames")

        # Ensure we have exactly the required number of frames
        assert (
            len(frames) == self.required_frames
        ), f"Expected {self.required_frames} frames, got {len(frames)}"

        # Process with TimeSformer processor
        return self.processor(images=frames, return_tensors="pt").pixel_values.squeeze(0).half()

    def get_model(self, identifier):
        """
        Load a TimeSformer model based on the identifier.

        Args:
            identifier (str): String identifier that starts with one of the
                supported TimeSformer variants.

        Returns:
            model: The loaded TimeSformer model.

        Raises:
            ValueError: If the identifier doesn't match any known model types.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                self.processor = AutoImageProcessor.from_pretrained(
                    model_name, use_fast=True)
                model = TimesformerModel.from_pretrained(
                    model_name, torch_dtype=torch.float16)

                # Update required frame count if specified in config
                if hasattr(model.config, "num_frames"):
                    self.required_frames = model.config.num_frames

                model = torch.compile(model)
                return model
        raise ValueError(
            f"Unknown model identifier: {identifier}. Available prefixes: "
            f"{', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses Timesformer model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from Timesformer model
                as a numpy array.

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1),
                where N is batch size (or 1 if single sample).
        """
        batch_size = features_np.shape[0]
        flattened_features = features_np.reshape(
            batch_size, self.required_frames, -1)

        return flattened_features
