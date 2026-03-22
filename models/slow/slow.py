import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class Slow:
    """Loads pre-trained Slow (single-pathway) model from PyTorchVideo."""

    def __init__(self):
        """Initializes the Slow model loader."""
        self.model_mappings = {
            "SLOW-R50": "slow_r50",
        }

        # Slow model normalization values (PyTorchVideo standard)
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]

        # Default parameters
        self.required_frames = 8   # Slow pathway uses 8 frames
        self.image_size = 224
        self.fps = 30.0

        # Layer name mapping for reference
        self.layer_mapping = {
            "blocks.1": "early visual (V1-like)",
            "blocks.2": "mid visual (V2-like)",
            "blocks.3": "late visual (V4-like)",
            "blocks.4": "high visual (IT-like)",
        }

        # Static flag — False because this is a video model
        self.static = False

    def preprocess_fn(self, input_data, orig_fps=None):
        """
        Preprocesses video frames for the Slow model.

        Args:
            input_data: list of PIL.Image or np.ndarray, or a single PIL.Image.
            orig_fps: if provided, resample from this fps to self.fps.

        Returns:
            Tensor of shape (1, C, T, H, W) ready for the Slow model.
        """
        # 1) Normalize to list of np.ndarray in HWC uint8
        if isinstance(input_data, Image.Image):
            frames = [np.array(input_data.convert("RGB"), dtype=np.uint8)]
        elif isinstance(input_data, list):
            frames = []
            for f in input_data:
                if isinstance(f, Image.Image):
                    frames.append(np.array(f.convert("RGB"), dtype=np.uint8))
                elif isinstance(f, np.ndarray):
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

        # 4) Resize each frame to image_size x image_size
        H, W = self.image_size, self.image_size
        resized = []
        for f in frames:
            if not isinstance(f, np.ndarray):
                f = np.array(f.convert("RGB"), dtype=np.uint8)
            resized.append(cv2.resize(f, (W, H),
                                      interpolation=cv2.INTER_LINEAR))

        # 5) Stack into (T, H, W, C), convert to float, normalize
        frames_np = np.stack(resized, axis=0).astype(np.float32) / 255.0

        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        frames_np = (frames_np - mean) / std

        # 6) Convert to tensor (T, H, W, C) -> (C, T, H, W) -> (1, C, T, H, W)
        tensor = torch.from_numpy(frames_np)       # (T, H, W, C)
        tensor = tensor.permute(3, 0, 1, 2)        # (C, T, H, W)
        # tensor shape: (C, T, H, W) — BBScore handles batching

        return tensor

    def get_model(self, identifier):
        """
        Loads the Slow model from PyTorchVideo via torch.hub.

        Args:
            identifier (str): Identifier for the Slow variant (e.g. 'SLOW-R50').

        Returns:
            model: The loaded Slow model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, hub_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                model = torch.hub.load(
                    "facebookresearch/pytorchvideo",
                    hub_name,
                    pretrained=True
                )
                return model

        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses Slow model output by flattening features."""
        import numpy as np

        batch_size, T = features_np.shape[0], features_np.shape[1]
        flattened_features = features_np.reshape(batch_size, T, -1)
        return flattened_features
