"""
Timebin model registry for V-JEPA2 with "last 256 tokens" truncation.

- Defines 18 timebins: [3, 5, 7, ..., 37] frames (first is 3, then +2 each).
- Each model has a unique model_id like "VJEPA2-TB{idx:02d}-F{frames}-LAST256".
- Wraps VJEPA2 to (a) tell the processor how many frames to take, and (b) truncate to the last 256 tokens.
"""

from typing import List, Dict, Any, Optional
import numpy as np

try:
    # This is provided in your environment according to "__init__ (1).py"
    from models import MODEL_REGISTRY
except Exception:
    MODEL_REGISTRY = {}

# Import the base model class from the uploaded package
from .vjepa2 import VJEPA2

import os
import torch
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


# All timebins share this pretrained checkpoint
BASE_IDENTIFIER = "facebook/vjepa2-vitg-fpc64-256"

# Precompute the 18 timebins: first 3, then +2 up to 18 bins total
TIMEBINS: List[int] = [3] + [3 + 2*i for i in range(1, 18)]  # [3,5,7,...,37]


class TimebinVJEPA(VJEPA2):
    """
    Thin wrapper over VJEPA2 that enforces frame-count per timebin and
    truncates the token sequence to the last 256 tokens during post-processing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _to_frame_list(self, video, repeat: int):
        """
        Normalize `video` to a list of PIL.Image frames.
        If only one frame is provided, repeat it `repeat` times.
        Supported inputs:
          - PIL.Image.Image
          - numpy.ndarray with shape [H,W,C] or [C,H,W] or [T,H,W,C] or [T,C,H,W]
          - torch.Tensor with the same shapes as ndarray
          - list/tuple of PIL.Image.Image
        """
        try:
            from PIL import Image
            import numpy as _np
            import torch as _torch
        except Exception:
            Image = None
            _np = None
            _torch = None

        def to_pil(arr):
            if Image is None:
                raise RuntimeError("PIL not available to convert frames.")
            if isinstance(arr, Image.Image):
                return arr
            if _np is not None and isinstance(arr, _np.ndarray):
                # [C,H,W] -> [H,W,C]
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = arr.transpose(1, 2, 0)
                return Image.fromarray(arr.astype('uint8'))
            if _torch is not None and isinstance(arr, _torch.Tensor):
                a = arr.detach().cpu()
                if a.ndim == 3 and a.shape[0] in (1, 3):  # [C,H,W]
                    a = a.permute(1, 2, 0)
                return Image.fromarray(a.numpy().astype('uint8'))
            raise TypeError(f"Unsupported frame type: {type(arr)}")

        # Case 1: list/tuple of frames
        if isinstance(video, (list, tuple)):
            frames = list(video)
            if len(frames) == 0:
                raise ValueError("Empty video input.")
            # ensure PIL
            frames = [to_pil(f) for f in frames]
            if len(frames) == 1:
                frames = frames * int(repeat)
            return frames[:repeat]

        # Case 2: single PIL image
        if Image is not None and isinstance(video, Image.Image):
            return [video] * int(repeat)

        # Case 3: numpy or torch
        if (_np is not None and isinstance(video, _np.ndarray)) or (_torch is not None and isinstance(video, _torch.Tensor)):
            # Accept [H,W,C], [C,H,W], [T,H,W,C], [T,C,H,W]
            if hasattr(video, "detach"):
                video = video.detach().cpu().numpy()
            arr = video
            if arr.ndim == 3:
                # single frame
                return [to_pil(arr)] * int(repeat)
            if arr.ndim == 4:
                # has time dim
                if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                    # probably [C,H,W] mistaken; treat as single frame
                    return [to_pil(arr)] * int(repeat)
                # assume [T,H,W,C] or [T,C,H,W]
                T = arr.shape[0]
                frames = []
                for t in range(T):
                    frames.append(to_pil(arr[t]))
                if len(frames) == 1:
                    frames = frames * int(repeat)
                return frames[:repeat]
            raise ValueError(f"Unsupported array/tensor shape: {arr.shape}")

        # Fallback: try to wrap as PIL and repeat
        return [to_pil(video)] * int(repeat)

    # Truncate to the last 256 tokens. This assumes the model returns a feature tensor
    # of shape [B, T, D] or [B, N_tokens, D]. We take the last min(256, N_tokens).
    def postprocess_fn(self, features_np: np.ndarray):
        arr = features_np
        # collapse any singleton dims first
        while arr.ndim > 3 and 1 in arr.shape:
            arr = arr.squeeze()

        # Typical shapes: [B, N, D] or [N, D]
        if arr.ndim == 3:
            B, N, D = arr.shape
            n_take = min(256, N)
            arr = arr[:, -n_take:, :]
        elif arr.ndim == 2:
            N, D = arr.shape
            n_take = min(256, N)
            arr = arr[-n_take:, :]
        else:
            # Fallback to parent behavior if unexpected shape
            arr = super().postprocess_fn(arr)

        return arr

    # If your VJEPA2 exposes a preprocess function, you can override it.
    # Otherwise, many HF processors accept "num_frames" as an argument; we conditionally
    # inject that where applicable. The exact signature may vary by your processor.
    def preprocess_fn(self, video, **processor_kwargs):
        processor_kwargs = dict(processor_kwargs)
        frames = self._to_frame_list(video, repeat=self.tb_frames)
        outputs = self.processor(frames, **processor_kwargs)
        return outputs["pixel_values_videos"][0].half()

    def get_model(self, identifier, vision_only=True):
        """
        Loads a V-JEPA2 model and its processor.

        Args:
            identifier (str): Prefix e.g. "VJEPA2-VITG-FPC64-256".
            vision_only (bool): Ignored; always loads the vision encoder.

        Returns:
            torch.nn.Module: The V-JEPA2 video model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        self.identifier = identifier
        self.tb_frames = int(identifier.split('-')[-2][1:])
        model = AutoModel.from_pretrained(BASE_IDENTIFIER)
        self.processor = AutoVideoProcessor.from_pretrained(BASE_IDENTIFIER)
        model = torch.compile(model.half())
        return model


def register_timebin_models(
    base_key_prefix: str = "vjepa2_tb",
    model_id_prefix: str = "VJEPA2-TB",
    variant_hint: str = "",  # e.g., "VITL-FPC64" if you want to encode a backbone hint
) -> Dict[str, Dict[str, Any]]:
    """
    Registers 18 timebin-specific models into MODEL_REGISTRY.

    Returns the dict of new entries for convenience.
    """
    new_entries: Dict[str, Dict[str, Any]] = {}
    for idx, frames in enumerate(TIMEBINS, start=1):
        key = f"{base_key_prefix}{idx:02d}"  # e.g., vjepa2_tb01
        mid = f"{model_id_prefix}{idx:02d}-F{frames}-LAST256"
        if variant_hint:
            mid = f"{mid}-{variant_hint}"

        entry = {
            "class": TimebinVJEPA,
            "model_id_mapping": mid,  # always load vitg-fpc64-256
        }
        MODEL_REGISTRY[key] = entry
        new_entries[key] = entry
    return new_entries


# If this module is imported, auto-register by default.
AUTO_REGISTERED = register_timebin_models()
