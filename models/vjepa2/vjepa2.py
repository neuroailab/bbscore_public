import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class VJEPA2:
    """Loads pre-trained V-JEPA 2 video model and processor."""

    def __init__(self):
        """Initializes the VJEPA2 loader."""
        # Mapping from prefix to (HF repo, frames-per-clip)
        self.model_mappings = {
            "VJEPA2-VITL-FPC64-256": ("facebook/vjepa2-vitl-fpc64-256", 64),
            "VJEPA2-VITH-FPC64-256": ("facebook/vjepa2-vith-fpc64-256", 64),
            "VJEPA2-VITG-FPC64-256": ("facebook/vjepa2-vitg-fpc64-256", 64),
            "VJEPA2-VITL-FPC32-256-DIVE": ("facebook/vjepa2-vitl-fpc32-256-diving48", 32),
            "VJEPA2-VITL-FPC16-256-SSV2": ("facebook/vjepa2-vitl-fpc16-256-ssv2", 16)
        }
        self.processor = None
        self.frame_count = None
        self.static = False

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses image or video-frame data for V-JEPA2.

        Args:
            input_data: PIL Image, file path (str), numpy array, or list of those (for video).
            fps:        Unused here—frames must already be extracted.

        Returns:
            torch.Tensor:
              - Single image → shape [1, frame_count, C, H, W]
              - Video frames list → shape [1, T, C, H, W]

        Raises:
            ValueError: If processor not initialized or input type unsupported.
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first.")

        # Single image: repeat to frame_count
        if isinstance(input_data, (str, Image.Image, np.ndarray)):
            if isinstance(input_data, str) and os.path.isfile(input_data):
                img = Image.open(input_data).convert("RGB")
            elif isinstance(input_data, np.ndarray):
                img = Image.fromarray(np.uint8(input_data)).convert("RGB")
            else:
                img = input_data.convert("RGB")

            outputs = self.processor(img, return_tensors="pt")
            pv = outputs["pixel_values_videos"]
            # pv = pv.repeat(1, 8, 1, 1, 1)
            return pv.squeeze(0).half()

        # Video: list of frames
        elif isinstance(input_data, list):
            # Build a numpy array of shape (T, C, H, W)
            if all(isinstance(f, np.ndarray) for f in input_data):
                video_array = np.stack(input_data, axis=0)
            else:
                video_array = np.stack([
                    np.array(f).transpose(2, 0, 1) if not isinstance(
                        f, np.ndarray) else f
                    for f in input_data
                ], axis=0)

            outputs = self.processor(video_array, return_tensors="pt")
            return outputs["pixel_values_videos"].squeeze(0).half()  # [::2]

        else:
            raise ValueError("Unsupported input type for preprocessing.")

    def get_model(self, identifier, vision_only=True):
        """
        Loads a V-JEPA2 model and its processor.

        Args:
            identifier (str): Prefix e.g. "VJEPA2-VITL-FPC64-256".
            vision_only (bool): Ignored; always loads the vision encoder.

        Returns:
            torch.nn.Module: The V-JEPA2 video model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, (repo, frames) in self.model_mappings.items():
            if identifier == prefix:
                self.frame_count = frames  # number of frames per clip
                # , torch_dtype=torch.float16)
                model = AutoModel.from_pretrained(repo)
                self.processor = AutoVideoProcessor.from_pretrained(repo)
                # model = torch.compile(model.half())
                return model.half()

        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np: np.ndarray):
        arr = features_np

        # Collapse any singleton dims first (same behavior as before)
        while arr.ndim > 3 and 1 in arr.shape:
            arr = arr.squeeze(1)

        if arr.ndim == 2:
            # [N_tot, D] where N_tot = 256 * K
            N_tot, D = arr.shape
            if N_tot % 256 != 0:
                raise ValueError(
                    f"Expected N_tot multiple of 256, got {N_tot}")
            K = N_tot // 256
            # [K, 256, D] -> [K, 256*D]
            arr = arr.reshape(K, 256, D)
            arr = arr.reshape(K, 256 * D)
            return arr

        elif arr.ndim == 3:
            # [B, N_tot, D] where N_tot = 256 * K
            B, N_tot, D = arr.shape
            if N_tot % 256 != 0:
                raise ValueError(
                    f"Expected N_tot multiple of 256, got {N_tot}")
            K = N_tot // 256
            # [B, K, 256, D] -> [B, K, 256*D]
            arr = arr.reshape(B, K, 256, D)
            arr = arr.reshape(B, K, 256 * D)
            return arr
