import ssl
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F

# Standard ImageNet mean/std
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class ResNeXtWSL:
    """Loads Weakly Supervised Learning (WSL) ResNeXt models from Facebook Research."""

    def __init__(self):
        self.model_mappings = {
            "resnext101_32x8d_wsl": "resnext101_32x8d_wsl",
        }
        self.static = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocess input for ResNeXt WSL.

        Args:
            input_data: PIL.Image or list of PIL.Image.
            fps: Unused.

        Returns:
            torch.Tensor of shape [B, 3, 224, 224], float32, normalized.
        """
        # 1) Normalize to a list of RGB PILs
        if isinstance(input_data, Image.Image):
            frames_pil = [input_data.convert("RGB")]
        elif isinstance(input_data, list):
            frames_pil = []
            for img in input_data:
                if not isinstance(img, Image.Image):
                    raise ValueError("All elements must be PIL.Image")
                frames_pil.append(img.convert("RGB"))
        else:
            raise ValueError(
                "Input must be a PIL.Image or a list of PIL.Image.")

        # 2) Stack all PILs -> np array
        np_frames = np.stack(
            [np.array(img, dtype=np.uint8) for img in frames_pil], axis=0
        )

        # 3) Convert to torch.Tensor: [B, 3, H, W], float in [0..1]
        arr = torch.from_numpy(np_frames)
        arr = arr.permute(0, 3, 1, 2).float().div(255.0)

        # 4a) Resize (shorter side to 256)
        arr = F.resize(
            arr, size=256, interpolation=Image.BILINEAR, antialias=True)

        # 4b) Center-crop to 224
        arr = F.center_crop(arr, output_size=(224, 224))

        # 5) Normalize
        arr = F.normalize(arr, NORMALIZE_MEAN, NORMALIZE_STD)

        return arr.half()

    def get_model(self, identifier):
        """
        Loads the ResNeXt WSL model via Torch Hub.
        """
        if identifier in self.model_mappings:
            hub_model_name = self.model_mappings[identifier]
            print(f"Loading {identifier} via torch.hub...")

            # Handle SSL context for Hub downloads if certificates are missing
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            # Load from Facebook Research Hub
            self.model = torch.hub.load(
                'facebookresearch/WSL-Images', hub_model_name)

            self.model.eval()
            print('Loaded')
            self.model = torch.compile(self.model.half())
            return self.model

        raise ValueError(
            f"Unknown model identifier: {identifier}. Available: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features):
        """
        Postprocesses output by flattening.
        """
        B, T = features.shape[0], features.shape[1]
        return features.reshape(B, T, -1)
