import os
import numpy as np
import torch
import requests
from PIL import Image
import torchvision.transforms.functional as F
from sklearn.datasets import get_data_home

from s3dg_howto100m import S3D

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class S3DHowTo100M:
    """Loads the S3D model pre-trained on HowTo100M."""

    def __init__(self):
        """Initializes the S3D loader."""
        self.model_mappings = {
            "s3d-HowTo100M": {
                "model_url": "https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth",
                "dict_url": "https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy",
            }
        }
        self.fps = 24
        self.static = False

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocess input for S3D HowTo100M.

        Args:
            input_data: PIL.Image or list of PIL.Image.
            fps: Original FPS of frames.

        Returns:
            torch.Tensor of shape [3, T, 256, 256], float32.
            Note: No mean/std normalization is applied, only division by 255.
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

        T_orig = len(frames_pil)

        # 2) (Optional) FPS-based downsampling
        if fps is not None:
            target_fps = getattr(self, "fps", None)
            if target_fps is not None and target_fps != fps and T_orig > 1:
                duration_sec = T_orig / float(fps)
                new_count = int(round(duration_sec * float(target_fps)))
                if new_count > 0:
                    indices = np.linspace(
                        0, T_orig - 1, num=new_count, dtype=int)
                    frames_pil = [frames_pil[i] for i in indices]

        # ------------------------------------------------------------------
        # 2.5) Ensure minimum of 6 frames (Padding)
        # ------------------------------------------------------------------
        min_frames = 6
        current_count = len(frames_pil)

        if current_count < min_frames and current_count > 0:
            frames_to_add = min_frames - current_count
            last_frame = frames_pil[-1]
            # Duplicate the last frame and extend the list
            frames_pil.extend([last_frame] * frames_to_add)

        # ------------------------------------------------------------------
        # 2.6) Ensure EVEN number of frames for space_to_depth operation
        # ------------------------------------------------------------------
        current_count = len(frames_pil)
        if current_count % 2 != 0 and current_count > 0:
            # Duplicate the last frame to make it even
            frames_pil.append(frames_pil[-1])
        # ------------------------------------------------------------------

        # 3) Stack all PILs → one big [T, H, W, 3] uint8 array
        np_frames = np.stack(
            [np.array(img, dtype=np.uint8) for img in frames_pil], axis=0
        )

        # 4) Convert to torch.Tensor: [T, 3, H, W], float in [0..1]
        arr = torch.from_numpy(np_frames)
        arr = arr.permute(0, 3, 1, 2).float().div(255.0)

        # 5) Resize to (256, 256) - Squashes if aspect ratio differs
        # Note: Model was trained with simple Resize(256, 256), no crop.
        arr = F.resize(arr, size=(256, 256),
                       interpolation=Image.BILINEAR, antialias=True)

        # 6) Permute to [C, T, H, W] for 3D CNN input
        arr = arr.permute(1, 0, 2, 3)

        return arr.half()

    def _download_file(self, url, dest_path):
        """Helper to download a file from a URL to a destination path."""
        if not os.path.exists(dest_path):
            print(f"Downloading {url} to {dest_path} ...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        return dest_path

    def get_model(self, identifier):
        """
        Loads the S3D HowTo100M model.
        """
        if identifier in self.model_mappings:
            urls = self.model_mappings[identifier]

            # Define weights directory
            weights_dir = os.path.join(
                get_data_home(), 'weights', self.__class__.__name__)
            os.makedirs(weights_dir, exist_ok=True)

            # 1. Download Dictionary (Required for S3D init)
            dict_filename = urls["dict_url"].split('/')[-1]
            dict_path = os.path.join(weights_dir, dict_filename)
            self._download_file(urls["dict_url"], dict_path)

            # 2. Download Model Weights
            model_filename = urls["model_url"].split('/')[-1]
            model_path = os.path.join(weights_dir, model_filename)
            self._download_file(urls["model_url"], model_path)

            print(f"Loading {identifier} weights...")

            # Instantiate the model
            # S3D typically takes (dict_path, vocab_size) in constructor
            self.model = S3D(dict_path, 512)

            # Load state dict
            state_dict = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

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
        return features.reshape(features.shape[0], features.shape[3], -1)
