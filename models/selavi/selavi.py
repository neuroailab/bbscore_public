import os
import torch
import numpy as np
from torchvision import transforms as T
from torchvision.io import read_video
from torch.nn.functional import interpolate


import urllib.request
from sklearn.datasets import get_data_home
from PIL import Image

from selavi.model import load_model
from selavi.video_transforms import clip_augmentation

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class SeLaVi:
    """Loads pre-trained SeLaVi (Masked Autoencoder) models."""

    def __init__(self):
        self.model_mappings = {
            "SeLaVi-Kinetics400": "https://dl.fbaipublicfiles.com/selavi/selavi_kinetics.pth",
            "SeLaVi-Kinetics-Sound": "https://dl.fbaipublicfiles.com/selavi/selavi_kinetics_sound.pth",
            "SeLaVi-AVE": "https://dl.fbaipublicfiles.com/selavi/selavi_ave.pth",
            "SeLaVi-VGG-Sound": "https://dl.fbaipublicfiles.com/selavi/selavi_vgg_sound.pth"
        }
        # We’ll keep a simple “normalize to [0,1]” processor
        self.processor = lambda vid: vid.float().div(255)
        self.fps = 30
        self.static = False

    def preprocess_fn(self, input_data, fps=None):
        """
        Fast preprocessing: read video in one shot, convert to tensor,[C,H,W], resample frame‐rate if needed.
        """
        # 1) LOAD / CONVERT TO TORCH TENSOR [T,H,W,3]
        if isinstance(input_data, str) and os.path.isfile(input_data):
            # read_video uses ffmpeg under the hood
            video, _, info = read_video(input_data, pts_unit='sec')
            # video: torch.Tensor [T, H, W, 3], dtype=uint8
            orig_fps = info.get('video_fps', self.fps)
        elif isinstance(input_data, torch.Tensor):
            # assume already [T, C, H, W] or [T, H, W, C]
            vid = input_data
            if vid.dim() == 4 and vid.shape[1] in (1, 3):
                # [T,C,H,W] → [T,H,W,C]
                video = vid.permute(0, 2, 3, 1)
            else:
                video = vid
            orig_fps = fps or self.fps
        elif isinstance(input_data, (list, np.ndarray)) or isinstance(input_data, Image.Image):
            if isinstance(input_data, Image.Image):
                input_data = [input_data]
            # list of PIL or np.ndarray frames → stack once
            arrs = []
            for f in input_data:
                if hasattr(f, 'to_numpy') or isinstance(f, np.ndarray):
                    a = np.asarray(f)
                else:
                    a = np.asarray(f.convert('RGB'))
                arrs.append(a)
            video = torch.from_numpy(np.stack(arrs, axis=0))  # [T,H,W,3]
            orig_fps = fps or self.fps
        else:
            raise ValueError(
                "Unsupported input, must be file path, tensor, or list of frames")

        # 2) RESAMPLE FRAME RATE (if needed)
        target_fps = self.fps
        use_fps = fps or orig_fps
        if orig_fps != target_fps and video.shape[0] > 1:
            T0 = video.shape[0]
            T1 = int(round(T0 * target_fps / use_fps))
            # simple indexing; avoids interpolation kernels
            idx = torch.linspace(0, T0-1, steps=T1).long()
            video = video[idx]

        # 3) TO TENSOR [T, C, H, W] + NORMALIZE
        video = video.permute(0, 3, 1, 2)      # → [T,C,H,W]
        video = self.processor(video)          # → float in [0,1]

        # 4) FINAL CLIP AUGMENTATION (expects [T,H,W,C])
        return clip_augmentation(video.permute(0, 2, 3, 1))

    def get_model(self, identifier):
        """
        Loads a SeLaVi model based on the identifier.

        Args:
            identifier (str): Identifier for the SeLaVi variant.

        Returns:
            model: The loaded SeLaVi model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_url in self.model_mappings.items():
            if identifier.startswith(prefix):
                if "Kinetics400" in prefix:
                    num_classes = 400
                elif "Kinetics-Sound" in prefix:
                    num_classes = 32
                elif "VGG-Sound" in prefix:
                    num_classes = 309
                elif "AVE" in prefix:
                    num_classes = 28
                else:
                    raise ValueError(f"Unknown identifier")

                # Find the matching model URL

                # Set up the directory structure using sklearn's get_data_home
                weights_dir = os.path.join(
                    get_data_home(), 'weights', self.__class__.__name__)
                os.makedirs(weights_dir, exist_ok=True)

                # Extract filename from URL and create full path
                filename = os.path.basename(model_url)
                weights_path = os.path.join(weights_dir, filename)

                # Download the weights file if it doesn't exist
                if not os.path.isfile(weights_path):
                    print(
                        f"Downloading {identifier} weights from {model_url}...")
                    try:
                        urllib.request.urlretrieve(model_url, weights_path)
                        print(f"Downloaded weights to {weights_path}")
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to download weights from {model_url}: {str(e)}")

                self.model = load_model(
                    vid_base_arch="r2plus1d_18",
                    aud_base_arch="resnet9",
                    use_mlp=True,
                    num_classes=num_classes,
                    pretrained=False,
                    norm_feat=False,
                    use_max_pool=False,
                    headcount=10,
                )

                self.model = self.model.video_network  # Remove audio network

                # Load weights
                state_dict_ = torch.load(
                    weights_path, map_location="cpu")['model']
                state_dict = {}
                for k, v in list(state_dict_.items()):
                    if k.startswith("module.video_network."):
                        k = k[len("module.video_network."):]
                        state_dict[k] = v
                self.model.load_state_dict(state_dict)
                self.model = torch.compile(self.model)
                return self.model
        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses SeLaVi model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from SeLaVi model
                as a numpy array. Expected shape:
                (batch_size, seq_len, feature_dim) or (seq_len, feature_dim)

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1),
                where N is batch size (or 1 if single sample).
        """
        batch_size = features_np.shape[0]
        T = features_np.shape[3]
        flattened_features = features_np.reshape(batch_size, T, -1)
        return flattened_features
