import os
import numpy as np
import torch
import torch.nn as nn
import requests
from PIL import Image
import torchvision.transforms.functional as F
from sklearn.datasets import get_data_home
from tqdm import tqdm

from vonenet import VOneNet

# VOneNet-specific normalization (0.5, 0.5, 0.5)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class Wrapper(nn.Module):
    """Helper class to match state_dict keys containing 'module.' prefix."""

    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.module = model


class VOneResNet50NS:
    """Loads the VOneResNet-50 (Non-Stochastic) model."""

    def __init__(self):
        self.model_mappings = {
            "voneresnet-50-non_stochastic": "https://brainscore-storage.s3.amazonaws.com/brainscore-vision/models/voneresnet-50-non_stochastic/voneresnet50_ns_e70.pth.tar"
        }
        self.static = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocess input for VOneResNet.

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

        # 5) Normalize (Mean/Std = 0.5 for VOneNet)
        arr = F.normalize(arr, NORMALIZE_MEAN, NORMALIZE_STD)

        return arr.half()

    def get_model(self, identifier):
        """
        Loads the VOneResNet model.
        """
        if identifier in self.model_mappings:
            url = self.model_mappings[identifier]

            weights_dir = os.path.join(
                get_data_home(), 'weights', self.__class__.__name__)
            os.makedirs(weights_dir, exist_ok=True)

            file_name = url.split('/')[-1]
            weights_path = os.path.join(weights_dir, file_name)

            # Download if not exists
            if not os.path.exists(weights_path):
                print(f"Downloading {identifier} weights from {url}...")
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024

                with tqdm(total=total_size, unit="B", unit_scale=True, desc=file_name) as progress_bar:
                    with open(weights_path, "wb") as file:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            file.write(data)

            print(f"Loading {identifier} ...")

            # Load checkpoint data to get flags
            ckpt_data = torch.load(weights_path, map_location='cpu')

            # Extract initialization flags
            flags = ckpt_data['flags']
            stride = flags['stride']
            simple_channels = flags['simple_channels']
            complex_channels = flags['complex_channels']
            k_exc = flags['k_exc']
            noise_mode = flags['noise_mode']
            noise_scale = flags['noise_scale']
            noise_level = flags['noise_level']
            model_arch_id = flags['arch'].replace(
                '_', '').lower()  # e.g. resnet50

            # Instantiate VOneNet
            model = VOneNet(
                model_arch=model_arch_id,
                stride=stride,
                k_exc=k_exc,
                simple_channels=simple_channels,
                complex_channels=complex_channels,
                noise_mode=noise_mode,
                noise_scale=noise_scale,
                noise_level=noise_level
            )

            # Load State Dict
            # The checkpoint usually has keys starting with 'module.'
            # We wrap the model to match these keys, then unwrap.
            model = Wrapper(model)
            model.load_state_dict(ckpt_data['state_dict'])
            self.model = model.module

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
