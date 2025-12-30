import os

import numpy as np
import torch

from collections import OrderedDict
from google.cloud import storage
from google.auth.credentials import AnonymousCredentials
from PIL import Image
from sklearn.datasets import get_data_home
from torchvision import transforms as T
import torchvision.transforms.functional as F

from .pixelnerf_model import GroupNormalize, PIXELNERF_LSTM_MODEL, load_model

NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class PIXELNERF_LSTM:
    """Loads pre-trained PIXELNERF self-supervised vision models."""

    def __init__(self):
        """Initializes the PIXELNERF loader."""
        self.model_mappings = {
            "PIXELNERF_LSTM": {"model": "gs://stanford_neuroai_models/pixelnerf/pixelnerf_lstm.pt",
                               "config": "gs://stanford_neuroai_models/pixelnerf/merged_conf.conf"}
        }
        # PIXELNERF normalization values
        self.static = False
        self.fps = 1000/60
        self.processor = T.Compose([T.Resize(64),
                                    T.ToTensor(),
                                    GroupNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocess a single PIL.Image or a list of PIL.Image using only torchvision,
        but applied in one shot to the entire [T,3,H,W] tensor—no per-frame loop.

        Args:
            input_data: PIL.Image or list of PIL.Image (must be RGB or convertible).
            fps:        (optional) original FPS of those frames. If self.fps exists
                        and differs, we downsample here.

        Returns:
            torch.Tensor of shape [T, 3, 224, 224], float32, normalized.
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
                indices = np.linspace(0, T_orig - 1, num=new_count, dtype=int)
                frames_pil = [frames_pil[i] for i in indices]

        T_frames = len(frames_pil)

        # 3) Stack all PILs → one big [T, H, W, 3] uint8 array
        np_frames = np.stack(
            [np.array(img, dtype=np.uint8) for img in frames_pil], axis=0
        )  # shape = [T, H, W, 3], dtype=uint8

        # 4) Convert to torch.Tensor: [T, 3, H, W], float in [0..1]
        # [T, H, W, 3], uint8
        arr = torch.from_numpy(np_frames)
        arr = arr.permute(0, 3, 1, 2).float().div(
            255.0)    # [T, 3, H, W], float32

        # 5a) Resize so that shorter side = 256 (preserving aspect ratio).
        arr = F.resize(
            arr, size=256, interpolation=Image.BILINEAR, antialias=True)
        # Now arr.shape == [T, 3, H₁, W₁], with min(H₁,W₁)=256.

        # 5b) Center-crop each frame to 224×224
        arr = F.center_crop(arr, output_size=(224, 224))

        # 6) **normalize** (channel‐wise)
        arr = F.normalize(arr, NORMALIZE_MEAN, NORMALIZE_STD)

        return arr[::2].half()

    def get_model(self, identifier):
        """
        Loads a PIXELNERF model based on the identifier.

        Args:
            identifier (str): Identifier for the PIXELNERF variant.

        Returns:
            model: The loaded PIXELNERF model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, _url in self.model_mappings.items():
            if identifier.startswith(prefix):
                # Define weights directory and ensure it exists.
                model_url, config_url = _url['model'], _url['config']
                weights_dir = os.path.join(
                    get_data_home(), 'weights', self.__class__.__name__)
                os.makedirs(weights_dir, exist_ok=True)

                # Determine local file name from the URL.
                model_file_name = model_url.split('/')[-1]
                model_path = os.path.join(weights_dir, model_file_name)

                config_file_name = config_url.split('/')[-1]
                config_path = os.path.join(weights_dir, config_file_name)

                # Download the model weights if they are not already present locally.
                if not os.path.exists(model_path):
                    print(
                        f"Downloading model weights from {model_url} to {model_path} ...")
                    if model_url.startswith("gs://"):
                        # Parse the GCS URL to get the bucket name and blob name
                        parts = model_url[5:].split('/', 1)
                        if len(parts) != 2:
                            raise ValueError("Invalid GCS URL format.")
                        bucket_name, blob_name = parts
                        client = storage.Client(
                            credentials=AnonymousCredentials())
                        bucket = client.bucket(bucket_name)
                        blob = bucket.blob(blob_name)
                        blob.download_to_filename(model_path)
                        # Parse the GCS URL to get the bucket name and blob name
                        print(config_path)
                        blob.download_to_filename(config_path)
                    else:
                        raise ValueError(f"Unsupported model URL: {model_url}")

                # Instantiate the model using the local weights file
                self.model = PIXELNERF_LSTM_MODEL()
                print('Loading LSTM Weights')
                self.model = load_model(self.model, model_path)
                print('Loaded')
                self.model = torch.compile(self.model.half())
                return self.model

        raise ValueError(
            f"Unknown model identifier: {identifier}. Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses PIXELNERF model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from PIXELNERF model
                as a numpy array.

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1).
        """
        batch_size, T = features_np.shape[0], features_np.shape[1]
        flattened_features = features_np.reshape(batch_size, T, -1)
        return flattened_features
