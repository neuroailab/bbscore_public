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

from .dfm_model import GroupNormalize, DFM_MODEL, DFM_LSTM_MODEL

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class DFM:
    """Loads pre-trained DFM self-supervised vision models."""

    def __init__(self, static=True):
        """Initializes the DFM loader."""
        self.model_mappings = {
            "DFM": "gs://stanford_neuroai_models/dfm/dfm.pt",
            "DFM_LSTM": "gs://stanford_neuroai_models/dfm/dfm_lstm.pt",
        }
        # DFM normalization values
        self.static = static
        self.size = (128, 128)
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.fps = 1000 / 60

    def preprocess_fn(self, input_data, fps=None):
        """
        1) Normalize to RGB PIL list
        2) (opt) FPS downsample
        3) Stack → [T,H,W,3] uint8 numpy
        4) To tensor [T,3,H,W], float [0..1]
        5) Resize shorter edge→128 (batched)
        6) Normalize with mean=0.5, std=0.5
        """
        # 1) PIL list
        if isinstance(input_data, Image.Image):
            frames_pil = [input_data.convert("RGB")]
        elif isinstance(input_data, list):
            frames_pil = []
            for img in input_data:
                if not isinstance(img, Image.Image):
                    raise ValueError("All elements must be PIL.Image")
                frames_pil.append(img.convert("RGB"))
        else:
            raise ValueError("Input must be a PIL.Image or list of PIL.Image.")

        T_orig = len(frames_pil)

        # 2) FPS downsampling
        if fps is not None:
            tgt = getattr(self, "fps", None)
            if tgt is not None and tgt != fps and T_orig > 1:
                duration = T_orig / float(fps)
                new_count = int(round(duration * float(tgt)))
                idxs = np.linspace(0, T_orig - 1, num=new_count, dtype=int)
                frames_pil = [frames_pil[i] for i in idxs]

        # 3) Stack to numpy [T,H,W,3]
        np_frames = np.stack([np.array(im, dtype=np.uint8)
                              for im in frames_pil], axis=0)

        # 4) To torch [T,3,H,W] float [0..1]
        arr = torch.from_numpy(np_frames)               # [T,H,W,3]
        arr = arr.permute(0, 3, 1, 2).float().div(255.)  # [T,3,H,W]

        # 5) Resize shorter edge == 128 (batched)
        arr = F.resize(
            arr,
            size=self.size,
            interpolation=Image.BILINEAR,
            antialias=True
        )

        # 6) Normalize exactly as GroupNormalize((.5,.5,.5),(.5,.5,.5))
        arr = F.normalize(arr, self.mean, self.std)

        return arr[::2].half()

    def get_model(self, identifier):
        """
        Loads a DFM model based on the identifier.

        Args:
            identifier (str): Identifier for the DFM variant.

        Returns:
            model: The loaded DFM model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_url in self.model_mappings.items():
            if identifier == prefix:
                # Define weights directory and ensure it exists.
                weights_dir = os.path.join(
                    get_data_home(), 'weights', self.__class__.__name__)
                os.makedirs(weights_dir, exist_ok=True)

                # Determine local file name from the URL.
                file_name = model_url.split('/')[-1]
                model_path = os.path.join(weights_dir, file_name)

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
                    else:
                        raise ValueError(f"Unsupported model URL: {model_url}")

                # Instantiate the model using the local weights file
                if self.static:
                    self.model = DFM_MODEL(model_path)
                else:
                    self.model = DFM_LSTM_MODEL(model_path)
                    print('Loading LSTM Weights')

                    # Load the checkpoint
                    ckpt = torch.load(model_path, map_location='cpu')

                    # Remove 'module.' prefix if present
                    new_ckpt = OrderedDict()
                    for k, v in ckpt.items():
                        new_key = k.replace('module.', '') if k.startswith(
                            'module.') else k
                        new_ckpt[new_key] = v

                    # Load the cleaned state_dict
                    self.model.load_state_dict(new_ckpt)
                    print('Loaded')

                self.model = torch.compile(self.model.half())
                return self.model

        raise ValueError(
            f"Unknown model identifier: {identifier}. Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses DFM model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from DFM model
                as a numpy array.

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1).
        """
        batch_size, T = features_np.shape[0], features_np.shape[1]
        flattened_features = features_np.reshape(batch_size, T, -1)

        return flattened_features


class DFM_LSTM(DFM):
    def __init__(
        self,
    ):
        super().__init__(static=False)
