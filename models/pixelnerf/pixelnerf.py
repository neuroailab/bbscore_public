import os

import cv2
import numpy as np
import torch

from google.cloud import storage
from google.auth.credentials import AnonymousCredentials
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import v2 as T
from sklearn.datasets import get_data_home
from pyhocon import ConfigFactory
from phys_extractors.models.pixelnerf.src.model import make_model

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class PixelNerf:
    """Loads pre-trained PixelNerf (Masked Autoencoder) models,
       with video-aware preprocessing via torchvision.transforms.v2."""

    def __init__(self):
        self.model_mappings = {
            "PN": {
                "model": "gs://stanford_neuroai_models/pixelnerf/pixel_nerf_latest",
                "config": "gs://stanford_neuroai_models/pixelnerf/merged_conf.conf"
            }
        }

        # v2 Compose will accept lists, tensors, PIL, numpy, etc.
        # any leading dims (frames) are preserved automatically :contentReference[oaicite:1]{index=1}
        self.processor = T.Compose([
            # PIL/ndarray → Tensor[..., C, H, W]
            T.ToImage(),
            T.ToDtype(torch.uint8, scale=True),    # ensure uint8 in [0,255]
            T.Resize(size=256, antialias=True),    # resize spatial dims
            T.CenterCrop(size=224),                # center‐crop spatial dims
            T.ToDtype(torch.float32, scale=True),  # scale to [0,1] float
            T.Normalize(                           # normalize per ImageNet stats
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # static vs dynamic for the model; unchanged
        self.static = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input for PixelNerf.

        Args:
          input_data:  str path to video file, PIL.Image, list of PIL/Image or numpy frames,
                      or a torch.Tensor of shape (T, C, H, W).

        Returns:
          Tensor of shape (T, C, 224, 224), dtype=float32.
        """
        # load video frames if given a path
        if isinstance(input_data, str):
            cap = cv2.VideoCapture(input_data)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # convert BGR→RGB PIL
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(img)
            cap.release()
            input_data = frames

        # wrap single-image inputs
        elif isinstance(input_data, Image.Image):
            input_data = [input_data]

        # convert raw numpy frames to PIL for consistency
        elif isinstance(input_data, list) and isinstance(input_data[0], np.ndarray):
            input_data = [
                Image.fromarray(np.uint8(f)).convert("RGB")
                for f in input_data
            ]

        # at this point, `input_data` is either:
        #  • list of PIL.Image,
        #  • torch.Tensor of shape (..., C, H, W)
        # Compose will recursively apply each transform and preserve list structure :contentReference[oaicite:2]{index=2}
        processed = self.processor(input_data[::2])

        # if we get a list back, stack into a video tensor
        if isinstance(processed, list):
            # each element is now a Tensor[C,224,224]
            return torch.stack(processed)  # → (T, C, H, W)

        # else it was already a tensor, e.g. you passed a torch.Tensor in
        return processed

    def get_model(self, identifier):
        """
        Loads a PixelNerf model based on the identifier.

        Args:
            identifier (str): Identifier for the PixelNerf variant.

        Returns:
            model: The loaded PixelNerf model.

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
                        blob.download_to_filename(config_path)
                    else:
                        raise ValueError(f"Unsupported model URL: {model_url}")

                current_dir = os.path.dirname(__file__)
                config_path = os.path.join(current_dir, 'merged_conf.conf')

                conf = ConfigFactory.parse_file(config_path)
                self.model = make_model(conf["model"])
                self.model.load_state_dict(
                    torch.load(model_path, map_location='cpu'))
                return self.model.encoder.half()
        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses PixelNerf model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from PixelNerf model
                as a numpy array. Expected shape:
                (batch_size, seq_len, feature_dim) or (seq_len, feature_dim)

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1),
                where N is batch size (or 1 if single sample).
        """
        batch_size, T = features_np.shape[0], features_np.shape[1]
        flattened_features = features_np.reshape(batch_size, T, -1)
        return flattened_features
