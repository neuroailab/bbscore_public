import os
import torch
import numpy as np
from PIL import Image
# Use torchvision v2 for video support
import torchvision.transforms.v2 as T_v2
from sklearn.datasets import get_data_home
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
from google.auth import default
from collections import OrderedDict

# Import the CORnet architectures
from .cornet_z import CORnet_Z
from .cornet_r import CORnet_R
from .cornet_rt import CORnet_RT
from .cornet_s import CORnet_S


class CORNET:
    """
    Wrapper for loading pre-trained CORnet models into the BBScore framework.
    Handles weight fetching from Google Cloud Storage and uses torchvision v2
    to correctly process both single images and video frame sequences.
    """

    def __init__(self):
        """Initializes the CORnet loader."""
        self.gcs_bucket_name = "stanford_neuroai_models"

        self.model_mappings = {
            "CORNET-Z": {
                "class": CORnet_Z,
                "gcs_path": "cornet/cornet_z-5c427c9c.pth",
            },
            "CORNET-RT": {
                "class": CORnet_RT,
                "gcs_path": "cornet/cornet_rt-933c001c.pth",
            },
            "CORNET-S": {
                "class": CORnet_S,
                "gcs_path": "cornet/cornet_s-1d3f7974.pth",
            },
        }

        # Use torchvision.transforms.v2 for native tensor and video support.
        # This pipeline processes a list of images or a single image into a
        # normalized tensor ready for the model.
        self.processor = T_v2.Compose([
            T_v2.ToImage(),  # Converts PIL/NumPy to tensor, moves C to the first dim
            # Converts to float and scales to [0.0, 1.0]
            T_v2.ToDtype(torch.float32, scale=True),
            T_v2.Resize(256, antialias=True),
            T_v2.CenterCrop(224),
            T_v2.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
        ])

        self.static = True

    def _fetch_gcs_weights(self, gcs_path: str, local_path: str):
        """Downloads weights from GCS if they don't exist locally."""
        if os.path.exists(local_path):
            print(f"Weights already exist at {local_path}, skipping download.")
            return

        print(
            f"Downloading model weights from gs://{self.gcs_bucket_name}/{gcs_path} to {local_path}...")
        try:
            credentials, project = default()
            client = storage.Client(credentials=credentials)
        except DefaultCredentialsError:
            print(
                "Warning: Could not find default Google Cloud credentials. Trying anonymous access.")
            client = storage.Client.create_anonymous_client()

        try:
            bucket = client.bucket(self.gcs_bucket_name)
            blob = bucket.blob(gcs_path)
            blob.download_to_filename(local_path)
            print("Download complete.")
        except Exception as e:
            raise ConnectionError(
                f"Failed to download from GCS: {e}. Ensure permissions are correct or run 'gcloud auth application-default login'.")

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses a single image or a list of video frames for CORnet.
        Leverages torchvision.transforms.v2 for efficient processing.
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first.")

        # The v2 transform pipeline can directly handle a single PIL image,
        # a list of PIL images, or a numpy array (H, W, C).
        if isinstance(input_data, Image.Image):
            images_to_process = [input_data.convert("RGB")]
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            images_to_process = [Image.open(p).convert("RGB")
                                 for p in input_data]
        else:
            images_to_process = input_data  # Assumes PIL, list of PILs, or np.ndarray

        # The processor handles single images, lists of images, or numpy arrays.
        # For a list of T images, it returns a tensor of shape [T, C, H, W].
        processed_tensor = self.processor(images_to_process)
        processed_tensor = torch.stack(processed_tensor)
        if processed_tensor.ndim == 3:
            processed_tensor = processed_tensor.unsqueeze(0)
        return processed_tensor

    def get_model(self, identifier):
        """Loads a CORnet model and its weights from GCS."""
        if identifier not in self.model_mappings:
            raise ValueError(
                f"Unknown model identifier: {identifier}. "
                f"Available identifiers: {', '.join(self.model_mappings.keys())}"
            )

        model_info = self.model_mappings[identifier]
        model_class = model_info["class"]
        gcs_path = model_info["gcs_path"]

        weights_dir = os.path.join(
            get_data_home(), 'weights', self.__class__.__name__)
        os.makedirs(weights_dir, exist_ok=True)
        local_weights_path = os.path.join(
            weights_dir, os.path.basename(gcs_path))

        self._fetch_gcs_weights(gcs_path, local_weights_path)

        model = model_class()

        checkpoint = torch.load(
            local_weights_path, map_location='cpu', weights_only=False)

        state_dict = checkpoint.get('state_dict', checkpoint)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

        return model

    def postprocess_fn(self, features_np):
        """Flattens the output of the model for benchmarking."""
        batch_size, t = features_np.shape[0], features_np.shape[1]
        return features_np.reshape(batch_size, t, -1)
