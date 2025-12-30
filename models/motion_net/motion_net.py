import os
import pickle
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from sklearn.datasets import get_data_home
from google.cloud import storage
from google.auth.credentials import AnonymousCredentials


class MotionNet(nn.Module):
    """
    PyTorch implementation of Rideaux & Welchman (2020).
    Original source: https://github.com/patrickmineault/your-head-is-there-to-move-you-around
    """

    def __init__(self, ckpt_path):
        super().__init__()

        # load from pickle (previously h5py)
        with open(ckpt_path, 'rb') as f:
            results = pickle.load(f)

        self.conv1 = nn.Conv3d(1,
                               128,
                               (6, 6, 6),
                               (1, 1, 1),
                               padding=(3, 3, 3)
                               )

        # Transpose weights to match PyTorch (Output, Input, D, H, W)
        self.conv1.weight.data = torch.tensor(
            results['wconv'].transpose((3, 2, 1, 0))
        ).unsqueeze(1)

        self.conv1.bias.data = torch.tensor(results['bconv'])

        self.relu = nn.ReLU()

        # Conv2
        self.conv2 = nn.Conv3d(128,
                               64,
                               (1, 27, 27),
                               (1, 9, 9),
                               padding=(0, 17, 17))

        self.conv2.weight.data = torch.tensor(
            results['wout'].reshape(27, 27, 128, 64).transpose((3, 2, 1, 0))
        ).unsqueeze(2)

        self.conv2.bias.data = torch.tensor(results['bout'])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        # Expects X: (B, 3, T, H, W) -> converts to grayscale by mean over channels
        # Result: (B, 1, T, H, W)
        X = X.mean(axis=1, keepdims=True)

        X = self.conv1(X)
        X = X[:, :, :-1, :, :]  # Trim temporal padding artifact if necessary
        X = self.relu(X)
        X = self.conv2(X)
        X = self.softmax(X)

        return X


class MotionNetLoader:
    """
    Loader for MotionNet compatible with TemplateModel.
    """

    def __init__(self):
        self.static = False
        self.model_url = "gs://stanford_neuroai_models/motion_net/motionnet.pkl"
        self.img_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989])
        ])

    def _download_weights(self):
        """Downloads weights to a local cache using GCS."""
        weights_dir = os.path.join(get_data_home(), 'weights', 'motion_net')
        os.makedirs(weights_dir, exist_ok=True)

        filename = self.model_url.split('/')[-1]
        dest_path = os.path.join(weights_dir, filename)

        if not os.path.exists(dest_path):
            print(f"Downloading {self.model_url} to {dest_path} ...")
            if self.model_url.startswith("gs://"):
                # Parse the GCS URL to get the bucket name and blob name
                parts = self.model_url[5:].split('/', 1)
                if len(parts) != 2:
                    raise ValueError("Invalid GCS URL format.")
                bucket_name, blob_name = parts

                # Download using Anonymous Credentials
                client = storage.Client(credentials=AnonymousCredentials())
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.download_to_filename(dest_path)
            else:
                raise ValueError(f"Unsupported model URL: {self.model_url}")

        return dest_path

    def preprocess_fn(self, input_data):
        """
        Preprocesses input into (1, C, T, H, W) tensor.
        """
        # 1. Convert inputs to list of PIL images
        if isinstance(input_data, Image.Image):
            frames = [input_data.convert("RGB")]
        elif isinstance(input_data, list):
            frames = [img.convert("RGB") for img in input_data]
        else:
            raise ValueError("Input must be PIL Image or list of PIL Images.")

        # 2. To Numpy [T, H, W, 3] -> [T, 3, H, W] -> Float Tensor [0-1]
        np_frames = np.stack([np.array(f) for f in frames])
        tensor_frames = torch.from_numpy(np_frames).float() / 255.0

        # [T, H, W, C] -> [T, C, H, W]
        tensor_frames = tensor_frames.permute(0, 3, 1, 2)

        # 3. Apply Resize and Normalize
        tensor_frames = self.img_transform(tensor_frames)

        # 4. Reshape to [B, C, T, H, W] (B=1)
        # Current shape: [T, C, H, W] -> Permute to [C, T, H, W]
        tensor_frames = tensor_frames.permute(1, 0, 2, 3)

        # Add Batch dimension
        return tensor_frames

    def get_model(self, identifier="MotionNet"):
        """
        Loads the MotionNet model.
        """
        if identifier != "MotionNet":
            raise ValueError(f"Unknown identifier: {identifier}")

        weight_path = self._download_weights()

        model = MotionNet(weight_path)

        return model

    def postprocess_fn(self, features_np):
        """
        Postprocesses output by flattening.
        """
        # features_np shape likely (B, C, T, H, W) or (B, Channels, T, H, W)
        B, T = features_np.shape[0], features_np.shape[3]
        # Flatten everything except batch
        return features_np.reshape(B, T, -1)
