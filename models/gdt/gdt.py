import os

import cv2
import numpy as np
import torch

from google.cloud import storage
from google.auth.credentials import AnonymousCredentials
from PIL import Image
from torchvision import transforms as T
from sklearn.datasets import get_data_home
from gdt_model.model import GDT
from gdt_model.video_transforms import clip_augmentation

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class GDTVideo:
    """Loads pre-trained GDT (Masked Autoencoder) models."""

    def __init__(self):
        """Initializes the GDT loader."""
        self.model_mappings = {
            "GDT-Kinetics400": "gs://stanford_neuroai_models/gdt/gdt_K400.pth",
            "GDT-IG65M": "gs://stanford_neuroai_models/gdt/gdt_IG65M.pth",
            "GDT-HowTo100M": "gs://stanford_neuroai_models/gdt/gdt_HT100M.pth",
        }

        # Default parameters
        self.processor = T.Compose([T.ToTensor()])
        self.fps = 30

        # Static flag
        self.static = False

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for GDT.

        Args:
            input_data: Video file path (str), list of frames, or tensor.

        Returns:
            dict: Preprocessed input for the model.

        Raises:
            ValueError: If the processor is not initialized or input is invalid.
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first.")

        frames = []

        # Handle different input types
        if isinstance(input_data, str) and os.path.isfile(input_data):
            # Load video from file path
            cap = cv2.VideoCapture(input_data)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frames = []

            # Read every frame from 0 to frame_count - 1
            for idx in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    # Break if there is an error or no frame is returned
                    break

                # Convert BGR (OpenCV’s default) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_img = Image.fromarray(frame)

                frames.append(self.processor(frame_img))

            cap.release()

        elif isinstance(input_data, list) or isinstance(input_data, Image.Image):
            if isinstance(input_data, Image.Image):
                input_data = [input_data]

            for frame in input_data:
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(np.uint8(frame)).convert("RGB")
                frames.append(self.processor(frame))

        else:  # Added Error for invalid input
            raise ValueError(
                "Input must be a video file path, list of frames, or tensor"
            )

        if fps is not None:
            target_fps = getattr(self, "fps", None)
            if target_fps is not None and target_fps != fps and len(frames) > 1:
                orig_count = len(frames)
                duration_sec = orig_count / float(fps)
                new_count = int(round(duration_sec * float(target_fps)))
                # pick evenly spaced frames
                indices = np.linspace(
                    0, orig_count - 1, num=new_count).astype(int)
                frames = [frames[i] for i in indices]

        frames = torch.stack(frames)
        return clip_augmentation(frames.permute(0, 2, 3, 1))

    def get_model(self, identifier):
        """
        Loads a GDT model based on the identifier.

        Args:
            identifier (str): Identifier for the GDT variant.

        Returns:
            model: The loaded GDT model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                # Define weights directory and ensure it exists.
                weights_dir = os.path.join(
                    get_data_home(), 'weights', self.__class__.__name__)
                os.makedirs(weights_dir, exist_ok=True)

                # Determine local file name from the URL.
                file_name = model_name.split('/')[-1]
                model_path = os.path.join(weights_dir, file_name)

                # Download the model weights if they are not already present locally.
                if not os.path.exists(model_path):
                    print(
                        f"Downloading model weights from {model_name} to {model_path} ...")
                    if model_name.startswith("gs://"):
                        # Parse the GCS URL to get the bucket name and blob name
                        parts = model_name[5:].split('/', 1)
                        if len(parts) != 2:
                            raise ValueError("Invalid GCS URL format.")
                        bucket_name, blob_name = parts
                        client = storage.Client(
                            credentials=AnonymousCredentials())
                        bucket = client.bucket(bucket_name)
                        blob = bucket.blob(blob_name)
                        blob.download_to_filename(model_path)
                    else:
                        raise ValueError(
                            f"Unsupported model URL: {model_name}")

                self.model = GDT(
                    vid_base_arch="r2plus1d_18",
                    aud_base_arch="resnet9",
                    pretrained=False,
                    norm_feat=False,
                    use_mlp=False,
                    num_classes=256,
                )
                self.model = self.model.video_network  # Remove audio network

                # Load weights
                state_dict_ = torch.load(
                    model_path, map_location="cpu", weights_only=False)['model']
                state_dict = {}
                for k, v in list(state_dict_.items()):
                    if k.startswith("video_network."):
                        k = k[len("video_network."):]
                        state_dict[k] = v
                self.model.load_state_dict(state_dict)
                self.model = torch.compile(self.model.half())
                return self.model
        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses GDT model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from GDT model
                as a numpy array. Expected shape:
                (batch_size, seq_len, feature_dim) or (seq_len, feature_dim)

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1),
                where N is batch size (or 1 if single sample).
        """
        batch_size, T = features_np.shape[0], features_np.shape[3]
        flattened_features = features_np.reshape(batch_size, T,  -1)
        return flattened_features
