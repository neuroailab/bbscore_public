import os
import numpy as np
from PIL import Image

import imp
import torch
import requests
import functools

from torchvision import transforms
from tqdm import tqdm

import openstl
from openstl.methods import method_maps
from sklearn.datasets import get_data_home

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

IMAGE_SIZES = (128, 160)


class TAU:
    """Loads pre-trained TAU video model (trained on KITTI)."""

    def __init__(self):
        """Initializes the video model loader."""
        self.model_mappings = {
            "TAU": {"model": "https://github.com/chengtan9907/OpenSTL/releases/download/kitti-weights/kitticaltech_tau_one_ep100.pth",
                    "config": "models/tau/tau_config.py"
                    }
        }

        # Current processor
        self.processor = transforms.Resize(IMAGE_SIZES)
        self.static = False
        self.fps = 10

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for SimVP.
        Strictly returns 10 frames.

        Args:
            input_data: PIL Image, file path (str), or numpy array.

        Returns:
            torch.Tensor: Preprocessed input tensor [T, C, H, W], where T=10.
        """
        frames = []

        # Handle different input types
        if isinstance(input_data, list) or isinstance(input_data, Image.Image):
            if isinstance(input_data, Image.Image):
                input_data = [input_data]
            frames = [transforms.ToTensor()(i) for i in input_data]

        elif isinstance(input_data, str) and os.path.isfile(input_data):
            try:
                import cv2
                cap = cv2.VideoCapture(input_data)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for idx in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(transforms.ToTensor()(frame))
                cap.release()
            except ImportError:
                raise ValueError(
                    "Input is a file path but opencv-python is not installed.")
        else:
            raise ValueError(
                "Input must be a video file path, list of frames, or tensor"
            )

        # Strictly sample 10 frames
        target_frames = 10
        total_frames = len(frames)

        if total_frames > 0:
            # Sample exactly 10 frames uniformly from the input
            indices = np.linspace(0, total_frames - 1,
                                  num=target_frames).astype(int)
            frames = [frames[i] for i in indices]

        # Stack frames: [T, C, H, W]
        frames = torch.stack(frames)

        # Resize: [T, C, H, W] -> [T, C, H_new, W_new]
        frames = self.processor(frames)

        return frames

    def _get_config(self, identifier, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        config = imp.load_source(identifier, config_path).__dict__
        config = {k: v for k, v in config.items() if not k.startswith("__")}
        return config

    def get_model(self, identifier):
        if identifier not in self.model_mappings:
            raise ValueError(f"Unknown model: {identifier}")

        config = self._get_config(
            identifier, self.model_mappings[identifier]['config'])

        config["method"] = config["method"]
        config['dataname'] = "kitticaltech"
        config['metrics'] = ['mse', 'mae']
        config['in_shape'] = [10, 3, *IMAGE_SIZES]

        m = method_maps[config["method"]](**config).model
        m.forward = functools.partial(m.forward, return_loss=False)
        self.model = m

        weights_dir = os.path.join(
            get_data_home(), 'weights', self.__class__.__name__)
        os.makedirs(weights_dir, exist_ok=True)

        file_name = self.model_mappings[identifier]['model'].split('/')[-1]
        weight_path = os.path.join(weights_dir, file_name)

        if not os.path.exists(weight_path):
            response = requests.get(
                self.model_mappings[identifier]['model'], stream=True)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024

            with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"{file_name}") as progress_bar:
                with open(weight_path, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)

            if total_size != 0 and progress_bar.n != total_size:
                raise RuntimeError("Could not download file")

        self.model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        self.model.eval()
        return self.model

    def postprocess_fn(self, features_np):
        """Postprocesses model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from model as numpy array.

        Returns:
            np.ndarray: Flattened feature tensor.
        """
        batch_size = features_np.shape[0]
        flattened_features = features_np.reshape(batch_size, 10, -1)
        return flattened_features
