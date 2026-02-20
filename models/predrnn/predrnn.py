import os
import numpy as np
from PIL import Image

import importlib.util
import torch
import requests
import functools

from torchvision import transforms
from tqdm import tqdm

import openstl
from openstl.methods import method_maps
from openstl.utils import reshape_patch
from sklearn.datasets import get_data_home

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

IMAGE_SIZES = (128, 160)


class PREDRNN:
    """Loads pre-trained video models (vision encoders only, by default)."""

    def __init__(self):
        """Initializes the video model loader."""
        self.model_mappings = {
            "PREDRNN": {"model": "https://github.com/chengtan9907/OpenSTL/releases/download/kitti-weights/kitticaltech_predrnn_one_ep100.pth",
                        "config": "models/predrnn/predrnn_config.py"
                        }
        }

        # Current processor
        self.processor = transforms.Resize(IMAGE_SIZES)
        self.static = False
        self.fps = 10

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for RESNET.

        Args:
            input_data: PIL Image, file path (str), or numpy array.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
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
                # frame_img = Image.fromarray(frame)

                frames.append(transforms.ToTensor()(frame))

            cap.release()

        elif isinstance(input_data, list) or isinstance(input_data, Image.Image):
            if isinstance(input_data, Image.Image):
                input_data = [input_data]
            frames = [transforms.ToTensor()(i) for i in input_data]

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

        # frames = torch.Tensor(video.to_numpy() / 255.0).permute(0, 3, 1, 2)
        frames = torch.stack(frames)
        frames = self.processor(frames)
        frames = frames.permute(0, 2, 3, 1)[None, :]  # BTHWC
        patch_size = 2

        batch_size, seq_length, img_height, img_width, num_channels = frames.shape
        a = frames.reshape(batch_size, seq_length,
                           img_height//patch_size, patch_size,
                           img_width//patch_size, patch_size,
                           num_channels)
        b = a.transpose(3, 4)
        patches = b.reshape(batch_size, seq_length,
                            img_height//patch_size,
                            img_width//patch_size,
                            patch_size*patch_size*num_channels)[0]
        return patches.half()

    def _get_config(self, identifier, config_path):
        spec = importlib.util.spec_from_file_location(identifier, config_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        config = mod.__dict__
        config = {k: v for k, v in config.items() if not k.startswith("__")}
        return config

    def get_model(self, identifier):
        if identifier not in self.model_mappings:
            raise ValueError(f"Unknown model: {identifier}")

        config = self._get_config(
            identifier, self.model_mappings[identifier]['config'])
        config["method"] = config["method"].lower()
        config['dataname'] = "kitticaltech"
        config['dataname'] = "kitticaltech"
        # not in use, just to initialize the model
        config['metrics'] = ['mse', 'mae']
        config['in_shape'] = [None, 3, *IMAGE_SIZES]

        m = method_maps[config["method"]](**config).model
        m.forward = functools.partial(m.forward, return_loss=False)
        self.model = m

        weights_dir = os.path.join(
            get_data_home(), 'weights', self.__class__.__name__)
        os.makedirs(weights_dir, exist_ok=True)

        # Determine local file name from the URL.
        file_name = self.model_mappings[identifier]['model'].split('/')[-1]
        weight_path = os.path.join(weights_dir, file_name)

        # Streaming, so we can iterate over the response.
        if not os.path.exists(weight_path):
            response = requests.get(
                self.model_mappings[identifier]['model'], stream=True)

            # Sizes in bytes.
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
        self.model = self.model.half()
        self.model = torch.compile(self.model)
        self.model.eval()
        return self.model

    def postprocess_fn(self, features_np):
        """Postprocesses model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from model as numpy array.

        Returns:
            np.ndarray: Flattened feature tensor.
        """
        batch_size, T = features_np.shape[0], features_np.shape[1]
        flattened_features = features_np.reshape(batch_size, T, -1)
        return flattened_features
