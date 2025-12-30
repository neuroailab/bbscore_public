from .slowfast_model import SLOWFAST_MODEL

import os
import copy
import numpy as np
import torch
from PIL import Image
from sklearn.datasets import get_data_home

import mmengine
import mmaction
from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate

from mmaction.datasets.transforms import FormatShape, PackActionInputs
from mmengine.registry import TRANSFORMS

# Register the missing transforms
TRANSFORMS.register_module()(FormatShape)
TRANSFORMS.register_module()(PackActionInputs)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class SlowFast:
    """Loads pre-trained MAE self-supervised vision models."""

    def __init__(self):
        """Initializes the MAE loader."""
        self.model_mappings = {
            "SlowFast": {"config": "models/slowfast/helpers/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py",
                         "checkpoint": "https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth"
                         }
        }
        # MAE normalization values
        self.static = False
        self.fps = 12.5
        self.processor = None

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for MAE.

        Args:
            input_data: PIL Image, file path (str), or numpy array.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        frames = []

        # Handle different input types
        if isinstance(input_data, list) or isinstance(input_data, Image.Image):
            if isinstance(input_data, Image.Image):
                input_data = [np.uint8(input_data)]

            for frame in input_data:
                if isinstance(frame, np.ndarray):
                    frame = np.uint8(frame)
                frames.append(frame)

        else:
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

        # Ensure exactly 32 frames
        num_frames = len(frames)
        if num_frames < 32:
            # fallback blank frame
            last_frame = frames[-1]
            frames += [last_frame] * (32 - num_frames)
        elif num_frames > 32:
            indices = np.linspace(0, num_frames - 1, 32, dtype=int)
            frames = [frames[i] for i in indices]

        # 1) Make a copy of the original list‐of‐dicts
        proc_copy = copy.deepcopy(self.processor)

        # 2) Mutate the copy
        for i, pipeline in enumerate(proc_copy):
            if pipeline["type"] in ["ThreeCrop", "TenCrop"]:
                proc_copy[i] = {
                    "type": "CenterCrop",
                    "crop_size": pipeline["crop_size"],
                }
            if pipeline["type"] == "SampleFrames":
                proc_copy[i].update({"num_clips": 1, "frame_interval": 1})
            if pipeline["type"] == "Resize":
                # Suppose `frames` is already in scope
                h, w = np.uint8(frames[0]).shape[:2]
                aspect_ratio = w / h
                if h < w:
                    new_h = 256
                    new_w = int(w * 256 / h)
                else:
                    new_w = 256
                    new_h = int(h * 256 / w)

                proc_copy[i] = {
                    "type": "Resize",
                    "scale": (new_w, new_h),   # (width, height)
                    "keep_ratio": False        # force exactly (new_h × new_w)
                }

                proc_copy.insert(i + 1, {
                    "type": "CenterCrop",
                    "crop_size": (new_w, 256)
                })

        # 3) Now create your Compose from the copy:
        processor = Compose(proc_copy[3:])
        frames = np.stack(frames)
        data = {'imgs': frames, 'num_clips': 1,
                'modality': 'RGB', 'clip_len': 32}
        data = processor(data)
        return data

    def get_model(self, identifier):
        """
        Loads a MAE model based on the identifier.

        Args:
            identifier (str): Identifier for the MAE variant.

        Returns:
            model: The loaded MAE model.

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
                config_path = model_url['config']
                checkpoint = model_url['checkpoint']
                config = mmengine.Config.fromfile(config_path)

                test_pipeline_cfg = config.test_pipeline
                # SampleFrames: clip_len x frame_interval (sampling interval) x num_clips
                # change every ThreeCrop and TenCrop to CenterCrop
                self.processor = test_pipeline_cfg

                self.model = SLOWFAST_MODEL(config, checkpoint, device="cuda")
                print('Loaded')
                self.model = torch.compile(self.model)
                self.model.eval()
                return self.model

        raise ValueError(
            f"Unknown model identifier: {identifier}. Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses MAE model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from MAE model
                as a numpy array.

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1).
        """
        batch_size, T = features_np.shape[0], features_np.shape[1]
        flattened_features = features_np.reshape(batch_size, 32, -1)
        return flattened_features
