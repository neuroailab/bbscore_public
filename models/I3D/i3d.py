import os
import copy
import numpy as np

from PIL import Image
from sklearn.datasets import get_data_home

import mmengine
import mmaction
from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate

from .i3d_model import I3D_MODEL
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class I3D_NONLOCAL:
    """Loads pre-trained I3D self-supervised vision models."""

    def __init__(self):
        """Initializes the I3D loader."""
        self.model_mappings = {
            "I3D": {"config": "models/I3D/helpers/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb.py",
                    "checkpoint": "https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148.pth"
                    }
        }
        # I3D normalization values
        self.static = False
        self.fps = 12.5
        self.processor = None

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for I3D.

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
        if num_frames < 5:
            # fallback blank frame
            last_frame = frames[-1] if frames else Image.new("RGB", (256, 256))
            frames += [last_frame] * (5 - num_frames)

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

        processor = Compose(proc_copy[3:])

        frames = np.stack(frames)
        data = {'imgs': frames, 'num_clips': 1,
                'modality': 'RGB', 'clip_len': len(frames)}
        data = processor(data)
        return data

    def get_model(self, identifier):
        """
        Loads a I3D model based on the identifier.

        Args:
            identifier (str): Identifier for the I3D variant.

        Returns:
            model: The loaded I3D model.

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

                self.model = I3D_MODEL(config, checkpoint, device="cuda")
                print('Loaded')
                self.model = torch.compile(self.model.half())
                return self.model

        raise ValueError(
            f"Unknown model identifier: {identifier}. Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """Postprocesses I3D model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from I3D model
                as a numpy array.

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1).
        """
        batch_size, T = features_np.shape[0], features_np.shape[3]
        flattened_features = features_np.reshape(batch_size, T, -1)
        return flattened_features
