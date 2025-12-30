import os
import subprocess
import urllib.request
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.datasets import get_data_home

# AVID specific imports
import avid_cma
from avid_cma.datasets import preprocessing
from avid_cma.utils import main_utils
from avid_cma.utils.logger import Logger

# Torch settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class AVID:
    """Loads pre-trained AVID models."""

    def __init__(self):
        """Initializes the AVID_CMA loader."""
        self.model_mappings = {
            "AVID-Kinetics400": {
                "model": (
                    "https://dl.fbaipublicfiles.com/avid-cma/checkpoints/"
                    "AVID_Kinetics_Cross-N1024_checkpoint.pth.tar"
                ),
                "config": (
                    "https://raw.githubusercontent.com/YingtianDt/AVID-CMA/"
                    "main/avid_cma/configs/main/avid/kinetics/Cross-N1024.yaml"
                ),
            },
            "AVID-CMA-Kinetics400": {
                "model": (
                    "https://dl.fbaipublicfiles.com/avid-cma/checkpoints/"
                    "AVID-CMA_Kinetics_InstX-N1024-PosW-N64-Top32_"
                    "checkpoint.pth.tar"
                ),
                "config": (
                    "https://raw.githubusercontent.com/YingtianDt/AVID-CMA/"
                    "main/avid_cma/configs/main/avid-cma/kinetics/"
                    "InstX-N1024-PosW-N64-Top32.yaml"
                ),
            },
            "AVID-Audioset": {
                "model": (
                    "https://dl.fbaipublicfiles.com/avid-cma/checkpoints/"
                    "AVID_Audioset_Cross-N1024_checkpoint.pth.tar"
                ),
                "config": (
                    "https://raw.githubusercontent.com/YingtianDt/AVID-CMA/"
                    "main/avid_cma/configs/main/avid/audioset/Cross-N1024.yaml"
                ),
            },
            "AVID-CMA-Audioset": {
                "model": (
                    "https://dl.fbaipublicfiles.com/avid-cma/checkpoints/"
                    "AVID-CMA_Audioset_InstX-N1024-PosW-N64-Top32_"
                    "checkpoint.pth.tar"
                ),
                "config": (
                    "https://raw.githubusercontent.com/YingtianDt/AVID-CMA/"
                    "main/avid_cma/configs/main/avid-cma/audioset/"
                    "InstX-N1024-PosW-N64-Top32.yaml"
                ),
            },
        }

        # Current processor
        self.processor = None
        self.image_size = 256
        self.fps = 16  # Default, will be updated by config

        # Static flag
        self.static = False

    def preprocess_fn(
        self,
        input_data: Union[str, List[Union[np.ndarray, Image.Image]], Image.Image],
        fps: Optional[float] = None
    ):
        """
        Preprocesses input data for AVID.

        Args:
            input_data: Video file path (str), list of frames, or PIL Image.
            fps: Target FPS if resampling is required.

        Returns:
            dict: Preprocessed input for the model.

        Raises:
            ValueError: If processor not initialized or input invalid.
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first."
            )

        frames = []

        # Handle different input types
        if isinstance(input_data, str) and os.path.isfile(input_data):
            # Load video from file path
            cap = cv2.VideoCapture(input_data)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)

            duration = frame_count / video_fps if video_fps > 0 else 0

            # Use self.fps (from config) unless override provided
            target_fps_val = self.fps if fps is None else fps

            desired_frame_count = int(round(target_fps_val * duration))
            if desired_frame_count < 1:
                desired_frame_count = frame_count

            indices = np.linspace(
                0, frame_count - 1, desired_frame_count, dtype=np.int32
            )

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                else:
                    frames.append(
                        Image.new(
                            "RGB",
                            (self.image_size, self.image_size),
                            color=0
                        )
                    )
            cap.release()

        elif isinstance(input_data, (list, Image.Image)):
            if isinstance(input_data, Image.Image):
                input_data = [input_data]

            for frame in input_data:
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(np.uint8(frame)).convert("RGB")

                # Resize if necessary before processor
                if (frame.width != self.image_size or
                        frame.height != self.image_size):
                    frame = frame.resize((self.image_size, self.image_size))
                frames.append(frame)

        else:
            raise ValueError(
                "Input must be a list of PIL images, numpy arrays, "
                "or a video file path."
            )

        # Handle FPS downsampling if input was a list of frames
        if fps is not None and isinstance(input_data, list):
            target_fps = self.fps
            if (target_fps is not None and
                    target_fps != fps and
                    len(frames) > 1):
                orig_count = len(frames)
                duration_sec = orig_count / float(fps)
                new_count = int(round(duration_sec * float(target_fps)))
                indices = np.linspace(
                    0, orig_count - 1, num=new_count
                ).astype(int)
                frames = [frames[i] for i in indices]

        # Process with AVID processor
        inputs = self.processor(frames)
        return inputs

    def get_model(self, identifier: str):
        """
        Loads a AVID model based on the identifier.

        Args:
            identifier (str): Identifier for the AVID variant.

        Returns:
            model: The loaded AVID model.

        Raises:
            ValueError: If the identifier is unknown.
            RuntimeError: If download fails.
        """
        for prefix, model_cfg in self.model_mappings.items():
            if identifier.startswith(prefix):
                weights_dir = os.path.join(
                    get_data_home(), 'weights', self.__class__.__name__
                )
                os.makedirs(weights_dir, exist_ok=True)

                weights_path = os.path.join(
                    weights_dir, os.path.basename(model_cfg['model'])
                )

                # Logic to name config file distinctively
                is_kinetics = "Kinetics" in model_cfg['model']
                config_name = 'Kinetics_' if is_kinetics else 'Audioset_'
                config_name += os.path.basename(model_cfg['config'])
                config_path = os.path.join(weights_dir, config_name)

                # Download Weights
                if not os.path.isfile(weights_path):
                    print(
                        f"Downloading {identifier} weights from "
                        f"{model_cfg['model']}..."
                    )
                    try:
                        subprocess.run(
                            ['wget', model_cfg['model'], '-O', weights_path],
                            check=True
                        )
                        print(f"Downloaded weights to {weights_path}")
                    except Exception as e:
                        print(f"wget failed, trying urllib: {e}")
                        urllib.request.urlretrieve(
                            model_cfg['model'], weights_path
                        )

                # Download Config
                if not os.path.isfile(config_path):
                    print(
                        f"Downloading {identifier} config from "
                        f"{model_cfg['config']}..."
                    )
                    try:
                        urllib.request.urlretrieve(
                            model_cfg['config'], config_path
                        )
                        print(f"Downloaded config to {config_path}")
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to download config: {str(e)}"
                        )

                # FIX 3: Safe file opening
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)

                cfg['model']['args']['checkpoint'] = weights_path
                logger = Logger()

                print(f"Loading Model Config: {cfg['model']['name']}")
                model = main_utils.build_model(cfg['model'], logger)
                model = model.video_model

                # Define dataloaders
                db_cfg = cfg['dataset']

                # FIX 2: Update FPS from config
                self.fps = db_cfg.get('video_fps', 16)
                self.image_size = db_cfg.get('crop_size', 256)

                self.processor = preprocessing.VideoPrep_Crop_CJ(
                    resize=(256, 256),
                    crop=(self.image_size, self.image_size),
                    augment=False,
                    pad_missing=True,
                )

                # FIX 4: Safety check for compile
                if hasattr(torch, 'compile'):
                    try:
                        model = torch.compile(model)
                    except Exception as e:
                        print(
                            f"Warning: torch.compile failed, using standard "
                            f"model. Error: {e}"
                        )

                return model

        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(
        self,
        features_np: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Postprocesses AVID model output.

        Args:
            features_np: Output features.

        Returns:
            np.ndarray: Flattened features.
        """
        batch_size = features_np.shape[0]
        T = features_np.shape[3]
        flattened_features = features_np.reshape(batch_size, T, -1)

        return flattened_features
