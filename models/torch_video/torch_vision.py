import os
import numpy as np
from PIL import Image

import torch
import torchvision.models.video as video_models
from torchvision import transforms

from torchvision.models.video import (
    MC3_18_Weights,
    MViT_V1_B_Weights,
    MViT_V2_S_Weights,
    R2Plus1D_18_Weights,
    R3D_18_Weights,
    S3D_Weights,
    Swin3D_B_Weights,
    Swin3D_S_Weights,
    Swin3D_T_Weights,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class TorchVideoModels:
    """Loads pre-trained video models (vision encoders only, by default)."""

    def __init__(self):
        """Initializes the video model loader."""
        self.model_mappings = {
            "MC3_18_Weights": video_models.mc3_18,
            "MViT": video_models.mvit,
            "MViT_V1_B_Weights": video_models.mvit_v1_b,
            "MViT_V2_S_Weights": video_models.mvit_v2_s,
            "R2Plus1D_18_Weights": video_models.r2plus1d_18,
            "R3D_18_Weights": video_models.r3d_18,
            "S3D_Weights": video_models.s3d,
            "Swin3D_B_Weights": video_models.swin3d_b,
            "Swin3D_S_Weights": video_models.swin3d_s,
            "Swin3D_T_Weights": video_models.swin3d_t,
            "SwinTransformer3d": video_models.SwinTransformer3d,
        }

        # Map model identifiers to their torchvision Weights
        self.VIDEO_MODEL_PROCESSORS = {
            "MC3_18_Weights": MC3_18_Weights.DEFAULT.transforms(),
            "R2Plus1D_18_Weights": R2Plus1D_18_Weights.DEFAULT.transforms(),
            "R3D_18_Weights": R3D_18_Weights.DEFAULT.transforms(),
            "S3D_Weights": S3D_Weights.DEFAULT.transforms(),
            "MViT_V1_B_Weights": MViT_V1_B_Weights.DEFAULT.transforms(),
            "MViT_V2_S_Weights": MViT_V2_S_Weights.DEFAULT.transforms(),
            "Swin3D_B_Weights": Swin3D_B_Weights.DEFAULT.transforms(),
            "Swin3D_S_Weights": Swin3D_S_Weights.DEFAULT.transforms(),
            "Swin3D_T_Weights": Swin3D_T_Weights.DEFAULT.transforms(),
            "SwinTransformer3d": Swin3D_B_Weights.DEFAULT.transforms(),  # fallback
            "MViT": MViT_V1_B_Weights.DEFAULT.transforms(),            # alias
        }

        self.fps_dict = {"MC3_18_Weights": 25,
                         "R2Plus1D_18_Weights": 25,
                         "R3D_18_Weights": 25,
                         "S3D_Weights": 24,
                         "MViT_V1_B_Weights": 7.5,
                         "MViT_V2_S_Weights": 7.5,
                         "MViT": 7.5,
                         "Swin3D_B_Weights": 15,
                         "Swin3D_S_Weights": 15,
                         "Swin3D_T_Weights": 15,
                         "SwinTransformer3d": 15}

        # Current processor
        self.processor = None
        self.fps = 15
        self.static = False

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

        num_frames = len(frames)
        if "MViT" in self.identifier:
            if num_frames < 16:
                # fallback blank frame
                last_frame = frames[-1] if frames else Image.new(
                    "RGB", (224, 224))
                frames += [last_frame] * (16 - num_frames)
            elif num_frames > 16:
                indices = np.linspace(0, num_frames - 1, 16, dtype=int)
                frames = [frames[i] for i in indices]
        elif "S3D" in self.identifier:
            if num_frames < 13:
                # fallback blank frame
                last_frame = frames[-1] if frames else Image.new(
                    "RGB", (224, 224))
                frames += [last_frame] * (13 - num_frames)

        frames = torch.stack(frames)
        frames = self.processor(frames)
        return frames.half()

    def get_model(self, identifier):
        if identifier not in self.model_mappings:
            raise ValueError(f"Unknown model: {identifier}")
        self.identifier = identifier
        self.fps = self.fps_dict[identifier]
        self.model = self.model_mappings[identifier](weights="DEFAULT")
        self.processor = self.VIDEO_MODEL_PROCESSORS[identifier]
        self.model = torch.compile(self.model.half().to('cuda'))
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
        if "MViT" in self.identifier:
            flattened_features = features_np.squeeze(1)
        else:
            T = features_np.shape[3]
            flattened_features = features_np.reshape(batch_size, T, -1)

        return flattened_features
