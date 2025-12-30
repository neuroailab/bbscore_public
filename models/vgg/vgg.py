import os
import torch
import timm
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as v2


class VGG:
    """Loads pre-trained VGG-19 models (vision encoder only) with optional video-frame support via torchvision v2."""

    def __init__(self):
        """Initializes the VGG-19 loader."""
        self.model_mappings = {
            "VGG19": "vgg19.tv_in1k",
        }
        # The unified v2 transform for both images and videos
        self.transforms = None
        self.static = True
        # will be overridden by model config

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses image or video-frame data for VGG-19 using torchvision v2.

        Args:
            input_data: PIL Image, file path (str), numpy array, or list of those (for video).
            fps:        Unused here—frames must already be extracted if video.

        Returns:
            torch.Tensor:
              - Single image → shape [1, C, H, W]
              - Video frames list → shape [1, T, C, H, W]

        Raises:
            ValueError: If transforms not initialized or input type unsupported.
        """
        if self.transforms is None:
            raise ValueError(
                "Transforms not initialized. Call get_model() first.")

        # Single image
        if isinstance(input_data, (str, Image.Image, np.ndarray)):
            if isinstance(input_data, str) and os.path.isfile(input_data):
                img = Image.open(input_data).convert("RGB")
            elif isinstance(input_data, np.ndarray):
                img = Image.fromarray(np.uint8(input_data)).convert("RGB")
            else:
                img = input_data.convert("RGB")

            px = self.transforms(img)    # → [C, H, W]
            return px       # → [1, C, H, W]

        # Video: list of frames
        elif isinstance(input_data, list):
            # Convert all inputs to PIL and collect
            pil_frames = []
            for frame in input_data:
                if isinstance(frame, np.ndarray):
                    pil_frames.append(Image.fromarray(
                        np.uint8(frame)).convert("RGB"))
                else:
                    pil_frames.append(frame.convert("RGB"))
            # ONE transform call on the entire list
            # v2.Compose will process the list and return a Tensor [T, C, H, W]
            batched = self.transforms(pil_frames)  # → [T, C, H, W]
            data = torch.stack(batched)
            return data

        else:
            raise ValueError("Unsupported input type for preprocessing.")

    def get_model(self, identifier, pretrained=True):
        """
        Loads a VGG-19 model in FP16 and its matching preprocessing pipeline
        (which now outputs float16 tensors).

        Args:
            identifier (str): Prefix of the model e.g. "VGG19" or "VGG19-BN".
            pretrained (bool): Whether to load ImageNet pre-trained weights.

        Returns:
            torch.nn.Module: The VGG-19 model in eval mode, in FP16.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                # Create the model, convert to FP16, then eval
                model = timm.create_model(model_name, pretrained=pretrained)
                model = model.half().eval()

                # Grab timm's data config for this model
                data_config = timm.data.resolve_model_data_config(model)
                size = data_config.get("input_size")[-2:]
                mean = data_config.get("mean", (0.485, 0.456, 0.406))
                std = data_config.get("std",  (0.229, 0.224, 0.225))

                # Build a single torchvision v2 pipeline that ends in float16
                self.transforms = v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8,   scale=True),
                    v2.Resize(size,            antialias=True),
                    v2.CenterCrop(size),
                    v2.ToDtype(torch.float32, scale=True),  # ← now float16
                    v2.Normalize(mean=mean, std=std),
                    v2.ToDtype(torch.float16, scale=False),
                ])

                return model

        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        batch_size, t = features_np.shape[0], features_np.shape[1]
        return features_np.reshape(batch_size, t, -1)
