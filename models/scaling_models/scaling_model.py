import functools
import json
from pathlib import Path
import ssl

import torch
import torchvision.models
import timm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# --- Pre-computation and Helpers ---

ssl._create_default_https_context = ssl._create_unverified_context


def load_image(image_filepath: str) -> Image.Image:
    """Loads an image file into a PIL Image."""
    return Image.open(image_filepath).convert("RGB")


def get_interpolation_mode(interpolation: str):
    """Returns the interpolation mode integer for albumentations."""
    if "linear" in interpolation or "bilinear" in interpolation:
        return 1
    elif "cubic" in interpolation or "bicubic" in interpolation:
        return 2
    elif "nearest" in interpolation:
        return 0
    else:
        raise NotImplementedError(
            f"Interpolation mode '{interpolation}' not implemented.")

# --- Base Loaders ---


class BaseModelLoader:
    """
    Base class for loading models. Subclasses must have a `MODEL_ID` attribute.
    """
    MODEL_ID = None

    def __init__(self, config_path="configs.json"):
        if not self.MODEL_ID:
            raise NotImplementedError(
                "Loader classes must have a MODEL_ID attribute.")

        config_file = Path(__file__).parent / config_path
        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found at: {config_file}")

        with open(config_file, "r") as f:
            full_configs = json.load(f)

        if self.MODEL_ID not in full_configs:
            raise ValueError(
                f"Model ID '{self.MODEL_ID}' not found in configuration file.")

        self.config = full_configs[self.MODEL_ID]
        self.model = None
        self.static = True

    def get_model(self, identifier=None):
        """Loads the PyTorch model based on the instance's configuration."""
        model_name, num_classes, ckpt_url, use_timm, timm_model_name, load_model_ema, output_head = \
            self.config["model_name"], self.config["num_classes"], self.config["checkpoint_url"], \
            self.config["use_timm"], self.config["timm_model_name"], self.config["load_model_ema"], self.config["output_head"]

        if use_timm:
            model = timm.create_model(
                timm_model_name, pretrained=False, num_classes=num_classes)
        else:
            model = getattr(torchvision.models, model_name)(weights=None)
            if num_classes != 1000 and output_head:
                *head_path, final_layer = output_head.split('.')
                head_module = functools.reduce(
                    getattr, head_path, model) if head_path else model
                in_features = getattr(head_module, final_layer).in_features
                bias = getattr(head_module, final_layer).bias is not None
                setattr(head_module, final_layer, torch.nn.Linear(
                    in_features, num_classes, bias=bias))

        state_dict = torch.hub.load_state_dict_from_url(
            ckpt_url, file_name=f"{self.config['model_id']}.pt", map_location="cpu"
        )
        state_dict = state_dict.get("state", {}).get(
            "model_ema_state_dict" if load_model_ema else "model", state_dict)
        state_dict = {k.replace("module.", ""): v for k,
                      v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model
        print(f"Model '{self.config['model_id']}' loaded successfully.")
        return self.model

    def preprocess_fn(self, images):
        """Preprocesses a batch of images."""
        if not isinstance(images, list):
            images = [images]
        loaded_images = [load_image(img) if isinstance(img, str) else (
            Image.fromarray(img) if isinstance(img, np.ndarray) else img) for img in images]

        interpolation = get_interpolation_mode(self.config['interpolation'])
        transforms = A.Compose([
            A.Resize(self.config['resize_size'], self.config['resize_size'],
                     p=1.0, interpolation=interpolation),
            A.CenterCrop(self.config['crop_size'],
                         self.config['crop_size'], p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        return torch.stack([transforms(image=np.array(img))["image"] for img in loaded_images])

    def postprocess_fn(self, model_output):
        """Post-processes model outputs."""
        B, T = model_output.shape[0], model_output.shape[1]
        model_output = model_output.reshape(B, T, -1)
        return model_output


# --- ResNet Models ---

class Resnet18ImagenetFullLoader(BaseModelLoader):
    MODEL_ID = "resnet18_imagenet_full"
    def __init__(self): super().__init__()


class Resnet34ImagenetFullLoader(BaseModelLoader):
    MODEL_ID = "resnet34_imagenet_full"
    def __init__(self): super().__init__()


class Resnet50ImagenetFullLoader(BaseModelLoader):
    MODEL_ID = "resnet50_imagenet_full"
    def __init__(self): super().__init__()


class Resnet101ImagenetFullLoader(BaseModelLoader):
    MODEL_ID = "resnet101_imagenet_full"
    def __init__(self): super().__init__()


class Resnet152ImagenetFullLoader(BaseModelLoader):
    MODEL_ID = "resnet152_imagenet_full"
    def __init__(self): super().__init__()


class Resnet18EcosetFullLoader(BaseModelLoader):
    MODEL_ID = "resnet18_ecoset_full"
    def __init__(self): super().__init__()


class Resnet34EcosetFullLoader(BaseModelLoader):
    MODEL_ID = "resnet34_ecoset_full"
    def __init__(self): super().__init__()


class Resnet50EcosetFullLoader(BaseModelLoader):
    MODEL_ID = "resnet50_ecoset_full"
    def __init__(self): super().__init__()


class Resnet101EcosetFullLoader(BaseModelLoader):
    MODEL_ID = "resnet101_ecoset_full"
    def __init__(self): super().__init__()


class Resnet152EcosetFullLoader(BaseModelLoader):
    MODEL_ID = "resnet152_ecoset_full"
    def __init__(self): super().__init__()


class Resnet50Imagenet1Seed0Loader(BaseModelLoader):
    MODEL_ID = "resnet50_imagenet_1_seed-0"
    def __init__(self): super().__init__()


class Resnet50Imagenet10Seed0Loader(BaseModelLoader):
    MODEL_ID = "resnet50_imagenet_10_seed-0"
    def __init__(self): super().__init__()


class Resnet50Imagenet100Seed0Loader(BaseModelLoader):
    MODEL_ID = "resnet50_imagenet_100_seed-0"
    def __init__(self): super().__init__()

# --- EfficientNet Models ---


class EfficientnetB0ImagenetFullLoader(BaseModelLoader):
    MODEL_ID = "efficientnet_b0_imagenet_full"
    def __init__(self): super().__init__()


class EfficientnetB1ImagenetFullLoader(BaseModelLoader):
    MODEL_ID = "efficientnet_b1_imagenet_full"
    def __init__(self): super().__init__()


class EfficientnetB2ImagenetFullLoader(BaseModelLoader):
    MODEL_ID = "efficientnet_b2_imagenet_full"
    def __init__(self): super().__init__()

# --- DeiT Models ---


class DeitSmallImagenetFullSeed0Loader(BaseModelLoader):
    MODEL_ID = "deit_small_imagenet_full_seed-0"
    def __init__(self): super().__init__()


class DeitBaseImagenetFullSeed0Loader(BaseModelLoader):
    MODEL_ID = "deit_base_imagenet_full_seed-0"
    def __init__(self): super().__init__()


class DeitLargeImagenetFullSeed0Loader(BaseModelLoader):
    MODEL_ID = "deit_large_imagenet_full_seed-0"
    def __init__(self): super().__init__()


class DeitSmallImagenet1Seed0Loader(BaseModelLoader):
    MODEL_ID = "deit_small_imagenet_1_seed-0"
    def __init__(self): super().__init__()


class DeitSmallImagenet10Seed0Loader(BaseModelLoader):
    MODEL_ID = "deit_small_imagenet_10_seed-0"
    def __init__(self): super().__init__()


class DeitSmallImagenet100Seed0Loader(BaseModelLoader):
    MODEL_ID = "deit_small_imagenet_100_seed-0"
    def __init__(self): super().__init__()

# --- ConvNeXt Models ---


class ConvnextTinyImagenetFullSeed0Loader(BaseModelLoader):
    MODEL_ID = "convnext_tiny_imagenet_full_seed-0"
    def __init__(self): super().__init__()


class ConvnextSmallImagenetFullSeed0Loader(BaseModelLoader):
    MODEL_ID = "convnext_small_imagenet_full_seed-0"
    def __init__(self): super().__init__()


class ConvnextBaseImagenetFullSeed0Loader(BaseModelLoader):
    MODEL_ID = "convnext_base_imagenet_full_seed-0"
    def __init__(self): super().__init__()


class ConvnextLargeImagenetFullSeed0Loader(BaseModelLoader):
    MODEL_ID = "convnext_large_imagenet_full_seed-0"
    def __init__(self): super().__init__()


class ConvnextSmallImagenet1Seed0Loader(BaseModelLoader):
    MODEL_ID = "convnext_small_imagenet_1_seed-0"
    def __init__(self): super().__init__()


class ConvnextSmallImagenet10Seed0Loader(BaseModelLoader):
    MODEL_ID = "convnext_small_imagenet_10_seed-0"
    def __init__(self): super().__init__()


class ConvnextSmallImagenet100Seed0Loader(BaseModelLoader):
    MODEL_ID = "convnext_small_imagenet_100_seed-0"
    def __init__(self): super().__init__()
