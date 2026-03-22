from pathlib import Path

import torch
from torch import nn
import torchvision
from transformers import AutoImageProcessor
from sklearn.datasets import get_data_home
from torch.hub import download_url_to_file
from transformers import AutoImageProcessor

from models.resnet import ResNet

class TDANN(ResNet):
    """All models are based off the ResNet-18 architecture. Note that the tdann was trained
    so that each weight was assigned to a coordinate on a cortical sheet which affected its 
    spatial loss term. Ie, the learned weights inherently reflect the forced topology"""

    def __init__(self):
        """Initializes the TDANN loader."""
        self.cache_dir = Path(get_data_home())
        self.cache_dir.mkdir(exist_ok=True)

        # map of id: osf weight file url
        self.model_mappings = {
            "tdann_simclr": "https://osf.io/58we2/download",
            "tdann_supervised": "https://osf.io/xz5ky/download",
        }

        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
        self.static = True

    def preprocess_fn(self, input_data):
        """
        Preprocesses input data for the model.

        Args:
            input_data: PIL Image, file path (str), or numpy array.

        Returns:
            torch.Tensor: Preprocessed input tensor.

        Raises:
            ValueError: If the input type is invalid.
        """
        return super().preprocess_fn(input_data)

    def _load_model_from_checkpoint(self, checkpoint_path: str):
        """With help from https://github.com/neuroailab/TDANN/"""
        model = torchvision.models.resnet18(weights=None)

        # drop the FC layer
        model.fc = nn.Identity()

        # load weights
        ckpt = torch.load(checkpoint_path)
        state_dict = ckpt["classy_state_dict"]["base_model"]["model"]["trunk"]

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("base_model") and "fc." not in k:
                remainder = k.split("base_model.")[-1]
                new_state_dict[remainder] = v

        model.load_state_dict(new_state_dict)

        # freeze all weights
        for param in model.parameters():
            param.requires_grad = False

        return torch.compile(model)

    def get_model(self, identifier):
        """
        Loads a TDANN model.

        Args:
            identifier (str): Identifier for the variant.

        Returns:
            model: The loaded model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        if identifier not in self.model_mappings:
            raise ValueError(
                f"Unknown model identifier: {identifier}. "
                f"Available: {', '.join(self.model_mappings.keys())}"
            )

        checkpoint_url = self.model_mappings[identifier]
        local_filename = f"{identifier}_checkpoint.pth"
        checkpoint_path = self.cache_dir / local_filename

        if not checkpoint_path.exists():
            download_url_to_file(checkpoint_url, checkpoint_path, progress=True)

        return self._load_model_from_checkpoint(checkpoint_path)

    def postprocess_fn(self, features_np):
        """Postprocesses model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from model
                as a numpy array.

        Returns:
            np.ndarray: postprocessed feature tensor.
        """
        return super().postprocess_fn(features_np)
