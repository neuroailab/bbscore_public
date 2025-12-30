import os
import torch
from PIL import Image
from sklearn.datasets import get_data_home
from torchvision import transforms

from .cwm_model import CWM_MODEL
from ccwm.cwm.cwm_predictor import CWMPredictor


class CWM:
    '''
    IMPORTANT: you need to have ccwm installed in your environment.
    '''

    def __init__(self):
        self.model_mappings = {
            "CWM170M_0.0": ("CWM170M_RGB_BigVideo_200k/model_00200000.pt", 0.0),
            "CWM170M_0.9": ("CWM170M_RGB_BigVideo_200k/model_00200000.pt", 0.9),
            "CWM170M_1.0": ("CWM170M_RGB_BigVideo_200k/model_00200000.pt", 1.0),
            "CWM170M_ONE_IMAGE": ("CWM170M_RGB_BigVideo_200k/model_00200000.pt", -1.0),
            "CWM1B_0.0": ("CWM1B_RGB_BigVideo_200k/model_00200000.pt", 0.0),
            "CWM1B_0.9": ("CWM1B_RGB_BigVideo_200k/model_00200000.pt", 0.9),
            "CWM1B_1.0": ("CWM1B_RGB_BigVideo_200k/model_00200000.pt", 1.0),
            "CWM1B_ONE_IMAGE": ("CWM1B_RGB_BigVideo_200k/model_00200000.pt", -1.0),
        }
        self.static = False
        self.fps = 10.0
        self.processor = None

    def preprocess_fn(self, input_data, fps=None):
        if not isinstance(input_data, list):
            input_data = [input_data]

        processed_frames = [input_data[0], input_data[-1]]

        if self.processor is None:
            raise ValueError(
                "Processor not initialized. Call get_model() first.")

        tensor_frames = [self.processor(frame) for frame in processed_frames]
        return torch.stack(tensor_frames, dim=0)

    def get_model(self, identifier):
        if identifier not in self.model_mappings:
            raise ValueError(f"Unknown model identifier: {identifier}")

        model_name, mask_ratio = self.model_mappings[identifier]

        if not hasattr(self, 'dummy_predictor'):
            self.dummy_predictor = CWMPredictor(model_name=model_name)

        resize_crop_transform = transforms.Compose([
            transforms.Resize(self.dummy_predictor.model.config.resolution),
            transforms.CenterCrop(
                self.dummy_predictor.model.config.resolution),
        ])

        self.processor = transforms.Compose([
            resize_crop_transform,
            self.dummy_predictor.in_transform,
        ])

        model = CWM_MODEL(model_name, mask_ratio)
        return model

    def postprocess_fn(self, features_np):
        return features_np.squeeze(0)
