import os
import re
import torch
import numpy as np

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale, Normalize

from dualneuron.twins.nets import V1GrayTaskDriven, V4GrayTaskDriven, V4ColorTaskDriven
from dualneuron.twins.activations import get_spatial_activation

class DualNeuron:
    """Loads pre-trained Dual Neuron models."""

    def __init__(self):
        """Initializes the Model loader."""
        
        self.model_mappings = {
            'gray_V1': 'V1GrayTaskDriven',
            'gray_V4': 'V4GrayTaskDriven',
            'rgb_V4': 'V4ColorTaskDriven',
        }
        self.image_sizes = {
            'gray_V1': (93, 93),
            'gray_V4': (100, 100),
            'rgb_V4': (100, 100),                   
        }
        
        # Static flag if this is image-model
        self.static = True
        
    @staticmethod
    def _parse_identifier(identifier):
        """Uniformizes the parsed identifier of the model"""
        if re.search(r'V1', identifier):
            area = 'V1'
        elif re.search(r'V4', identifier):
            area = 'V4'
        else:
            raise ValueError("Need either 'V1' or 'V4' in identifier string for DualNeuron model")
        
        if re.search(r'rgb', identifier):
            colorscale = 'rgb'
        elif re.search(r'gray', identifier) or re.search(r'grey', identifier):
            colorscale = 'gray'
        else:
            raise ValueError("Need either 'rgb' or 'gray' in identifier string for DualNeuron model")

        if area == 'V1' and colorscale == 'rgb':
            raise ValueError("DualNeuron does not support rgb input in V1")

        parsed_identifier = colorscale + '_' + area
        
        return parsed_identifier
        

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
        
        if isinstance(input_data, str) and os.path.isfile(input_data):
            data = Image.open(input_data)
            
        elif isinstance(input_data, Image.Image) or isinstance(input_data, np.ndarray):
            data = input_data
            
        elif isinstance(input_data, torch.Tensor):
            data = input_data
        else:
            raise ValueError("Input must be image file, Image.Image object, or tensor")
        
        # reshape data to correct shape
        transform_list = [
            ToTensor(),
            Resize(self.image_size[0]),
        ]
    
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if self.grayscale:
            transform_list.append(Grayscale())
            
            mean_gray = sum(mean) / 3
            std_gray = sum(std) / 3
            transform_list.append(Normalize([mean_gray], [std_gray]))
        else:
            transform_list.append(Normalize(mean, std))
            
        transform = Compose(transform_list)
        preprocessed_data = transform(data)
        
        return preprocessed_data

    def get_model(self, identifier='V1 gray'):
        """
        Loads a model based on the identifier.

        Args:
            identifier (str): Identifier for the model variant.

        Returns:
            The loaded model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        
        id = self.model_mappings[self._parse_identifier(identifier)]
        
        if id == 'V1GrayTaskDriven':
            self.model = V1GrayTaskDriven(ensemble=True, centered=True)
            self.grayscale = True
        elif id == 'V4GrayTaskDriven':
            self.model = V4GrayTaskDriven(ensemble=True, centered=True)
            self.grayscale = True
        elif id == 'V4ColorTaskDriven':
            self.model = V4ColorTaskDriven(ensemble=True, centered=True)
            self.grayscale = False

        self.image_size = self.image_sizes[self._parse_identifier(identifier)]

        return self.model

    def postprocess_fn(self, features_np, standardize = False):
        """Postprocesses model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from DualNeuron model

        Returns:
            np.ndarray: Flattened feature tensor of shape (N, -1),
                where N is batch size (or 1 if single sample).
                
        Turns out they do all the work for you inside dualneuron
        """
        
        stim_by_neuron = np.array(get_spatial_activation(torch.Tensor(features_np)))
        
        if standardize and stim_by_neuron.shape[0] > 1: # can't standardize if just 1 stimulus
            return (stim_by_neuron - stim_by_neuron.mean(axis=0)) / stim_by_neuron.std(axis=0)
        else:
            return stim_by_neuron
