import os
 
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel
 
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

class SigLIP:
    def __init__(self):
        """Initializes the SigLIP loader."""
        self.model_mappings = {
            "SigLIP-Base": "google/siglip-base-patch16-224",
            "SigLIP-Large": "google/siglip-large-patch16-384",
        }
 
        self.processor = None
        self.input_size = 224  # Default size
 
        # Static flag
        self.static = True

    def get_model(self, identifier):
        """
        Loads a SigLIP vision encoder.
 
        Args:
            identifier (str): Identifier for the SigLIP variant
                (e.g., "SigLIP-Base").
 
        Returns:
            model: The compiled SigLIP vision model.
 
        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_name in self.model_mappings.items():
            if identifier == prefix:
                # Update input size for siglip-large-patch16-384
                if "Large" in identifier:
                    self.input_size = 384
                
                # Load vision model directly
                model = SiglipVisionModel.from_pretrained(model_name, torch_dtype=torch.float16)
                self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
                model = torch.compile(model)

                return model
        
        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available prefixes: {', '.join(self.model_mappings.keys())}"
        )
    
    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses image data for SigLIP.
 
        Args:
            input_data: PIL Image, file path (str), numpy array, or list.
            fps: unused, kept for API compatibility.
 
        Returns:
            torch.Tensor: Preprocessed pixel values in float16.
 
        Raises:
            ValueError: If processor not initialized or input is invalid.
        """
        if self.processor is None:
            raise ValueError(
                "Processor not initialized.  Call get_model() first.")

        # Check input format
        if isinstance(input_data, str) and os.path.isfile(input_data):
            img = Image.open(input_data).convert("RGB")
            if img.width != self.input_size or img.height != self.input_size:
                img = img.resize((self.input_size, self.input_size))
        elif isinstance(input_data, np.ndarray):
            img = Image.fromarray(np.uint8(input_data)).convert("RGB")
            if img.width != self.input_size or img.height != self.input_size:
                img = img.resize((self.input_size, self.input_size))
        elif isinstance(input_data, Image.Image):
            img = input_data.convert("RGB")
            if img.width != self.input_size or img.height != self.input_size:
                img = img.resize((self.input_size, self.input_size))
        elif isinstance(input_data, list):
            img = []
            for i in input_data:
                img += [i.convert("RGB").resize((self.input_size,
                                                 self.input_size))]
        else:
            raise ValueError(
                "Input must be a PIL Image, file path, or numpy array")
        
        return self.processor(images=img, return_tensors="pt").pixel_values.half()

    def postprocess_fn(self, features_np):
        """Postprocesses SigLIP vision encoder output.
 
        Args:
            features_np (np.ndarray): Output features from SigLIP
                vision encoder as a numpy array.
 
        Returns:
            np.ndarray: Reshaped feature tensor of shape
                (batch, num_patches, -1).
        """
        batch_size, num_patches = features_np.shape[0], features_np.shape[1]
        flattened_features = features_np.reshape(batch_size, num_patches, -1)
        return flattened_features