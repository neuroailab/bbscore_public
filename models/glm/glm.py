import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Glm4vForConditionalGeneration, Glm4vMoeForConditionalGeneration

# This wrapper class correctly unpacks the dictionary of inputs from the processor.


class VLM:
    """
    Wrapper for Zhipu AI's GLM-V series of Vision-Language Models.
    """

    def __init__(self):
        """Initializes the GLM-V loader."""
        self.model_mappings = {
            "GLM-4.5V": "zai-org/GLM-4.5V",
            "GLM-4.1V-9B": "zai-org/GLM-4.1V-9B-Thinking",
        }
        self.static = True
        self.processor = None
        self.model_name = ""
        self.model = None
        self.original_forward = None

    def preprocess_fn(self, input_data, fps=None):
        """
        Return ONLY NumPy images (no tensors). Output is a list[np.ndarray],
        each shaped (256, 256, 3), dtype=uint8, RGB.
        """
        def to_np_rgb(img):
            if not isinstance(img, Image.Image):
                raise ValueError(
                    "preprocess_fn expects PIL.Image or list[PIL.Image].")
            img = img.resize((256, 256), Image.BICUBIC)  # Resize to 256x256
            return np.array(img.convert("RGB"), dtype=np.uint8)

        if isinstance(input_data, list):
            return [to_np_rgb(img) for img in input_data]
        elif isinstance(input_data, Image.Image):
            return [to_np_rgb(input_data)]
        else:
            raise ValueError(
                "Input must be a PIL Image or a list of PIL Images.")

    def custom_forward(self, inputs):
        """
        Feature extraction forward logic for processing images.
        Returns model outputs that can be captured by hooks.
        """
        # ---- chat prompt per image ----
        prompt = "Describe the visual content of this image in detail."

        messages = [
            {"role": "user", "content": [
                {"type": "image"},  # we'll attach the image separately
                {"type": "text", "text": prompt}
            ]}
        ]  # * len(inputs)#.shape[0]
        chat_str = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 3. Tokenize the chat prompt (text) and preprocess image
        vision_inputs = self.processor(
            images=inputs, text=chat_str, return_tensors="pt")

        # 5. Forward pass
        outputs = self.original_forward(**vision_inputs.to('cuda'))
        return outputs

    def hybrid_forward(self, *args, **kwargs):
        """
        Hybrid forward that handles both custom inputs and model's internal calls.
        """
        # Check if this is your custom call (image input from FeatureExtractor)
        # This will be the case when FeatureExtractor calls model(inputs)
        if (len(args) == 1 and len(kwargs) == 0 and
            not isinstance(args[0], torch.Tensor) and
                not (isinstance(args[0], dict) and 'input_ids' in args[0])):
            # This is your custom forward call with image inputs
            return self.custom_forward(args[0])
        else:
            # This is the model's internal forward call during the custom_forward
            return self.original_forward(*args, **kwargs)

    def forward(self, inputs):
        """
        Main forward method that delegates to hybrid_forward.
        """
        return self.hybrid_forward(inputs)

    def get_model(self, identifier):
        """
        Loads a GLM-V model, configures it, and wraps it.
        """
        if identifier not in self.model_mappings:
            raise ValueError(f"Unknown model identifier: {identifier}")

        model_path = self.model_mappings[identifier]
        self.model_name = identifier

        print("⚠️  GLM requires Batch Size = 1 !")
        print(
            f"Loading model and processor for {identifier} from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True)

        ModelClass = Glm4vMoeForConditionalGeneration if "4.5V" in identifier else Glm4vForConditionalGeneration

        actual_model = ModelClass.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=True,
            device_map="cuda"
        )

        config_owner = actual_model
        if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'language_model'):
            config_owner = actual_model.model.language_model

        if hasattr(config_owner, 'config'):
            config_owner.config.output_hidden_states = True
            print("Successfully set `output_hidden_states=True`.")
        else:
            raise AttributeError("Could not find the model's configuration.")

        self.model = actual_model
        # Store the original forward method
        self.original_forward = self.model.forward
        # Replace with hybrid forward
        self.model.forward = self.hybrid_forward

        return self.model

    def postprocess_fn(self, features):
        """
        Post-processes the model's features from the hook.
        """
        return features.unsqueeze(0)
