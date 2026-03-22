import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


class Wav2Vec2:
    def __init__(self):
        self.model_mappings = {
            "Wav2Vec2-Base": "facebook/wav2vec2-base",
            "Wav2Vec2-Large": "facebook/wav2vec2-large",
            "Wav2Vec2-Large-960h": "facebook/wav2vec2-large-960h",
        }
        self.processor = None
        self.static = True

    def get_model(self, identifier):
        hf_name = self.model_mappings[identifier]
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(hf_name)
        model = Wav2Vec2Model.from_pretrained(hf_name)
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed for Wav2Vec2: {e}")
        return model

    def preprocess_fn(self, input_data):
        """
        Args:
            input_data: np.ndarray of shape (num_samples,) at 16kHz
        Returns:
            Tensor of shape (1, num_samples)
        """
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()
        inputs = self.processor(
            input_data, sampling_rate=16000,
            return_tensors="pt", padding=True
        )
        return inputs.input_values.squeeze(0)

    def postprocess_fn(self, features):
        """Mean-pool encoder output (B, seq_len, hidden_dim) -> (B, hidden_dim)."""
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if features.ndim == 3:
            return features.mean(axis=1)
        return features.reshape(features.shape[0], -1)
