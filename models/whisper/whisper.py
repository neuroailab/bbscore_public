import numpy as np
import torch
from transformers import WhisperModel, WhisperFeatureExtractor


class Whisper:
    def __init__(self):
        self.model_mappings = {
            "Whisper-Tiny": "openai/whisper-tiny",
            "Whisper-Base": "openai/whisper-base",
            "Whisper-Small": "openai/whisper-small",
            "Whisper-Medium": "openai/whisper-medium",
            "Whisper-Large-v3": "openai/whisper-large-v3",
        }
        self.processor = None
        self.static = True
        self.audio_context_duration = 30.0  # Whisper expects 30s input

    def get_model(self, identifier):
        hf_name = self.model_mappings[identifier]
        self.processor = WhisperFeatureExtractor.from_pretrained(hf_name)
        full_model = WhisperModel.from_pretrained(hf_name)
        model = full_model.encoder
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed for Whisper: {e}")
        return model

    def preprocess_fn(self, input_data):
        """
        Args:
            input_data: np.ndarray of shape (num_samples,) at 16kHz
        Returns:
            Tensor of shape (1, n_mels, time_steps)
        """
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()
        inputs = self.processor(
            input_data, sampling_rate=16000,
            return_tensors="pt"
        )
        return inputs.input_features

    def postprocess_fn(self, features):
        """Mean-pool encoder output (B, seq_len, hidden_dim) -> (B, hidden_dim)."""
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if features.ndim == 3:
            return features.mean(axis=1)
        return features.reshape(features.shape[0], -1)
