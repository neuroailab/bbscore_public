import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
import os
from models.ssast.ast_models import ASTModel

# AudioSet normalization statistics — must match convert_lebel_to_spectrograms.py
AUDIOSET_MEAN = -4.2677393
AUDIOSET_STD  =  4.5689974

class SSASTModel(nn.Module):
    """
    Thin nn.Module wrapping ASTModel for BBScore hook-based feature extraction.

    Recommended layer for feature extraction: 'ast.v.norm'
    (hook output shape: (B, N_patches + n_cls_tokens, 768);
     postprocess_fn mean-pools this to (B, 768))
    """

    _CONFIGS = {
        'ssast_patch': {
            'url': 'https://www.dropbox.com/scl/fi/0q6hzz06qcjx0il370k83/SSAST-Base-Patch-400.pth?rlkey=0jxlz3zh9ryk9zc68zr2jmml9&dl=1',
            'checkpoint_name': 'SSAST-Base-Patch-400.pth',
            'fshape': 16, 'tshape': 16,
            'fstride': 16, 'tstride': 16,
            'input_fdim': 128, 'input_tdim': 1024,
            'model_size': 'base',
        },
        'ssast_frame': {
            'url': 'https://www.dropbox.com/scl/fi/ls5b3v6r4182fbrlld9t2/SSAST-Base-Frame-400.pth?rlkey=jaeuyuxq2qetizh7ybr8rrhli&dl=1',
            'checkpoint_name': 'SSAST-Base-Frame-400.pth',
            'fshape': 128, 'tshape': 2,
            'fstride': 128, 'tstride': 2,
            'input_fdim': 128, 'input_tdim': 1024,
            'model_size': 'base',
        },
    }

    def __init__(self, model_identifier: str):
        super().__init__()
        config = self._CONFIGS.get(model_identifier)
        if config is None:
            print(f'[SSAST] Unknown identifier "{model_identifier}". Defaulting to ssast_patch.')
            config = self._CONFIGS['ssast_patch']

        # Store expected input dimensions for forward pass padding.
        self.input_fdim = config['input_fdim']
        self.input_tdim = config['input_tdim']

        # ASTModel(pretrain_stage=False) requires a valid checkpoint path at
        # construction time — download first, then pass the path.
        cached_file = self._get_checkpoint(config['url'], config['checkpoint_name'])

        self.ast = ASTModel(
            label_dim=527,
            fshape=config['fshape'],
            tshape=config['tshape'],
            fstride=config['fstride'],
            tstride=config['tstride'],
            input_fdim=config['input_fdim'],
            input_tdim=config['input_tdim'],
            model_size=config['model_size'],
            pretrain_stage=False,
            load_pretrained_mdl_path=cached_file,
        )

    @staticmethod
    def _get_checkpoint(url: str, filename: str) -> str:
        cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
        os.makedirs(cache_dir, exist_ok=True)
        cached_file = os.path.join(cache_dir, filename)
        if not os.path.exists(cached_file):
            print(f'[SSAST] Downloading weights to {cached_file} ...')
            torch.hub.download_url_to_file(url, cached_file)
        return cached_file

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, F) or (B, T, F) normalized log-mel spectrogram, float32.
               preprocess_fn returns (T, F); framework may or may not add batch dim.
        Returns:
            (B, D) or (D,) token-averaged features (depends on input dimensionality).
        """
        # Handle both (T, F) and (B, T, F) inputs by adding batch dim if needed.
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (T, F) → (1, T, F)
        
        # Pad time dimension to self.input_tdim (1024 for both patch and frame models).
        # Shape is now (B, T, 128); we need to pad T to input_tdim.
        B, T, F = x.shape
        if T < self.input_tdim:
            pad_amount = self.input_tdim - T
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_amount))  # pad on time dim
        elif T > self.input_tdim:
            x = x[:, :self.input_tdim, :]  # truncate if too long
        
        return self.ast(x)


class SSASTWrapper:
    """BBScore factory class for SSAST models."""

    # Shared MelSpectrogram transform — instantiated once per process.
    _mel_transform: T.MelSpectrogram = None

    def __init__(self):
        self.static = True  # single-clip audio model, not a sequential video model

    def get_model(self, identifier: str) -> SSASTModel:
        return SSASTModel(identifier)

    @classmethod
    def _build_mel_transform(cls) -> T.MelSpectrogram:
        if cls._mel_transform is None:
            # Parameters must exactly match convert_lebel_to_spectrograms.py
            cls._mel_transform = T.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=160,
                n_mels=128,
                f_min=0.0,
                f_max=8000,
                power=2.0,
                norm='slaney',
                mel_scale='htk',
            )
        return cls._mel_transform

    def preprocess_fn(self, input_data):
        """
        Convert a stimulus to a normalized log-mel spectrogram tensor.

        Accepts:
          - str: path to a pre-computed .npy spectrogram (T, 128) saved by
                 convert_lebel_to_spectrograms.py — loaded and returned directly.
          - np.ndarray / torch.Tensor: raw 1-D waveform at 16 kHz — converted
                 on-the-fly using the same parameters as convert_lebel_to_spectrograms.py.

        Returns:
            torch.Tensor of shape (T, 128), float32, normalized.
        """
        if isinstance(input_data, str):
            if not input_data.endswith('.npy'):
                raise ValueError(
                    f'String inputs must be paths to .npy spectrogram files, got: {input_data}'
                )
            spec = np.load(input_data).astype(np.float32)
            return torch.from_numpy(spec)  # already (T, 128)

        if isinstance(input_data, np.ndarray):
            waveform = torch.from_numpy(input_data).float()
        elif isinstance(input_data, torch.Tensor):
            waveform = input_data.float()
        else:
            raise TypeError(f'[SSAST] Unsupported preprocess_fn input type: {type(input_data)}')

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)

        mel_transform = self._build_mel_transform()
        mel_spec = mel_transform(waveform)              # (1, 128, T)
        log_mel = torch.log(mel_spec + 1e-7)            # (1, 128, T)
        log_mel = (log_mel - AUDIOSET_MEAN) / (AUDIOSET_STD + 1e-8)
        return log_mel.squeeze(0).T                     # (T, 128)

    def postprocess_fn(self, features):
        """
        Mean-pool over the token/sequence dimension.

        Hooks on transformer layers (e.g. 'ast.v.norm') yield (B, N, D).
        Returns (B, D), matching the convention in wav2vec2.postprocess_fn.
        """
        if isinstance(features, torch.Tensor):
            if features.dim() == 3:
                return features.mean(dim=1)
            return features
        if isinstance(features, np.ndarray):
            if features.ndim == 3:
                return features.mean(axis=1)
            return features
        return features

