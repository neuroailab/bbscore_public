import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from phys_extractors.models.mcvd_pytorch.load_model_from_ckpt import load_model, get_readout_sampler, init_samples
from phys_extractors.models.mcvd_pytorch.datasets import data_transform
from phys_extractors.models.mcvd_pytorch.runners.ncsn_runner import conditioning_fn


class MCVD_Wrapper(nn.Module):
    def __init__(self, weights_path, cfg_path):

        super().__init__()
        device = 'cuda'
        self.scorenet, self.config = load_model(weights_path, device, cfg_path)
        self.sampler = get_readout_sampler(self.config)

    def forward(self, videos):
        input_frames = data_transform(self.config, videos)
        # Match precision with the model (scorenet)
        target_dtype = next(self.scorenet.parameters()).dtype
        input_frames = input_frames.to(dtype=target_dtype)

        expected_total = self.config.data.num_frames_cond + self.config.data.num_frames
        if expected_total > videos.shape[1]:
            added_frames = expected_total - videos.shape[1]
            input_frames = torch.cat(
                [input_frames] +
                [input_frames[:, -1].unsqueeze(1)] * added_frames,
                dim=1
            )

        chunk_size = self.config.data.num_frames_cond + self.config.data.num_frames
        num_total_frames = input_frames.shape[1]

        for j in range(0, num_total_frames, chunk_size):
            chunk = input_frames[:, j:j + chunk_size, :, :, :]

            # Pad the chunk if it's smaller than expected
            if chunk.shape[1] < chunk_size:
                pad_len = chunk_size - chunk.shape[1]
                last_frame = chunk[:, -1:].repeat(1, pad_len, 1, 1, 1)
                chunk = torch.cat([chunk, last_frame], dim=1)

            real, cond, cond_mask = conditioning_fn(
                self.config,
                chunk,
                num_frames_pred=self.config.data.num_frames,
                prob_mask_cond=getattr(
                    self.config.data, 'prob_mask_cond', 0.0),
                prob_mask_future=getattr(
                    self.config.data, 'prob_mask_future', 0.0)
            )

            # Match precision
            cond = cond.to(dtype=target_dtype)

            init = init_samples(len(real), self.config).to(dtype=target_dtype)

            with torch.inference_mode(), torch.amp.autocast('cuda'):
                pred, _, _, mid = self.sampler(
                    init, self.scorenet, cond=cond,
                    cond_mask=cond_mask,
                    subsample=100, verbose=False
                )

        return pred  # torch.stack(output, dim=1)
