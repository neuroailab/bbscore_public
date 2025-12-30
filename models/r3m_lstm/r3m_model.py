import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict
from r3m import load_r3m


def load_model(
    model, model_path, state_dict_key="state_dict"
):
    params = torch.load(model_path, map_location="cpu")
    sd = params

    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module.") and 'r3m' in k:
            # remove 'module.' of dataparallel/DDP
            name = 'encoder.r3m.module.' + k[19:]
        elif k.startswith("module.") and 'dynamics' in k:
            name = k[7:]
        elif k.startswith("module."):
            name = k[7:]
        else:
            name = k
        new_sd[name] = v
    model.load_state_dict(new_sd)
    print(f"Loaded parameters from {model_path}")

    return model


class R3M_pretrained(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        self.r3m = load_r3m(model_name)
        self.latent_dim = 2048  # resnet50 final fc in_features

    def get_encoder_feats(self, x):
        feats = self.r3m(x * 255.0)
        feats = torch.flatten(feats, start_dim=1)  # (Bs, -1)
        return feats

    def forward(self, videos, n_past=None):
        bs, num_frames, num_channels, h, w = videos.shape
        videos = videos.flatten(0, 1)
        input_states = self.get_encoder_feats(videos)
        input_states = input_states.reshape(bs, num_frames, -1)
        return input_states


class LSTM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(self.latent_dim, 1024, batch_first=True)
        self.regressor = nn.Linear(1024, self.latent_dim)

    def forward_step(self, feats):
        assert feats.ndim == 3
        # note: for lstms, hidden is the last timestep output
        _, hidden = self.lstm(feats)
        # assumes n_layers=1
        x = torch.squeeze(hidden[0].permute(1, 0, 2), dim=1)
        x = self.regressor(x)
        return x

    def forward(self, observed_encoder_states, rollout_steps, n_context):
        observed_dynamics_states = []
        prev_states = observed_encoder_states
        for step in range(rollout_steps):
            # dynamics model predicts next latent from past latents
            prev_states_ = prev_states[:, step:step+n_context]
            pred_state = self.forward_step(prev_states_)
            observed_dynamics_states.append(pred_state)

        return torch.stack(observed_dynamics_states, axis=1)


# Given sequence of images, predicts next latent
class R3M_LSTM_MODEL(nn.Module):
    def __init__(self, n_past=7):
        super().__init__()
        self.n_past = n_past
        self.encoder = R3M_pretrained()
        dynamics_kwargs = {"latent_dim": self.encoder.latent_dim}
        self.dynamics = LSTM(**dynamics_kwargs)

    def forward(self, x, n_past=None):
        observed_rollout_steps = max(1, x[:, self.n_past:].shape[1])
        encoder_output = self.encoder(x, self.n_past)
        dynamics_output = self.dynamics(encoder_output,
                                        observed_rollout_steps,
                                        self.n_past)

        return dynamics_output
