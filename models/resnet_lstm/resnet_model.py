import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict
from transformers import ResNetModel


def load_model(
    model, model_path, state_dict_key="state_dict"
):
    params = torch.load(model_path, map_location="cpu")
    sd = params

    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module.") and 'dynamics' in k:
            name = k[7:]
        elif k.startswith("module."):
            name = k[7:]
        else:
            name = k
        new_sd[name] = v
    model.load_state_dict(new_sd)
    print(f"Loaded parameters from {model_path}")

    return model


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class ResNet50(nn.Module):
    def __init__(self):

        super().__init__()
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.latent_dim = 8192

    def get_encoder_feats(self, images):
        '''
        images: [B, C, H, W], Image is normalized with imagenet norm
        '''
        input_dict = {'pixel_values': images}

        decoder_outputs = self.model(**input_dict, output_hidden_states=True)

        features = decoder_outputs.last_hidden_state

        features = features.reshape(features.shape[0], -1)

        features = nn.AdaptiveAvgPool1d(8192)(features.float())

        return features

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
class RESNET_LSTM_MODEL(nn.Module):
    def __init__(self, n_past=7):
        super().__init__()
        self.n_past = n_past
        self.encoder = ResNet50()
        dynamics_kwargs = {"latent_dim": self.encoder.latent_dim}
        self.dynamics = LSTM(**dynamics_kwargs)

    def forward(self, x, n_past=None):
        observed_rollout_steps = max(1, x[:, self.n_past:].shape[1])
        encoder_output = self.encoder(x, self.n_past)
        dynamics_output = self.dynamics(encoder_output,
                                        observed_rollout_steps,
                                        self.n_past)

        return dynamics_output
