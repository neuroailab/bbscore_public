import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict
from einops import rearrange
from torch.functional import F


class DFM_MODEL(nn.Module):
    def __init__(self, weights_path):

        super().__init__()
        from phys_extractors.models.DFM_physion.PixelNeRF import PixelNeRFModelCond
        render_settings = {
            "n_coarse": 64,
            "n_fine": 64,
            "n_coarse_coarse": 32,
            "n_coarse_fine": 0,
            "num_pixels": 64 ** 2,
            "n_feats_out": 64,
            "num_context": 1,
            "sampling": "patch",
            "cnn_refine": False,
            "self_condition": False,
            "lindisp": False,
        }

        self.model = PixelNeRFModelCond(
            near=1.0,
            far=2,
            model='dit',
            use_first_pool=False,
            mode='cond',
            feats_cond=True,
            use_high_res_feats=True,
            render_settings=render_settings,
            use_viewdir=False,
            image_size=128,
            use_abs_pose=False,
        )

        if weights_path:
            print('Loading Weights')
            ckpt = torch.load(weights_path, map_location='cpu')
            self.model.load_state_dict(ckpt)
            print('Loaded')

    def forward(self, image):
        b, c, h, w = image.shape
        t = torch.zeros((b), device=image.device, dtype=torch.long)
        feature_map = self.model.get_feats(image, t, abs_camera_poses=None)
        return feature_map


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


class DFM_LSTM_MODEL(nn.Module):
    def __init__(self, weights_path):
        super().__init__()

        self.encoder = DFM_MODEL(None)
        self.n_past = 7

        dynamics_kwargs = {"latent_dim": 8192}
        self.dynamics = LSTM(**dynamics_kwargs)

    def forward(self, x):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        # x is (Bs, T, 3, H, W)
        observed_rollout_steps = max(1, x[:, self.n_past:].shape[1])

        b, num_context, c, h, w = x.shape

        x = rearrange(x, "b t h w c -> (b t) h w c")
        encoder_output = self.encoder(x)

        # To downscale it by a factor of 4, we are reducing the size of H and W
        # Calculate the new dimensions
        H_new = encoder_output.shape[-1] // 4
        W_new = encoder_output.shape[-2] // 4

        # Now, we will use the interpolate function from the torch.nn.functional module
        feature_map = F.interpolate(encoder_output, size=(H_new, W_new),
                                    mode='bilinear', align_corners=False)

        feature_map = feature_map.reshape(feature_map.shape[0], -1)

        feature_map = nn.AdaptiveAvgPool1d(8192)(feature_map.float().cpu())
        feature_map = feature_map.reshape(b, num_context, -1)

        if torch.cuda.is_available():
            feature_map = feature_map.cuda()

        dynamics_output = self.dynamics(feature_map,
                                        observed_rollout_steps,
                                        self.n_past)

        return dynamics_output


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
