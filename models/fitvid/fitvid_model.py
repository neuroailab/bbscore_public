import torch
import torch.nn as nn
from collections import OrderedDict
from phys_extractors.models.FitVid import fitvid

# Given sequence of images, predicts next latent


class FitVidEncoder(nn.Module):
    def __init__(self, weights_path, n_past=7):
        super().__init__()
        self.n_past = n_past
        self.model = fitvid.FitVid(n_past=n_past, train=False)
        params = torch.load(weights_path, map_location="cpu")
        new_sd = OrderedDict()
        for k, v in params.items():
            name = k[7:] if k.startswith("module.") else k
            new_sd[name] = v
        self.model.load_state_dict(new_sd)
        print(f"Loaded parameters from {weights_path}")

    def forward(self, x):
        with torch.no_grad():
            output = self.model(x, n_past=x.shape[1])
        features = output['h_preds']
        return features
