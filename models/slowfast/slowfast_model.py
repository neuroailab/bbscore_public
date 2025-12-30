import torch
import torch.nn as nn

from mmaction.apis import init_recognizer
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose, pseudo_collate


class SLOWFAST_MODEL(nn.Module):
    def __init__(self, config, checkpoint, device="cuda"):
        super().__init__()
        self.model = init_recognizer(config, checkpoint, device)
        self.model.eval()
        init_default_scope(self.model.cfg.get('default_scope', 'mmaction'))

    def forward(self, inputs):
        data = inputs
        data["inputs"] = [d.to("cuda") for d in data["inputs"]]
        return self.model.test_step(data)[0]
