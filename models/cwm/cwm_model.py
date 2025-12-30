import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
from typing import Tuple, Union, List, Dict
import tqdm
import random

from ccwm.cwm.cwm_predictor import CWMPredictor
from ccwm.cwm.utils.image_processing import create_images_from_patches, patchify_image, unpatchify_image
from ccwm.cwm.utils.sequence_construction import (
    get_pos_idxs, shuffle_and_trim_values_and_positions
)

from ccwm.utils.sequence_construction import get_rope_pos_idxs
from ccwm.utils.model_wrapper import ModelFactory


class CWM_MODEL(nn.Module):
    def __init__(self, model_name, mask_ratio):
        super().__init__()
        self.device = 'cuda'
        self.ctx = torch.amp.autocast(
            device_type='cuda' if 'cuda' in self.device else 'cpu', dtype=torch.bfloat16)
        self.predictor = CWMPredictor(model_name=model_name)
        self.model = self.predictor.model
        self.in_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.inv_in_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
            torchvision.transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            torchvision.transforms.ToPILImage()
        ])
        self.mask_ratio = mask_ratio

    def resize_crop_transform(self, image, resolution):
        resize_crop_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resolution),
            torchvision.transforms.CenterCrop(resolution),
        ])
        return resize_crop_transform(image)

    def forward(self, x):
        B, T, C, H, W = x.shape

        batch_predictions = []
        for i in range(B):
            video_tensor = x[i]

            results = self.forward_step(
                video_tensor[0],
                video_tensor[-1],
                frame_gap=T - 1,
                mask_ratio=self.mask_ratio
            )

            batch_predictions.append(results)

        output = torch.cat(batch_predictions, dim=0)
        return output

    def forward_step(self,
                     frame0: Union[Image.Image, np.ndarray],
                     frame1: Union[Image.Image, np.ndarray],
                     frame_gap: int,
                     mask_ratio: float = 0.9,
                     unmask_indices: List[int] = None,
                     seed: int = 0,
                     **kwargs):

        self._set_seed(seed)
        num_patches_per_image = int(self.model.config.block_size / 2)
        resolution = self.model.config.resolution

        if unmask_indices is None:
            num_unmasked_patches = int(
                num_patches_per_image * (1 - max(0, mask_ratio)))
            unmask_indices = np.random.choice(
                num_patches_per_image, num_unmasked_patches, replace=False).tolist()

        # frame0: [1, 256, 256, 3]
        frame0 = frame0.unsqueeze(0).permute(0, 2, 3, 1)

        # Patchify frame0
        # patches0: [1, 1024, 192=8*8*3]
        patches0 = patchify_image(frame0, self.model.config.patch_size)

        if mask_ratio >= 0:
            frame1 = frame1.unsqueeze(0).permute(0, 2, 3, 1)
            patches1 = patchify_image(frame1, self.model.config.patch_size)
            mask_indices = [i for i in list(
                range(num_patches_per_image)) if i not in unmask_indices]
            patches1[:, mask_indices, :] = 0.0

            # seq: (2048, 192)
            seq = torch.cat([
                patches0.reshape(-1, patches0.shape[-1]),
                patches1.reshape(-1, patches1.shape[-1])])
        else:
            seq = patches0.reshape(-1, patches0.shape[-1])

        if 'CWM2' in self.model.config.model_class:
            # patches_0_dummy: [1, 1024, 1]. First create a dummy tensor to pass to get_rope_pos_idxs
            patches_0_dummy = torch.zeros_like(patches0[:, :, :1])

            # patches_0_pos_idx: [1, 1024, 1, 4]
            patches_0_pos_idx = get_rope_pos_idxs(patches_0_dummy, 0, 0)

            # pos: (2048, 4)
            if mask_ratio >= 0:
                patches_1_dummy = torch.zeros_like(patches1[:, :, :1])
                patches_1_pos_idx = get_rope_pos_idxs(
                    patches_1_dummy, frame_gap, 0)

                pos = torch.cat([
                    patches_0_pos_idx.reshape(-1, patches_0_pos_idx.shape[-1]),
                    patches_1_pos_idx.reshape(-1, patches_1_pos_idx.shape[-1])],
                    dim=0
                )
            else:
                pos = patches_0_pos_idx.reshape(-1,
                                                patches_0_pos_idx.shape[-1])

            mask_value = 1
            pos_mask_indices = [
                m + num_patches_per_image for m in mask_indices]
            pos[pos_mask_indices, 3] = mask_value

            # mask is not used for CWM2
            mask = torch.zeros(pos.shape[0]).float()

        else:
            # Define positional indexes for sequence
            # patches_0_pos_idx: [1, 1024]
            patches_0_pos_idx = get_pos_idxs(patches0, 0)

            # pos: (2048,)
            if mask_ratio >= 0:
                patches_1_pos_idx = get_pos_idxs(
                    patches1, num_patches_per_image)
                pos = torch.cat([patches_0_pos_idx.reshape(-1).long(),
                                 patches_1_pos_idx.reshape(-1).long()])
            else:
                pos = patches_0_pos_idx.reshape(-1).long()

            mask = torch.ones_like(pos).float()
            mask[:num_patches_per_image] = 0.0  # unmask frame0
            if mask_ratio >= 0:
                indices_to_unmask = [num_patches_per_image +
                                     idx for idx in unmask_indices]
                mask[indices_to_unmask] = 0.0

        with self.ctx:
            patches, _ = self.model(seq.unsqueeze(0).to(self.device), pos.unsqueeze(
                0).to(self.device), tgt=None, mask=mask.unsqueeze(0).to(self.device))
        patches = patches.detach().cpu().float()

        return patches

    def _set_seed(self, seed: int):
        """
        Set the seed for reproducibility.

        Parameters:
            seed: int, the seed to set
        """
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
