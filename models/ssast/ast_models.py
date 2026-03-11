# -*- coding: utf-8 -*-
# @Time    : 7/16/21 3:12 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import timm
from timm.models.layers import to_2tuple, trunc_normal_
import numpy as np

# Override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        # return: (B, N, E)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class ASTModel(nn.Module):
    def __init__(self, label_dim=527, fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024, model_size='base',
                 pretrain_stage=True, load_pretrained_mdl_path=None):

        super(ASTModel, self).__init__()
        
        # Override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # Pretrain the AST models
        if pretrain_stage == True:
            if load_pretrained_mdl_path != None:
                raise ValueError('Setting load_pretrained_mdl_path at pretraining stage is useless.')
            if fstride != fshape or tstride != tshape:
                raise ValueError('fstride != fshape or tstride != tshape, they must be same at the pretraining stage')

            if model_size == 'tiny':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 3, 12
                self.cls_token_num = 2
            elif model_size == 'small':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 6, 12
                self.cls_token_num = 2
            elif model_size == 'base':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == 'base_nokd':
                self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception('Model size must be one of tiny, small, base, base_nokd')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            
            self.softmax = nn.Softmax(dim=-1)
            self.lsoftmax = nn.LogSoftmax(dim=-1)
            self.fshape, self.tshape = fshape, tshape
            self.fstride, self.tstride = fstride, tstride
            self.input_fdim, self.input_tdim = input_fdim, input_tdim
            self.p_input_fdim, self.p_input_tdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False), nn.Parameter(torch.tensor(input_tdim), requires_grad=False)

            self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))

            self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
            self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

            self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = self.p_f_dim * self.p_t_dim
            self.num_patches = num_patches
            self.v.patch_embed.num_patches = num_patches
            
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
            self.v.patch_embed.proj = new_proj

            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

        elif pretrain_stage == False:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if load_pretrained_mdl_path == None:
                raise ValueError('Please set load_pretrained_mdl_path to load a pretrained models.')
            
            sd = torch.load(load_pretrained_mdl_path, map_location=device)
            try:
                p_fshape, p_tshape = sd['module.v.patch_embed.proj.weight'].shape[2], sd['module.v.patch_embed.proj.weight'].shape[3]
                p_input_fdim, p_input_tdim = sd['module.p_input_fdim'].item(), sd['module.p_input_tdim'].item()
            except:
                if 'v.patch_embed.proj.weight' in sd:
                    p_fshape, p_tshape = sd['v.patch_embed.proj.weight'].shape[2], sd['v.patch_embed.proj.weight'].shape[3]
                    p_input_fdim, p_input_tdim = sd['p_input_fdim'].item(), sd['p_input_tdim'].item()
                else: 
                    # If all fails, assume defaults for ssast-base-frame-400
                    print("Warning: Could not infer shapes from state_dict, assuming ssast-base-frame-400 defaults (128x2).")
                    p_fshape, p_tshape = 128, 2
                    p_input_fdim, p_input_tdim = 128, 1024

            print('now load a SSL pretrained models from ' + load_pretrained_mdl_path)
            
            audio_model = ASTModel(fstride=p_fshape, tstride=p_tshape, fshape=p_fshape, tshape=p_tshape,
                                   input_fdim=p_input_fdim, input_tdim=p_input_tdim, pretrain_stage=True, model_size=model_size)
            
            # Handle DataParallel wrapping
            state_dict_keys = list(sd.keys())
            if state_dict_keys and state_dict_keys[0].startswith('module.'):
                audio_model = torch.nn.DataParallel(audio_model)
                audio_model.load_state_dict(sd, strict=False)
                self.v = audio_model.module.v
                self.original_embedding_dim = self.v.pos_embed.shape[2]
                self.cls_token_num = audio_model.module.cls_token_num
                p_f_dim, p_t_dim = audio_model.module.p_f_dim, audio_model.module.p_t_dim
            else:
                audio_model.load_state_dict(sd, strict=False)
                self.v = audio_model.v
                self.original_embedding_dim = self.v.pos_embed.shape[2]
                self.cls_token_num = audio_model.cls_token_num
                p_f_dim, p_t_dim = audio_model.p_f_dim, audio_model.p_t_dim

            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = f_dim * t_dim
            p_num_patches = p_f_dim * p_t_dim
            self.v.patch_embed.num_patches = num_patches
            
            if fshape != p_fshape or tshape != p_tshape:
                raise ValueError(f'The patch shape of pretraining and fine-tuning is not consistant: p_f={p_fshape}, p_t={p_tshape}, f={fshape}, t={tshape}')

            if fstride != p_fshape or tstride != p_tshape:
                new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
                self.v.patch_embed.proj = new_proj

            new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach().reshape(1, p_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim)
            if t_dim < p_t_dim:
                new_pos_embed = new_pos_embed[:, :, :, int(p_t_dim/2) - int(t_dim / 2): int(p_t_dim/2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear') 
                # Note: '8' corresponds to p_f_dim for default 16x16 patch? 
                # But here we use Frame-based (128x2). Input fdim=128. fstride=128. So f_dim = 1.
                # If f_dim=1, using '8' is wrong if p_f_dim=1.
                # Let's check p_f_dim for frame-based 128x2.
                # p_f_dim = input_fdim // fstride = 128 // 128 = 1.
                # So we should interpolate to (f_dim, t_dim).
                
            # Correct logic:
            if f_dim < p_f_dim:
                new_pos_embed = new_pos_embed[:, :, int(p_f_dim/2) - int(f_dim / 2): int(p_f_dim/2) - int(f_dim / 2) + t_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))
            
            self.fshape = fshape
            self.tshape = tshape
            self.fstride = fstride
            self.tstride = tstride
            self.input_fdim = input_fdim
            self.input_tdim = input_tdim


    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def finetuningavgtok(self, x):
        B = x.shape[0]
        # x is (B, 1, F, T) -> patch_embed -> (B, N, D)
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Dynamic variable-length support
        current_len = x.shape[1]
        target_len = self.v.pos_embed.shape[1]
        
        if current_len != target_len:
             pos_embed = self.v.pos_embed # (1, M, D)
             cls_pos = pos_embed[:, :self.cls_token_num, :]
             patch_pos = pos_embed[:, self.cls_token_num:, :] # (1, M-cls, D)
             
             # Interpolate strictly on the N tokens dimension. 
             # Since it's flattened, this treats it as 1D sequence for interpolation.
             # (1, M-cls, D) -> (1, D, M-cls)
             patch_pos = patch_pos.transpose(1, 2) 
             patch_pos = torch.nn.functional.interpolate(patch_pos, size=(current_len - self.cls_token_num), mode='linear', align_corners=False)
             patch_pos = patch_pos.transpose(1, 2) # (1, current-cls, D)
             
             pos_embed = torch.cat((cls_pos, patch_pos), dim=1)
             x = x + pos_embed
        else:
             x = x + self.v.pos_embed

        x = self.v.pos_drop(x)

        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        return x

    def forward(self, x, task='ft_avgtok'):
        # expect input x = (batch_size, time_frame_num, frequency_bins)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3) # (B, 1, F, T)

        if task == 'ft_avgtok':
            return self.finetuningavgtok(x)
        return self.finetuningavgtok(x)
