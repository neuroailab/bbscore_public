import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import moten
from moten import core


class MotionEnergyModule(nn.Module):
    """
    A PyTorch nn.Module wrapper for the Moten pyramid.

    Optimized for batch processing. Matches moten's exact filter alignment,
    padding logic, spatial masking, and double precision computation to 
    ensure numerical equivalence.
    """

    def __init__(self, pyramid):
        super().__init__()
        self.nfilters = pyramid.nfilters
        self.projection = nn.Identity()

        # --- 1. Pre-compute filters ---
        s_sin_weights = []
        s_cos_weights = []
        t_sin_weights = []
        t_cos_weights = []

        self.temporal_width = pyramid.definition.filter_temporal_width

        # Moten hardcodes a threshold to zero out small spatial values
        mask_threshold = 0.001

        for i in range(self.nfilters):
            # These returns are typically float64
            sg0, sg90, tg0, tg90 = pyramid.get_filter_spatiotemporal_quadratures(
                i)

            # --- CRITICAL FIX 1: Spatial Masking ---
            # Moten zeros out spatial weights where the amplitude is low.
            # We must replicate this mask before conversion.
            mask = (np.abs(sg0) + np.abs(sg90)) > mask_threshold
            sg0[~mask] = 0.0
            sg90[~mask] = 0.0

            s_sin_weights.append(sg0)
            s_cos_weights.append(sg90)
            t_sin_weights.append(tg0)
            t_cos_weights.append(tg90)

        # --- 2. Convert to PyTorch Tensors (Double Precision) ---
        # We use .double() to match moten's internal numpy precision (float64).
        # Computing in float32 results in ~1e-4 error; float64 reduces this to ~1e-7 or less.

        # Spatial: (N_filters, 1, H, W)
        w_s_sin = torch.from_numpy(
            np.stack(s_sin_weights)).double().unsqueeze(1)
        w_s_cos = torch.from_numpy(
            np.stack(s_cos_weights)).double().unsqueeze(1)

        # Temporal: (N_filters, 1, T_kernel)
        w_t_sin = torch.from_numpy(
            np.stack(t_sin_weights)).double().unsqueeze(1)
        w_t_cos = torch.from_numpy(
            np.stack(t_cos_weights)).double().unsqueeze(1)

        # --- CRITICAL FIX 2: Temporal Flip ---
        # Moten's filter definition: Index 0 is Future (t+N), Index N is Past (t-N).
        # PyTorch Conv1d (Correlation): Index 0 aligns with t, Index N aligns with t+N.
        # To match the "convolution" behavior (accessing past), we must flip the weights.
        w_t_sin = torch.flip(w_t_sin, dims=[-1])
        w_t_cos = torch.flip(w_t_cos, dims=[-1])

        self.register_buffer('w_s_sin', w_s_sin)
        self.register_buffer('w_s_cos', w_s_cos)
        self.register_buffer('w_t_sin', w_t_sin)
        self.register_buffer('w_t_cos', w_t_cos)

        self.epsilon = 1e-05

        # --- 3. Pre-calculate Asymmetric Padding ---
        # Moten centers the filter at index: ceil(width/2) - 1 (0-based).
        # Example Width 16: Center Index = 7.
        # After flipping the kernel, the center moves to: (Width - 1) - 7 = 8.
        # In Conv1d, Kernel[0] aligns with Input[t]. We want Kernel[8] to align with Input[t].
        # Therefore, we need to Pad Left by 8 so that effective t is shifted.

        center_idx_moten = int(np.ceil(self.temporal_width / 2.0)) - 1
        flip_center_idx = (self.temporal_width - 1) - center_idx_moten

        self.pad_left = flip_center_idx
        self.pad_right = (self.temporal_width - 1) - self.pad_left

    def forward(self, x):
        """
        Args:
            x (torch.Tensor or np.ndarray): (Batch, Time, Height, Width)
        """
        # 1. Prepare Input
        if not isinstance(x, torch.Tensor):
            device = self.w_s_sin.device
            x = torch.from_numpy(x).to(device)

        # Ensure we compute in Double Precision to match Moten
        x = x.double()

        if x.ndim == 3:
            x = x.unsqueeze(0)

        B, T, H, W = x.shape

        # 2. Spatial Convolution (Global Dot Product)
        # Reshape to (Batch*Time, 1, H, W)
        x_spatial = x.view(B * T, 1, H, W)

        # F.conv2d with double weights + double input -> double output
        g_sin = F.conv2d(x_spatial, self.w_s_sin).view(B, T, self.nfilters)
        g_cos = F.conv2d(x_spatial, self.w_s_cos).view(B, T, self.nfilters)

        # 3. Temporal Convolution
        # Reshape to (Batch, Channels, Time)
        g_sin = g_sin.permute(0, 2, 1)
        g_cos = g_cos.permute(0, 2, 1)

        # Apply Asymmetric Padding to Time dimension
        # F.pad tuple format for 3D input is (pad_left, pad_right) for the last dimension
        g_sin_pad = F.pad(g_sin, (self.pad_left, self.pad_right))
        g_cos_pad = F.pad(g_cos, (self.pad_left, self.pad_right))

        # Convolve (S_sin * T_cos + S_cos * T_sin, etc.)
        # Groups=nfilters ensures channel-wise convolution (Depthwise)
        term1 = F.conv1d(g_sin_pad, self.w_t_cos, groups=self.nfilters)
        term2 = F.conv1d(g_cos_pad, self.w_t_sin, groups=self.nfilters)
        term3 = F.conv1d(g_cos_pad, self.w_t_cos, groups=self.nfilters)
        term4 = F.conv1d(g_sin_pad, self.w_t_sin, groups=self.nfilters)

        # 4. Combine
        resp_sin = term1 + term2
        resp_cos = term3 - term4

        # Calculate Energy
        energy = torch.sqrt(resp_sin**2 + resp_cos**2)
        energy = torch.log(energy + self.epsilon)

        # Permute to (Batch, Time, Features)
        out = energy.permute(0, 2, 1)

        # Cast back to float32 for downstream tasks (standard PyTorch behavior)
        return self.projection(out).float()


class MotionEnergyLoader:
    def __init__(self):
        self.static = False
        self.frame_size = (224, 224)
        self.fps = 25
        self.MODEL_ID = "motion-energy"

    def preprocess_fn(self, input_data):
        if isinstance(input_data, list):
            # (T, H, W, C) -> (T, H, W)
            frames = [np.array(img.convert("RGB").resize(
                self.frame_size)) for img in input_data]
            frames = np.stack(frames, axis=0)
        elif isinstance(input_data, np.ndarray):
            frames = input_data
            if frames.ndim == 3:
                frames = frames[np.newaxis, ...]
        else:
            raise ValueError(
                "Input must be a list of PIL Images or a numpy array.")

        # Convert to luminance
        luminance = moten.io.imagearray2luminance(frames, size=self.frame_size)
        return luminance

    def get_model(self, identifier=None):
        if identifier and identifier != self.MODEL_ID:
            raise ValueError(f"Unknown identifier: {identifier}")

        pyramid = moten.get_default_pyramid(
            vhsize=self.frame_size, fps=self.fps)
        model = MotionEnergyModule(pyramid)
        model.eval()
        return model

    def postprocess_fn(self, features_np):
        if isinstance(features_np, torch.Tensor):
            features_np = features_np.detach().cpu().numpy()
        features_np = features_np.squeeze(1)
        return features_np
