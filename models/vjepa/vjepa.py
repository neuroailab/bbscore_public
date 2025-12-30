import os
import urllib.request
from sklearn.datasets import get_data_home
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import phys_extractors.models.jepa_physics.jepa.src.models.vision_transformer as vit

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class VJEPA:
    """Loads and runs pre-trained Video Joint-Embedding Predictive Architecture (V-JEPA) models with optimized preprocessing, compilation, mixed-precision, and JIT tracing for maximum inference speed."""

    def __init__(self):
        """Initializes the V-JEPA loader, sets up transforms, device, and caches, and enables CUDNN optimizations."""
        # Model mappings
        self.model_mappings = {
            "VJEPA-LARGE": "https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar",
            "VJEPA-HUGE": "https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar",
        }

        # Video and ViT parameters
        self.num_frames = 16
        self.fps = 10
        self.crop_size = 224
        self.patch_size = 16
        self.tubelet_size = 2
        self.uniform_power = True
        self.use_sdpa = True
        self.use_SiLU = False
        self.tight_SiLU = False

        # Normalization constants
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.MEAN = torch.tensor(mean).view(3, 1, 1, 1)
        self.STD = torch.tensor(std).view(3, 1, 1, 1)

        # Device setup
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Simple PIL-to-tensor transform
        self._to_tensor = transforms.ToTensor()
        self.static = False

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocess frames into [C, T, H, W] tensor, cast to half precision on GPU.
        """
        # Build [T, C, H, W]
        if isinstance(input_data, Image.Image):
            frames = self._to_tensor(input_data).unsqueeze(0)
            orig_fps = fps or self.fps
        else:
            tensors = []
            for img in input_data:
                if isinstance(img, Image.Image):
                    tensors.append(self._to_tensor(img))
                else:
                    arr = torch.from_numpy(
                        img).float().permute(2, 0, 1).div(255.0)
                    tensors.append(arr)
            frames = torch.stack(tensors, dim=0)
            orig_fps = fps or self.fps

        # Temporal sampling
        if orig_fps and fps and orig_fps != fps:
            T = frames.shape[0]
            new_T = max(int(round(T / orig_fps * fps)), 1)
            idx = torch.linspace(0, T - 1, new_T).long()
            frames = frames[idx]
        T = frames.size(0)
        if T > self.num_frames:
            idx = torch.linspace(0, T - 1, self.num_frames).long()
            frames = frames[idx]
        elif T < self.num_frames:
            pad = frames[-1:].expand(self.num_frames - T, -1, -1, -1)
            frames = torch.cat([frames, pad], dim=0)

        # Resize and normalize
        frames = F.interpolate(frames, size=(self.crop_size, self.crop_size),
                               mode='bilinear', align_corners=False)
        # Reorder to [C, T, H, W]
        frames = frames.permute(1, 0, 2, 3)
        # Normalize
        frames = (frames - self.MEAN) / self.STD
        # Mixed precision
        if self.device.type == 'cuda':
            frames = frames.half()
        return frames

    def get_model(self, identifier):
        """
        Loads (cached) model, applies half precision, compiles/JIT traces, and wraps inference in no_grad+autocast.
        """
        # URL lookup and download
        model_url = next((url for prefix, url in self.model_mappings.items(
        ) if identifier.startswith(prefix)), None)
        if not model_url:
            raise ValueError(f"Unknown model identifier: {identifier}")
        weights_dir = os.path.join(
            get_data_home(), 'weights', self.__class__.__name__)
        os.makedirs(weights_dir, exist_ok=True)
        path = os.path.join(weights_dir, os.path.basename(model_url))
        if not os.path.isfile(path):
            urllib.request.urlretrieve(model_url, path)

        # Load and build
        state = torch.load(path, map_location='cpu')
        model = vit.vit_large(img_size=self.crop_size, patch_size=self.patch_size,
                              num_frames=self.num_frames, tubelet_size=self.tubelet_size,
                              uniform_power=self.uniform_power, use_sdpa=self.use_sdpa,
                              use_SiLU=self.use_SiLU, tight_SiLU=self.tight_SiLU)
        model.load_state_dict({k.replace(
            'module.backbone.', ''): v for k, v in state['encoder'].items()}, strict=False)
        model.eval()
        # Device and half
        model.to(self.device)
        if self.device.type == 'cuda':
            model.half()

        # Freeze
        for p in model.parameters():
            p.requires_grad = False

        # Compile and/or JIT trace
        try:
            model = torch.compile(model)
        except Exception:
            pass
        if self.device.type == 'cuda':
            # JIT trace once with dummy input
            dummy = torch.randn(1, 3, self.num_frames, self.crop_size,
                                self.crop_size, device=self.device).half()
            try:
                model = torch.jit.trace(model, dummy, strict=False).eval()
            except Exception:
                pass

        # Wrap forward for optimized inference
        orig_forward = model.forward

        def forward_opt(x):
            with torch.inference_mode():
                if self.device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        return orig_forward(x)
                return orig_forward(x)
        model.forward = forward_opt

        return model

    def postprocess_fn(self, features_np):
        """Flattens features to shape (batch, num_frames, feat_dim)."""
        bs = features_np.shape[0]
        features_np = features_np.squeeze(1)
        return features_np.reshape(bs, self.num_frames, -1)
