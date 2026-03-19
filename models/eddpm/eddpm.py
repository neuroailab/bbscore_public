"""
ED-DPM 256x256 classifier wrapper for BBScore feature extraction.

Exposes two model identifiers:
    eddpm_classifier_256      -- base 256x256 classifier checkpoint
    eddpm_classifier_256_ect  -- 256x256 classifier + 0.1 ECT checkpoint

Tap layers available for BBScore hooks:
    tap_first_conv_256, tap_down_64_end, tap_down_32_end,
    tap_down_16_end, tap_down_8_end, tap_middle_8, tap_logits
"""

import hashlib
import os
from pathlib import Path
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Lazy loader for the external ED-DPM package.
# Resolved via $EDDPM_ROOT or fallback to ../../../ED-DPM relative to this
# file (i.e. <monorepo>/ED-DPM alongside <monorepo>/bbscore_public).
# Deferred so that BBScore can import the model registry without requiring
# the ED-DPM source tree to be present.
# ---------------------------------------------------------------------------
_eddpm_cache = {}


def _load_eddpm_deps():
    """Import the external ED-DPM symbols on first use and cache them."""
    if _eddpm_cache:
        return _eddpm_cache

    root = os.environ.get("EDDPM_ROOT")
    if root is None:
        root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "..", "ED-DPM")
        )
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)

    try:
        from entropy_driven_guided_diffusion.script_util import (
            create_classifier,
            classifier_defaults,
        )
        from entropy_driven_guided_diffusion.nn import timestep_embedding
    except ImportError as exc:
        raise ImportError(
            "Could not import the ED-DPM package. "
            "Set EDDPM_ROOT to the path containing "
            "entropy_driven_guided_diffusion/, or install ED-DPM "
            "(pip install -e path/to/ED-DPM)."
        ) from exc

    _eddpm_cache["create_classifier"] = create_classifier
    _eddpm_cache["classifier_defaults"] = classifier_defaults
    _eddpm_cache["timestep_embedding"] = timestep_embedding
    return _eddpm_cache


# ---------------------------------------------------------------------------
# Checkpoint definitions and resolution helpers
# ---------------------------------------------------------------------------
_CHECKPOINT_SOURCES = {
    "eddpm_classifier_256": {
        "filename": "256x256_classifier.pt",
        "legacy_filenames": ("256x256-classifier.pt",),
        "url": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt",
        "gdrive_id": None,
        "sha256": None,
    },
    "eddpm_classifier_256_ect": {
        "filename": "256x256_classifier+0.1ECT.pt",
        "legacy_filenames": (),
        "url": None,
        "gdrive_id": "1eiM62KKa6hdoTSWc1LtWwBfs9BXH4fJ_",
        "sha256": None,
    },
}

_TARGET_RESOLUTIONS = {64, 32, 16, 8}


def _weights_dir():
    """Return the persistent ED-DPM cache directory under sklearn data home."""
    try:
        from sklearn.datasets import get_data_home
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required to resolve the ED-DPM checkpoint cache path. "
            "Install scikit-learn or set EDDPM_CHECKPOINT_DIR."
        ) from exc
    path = Path(get_data_home()) / "weights" / "eddpm"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checksum(path, expected_sha256):
    if not expected_sha256:
        return
    got = _sha256(path)
    if got != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch for checkpoint '{path}': expected {expected_sha256}, got {got}"
        )


def _download_http(url, dst):
    tmp_path = Path(f"{dst}.tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp_path))
        os.replace(tmp_path, dst)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download checkpoint from URL: {url}") from exc


def _download_gdrive(file_id, dst):
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required to download the ED-DPM ECT checkpoint. "
            "Install gdown or provide EDDPM_CHECKPOINT_DIR."
        ) from exc
    url = f"https://drive.google.com/uc?id={file_id}"
    tmp_path = Path(f"{dst}.tmp")
    try:
        downloaded = gdown.download(url=url, output=str(tmp_path), quiet=False)
        if downloaded is None:
            raise RuntimeError("gdown returned no output path")
        os.replace(tmp_path, dst)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Failed to download checkpoint from Google Drive id: {file_id}"
        ) from exc


def _safe_load_state_dict(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    return state


def _ensure_checkpoint(identifier):
    """Return a local checkpoint path, downloading into cache if needed."""
    source = _CHECKPOINT_SOURCES.get(identifier)
    if source is None:
        raise ValueError(
            f"Unknown identifier: {identifier}. Available: {list(_CHECKPOINT_SOURCES)}"
        )

    filenames = (source["filename"], *source.get("legacy_filenames", ()))

    env_dir = os.environ.get("EDDPM_CHECKPOINT_DIR")
    if env_dir:
        base = Path(env_dir).expanduser()
        for filename in filenames:
            candidate = base / filename
            if candidate.is_file():
                _validate_checksum(candidate, source.get("sha256"))
                return str(candidate)

    cache_dir = _weights_dir()
    for filename in filenames:
        candidate = cache_dir / filename
        if candidate.is_file():
            _validate_checksum(candidate, source.get("sha256"))
            return str(candidate)

    checkpoint_path = cache_dir / source["filename"]
    if source.get("url"):
        _download_http(source["url"], checkpoint_path)
    elif source.get("gdrive_id"):
        _download_gdrive(source["gdrive_id"], checkpoint_path)
    else:
        raise RuntimeError(
            f"No download source configured for ED-DPM checkpoint: {identifier}"
        )

    _validate_checksum(checkpoint_path, source.get("sha256"))
    return str(checkpoint_path)


# ---------------------------------------------------------------------------
# Tapped classifier wrapper
# ---------------------------------------------------------------------------
class EDDPMTappedClassifier(nn.Module):
    """Wraps the ED-DPM EncoderUNetModel classifier with named Identity
    tap points so that BBScore's FeatureExtractor can hook them by name.

    The forward pass replicates the proven logic from nsd-eddpm.py:
    walk through input_blocks, capture the *last* output at each target
    spatial resolution, run middle_block, then compute logits.
    """

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.tap_first_conv_256 = nn.Identity()
        self.tap_down_64_end = nn.Identity()
        self.tap_down_32_end = nn.Identity()
        self.tap_down_16_end = nn.Identity()
        self.tap_down_8_end = nn.Identity()
        self.tap_middle_8 = nn.Identity()
        self.tap_logits = nn.Identity()

    def forward(self, x, timesteps=None):
        if timesteps is None:
            timesteps = torch.zeros(
                x.shape[0], device=x.device, dtype=torch.long)

        timestep_embedding = _eddpm_cache["timestep_embedding"]
        emb = self.classifier.time_embed(
            timestep_embedding(timesteps, self.classifier.model_channels)
        )

        cast_dtype = self.classifier.dtype
        out_dtype = x.dtype
        h = x.type(cast_dtype)

        last_at_res: dict[int, torch.Tensor] = {}

        for idx, module in enumerate(self.classifier.input_blocks):
            h = module(h, emb)
            spatial = h.shape[-1]

            if idx == 0:
                self.tap_first_conv_256(h.type(out_dtype))

            if spatial in _TARGET_RESOLUTIONS:
                last_at_res[spatial] = h.type(out_dtype)

        self.tap_down_64_end(last_at_res[64])
        self.tap_down_32_end(last_at_res[32])
        self.tap_down_16_end(last_at_res[16])
        self.tap_down_8_end(last_at_res[8])

        h = self.classifier.middle_block(h, emb)
        self.tap_middle_8(h.type(out_dtype))

        logits = self.classifier.out(h.type(out_dtype))
        return self.tap_logits(logits)


# ---------------------------------------------------------------------------
# BBScore model wrapper
# ---------------------------------------------------------------------------
class EDDPMModelWrapper:
    """BBScore-compatible wrapper for the ED-DPM 256x256 classifier family."""

    def __init__(self):
        self.processor = transforms.Compose([
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.static = True

    def preprocess_fn(self, input_data, fps=None):
        """Transform raw input into a preprocessed tensor.

        Returns (3, 256, 256) for a single image so the DataLoader can batch
        correctly, or (B, 3, 256, 256) when given a list of images.
        """
        if isinstance(input_data, str) and os.path.isfile(input_data):
            return self.processor(Image.open(input_data).convert("RGB"))
        elif isinstance(input_data, np.ndarray):
            return self.processor(
                Image.fromarray(np.uint8(input_data)).convert("RGB"))
        elif isinstance(input_data, Image.Image):
            return self.processor(input_data.convert("RGB"))
        elif isinstance(input_data, list):
            return torch.stack(
                [self.processor(i.convert("RGB")) for i in input_data])
        else:
            raise ValueError(
                "Input must be a PIL Image, file path, or numpy array")

    def get_model(self, identifier):
        """Instantiate and return an EDDPMTappedClassifier with weights."""
        if identifier not in _CHECKPOINT_SOURCES:
            raise ValueError(
                f"Unknown identifier: {identifier}. "
                f"Available: {list(_CHECKPOINT_SOURCES)}")

        deps = _load_eddpm_deps()
        ckpt_path = _ensure_checkpoint(identifier)

        defaults = deps["classifier_defaults"]()
        defaults["image_size"] = 256
        classifier = deps["create_classifier"](**defaults)

        state_dict = _safe_load_state_dict(ckpt_path)
        classifier.load_state_dict(state_dict)

        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad_(False)

        model = EDDPMTappedClassifier(classifier)
        model.eval()
        return model

    def postprocess_fn(self, features):
        """Postprocess features for BBScore ridge metrics.

        BBScore's extractor may introduce a singleton sequence/time dimension
        (e.g. stacking outputs as (B, T, C, H, W) for static models). This
        helper:
          - pools large spatial maps down to 8x8
          - flattens everything except the sample/batch dimension
          - returns a numpy array (no (B, T, -1) reshaping)
        """
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(np.asarray(features))

        if features.ndim < 2:
            return features.detach().cpu().numpy()

        if features.ndim == 2:
            processed = features
        elif features.ndim == 4:
            # (B, C, H, W)
            if features.shape[-2] > 8 or features.shape[-1] > 8:
                features = F.adaptive_avg_pool2d(features, (8, 8))
            processed = features.flatten(start_dim=1)
        elif features.ndim == 5:
            # (B, T, C, H, W) -- pool each spatial map, then flatten.
            B, T, C, H, W = features.shape
            inner = features.reshape(B * T, C, H, W)
            if H > 8 or W > 8:
                inner = F.adaptive_avg_pool2d(inner, (8, 8))
            processed = inner.flatten(start_dim=1).reshape(B, -1)
        else:
            processed = features.flatten(start_dim=1)

        return processed.detach().cpu().numpy()
