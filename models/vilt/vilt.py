import os
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import ViltConfig, ViltModel, ViltProcessor


def _ensure_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _to_pil(img: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()

        if img.ndim == 4:
            raise ValueError("Expected a single image tensor, got a batched tensor")
        if img.ndim == 2:
            img = img.unsqueeze(0)

        # CHW
        if img.ndim == 3 and img.shape[0] in (1, 3):
            return to_pil_image(img).convert("RGB")

        # HWC
        if img.ndim == 3 and img.shape[-1] in (1, 3):
            arr = img.numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")

        raise ValueError(f"Unsupported tensor image shape: {tuple(img.shape)}")

    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return Image.fromarray(img).convert("RGB")
        if img.ndim == 3:
            arr = img
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")

    if isinstance(img, str) and os.path.isfile(img):
        return Image.open(img).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(img)}")


class CaptionResolver:
    """
    Builds the text prompt for ViLT.

    Env vars:
      BBSCORE_VILT_PROMPT
      BBSCORE_VILT_PROMPT_MODE

    Supported modes:
      - fixed
      - coco_first
      - coco_all
      - fixed_plus_coco_first
      - fixed_plus_coco_all
    """

    def __init__(self):
        self.fixed_prompt = os.environ.get("BBSCORE_VILT_PROMPT", "").strip()
        self.mode = os.environ.get("BBSCORE_VILT_PROMPT_MODE", "fixed").strip().lower()
        print("fixed_prompt", self.fixed_prompt)
        print("mode", self.mode)

        valid_modes = {
            "fixed",
            "coco_first",
            "coco_all",
            "fixed_plus_coco_first",
            "fixed_plus_coco_all",
        }
        if self.mode not in valid_modes:
            raise ValueError(
                f"Unsupported BBSCORE_VILT_PROMPT_MODE={self.mode!r}. "
                f"Use one of {sorted(valid_modes)}."
            )

    def build_prompt(self, sample: Dict[str, Any] = None) -> str:
        fixed = self.fixed_prompt
        captions = []

        if isinstance(sample, dict):
            caps = sample.get("captions", None)
            if caps is not None:
                if isinstance(caps, (list, tuple)):
                    captions = [str(x).strip() for x in caps if str(x).strip()]
                else:
                    captions = [str(caps).strip()]

        first_caption = captions[0] if captions else ""
        all_captions = " ".join(captions) if captions else ""

        if self.mode == "fixed":
            return fixed

        if self.mode == "coco_first":
            return first_caption or fixed

        if self.mode == "coco_all":
            return all_captions or fixed

        if self.mode == "fixed_plus_coco_first":
            return f"{fixed} {first_caption}".strip()

        if self.mode == "fixed_plus_coco_all":
            return f"{fixed} {all_captions}".strip()

        return fixed


class _ViLTBackboneForBBScore(nn.Module):
    """
    Thin wrapper so BBScore can hook layers like:
      - vilt.embeddings
      - vilt.encoder.layer.0
      - ...
      - vilt.encoder.layer.11
      - vilt.layernorm
    """

    def __init__(
        self,
        model_name: str,
        max_text_len: int = 40,
        add_pooling_layer: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_text_len = max_text_len

        self.random_init = os.environ.get('BBSCORE_VILT_RANDOM_INIT', '0') == '1'
        self.random_seed = int(os.environ.get('BBSCORE_VILT_RANDOM_SEED', '0'))

        self.processor = ViltProcessor.from_pretrained(
            model_name,
            use_fast=False,
        )

        if self.random_init:
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)

            config = ViltConfig.from_pretrained(model_name)
            config.add_pooling_layer = add_pooling_layer
            self.vilt = ViltModel(config)
            print(f'[ViLT] Using randomly initialized weights with seed {self.random_seed}.')
        else:
            self.vilt = ViltModel.from_pretrained(
                model_name,
                add_pooling_layer=add_pooling_layer,
                use_safetensors=True,
            )

    def _normalize_batch(
        self,
        batch: Union[torch.Tensor, Dict[str, Any], List[Any], tuple],
    ) -> Dict[str, Any]:
        if isinstance(batch, torch.Tensor):
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)
            if batch.ndim != 4:
                raise ValueError(
                    f"Expected tensor batch of shape [B, C, H, W], got {tuple(batch.shape)}"
                )
            images = [_to_pil(img) for img in batch]
            texts = [""] * len(images)
            return {"images": images, "texts": texts}

        if isinstance(batch, dict):
            images = None
            for key in ("image", "images", "img", "pixel_values"):
                if key in batch:
                    images = batch[key]
                    break
            if images is None:
                raise KeyError(
                    "ViLT batch dict must contain one of: image, images, img, pixel_values"
                )

            texts = None
            for key in ("text", "texts", "prompt", "prompts"):
                if key in batch:
                    texts = batch[key]
                    break

            images = _ensure_list(images)
            images = [_to_pil(img) for img in images]

            if texts is None:
                texts = [""] * len(images)
            else:
                texts = _ensure_list(texts)
                if len(texts) == 1 and len(images) > 1:
                    texts = texts * len(images)
                if len(texts) != len(images):
                    raise ValueError(
                        f"Need one prompt per image, got {len(texts)} prompts for {len(images)} images"
                    )
                texts = [str(t) for t in texts]

            return {"images": images, "texts": texts}

        if isinstance(batch, (list, tuple)):
            images = [_to_pil(x) for x in batch]
            texts = [""] * len(images)
            return {"images": images, "texts": texts}

        raise TypeError(f"Unsupported batch type for ViLT: {type(batch)}")

    def forward(self, batch):
        batch = self._normalize_batch(batch)
        images, texts = batch["images"], batch["texts"]

        enc = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
        )

        device = next(self.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        outputs = self.vilt(**enc)
        return outputs.last_hidden_state


class ViLTBBScore:
    """
    BBScore wrapper for ViLT.

    Env vars:
      BBSCORE_VILT_PROMPT
      BBSCORE_VILT_PROMPT_MODE
      BBSCORE_VILT_POOL_MODE
      BBSCORE_VILT_MAX_TEXT_LEN
    """

    def __init__(self):
        self.pool_mode = os.environ.get("BBSCORE_VILT_POOL_MODE", "cls").strip().lower()
        self.max_text_len = int(os.environ.get("BBSCORE_VILT_MAX_TEXT_LEN", "40"))
        self.caption_resolver = CaptionResolver()

        if self.pool_mode not in {"cls", "mean"}:
            raise ValueError(
                f"Unsupported BBSCORE_VILT_POOL_MODE={self.pool_mode!r}. "
                "Use 'cls' or 'mean'."
            )

        self.static = True
        self.model = None
        self.expects_metadata = True

    def preprocess_fn(self, input_data, fps=None):
        """
        Accept either:
          - a plain image
          - a dict with keys like {"image": ..., "captions": ...}
        """
        if isinstance(input_data, dict):
            sample = dict(input_data)
            sample["text"] = self.caption_resolver.build_prompt(sample)
            return sample

        return {
            "image": input_data,
            "text": self.caption_resolver.build_prompt(None),
        }

    def get_model(self, identifier):
        self.model = _ViLTBackboneForBBScore(
            model_name=identifier,
            max_text_len=self.max_text_len,
            add_pooling_layer=True,
        )
        return self.model

    def postprocess_fn(self, features):
        if isinstance(features, tuple):
            features = features[0]

        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)

        # Some ViLT hook targets produce [B, 1, T, D].
        # Treat that as [B, T, D].
        if features.ndim == 4 and features.shape[1] == 1:
            features = features.squeeze(1)

        if features.ndim == 2:
            return features

        if features.ndim == 3:
            if self.pool_mode == "cls":
                return features[:, 0, :]
            if self.pool_mode == "mean":
                return features.mean(dim=1)

        raise ValueError(
            f"Unexpected hooked feature shape {tuple(features.shape)} for ViLT."
        )
