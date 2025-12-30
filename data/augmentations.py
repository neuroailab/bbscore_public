import math
import numpy as np
import random
import re
import PIL
from PIL import Image, ImageEnhance, ImageOps
from typing import List, Union, Tuple, Any, Dict, Optional, Callable
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

_PIL_VER = tuple([int(x) for x in PIL.__version__.split(".")[:2]])
_FILL = (128, 128, 128)
_MAX_LEVEL = 10.0
_HPARAMS_DEFAULT = {
    "translate_const": 250,
    "img_mean": _FILL,
}
_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _interpolation(kwargs):
    interpolation = kwargs.pop("resample", Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    if "fillcolor" in kwargs and _PIL_VER < (5, 0):
        kwargs.pop("fillcolor")
    kwargs["resample"] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15), round(math.sin(angle), 15), 0.0,
            round(-math.sin(angle), 15), round(math.cos(angle), 15), 0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f
        matrix[2], matrix[5] = transform(-rotn_center[0] -
                                         post_trans[0], -rotn_center[1] - post_trans[1], matrix)
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs["resample"])


def auto_contrast(img, **__): return ImageOps.autocontrast(img)
def invert(img, **__): return ImageOps.invert(img)
def equalize(img, **__): return ImageOps.equalize(img)
def solarize(img, thresh, **__): return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        lut.append(min(255, i + add) if i < thresh else i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    return img if bits_to_keep >= 8 else ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **
             __): return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__): return ImageEnhance.Color(img).enhance(factor)


def brightness(
    img, factor, **__): return ImageEnhance.Brightness(img).enhance(factor)
def sharpness(
    img, factor, **__): return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v): return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, _hparams): return (
    _randomly_negate((level / _MAX_LEVEL) * 30.0),)


def _enhance_level_to_arg(level, _hparams): return (
    (level / _MAX_LEVEL) * 1.8 + 0.1,)


def _enhance_increasing_level_to_arg(level, _hparams): return (
    1.0 + _randomly_negate((level / _MAX_LEVEL) * 0.9),)


def _shear_level_to_arg(level, _hparams): return (
    _randomly_negate((level / _MAX_LEVEL) * 0.3),)


def _translate_abs_level_to_arg(level, hparams): return (
    _randomly_negate((level / _MAX_LEVEL) * float(hparams["translate_const"])),)


def _translate_rel_level_to_arg(level, hparams): return (
    _randomly_negate((level / _MAX_LEVEL) * hparams.get("translate_pct", 0.45)),)


def _posterize_level_to_arg(level, _hparams): return (
    int((level / _MAX_LEVEL) * 4),)


def _posterize_increasing_level_to_arg(level, hparams): return (
    4 - _posterize_level_to_arg(level, hparams)[0],)


def _posterize_original_level_to_arg(level, _hparams): return (
    int((level / _MAX_LEVEL) * 4) + 4,)


def _solarize_level_to_arg(level, _hparams): return (
    int((level / _MAX_LEVEL) * 256),)


def _solarize_increasing_level_to_arg(level, _hparams): return (
    256 - _solarize_level_to_arg(level, _hparams)[0],)


def _solarize_add_level_to_arg(level, _hparams): return (
    int((level / _MAX_LEVEL) * 110),)


LEVEL_TO_ARG = {
    "AutoContrast": None, "Equalize": None, "Invert": None, "Rotate": _rotate_level_to_arg,
    "Posterize": _posterize_level_to_arg, "PosterizeIncreasing": _posterize_increasing_level_to_arg,
    "PosterizeOriginal": _posterize_original_level_to_arg, "Solarize": _solarize_level_to_arg,
    "SolarizeIncreasing": _solarize_increasing_level_to_arg, "SolarizeAdd": _solarize_add_level_to_arg,
    "Color": _enhance_level_to_arg, "ColorIncreasing": _enhance_increasing_level_to_arg,
    "Contrast": _enhance_level_to_arg, "ContrastIncreasing": _enhance_increasing_level_to_arg,
    "Brightness": _enhance_level_to_arg, "BrightnessIncreasing": _enhance_increasing_level_to_arg,
    "Sharpness": _enhance_level_to_arg, "SharpnessIncreasing": _enhance_increasing_level_to_arg,
    "ShearX": _shear_level_to_arg, "ShearY": _shear_level_to_arg,
    "TranslateX": _translate_abs_level_to_arg, "TranslateY": _translate_abs_level_to_arg,
    "TranslateXRel": _translate_rel_level_to_arg, "TranslateYRel": _translate_rel_level_to_arg,
}

NAME_TO_OP = {
    "AutoContrast": auto_contrast, "Equalize": equalize, "Invert": invert, "Rotate": rotate,
    "Posterize": posterize, "PosterizeIncreasing": posterize, "PosterizeOriginal": posterize,
    "Solarize": solarize, "SolarizeIncreasing": solarize, "SolarizeAdd": solarize_add,
    "Color": color, "ColorIncreasing": color, "Contrast": contrast, "ContrastIncreasing": contrast,
    "Brightness": brightness, "BrightnessIncreasing": brightness, "Sharpness": sharpness,
    "SharpnessIncreasing": sharpness, "ShearX": shear_x, "ShearY": shear_y,
    "TranslateX": translate_x_abs, "TranslateY": translate_y_abs,
    "TranslateXRel": translate_x_rel, "TranslateYRel": translate_y_rel,
}


class AugmentOp:
    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.name = name
        self.op_function = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = {
            "fillcolor": hparams["img_mean"] if "img_mean" in hparams else _FILL,
            "resample": hparams["interpolation"] if "interpolation" in hparams else _RANDOM_INTERPOLATION,
        }
        self.magnitude_std = self.hparams.get("magnitude_std", 0)

    def apply_op(self, img: Image.Image) -> Image.Image:
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        current_magnitude = self.magnitude
        if self.magnitude_std > 0:
            current_magnitude = random.gauss(
                current_magnitude, self.magnitude_std)
        current_magnitude = min(_MAX_LEVEL, max(0, current_magnitude))
        args = self.level_fn(
            current_magnitude, self.hparams) if self.level_fn else ()
        return self.op_function(img, *args, **self.kwargs)


_RAND_TRANSFORMS = [
    "AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "Solarize", "SolarizeAdd",
    "Color", "Contrast", "Brightness", "Sharpness", "ShearX", "ShearY",
    "TranslateXRel", "TranslateYRel",
]
_RAND_INCREASING_TRANSFORMS = [
    "AutoContrast", "Equalize", "Invert", "Rotate", "PosterizeIncreasing", "SolarizeIncreasing",
    "SolarizeAdd", "ColorIncreasing", "ContrastIncreasing", "BrightnessIncreasing",
    "SharpnessIncreasing", "ShearX", "ShearY", "TranslateXRel", "TranslateYRel",
]
_RAND_CHOICE_WEIGHTS_0 = {
    "Rotate": 0.3, "ShearX": 0.2, "ShearY": 0.2, "TranslateXRel": 0.1, "TranslateYRel": 0.1,
    "Color": 0.025, "Sharpness": 0.025, "AutoContrast": 0.025, "Solarize": 0.005,
    "SolarizeAdd": 0.005, "Contrast": 0.005, "Brightness": 0.005, "Equalize": 0.005,
    "Posterize": 0, "Invert": 0,
}


def _select_rand_weights(weight_idx=0, transforms_list=None):
    transforms_list = transforms_list or _RAND_TRANSFORMS
    if weight_idx != 0:
        raise ValueError("Only weight_idx 0 is supported")
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    # Ensure all transforms are in weights
    probs = [rand_weights[k] for k in transforms_list if k in rand_weights]
    if len(probs) != len(transforms_list):  # Fallback if a transform is not in weights
        print(f"Warning: Not all transforms in `transforms_list` found in `_RAND_CHOICE_WEIGHTS_0`. Using uniform probabilities.")
        return None  # Will lead to uniform choice in np.random.choice
    probs_sum = np.sum(probs)
    if probs_sum == 0:
        return None  # Avoid division by zero if all weights are 0
    return np.array(probs) / probs_sum


def rand_augment_ops(magnitude=10, hparams=None, transforms_list=None):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms_list = transforms_list or _RAND_TRANSFORMS
    return [
        AugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams)
        for name in transforms_list
    ]


class RandAugment:
    def __init__(self, ops: List[AugmentOp], num_layers=2, choice_weights: Optional[np.ndarray] = None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img_or_list_of_imgs: Union[Image.Image, List[Image.Image]]) -> Union[Image.Image, List[Image.Image]]:
        ops_to_apply = np.random.choice(
            self.ops, self.num_layers, replace=self.choice_weights is None, p=self.choice_weights
        )
        if isinstance(img_or_list_of_imgs, list):  # Video frames
            processed_frames = []
            for img_frame in img_or_list_of_imgs:
                processed_frame = img_frame
                for op_instance in ops_to_apply:
                    processed_frame = op_instance.apply_op(processed_frame)
                processed_frames.append(processed_frame)
            return processed_frames
        else:  # Single image
            processed_img = img_or_list_of_imgs
            for op_instance in ops_to_apply:
                processed_img = op_instance.apply_op(processed_img)
            return processed_img


def rand_augment_transform(config_str: str, hparams: Dict[str, Any]) -> RandAugment:
    magnitude = _MAX_LEVEL
    num_layers = 2
    weight_idx = None
    transforms_list = _RAND_TRANSFORMS
    config = config_str.split("-")
    assert config[
        0] == "rand", f"RandAugment config string must start with 'rand', got {config_str}"
    config = config[1:]
    for c in config:
        cs = re.split(r"(\d.*)", c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == "mstd":
            hparams.setdefault("magnitude_std", float(val))
        elif key == "inc":
            if bool(int(val)):
                transforms_list = _RAND_INCREASING_TRANSFORMS  # Check int(val)
        elif key == "m":
            magnitude = int(val)
        elif key == "n":
            num_layers = int(val)
        elif key == "w":
            weight_idx = int(val)
        else:
            raise NotImplementedError(f"Unknown RandAugment config key: {key}")

    ra_ops = rand_augment_ops(
        magnitude=magnitude, hparams=hparams, transforms_list=transforms_list)
    choice_weights = None
    if weight_idx is not None:
        choice_weights = _select_rand_weights(
            weight_idx, transforms_list=transforms_list)
        if choice_weights is None and len(ra_ops) > 0:  # Fallback for safety
            print(
                f"Warning: could not select random weights for index {weight_idx}, using uniform.")

    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)


def _ra_worker(args):
    """
    Worker for ProcessPool: takes (PIL.Image, RandAugment instance), returns augmented PIL.
    """
    frame, ra = args
    return ra(frame)


def randaugment_parallel(
    pil_frames: list,
    ra_instance: RandAugment,
    num_workers: int = None
) -> list:
    """
    Uses multiprocessing.Pool to apply RandAugment to each frame in parallel.
    """
    if num_workers is None:
        # max 8, or CPU count, or #frames
        num_workers = min(cpu_count(), len(pil_frames), 8)

    # Build tasks: each is (frame, ra_instance)
    tasks = [(pil_frames[i], ra_instance) for i in range(len(pil_frames))]
    with Pool(num_workers) as pool:
        augmented = pool.map(_ra_worker, tasks)
    return augmented


def randaugment_threaded(
    pil_frames: list,
    ra_instance: RandAugment,
    num_workers: int = None
) -> list:
    """
    Uses ThreadPoolExecutor to apply RandAugment to each frame in parallel.
    Pillow-SIMD releases the GIL, so threads can help too.
    """
    if num_workers is None:
        num_workers = min(cpu_count() - 1, len(pil_frames),
                          16)  # leave 1 core for OS

    augmented = [None] * len(pil_frames)
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        futures = {exe.submit(
            ra_instance, pil_frames[i]): i for i in range(len(pil_frames))}
        for fut in as_completed(futures):
            idx = futures[fut]
            augmented[idx] = fut.result()
    return augmented
