from models import MODEL_REGISTRY
from .resnext_wsl import ResNeXtWSL

MODEL_REGISTRY["resnext101_32x8d_wsl"] = {
    "class": ResNeXtWSL, "model_id_mapping": "resnext101_32x8d_wsl"
}
