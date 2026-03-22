from models import MODEL_REGISTRY
from .vilt import ViLTBBScore

MODEL_REGISTRY["vilt_b32_mlm"] = {
    "class": ViLTBBScore,
    "model_id_mapping": "dandelin/vilt-b32-mlm",
}
