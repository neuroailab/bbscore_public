from models import MODEL_REGISTRY
from .vjepa import VJEPA

MODEL_REGISTRY["vjepa_huge"] = {
    "class": VJEPA, "model_id_mapping": "VJEPA-HUGE"}
MODEL_REGISTRY["vjepa_large"] = {
    "class": VJEPA, "model_id_mapping": "VJEPA-LARGE"}
