from models import MODEL_REGISTRY
from .pixel import PIXEL

MODEL_REGISTRY["pixel"] = {
    "class": PIXEL, "model_id_mapping": "PIXEL"}
