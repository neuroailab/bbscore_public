from models import MODEL_REGISTRY
from .pixelnerf import PixelNerf

MODEL_REGISTRY["pixelnerf"] = {
    "class": PixelNerf, "model_id_mapping": "PN"}
