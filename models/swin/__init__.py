from models import MODEL_REGISTRY
from .swin import Swin

MODEL_REGISTRY["swin_tiny"] = {"class": Swin, "model_id_mapping": "SWIN-TINY"}
MODEL_REGISTRY["swin_small"] = {
    "class": Swin, "model_id_mapping": "SWIN-SMALL"}
MODEL_REGISTRY["swin_base"] = {"class": Swin, "model_id_mapping": "SWIN-BASE"}
MODEL_REGISTRY["swin_large"] = {
    "class": Swin, "model_id_mapping": "SWIN-LARGE"}
