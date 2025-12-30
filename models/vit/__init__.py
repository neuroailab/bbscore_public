from models import MODEL_REGISTRY
from .vit import VisionTransformer

MODEL_REGISTRY["vit_base"] = {
    "class": VisionTransformer, "model_id_mapping": "VIT-BASE"}
MODEL_REGISTRY["vit_large"] = {
    "class": VisionTransformer, "model_id_mapping": "VIT-LARGE"}
MODEL_REGISTRY["vit_huge"] = {
    "class": VisionTransformer, "model_id_mapping": "VIT-HUGE"}
