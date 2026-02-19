from models import MODEL_REGISTRY
from .hubert import HuBERT

MODEL_REGISTRY["hubert_base"] = {
    "class": HuBERT, "model_id_mapping": "HuBERT-Base"}
MODEL_REGISTRY["hubert_large"] = {
    "class": HuBERT, "model_id_mapping": "HuBERT-Large"}
MODEL_REGISTRY["hubert_xl"] = {
    "class": HuBERT, "model_id_mapping": "HuBERT-XL"}
