from models import MODEL_REGISTRY
from .blip import BLIP

MODEL_REGISTRY["blip_base"] = {
    "class": BLIP, "model_id_mapping": "BLIP-BASE"}
MODEL_REGISTRY["blip_large"] = {
    "class": BLIP, "model_id_mapping": "BLIP-LARGE"}
MODEL_REGISTRY["blip_vqa"] = {
    "class": BLIP, "model_id_mapping": "BLIP-VQA"}
MODEL_REGISTRY["blip_itm"] = {
    "class": BLIP, "model_id_mapping": "BLIP-ITM"}
