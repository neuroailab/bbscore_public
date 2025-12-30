from models import MODEL_REGISTRY
from .dino import DINO

MODEL_REGISTRY["dino_vitb8"] = {
    "class": DINO, "model_id_mapping": "DINO-VITB8"}
MODEL_REGISTRY["dino_vitb16"] = {
    "class": DINO, "model_id_mapping": "DINO-VITB16"}
MODEL_REGISTRY["dino_vits8"] = {
    "class": DINO, "model_id_mapping": "DINO-VITS8"}
MODEL_REGISTRY["dino_vits16"] = {
    "class": DINO, "model_id_mapping": "DINO-VITS16"}
