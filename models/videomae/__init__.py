from models import MODEL_REGISTRY
from .videomae import VideoMAE

MODEL_REGISTRY["videomae_base"] = {
    "class": VideoMAE, "model_id_mapping": "VIDEOMAE-BASE"}
MODEL_REGISTRY["videomae_large"] = {
    "class": VideoMAE, "model_id_mapping": "VIDEOMAE-LARGE"}
