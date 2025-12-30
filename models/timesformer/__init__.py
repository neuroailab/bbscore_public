from models import MODEL_REGISTRY
from .timesformer import TimeSformer

MODEL_REGISTRY["timesformer_base"] = {
    "class": TimeSformer, "model_id_mapping": "TIMESFORMER-BASE"}
MODEL_REGISTRY["timesformer_hr"] = {
    "class": TimeSformer, "model_id_mapping": "TIMESFORMER-HR"}
MODEL_REGISTRY["timesformer_l"] = {
    "class": TimeSformer, "model_id_mapping": "TIMESFORMER-L"}
