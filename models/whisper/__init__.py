from models import MODEL_REGISTRY
from .whisper import Whisper

MODEL_REGISTRY["whisper_tiny"] = {
    "class": Whisper, "model_id_mapping": "Whisper-Tiny"}
MODEL_REGISTRY["whisper_base"] = {
    "class": Whisper, "model_id_mapping": "Whisper-Base"}
MODEL_REGISTRY["whisper_small"] = {
    "class": Whisper, "model_id_mapping": "Whisper-Small"}
MODEL_REGISTRY["whisper_medium"] = {
    "class": Whisper, "model_id_mapping": "Whisper-Medium"}
MODEL_REGISTRY["whisper_large_v3"] = {
    "class": Whisper, "model_id_mapping": "Whisper-Large-v3"}
