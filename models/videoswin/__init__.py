from models import MODEL_REGISTRY
from .videoswin import VideoSwin

MODEL_REGISTRY["VideoSwin-B"] = {
    "class": VideoSwin, "model_id_mapping": "VideoSwin-B"}
MODEL_REGISTRY["VideoSwin-L"] = {
    "class": VideoSwin, "model_id_mapping": "VideoSwin-L"}
