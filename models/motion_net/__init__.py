from models import MODEL_REGISTRY
from .motion_net import MotionNetLoader

MODEL_REGISTRY["MotionNet"] = {
    "class": MotionNetLoader,
    "model_id_mapping": "MotionNet"
}
