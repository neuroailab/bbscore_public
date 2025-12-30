from models import MODEL_REGISTRY
from .r3m_resnet import R3M

MODEL_REGISTRY["r3m"] = {
    "class": R3M, "model_id_mapping": "R3M-ResNet50"}
