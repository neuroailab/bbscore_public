from models import MODEL_REGISTRY
from .x3d import X3D

MODEL_REGISTRY["X3D"] = {
    "class": X3D, "model_id_mapping": "X3D"}
