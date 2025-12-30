from models import MODEL_REGISTRY
from .i3d import I3D_NONLOCAL

MODEL_REGISTRY["i3d-nonlocal"] = {
    "class": I3D_NONLOCAL, "model_id_mapping": "I3D"}
