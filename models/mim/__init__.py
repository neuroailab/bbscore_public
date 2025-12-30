from models import MODEL_REGISTRY
from .mim import MIM

MODEL_REGISTRY["mim"] = {
    "class": MIM, "model_id_mapping": "MIM"}
