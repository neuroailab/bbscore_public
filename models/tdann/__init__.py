from models import MODEL_REGISTRY
from .tdann import TDANN

MODEL_REGISTRY["tdann_simclr"] = {
    "class": TDANN,
    "model_id_mapping": "tdann_simclr"
}
MODEL_REGISTRY["tdann_supervised"] = {
    "class": TDANN,
    "model_id_mapping": "tdann_supervised"
}

