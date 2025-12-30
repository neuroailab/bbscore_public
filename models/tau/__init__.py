from models import MODEL_REGISTRY
from .tau import TAU


MODEL_REGISTRY["TAU"] = {
    "class": TAU, "model_id_mapping": "TAU"
}
