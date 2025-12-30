from models import MODEL_REGISTRY
from .simvp import SimVP

MODEL_REGISTRY["SimVP"] = {
    "class": SimVP, "model_id_mapping": "SimVP"
}
