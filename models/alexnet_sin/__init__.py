from models import MODEL_REGISTRY
from .alexnet_sin import AlexNetSIN

MODEL_REGISTRY["alexnet_sin"] = {
    "class": AlexNetSIN, "model_id_mapping": "alexnet_sin"
}
