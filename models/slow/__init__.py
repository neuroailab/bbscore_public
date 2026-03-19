from models import MODEL_REGISTRY
from .slow import Slow

MODEL_REGISTRY["slow_r50"] = {
    "class": Slow,
    "model_id_mapping": "SLOW-R50"
}
