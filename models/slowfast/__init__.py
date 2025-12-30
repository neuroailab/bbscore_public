from models import MODEL_REGISTRY
from .slowfast import SlowFast

MODEL_REGISTRY["slowfast"] = {
    "class": SlowFast, "model_id_mapping": "SlowFast"}
