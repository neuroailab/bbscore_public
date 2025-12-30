from models import MODEL_REGISTRY
from .predrnn import PREDRNN

MODEL_REGISTRY["predrnn"] = {
    "class": PREDRNN, "model_id_mapping": "PREDRNN"}
