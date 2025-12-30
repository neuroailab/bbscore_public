from models import MODEL_REGISTRY
from .llama3 import LLAMA3

MODEL_REGISTRY["llama3"] = {
    "class": LLAMA3, "model_id_mapping": "Llama3"}
