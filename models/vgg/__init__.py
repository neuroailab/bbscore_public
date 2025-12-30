from models import MODEL_REGISTRY
from .vgg import VGG

MODEL_REGISTRY["vgg19"] = {
    "class": VGG, "model_id_mapping": "VGG19"}
