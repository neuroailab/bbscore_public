from models import MODEL_REGISTRY
from .cornet import CORNET

# Register CORnet-Z
MODEL_REGISTRY["cornet_z"] = {
    "class": CORNET,
    "model_id_mapping": "CORNET-Z"
}

# Register CORnet-RT
MODEL_REGISTRY["cornet_rt"] = {
    "class": CORNET,
    "model_id_mapping": "CORNET-RT"
}

# Register CORnet-S
MODEL_REGISTRY["cornet_s"] = {
    "class": CORNET,
    "model_id_mapping": "CORNET-S"
}
