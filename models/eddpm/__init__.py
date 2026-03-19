from models import MODEL_REGISTRY
from .eddpm import EDDPMModelWrapper

MODEL_REGISTRY["eddpm_classifier_256"] = {
    "class": EDDPMModelWrapper,
    "model_id_mapping": "eddpm_classifier_256",
}
MODEL_REGISTRY["eddpm_classifier_256_ect"] = {
    "class": EDDPMModelWrapper,
    "model_id_mapping": "eddpm_classifier_256_ect",
}
