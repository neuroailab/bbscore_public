from models import MODEL_REGISTRY
from .dualneuron import DualNeuron

MODEL_REGISTRY['dualneuron_v1gray'] = {
    "class": DualNeuron,
    "model_id_mapping": "gray_V1"
}
MODEL_REGISTRY['dualneuron_v4gray'] = {
    "class": DualNeuron,
    "model_id_mapping": "gray_V4"
}
MODEL_REGISTRY['dualneuron_v4color'] = {
    "class": DualNeuron,
    "model_id_mapping": "rgb_V4"
}