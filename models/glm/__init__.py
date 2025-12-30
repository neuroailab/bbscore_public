from models import MODEL_REGISTRY
from .glm import VLM

# Register GLM-4.5V
MODEL_REGISTRY["glm-4.5v"] = {
    "class": VLM,
    "model_id_mapping": "GLM-4.5V"
}

# Register GLM-4.1V-9B-Thinking
MODEL_REGISTRY["glm-4.1v-9b-thinking"] = {
    "class": VLM,
    "model_id_mapping": "GLM-4.1V-9B"
}
