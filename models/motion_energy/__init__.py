from models import MODEL_REGISTRY
from .motion import MotionEnergyLoader

MODEL_REGISTRY["motion-energy"] = {
    "class": MotionEnergyLoader,
    "model_id_mapping": "motion-energy"
}
