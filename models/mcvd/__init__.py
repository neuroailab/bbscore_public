from models import MODEL_REGISTRY
from .mcvd import MCVD

MODEL_REGISTRY["mcvd_ego"] = {
    "class": MCVD, "model_id_mapping": "MCVD-EGO"}
MODEL_REGISTRY["mcvd_phys"] = {
    "class": MCVD, "model_id_mapping": "MCVD-PHYS"}
