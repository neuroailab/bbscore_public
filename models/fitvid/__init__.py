from models import MODEL_REGISTRY
from .fitvid import FITVID

MODEL_REGISTRY["fitvid_ego"] = {
    "class": FITVID, "model_id_mapping": "FITVID-EGO"}
MODEL_REGISTRY["fitvid_phys"] = {
    "class": FITVID, "model_id_mapping": "FITVID-PHYS"}
