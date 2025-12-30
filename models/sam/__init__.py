from models import MODEL_REGISTRY
from .sam import SAM

MODEL_REGISTRY["sam_base"] = {"class": SAM, "model_id_mapping": "SAM-BASE"}
MODEL_REGISTRY["sam_large"] = {"class": SAM, "model_id_mapping": "SAM-LARGE"}
MODEL_REGISTRY["sam_huge"] = {"class": SAM, "model_id_mapping": "SAM-HUGE"}
