from models import MODEL_REGISTRY
from .avid import AVID

MODEL_REGISTRY["avid_kinetics"] = {
    "class": AVID, "model_id_mapping": "AVID-Kinetics400"}
MODEL_REGISTRY["avid_audioset"] = {
    "class": AVID, "model_id_mapping": "AVID-Audioset"}
MODEL_REGISTRY["avid_cma_kinetics"] = {
    "class": AVID, "model_id_mapping": "AVID-CMA-Kinetics400"}
MODEL_REGISTRY["avid_cma_audioset"] = {
    "class": AVID, "model_id_mapping": "AVID-CMA-Audioset"}
