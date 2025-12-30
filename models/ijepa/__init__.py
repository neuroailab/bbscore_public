from models import MODEL_REGISTRY
from .ijepa import IJEPA

MODEL_REGISTRY["ijepa_huge_14_1k"] = {
    "class": IJEPA, "model_id_mapping": "IJEPA-HUGE14-1K"}
MODEL_REGISTRY["ijepa_huge_14_22k"] = {
    "class": IJEPA, "model_id_mapping": "IJEPA-HUGE14-22K"}
MODEL_REGISTRY["ijepa_huge_16_1k"] = {
    "class": IJEPA, "model_id_mapping": "IJEPA-HUGE16-1K"}
MODEL_REGISTRY["ijepa_giant"] = {
    "class": IJEPA, "model_id_mapping": "IJEPA-GIANT"}
