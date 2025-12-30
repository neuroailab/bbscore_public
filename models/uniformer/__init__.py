from models import MODEL_REGISTRY
from .uniformer import UniFormerV2

MODEL_REGISTRY["UniFormer-V2-B"] = {
    "class": UniFormerV2, "model_id_mapping": "UniFormer-V2-B"}
MODEL_REGISTRY["UniFormer-V2-L"] = {
    "class": UniFormerV2, "model_id_mapping": "UniFormer-V2-L"}
