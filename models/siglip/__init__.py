from models import MODEL_REGISTRY
from .siglip import SigLIP
 
MODEL_REGISTRY["siglip_base"] = {
    "class": SigLIP, "model_id_mapping": "SigLIP-Base"}
MODEL_REGISTRY["siglip_large"] = {
    "class": SigLIP, "model_id_mapping": "SigLIP-Large"}
 