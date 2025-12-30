from .timebin_registry import AUTO_REGISTERED
from models import MODEL_REGISTRY
from .vjepa2 import VJEPA2

MODEL_REGISTRY["vjepa2_large_64"] = {
    "class": VJEPA2, "model_id_mapping": "VJEPA2-VITL-FPC64-256"}
MODEL_REGISTRY["vjepa2_huge_64"] = {
    "class": VJEPA2, "model_id_mapping": "VJEPA2-VITH-FPC64-256"}
MODEL_REGISTRY["vjepa2_great_64"] = {
    "class": VJEPA2, "model_id_mapping": "VJEPA2-VITG-FPC64-256"}
MODEL_REGISTRY["vjepa2_large_32"] = {
    "class": VJEPA2, "model_id_mapping": "VJEPA2-VITL-FPC32-256-DIVE"}
MODEL_REGISTRY["vjepa2_large_16_ssv2"] = {
    "class": VJEPA2, "model_id_mapping": "VJEPA2-VITL-FPC16-256-SSV2"}

# Auto-register 18 timebin models with last-256 truncation
# vjepa2_tb01 vjepa2_tb02 vjepa2_tb03 vjepa2_tb04 ... vjepa2_tb17  vjepa2_tb18
