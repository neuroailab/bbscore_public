from models import MODEL_REGISTRY
from .clip import CLIP

MODEL_REGISTRY["clip_vit_b_32"] = {
    "class": CLIP, "model_id_mapping": "CLIP-VIT-B-32"}
MODEL_REGISTRY["clip_vit_b_16"] = {
    "class": CLIP, "model_id_mapping": "CLIP-VIT-B-16"}
MODEL_REGISTRY["clip_vit_l_14"] = {
    "class": CLIP, "model_id_mapping": "CLIP-VIT-L-14"}
MODEL_REGISTRY["clip_vit_l_14_336"] = {
    "class": CLIP, "model_id_mapping": "CLIP-VIT-L-14-336"}
