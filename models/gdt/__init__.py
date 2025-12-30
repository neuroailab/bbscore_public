from models import MODEL_REGISTRY
from .gdt import GDTVideo

MODEL_REGISTRY["gdt_ht100"] = {
    "class": GDTVideo, "model_id_mapping": "GDT-HowTo100M"}
MODEL_REGISTRY["gdt_ig65"] = {
    "class": GDTVideo, "model_id_mapping": "GDT-IG65M"}
MODEL_REGISTRY["gdt_kinetics400"] = {
    "class": GDTVideo, "model_id_mapping": "GDT-Kinetics400"}
