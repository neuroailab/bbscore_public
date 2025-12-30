from models import MODEL_REGISTRY
from .selavi import SeLaVi

MODEL_REGISTRY["selavi_kinetics400"] = {
    "class": SeLaVi, "model_id_mapping": "SeLaVi-Kinetics400"}
MODEL_REGISTRY["selavi_kinetics_sound"] = {
    "class": SeLaVi, "model_id_mapping": "SeLaVi-Kinetics-Sound"}
MODEL_REGISTRY["selavi_ave"] = {
    "class": SeLaVi, "model_id_mapping": "SeLaVi-AVE"}
MODEL_REGISTRY["selavi_vgg"] = {
    "class": SeLaVi, "model_id_mapping": "SeLaVi-VGG-Sound"}
