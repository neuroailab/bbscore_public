from models import MODEL_REGISTRY
from .convnext import ConvNeXt

MODEL_REGISTRY["convnext_tiny"] = {
    "class": ConvNeXt, "model_id_mapping": "CONVNEXT-TINY"}
MODEL_REGISTRY["convnext_small"] = {
    "class": ConvNeXt, "model_id_mapping": "CONVNEXT-SMALL"}
MODEL_REGISTRY["convnext_base"] = {
    "class": ConvNeXt, "model_id_mapping": "CONVNEXT-BASE"}
MODEL_REGISTRY["convnext_large"] = {
    "class": ConvNeXt, "model_id_mapping": "CONVNEXT-LARGE"}
MODEL_REGISTRY["convnext_xlarge"] = {
    "class": ConvNeXt, "model_id_mapping": "CONVNEXT-XLARGE"}
