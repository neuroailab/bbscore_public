from models import MODEL_REGISTRY
from .alexnet import AlexNet

MODEL_REGISTRY["alexnet_untrained"] = {
    "class": AlexNet, "model_id_mapping": "AlexNet-Untrained"}

MODEL_REGISTRY["alexnet_trained"] = {
    "class": AlexNet, "model_id_mapping": "AlexNet-ImageNet"}

MODEL_REGISTRY["alexnet_barcode"] = {
    "class": AlexNet, "model_id_mapping": "AlexNet-Barcode"}
