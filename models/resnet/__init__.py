from models import MODEL_REGISTRY
from .resnet import ResNet

MODEL_REGISTRY["resnet18"] = {
    "class": ResNet, "model_id_mapping": "ResNet18"}
MODEL_REGISTRY["resnet34"] = {
    "class": ResNet, "model_id_mapping": "ResNet34"}
MODEL_REGISTRY["resnet50"] = {
    "class": ResNet, "model_id_mapping": "ResNet50"}
MODEL_REGISTRY["resnet50dino"] = {
    "class": ResNet, "model_id_mapping": "ResNet50-DINO"}
