from models import MODEL_REGISTRY
from .robust import ROBUST

MODEL_REGISTRY["swim_l_robust"] = {
    "class": ROBUST, "model_id_mapping": "Robust-Swin-L"}
MODEL_REGISTRY["convnextv2_l + swin-l"] = {
    "class": ROBUST, "model_id_mapping": "ConvNeXtV2-L + Swin-L"}
MODEL_REGISTRY["wideresnet101robust"] = {
    "class": ROBUST, "model_id_mapping": "RaWideResNet-101-2"}
MODEL_REGISTRY["vit_b_convstem_robust"] = {
    "class": ROBUST, "model_id_mapping": "ViT-B-ConvStem"}
MODEL_REGISTRY["resnet50robust"] = {
    "class": ROBUST, "model_id_mapping": "Robust-ResNet-50"}
MODEL_REGISTRY["deit_b_robust"] = {
    "class": ROBUST, "model_id_mapping": "DeiT-B"}
MODEL_REGISTRY["resnet50noisymix"] = {
    "class": ROBUST, "model_id_mapping": "NoisyMix-ResNet-50"}
