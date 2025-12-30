from models import MODEL_REGISTRY
# assuming your class is defined in torch_video.py
from .torch_vision import TorchVideoModels

# Model registrations
MODEL_REGISTRY["mc3_18"] = {
    "class": TorchVideoModels, "model_id_mapping": "MC3_18_Weights"}

MODEL_REGISTRY["mvit"] = {
    "class": TorchVideoModels, "model_id_mapping": "MViT"}

MODEL_REGISTRY["mvit_v1_b"] = {
    "class": TorchVideoModels, "model_id_mapping": "MViT_V1_B_Weights"}

MODEL_REGISTRY["mvit_v2_s"] = {
    "class": TorchVideoModels, "model_id_mapping": "MViT_V2_S_Weights"}

MODEL_REGISTRY["r2plus1d_18"] = {
    "class": TorchVideoModels, "model_id_mapping": "R2Plus1D_18_Weights"}

MODEL_REGISTRY["r3d_18"] = {
    "class": TorchVideoModels, "model_id_mapping": "R3D_18_Weights"}

MODEL_REGISTRY["s3d"] = {
    "class": TorchVideoModels, "model_id_mapping": "S3D_Weights"}

MODEL_REGISTRY["swin3d_b"] = {
    "class": TorchVideoModels, "model_id_mapping": "Swin3D_B_Weights"}

MODEL_REGISTRY["swin3d_s"] = {
    "class": TorchVideoModels, "model_id_mapping": "Swin3D_S_Weights"}

MODEL_REGISTRY["swin3d_t"] = {
    "class": TorchVideoModels, "model_id_mapping": "Swin3D_T_Weights"}

MODEL_REGISTRY["swin_transformer"] = {
    "class": TorchVideoModels, "model_id_mapping": "SwinTransformer3d"}

MODEL_REGISTRY["video_resnet"] = {
    "class": TorchVideoModels, "model_id_mapping": "VideoResNet"}
