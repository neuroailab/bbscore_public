from models import MODEL_REGISTRY
from .ssast_wrapper import SSASTWrapper

MODEL_REGISTRY["ssast_patch"] = {
    "class": SSASTWrapper,
    "model_id_mapping": "ssast_patch",
}
MODEL_REGISTRY["ssast_frame"] = {
    "class": SSASTWrapper,
    "model_id_mapping": "ssast_frame",
}

