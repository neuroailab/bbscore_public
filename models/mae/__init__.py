from models import MODEL_REGISTRY
from .mae import MAE

MODEL_REGISTRY["mae_base"] = {
    "class": MAE, "model_id_mapping": "MAE-BASE"}
MODEL_REGISTRY["mae_large"] = {
    "class": MAE, "model_id_mapping": "MAE-LARGE"}
