from models import MODEL_REGISTRY
from .dinov2_lstm import DINOV2_LSTM

MODEL_REGISTRY["dinov2_lstm"] = {
    "class": DINOV2_LSTM, "model_id_mapping": "DINOV2_LSTM"}
