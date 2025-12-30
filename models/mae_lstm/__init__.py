from models import MODEL_REGISTRY
from .mae_lstm import MAE_LSTM

MODEL_REGISTRY["mae_lstm"] = {
    "class": MAE_LSTM, "model_id_mapping": "MAE_LSTM"}
