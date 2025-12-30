from models import MODEL_REGISTRY
from .dfm import DFM, DFM_LSTM

MODEL_REGISTRY["dfm"] = {
    "class": DFM, "model_id_mapping": "DFM"}

MODEL_REGISTRY["dfm_lstm"] = {
    "class": DFM_LSTM, "model_id_mapping": "DFM_LSTM"}
