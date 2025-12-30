from models import MODEL_REGISTRY
from .r3m_lstm import R3M_LSTM

MODEL_REGISTRY["r3m_lstm_physion"] = {
    "class": R3M_LSTM, "model_id_mapping": "R3M_LSTM_PHYS"}

MODEL_REGISTRY["r3m_lstm_ego4d"] = {
    "class": R3M_LSTM, "model_id_mapping": "R3M_LSTM_EGO4D"}
