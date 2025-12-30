from models import MODEL_REGISTRY
from .resnet_lstm import RESNET_LSTM

MODEL_REGISTRY["resnet_lstm"] = {
    "class": RESNET_LSTM, "model_id_mapping": "RESNET_LSTM"}
