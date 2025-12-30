from models import MODEL_REGISTRY
from .pixelnerf_lstm import PIXELNERF_LSTM

MODEL_REGISTRY["pixelnerf_lstm"] = {
    "class": PIXELNERF_LSTM, "model_id_mapping": "PIXELNERF_LSTM"}
