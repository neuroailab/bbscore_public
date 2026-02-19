from models import MODEL_REGISTRY
from .wav2vec2 import Wav2Vec2

MODEL_REGISTRY["wav2vec2_base"] = {
    "class": Wav2Vec2, "model_id_mapping": "Wav2Vec2-Base"}
MODEL_REGISTRY["wav2vec2_large"] = {
    "class": Wav2Vec2, "model_id_mapping": "Wav2Vec2-Large"}
MODEL_REGISTRY["wav2vec2_large_960h"] = {
    "class": Wav2Vec2, "model_id_mapping": "Wav2Vec2-Large-960h"}
