from models import MODEL_REGISTRY
from .gpt2 import GPT2

MODEL_REGISTRY["gpt2_small"] = {
    "class": GPT2, "model_id_mapping": "GPT2-Small"}

MODEL_REGISTRY["gpt2_medium"] = {
    "class": GPT2, "model_id_mapping": "GPT2-Medium"}

MODEL_REGISTRY["gpt2_large"] = {
    "class": GPT2, "model_id_mapping": "GPT2-Large"}

MODEL_REGISTRY["gpt2_xl"] = {
    "class": GPT2, "model_id_mapping": "GPT2-XL"}
