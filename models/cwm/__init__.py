from models import MODEL_REGISTRY
from .cwm import CWM

MODEL_REGISTRY["cwm_170m_0.0"] = {
    "class": CWM, "model_id_mapping": "CWM170M_0.0"}
MODEL_REGISTRY["cwm_170m_0.9"] = {
    "class": CWM, "model_id_mapping": "CWM170M_0.9"}
MODEL_REGISTRY["cwm_170m_1.0"] = {
    "class": CWM, "model_id_mapping": "CWM170M_1.0"}
MODEL_REGISTRY["cwm_170m_one_image"] = {
    "class": CWM, "model_id_mapping": "CWM170M_ONE_IMAGE"}
MODEL_REGISTRY["cwm_1b_0.0"] = {
    "class": CWM, "model_id_mapping": "CWM1B_0.0"}
MODEL_REGISTRY["cwm_1b_0.9"] = {
    "class": CWM, "model_id_mapping": "CWM1B_0.9"}
MODEL_REGISTRY["cwm_1b_1.0"] = {
    "class": CWM, "model_id_mapping": "CWM1B_1.0"}
MODEL_REGISTRY["cwm_1b_one_image"] = {
    "class": CWM, "model_id_mapping": "CWM1B_ONE_IMAGE"}
