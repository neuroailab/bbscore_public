from models import MODEL_REGISTRY
from .convlstm import CONVLSTM

MODEL_REGISTRY["convlstm"] = {
    "class": CONVLSTM, "model_id_mapping": "CONVLSTM"}
