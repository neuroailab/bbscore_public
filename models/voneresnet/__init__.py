from models import MODEL_REGISTRY
from .voneresnet import VOneResNet50NS

MODEL_REGISTRY["voneresnet-50-non_stochastic"] = {
    "class": VOneResNet50NS, "model_id_mapping": "voneresnet-50-non_stochastic"
}
