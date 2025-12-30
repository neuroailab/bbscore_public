from models import MODEL_REGISTRY
from .s3d import S3DHowTo100M

MODEL_REGISTRY["s3d-HT100"] = {
    "class": S3DHowTo100M, "model_id_mapping": "s3d-HowTo100M"
}
