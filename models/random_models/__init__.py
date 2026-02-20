from models import MODEL_REGISTRY
from .random_models import (
    RandomConvnextLargeImagenetFullSeed0Loader,
    RandomResnext101_32x8d_wsl,
    RandomS3DHowTo100M,
    RandomVideoMAEV1L,
)

# Register Random ConvNeXt
MODEL_REGISTRY["Random-convnext_large_imagenet_full_seed-0"] = {
    "class": RandomConvnextLargeImagenetFullSeed0Loader,
    "model_id_mapping": None
}

# Register Random ResNeXt WSL
MODEL_REGISTRY["Random-resnext101_32x8d_wsl"] = {
    "class": RandomResnext101_32x8d_wsl,
    "model_id_mapping": None
}

# Register Random S3D
MODEL_REGISTRY["Random-s3d-HowTo100M"] = {
    "class": RandomS3DHowTo100M,
    "model_id_mapping": None
}

# Register Random VideoMAE
MODEL_REGISTRY["Random-VideoMAE-V1-L"] = {
    "class": RandomVideoMAEV1L,
    "model_id_mapping": None
}
