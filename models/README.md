# Models Documentation

This directory contains wrappers for various deep learning models. These wrappers unify the interface so BBScore can extract features from standard libraries (TorchVision, HuggingFace) or custom repositories easily.

## Available Models
To list all available model keys via command line:
```bash
python -c "from models import MODEL_REGISTRY; print(list(MODEL_REGISTRY.keys()))"
```

Common models include: `resnet50`, `videomae_base`, `clip_vit_b_32`, `vjepa_huge`, `slowfast`, etc.

## Adding a New Model

To add a model that isn't currently supported:

1.  **Create a Folder:** Create a new directory in `models/` (e.g., `models/mymodel/`).
2.  **Create the Wrapper:** Create a python file (e.g., `mymodel.py`) containing a class that implements the following methods:
    *   `__init__`: Define model mappings/variants.
    *   `get_model(identifier)`: Load weights and return the PyTorch module.
    *   `preprocess_fn(input_data)`: Transform raw input (images/video frames) into the tensor format the model expects.
    *   `postprocess_fn(features)`: Flatten or reshape output features (usually to `[Batch, Time, Features]` or `[Batch, Features]`).
3.  **Register:**
    *   Create an `__init__.py` in your folder.
    *   Import your class and add it to the global registry.

**Example `models/mymodel/__init__.py`:**
```python
from models import MODEL_REGISTRY
from .mymodel import MyModelWrapper

MODEL_REGISTRY["my_cool_model"] = {
    "class": MyModelWrapper, 
    "model_id_mapping": "variant-1"
}
```
4. **Global Import:** Add `mymodel` to the list in `models/__init__.py`.
