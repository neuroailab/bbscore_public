from models import MODEL_REGISTRY
from . import scaling_model
from .scaling_model import BaseModelLoader
import inspect


# --- Register all hardcoded model loaders ---

# Find all classes defined in the `all_models` module
for name, cls in inspect.getmembers(scaling_model, inspect.isclass):
    # Check if it's a valid, non-base model loader class
    if issubclass(cls, BaseModelLoader) and hasattr(cls, 'MODEL_ID') and cls.MODEL_ID:
        # The key is the model_id itself, and the value is the class
        MODEL_REGISTRY[cls.MODEL_ID] = {
            "class": cls,
            "model_id_mapping": cls.MODEL_ID  # The mapping is just the ID itself
        }
