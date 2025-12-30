# Data Documentation

This directory handles the downloading, caching, and loading of datasets.

*   **Stimulus Sets:** Raw media (images/videos).
*   **Assemblies:** Neural recordings (fMRI voxels, spike rates) or behavioral labels.

## Storage
All data is downloaded to the path defined in your environment variable:
`export SCIKIT_LEARN_DATA=/path/to/storage`

## Adding New Data

1.  Create a file in `data/` (e.g., `MyDataset.py`).
2.  Inherit from `BaseDataset`.
3.  Implement:
    *   `__init__`: Set up paths. Use `self.fetch()` or `self.fetch_and_extract()` to download data from URLs, S3, or Google Drive automatically.
    *   `__len__`: Number of samples.
    *   `__getitem__`: Return the specific sample (image tensor or neural vector).
4.  If creating a Benchmark, import these classes in your `benchmarks/` script.

**Note on S3/GCS:**
The `BaseDataset` class has built-in helper methods to download files from AWS S3 or Google Cloud Storage if you have credentials set up.

Example (`data/MyDataset.py`):
```python
import os
from typing import Optional, Callable
import torch
from PIL import Image
from .base import BaseDataset


class MyStimulusSet(BaseDataset):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        preprocess: Optional[Callable] = None,
    ):
        super().__init__(root_dir)
        self.preprocess = preprocess
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
      #logic for loading paths to stimulus images.
      pass


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.preprocess:
            image = self.preprocess(image)
        return image


class MyAssembly(BaseDataset):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        # ... other arguments ...
    ):
        super().__init__(root_dir)
        # ... Initialization logic ...
        pass

    def get_assembly(self, arg1: str, ...): # example of extra argument.
        # ... Logic to load and prepare fMRI data ...
        # Should return a numpy array (the assembly) and the ceiling.
        pass


    def __len__(self):
        # ... return something
        pass


    def __getitem__(self, idx):
        pass
```
