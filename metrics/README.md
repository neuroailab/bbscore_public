# Metrics Documentation

Metrics define *how* we compare model activations to brain data.

## Available Metrics
To list metrics:
```bash
python -c "from metrics import METRICS; print(list(METRICS.keys()))"
```

*   **Regression:** `ridge` (Ridge Regression), `pls` (Partial Least Squares).
*   **Similarity:** `rsa` (Representational Similarity Analysis).
*   **Classification:** `accuracy` (for behavioral tasks).

## Adding a New Metric

1.  Create a file in `metrics/` (e.g., `mymetric.py`).
2.  Create a class inheriting from `BaseMetric`.
3.  Implement `compute_raw(source, target)`.
    *   `source`: Model activations (numpy array).
    *   `target`: Brain data (numpy array).
    *   Return a dictionary of scores (e.g., `{'pearson': 0.5}`).
4.  Register it in `metrics/__init__.py`.

**Example:**
```python
from .base import BaseMetric
import numpy as np

class SimpleDiffMetric(BaseMetric):
    def compute_raw(self, source, target, **kwargs):
        # Dummy example: simple mean difference
        score = np.mean(source - target)
        return {"simple_diff": score}
```
