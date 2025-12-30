# Benchmarks Documentation

A "Benchmark" in BBScore represents a specific task. It connects a **Stimulus Set** (images/videos shown to a subject) with an **Assembly** (the recorded brain activity or behavioral response).

## Available Benchmarks
To list benchmarks:
```bash
python -c "from benchmarks import BENCHMARK_REGISTRY; print(list(BENCHMARK_REGISTRY.keys()))"
```

Common benchmarks include:
*   **NSD (Natural Scenes Dataset):** `NSDV1Shared`, `NSDV2Shared`...
*   **Algonauts:** `Algonauts2025Full`
*   **Primate Data:** `MajajHong2015V4`, `FreemanZiemba2013V1`

## Adding a New Benchmark

1.  **Create Directory:** Create `benchmarks/MyBenchmark/`.
2.  **Create Class:** Create `mybenchmark.py`. Inherit from `BenchmarkScore` (for standard tasks) or `OnlineBenchmarkScore` (for large datasets processed in streams).
3.  **Implement `__init__`:**
    *   Define which Data classes to use (`stimulus_train_class`, `assembly_class`).
    *   Set parameters like `batch_size`.
4.  **Register:** Add to `BENCHMARK_REGISTRY`.

**Example:**
```python
from benchmarks.BBS import BenchmarkScore
from data.MyData import MyStimuli, MyNeuralData
from benchmarks import BENCHMARK_REGISTRY

class MyNewBenchmark(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug=False, batch_size=64):
        super().__init__(
            stimulus_train_class=MyStimuli,
            assembly_class=MyNeuralData,
            model_identifier=model_identifier,
            layer_name=layer_name,
            debug=debug
        )

BENCHMARK_REGISTRY["MyNewBenchmark"] = MyNewBenchmark
```
