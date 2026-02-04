from benchmarks.BBS import BenchmarkScore
from data.LeBel2023 import LeBel2023StimulusSet, LeBel2023Assembly
from benchmarks import BENCHMARK_REGISTRY

# Base class for single-subject LeBel benchmarks


class LeBel2023Base(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, subject_id, debug=False, batch_size=4):
        super().__init__(
            stimulus_train_class=LeBel2023StimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=LeBel2023Assembly,
            assembly_train_kwargs={'subjects': [subject_id]},
            assembly_test_kwargs={'subjects': [subject_id]},
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


# Register benchmarks for all 8 subjects (UTS01 to UTS08)
subjects = [f"UTS{i:02d}" for i in range(1, 9)]

for subj in subjects:
    class_name = f"LeBel2023{subj}"

    # Dynamic class creation to ensuring distinct types in registry
    class DynamicBenchmark(LeBel2023Base):
        def __init__(self, model_identifier, layer_name, debug=False, batch_size=4):
            super().__init__(model_identifier, layer_name, subj, debug, batch_size)

    DynamicBenchmark.__name__ = class_name
    BENCHMARK_REGISTRY[class_name] = DynamicBenchmark

# Generic fallback (defaults to UTS01)


class LeBel2023Benchmark(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=LeBel2023StimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=LeBel2023Assembly,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["LeBel2023"] = LeBel2023Benchmark
