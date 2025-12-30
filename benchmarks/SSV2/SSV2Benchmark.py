from benchmarks.BBS import BenchmarkScore
from benchmarks.BBS_online import OnlineBenchmarkScore
from data.SSV2_pruned import (
    SSV2PrunedStimulusTrainSet,
    AugmentedSSV2PrunedStimulusTrainSet,
    SSV2PrunedStimulusTestSet
)
from benchmarks import BENCHMARK_REGISTRY


class SSV2Benchmark(OnlineBenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=(SSV2PrunedStimulusTrainSet,
                                  SSV2PrunedStimulusTrainSet),
            stimulus_test_class=SSV2PrunedStimulusTestSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=40,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug,
        )


BENCHMARK_REGISTRY["SSV2"] = SSV2Benchmark


class AugmentedSSV2Benchmark(OnlineBenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=(
                AugmentedSSV2PrunedStimulusTrainSet, SSV2PrunedStimulusTrainSet),
            stimulus_test_class=SSV2PrunedStimulusTestSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=40,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug,
        )


BENCHMARK_REGISTRY["AugmentedSSV2"] = AugmentedSSV2Benchmark
