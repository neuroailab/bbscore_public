from benchmarks.BBS import BenchmarkScore
from benchmarks.BBS_online import OnlineBenchmarkScore
from data.PhysionDetection import (
    PhysionContactDetectionTrain,
    # PhysionContactDetectionAugmentedTrain,
    PhysionContactDetectionTest,
    PhysionIntraContactDetectionTrain,
    # PhysionIntraContactDetectionAugmentedTrain,
    PhysionIntraContactDetectionTest
)
from data.PhysionPrediction import (
    PhysionContactPredictionTrain,
    # PhysionContactPredictionAugmentedTrain,
    PhysionContactPredictionTest,
    PhysionIntraContactPredictionTrain,
    # PhysionIntraContactPredictionAugmentedTrain,
    PhysionIntraContactPredictionTest
)
from benchmarks import BENCHMARK_REGISTRY


class PhysionContactDetection(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=PhysionContactDetectionTrain,
            stimulus_test_class=PhysionContactDetectionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            batch_size=batch_size,
            num_workers=16,
            debug=debug,
            save_features=True
        )


BENCHMARK_REGISTRY["PhysionContactDetection"] = PhysionContactDetection


class PhysionIntraContactDetection(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=PhysionIntraContactDetectionTrain,
            stimulus_test_class=PhysionIntraContactDetectionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            batch_size=batch_size,
            num_workers=16,
            debug=debug,
            save_features=True
        )


BENCHMARK_REGISTRY["PhysionIntraContactDetection"] = PhysionIntraContactDetection


class OnlinePhysionContactDetection(OnlineBenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 32):
        super().__init__(
            stimulus_train_class=(
                PhysionContactDetectionTrain, PhysionContactDetectionTrain),
            stimulus_test_class=PhysionContactDetectionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=2,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug,
        )


BENCHMARK_REGISTRY["OnlinePhysionContactDetection"] = OnlinePhysionContactDetection


class OnlinePhysionIntraContactDetection(OnlineBenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 32):
        super().__init__(
            stimulus_train_class=(
                PhysionIntraContactDetectionTrain, PhysionIntraContactDetectionTrain),
            stimulus_test_class=PhysionIntraContactDetectionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=2,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug,
        )


BENCHMARK_REGISTRY["OnlinePhysionIntraContactDetection"] = OnlinePhysionIntraContactDetection


class PhysionContactPrediction(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=PhysionContactPredictionTrain,
            stimulus_test_class=PhysionContactPredictionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            batch_size=batch_size,
            num_workers=16,
            debug=debug,
            save_features=True
        )


BENCHMARK_REGISTRY["PhysionContactPrediction"] = PhysionContactPrediction


class PhysionIntraContactPrediction(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=PhysionIntraContactPredictionTrain,
            stimulus_test_class=PhysionIntraContactPredictionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            batch_size=batch_size,
            num_workers=16,
            debug=debug,
            save_features=True
        )


BENCHMARK_REGISTRY["PhysionIntraContactPrediction"] = PhysionIntraContactPrediction


class OnlinePhysionContactPrediction(OnlineBenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 32):
        super().__init__(
            stimulus_train_class=(
                PhysionContactPredictionTrain, PhysionContactPredictionTrain),
            stimulus_test_class=PhysionContactPredictionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=2,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug,
        )


BENCHMARK_REGISTRY["OnlinePhysionContactPrediction"] = OnlinePhysionContactPrediction


class OnlinePhysionIntraContactPrediction(OnlineBenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 32):
        super().__init__(
            stimulus_train_class=(
                PhysionIntraContactPredictionTrain, PhysionIntraContactPredictionTrain),
            stimulus_test_class=PhysionIntraContactPredictionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=2,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug,
        )


BENCHMARK_REGISTRY["OnlinePhysionIntraContactPrediction"] = OnlinePhysionIntraContactPrediction
