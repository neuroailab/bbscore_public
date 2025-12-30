from benchmarks.BBS import BenchmarkScore
from benchmarks.BBS_online import OnlineBenchmarkScore
from data.PhysionDetection import (
    PhysionPlacementDetectionTrain,
    PhysionPlacementDetectionTest,
    PhysionIntraPlacementDetectionTrain,
    PhysionIntraPlacementDetectionTest
)
from data.PhysionPrediction import (
    PhysionPlacementPredictionTrain,
    PhysionPlacementPredictionTest,
    PhysionIntraPlacementPredictionTrain,
    PhysionIntraPlacementPredictionTest
)
from benchmarks import BENCHMARK_REGISTRY


class PhysionPlacementDetection(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=PhysionPlacementDetectionTrain,
            stimulus_test_class=PhysionPlacementDetectionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            batch_size=batch_size,
            num_workers=16,
            debug=debug,
            save_features=True
        )


BENCHMARK_REGISTRY["PhysionPlacementDetection"] = PhysionPlacementDetection


class PhysionIntraPlacementDetection(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=PhysionIntraPlacementDetectionTrain,
            stimulus_test_class=PhysionIntraPlacementDetectionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            batch_size=batch_size,
            num_workers=16,
            debug=debug,
            save_features=True
        )


BENCHMARK_REGISTRY["PhysionIntraPlacementDetection"] = PhysionIntraPlacementDetection


class OnlinePhysionPlacementDetection(OnlineBenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 32):
        super().__init__(
            stimulus_train_class=(
                PhysionPlacementDetectionTrain, PhysionPlacementDetectionTrain),
            stimulus_test_class=PhysionPlacementDetectionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=256,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug,
        )


BENCHMARK_REGISTRY["OnlinePhysionPlacementDetection"] = OnlinePhysionPlacementDetection


class OnlinePhysionIntraPlacementDetection(OnlineBenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 32):
        super().__init__(
            stimulus_train_class=(
                PhysionIntraPlacementDetectionTrain, PhysionIntraPlacementDetectionTrain),
            stimulus_test_class=PhysionIntraPlacementDetectionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=256,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug,
        )


BENCHMARK_REGISTRY["OnlinePhysionIntraPlacementDetection"] = OnlinePhysionIntraPlacementDetection


class PhysionPlacementPrediction(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=PhysionPlacementPredictionTrain,
            stimulus_test_class=PhysionPlacementPredictionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            batch_size=batch_size,
            num_workers=16,
            debug=debug,
            save_features=True
        )


BENCHMARK_REGISTRY["PhysionPlacementPrediction"] = PhysionPlacementPrediction


class PhysionIntraPlacementPrediction(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=PhysionIntraPlacementPredictionTrain,
            stimulus_test_class=PhysionIntraPlacementPredictionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            batch_size=batch_size,
            num_workers=16,
            debug=debug,
            save_features=True
        )


BENCHMARK_REGISTRY["PhysionIntraPlacementPrediction"] = PhysionIntraPlacementPrediction


class OnlinePhysionPlacementPrediction(OnlineBenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 32):
        super().__init__(
            stimulus_train_class=(
                PhysionPlacementPredictionTrain, PhysionPlacementPredictionTrain),
            stimulus_test_class=PhysionPlacementPredictionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=256,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug,
        )


BENCHMARK_REGISTRY["OnlinePhysionPlacementPrediction"] = OnlinePhysionPlacementPrediction


class OnlinePhysionIntraPlacementPrediction(OnlineBenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 32):
        super().__init__(
            stimulus_train_class=(
                PhysionIntraPlacementPredictionTrain, PhysionIntraPlacementPredictionTrain),
            stimulus_test_class=PhysionIntraPlacementPredictionTest,
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=256,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug,
        )


BENCHMARK_REGISTRY["OnlinePhysionIntraPlacementPrediction"] = OnlinePhysionIntraPlacementPrediction
