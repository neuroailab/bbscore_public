from benchmarks.BBS import BenchmarkScore
from data.StaticFullFieldSineGratings import StaticFullFieldSineGratingsStimulusSet
from benchmarks import BENCHMARK_REGISTRY


class V1StaticFullFieldSineGratings(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=StaticFullFieldSineGratingsStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            batch_size=batch_size,
            num_workers=16,
            debug=debug,
        )


BENCHMARK_REGISTRY["V1StaticFullFieldSineGratings"] = V1StaticFullFieldSineGratings
