from benchmarks.BBS import BenchmarkScore
from data.NSDShared import (
    NSDStimulusSet,
    NSDAssemblyV1,
    NSDAssemblyV1d,
    NSDAssemblyV1v,
    NSDAssemblyV2,
    NSDAssemblyV2v,
    NSDAssemblyV2d,
    NSDAssemblyV3,
    NSDAssemblyV3d,
    NSDAssemblyV3v,
    NSDAssemblyV4,
    NSDAssemblyLateral,
    NSDAssemblyVentral,
    NSDAssemblyParietal,
    NSDAssemblyMidLateral,
    NSDAssemblyMidVentral,
    NSDAssemblyMidParietal,
    NSDAssemblyHighLateral,
    NSDAssemblyHighVentral,
    NSDAssemblyHighParietal,
)
from benchmarks import BENCHMARK_REGISTRY


class NSDV1Shared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV1,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDV1Shared"] = NSDV1Shared


class NSDV1dShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV1d,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDV1dShared"] = NSDV1dShared


class NSDV1vShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV1v,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDV1vShared"] = NSDV1vShared


class NSDV2Shared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV2,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDV2Shared"] = NSDV2Shared


class NSDV2dShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV2d,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDV2dShared"] = NSDV2dShared


class NSDV2vShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV2v,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDV2vShared"] = NSDV2vShared


class NSDV3Shared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV3,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDV3Shared"] = NSDV3Shared


class NSDV3dShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV3d,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDV3dShared"] = NSDV3dShared


class NSDV3vShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV3v,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDV3vShared"] = NSDV3vShared


class NSDV4Shared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV4,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDV4Shared"] = NSDV4Shared


class NSDLateralShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyLateral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDLateralShared"] = NSDLateralShared


class NSDVentralShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyVentral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDVentralShared"] = NSDVentralShared


class NSDParietalShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyParietal,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDParietalShared"] = NSDParietalShared


class NSDHighLateralShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyHighLateral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDHighLateralShared"] = NSDHighLateralShared


class NSDHighVentralShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyHighVentral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDHighVentralShared"] = NSDHighVentralShared


class NSDHighParietalShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyHighParietal,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDHighParietalShared"] = NSDHighParietalShared


class NSDMidLateralShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyMidLateral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDMidLateralShared"] = NSDMidLateralShared


class NSDMidVentralShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyMidVentral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDMidVentralShared"] = NSDMidVentralShared


class NSDMidParietalShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyMidParietal,
            batch_size=batch_size,
            num_workers=0,
            debug=debug
        )


BENCHMARK_REGISTRY["NSDMidParietalShared"] = NSDMidParietalShared
