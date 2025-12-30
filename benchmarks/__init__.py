BENCHMARK_REGISTRY = {}

# Import Benchmark to trigger registrations
from benchmarks.NSD import NSDSharedBenchmark
from benchmarks.BMD import BMDBenchmark
from benchmarks.FreemanZiemba2013 import FreemanZiemba2013Benchmark
from benchmarks.MajajHong2015 import MajajHong2015Benchmark
from benchmarks.TVSD import TVSDBenchmark, TVSDMonkeyTemporal, TVSDMarkov, TVSDOnlineMarkov
from benchmarks.Physion import PhysionContact
from benchmarks.Physion import PhysionPlacement
from benchmarks.ImageNet2012 import ImageNet2012
from benchmarks.SSV2 import SSV2Benchmark
from benchmarks.V1SineGratings import V1SineGratingsBenchmark
from benchmarks.Algonauts2025 import AlgonautsBenchmark
from benchmarks.iEEG import iEEGBenchmark
# If you have other benchmark folders, import them here as well:
# from benchmarks.AnotherFolder import AnotherBenchmarkModule
