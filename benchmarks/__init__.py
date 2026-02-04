BENCHMARK_REGISTRY = {}

# Import Benchmark to trigger registrations
from benchmarks.NSD import NSDSharedBenchmark
from benchmarks.TVSD import TVSDBenchmark
from benchmarks.Physion import PhysionContact
from benchmarks.Physion import PhysionPlacement
from benchmarks.SSV2 import SSV2Benchmark
from benchmarks.V1SineGratings import V1SineGratingsBenchmark
from benchmarks.LeBel2023 import LeBel2023Benchmark
