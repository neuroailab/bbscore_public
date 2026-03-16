"""
Route Learning Benchmarks for BBScore.

Brain-to-brain comparison benchmarks using AssemblyBenchmarkScorer.
Each benchmark wires Subject A's hippocampal assembly as the source and
Subject B's as the target, then runs ridge regression (or other metrics)
to measure how well one person's hippocampal activity predicts the other's.

Pattern follows TVSDMonkeyFV1toNV1, TVSDMonkeyNV1toFV1, etc.

Usage:
    python run.py --benchmark RouteLearningExp1s01toExp1s02Hippo --metric ridge

    # Or from Python:
    from benchmarks.RouteLearning.RouteLearningBenchmark import (
        RouteLearningExp1s01toExp1s02Hippo
    )
    bench = RouteLearningExp1s01toExp1s02Hippo()
    bench.add_metric('ridge')
    results = bench.run()
"""

from benchmarks.BBS import AssemblyBenchmarkScorer
from data.RouteLearning import (
    # Experiment 1: bilateral hippocampus
    RouteLearningExp1s01Hippo,
    RouteLearningExp1s02Hippo,
    RouteLearningExp1s03Hippo,
    RouteLearningExp1s04Hippo,
    RouteLearningExp1s05Hippo,
    # Experiment 1: left hippocampus
    RouteLearningExp1s01LeftHippo,
    RouteLearningExp1s02LeftHippo,
    # Experiment 1: right hippocampus
    RouteLearningExp1s01RightHippo,
    RouteLearningExp1s02RightHippo,
    # Experiment 2: bilateral hippocampus
    RouteLearningExp2s01Hippo,
    RouteLearningExp2s02Hippo,
    RouteLearningExp2s03Hippo,
)
from benchmarks import BENCHMARK_REGISTRY


# ──────────────────────────────────────────────────────────────
# Experiment 1: Bilateral hippocampus, subject pairs
# ──────────────────────────────────────────────────────────────

class RouteLearningExp1s01toExp1s02Hippo(AssemblyBenchmarkScorer):
    """Exp1 sub01 → sub02, bilateral hippocampus (ridge: brain→brain)."""
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=RouteLearningExp1s01Hippo,
            target_assembly_class=RouteLearningExp1s02Hippo,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug,
        )


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s02Hippo"] = (
    RouteLearningExp1s01toExp1s02Hippo
)


class RouteLearningExp1s02toExp1s01Hippo(AssemblyBenchmarkScorer):
    """Exp1 sub02 → sub01, bilateral hippocampus (reverse direction)."""
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=RouteLearningExp1s02Hippo,
            target_assembly_class=RouteLearningExp1s01Hippo,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug,
        )


BENCHMARK_REGISTRY["RouteLearningExp1s02toExp1s01Hippo"] = (
    RouteLearningExp1s02toExp1s01Hippo
)


class RouteLearningExp1s01toExp1s03Hippo(AssemblyBenchmarkScorer):
    """Exp1 sub01 → sub03, bilateral hippocampus."""
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=RouteLearningExp1s01Hippo,
            target_assembly_class=RouteLearningExp1s03Hippo,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug,
        )


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s03Hippo"] = (
    RouteLearningExp1s01toExp1s03Hippo
)


class RouteLearningExp1s01toExp1s04Hippo(AssemblyBenchmarkScorer):
    """Exp1 sub01 → sub04, bilateral hippocampus."""
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=RouteLearningExp1s01Hippo,
            target_assembly_class=RouteLearningExp1s04Hippo,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug,
        )


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s04Hippo"] = (
    RouteLearningExp1s01toExp1s04Hippo
)


class RouteLearningExp1s01toExp1s05Hippo(AssemblyBenchmarkScorer):
    """Exp1 sub01 → sub05, bilateral hippocampus."""
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=RouteLearningExp1s01Hippo,
            target_assembly_class=RouteLearningExp1s05Hippo,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug,
        )


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s05Hippo"] = (
    RouteLearningExp1s01toExp1s05Hippo
)


# ──────────────────────────────────────────────────────────────
# Experiment 1: Left hippocampus
# ──────────────────────────────────────────────────────────────

class RouteLearningExp1s01toExp1s02LeftHippo(AssemblyBenchmarkScorer):
    """Exp1 sub01 → sub02, left hippocampus only."""
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=RouteLearningExp1s01LeftHippo,
            target_assembly_class=RouteLearningExp1s02LeftHippo,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug,
        )


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s02LeftHippo"] = (
    RouteLearningExp1s01toExp1s02LeftHippo
)


# ──────────────────────────────────────────────────────────────
# Experiment 1: Right hippocampus
# ──────────────────────────────────────────────────────────────

class RouteLearningExp1s01toExp1s02RightHippo(AssemblyBenchmarkScorer):
    """Exp1 sub01 → sub02, right hippocampus only."""
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=RouteLearningExp1s01RightHippo,
            target_assembly_class=RouteLearningExp1s02RightHippo,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug,
        )


BENCHMARK_REGISTRY["RouteLearningExp1s01toExp1s02RightHippo"] = (
    RouteLearningExp1s01toExp1s02RightHippo
)


# ──────────────────────────────────────────────────────────────
# Experiment 2: Bilateral hippocampus
# ──────────────────────────────────────────────────────────────

class RouteLearningExp2s01toExp2s02Hippo(AssemblyBenchmarkScorer):
    """Exp2 sub01 → sub02, bilateral hippocampus."""
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=RouteLearningExp2s01Hippo,
            target_assembly_class=RouteLearningExp2s02Hippo,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug,
        )


BENCHMARK_REGISTRY["RouteLearningExp2s01toExp2s02Hippo"] = (
    RouteLearningExp2s01toExp2s02Hippo
)


class RouteLearningExp2s02toExp2s01Hippo(AssemblyBenchmarkScorer):
    """Exp2 sub02 → sub01, bilateral hippocampus (reverse)."""
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=RouteLearningExp2s02Hippo,
            target_assembly_class=RouteLearningExp2s01Hippo,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug,
        )


BENCHMARK_REGISTRY["RouteLearningExp2s02toExp2s01Hippo"] = (
    RouteLearningExp2s02toExp2s01Hippo
)


class RouteLearningExp2s01toExp2s03Hippo(AssemblyBenchmarkScorer):
    """Exp2 sub01 → sub03, bilateral hippocampus."""
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=RouteLearningExp2s01Hippo,
            target_assembly_class=RouteLearningExp2s03Hippo,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug,
        )


BENCHMARK_REGISTRY["RouteLearningExp2s01toExp2s03Hippo"] = (
    RouteLearningExp2s01toExp2s03Hippo
)