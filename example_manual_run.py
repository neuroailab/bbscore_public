import os
import numpy as np
import torch

from benchmarks.BBS import BenchmarkScore
from data.NSDShared import NSDStimulusSet, NSDAssemblyV1


def test_pipeline(model_identifier, layer_name, metric_name):
    print(
        f"Testing with model: {model_identifier}, layer: {layer_name}, metric: {metric_name}")

    # Setup Benchmark Manually
    pipeline = BenchmarkScore(
        NSDStimulusSet,
        model_identifier=model_identifier,
        layer_name=layer_name,
        assembly_class=NSDAssemblyV1,
        batch_size=32,  # Adjust batch size as needed
        num_workers=0   # Adjust for your system (0 is good for debugging)
    )

    # Add the desired metric (using ridge as an example)
    pipeline.add_metric(metric_name)

    # Run the pipeline
    results = pipeline.run()

    print("\n--- Results ---")
    print(f"\n{metric_name.upper()} Metric Results:")
    for key, value in results['metrics'].items():
        print(f"  {key}: {value}")

    print("Pipeline test completed.")


if __name__ == "__main__":
    # Example usage:
    test_pipeline(model_identifier="videomae_base",
                  layer_name="encoder.layer.11", metric_name="ridge")
