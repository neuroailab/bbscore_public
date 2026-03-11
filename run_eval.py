"""
run_eval.py

Runs RSA comparisons between SSAST and Wav2Vec2 on LeBel2023Audio for
Heschl's Gyrus (A1). Run this on Friday after:
  1. Your spectrogram conversion script has been run on the full LeBel stimuli
  2. Lillian's SSAST wrapper is merged into the ssast branch

Usage:
    python run_eval.py

BEFORE RUNNING — Lillian needs to fill in:
  - SSAST_LAYERS: the layer names her wrapper exposes
  - WAV2VEC2_LAYERS: the corresponding Wav2Vec2 layers to compare against

Results are saved to results/rsa_results.csv
"""

import subprocess
import csv
import os
from pathlib import Path
from datetime import datetime

# ── Subjects ──────────────────────────────────────────────────────────────────
SUBJECTS = ["UTS01", "UTS02", "UTS03"]

# ── Layers ────────────────────────────────────────────────────────────────────
# Lillian: fill in the exact layer name strings your wrapper exposes.
# Aim for a spread: early, middle, and late transformer blocks.
# Example format (replace with real names):
#   "patch_embed"           ← before transformer blocks (earliest)
#   "blocks.2"              ← early transformer block
#   "blocks.5"              ← middle
#   "blocks.9"              ← late
#   "blocks.11"             ← deepest

SSAST_LAYERS = [
    "FILL_IN_EARLY_LAYER",    # e.g. patch embedding / block 0-2
    "FILL_IN_MIDDLE_LAYER",   # e.g. block 5-6
    "FILL_IN_LATE_LAYER",     # e.g. block 10-11
]

# Wav2Vec2 baseline layers — mirroring early/middle/late
# These are the standard layer names for wav2vec2_base in BBScore
WAV2VEC2_LAYERS = [
    "_orig_mod.encoder.layers.0",   # early
    "_orig_mod.encoder.layers.5",   # middle
    "_orig_mod.encoder.layers.11",  # late
]

# ── Config ────────────────────────────────────────────────────────────────────
METRIC    = "rsa"
BENCHMARK = "LeBel2023Audio"   # will be formatted as LeBel2023Audio{subject}

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / f"rsa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def run_benchmark(model: str, layer: str, subject: str) -> dict:
    """Run a single BBScore eval and return a result dict."""
    benchmark = f"{BENCHMARK}{subject}"
    cmd = [
        "python", "run.py",
        "--model",     model,
        "--layer",     layer,
        "--benchmark", benchmark,
        "--metric",    METRIC,
    ]

    print(f"\n{'='*60}")
    print(f"  Model:     {model}")
    print(f"  Layer:     {layer}")
    print(f"  Subject:   {subject}")
    print(f"  Benchmark: {benchmark}")
    print(f"{'='*60}")

    result = {
        "model":     model,
        "layer":     layer,
        "subject":   subject,
        "benchmark": benchmark,
        "rsa_score": None,
        "status":    "pending",
        "error":     "",
    }

    try:
        output = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(output.stdout)

        # Parse RSA score from BBScore output
        # BBScore prints something like "Score: 0.1234" — adjust if format differs
        for line in output.stdout.splitlines():
            if "score" in line.lower():
                parts = line.split()
                for part in parts:
                    try:
                        result["rsa_score"] = float(part)
                        break
                    except ValueError:
                        continue

        result["status"] = "success"

    except subprocess.CalledProcessError as e:
        print(f"  ERROR: {e.stderr}")
        result["status"] = "error"
        result["error"]  = e.stderr.strip().splitlines()[-1] if e.stderr else "unknown"

    return result


def main():
    # Warn if layers haven't been filled in
    unfilled = [l for l in SSAST_LAYERS if "FILL_IN" in l]
    if unfilled:
        print("WARNING: SSAST_LAYERS contains placeholders. Fill these in before running:")
        for l in unfilled:
            print(f"  {l}")
        print("Continuing with Wav2Vec2 only for now...\n")

    results = []

    # ── SSAST runs ────────────────────────────────────────────────────────────
    real_ssast_layers = [l for l in SSAST_LAYERS if "FILL_IN" not in l]
    for subject in SUBJECTS:
        for layer in real_ssast_layers:
            result = run_benchmark(model="ssast", layer=layer, subject=subject)
            results.append(result)
            save_results(results)  # save after each run in case of crash

    # ── Wav2Vec2 baseline runs ─────────────────────────────────────────────────
    for subject in SUBJECTS:
        for layer in WAV2VEC2_LAYERS:
            result = run_benchmark(model="wav2vec2_base", layer=layer, subject=subject)
            results.append(result)
            save_results(results)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Layer':<35} {'Subject':<10} {'RSA Score':<10} {'Status'}")
    print("-" * 90)
    for r in results:
        score = f"{r['rsa_score']:.4f}" if r['rsa_score'] is not None else "N/A"
        print(f"{r['model']:<20} {r['layer']:<35} {r['subject']:<10} {score:<10} {r['status']}")

    print(f"\nFull results saved to: {RESULTS_FILE}")


def save_results(results: list):
    """Save results to CSV after each run so nothing is lost if something crashes."""
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "layer", "subject", "benchmark", "rsa_score", "status", "error"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
