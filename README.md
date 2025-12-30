# BBScore: Brain-Behavior Scoring Framework

BBScore is a comprehensive framework for benchmarking deep learning models against neural (fMRI, ephys) and behavioral datasets. It handles the complex pipeline of loading model weights, preprocessing stimuli (images/videos), extracting feature activations, and scoring them against biological data.

## 🚀 Quick Start (Demos)

If you just want to see how the data looks or run a simple analysis without installing everything locally, try these Colab notebooks:

- **Simple Notebook For Loading MongoDB Data and Plotting**  
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m3VQtF7RKvEXv4Qi1pyidwOtJ8woIpex?usp=sharing)

- **Data Analysis Notebook (By Josh Wilson)**  
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w9yTRZONhnucXbrbCzw5qn1HbyXxp314?usp=sharing)

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd BBScore
```

### 2. Set up a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.
```bash
# Create the environment
python3 -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Crucial!)
BBScore needs to know where to save heavy files (datasets and model weights). You **must** set the `SCIKIT_LEARN_DATA` variable.

**Linux/macOS:**
```bash
# Replace /path/to/storage with a real path on your disk with ample space (50GB+)
export SCIKIT_LEARN_DATA="/path/to/storage/bbscore_data"
export RESULTS_PATH="/path/to/storage/bbscore_results" # Optional, defaults to SCIKIT_LEARN_DATA
```

To make this permanent, add those lines to your `~/.bashrc` or `~/.zshrc`.

---

## 🏃‍♀️ How to Run a Benchmark

The main entry point is `run.py`.

### Basic Usage
```bash
python run.py --model <model_name> --layer <layer_name> --benchmark <benchmark_name> --metric <metric_name>
```

**Example:**
Run a Ridge Regression benchmark using the `videomae_base` model on the `NSD` dataset.
```bash
python run.py --model videomae_base --layer encoder.layer.11 --benchmark NSDV1Shared --metric ridge
```

### Key Arguments
*   `--model`: The identifier of the model (e.g., `resnet50`, `videomae_base`).
*   `--layer`: The specific layer to extract features from (e.g., `layer4`, `encoder.layer.11`).
*   `--benchmark`: The dataset/task identifier (e.g., `NSDV1Shared`, `MajajHong2015V4`).
*   `--metric`: The scoring method (e.g., `ridge`, `rsa`, `pls`).
*   `--batch-size`: Batch size for feature extraction (default: 4).
*   `--debug`: Runs without saving to the database/remote tracking.

### Running Batch Evaluations (`eval.py`)
If you want to run many combinations defined in a script:
```bash
python eval.py
```
*Note: You can modify the `experiments` list inside `eval.py` to customize your batch run.*

---

## 🔍 What Models/Benchmarks are available?

To see a list of all registered components available in your current installation, run this Python snippet:

```python
from models import MODEL_REGISTRY
from benchmarks import BENCHMARK_REGISTRY
from metrics import METRICS

print("--- Available Models ---")
print("\n".join(sorted(MODEL_REGISTRY.keys())))

print("\n--- Available Benchmarks ---")
print("\n".join(sorted(BENCHMARK_REGISTRY.keys())))

print("\n--- Available Metrics ---")
print("\n".join(sorted(METRICS.keys())))
```

---

## 📂 Project Structure

*   `benchmarks/`: Definitions tying data and scoring together (e.g., `NSD`, `Algonauts`).
*   `data/`: Scripts to download and preprocess datasets (Stimuli and Neural assemblies).
*   `metrics/`: Mathematical implementations of scores (Ridge, RSA, PLS).
*   `models/`: Wrappers for deep learning models (HuggingFace, TorchVision, Custom).
*   `mongo_utils/`: Helpers for database injection (Advanced use).
