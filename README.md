# BBScore: Neural Benchmarking Framework

BBScore is a framework for benchmarking deep learning models against neural (fMRI, electrophysiology) and behavioral datasets. It handles model loading, stimulus preprocessing, feature extraction, and comparison with biological data.

=======
- **Simple Notebook For Loading MongoDB Data and Plotting**
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14T5IuhZJ6PZaOvhpUMH-uIFGrzyVb7Wt?usp=sharing)

- **Data Analysis Notebook (By Josh Wilson)**
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Hwkro4UnmqsXWso6MEz25P1b3a0RWpbg?usp=sharing)

## Quick Start

### 1. Check Your System

Before installing, check if your machine can run BBScore:

```bash
python check_system.py --quick
```

For a detailed check with a specific configuration:
```bash
python check_system.py --model resnet50 --benchmark OnlineTVSDV1 --metric ridge
```

### 2. Install (Recommended: Use the Install Script)

```bash
# Make the script executable
chmod +x install.sh

# Run the installer
./install.sh
```

Or install manually:
```bash
# Create conda environment
conda create -n bbscore python=3.11 -y
conda activate bbscore

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
# Required: Set data directory (50GB+ free space recommended)
export SCIKIT_LEARN_DATA="/path/to/your/data/bbscore_data"

# Add to your shell config for persistence
echo 'export SCIKIT_LEARN_DATA="/path/to/your/data/bbscore_data"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Run Your First Benchmark

```bash
# Simple example with a small model
python run.py --model resnet18 --layer layer4 --benchmark V1SineGratingsBenchmark --metric ridge

# Video benchmark with online metric (lower memory)
python run.py --model resnet50 --layer layer4 --benchmark OnlineTVSDV1 --metric online_linear_regressor
```

---

## For Students: Step-by-Step Guide

### What You Need

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16+ GB |
| GPU | None (CPU works) | 4+ GB VRAM |
| Disk | 50 GB | 100+ GB |
| Python | 3.9+ | 3.11 |

**No GPU?** Use:
- `ridge` metric instead of `online_linear_regressor`
- Smaller models (`resnet18`, `efficientnet_b0`)
- Online benchmarks (`OnlineTVSDV1`, `OnlineTVSDV4`)

### Recommended Workflow

1. **Start with small experiments:**
   ```bash
   python run.py --model resnet18 --layer layer4 --benchmark OnlineTVSDV1 --metric ridge
   ```

2. **Scale up gradually:**
   ```bash
   python run.py --model dinov2_base --layer blocks.11 --benchmark OnlineTVSDV1 --metric online_linear_regressor
   ```

3. **Using OnlineLinearRegressor:**
   ```python
   # Default: MSE loss with L2 regularization
   from metrics import OnlineLinearRegressor

   metric = OnlineLinearRegressor(
       input_feature_dim=768,
       loss_type='mse',  # Default: MSE + L2 regularization
       n_epochs=100,
   )
   ```

---

## Available Components

### Benchmarks

| Benchmark | Type | Memory | Description |
|-----------|------|--------|-------------|
| `OnlineTVSDV1` | Video | Low | Macaque V1 neural responses |
| `OnlineTVSDV4` | Video | Low | Macaque V4 neural responses |
| `OnlineTVSDIT` | Video | Low | Macaque IT neural responses |
| `NSDV1Shared` | Image | Medium | Human fMRI V1 (NSD dataset) |
| `V1SineGratingsBenchmark` | Image | Very Low | Synthetic V1 gratings |
| `SSV2Benchmark` | Video | High | Something-Something-V2 |
| `LeBel2023` | Text/fMRI | Low | Language comprehension fMRI |

### Models (Examples)

| Model | Parameters | VRAM | Type |
|-------|------------|------|------|
| `resnet18` | 11M | 2 GB | Image |
| `resnet50` | 26M | 3 GB | Image |
| `dinov2_base` | 86M | 4 GB | Image |
| `dinov2_large` | 304M | 8 GB | Image |
| `videomae_base` | 87M | 8 GB | Video |
| `clip_vit_b32` | 151M | 4 GB | Image |

### Metrics

| Metric | GPU Required | Description |
|--------|--------------|-------------|
| `ridge` | No | Ridge regression (sklearn) |
| `online_linear_regressor` | Yes | Online ridge with SGD and L2 regularization |
| `pls` | No | Partial Least Squares |
| `rsa` | No | Representational Similarity Analysis |

---

## Command Reference

### Basic Run
```bash
python run.py --model <MODEL> --layer <LAYER> --benchmark <BENCHMARK> --metric <METRIC>
```

### Common Options
```bash
--batch-size 8       # Adjust based on your GPU memory
--device cuda:0      # Specify GPU
```

### Examples

```bash
# Image model on NSD (human fMRI)
python run.py --model resnet50 --layer layer4 --benchmark NSDV1Shared --metric ridge

# Video model on TVSD (macaque ephys)
python run.py --model videomae_base --layer encoder.layer.11 --benchmark OnlineTVSDV1 --metric online_linear_regressor

# DINO on V4
python run.py --model dinov2_base --layer blocks.11 --benchmark OnlineTVSDV4 --metric ridge

# Fast alpha search for high-dimensional features
python run.py --model dinov2_large --layer blocks.23 --benchmark NSDV1Shared --metric ridge --subsample-features-for-alpha 2000
```

### List Available Options
```bash
python check_system.py --list
```

---

## Troubleshooting

### Out of Memory (GPU)
```bash
# Reduce batch size
python run.py ... --batch-size 2

# Use CPU
python run.py ... --device cpu --metric ridge
```

### Out of Memory (RAM)
- Use `Online*` benchmarks instead of standard ones
- Use smaller models

### Slow Training
- Install cuML for 50x faster ridge regression: https://docs.rapids.ai/api/cuml/stable/
- Use `--subsample-features-for-alpha 2000` for faster alpha search

### Dataset Download Issues
- Ensure `SCIKIT_LEARN_DATA` is set to a writable directory
- Check you have enough disk space
- Some datasets require AWS credentials (see `data/` folder)

---

## Loss Functions for OnlineLinearRegressor

When using `OnlineLinearRegressor`, you can choose different loss functions:

| Loss Type | Description |
|-----------|-------------|
| `mse` | **Default** - Mean squared error with L2 regularization |
| `correlation` | Pearson correlation loss |
| `combined` | MSE + correlation (tune `correlation_weight`) |
| `ccc` | Concordance Correlation Coefficient (combines correlation + scale) |
| `ccc_mse` | CCC + MSE combined |

**Example:**
```python
from metrics import OnlineLinearRegressor

# Default configuration (MSE + L2)
metric = OnlineLinearRegressor(
    input_feature_dim=768,
    loss_type='mse',  # Default
    n_epochs=100,
)

# Alternative: CCC loss is also available
metric_ccc = OnlineLinearRegressor(
    input_feature_dim=768,
    loss_type='ccc',
    n_epochs=100,
)
```

---

## Project Structure

```
bbscore_public/
├── benchmarks/          # Benchmark definitions (NSD, TVSD, Physion, etc.)
├── data/                # Dataset loaders and downloaders
├── metrics/             # Scoring methods (ridge, RSA, PLS, online)
│   └── losses.py        # Loss functions (MSE, CCC, Pearson, etc.)
├── models/              # Model wrappers (HuggingFace, TorchVision)
├── run.py               # Main entry point
├── eval.py              # Batch evaluation script
├── check_system.py      # System diagnostic tool
├── install.sh           # Installation script
└── requirements.txt     # Python dependencies
```

---

## Getting Help

1. **Check your system:** `python check_system.py`
2. **List options:** `python check_system.py --list`

