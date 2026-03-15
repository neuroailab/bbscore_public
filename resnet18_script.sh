#!/bin/bash
#SBATCH --job-name=resnet_18
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/lianeozo/logs/bbscore_%j.out
#SBATCH --error=/scratch/users/lianeozo/logs/bbscore_%j.err

# ── Environment ──────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bbscore

export SCIKIT_LEARN_DATA=/scratch/users/lianeozo/bbscore_data

cd ~/bbscore_public

# Create log directory if it doesn't exist
mkdir -p /scratch/users/lianeozo/logs

RESULTS_DIR=/scratch/users/lianeozo/bbscore_data/results

# ── Helper function ───────────────────────────────────────────────────────────
run_experiment() {
    local model=$1
    local layer=$2
    local benchmark=$3
    local metric=$4

    # Build expected results filename — matches BBScore naming convention
    local safe_layer=$(echo "$layer" | tr '/' '_')
    local result_file="${RESULTS_DIR}/${model}_${safe_layer}_${benchmark}.pkl"

    # Skip if result already exists
    if [ -f "$result_file" ]; then
        echo "SKIPPING (already done): $model | $layer | $benchmark | $metric"
        return
    fi

    echo "========================================================"
    echo "Running: model=$model | layer=$layer"
    echo "         benchmark=$benchmark | metric=$metric"
    echo "Start time: $(date)"
    echo "========================================================"

    python run.py \
        --model "$model" \
        --layer "$layer" \
        --benchmark "$benchmark" \
        --metric "$metric" \
        --batch-size 32

    if [ $? -eq 0 ]; then
        echo "SUCCESS: $model | $layer | $benchmark | $metric"
    else
        echo "FAILED:  $model | $layer | $benchmark | $metric"
    fi

    echo "End time: $(date)"
    echo ""
}

# ── Define layer lists ────────────────────────────────────────────────────────

RESNET18_LAYERS=(
        "layer1.0.bn1"
        "layer3.0.conv2"
        "layer2.0.bn2"
        "layer4.0.bn1"
        "layer1"
        "layer2"
        "layer3"
        "layer4"
)

BENCHMARKS=(
    "TVSDV110msBins"
    "TVSDV410msBins"
    "TVSDIT10msBins"
)

METRIC="temporal_rsa"

# ── ResNet-18 runs ────────────────────────────────────────────────────────────
echo "########################################################"
echo "Starting ResNet-18 runs"
echo "########################################################"

for layer in "${RESNET18_LAYERS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        run_experiment "resnet18_imagenet_full" "$layer" "$benchmark" "$METRIC"
    done
done

echo "########################################################"
echo "All runs complete: $(date)"
echo "########################################################"