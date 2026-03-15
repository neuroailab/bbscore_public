#!/bin/bash
#SBATCH --job-name=bbscore_vit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/lianeozo/logs/bbscore_vit_%j.out
#SBATCH --error=/scratch/users/lianeozo/logs/bbscore_vit_%j.err

# ── Environment ──────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bbscore

export SCIKIT_LEARN_DATA=/scratch/users/lianeozo/bbscore_data

cd ~/bbscore_public

mkdir -p /scratch/users/lianeozo/logs

RESULTS_DIR=/scratch/users/lianeozo/bbscore_data/results

# ── Helper function ───────────────────────────────────────────────────────────
run_experiment() {
    local model=$1
    local layer=$2
    local benchmark=$3
    local metric=$4

    local safe_layer=$(echo "$layer" | tr '/' '_')
    local result_file="${RESULTS_DIR}/${model}_${safe_layer}_${benchmark}.pkl"

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
        --batch-size 4

    if [ $? -eq 0 ]; then
        echo "SUCCESS: $model | $layer | $benchmark | $metric"
    else
        echo "FAILED:  $model | $layer | $benchmark | $metric"
    fi

    echo "End time: $(date)"
    echo ""
}

# ── Layer and benchmark lists ─────────────────────────────────────────────────

VIT_BASE_LAYERS=(
    "_orig_mod.vit.encoder.layer.0"
    "_orig_mod.vit.encoder.layer.1"
    "_orig_mod.vit.encoder.layer.2"
    "_orig_mod.vit.encoder.layer.3"
    "_orig_mod.vit.encoder.layer.4"
    "_orig_mod.vit.encoder.layer.5"
    "_orig_mod.vit.encoder.layer.6"
    "_orig_mod.vit.encoder.layer.7"
    "_orig_mod.vit.encoder.layer.8"
    "_orig_mod.vit.encoder.layer.9"
    "_orig_mod.vit.encoder.layer.10"
    "_orig_mod.vit.encoder.layer.11"
)

BENCHMARKS=(
    "TVSDV110msBins"
    "TVSDV410msBins"
    "TVSDIT10msBins"
)

# ── ViT-Base runs ─────────────────────────────────────────────────────────────
echo "########################################################"
echo "Starting ViT-Base runs"
echo "########################################################"

for layer in "${VIT_BASE_LAYERS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        run_experiment "vit_base" "$layer" "$benchmark" "temporal_rsa"
    done
done

echo "########################################################"
echo "All ViT-Base runs complete: $(date)"
echo "########################################################"