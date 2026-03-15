#!/bin/bash
#SBATCH --job-name=bbscore_deit_distilled
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/lianeozo/logs/bbscore_deit_distilled_%j.out
#SBATCH --error=/scratch/users/lianeozo/logs/bbscore_deit_distilled_%j.err

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

DEIT_DISTILLED_LAYERS=(
    "blocks.0"
    "blocks.1"
    "blocks.2"
    "blocks.3"
    "blocks.4"
    "blocks.5"
    "blocks.6"
    "blocks.7"
    "blocks.8"
    "blocks.9"
    "blocks.10"
    "blocks.11"
)

BENCHMARKS=(
    "TVSDV110msBins"
    "TVSDV410msBins"
    "TVSDIT10msBins"
)

# ── DeiT-Small Distilled runs ─────────────────────────────────────────────────
echo "########################################################"
echo "Starting DeiT-Small Distilled runs"
echo "########################################################"

for layer in "${DEIT_DISTILLED_LAYERS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        run_experiment "deit_small_distilled_patch16_224" "$layer" "$benchmark" "temporal_rsa"
    done
done

echo "########################################################"
echo "All DeiT-Small Distilled runs complete: $(date)"
echo "########################################################"