#!/bin/bash
#SBATCH --job-name=deit_small
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/lianeozo/logs/bbscore_deit_rsa_%j.out
#SBATCH --error=/scratch/users/lianeozo/logs/bbscore_deit_rsa_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bbscore

export SCIKIT_LEARN_DATA=/scratch/users/lianeozo/bbscore_data

cd ~/bbscore_public

mkdir -p /scratch/users/lianeozo/logs

run_experiment() {
    local model=$1
    local layer=$2
    local benchmark=$3
    local metric=$4

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

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "SUCCESS: $model | $layer | $benchmark | $metric"
    else
        echo "FAILED:  $model | $layer | $benchmark | $metric"
        echo "Exit code: $exit_code — aborting job."
        exit 1
    fi

    echo "End time: $(date)"
    echo ""
}

DEIT_SMALL_LAYERS=(
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

echo "########################################################"
echo "Starting DeiT-Small temporal_rsa runs"
echo "########################################################"

for layer in "${DEIT_SMALL_LAYERS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        run_experiment "deit_small_imagenet_full_seed-0" "$layer" "$benchmark" "temporal_rsa"
    done
done

echo "########################################################"
echo "All DeiT-Small temporal_rsa runs complete: $(date)"
echo "########################################################"