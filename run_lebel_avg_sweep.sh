#!/usr/bin/env bash
# Run story-averaged LeBel benchmark for multiple layers (base and/or instruct).
# Use on FarmShare with: conda activate bbscore, set HF_TOKEN, SCIKIT_LEARN_DATA (or RESULTS_PATH).
#
# Usage:
#   ./run_lebel_avg_sweep.sh base          # layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
#   ./run_lebel_avg_sweep.sh instruct       # same layers for instruct
#   ./run_lebel_avg_sweep.sh both           # base then instruct
#
# Optional: set LAYERS="0 16 30" to run only those layers.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default layers (0..30, step 2; exclude 31)
LAYERS="${LAYERS:-0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30}"

# Use full path to bbscore conda Python to avoid SLURM losing conda activate
BBSCORE_PYTHON="${HOME}/miniconda3/envs/bbscore/bin/python"
if [ -x "$BBSCORE_PYTHON" ]; then
  PYTHON="${PYTHON:-$BBSCORE_PYTHON}"
else
  PYTHON="${PYTHON:-python}"
fi
echo "Using Python: $PYTHON"
"$PYTHON" -c "import numpy" || { echo "ERROR: numpy not found in $PYTHON - wrong env"; exit 1; }

run_model() {
  local model=$1
  for layer in $LAYERS; do
    echo "========== ${model} layer ${layer} =========="
    "$PYTHON" run_lebel_avg.py --model "$model" --layer "$layer" || true
  done
}

case "${1:-base}" in
  base)    run_model "llama3_8b_base" ;;
  instruct) run_model "llama3_8b_instruct" ;;
  both)    run_model "llama3_8b_base"; run_model "llama3_8b_instruct" ;;
  *)       echo "Usage: $0 base|instruct|both"; exit 1 ;;
esac
