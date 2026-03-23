#!/bin/bash
#SBATCH --job-name=download_tvsd
#SBATCH --partition=normal
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/users/lianeozo/logs/download_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bbscore
export SCIKIT_LEARN_DATA=/scratch/users/lianeozo/bbscore_data

cd ~/bbscore_public
python -c "
from data.TVSD import TVSDAssemblyV110msBins
a = TVSDAssemblyV110msBins()
a.prepare_data(train=True)
print('Done')
"
