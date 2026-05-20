#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --gres=gpu:3
#SBATCH --constraint=GPUMEM80GB
#SBATCH --mem=256G
#SBATCH --time=23:59:59

# Usage:
#   sbatch --job-name=<name> submit_cv.sh <config_file>
#
# The config file path is relative to the ML/ directory.

CONFIG=${1:?"Usage: sbatch --job-name=<name> submit_cv.sh <config_file>"}

module load miniforge3
conda activate blink2-cuda

N_GPUS=$(nvidia-smi -L | wc -l)
echo "Config : ${CONFIG}"
echo "GPUs   : ${N_GPUS}"

python crossval.py -c "${CONFIG}" --n-gpus ${N_GPUS}
