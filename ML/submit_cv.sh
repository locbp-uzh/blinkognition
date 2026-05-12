#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH --constraint=GPUMEM80GB
#SBATCH --mem=256G
#SBATCH --time=23:59:59

# Usage (called by submit_all_cv.sh):
#   sbatch --job-name=<name> submit_cv.sh <config_file>
#
# The config file path is relative to the ML/ directory.

CONFIG=${1:?"Usage: sbatch submit_cv.sh <config_file>"}

N_GPUS=$(nvidia-smi -L | wc -l)
echo "Config : ${CONFIG}"
echo "GPUs   : ${N_GPUS}"

/home/pabriv/data/conda/envs/blink2-cuda/bin/python crossval.py -c "${CONFIG}" --n-gpus ${N_GPUS}
