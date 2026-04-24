#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --constraint=GPUMEM80GB
#SBATCH --mem=256G
#SBATCH --time=23:59:59
#SBATCH --job-name=loocv_tcn
#SBATCH --output=loocv_%j.out
#SBATCH --error=loocv_%j.err

# ============================================================================
# Nested leave-one-out CV across DK experiments (20 training runs, multi-GPU)
#
# Usage:
#   sbatch submit_loocv.sh
#
# Or run directly on the login node (single GPU):
#   python loocv.py -c config_loocv.yaml --n-gpus 1
# ============================================================================

N_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected ${N_GPUS} GPUs"

python loocv.py -c config_loocv.yaml --n-gpus ${N_GPUS}
