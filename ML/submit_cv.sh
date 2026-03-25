#!/bin/bash
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks (or tasks per node)
#SBATCH --cpus-per-task=16             # Number of CPU cores per task (increased for multi-GPU)
#SBATCH --gres=gpu:4                   # Number of GPUs for parallel augmentation sweep
#SBATCH --constraint=GPUMEM80GB        # Constraints the job to A100 or H100 GPUs
#SBATCH --mem=256G                     # Memory per node (increased for multi-GPU)
#SBATCH --time=23:59:59                # Maximum execution time (HH:MM:SS)

# ============================================================================
# Cross-validation with optional augmentation sweep
# ============================================================================
#
# Standard mode (single GPU, no augmentation sweep):
#   python crossval.py -c config_cv.yaml
#
# Augmentation sweep mode (multi-GPU):
#   python crossval.py -c config_cv.yaml --models tcn,resnet1d --aug-factors 0,2,3,5 --n-gpus 4
#
# Or configure via config_cv.yaml:
#   augmentation_sweep:
#     enabled: true
#     factors: [0, 2, 3, 5]
#   system:
#     n_gpus: 4
# ============================================================================

# Detect number of GPUs available
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected ${N_GPUS} GPUs"

# Run cross-validation with augmentation sweep
python crossval.py -c config_cv.yaml --n-gpus ${N_GPUS}
