#!/bin/bash
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks (or tasks per node)
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --constraint=GPUMEM80GB        # Constraints the job to A100 or H100 GPUs
#SBATCH --gres=gpu:1                   # Number of GPUs
#SBATCH --mem=128G                      # Memory per node
#SBATCH --time=23:59:59                # Maximum execution time (HH:MM:SS)

python train.py -c config_train.yaml