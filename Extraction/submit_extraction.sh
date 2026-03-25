#!/bin/bash
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks (or tasks per node)
#SBATCH --cpus-per-task=16              # Number of CPU cores per task
#SBATCH --mem=256G                      # Memory per node
#SBATCH --time=23:59:59                # Maximum execution time (HH:MM:SS)

python run_pipeline.py -c config.yaml