#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --constraint=GPUMEM80GB
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=23:59:59

module load miniforge3
conda activate blink2-cuda

python train.py -c config_train.yaml