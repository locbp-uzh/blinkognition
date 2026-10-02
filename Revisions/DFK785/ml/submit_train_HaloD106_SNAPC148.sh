#!/bin/bash
# Train the final Halo-D106 vs SNAP-C148 model for the DFK785 mixture experiment on UZH S3IT.
# Submit from ML/:  sbatch ../Revisions/DFK785/ml/submit_train_HaloD106_SNAPC148.sh
#SBATCH --job-name=train_halo_snap_final
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --constraint=GPUMEM80GB
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=23:59:00

module load miniforge3
conda activate blink2-cuda

python train.py -c ../Revisions/DFK785/ml/config_train_HaloD106_SNAPC148.yaml
