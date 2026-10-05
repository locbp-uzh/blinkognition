#!/bin/bash
# Classify the DFK785 traces with the final model on the same GPU type and precision it was
# trained on (the Wasserstein threshold is precision-sensitive, see docs/classify.md), after a
# check on the model's own test set. Submit from ML/:
#   sbatch ../Revisions/DFK785/ml/submit_classify_DFK785.sh
#SBATCH --job-name=classify_dfk785
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --constraint=GPUMEM80GB
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Batch shells never ran conda init: set up lmod and conda here (templates/gpu_job.sh).
source /etc/profile
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate blink2-cuda || { echo "conda activate blink2-cuda failed" >&2; exit 1; }
echo "Node: $(hostname)  Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
set -e

MODEL=../Results/Train/2026-10-05_12-57-46_HaloD106_SNAPC148_cnngru_minmax_mirror_final_training_HaloD106_SNAPC148
CHECK_IN=../Results/Revisions/DFK785/classify/testset_inputs
python ../Revisions/DFK785/ml/testset_check.py export --model-dir "$MODEL" --out "$CHECK_IN"
python classify.py -c ../Revisions/DFK785/ml/config_classify_testset.yaml
RUN=$(ls -d ../Results/Revisions/DFK785/classify/*_testset_check | tail -1)
python ../Revisions/DFK785/ml/testset_check.py compare --ref "$CHECK_IN/training_reference.csv" --run "$RUN"
python classify.py -c ../Revisions/DFK785/ml/config_classify_DFK785.yaml
