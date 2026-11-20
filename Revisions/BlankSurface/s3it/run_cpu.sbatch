#!/bin/bash
# A BlankSurface analysis script (count.py, projections.py) on an S3IT CPU node, in picasso-env (the published
# run's environment). Run from the root of the review clone:
#
#   cd ~/blinkognition_review1 && sbatch Revisions/BlankSurface/s3it/run_cpu.sbatch count.py
#SBATCH --job-name=blanksurface_cpu
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
source /etc/profile
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /home/pabriv/data/conda/envs/picasso-env || { echo "conda activate picasso-env failed" >&2; exit 1; }
set -euo pipefail
SCRIPT=${1:?script in Revisions/BlankSurface}
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "$(date '+%F %T') host $(hostname), repo $(git log --oneline -1), script $SCRIPT"
python "Revisions/BlankSurface/$SCRIPT"
echo "$(date '+%F %T') done"
