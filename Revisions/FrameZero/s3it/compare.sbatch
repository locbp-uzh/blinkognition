#!/bin/bash
# FrameZero: compare.py (rerun against the published extraction) on an S3IT CPU node. Run from the
# root of the review clone (after rerun.sbatch has finished):
#
#   cd ~/blinkognition_review1 && sbatch Revisions/FrameZero/s3it/compare.sbatch
#
# picasso-env = the published run's environment.
#SBATCH --job-name=framezero_cmp
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
source /etc/profile
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /home/pabriv/data/conda/envs/picasso-env || { echo "conda activate picasso-env failed" >&2; exit 1; }
set -euo pipefail
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "$(date '+%F %T') host $(hostname), repo $(git log --oneline -1)"
python -c "import picasso, numpy, sklearn; print('picasso', picasso.__version__, 'numpy', numpy.__version__, 'sklearn', sklearn.__version__)"
python Revisions/FrameZero/compare.py
echo "$(date '+%F %T') done"
