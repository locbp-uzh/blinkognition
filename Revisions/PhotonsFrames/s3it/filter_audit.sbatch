#!/bin/bash
# Revisions/PhotonsFrames/filter_audit.py on UZH S3IT, reading the published extraction in place
# (--remote). Run from the root of the review clone:
#
#   cd ~/blinkognition_review1 && sbatch Revisions/PhotonsFrames/s3it/filter_audit.sbatch --where background
#
# picasso-env is the environment that wrote the published run. n_jobs in config.yaml (16) matches the
# CPUs requested. Outputs: Results/Revisions/PhotonsFrames/filter_audit[_background]/run_NNN/ in the clone.
#SBATCH --job-name=filter_audit
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
source /etc/profile
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /home/pabriv/data/conda/envs/picasso-env || { echo "conda activate picasso-env failed" >&2; exit 1; }
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "$(date '+%F %T') host $(hostname), repo $(git log --oneline -1)"
python -c "import numpy, pandas, sklearn, joblib; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__, 'joblib', joblib.__version__)"
python Revisions/PhotonsFrames/filter_audit.py --remote "$@"
echo "$(date '+%F %T') done"
