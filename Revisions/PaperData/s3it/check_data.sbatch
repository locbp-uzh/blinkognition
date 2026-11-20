#!/bin/bash
# Step 0 of the first-round review on UZH S3IT: Revisions/PaperData/check_data.py over the paper's
# published extraction, read in place (never written to). Run from the root of the review clone:
#
#   cd ~/blinkognition_review1 && sbatch Revisions/PaperData/s3it/check_data.sbatch [check_data.py args]
#
# picasso-env is the environment that wrote the published pickles (Picasso 0.9.3) and has nd2.
# Outputs: Results/Revisions/PaperData/check/run_NNN/ in the clone; fetch with rsync.
#SBATCH --job-name=paperdata_check
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
source /etc/profile
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /home/pabriv/data/conda/envs/picasso-env || { echo "conda activate picasso-env failed" >&2; exit 1; }
set -euo pipefail
echo "$(date '+%F %T') host $(hostname), repo $(git log --oneline -1)"
python -c "import numpy, pandas, nd2; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'nd2', nd2.__version__)"
python Revisions/PaperData/check_data.py "$@"
echo "$(date '+%F %T') done"
