#!/bin/bash
# FrameZero: the paper's extraction (extract, combine, filter; no diagnose) rerun on the published
# localizations with Picasso's removal of groups touching frame 0 switched off (keep_edges/). Run from the
# root of the review clone:
#
#   cd ~/blinkognition_review1 && sbatch Revisions/FrameZero/s3it/rerun.sbatch
#
# picasso-env = the published run's environment (Picasso 0.9.3).
#SBATCH --job-name=framezero
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=23:59:00
#SBATCH --output=slurm-%j.out
source /etc/profile
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /home/pabriv/data/conda/envs/picasso-env || { echo "conda activate picasso-env failed" >&2; exit 1; }
set -euo pipefail
export PYTHONPATH=$PWD/Revisions/FrameZero/keep_edges FRAMEZERO_KEEP_EDGE_GROUPS=1 QT_QPA_PLATFORM=offscreen
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "$(date '+%F %T') host $(hostname), repo $(git log --oneline -1)"
python -c "import picasso, numpy, sklearn; print('picasso', picasso.__version__, 'numpy', numpy.__version__, 'sklearn', sklearn.__version__)"
python -c "import picasso.postprocess as p; print('link patched:', p.link.__name__ == '_link_keep_edges')"
RUN=$(python Revisions/FrameZero/stage.py)
echo "run folder $RUN"
for step in extract combine filter; do
  echo "$(date '+%F %T') $step"
  python Extraction/$step.py -c "$RUN/framezero_config.yaml" -r "$RUN"
done
echo "$(date '+%F %T') done"
