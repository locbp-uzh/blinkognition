#!/bin/bash
# The DualColor 640-only extraction of one dataset on UZH S3IT (TRAINING_NOTES.md point 13): the Mac chain
# extraction/run_local.sh unchanged (staging, paramfinder, localize, extract, combine/filter/diagnose, the
# Picasso MLE patch), in conda env picasso088 = Picasso 0.8.8 with the package versions of the Mac
# picasso-env that extracted DFK842/843/846 (S3IT's picasso-env has Picasso 0.9.3). The raw data are read in
# place (DUALCOLOR_DATA_ROOT, default the experimenter's share); the staging root needs symlinks (scratch).
#
#   cd <clone> && sbatch Revisions/DualColor/s3it/extract.sbatch <dataset> [n_workers] [gt405|gt488]
#
# With gt405 / gt488 the ground-truth-mode runs for paper_in.py (run_local.sh's gt modes; set
# VESICLES_RUN=run_NNN to stage from a vesicles run before any trace assignment exists, and
# GT_SNAPSHOTS_ONLY=1 to localize only the vesicle-channel snapshots, as round 6).
#
# Outputs: <clone>/Results/Revisions/DualColor/<dataset>/Extraction/<run>; the link map and step logs in the
# staging root (<stage>/<dataset>/link_map.tsv, <stage>/runs/); copy the link map to
# Inputs/DualColor_640only/<dataset>/link_map.tsv where assign_traces.py runs.
#SBATCH --job-name=dc_extract
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
source /etc/profile
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /home/pabriv/data/conda/envs/picasso088 || { echo "conda activate picasso088 failed" >&2; exit 1; }
set -euo pipefail
DS=${1:?dataset}
NW=${2:-${SLURM_CPUS_PER_TASK:-8}}
MODE=${3:-}
STAGE=${STAGE_ROOT:-/scratch/$USER/dualcolor_stage/stage_${MODE:-640only}}
export DUALCOLOR_DATA_ROOT=${DUALCOLOR_DATA_ROOT:-/shares/locbp.chem.uzh/dkossm/Blinkognition}
export PY=$(command -v python)
# one worker process per core: no nested thread pools (numba / BLAS would otherwise size to the whole node)
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "$(date '+%F %T') host $(hostname), $NW workers, python $PY"
"$PY" -c "import picasso, numpy, numba, scipy; print('picasso', picasso.__version__, 'numpy', numpy.__version__, 'numba', numba.__version__, 'scipy', scipy.__version__)"
echo "data root $DUALCOLOR_DATA_ROOT, stage $STAGE, repo $(git log --oneline -1)"
Revisions/DualColor/extraction/run_local.sh "$DS" "$STAGE" "$NW" ${MODE:+"$MODE"}
echo "$(date '+%F %T') done"
