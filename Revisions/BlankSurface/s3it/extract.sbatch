#!/bin/bash
# One BlankSurface extraction (R3.1) on UZH S3IT: stage the slides of one vesicle channel at one protein
# gradient (stage.py), then the paper's pipeline from the localize step (Extraction/run_pipeline.py), with
# synchronous Picasso MLE fits (picasso_sync/). Run from the root of the review clone:
#
#   cd ~/blinkognition_review1 && sbatch Revisions/BlankSurface/s3it/extract.sbatch 488 10000
#
# picasso-env = the published run's environment (Picasso 0.9.3). The run config and link map are copied
# into the pipeline's run folder at the end.
#SBATCH --job-name=blanksurface
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
source /etc/profile
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /home/pabriv/data/conda/envs/picasso-env || { echo "conda activate picasso-env failed" >&2; exit 1; }
set -euo pipefail
GT=${1:?vesicle channel 488|405}
GRAD=${2:?protein gradient}
export PYTHONPATH=$PWD/Revisions/BlankSurface/picasso_sync BLANKSURFACE_PICASSO_SYNC_MLE=1 QT_QPA_PLATFORM=offscreen
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "$(date '+%F %T') host $(hostname), repo $(git log --oneline -1), gt $GT, gradient $GRAD"
python -c "import picasso, numpy, sklearn; print('picasso', picasso.__version__, 'numpy', numpy.__version__, 'sklearn', sklearn.__version__)"
python -c "import picasso.gaussmle as g; print('gaussmle_async patched:', g.gaussmle_async.__name__ == '_gaussmle_sync')"
CFG=$(python Revisions/BlankSurface/stage.py --gt "$GT" --gradient "$GRAD")
echo "config $CFG"
python Extraction/run_pipeline.py -c "$CFG" --start-from localize
OUTBASE=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_folder_base'])" "$CFG")
RUN=$(ls -td "$OUTBASE"/*/ | head -1)
cp "$CFG" "$RUN/blanksurface_config.yaml"
cp "$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['input_folder'])" "$CFG")/link_map.tsv" "$RUN/"
echo "run folder $RUN"
echo "$(date '+%F %T') done"
