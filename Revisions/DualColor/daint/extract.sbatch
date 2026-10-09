#!/bin/bash
# Extract the 640 traces of one dual-color dataset on Daint with the paper pipeline
# (Extraction/run_pipeline.py, single-channel mode) inside the picasso-env container.
#   cd ~/blinkognition_revisions && sbatch Revisions/DualColor/daint/extract.sbatch DFK788
# Raw data: $SCRATCH/dual_color/<data_folder> (rsynced from /Volumes/MediumBerth/Agent/dual_color).
# Staged links and Results/ live on scratch (purged after 30 days untouched: fetch promptly).
#SBATCH --job-name=dc_extract
#SBATCH --account=uzh67
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=Revisions/DualColor/daint/logs/%x-%j.out
set -euo pipefail
DS=${1:?usage: sbatch extract.sbatch <dataset>}
REPO=$HOME/blinkognition_revisions
WORK=$SCRATCH/dualcolor
mkdir -p "$WORK/stage" "$WORK/results"
[ -e "$REPO/Inputs/DualColor_640only" ] || ln -s "$WORK/stage" "$REPO/Inputs/DualColor_640only"
[ -e "$REPO/Results" ] || ln -s "$WORK/results" "$REPO/Results"
echo "Dataset $DS on $(hostname), repo $(git -C "$REPO" log --oneline -1)"
srun --environment=picasso-env bash -c "
  set -euo pipefail
  cd $REPO
  export DUALCOLOR_DATA_ROOT=$SCRATCH/dual_color
  python Revisions/DualColor/stage_inputs.py $DS --stage-root Inputs/DualColor_640only
  cd Extraction
  python run_pipeline.py -c ../Revisions/DualColor/extraction/config_640only_$DS.yaml
"
