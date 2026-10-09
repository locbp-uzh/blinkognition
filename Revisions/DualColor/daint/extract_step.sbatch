#!/bin/bash
# One step of the extraction of one dual-color dataset, sized for the 30-min debug partition
# (run_debug_chain.sh submits them in order: paramfinder, localize, extract, rest):
#   sbatch Revisions/DualColor/daint/extract_step.sbatch <dataset> <paramfinder|localize|extract|rest>
# The steps are those of Extraction/run_pipeline.py: paramfinder (with the input staging),
# localize (prints the run folder, saved to $SCRATCH/dualcolor/runs/<dataset>.txt), extract,
# and rest = run_pipeline.py --start-from combine (combine, filter, diagnose).
#SBATCH --job-name=dc_step
#SBATCH --account=uzh67
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=Revisions/DualColor/daint/logs/%x-%j.out
set -euo pipefail
DS=${1:?dataset}; STEP=${2:?step}
REPO=$HOME/blinkognition_revisions
WORK=$SCRATCH/dualcolor
mkdir -p "$WORK/stage" "$WORK/results" "$WORK/runs"
[ -e "$REPO/Inputs/DualColor_640only" ] || ln -s "$WORK/stage" "$REPO/Inputs/DualColor_640only"
[ -e "$REPO/Results" ] || ln -s "$WORK/results" "$REPO/Results"
CFG=../Revisions/DualColor/extraction/config_640only_$DS.yaml
RUNFILE=$WORK/runs/$DS.txt
echo "Dataset $DS step $STEP on $(hostname) $(date +%H:%M:%S), repo $(git -C "$REPO" log --oneline -1)"
case $STEP in
  paramfinder) CMD="cd $REPO && DUALCOLOR_DATA_ROOT=$SCRATCH/dual_color python Revisions/DualColor/stage_inputs.py $DS --stage-root Inputs/DualColor_640only && cd Extraction && python paramfinder.py -c $CFG" ;;
  localize)    CMD="cd $REPO/Extraction && python localize.py -c $CFG 2>&1 | tee $WORK/runs/$DS.localize.log; grep -o 'Output folder: .*' $WORK/runs/$DS.localize.log | tail -1 | sed 's/Output folder: //' > $RUNFILE; test -s $RUNFILE" ;;
  extract)     CMD="cd $REPO/Extraction && python extract.py -c $CFG -r \$(cat $RUNFILE)" ;;
  rest)        CMD="cd $REPO/Extraction && python run_pipeline.py -c $CFG --start-from combine -r \$(cat $RUNFILE)" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
# Picasso 0.8.8 race in the CLI MLE fit (loses spots with many threads): see picasso_sync/sitecustomize.py
FIX="export PYTHONPATH=$REPO/Revisions/DualColor/daint/picasso_sync DUALCOLOR_PICASSO_SYNC_MLE=1"
srun --environment=picasso-env bash -c "set -euo pipefail; $FIX; $CMD"
echo "Done $DS $STEP $(date +%H:%M:%S)"
