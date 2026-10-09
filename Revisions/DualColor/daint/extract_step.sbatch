#!/bin/bash
# One step of the extraction of one dual-color dataset, sized for the 30-min debug partition
# (run_debug_chain.sh submits them in order: paramfinder, localize, extract, rest):
#   sbatch Revisions/DualColor/daint/extract_step.sbatch <dataset> <paramfinder|localize|extract|rest> [gt405|gt488]
# With gt405 / gt488: the paper's ground-truth mode on the single-protein slides of that vesicle
# channel (config_gt<ch>_<dataset>.yaml, staging Inputs/DualColor_gt<ch>, run file <dataset>_gt<ch>.txt).
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
DS=${1:?dataset}; STEP=${2:?step}; MODE=${3:-}
REPO=$HOME/blinkognition_revisions
WORK=$SCRATCH/dualcolor
if [ -n "$MODE" ]; then
  case $MODE in gt405|gt488) ;; *) echo "unknown mode $MODE" >&2; exit 2 ;; esac
  STAGE=stage_$MODE; INPUTS=Inputs/DualColor_$MODE; CFG=../Revisions/DualColor/extraction/config_${MODE}_$DS.yaml
  TAG=${DS}_$MODE; GTSUF=gt; GTARG="--gt ${MODE#gt}"
else
  STAGE=stage; INPUTS=Inputs/DualColor_640only; CFG=../Revisions/DualColor/extraction/config_640only_$DS.yaml
  TAG=$DS; GTSUF=no_gt; GTARG=""
fi
mkdir -p "$WORK/$STAGE" "$WORK/results" "$WORK/runs"
[ -e "$REPO/$INPUTS" ] || ln -s "$WORK/$STAGE" "$REPO/$INPUTS"
[ -e "$REPO/Results" ] || ln -s "$WORK/results" "$REPO/Results"
RUNFILE=$WORK/runs/$TAG.txt
echo "Dataset $TAG step $STEP on $(hostname) $(date +%H:%M:%S), repo $(git -C "$REPO" log --oneline -1)"
case $STEP in
  # paramfinder keeps its Optuna study in the staged experiment folder and LOADS it if present
  # (load_if_exists), replaying old trials: a rerun must start a fresh study.
  paramfinder) STUDY=$WORK/$STAGE/$DS/$DS; TS=$(date +%Y%m%d_%H%M%S)
               for f in optimization_$GTSUF.db optimized_localization_params_$GTSUF.yaml; do
                 [ -e "$STUDY/$f" ] && mv "$STUDY/$f" "$STUDY/$f.before_$TS" && echo "moved aside $f"; done
               CMD="cd $REPO && DUALCOLOR_DATA_ROOT=$SCRATCH/dual_color python Revisions/DualColor/stage_inputs.py $DS --stage-root $INPUTS $GTARG && cd Extraction && python paramfinder.py -c $CFG" ;;
  localize)    CMD="cd $REPO/Extraction && python localize.py -c $CFG 2>&1 | tee $WORK/runs/$TAG.localize.log; grep -o 'Output folder: .*' $WORK/runs/$TAG.localize.log | tail -1 | sed 's/Output folder: //' > $RUNFILE; test -s $RUNFILE" ;;
  extract)     CMD="cd $REPO/Extraction && python extract.py -c $CFG -r \$(cat $RUNFILE)" ;;
  rest)        CMD="cd $REPO/Extraction && python run_pipeline.py -c $CFG --start-from combine -r \$(cat $RUNFILE)" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
# Picasso 0.8.8 race in the CLI MLE fit (loses spots with many threads): see picasso_sync/sitecustomize.py
FIX="export PYTHONPATH=$REPO/Revisions/DualColor/daint/picasso_sync DUALCOLOR_PICASSO_SYNC_MLE=1"
srun --environment=picasso-env bash -c "set -euo pipefail; $FIX; $CMD"
echo "Done $TAG $STEP $(date +%H:%M:%S)"
