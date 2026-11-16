#!/usr/bin/env bash
# The paper extraction (640-only) of one dual-color dataset on this Mac: the local counterpart of
# daint/extract_step.sbatch + daint/run_debug_chain.sh, with the same steps, config and Picasso fix.
#
#   Revisions/DualColor/extraction/run_local.sh <dataset> <stage root> [n_workers]
#
# <stage root> must allow symlinks (the MediumBerth volume is exFAT and does not): the movies are
# staged there by stage_inputs.py and Picasso writes its _locs files next to the links. A copy of
# extraction/config_640only_<dataset>.yaml with input_folder pointing into the stage root and the
# given n_workers (default 8) is written there; every other setting is the repository config's.
# Outputs go where the config says (Results/Revisions/DualColor/<dataset>/Extraction/<run>). The run
# folder is saved to <stage root>/runs/<dataset>.txt; one log per step in <stage root>/runs/.
set -euo pipefail
DS=${1:?dataset}; STAGE=${2:?stage root}; NW=${3:-8}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PY=/Users/privera/opt/anaconda3/envs/picasso-env/bin/python
STAGE=$(mkdir -p "$STAGE" && cd "$STAGE" && pwd)
mkdir -p "$STAGE/runs"
CFG=$STAGE/config_640only_${DS}.local.yaml
sed -e "s|^input_folder: .*|input_folder: $STAGE/$DS   # local staging (run_local.sh)|" \
    -e "s|^n_workers: .*|n_workers: $NW  # local run (run_local.sh)|" \
    "$REPO/Revisions/DualColor/extraction/config_640only_$DS.yaml" > "$CFG"
# Picasso 0.8.8 race in the CLI MLE fit (loses spots with many threads): see daint/picasso_sync/sitecustomize.py
export PYTHONPATH=$REPO/Revisions/DualColor/daint/picasso_sync DUALCOLOR_PICASSO_SYNC_MLE=1 QT_QPA_PLATFORM=offscreen
LOG=$STAGE/runs/$DS
echo "$DS: staging $(date +%H:%M:%S), repo $(git -C "$REPO" log --oneline -1)"
(cd "$REPO" && "$PY" Revisions/DualColor/stage_inputs.py "$DS" --stage-root "$STAGE") > "$LOG.stage.log" 2>&1
# paramfinder keeps its Optuna study in the staged experiment folder and LOADS it if present: a rerun
# must start a fresh study
TS=$(date +%Y%m%d_%H%M%S)
for f in optimization_no_gt.db optimized_localization_params_no_gt.yaml; do
  [ -e "$STAGE/$DS/$DS/$f" ] && mv "$STAGE/$DS/$DS/$f" "$STAGE/$DS/$DS/$f.before_$TS" && echo "moved aside $f"
done
cd "$REPO/Extraction"
echo "$DS: paramfinder $(date +%H:%M:%S)"; "$PY" paramfinder.py -c "$CFG" > "$LOG.paramfinder.log" 2>&1
echo "$DS: localize $(date +%H:%M:%S)"; "$PY" localize.py -c "$CFG" > "$LOG.localize.log" 2>&1
grep -o 'Output folder: .*' "$LOG.localize.log" | tail -1 | sed 's/Output folder: //' > "$STAGE/runs/$DS.txt"
test -s "$STAGE/runs/$DS.txt"
RUN=$(cat "$STAGE/runs/$DS.txt")
echo "$DS: extract $(date +%H:%M:%S) -> $RUN"; "$PY" extract.py -c "$CFG" -r "$RUN" > "$LOG.extract.log" 2>&1
echo "$DS: combine, filter, diagnose $(date +%H:%M:%S)"
"$PY" run_pipeline.py -c "$CFG" --start-from combine -r "$RUN" > "$LOG.rest.log" 2>&1
echo "$DS: done $(date +%H:%M:%S)"
