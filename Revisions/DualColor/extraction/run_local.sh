#!/usr/bin/env bash
# The paper extraction (640-only) of one dual-color dataset on this Mac: the local counterpart of
# daint/extract_step.sbatch + daint/run_debug_chain.sh, with the same steps, config and Picasso fix.
#
#   Revisions/DualColor/extraction/run_local.sh <dataset> <stage root> [n_workers] [gt405|gt488]
#
# With gt405 / gt488: the paper's ground-truth mode on the single-protein slides of that vesicle
# channel, for paper_in.py (TRAINING_NOTES.md point 7), as extract_step.sbatch's gt mode: config
# config_gt<ch>_<dataset>.yaml, staging with stage_inputs.py --gt <ch>, then paramfinder and localize
# only (paper_in.py reads the vesicle-channel localizations; extract/combine/filter are not needed).
# Outputs Results/Revisions/DualColor/<dataset>/Extraction_gt<ch>/<run>; run file <dataset>_gt<ch>.txt.
# Use a separate stage root per gt mode (e.g. .../stage_gt405 and .../stage_gt488): both modes keep
# their Optuna study and optimized_localization_params_gt.yaml in <stage root>/<dataset>/<dataset>.
#
# <stage root> must allow symlinks (the MediumBerth volume is exFAT and does not): the movies are
# staged there by stage_inputs.py and Picasso writes its _locs files next to the links. A copy of
# extraction/config_640only_<dataset>.yaml with input_folder pointing into the stage root and the
# given n_workers (default 8) is written there; every other setting is the repository config's.
# Outputs go where the config says (Results/Revisions/DualColor/<dataset>/Extraction/<run>). The run
# folder is saved to <stage root>/runs/<dataset>.txt; one log per step in <stage root>/runs/.
set -euo pipefail
DS=${1:?dataset}; STAGE=${2:?stage root}; NW=${3:-8}; MODE=${4:-}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PY=/Users/privera/opt/anaconda3/envs/picasso-env/bin/python
STAGE=$(mkdir -p "$STAGE" && cd "$STAGE" && pwd)
mkdir -p "$STAGE/runs"
case $MODE in
  "") CFGNAME=config_640only_$DS; TAG=$DS; GTSUF=no_gt; GTARG=() ;;
  gt405|gt488) CFGNAME=config_${MODE}_$DS; TAG=${DS}_$MODE; GTSUF=gt; GTARG=(--gt "${MODE#gt}") ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
CFG=$STAGE/$CFGNAME.local.yaml
sed -e "s|^input_folder: .*|input_folder: $STAGE/$DS   # local staging (run_local.sh)|" \
    -e "s|^n_workers: .*|n_workers: $NW  # local run (run_local.sh)|" \
    "$REPO/Revisions/DualColor/extraction/$CFGNAME.yaml" > "$CFG"
# Picasso 0.8.8 race in the CLI MLE fit (loses spots with many threads): see daint/picasso_sync/sitecustomize.py
export PYTHONPATH=$REPO/Revisions/DualColor/daint/picasso_sync DUALCOLOR_PICASSO_SYNC_MLE=1 QT_QPA_PLATFORM=offscreen
LOG=$STAGE/runs/$TAG
echo "$TAG: staging $(date +%H:%M:%S), repo $(git -C "$REPO" log --oneline -1)"
# VESICLES_RUN=run_NNN (gt modes): pair FOVs with that vesicles run instead of the trace_assignment's
[ -n "$MODE" ] && [ -n "${VESICLES_RUN:-}" ] && GTARG+=(--vesicles-run "$VESICLES_RUN")
(cd "$REPO" && "$PY" Revisions/DualColor/stage_inputs.py "$DS" --stage-root "$STAGE" ${GTARG[@]+"${GTARG[@]}"}) > "$LOG.stage.log" 2>&1
# paramfinder keeps its Optuna study in the staged experiment folder and LOADS it if present: a rerun
# must start a fresh study
TS=$(date +%Y%m%d_%H%M%S)
for f in optimization_$GTSUF.db optimized_localization_params_$GTSUF.yaml; do
  [ -e "$STAGE/$DS/$DS/$f" ] && mv "$STAGE/$DS/$DS/$f" "$STAGE/$DS/$DS/$f.before_$TS" && echo "moved aside $f"
done
cd "$REPO/Extraction"
echo "$TAG: paramfinder $(date +%H:%M:%S)"; "$PY" paramfinder.py -c "$CFG" > "$LOG.paramfinder.log" 2>&1
# GT_SNAPSHOTS_ONLY=1 (gt modes only): localize the vesicle-channel snapshots alone. paper_in.py reads only
# those (its 640 ROIs are the 640-only run's), so the 640 localization of a gt run is never used; the 640
# links move into a subfolder (find_nd2_files is non-recursive) after the paramfinder, which needs them.
# Round 6 localized five of its six gt runs this way (checks/20261117_round6_paper_in, relocalize).
if [ -n "$MODE" ] && [ "${GT_SNAPSHOTS_ONLY:-0}" = 1 ]; then
  for d in "$STAGE/$DS/$DS"/*/; do
    mkdir -p "$d/_640_not_localized"
    for f in "$d"*_640nm_*; do [ -e "$f" ] && mv "$f" "$d/_640_not_localized/"; done
  done
  echo "$TAG: GT_SNAPSHOTS_ONLY, $(ls "$STAGE/$DS/$DS"/*/*.nd2 | wc -l | tr -d ' ') snapshot movies to localize"
fi
echo "$TAG: localize $(date +%H:%M:%S)"; "$PY" localize.py -c "$CFG" > "$LOG.localize.log" 2>&1
grep -o 'Output folder: .*' "$LOG.localize.log" | tail -1 | sed 's/Output folder: //' > "$STAGE/runs/$TAG.txt"
test -s "$STAGE/runs/$TAG.txt"
RUN=$(cat "$STAGE/runs/$TAG.txt")
# localize.py reports failed movies but exits 0; a failed movie has no localizations (paper_in.py would
# score all its ROIs OUT), so stop here and name it
if grep -q 'Localization complete: [0-9]* succeeded, [1-9][0-9]* failed' "$LOG.localize.log"; then
  grep 'Failed to process' "$LOG.localize.log" | sed 's/: Picasso localization failed.*//' >&2
  echo "$TAG: localize had failed movies (above); rerun or fix them before using $RUN" >&2; exit 3
fi
if [ -n "$MODE" ]; then echo "$TAG: done $(date +%H:%M:%S) -> $RUN"; exit 0; fi
echo "$DS: extract $(date +%H:%M:%S) -> $RUN"; "$PY" extract.py -c "$CFG" -r "$RUN" > "$LOG.extract.log" 2>&1
echo "$DS: combine, filter, diagnose $(date +%H:%M:%S)"
"$PY" run_pipeline.py -c "$CFG" --start-from combine -r "$RUN" > "$LOG.rest.log" 2>&1
echo "$DS: done $(date +%H:%M:%S)"
