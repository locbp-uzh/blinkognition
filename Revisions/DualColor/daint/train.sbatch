#!/bin/bash
# Train and apply train_pure.py models on one Daint node (4x GH200) inside the blink2-cuda
# container, one config per GPU, up to four at a time:
#   cd ~/blinkognition_revisions && sbatch Revisions/DualColor/daint/train.sbatch Revisions/DualColor/ml/<a>.yaml [<b>.yaml ...]
# An argument that is a model run folder (Results/Revisions/DualColor/models/<run>, optionally
# <run>:<n_mc>) runs classify_background.py on it instead (the background-inference control).
# Needs Results/ (a symlink to $SCRATCH/dualcolor/results, made by extract.sbatch) with the
# extraction runs and, rsynced from local, the trace_assignment runs and (for background
# traces) the vesicles runs they name (fovs.csv, manifest.yaml).
#SBATCH --job-name=dc_train
#SBATCH --account=uzh67
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=Revisions/DualColor/daint/logs/%x-%j.out
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: sbatch train.sbatch <config.yaml> [...]" >&2; exit 1; }
REPO=$HOME/blinkognition_revisions
echo "Configs: $* on $(hostname), repo $(git -C "$REPO" log --oneline -1)"
srun --environment=blink2-cuda bash -c '
  cd '"$REPO"'
  nvidia-smi --query-gpu=index,name --format=csv,noheader
  i=0; pids=(); rc=0
  for cfg in '"$*"'; do
    [ ${#pids[@]} -ge 4 ] && { wait "${pids[0]}" || rc=1; pids=("${pids[@]:1}"); }
    if [ -d "${cfg%:*}" ]; then      # a trained model run[:n_mc]: the background-inference control on it
      run=${cfg%:*}; name=bginf_$(basename "$run"); cmd=(python Revisions/DualColor/classify_background.py "$run" --n-boot 0)
      [ "$run" != "$cfg" ] && cmd+=(--n-mc "${cfg##*:}")
    else
      name=$(basename "$cfg" .yaml); cmd=(python Revisions/DualColor/train_pure.py -c "$cfg")
    fi
    CUDA_VISIBLE_DEVICES=$((i % 4)) "${cmd[@]}" > "Revisions/DualColor/daint/logs/train_${name}_${SLURM_JOB_ID}.log" 2>&1 &
    pids+=($!); i=$((i + 1))
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done; exit $rc
'
