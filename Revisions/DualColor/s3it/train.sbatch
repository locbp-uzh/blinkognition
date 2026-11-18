#!/bin/bash
# Train and apply train_pure.py models on UZH S3IT, one GPU, all tasks of the job concurrently on it
# (TRAINING_NOTES.md point 13: DFK851 design B moved here from Daint, which was in maintenance on
# 2026-11-18). The S3IT counterpart of daint/train.sbatch, with the same task syntax:
#   cd <clone> && sbatch Revisions/DualColor/s3it/train.sbatch <config.yaml> [...] [apply:<out>:<DS>=<ta run>:<run,run,...>]
# A config runs train_pure.py -c <config>; an apply: token runs apply_models.py (trained runs applied
# unchanged to that dataset's mixed traces); an optional fifth field, comma-separated, adds options:
# passes (--save-passes), compare (--compare), e.g. apply:<out>:<DS>=<ta run>:<runs>:passes,compare. Conda env blink2-cuda (torch 2.5.1, as the Daint container).
# Needs Results/ with the inputs that daint/sync_inputs.sh sends (SYNC_DEST=cluster:<clone>/) and, for
# apply:, the model run folders it names (Results/Revisions/DualColor/models/*_<run>).
# Logs: Revisions/DualColor/s3it/logs/train_<name>_<jobid>.log, one per task.
#SBATCH --job-name=dc_train
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=Revisions/DualColor/s3it/logs/%x-%j.out
source /etc/profile
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /home/pabriv/data/conda/envs/blink2-cuda || { echo "conda activate blink2-cuda failed" >&2; exit 1; }
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: sbatch train.sbatch <config.yaml|apply:...> [...]" >&2; exit 1; }
echo "$(date '+%F %T') tasks: $* on $(hostname), repo $(git log --oneline -1)"
python -c "import torch, numpy, pandas, sklearn; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
mkdir -p Revisions/DualColor/s3it/logs
pids=(); rc=0
for cfg in "$@"; do
  if [[ $cfg == apply:* ]]; then   # apply:<out>:<DS>=<ta run>:<run,run,...>: apply_models.py
    IFS=: read -r _ out dsr runs flags <<< "$cfg"; name=apply_$(basename "$out")
    cmd=(python Revisions/DualColor/apply_models.py --runs ${runs//,/ } --datasets "${dsr%%=*}" --ta-runs "$dsr" --out "$out")
    [[ ,$flags, == *,passes,* ]] && cmd+=(--save-passes)
    [[ ,$flags, == *,compare,* ]] && cmd+=(--compare)
  else
    name=$(basename "$cfg" .yaml); cmd=(python Revisions/DualColor/train_pure.py -c "$cfg")
  fi
  "${cmd[@]}" > "Revisions/DualColor/s3it/logs/train_${name}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "$(date '+%F %T') done, rc $rc"
exit $rc
