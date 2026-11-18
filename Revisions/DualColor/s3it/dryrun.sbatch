#!/bin/bash
# train_pure.py --dry-run of configs on UZH S3IT (CPU), in the env and on the inputs that s3it/train.sbatch
# will use, with output_root moved to <out dir> so that no run folder lands in models/. Each run's
# data_split.csv and log are then compared with the local dry run's (checks/20261118_dfk851/dryrun_check.py
# writes those) before any training (TRAINING_NOTES.md point 13). Also records the open-file limit (the
# mirror's loky workers: Daint job 5377506 failed on "Too many open files").
#   cd <clone> && sbatch Revisions/DualColor/s3it/dryrun.sbatch <out dir> <config.yaml> [...]
#SBATCH --job-name=dc_dryrun
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=Revisions/DualColor/s3it/logs/%x-%j.out
source /etc/profile
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /home/pabriv/data/conda/envs/blink2-cuda || { echo "conda activate blink2-cuda failed" >&2; exit 1; }
set -euo pipefail
OUT=${1:?out dir}; shift
mkdir -p "$OUT/cfg" "$OUT/logs"
echo "$(date '+%F %T') $(hostname), repo $(git log --oneline -1), open-file limit $(ulimit -n), cpus ${SLURM_CPUS_PER_TASK:-?}"
python -c "import torch, numpy, pandas, sklearn, joblib; print('torch', torch.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__, 'joblib', joblib.__version__)"
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  python - "$cfg" "$OUT" <<'EOF'
import sys, yaml
from pathlib import Path
cfg, out = Path(sys.argv[1]), Path(sys.argv[2])
c = yaml.safe_load(open(cfg))
c["output_root"] = str(out / "runs" / cfg.stem)
(out / "cfg" / cfg.name).write_text(yaml.safe_dump(c, sort_keys=False))
EOF
  python Revisions/DualColor/train_pure.py -c "$OUT/cfg/$name.yaml" --dry-run > "$OUT/logs/$name.log" 2>&1 \
    && echo "$name: $(grep 'Dry run:' "$OUT/logs/$name.log")" || { echo "$name failed, see $OUT/logs/$name.log"; exit 1; }
done
echo "$(date '+%F %T') done"
