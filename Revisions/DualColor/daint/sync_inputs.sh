#!/usr/bin/env bash
# Send the inputs of train_pure.py runs from this Mac to Daint (Results/ there is a link into
# $SCRATCH/dualcolor/results, which the 30-day scratch purge empties):
#   Revisions/DualColor/daint/sync_inputs.sh <config.yaml> [--dry-run]
# For every dataset the config loads (train, test and holdout datasets) it sends the pinned
# trace_assignment run (trace_assignment_runs, else the latest), the pinned paper_in run (with
# train_label_filter: paper_in) and the *_filtered_<input_channel>.pkl traces (minmax by default) of the
# extraction run that the trace_assignment manifest names; nothing else (no movies, no localizations). AppleDouble files
# are skipped. Run from anywhere; paths are taken relative to the repository root.
set -euo pipefail
CFG=${1:?config}; shift || true
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PY=/Users/privera/opt/anaconda3/envs/blink2env/bin/python
LIST=$(mktemp)
"$PY" - "$REPO" "$CFG" > "$LIST" <<'EOF'
import sys
from pathlib import Path
import yaml
repo, cfg = Path(sys.argv[1]), yaml.safe_load(open(sys.argv[2]))
sys.path.insert(0, str(repo / "Revisions/DualColor"))
from dc import load_config, run_dir
dss = dict.fromkeys(cfg["train_datasets"] + cfg["test_datasets"] + list(cfg.get("holdout_datasets") or []))
for ds in dss:
    c = load_config(ds)
    ta = run_dir(c, "trace_assignment", (cfg.get("trace_assignment_runs") or {}).get(ds))
    print(ta.relative_to(repo))
    ext = repo / yaml.safe_load(open(ta / "manifest.yaml"))["extraction_run"]
    for p in sorted((ext / "ProteinTraces" / "Filtered").glob(f"*_filtered_{cfg.get('input_channel', 'minmax')}.pkl")):
        if not p.name.startswith("._"):
            print(p.relative_to(repo))
    if cfg.get("train_label_filter") == "paper_in":
        print(run_dir(c, "paper_in", (cfg.get("paper_in_runs") or {}).get(ds)).relative_to(repo))
EOF
echo "$(wc -l < "$LIST") paths:"; cat "$LIST"
# -R keeps the paths, -K follows the Results link on Daint as a directory
cd "$REPO" && rsync -aRK --exclude '._*' --exclude '.DS_Store' "$@" --files-from="$LIST" -r . daint:blinkognition_revisions/
rm -f "$LIST"
