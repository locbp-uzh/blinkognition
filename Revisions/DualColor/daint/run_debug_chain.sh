#!/usr/bin/env bash
# Run the extraction of the dual-color datasets as a chain of debug jobs on Daint, from the
# local machine (the debug partition allows one running and two submitted jobs per user):
#   nohup Revisions/DualColor/daint/run_debug_chain.sh DFK785 DFK788 DFK789 > chain.log 2>&1 &
# Each step is submitted when the previous one has COMPLETED; any other end state stops the
# chain. A dataset step already marked done in $SCRATCH/dualcolor/runs/<ds>.<step>.done is skipped.
set -uo pipefail
R='cd ~/blinkognition_revisions'
for ds in "$@"; do
  for step in paramfinder localize extract rest; do
    if ssh -o BatchMode=yes daint "test -f \$SCRATCH/dualcolor/runs/$ds.$step.done" 2>/dev/null; then
      echo "$(date +%H:%M) $ds $step already done"; continue
    fi
    jid=$(ssh -o BatchMode=yes daint "$R && sbatch --parsable Revisions/DualColor/daint/extract_step.sbatch $ds $step" 2>/dev/null | tail -1)
    [[ "$jid" =~ ^[0-9]+$ ]] || { echo "$(date +%H:%M) $ds $step: submit failed ($jid)"; exit 1; }
    echo "$(date +%H:%M) $ds $step submitted as $jid"
    while ssh -o BatchMode=yes daint "squeue -h -j $jid" 2>/dev/null | grep -q .; do sleep 30; done
    state=$(ssh -o BatchMode=yes daint "sacct -n -X -j $jid -o State,Elapsed" 2>/dev/null | head -1 | xargs)
    echo "$(date +%H:%M) $ds $step job $jid: $state"
    [[ "$state" == COMPLETED* ]] || { echo "stopping chain"; exit 1; }
    ssh -o BatchMode=yes daint "touch \$SCRATCH/dualcolor/runs/$ds.$step.done"
  done
done
echo "$(date +%H:%M) chain finished"
