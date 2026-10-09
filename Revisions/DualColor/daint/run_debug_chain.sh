#!/usr/bin/env bash
# Run the extraction of the dual-color datasets as a chain of debug jobs on Daint, from the
# local machine (the debug partition allows one running and two submitted jobs per user):
#   nohup Revisions/DualColor/daint/run_debug_chain.sh DFK785 DFK788 DFK789 > chain.log 2>&1 &
# Each step is submitted when the previous one has COMPLETED; any other end state stops the
# chain. A dataset step already marked done in $SCRATCH/dualcolor/runs/<ds>.<step>.done is skipped.
# MODE=gt405|gt488 runs the paper's ground-truth mode (extract_step.sbatch third argument; done
# markers <ds>_<mode>.<step>.done); STEPS overrides the step list, e.g. STEPS="paramfinder localize".
set -uo pipefail
R='cd ~/blinkognition_revisions'
MODE=${MODE:-}; STEPS=${STEPS:-paramfinder localize extract rest}
for ds in "$@"; do
  tag=$ds${MODE:+_$MODE}
  for step in $STEPS; do
    if ssh -n -o BatchMode=yes daint "test -f \$SCRATCH/dualcolor/runs/$tag.$step.done" 2>/dev/null; then
      echo "$(date +%H:%M) $tag $step already done"; continue
    fi
    jid=$(ssh -n -o BatchMode=yes daint "$R && sbatch --parsable Revisions/DualColor/daint/extract_step.sbatch $ds $step $MODE" 2>/dev/null | tail -1)
    [[ "$jid" =~ ^[0-9]+$ ]] || { echo "$(date +%H:%M) $tag $step: submit failed ($jid)"; exit 1; }
    echo "$(date +%H:%M) $tag $step submitted as $jid"
    while ssh -n -o BatchMode=yes daint "squeue -h -j $jid" 2>/dev/null | grep -q .; do sleep 30; done
    state=$(ssh -n -o BatchMode=yes daint "sacct -n -X -j $jid -o State,Elapsed" 2>/dev/null | head -1 | xargs)
    echo "$(date +%H:%M) $tag $step job $jid: $state"
    [[ "$state" == COMPLETED* ]] || { echo "stopping chain"; exit 1; }
    ssh -n -o BatchMode=yes daint "touch \$SCRATCH/dualcolor/runs/$tag.$step.done"
  done
done
echo "$(date +%H:%M) chain finished"
