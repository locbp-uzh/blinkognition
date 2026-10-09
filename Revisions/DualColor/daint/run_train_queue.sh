#!/usr/bin/env bash
# Submit train.sbatch jobs to the Daint debug partition from the local machine, one per line
# of a queue file, as slots free (debug allows two submitted jobs per user):
#   nohup Revisions/DualColor/daint/run_train_queue.sh <queue file> > <log> 2>&1 &
# Each line holds the arguments of one job (configs, or model runs for classify_background.py).
# A token @bginf:<run_name> stands for the latest model run *_<run_name> on Daint, used as a
# background-inference task: the line waits until that run has finished ('Done:' in its log),
# and its MC passes are cut from the model's n_mc (100) so that the ~25,400 background traces
# take at most BGINF_MC_BUDGET_S seconds at the MC time per trace the run measured on its test
# set (deterministic pass and the bootstrap summary are not included: --n-boot 0 on Daint).
set -uo pipefail
QUEUE=$1
BUDGET=${BGINF_MC_BUDGET_S:-900}
N_BG=25400
R='cd ~/blinkognition_revisions'
M=Results/Revisions/DualColor/models
d() { ssh -n -o BatchMode=yes daint "$@" 2>/dev/null; }   # -n: do not eat the queue on stdin

resolve() {   # @bginf:<run_name> -> <run dir>[:<n_mc>]
  local name=${1#@bginf:} run t n
  until run=$(d "$R && ls -d $M/*_$name 2>/dev/null | tail -1") && [ -n "$run" ] \
        && d "$R && grep -q '^Done:' $run/$name.log"; do sleep 60; done
  t=$(d "$R && grep 'MC time per sample' $run/$name.log | tail -1 | awk '{print \$5}'")
  n=$(awk -v t="$t" -v b="$BUDGET" -v nb="$N_BG" 'BEGIN { if (t == "" || t <= 0) { print 100; exit }
        n = int(100 * b / (nb * t)); print (n >= 100 ? 100 : (n < 10 ? 10 : n)) }')
  [ "$n" -ge 100 ] && echo "$run" || echo "$run:$n"
}

grep -v '^\s*#' "$QUEUE" | grep -v '^\s*$' | while read -r line; do
  args=()
  for tok in $line; do
    if [[ $tok == @bginf:* ]]; then args+=("$(resolve "$tok")"); else args+=("$tok"); fi
  done
  while [ "$(d 'squeue -u $USER -h -n dc_train | wc -l')" -ge 2 ]; do sleep 60; done
  jid=$(d "$R && sbatch --parsable -p debug -t 00:30:00 Revisions/DualColor/daint/train.sbatch ${args[*]}" | tail -1)
  echo "$(date +%H:%M) job $jid: ${args[*]}"
done
echo "$(date +%H:%M) queue submitted"
