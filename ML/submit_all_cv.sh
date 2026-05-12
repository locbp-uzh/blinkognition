#!/bin/bash
# Submit all cross-validation jobs to SLURM.
#
# Run from the ML/ directory on the HPC:
#   bash submit_all_cv.sh
#
# Optional: restrict to a subset by passing a glob pattern:
#   bash submit_all_cv.sh "configs_cv/config_cv_protein_mirrored_4prot_*.yaml"
#
# Each job gets a --job-name derived from the config filename so that
# squeue output is readable.

PATTERN="${1:-configs_cv/config_cv_*.yaml}"
SCRIPT="submit_cv.sh"

count=0
for cfg in ${PATTERN}; do
    [ -f "$cfg" ] || { echo "No files match: ${PATTERN}"; exit 1; }
    # Derive a short job name from the filename (strip prefix and suffix)
    jname=$(basename "$cfg" .yaml | sed 's/^config_cv_//')
    sbatch --job-name="${jname}" --account=locbp.chem.uzh --partition=standard \
           "${SCRIPT}" "${cfg}"
    echo "Submitted: ${cfg} (job-name: ${jname})"
    count=$((count + 1))
done

echo ""
echo "Total jobs submitted: ${count}"
