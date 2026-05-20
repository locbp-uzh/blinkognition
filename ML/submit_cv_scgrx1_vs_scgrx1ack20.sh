#!/bin/bash
# Cross-validation: scGrx1 vs scGrx1AcK20
# 9 jobs: protein (mirrored + not-mirrored) and background × minmax/zscored/both

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs_cv"

CONFIG_FILES=(
    "config_cv_protein_mirrored_scgrx1_scgrx1ack20_minmax.yaml"
    "config_cv_protein_mirrored_scgrx1_scgrx1ack20_zscored.yaml"
    "config_cv_protein_mirrored_scgrx1_scgrx1ack20_both.yaml"
    "config_cv_protein_notmirrored_scgrx1_scgrx1ack20_minmax.yaml"
    "config_cv_protein_notmirrored_scgrx1_scgrx1ack20_zscored.yaml"
    "config_cv_protein_notmirrored_scgrx1_scgrx1ack20_both.yaml"
    "config_cv_background_notmirrored_scgrx1_scgrx1ack20_minmax.yaml"
    "config_cv_background_notmirrored_scgrx1_scgrx1ack20_zscored.yaml"
    "config_cv_background_notmirrored_scgrx1_scgrx1ack20_both.yaml"
)

echo "Submitting ${#CONFIG_FILES[@]} CV jobs (scGrx1 vs scGrx1AcK20)..."

for config in "${CONFIG_FILES[@]}"; do
    config_path="${CONFIG_DIR}/${config}"
    job_name="${config%.yaml}"
    job_name="${job_name#config_}"

    if [[ ! -f "$config_path" ]]; then
        echo "WARNING: not found: $config_path"
        continue
    fi

    echo "Submitting: $job_name"
    sbatch --job-name="$job_name" <<EOF
#!/bin/bash
#SBATCH --account=locbp.chem.uzh
#SBATCH --partition=standard
#SBATCH --gres=gpu:3
#SBATCH --constraint=GPUMEM80GB
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=23:59:00
#SBATCH --output=${SCRIPT_DIR}/logs/${job_name}_%j.out
#SBATCH --error=${SCRIPT_DIR}/logs/${job_name}_%j.err

mkdir -p ${SCRIPT_DIR}/logs
module load miniforge3
conda activate blink2-cuda
cd ${SCRIPT_DIR}
python crossval.py -c "${config_path}"
EOF
done

echo "Done. Monitor with: squeue -u \$USER"
echo "Results: ../Results/CrossVal/"
