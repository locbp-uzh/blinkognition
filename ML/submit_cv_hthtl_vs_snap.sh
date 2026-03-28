#!/bin/bash
# Master script to submit all 9 cross-validation jobs
# Compares orig_conv_gru, resnet1d, and tcn models on HTHTL and snap proteins
# across 3 datasets (mirrored, not-mirrored, background) and 3 channel configs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs_cv"

# List of config files to submit
CONFIG_FILES=(
    "config_cv_mirrored_hthtl_snap_both.yaml"
    "config_cv_mirrored_hthtl_snap_minmax.yaml"
    "config_cv_mirrored_hthtl_snap_zscored.yaml"
    "config_cv_notmirrored_hthtl_snap_both.yaml"
    "config_cv_notmirrored_hthtl_snap_minmax.yaml"
    "config_cv_notmirrored_hthtl_snap_zscored.yaml"
    "config_cv_background_hthtl_snap_both.yaml"
    "config_cv_background_hthtl_snap_minmax.yaml"
    "config_cv_background_hthtl_snap_zscored.yaml"
)

echo "Submitting ${#CONFIG_FILES[@]} cross-validation jobs..."
echo ""

for config in "${CONFIG_FILES[@]}"; do
    config_path="${CONFIG_DIR}/${config}"
    job_name="${config%.yaml}"
    job_name="${job_name#config_}"  # Remove "config_" prefix for cleaner job names

    if [[ ! -f "$config_path" ]]; then
        echo "WARNING: Config file not found: $config_path"
        continue
    fi

    echo "Submitting: $job_name"

    sbatch --job-name="$job_name" <<EOF
#!/bin/bash
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --constraint=GPUMEM80GB
#SBATCH --time=23:59:59
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err

# Create logs directory if it doesn't exist
mkdir -p ${SCRIPT_DIR}/logs

# Activate conda environment
source ~/.bashrc
conda activate blink2cuda

# Change to ML directory
cd ${SCRIPT_DIR}

# Run cross-validation
python crossval.py -c "${config_path}"
EOF

    echo "  -> Submitted $job_name"
    echo ""
done

echo "All jobs submitted. Check status with: squeue -u \$USER"
echo "Results will appear in: ../Results/CrossVal/"
