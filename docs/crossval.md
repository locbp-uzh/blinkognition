# Cross-Validation (crossval.py)

Compare multiple model architectures using K-fold cross-validation with optional augmentation sweep and multi-GPU parallelization.

## Overview

`crossval.py` performs systematic model comparison using stratified K-fold cross-validation. It supports comparing multiple models, sweeping augmentation factors, and running parallel evaluations across multiple GPUs.

## Quick Start

```bash
cd ML

# Basic cross-validation (compare models)
python crossval.py -c config_cv.yaml --models tcn,resnet1d,orig_conv_gru

# Augmentation sweep (compare models x augmentation factors)
python crossval.py -c config_cv.yaml --aug-factors 0,2,3,5 --n-gpus 4
```

## Key Features

- Stratified K-fold cross-validation
- Multi-model comparison in a single run
- Augmentation factor sweep (compare different augmentation levels)
- Multi-GPU parallelization for augmentation sweeps
- Monte Carlo dropout uncertainty quantification
- Wasserstein distance threshold optimization
- Timing metrics (time per epoch) for fair comparison
- Automated visualization (comparison plots, heatmaps, bar charts)



## Configuration File

### Basic Template (config_cv.yaml)

```yaml
io:
  output_root: "../Results/CrossVal"
  run_name: "crossval"

data:
  traces_path: "../Data/traces"
  dataset:
    ProteinA:
      channels: ["minmax", "zscored"]
    ProteinB:
      channels: ["minmax", "zscored"]

trim_end: 2000
max_traces_per_class: 5000
balance_train: true
balance_test: true

compare:
  models: ["resnet1d", "tcn", "orig_conv_gru"]

cv:
  n_splits: 5
  shuffle: true
  seed: 840410

optimization:
  batch_size: 64
  lr: 0.0004
  weight_decay: 0.0004
  max_epochs: 500
  patience_limit: 10

  # Early stopping configuration
  early_stopping:
    auc_min_delta: 0.005    # Minimum AUC improvement required (0.5%)
    loss_mode: "relative"   # "relative", "absolute", or "none"
    loss_tolerance: 1.10    # For relative: loss <= best_loss * 1.10 (max 10% higher)
    # Alternative checkpoint (uncomment to enable):
    # auc_tolerance: 0.005  # Max AUC drop from best to allow alternative checkpoint
    # loss_min_delta: 0.01  # Min loss improvement required for alternative checkpoint

uncertainty:
  mc_dropout:
    enabled: false
    n_mc: 100
    trace_loss: 50.0              # Maximum trace loss % for auto threshold selection
    # wasserstein_threshold: 0.1  # Optional: fixed Wasserstein threshold (overrides trace_loss)

system:
  n_gpus: 1

# Augmentation sweep configuration
augmentation_sweep:
  enabled: false
  factors: [0, 2, 3, 5]  # 0 = no augmentation
  augmentation_params:
    time_warp_sigma: 0.03
    noise_sigma: 0.02
    magnitude_jitter: 0.02
```

## Usage Modes

### Standard Mode: Model Comparison

Compare multiple models without augmentation:

```bash
python crossval.py -c config_cv.yaml --models tcn,resnet1d,orig_conv_gru --k 5
```

Output:
- Per-fold metrics (AUC, balanced accuracy, time per epoch)
- Aggregated confusion matrices (mean and std)
- Model comparison scatter plot (AUC vs time per epoch)
- Loss curves for one random fold per model

### Augmentation Sweep Mode

Compare models across different augmentation levels using multi-GPU parallelization:

```bash
# Via CLI
python crossval.py -c config_cv.yaml --aug-factors 0,2,3,5 --n-gpus 4

# Via config file (set augmentation_sweep.enabled: true)
python crossval.py -c config_cv.yaml
```

Output:
- Heatmap visualization (model x augmentation factor)
- Grouped bar chart (AUC by augmentation factor)
- Aggregated metrics CSV
- Per-fold metrics CSV

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `-c, --config` | Path to configuration YAML file (required) |
| `--models` | Comma-separated model names to compare |
| `--k` | Number of CV folds (overrides config) |
| `--aug-factors` | Comma-separated augmentation factors (enables sweep mode) |
| `--n-gpus` | Number of GPUs for parallel execution |

## Multi-GPU Parallelization

The augmentation sweep uses dynamic work stealing to distribute (model, augmentation_factor) combinations across multiple GPUs:

```bash
# Request 4 GPUs via SLURM
#SBATCH --gres=gpu:4

# Auto-detect and use all available GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
python crossval.py -c config_cv.yaml --aug-factors 0,2,3,5 --n-gpus ${N_GPUS}
```

With dynamic work stealing, each GPU worker continuously pulls tasks from a shared queue. When a GPU finishes a task, it immediately picks up the next available task instead of waiting for other GPUs. This ensures GPUs don't sit idle while work remains.

Example: With 3 models, 4 augmentation factors (12 tasks), and 4 GPUs, if GPU 1 finishes a fast task early, it immediately starts the next task rather than waiting for slower GPUs to complete their current work.

## Output Structure

### Standard Mode

```
Results/CrossVal/<timestamp>_crossval_5fold_<dataset>/
├── cv_folds_metrics.csv           # Per-fold metrics for all models
├── cv_confmat_pre_mc.csv          # Confusion matrices before MC dropout
├── cv_confmat_post_mc_best.csv    # Confusion matrices after MC dropout
├── cv_wd_sweep.csv                # Wasserstein distance sweep results
├── pre_mc_confmat_<model>_mean.png
├── pre_mc_confmat_<model>_std.png
├── loss_curve_<model>_fold<N>.png # Loss curve for random fold
├── model_comparison/
│   ├── plot.png                   # AUC vs time per epoch scatter
│   └── data.csv                   # Comparison data
└── config_snapshot.json
```

### Augmentation Sweep Mode

```
Results/CrossVal/<timestamp>_crossval_5fold_<dataset>/
├── cv_folds_metrics.csv               # All fold-level metrics
├── augmentation_sweep/
│   ├── data.csv                       # Aggregated (model x aug_factor) stats
│   ├── heatmap.png / .pdf             # AUC heatmap (model x aug_factor)
│   ├── cv_confmat.csv                 # Pre-MC confusion matrix data (includes aug_label)
│   ├── cv_confmat_filtered.csv        # Post-MC filtered confusion matrix data
│   ├── confmat_<model>_<aug>.png/.pdf
│   └── confmat_<model>_<aug>_filtered.png/.pdf
├── MCD_results/
│   ├── cv_wd_sweep.csv                # Per-fold WD threshold sweep (model x aug x fold)
│   └── wd_sweep.png / .pdf            # Retention vs accuracy plot per model
├── loss_curve_<model>_<aug>_fold<N>.png
└── config_snapshot.json
```

## Augmentation Parameters

The augmentation sweep applies data augmentation only to training data:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `time_warp_sigma` | 0.5 | Time warping strength |
| `noise_sigma` | 0.5 | Gaussian noise amplitude |
| `magnitude_jitter` | 0.5 | Magnitude scaling jitter |

Augmentation factor controls how many augmented copies are created:
- `0` = No augmentation (baseline)
- `2` = Original + 2x augmented copies
- `5` = Original + 5x augmented copies

## Metrics Reported

### Per-Fold Metrics

- `auc_macro_ovr`: Macro-averaged AUC (one-vs-rest)
- `bal_acc`: Balanced accuracy
- `loss`: Validation loss
- `epochs_trained`: Number of epochs until early stopping
- `time_per_epoch_sec`: Training time per epoch (for fair comparison)

### Aggregated Metrics (Augmentation Sweep)

- `auc_mean`, `auc_std`: Mean and std AUC across folds
- `bal_acc_mean`, `bal_acc_std`: Mean and std balanced accuracy
- `time_per_epoch_mean`, `time_per_epoch_std`: Training speed

## HPC Submission

The default `submit_cv.sh` is configured for multi-GPU augmentation sweeps:

```bash
#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --constraint=GPUMEM80GB

N_GPUS=$(nvidia-smi -L | wc -l)
python crossval.py -c config_cv.yaml --n-gpus ${N_GPUS}
```

## Example Workflows

### Workflow 1: Quick Model Comparison

```bash
# Compare 3 models with 5-fold CV
python crossval.py -c config_cv.yaml --models tcn,resnet1d,orig_conv_gru --k 5
```

### Workflow 2: Find Optimal Augmentation

```bash
# Sweep augmentation factors across models
python crossval.py -c config_cv.yaml \
  --models tcn,resnet1d \
  --aug-factors 0,2,3,5 \
  --n-gpus 4
```

### Workflow 3: Production Comparison on HPC

```bash
# Edit config_cv.yaml with your dataset and models
# Submit to SLURM
sbatch submit_cv.sh
```

## Interpreting Results

### Model Comparison Plot

The scatter plot shows AUC (y-axis) vs time per epoch (x-axis) with error bars. Ideal models appear in the upper-left (high AUC, fast training).

### Augmentation Heatmap

Rows are models, columns are augmentation factors. Color intensity shows AUC. Look for:
- Best augmentation factor per model
- Models that benefit most from augmentation
- Diminishing returns at high augmentation

### Grouped Bar Chart

Shows AUC with error bars grouped by augmentation factor. Useful for comparing models at each augmentation level.

## Troubleshooting

### GPU Out of Memory

Reduce batch size or max_traces_per_class:
```yaml
optimization:
  batch_size: 32
max_traces_per_class: 2000
```

### Multi-GPU Issues

Ensure CUDA_VISIBLE_DEVICES is not set, or set to all GPUs:
```bash
unset CUDA_VISIBLE_DEVICES
python crossval.py -c config_cv.yaml --n-gpus 4
```

### Slow Augmentation Sweep

- Use fewer augmentation factors: `--aug-factors 0,3`
- Reduce max_epochs in config
- Use more GPUs

## See Also

- [Training](train.md) - Single model training
- Available models: `orig_conv_gru`, `resnet1d`, `tcn` (see `ML/models.py`)
