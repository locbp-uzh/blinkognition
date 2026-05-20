# Training Models (train.py)

Train classification models on labeled protein trace data with uncertainty quantification.

## Overview

`train.py` is the main training pipeline for supervised classification of protein traces. It handles dataset loading, model initialization, training with early stopping, evaluation, and uncertainty quantification via Monte Carlo dropout.

## Quick Start

```bash
cd ML
python train.py -c config_train.yaml
```

## Features

- Monte Carlo dropout uncertainty quantification
- Cross-platform acceleration (CUDA/MPS/CPU)
- Mixed precision training (FP16/BF16/FP32)
- Early stopping with patience
- Comprehensive logging and checkpointing
- Class-balanced training
- Gradient clipping
- Label smoothing

## Configuration File

### Basic Template

```yaml
io:
  output_root: "../Results/"
  run_name: "my_experiment"

data:
  traces_path: "../Data/traces"
  dataset:
    ProteinA:
      channels: ["minmax", "zscored"]
    ProteinB:
      channels: ["minmax", "zscored"]
  trim_end: 0
  max_traces_per_class: 200000
  balance_train: true
  balance_test: false  # Set to true to balance test set by subsampling majority classes

model:
  name: "resnet1d"

optimization:
  batch_size: 64
  max_epochs: 100
  patience_limit: 10
  lr: 0.001
  weight_decay: 0.0001
  label_smoothing: 0.05
  threshold_metric: "argmax"  # Use standard argmax (no threshold optimization)


uncertainty:
  mc_dropout:
    enabled: true
    n_mc: 100
    trace_loss: 50.0
    # wasserstein_threshold: 0.1  # Optional: fixed threshold

system:
  compile: false
  seed: 840410
  num_workers: 4
  verbose: false
```

## Configuration Parameters

### I/O Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_root` | str | `"../Results/"` | Directory for saving results |
| `run_name` | str | `"run"` | Descriptive name for this experiment |

### Data Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `traces_path` | str | `"../Data/traces"` | Path to directory containing trace pickle files |
| `dataset` | dict | required | Dictionary mapping protein names to channels |
| `trim_end` | int | `0` | Number of timesteps to trim from end (0 = no trimming) |
| `max_traces_per_class` | int | `null` | Maximum traces to load per class (null = no limit) |
| `balance_train` | bool | `true` | Use weighted sampling to balance classes during training |
| `balance_test` | bool | `false` | Balance test set by subsampling majority classes to match minority class size |

#### Dataset Dictionary Format

```yaml
dataset:
  PROTEIN_NAME:
    channels: ["channel1", "channel2"]
  ANOTHER_PROTEIN:
    channels: ["channel1", "channel2"]
```

The system discovers files matching:
- Pattern: `{PROTEIN_NAME}*{channel}*.pkl`
- Example: For `ProteinA` with channels `["minmax", "zscored"]`, it finds:
  - `ProteinA_minmax.pkl` (channel 0)
  - `ProteinA_zscored.pkl` (channel 1)

### Model Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Model architecture name (one of: `orig_conv_gru`, `resnet1d`, `tcn`) |
| `dropout` | float | from defaults | Global dropout rate (overrides hardcoded defaults) |
| `kwargs` | dict | `{}` | Additional model-specific parameters |

Available models:
- `orig_conv_gru` - Original baseline ConvGRU model
- `resnet1d` - 1D ResNet
- `tcn` - Temporal Convolutional Network

### Optimization Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | `64` | Training batch size |
| `max_epochs` | int | `100` | Maximum training epochs |
| `patience_limit` | int | `10` | Early stopping patience (epochs without improvement) |
| `lr` | float | `0.001` | Learning rate |
| `weight_decay` | float | `0.0` | AdamW weight decay |
| `label_smoothing` | float | `0.05` | Label smoothing factor (0.0 = no smoothing) |
| `clip_grad_norm` | float | `1.0` | Gradient clipping threshold |
| `threshold_metric` | str | `"argmax"` | Threshold selection method (see below) |

### Early Stopping Configuration

The training uses combined early stopping that requires both metric improvement AND loss stability:

```yaml
optimization:
  early_stopping:
    auc_min_delta: 0.005    # Minimum AUC improvement required (0.5%)
    loss_mode: "relative"   # "relative", "absolute", or "none"
    loss_tolerance: 1.10    # For relative: loss <= best_loss * 1.10 (max 10% higher)
    # Alternative checkpoint (uncomment to enable):
    # auc_tolerance: 0.005  # Max AUC drop from best to allow alternative checkpoint
    # loss_min_delta: 0.01  # Min loss improvement required for alternative checkpoint
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auc_min_delta` | float | `0.005` | Minimum AUC improvement to count as progress (0.005 = 0.5%) |
| `loss_mode` | str | `"relative"` | How to constrain loss: "relative", "absolute", or "none" |
| `loss_tolerance` | float | `1.10` | Loss constraint value (interpretation depends on mode) |
| `auc_tolerance` | float | `null` | (Optional) Max AUC drop from best to allow alternative checkpoint |
| `loss_min_delta` | float | `null` | (Optional) Min loss improvement required for alternative checkpoint |

This mechanism prevents saving checkpoints where AUC improved but loss spiked significantly, which often indicates overfitting.

**Primary checkpoint condition** (always active): A checkpoint is saved when:
1. AUC improves by at least `auc_min_delta`
2. AND current loss is within tolerance of the best loss seen

**Alternative checkpoint condition** (optional, enable by setting both `auc_tolerance` and `loss_min_delta`): A checkpoint is also saved when:
1. AUC is within `auc_tolerance` of the best AUC (even if not improving)
2. AND loss improves by at least `loss_min_delta`

The alternative condition allows saving checkpoints with slightly lower AUC but significantly better loss, which often yields higher accuracy at inference time.

Loss constraint modes:
- `"relative"`: Current loss must be <= best_loss * loss_tolerance (e.g., 1.10 = max 10% higher)
- `"absolute"`: Current loss must be <= best_loss + loss_tolerance
- `"none"`: No loss constraint (traditional AUC-only early stopping)

### Uncertainty Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mc_dropout.enabled` | bool | `true` | Perform Monte Carlo dropout inference |
| `mc_dropout.n_mc` | int | `100` | Number of forward passes |
| `mc_dropout.trace_loss` | float | `50.0` | Max trace loss % for auto threshold selection |
| `mc_dropout.wasserstein_threshold` | float | `null` | Optional: fixed Wasserstein threshold (overrides trace_loss) |

MC dropout runs `n_mc` stochastic forward passes (dropout active at inference), computes per-sample Wasserstein distances as an uncertainty proxy, and filters predictions above a learned threshold.

### System Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compile` | bool | `false` | Use `torch.compile` (experimental, requires PyTorch 2.0+) |
| `seed` | int | `null` | Random seed for reproducibility |
| `num_workers` | int | `4` | DataLoader worker processes |
| `verbose` | bool | `false` | Print detailed metrics during training |

#### Mixed Precision Behavior

- CUDA SM80+ (A100/H100): BF16 by default
- CUDA SM70+ (V100): FP16 by default
- MPS (Apple Silicon): FP16 by default
- CPU: FP32 always

## Usage Examples

### Basic Training

```bash
cd ML
python train.py -c config_train.yaml
```

### Training with Custom Parameters

```yaml
model:
  name: "tcn"
  kwargs:
    num_channels: [128, 128, 256, 256]
    kernel_size: 5
    dropout: 0.15
```

### Training on HPC (SLURM)

```bash
# Edit submit_train.sh to specify resources
sbatch submit_train.sh
```

## Output Structure

Each training run creates a timestamped directory:

```
Results/YYYY-MM-DD_HH-MM-SS_<run_name>_training_<dataset>/
├── best_model.pth                      # Best model checkpoint (state_dict)
├── checkpoint.pth                      # Latest checkpoint (includes optimal_threshold if used)
├── loss_curve.png                      # Train/val loss plot
├── confusion_matrix.png                # Test set confusion matrix (argmax or optimal threshold)
├── confusion_matrix_argmax.png         # Test set confusion matrix (only if threshold optimization used)
├── test_metrics.json                   # Comprehensive test metrics
├── config_full.json                    # Complete configuration snapshot
├── config_summary.json                 # Run summary (device, amp_dtype, num_classes, etc.)
├── <run_name>_training_<dataset>.log   # Training log
└── MCD_results/                        # Monte Carlo dropout results
    ├── filtered_confusion_matrix_threshX.XX.png
    ├── balanced_accuracy_vs_uncertainty.png
    └── mc_dropout_metrics.json         # MC dropout metrics with threshold info
```

See the output structure table above for descriptions of all output files.

## Training Pipeline

### 1. Initialization

```
Load config → Set seed → Detect accelerator → Build dataset
```

### 2. Model Building

```
Initialize model → Move to device → Optionally compile
```

### 3. Training Loop

```
For each epoch:
  Train on training set with class balancing
  Evaluate on validation set
  Check for improvement (patience tracking)
  Save checkpoint if best validation loss
  If patience exhausted: stop early
```

### 4. Threshold Selection

The system provides flexible threshold selection via the `threshold_metric` parameter:

**Default Behavior (argmax):**
- `threshold_metric: "argmax"` (recommended for balanced datasets)
- Uses standard argmax prediction (no threshold optimization)
- Fastest inference, simplest interpretation
- Produces single confusion matrix

**Threshold Optimization (Binary Classification):**
When `threshold_metric` is set to an optimization metric, the system sweeps 101 candidate thresholds (0.00 to 1.00) on the validation set:

- `"min_recall"` - Maximizes the minimum recall across classes (ensures neither class has poor performance)
- `"balanced_accuracy"` - Maximizes average per-class recall (may allow imbalance between classes)
- `"mcc"` - Matthews Correlation Coefficient (robust to class imbalance)
- `"f1"` - F1-score (harmonic mean of precision and recall)
- `"youden"` - Youden's J statistic (sensitivity + specificity - 1)

**Threshold Optimization (Multiclass Classification):**
- Uses global confidence threshold approach
- Sweeps 51 candidate thresholds
- Optimizes macro F1-score (equal weight to all classes)
- Requires maximum predicted probability to exceed threshold

The optimal threshold (if computed) is saved in the checkpoint and used for:
- Test set evaluation
- Monte Carlo dropout predictions
- Wasserstein uncertainty filtering

**When to Use Threshold Optimization:**

Use `"argmax"` (default) when:
- Your dataset is balanced or well-regularized
- You want the simplest, fastest predictions
- You trust the model's probability calibration

Use threshold optimization when:
- You have class imbalance despite balancing techniques
- You need to explicitly control the precision/recall tradeoff
- Your application requires specific minimum recalls per class

**Output Differences:**
- `threshold_metric: "argmax"` → Single confusion matrix
- Other threshold metrics → Two confusion matrices (argmax baseline + optimized)

### 5. Evaluation

```
Load best model → Apply optimal threshold → Evaluate on test set
```

Reports comprehensive metrics:
- **Binary**: Accuracy (argmax vs optimal), balanced accuracy, AUC-ROC, AUC-PR, MCC
- **Multiclass**: Accuracy (argmax vs optimal), balanced accuracy, AUC-ROC, macro F1
- Generates confusion matrices based on threshold_metric setting

### 7. Monte Carlo Dropout

If `mc_dropout.enabled: true`:
```
For each test sample:
  Run n_mc forward passes with dropout ON
  Apply optimal threshold to mean predictions
  Aggregate predictions → Compute uncertainty
Filter unreliable predictions → Re-evaluate
```

## Common Workflows

### Workflow 1: Quick Experiment

```bash
# 1. Edit config
vim config_train.yaml

# 2. Train
python train.py -c config_train.yaml

# 3. Check results
ls -lh ../Results/2025-*
```

### Workflow 2: Systematic Comparison

```bash
# Compare multiple models on same dataset
for model in tcn resnet1d orig_conv_gru; do
  sed -i "s/name: .*/name: \"$model\"/" config_train.yaml
  python train.py -c config_train.yaml
done
```

## Monitoring Training

### Real-time Monitoring

```bash
# Watch training log
tail -f ../Results/2025-*/my_experiment_training_*.log
```

### SLURM Job Monitoring

```bash
# Check job status
squeue -u $USER

# View output
cat slurm-<jobid>.out
```

## Troubleshooting

### GPU Out of Memory

Reduce batch size:
```yaml
optimization:
  batch_size: 32  # or 16
```

### NaN Loss

1. Lower learning rate:
```yaml
optimization:
  lr: 0.0001
```

2. Increase gradient clipping:
```yaml
optimization:
  clip_grad_norm: 5.0
```

### Slow Training

1. Enable compilation (PyTorch 2.0+, experimental):
```yaml
system:
  compile: true
```

2. Use fewer workers on systems with limited CPU:
```yaml
system:
  num_workers: 2
```

3. Disable verbose mode:
```yaml
system:
  verbose: false
```

### Model Not Loading

Check that model name matches exactly (names are case-sensitive):

Available names: `orig_conv_gru`, `resnet1d`, `tcn`

## Advanced Options

### Custom Model Parameters

Override hardcoded defaults:
```yaml
model:
  name: "tcn"
  kwargs:
    num_channels: [128, 128, 256, 256, 512]
    kernel_size: 5
    dropout: 0.2
    norm: "bn"  # instead of default "gn"
```

### Per-Model Configuration

```yaml
model:
  name: "resnet1d"
  per_model:
    resnet1d:
      base_filters: 128
      num_blocks: 4
```

### Training Without Validation Split

Not directly supported, but you can:
1. Set `patience_limit: 1000` (effectively disable early stopping)
2. Use all data in training (system auto-splits 70/15/15)

### Resume Training

Currently not supported. Workaround:
1. Use same config with different `run_name`
2. Manually load checkpoint in custom training script

## Best Practices

1. Use class balancing (`balance_train: true`) for imbalanced datasets
2. Enable Monte Carlo dropout for uncertainty quantification
3. Set random seed for reproducibility
4. Use descriptive `run_name` for easy identification
5. Monitor training logs for convergence issues
6. Check confusion matrix for systematic errors

## See Also

- [Cross-Validation](crossval.md) - Multi-fold evaluation
