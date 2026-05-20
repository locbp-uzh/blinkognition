# Trace Extraction Pipeline

Complete pipeline for extracting protein trace signals from ND2 microscopy movies.

## Overview

The extraction pipeline processes single-molecule localization microscopy (SMLM) data through six steps:

1. **Parameter Optimization** - Bayesian optimization of localization parameters
2. **Localize** - CPU-based Picasso localization with parallel processing
3. **Extract** - Extract intensity traces from localizations
4. **Combine** - Combine per-movie traces by protein
5. **Filter** - Quality filtering and ML preparation
6. **Diagnose** - Trace quality analysis and diagnostic reports

All outputs are organized in versioned run folders with full parameter traceability.

## Features

- **Flexible Analysis Modes**: Supports both dual-channel (with ground truth) and single-channel (no ground truth) analysis
- **CPU-based Localization**: Uses Picasso's MLE (maximum likelihood estimation) method
- **Automatic FOV Detection**: Reads image dimensions from ND2 metadata
- **Automatic Parameter Optimization**: Bayesian optimization finds optimal gradients and boxsize per protein/experiment
- **Parallel Processing**: Multi-CPU support for localization and trace extraction steps
- **Structured Data Organization**: Experiment → Protein → Files hierarchy
- **Multi-channel Support**: Any wavelengths with channel-specific gradients
- **Ground Truth Labeling**: Optional automatic IN/OUT classification based on colocalization
- **Background Trace Extraction**: Optional extraction of background traces from non-localization pixels for comparison
- **Versioned Output**: Run folders named by proteins and gradient settings
- **Orchestrated Execution**: Single command runs all steps with auto-folder detection
- **Flexible Resuming**: Restart from any step for parameter tuning

## Quick Start

### Environment Setup

```bash
# Create picasso environment
conda env create -f picasso-env.yaml
conda activate picasso-env
```

### Run Complete Pipeline

```bash
# Full pipeline (optimization through filtering)
python Extraction/run_pipeline.py -c Extraction/config.yaml

# Output with optimized parameters:
# Results/Extract/<proteins>_optparam_001/

# Output with config defaults (no optimization):
# Results/Extract/<proteins>_gradP-12000_gradGT-4000_001/  (with ground truth)
# Results/Extract/<proteins>_gradP-12000_001/              (no ground truth)
```

### Resume from Specific Step

```bash
# Re-run filtering with different thresholds
python Extraction/run_pipeline.py -c Extraction/config.yaml \
  --start-from filter \
  -r Results/Extract/<your_run_folder>
```

## Analysis Modes

The pipeline supports two analysis modes controlled by the `has_ground_truth` flag in config.yaml:

### Mode 1: With Ground Truth (Dual-Channel)
```yaml
has_ground_truth: true
protein_channel: "640"
ground_truth_channel: "488"
```

- Analyzes both protein and ground truth channels
- Produces IN/OUT labeled traces based on colocalization
- Uses composite objective for parameter optimization (rewards IN traces)
- Output files: `{Protein}_IN_*.pkl` and `{Protein}_OUT_*.pkl`

### Mode 2: No Ground Truth (Single-Channel)
```yaml
has_ground_truth: false
protein_channel: "640"
# ground_truth_channel not used
```

- Analyzes only protein channel
- Produces unlabeled traces
- Uses simplified objective for parameter optimization
- Output files: `{Protein}_*.pkl`

**Both modes can coexist** - the system uses mode-specific file naming (_gt vs _no_gt suffixes) to prevent conflicts.

## Pipeline Steps

### Step 0: Parameter Optimization (paramfinder.py)

Optimizes localization parameters using Bayesian optimization:
- Runs on first 1000 frames of a random movie per protein/experiment
- Uses Optuna (TPE sampler) with 20 trials by default
- Optimizes gradient thresholds only (boxsize fixed in config.yaml for consistent intensity integration)
- Saves results to `{Protein}/optimized_localization_params_{gt|no_gt}.yaml`
- Creates visual QC images in `{Protein}/visualQC/{gt|no_gt}/`

**Objective Functions**:
- With ground truth: `num_single * (1 + num_IN/num_single) * (1 - overlap_ratio)^2`
- No ground truth: `num_single * (1 - overlap_ratio)^2`

**Outputs**:
- `optimization_{gt|no_gt}.db` - Optuna database for resuming
- `optimized_localization_params_{gt|no_gt}.yaml` - Best parameters
- `visualQC/{gt|no_gt}/trial_*.tif` - QC images for each trial

### Step 1: Localization (localize.py)

Runs Picasso localization on ND2 movies:
- Uses CPU-based MLE (maximum likelihood estimation) method
- Auto-detects FOV size from ND2 metadata
- Automatically loads optimized parameters from protein folders
- Finds picasso-env Python automatically
- Processes files based on analysis mode (both channels or protein only)
- Parallel processing of movies across multiple CPUs
- Creates versioned run folder with descriptive name
- Saves localization parameters for traceability

**Inputs**:
- ND2 movies in `Data/movies/{Experiment}/{Protein}/`
- Optimized parameters from `optimized_localization_params_{gt|no_gt}.yaml` (if exists)
- Configuration: gradient thresholds, camera parameters (fallback if no optimized params)

**Outputs**:
- `Results/Extract/{proteins}_gradP-{X}_gradGT-{Y}_{N}/` (with ground truth)
- `Results/Extract/{proteins}_gradP-{X}_{N}/` (no ground truth)
  - `localization_params.yaml` - All parameters and input path
  - `Exp*/Protein/*_locs.hdf5` - Localization coordinates
  - `Exp*/Protein/*_locs.yaml` - Metadata

### Step 2: Extract Traces (extract.py)

Extracts intensity time series from localizations:
- **With ground truth**: Pairs protein and ground truth channel movies
- **No ground truth**: Processes protein channel only
- Links protein localizations across frames
- Clusters ground truth localizations (if applicable)
- Extracts intensity traces for each ROI
- Labels traces as IN/OUT based on ground truth proximity (if applicable)
- **Background traces**: Optionally extracts traces from non-localization pixels for comparison
- Parallel processing of movie pairs across multiple CPUs

**Background Trace Extraction**:
When `n_background_traces > 0` in config, the pipeline extracts background traces from pixels that are not part of any protein ROI. This provides baseline noise characteristics for comparison with protein traces:
- Samples random positions from unoccupied pixels (respecting edge distance)
- Uses same ROI size (boxsize) as protein localizations
- Useful for quality assessment and noise characterization

**Inputs**:
- Run folder from localize.py
- Configuration: linking distance, dark time, ground truth threshold, n_background_traces

**Outputs**:
- `{RunFolder}/Exp*/Protein/*_traces.pkl` - DataFrame with:
  - `x, y`: ROI position
  - `trace`: Intensity time series (e.g., 6000 frames)
  - `overlap`: "single" or "overlapping"
  - `ground_truth`: "IN" or "OUT" (only if has_ground_truth=true)
- `{RunFolder}/Exp*/Protein/*_background_traces.pkl` - Background traces (if enabled):
  - `x, y`: ROI position
  - `trace`: Intensity time series
  - `overlap`: Always "single" (non-overlapping by design)
- `{RunFolder}/background_file_list.pkl` - List of background trace files (if enabled)

### Step 3: Combine Traces (combine.py)

Combines per-movie traces by protein and applies preprocessing:
- Groups traces by protein (from filename)
- Assigns unique IDs for provenance
- **With ground truth**: Separates by ground_truth (IN vs OUT)
- **No ground truth**: Keeps all single (non-overlapping) traces together
- **Background traces**: Processes background traces separately (if enabled)
- Converts to matrices (rows=frames, cols=traces)
- Applies multiple normalizations

**Background removal method**: A 2-component GMM is fit to each trace; the noise component mean is subtracted as background. This is robust for traces that lack a clear flat tail.

**Background contamination filtering** (applied before saving background traces):
1. **Spatial buffer**: reject background ROIs within `background_buffer` pixels of any protein ROI
2. **Robust outlier score**: reject traces where `(max − median) / MAD > background_rob_threshold`. This catches brief contamination events (spikes) that inflate the trace without being detectable by simple variance measures.

**Normalizations**:
1. **Raw**: No preprocessing
2. **Background Removed**: Subtract per-trace GMM noise component mean
3. **Z-scored**: Standardize (mean=0, std=1) from background-removed
4. **Min-max**: Scale to [0,1] from background-removed

**Outputs (with ground truth)**:
- `{RunFolder}/ProteinTracesIN/AllRaw/{Protein}_IN_raw.pkl`
- `{RunFolder}/ProteinTracesIN/AllBG/{Protein}_IN_bg_rm.pkl`
- `{RunFolder}/ProteinTracesIN/AllNorm/{Protein}_IN_zscored.pkl`
- `{RunFolder}/ProteinTracesIN/AllNorm/{Protein}_IN_minmax.pkl`
- `{RunFolder}/UniqueIDs/{Protein}_uniqueID_{all|IN|OUT}.pkl` - Metadata
- (same for `ProteinTracesOUT/`)

**Outputs (no ground truth)**:
- `{RunFolder}/ProteinTraces/AllRaw/{Protein}_raw.pkl`
- `{RunFolder}/ProteinTraces/AllBG/{Protein}_bg_rm.pkl`
- `{RunFolder}/ProteinTraces/AllNorm/{Protein}_zscored.pkl`
- `{RunFolder}/ProteinTraces/AllNorm/{Protein}_minmax.pkl`
- `{RunFolder}/UniqueIDs/{Protein}_uniqueID_{all|single}.pkl` - Metadata

**Outputs (background traces, if enabled)**:
- `{RunFolder}/BackgroundTraces/AllRaw/{Protein}_background_raw.pkl`
- `{RunFolder}/BackgroundTraces/AllBG/{Protein}_background_bg_rm.pkl`
- `{RunFolder}/BackgroundTraces/AllNorm/{Protein}_background_zscored.pkl`
- `{RunFolder}/BackgroundTraces/AllNorm/{Protein}_background_minmax.pkl`
- `{RunFolder}/UniqueIDs/{Protein}_uniqueID_background.pkl` - Metadata

**Note**: `AllRaw/`, `AllBG/`, `AllNorm/` contain all traces before quality filtering. Background traces are not peak-filtered; their `Filtered/` subfolder is unused.

### Step 4: Filter Traces (filter.py)

Quality filtering for machine learning. Reads from the `AllRaw/`, `AllBG/`, `AllNorm/` subfolders produced by combine.py and writes filtered outputs to `Filtered/`.

**Peak detection**: Runs GMM classification on the background-removed trace; consecutive frames assigned to the signal component form peaks. SNR gate (`snr_min_separation`) rejects pure-noise traces before peak detection.

**Filtering criteria**:
- Minimum number of peaks (`min_peak_number`)
- First peak must occur before `first_peak_time`
- Second peak must follow the first within `delta_first_second` frames
- Last peak must occur after `last_peak_time` (from start)
- Optional: Denoise by flattening frames below signal threshold (`flatten_background`)
- Optional: Data augmentation with time-reversal mirroring (`include_mirrored`)

**Outputs (with ground truth)**:
- `{RunFolder}/ProteinTracesIN/Filtered/{Protein}_IN_filtered_raw.pkl`
- `{RunFolder}/ProteinTracesIN/Filtered/{Protein}_IN_filtered_bg_rm.pkl`
- `{RunFolder}/ProteinTracesIN/Filtered/{Protein}_IN_filtered_zscored.pkl` ← ML-ready
- `{RunFolder}/ProteinTracesIN/Filtered/{Protein}_IN_filtered_minmax.pkl`  ← ML-ready
- (same for `ProteinTracesOUT/Filtered/`)

**Outputs (no ground truth)**:
- `{RunFolder}/ProteinTraces/Filtered/{Protein}_filtered_raw.pkl`
- `{RunFolder}/ProteinTraces/Filtered/{Protein}_filtered_bg_rm.pkl`
- `{RunFolder}/ProteinTraces/Filtered/{Protein}_filtered_zscored.pkl`
- `{RunFolder}/ProteinTraces/Filtered/{Protein}_filtered_minmax.pkl`

### Step 5: Diagnostics (diagnose.py)

Analyzes trace quality and generates diagnostic reports to help understand filtering decisions:
- Computes quality metrics for all traces (before filtering)
- Shows which filtering criteria each trace passes/fails
- Generates detailed reports with statistics
- Creates visualizations of passed vs. failed traces
- Plots distributions of quality metrics
- **Background comparison**: Compares protein traces with background traces (if available)

**Quality Metrics Computed**:
- **SNR**: GMM component separation — `(signal_mean − noise_mean) / noise_std`
- **Peak count**: Number of peaks detected via GMM classification on background-removed traces
- **Peak timing**: First and last peak positions
- **Peak spacing**: Distance between first two peaks
- **Signal statistics**: Mean, std, max intensity
- **Filter status**: Pass/fail with specific failure reasons

**Background Trace Analysis** (if `n_background_traces > 0`):
When background traces are available, the diagnostic step also:
- Computes basic metrics for background traces (SNR, std, peak count)
- Plots example background traces for visual inspection
- Compares protein vs background distributions (SNR, variability, peaks)
- Calculates SNR separation between protein and background

**Outputs**:
- `{RunFolder}/Diagnostics/IN/{Protein}_IN_trace_examples/` - Visual examples (plot.pdf + data CSV)
- `{RunFolder}/Diagnostics/OUT/{Protein}_OUT_trace_examples/`
- `{RunFolder}/Diagnostics/Background/{Protein}_background_trace_examples/`
- (No ground truth: `Diagnostics/Protein/` instead of `IN/`/`OUT/`)

**Usage**:
```bash
# Run automatically after filtering (if enabled in config)
python Extraction/run_pipeline.py -c Extraction/config.yaml

# Run manually on existing filtered data
python Extraction/diagnose.py -c Extraction/config.yaml -r Results/Extract/YOUR_RUN_FOLDER

# Analyze specific protein only
python Extraction/diagnose.py -c Extraction/config.yaml -r YOUR_RUN_FOLDER --protein <ProteinName>

# Control number of example traces
python Extraction/diagnose.py -c Extraction/config.yaml -r YOUR_RUN_FOLDER --n-examples 24
```

**Configuration**:
```yaml
diagnostics:
  enabled: True                        # Run after filtering in pipeline
  n_examples: 12                       # Number of example traces to plot
  proteins: null                       # Specific proteins to diagnose (null = all)
  # OR: proteins: ["ProteinA"]            # Analyze only ProteinA
```

## Configuration

### Required Parameters

```yaml
# Analysis Mode
has_ground_truth: true    # true for dual-channel, false for single-channel

# I/O
input_folder: Data/movies
output_folder_base: Results/Extract

# Proteins to process
proteins:
  - ProteinA
  - ProteinB

# Channels
protein_channel: "640"           # Protein signal channel
ground_truth_channel: "488"      # Ground truth channel (ignored if has_ground_truth: false)

# Localization (default values - can be optimized automatically)
boxsize: 7
gradient_protein: 12000          # Protein channel sensitivity
gradient_ground_truth: 4000      # Ground truth channel sensitivity (only if has_ground_truth: true)
drift: 0
baseline: 79
sensitivity: 16.0
gain: 300
quantum_efficiency: 0.93

# Parameter Optimization (optional)
optimization:
  mode: "per_experiment"         # "per_experiment" (default) or "per_protein"
  aggregation: "mean"            # How to combine per-protein scores in per_experiment mode
  n_trials: 20                   # Number of Optuna trials per experiment
  search_space:
    gradient_protein_min: 10000
    gradient_protein_max: 80000
    gradient_protein_step: 10000
    gradient_ground_truth_min: 1000
    gradient_ground_truth_max: 10000
    gradient_ground_truth_step: 1000

# Trace extraction
max_distance: 2.0                      # Linking distance (pixels)
max_darktime: 6000                     # Max dark time (frames)
min_on_ground_truth: 3                 # Min locs to call ground truth spot (if applicable)
max_dist_closest_ground_truth: 2.0     # IN/OUT threshold (if applicable)

# Background trace extraction (optional)
n_background_traces: 100               # Number of background traces per movie (0 to disable)
background_buffer: 5                   # Reject background ROIs within N pixels of any protein ROI
background_rob_threshold: 8.0         # Reject background traces where (max-median)/MAD > threshold

# Movie settings
movie_length: 6000        # Expected frames
# size_FOV: 230           # Auto-detected from ND2 metadata (can override if needed)

gmm_proba_threshold: 0.8  # Posterior probability threshold for GMM signal classification
snr_min_separation: 2.0   # Min GMM component separation to accept a trace; 0 to disable

# Filtering
min_peak_number: 3        # Min peaks required
first_peak_time: 1000     # Latest frame for first peak
last_peak_time: 200       # Earliest frame for last peak (from end)

# Diagnostics
diagnostics:
  enabled: True           # Run diagnostic analysis after filtering
  n_examples: 12          # Number of example traces to plot
  proteins: null          # Specific proteins to diagnose (null = all proteins)

# Parallel Processing
n_workers: -1             # -1 or null: use all CPUs, 1: sequential, N>1: use N CPUs
```

See `Extraction/config.yaml` for a complete annotated configuration example.

## Data Organization

### Input Structure
```
Data/movies/
├── Exp1/
│   ├── ProteinA/
│   │   ├── optimized_localization_params_gt.yaml      # Auto-generated by paramfinder.py
│   │   ├── optimization_gt.db                         # Auto-generated Optuna database
│   │   ├── visualQC/gt/trial_*.tif                   # Auto-generated QC images
│   │   ├── ProteinA_640nm_movie_0001.nd2
│   │   ├── ProteinA_488nm_movie_0001.nd2  # Optional (only if has_ground_truth: true)
│   │   └── ...
│   └── ProteinB/
│       └── ...
└── Exp2/
    └── ...
```

### Output Structure (with ground truth)
```
Results/Extract/<your_run_folder>/
├── run_info.yaml                      # Run metadata (proteins, channels, etc.)
│
├── FileLists/
│   ├── trace_file_list.pkl            # Successfully processed trace files
│   ├── background_file_list.pkl       # Background trace files (if n_background_traces > 0)
│   └── failed_file_list.pkl           # Failed files (if any)
│
├── UniqueIDs/
│   ├── ProteinA_uniqueID_all.pkl          # All traces with metadata
│   ├── ProteinA_uniqueID_IN.pkl           # IN traces with metadata
│   ├── ProteinA_uniqueID_OUT.pkl
│   └── ProteinA_uniqueID_background.pkl
│
├── Exp1/ProteinA/                         # Per-movie outputs (localization + extraction)
│   ├── *_locs.hdf5
│   ├── *_locs.yaml
│   ├── *_traces.pkl
│   └── *_background_traces.pkl
│
├── ProteinTracesIN/
│   ├── AllRaw/     ProteinA_IN_raw.pkl
│   ├── AllBG/      ProteinA_IN_bg_rm.pkl
│   ├── AllNorm/    ProteinA_IN_zscored.pkl
│   │               ProteinA_IN_minmax.pkl
│   └── Filtered/   ProteinA_IN_filtered_raw.pkl
│                   ProteinA_IN_filtered_bg_rm.pkl
│                   ProteinA_IN_filtered_zscored.pkl    ← ML-ready
│                   ProteinA_IN_filtered_minmax.pkl     ← ML-ready
│
├── ProteinTracesOUT/
│   └── (same structure)
│
├── BackgroundTraces/
│   ├── AllRaw/     ProteinA_background_raw.pkl
│   ├── AllBG/      ProteinA_background_bg_rm.pkl
│   ├── AllNorm/    ProteinA_background_zscored.pkl
│   │               ProteinA_background_minmax.pkl
│   └── Filtered/   (unused — background traces are not peak-filtered)
│
├── Diagnostics/
│   ├── IN/
│   │   └── ProteinA_IN_trace_examples/    # plot.pdf + data_panel_traces.csv
│   ├── OUT/
│   │   └── ProteinA_OUT_trace_examples/
│   └── Background/
│       └── ProteinA_background_trace_examples/
│
└── ... (same for ProteinB)

Original Data Folders:
input_folder/Exp1/ProteinA/
├── optimized_localization_params_gt.yaml   # Optimized parameters (from Step 0)
├── used_localization_params_gt.yaml        # Actual parameters used (from Step 2)
├── optimization_gt.db                      # Optuna database
├── visualQC/gt/                            # QC images from optimization
└── *_640nm_*.nd2                           # Original movies
```

**Key File Naming:**
- `run_info.yaml` - Run metadata only (no parameters, since they vary by protein)
- `AllRaw/`, `AllBG/`, `AllNorm/` - All traces before quality filtering
- `Filtered/` - Quality-filtered (ML-ready) traces
- Parameters saved in protein folders, not in run folder

## Common Workflows

### Full Pipeline from Scratch
```bash
python Extraction/run_pipeline.py -c Extraction/config.yaml
```

### Re-run Filtering with Different Thresholds
```bash
# 1. Edit config.yaml - change min_peak_number, thresholds, etc.
# 2. Resume from filter (will also re-run diagnostics)
python Extraction/run_pipeline.py -c Extraction/config.yaml \
  --start-from filter \
  -r Results/Extract/<your_run_folder>
```

### Run Diagnostics Only
```bash
# Analyze existing filtered data without re-filtering
python Extraction/run_pipeline.py -c Extraction/config.yaml \
  --start-from diagnose \
  -r Results/Extract/<your_run_folder>

# Or run diagnose.py directly for more control
python Extraction/diagnose.py -c Extraction/config.yaml \
  -r Results/Extract/<your_run_folder> \
  --protein <ProteinName> --n-examples 20
```

### Try Different Analysis Mode
```bash
# 1. Edit config.yaml - change has_ground_truth: false
# 2. Run parameter optimization if needed
python Extraction/paramfinder.py -c Extraction/config.yaml
# 3. Run full pipeline (creates NEW run folder with mode-specific naming)
python Extraction/run_pipeline.py -c Extraction/config.yaml
```

### Re-run Combining/Filtering After Data Fixes
```bash
# Resume from combine
python Extraction/run_pipeline.py -c Extraction/config.yaml \
  --start-from combine \
  -r Results/Extract/<your_run_folder>
```

## Using Config Defaults (Convenience)

Set defaults in `config.yaml`:
```yaml
start_from: filter
resume_run_folder: Results/Extract/<your_run_folder>
```

Then just run:
```bash
python Extraction/run_pipeline.py -c Extraction/config.yaml
```

Override via command-line:
```bash
python Extraction/run_pipeline.py -c Extraction/config.yaml --start-from combine
```

## Individual Step Execution

For maximum control, run steps individually:

```bash
PYTHON=~/opt/anaconda3/envs/picasso-env/bin/python

# Step 0 (Optional): Optimize parameters
$PYTHON Extraction/paramfinder.py -c Extraction/config.yaml

# Step 1: Localization
$PYTHON Extraction/localize.py -c Extraction/config.yaml

# Get run folder from output
RUN_FOLDER="Results/Extract/<your_run_folder>"

# Step 2: Extract traces
$PYTHON Extraction/extract.py -c Extraction/config.yaml -r $RUN_FOLDER

# Step 3: Combine by protein
$PYTHON Extraction/combine.py -c Extraction/config.yaml -r $RUN_FOLDER

# Step 4: Filter for ML
$PYTHON Extraction/filter.py -c Extraction/config.yaml -r $RUN_FOLDER

# Step 5: Quality diagnostics (optional)
$PYTHON Extraction/diagnose.py -c Extraction/config.yaml -r $RUN_FOLDER
```

## Hardware Requirements

### Mac (Development)
- **RAM**: 16+ GB
- **Disk**: ~2GB per 100 movies
- **CPUs**: Multi-core recommended for parallelization
- **Speed**: ~1-5 movies/minute per core (CPU-based MLE)
- **Parallelization**: Automatically uses all available CPU cores by default

### HPC (Production)
- **CPUs**: Multi-core for parallel processing of multiple movies
- **RAM**: 32+ GB
- **Speed**: Scales with number of CPU cores available
- **Parallelization**: Set `n_workers` in config to control CPU usage

## Troubleshooting

### "Picasso not found"
```bash
conda activate picasso-env
python -m picasso --help

# If not installed:
pip install picassosr
```


### "No paired movies found"
- Check `protein_channel` and `ground_truth_channel` match your filenames
- Verify folder structure matches expected hierarchy
- Check protein names in config match folder names
- If using single-channel mode (has_ground_truth: false), ensure protein_channel files exist

### "No traces passed filtering" or "Very few traces pass filtering"
1. **Run diagnostics to understand why**:
   ```bash
   python Extraction/diagnose.py -c Extraction/config.yaml -r YOUR_RUN_FOLDER
   ```
   This will show you:
   - How many traces fail each criterion
   - Quality metric distributions (SNR, peak count, etc.)
   - Visual examples of passed vs. failed traces

2. **Common solutions based on diagnostic output**:
   - If SNR is too low: Lower `snr_min_separation` (e.g., 3.0 → 2.0) or `gmm_proba_threshold`
   - If traces fail `min_peak_number`: Lower the threshold (e.g., 8 → 5)
   - If traces fail `first_peak_time`: Increase the window (e.g., 1000 → 1500)
   - If traces fail `delta_first_second`: Increase max spacing (e.g., 1000 → 1500)
   - Check `movie_length` matches your data

3. **After adjusting config, re-filter**:
   ```bash
   python Extraction/run_pipeline.py -c Extraction/config.yaml \
     --start-from filter -r YOUR_RUN_FOLDER
   ```

### Run folder not created
- Check `output_folder_base` path exists
- Verify write permissions
- Check disk space

### Slow processing performance
- Increase `n_workers` to use more CPUs (set to -1 to use all available cores)
- Verify CPU utilization during processing
- Consider processing smaller batches of movies

### Parallel processing errors
- Reduce `n_workers` if encountering memory issues
- Set `n_workers: 1` for sequential processing to debug issues
- Check available RAM (each worker needs memory for movie processing)

## Integration with ML Pipeline

The filtered outputs are ready for ML training:

```python
import pandas as pd

# Load filtered traces for a protein
grx1_in = pd.read_pickle('Results/Extract/.../ProteinTracesIN/Filtered/ProteinA_IN_filtered_zscored.pkl')
grx1_out = pd.read_pickle('Results/Extract/.../ProteinTracesOUT/Filtered/ProteinA_OUT_filtered_zscored.pkl')

# Shape: (6000 frames × N traces)
print(f"IN traces: {grx1_in.shape}")
print(f"OUT traces: {grx1_out.shape}")

# Use with ML training pipeline
# (see ML/train.py)
```

Or directly in ML config:
```yaml
# ML/config_train.yaml
data:
  data_path: /path/to/Results/Extract/<your_run_folder>
  trace_normalization: zscored_filtered
```

## References

- **Picasso**: https://github.com/jungmannlab/picasso
- **ND2 Format**: https://github.com/tlambert03/nd2
