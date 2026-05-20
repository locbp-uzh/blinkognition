# Blinkognition

A pipeline for classifying single-molecule fluorescence traces by protein identity. Raw ND2 movies are processed into filtered, normalized traces; deep-learning models are trained and cross-validated on those traces; and photophysical features are extracted from model-selected, high-certainty traces. Supplementary vesicle characterization scripts (DLS, FCS, cargo exchange, occupancy) are provided in a separate module.

## Repository Structure

```
blinkognition/
├── Extraction/         Trace extraction pipeline (ND2 → ML-ready traces)
├── Features/           Photophysical feature extraction from classified traces
├── ML/                 Model training and cross-validation
├── VesicleAnalysis/    Vesicle characterization (DLS, FCS, cargo exchange, occupancy)
├── Controls/           Validation experiments for the trained classifier
├── assets/colormaps/   Crameri colormaps (used by ML/utils.py and VesicleAnalysis)
├── docs/               Per-module documentation
└── Inputs/             Downloaded data (not committed; see below)
```

## Input Data

The raw trace files and sample movies are archived on Zenodo (DOI: pending). Download
them into `Inputs/` before running the pipeline:

```bash
# Filtered traces for ML, Features, and Controls (~5.2 GB)
python download_data.py --dataset traces

# Sample ND2 movies for re-running the Extraction pipeline
python download_data.py --dataset movies

# Both datasets
python download_data.py --dataset all
```

All config files in `ML/`, `Features/`, and `Controls/` are pre-configured to read
from `Inputs/FinalTraces/` so no path edits are needed after downloading.

## Environments

```bash
# NVIDIA GPU (HPC cluster)
conda env create -f blink2cuda.yaml

# Apple Silicon
conda env create -f blink2mac.yaml

# VesicleAnalysis and trace Extraction requires a separate Picasso environment
conda env create -f picasso-env.yaml
```

`blink2cuda` and `blink2mac` cover ML, Features, and Controls.
`picasso-env` is required for VesicleAnalysis and Extraction.

## Modules

### Extraction

Processes ND2 movies into ML-ready trace files through six sequential steps: parameter optimization, localization, trace extraction, combining per-movie outputs, quality filtering, and diagnostics. Supports both dual-channel (with colocalized ground truth for IN/OUT labeling) and single-channel modes.

```bash
cd Extraction
python run_pipeline.py -c config.yaml
```

See `docs/extraction_pipeline.md`.

### ML: Training

Trains a supervised classifier on labeled trace data with early stopping, MC Dropout uncertainty quantification, and a Wasserstein-distance threshold sweep for filtering low-certainty predictions.

```bash
cd ML
python train.py -c config_train.yaml
# or: sbatch submit_train.sh
```

See `docs/train.md`.

### ML: Cross-Validation

Compares model architectures and augmentation strategies using stratified K-fold cross-validation with optional multi-GPU parallelization.

```bash
cd ML
python crossval.py -c configs_cv/config_cv_<name>.yaml
```

See `docs/crossval.md`.

#### Reproducing the paper's CV experiments

The paper compares three protein pairs across three channel normalizations and three trace types (protein mirrored, protein not-mirrored, background not-mirrored), giving 27 jobs in total. Pre-built configs are in `ML/configs_cv/`. On an HPC cluster with SLURM, submit all 27 jobs in three batches:

```bash
cd ML
bash submit_cv_halod106_vs_snapc148.sh       # HaloD106 vs SNAPC148 (9 jobs)
bash submit_cv_halod106_vs_halok117.sh       # HaloD106 vs HaloK117 (9 jobs)
bash submit_cv_scgrx1_vs_scgrx1ack20.sh     # scGrx1 vs scGrx1AcK20 (9 jobs)
```

Each job runs on 3× A100/H100 80 GB GPUs and takes up to 24 hours. Results are written to `Results/CrossVal/<run_name>/`. To run a single config locally (e.g., for debugging):

```bash
cd ML
python crossval.py -c configs_cv/config_cv_protein_mirrored_halod106_snapc148_minmax.yaml
```

Config naming convention: `config_cv_{trace_type}_{protein1}_{protein2}_{channel}.yaml`
- `trace_type`: `protein_mirrored`, `protein_notmirrored`, `background_notmirrored`
- `protein1/2`: `halod106`, `halok117`, `snapc148`, `scgrx1`, `scgrx1ack20`
- `channel`: `minmax`, `zscored`, `both`

### Feature Extraction

Extracts per-trace photophysical features (mean ON time, mean OFF time, duty cycle) from traces selected by MC Dropout certainty. Produces violin plots and CSVs with optional statistical testing between two protein classes.

```bash
cd Features
python features.py -c config.yaml
```

See `docs/features.md`.

### Vesicle Analysis

Standalone scripts for vesicle characterization experiments. Each script reads a YAML config and writes structured output.

```bash
cd VesicleAnalysis
conda activate picasso-env
python cargo_exchange.py -c config.yaml
python occupation.py -c config.yaml
python dls_analysis.py -c config.yaml
python fcs_analysis.py -c config.yaml
```

See `docs/analysis_scripts.md`.

### Control Experiments

Three experiments for validating a trained classifier: label scrambling, noise-trace classification, and a random-forest baseline on handcrafted blinking statistics.

```bash
cd Controls/label_scrambling  &&  python scramble_train.py  -c config_scramble.yaml
cd Controls/noise_classification && python classify_noise.py -c config_noise.yaml
cd Controls/random_forest     &&  python rf_pipeline.py    -c config_rf.yaml
```

See `docs/controls.md`.
