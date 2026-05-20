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

After downloading, update the `traces_path` (and background path) in
`ML/config_train.yaml`, `ML/config_cv.yaml`, `Features/config.yaml`, and
the Controls configs to point to `Inputs/FinalTraces/...`.

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
# or: sbatch submit_cv.sh
```

See `docs/crossval.md`.

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
