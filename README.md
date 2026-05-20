# Blinkognition

A pipeline for classifying single-molecule fluorescence traces by protein identity. Raw ND2 movies are processed into filtered, normalized traces; deep-learning models are trained and cross-validated on those traces; and photophysical features are extracted from model-selected, high-certainty traces. Supplementary vesicle characterization scripts (DLS, FCS, cargo exchange, occupancy) are provided in a separate module.

The old codebase for the 2023 JACS paper can still be found at: https://gitlab.uzh.ch/locbp/public/blinkognition

## Repository Structure

```
blinkognition/
├── Extraction/         Trace extraction pipeline (ND2 → ML-ready traces)
├── Features/           Photophysical feature extraction from classified traces
├── ML/                 Model training and cross-validation
├── Controls/           Validation experiments for the trained classifier
├── assets/colormaps/   Crameri colormaps (used by ML/utils.py)
├── docs/               Per-module documentation
└── Inputs/             Downloaded data (not committed; see below)
```

## Input Data

Two datasets are archived on Zenodo (DOI: pending). They serve different purposes
and are independent of each other.

### Filtered traces (for ML, Features, and Controls)

```bash
python download_data.py --dataset traces
```

Downloads pre-processed, filtered traces (~5.2 GB) into `Inputs/FinalTraces/`. This
is the dataset used to train the classifier, run cross-validation, and compute
photophysical features. All `ML/`, `Controls/label_scrambling/`, and
`Controls/random_forest/` configs read from here — no path edits needed after
downloading.

### Sample movies (for evaluating the Extraction pipeline)

```bash
python download_data.py --dataset movies
```

Downloads a representative set of raw ND2 movies (~5.3 GB) into `Inputs/SampleMovies/`.
These are provided so that the full Extraction pipeline can be tested
end-to-end and experimented with. The output goes to `Results/Extraction/` and is not used as input to ML because the sample movies cover only a subset of the data and do not produce enough traces for
training. To reproduce the ML experiments, use the pre-processed traces above.

Two configs require a path set after training:
- `Features/config.yaml` — set `mcd_filter.npz_path` to the `.npz` file from your training run
- `Controls/noise_classification/config_noise.yaml` — set `pretrained_model_dir` to your training run folder

## Environments

```bash
# NVIDIA GPU (HPC cluster)
conda env create -f blink2cuda.yaml

# Apple Silicon
conda env create -f blink2mac.yaml

# Trace Extraction requires a separate Picasso environment
conda env create -f picasso-env.yaml
```

`blink2cuda` and `blink2mac` cover ML, Features, and Controls.
`picasso-env` is required for Extraction.

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
python crossval.py -c config_cv.yaml
```

Edit `ML/config_cv.yaml` to choose proteins, channels, models, and augmentation settings. The file is annotated with the exact values used in the paper. See `docs/crossval.md` for all options.

#### Reproducing the paper's CV experiments

The paper compared three protein pairs across two normalizations (minmax, zscored) and three trace variants (protein with mirroring, protein without mirroring, background traces), using all three model architectures and an augmentation factor sweep — 27 combinations in total. To reproduce any one of them, edit `config_cv.yaml` with the settings below and run `python crossval.py -c config_cv.yaml`.

Key settings for all paper runs:

```yaml
compare:
  models: ["orig_conv_gru", "resnet1d", "tcn"]

cv:
  n_splits: 4

augmentation_sweep:
  enabled: true
  factors: [0, 3, 5]
  augmentation_params:
    time_warp_sigma: 0.5
    noise_sigma: 0.5
    magnitude_jitter: 0.5
    include_mirror: true  # set false for the not-mirrored variant

system:
  n_gpus: 3  # each run used 3× A100/H100 80 GB on SLURM
```

Per-comparison settings:

| Protein pair | `dataset` keys | `traces_path` | `channels` |
|---|---|---|---|
| HaloD106 vs SNAPC148 | `HaloD106`, `SNAPC148` | `ProteinTracesIN/Filtered` | `["minmax", "zscored"]` |
| HaloD106 vs HaloK117 | `HaloD106`, `HaloK117` | `ProteinTracesIN/Filtered` | `["minmax", "zscored"]` |
| scGrx1 vs scGrx1AcK20 | `scGrx1`, `scGrx1AcK20` | `ProteinTracesIN/Filtered` | `["minmax", "zscored"]` |
| Background (noise control) | same pairs | `BackgroundTraces/Filtered` | `["minmax", "zscored"]` |

On SLURM, submit each run as a batch job:

```bash
cd ML
sbatch --job-name=cv_halod106_snapc148 \
  --account=<account> --partition=standard \
  --gres=gpu:3 --constraint=GPUMEM80GB \
  --cpus-per-task=12 --mem=192G --time=23:59:00 \
  --wrap="module load miniforge3 && conda activate blink2-cuda && python crossval.py -c config_cv.yaml"
```

Results are written to `Results/CrossVal/<run_name>/`.

### Feature Extraction

Extracts per-trace photophysical features (mean ON time, mean OFF time, duty cycle) from traces selected by MC Dropout certainty. Produces violin plots and CSVs with optional statistical testing between two protein classes.

```bash
cd Features
python features.py -c config.yaml
```

See `docs/features.md`.

### Control Experiments

Three experiments for validating a trained classifier: label scrambling, noise-trace classification, and a random-forest baseline on handcrafted blinking statistics.

```bash
cd Controls/label_scrambling  &&  python scramble_train.py  -c config_scramble.yaml
cd Controls/noise_classification && python classify_noise.py -c config_noise.yaml
cd Controls/random_forest     &&  python rf_pipeline.py    -c config_rf.yaml
```

See `docs/controls.md`.
