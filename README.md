# Blinkognition

Code for Puentener et al. 2026 — single-molecule protein trace analysis pipeline.

## Repository Structure

```
blinkognition/
├── Extraction/         Trace extraction pipeline (ND2 → ML-ready traces)
├── ML/                 Model training and cross-validation
├── VesicleAnalysis/    Vesicle experiments (GT correlation, cargo exchange, occupancy, DLS/FCS)
├── assets/             Colormaps, plotting utilities, paper figure scripts
├── docs/               Documentation for each component
└── Results/            Output directory (Extract/, Train/, CrossVal/)
```

## Environment Setup

```bash
# NVIDIA GPUs (HPC cluster)
conda env create -f blink2cuda.yaml

# Apple Silicon
conda env create -f blink2mac.yaml

# VesicleAnalysis scripts require the picasso environment
conda env create -f picasso-env.yaml
```

## Extraction Pipeline

Processes ND2 microscopy movies into ML-ready trace files. Uses GMM-based background removal throughout.

```bash
cd Extraction

# Run full pipeline
python run_pipeline.py -c config.yaml

# Start from a specific step
# Edit config.yaml: start_from: localize  (or paramfinder, extract, combine, filter, diagnose)

# Control parallelization via config.yaml
# n_workers: -1    # all CPUs
# n_workers: 16    # specific count

# SLURM
sbatch submit_extraction.sh
```

Supports two modes via `has_ground_truth` in config:
- `true` — dual-channel analysis, traces labeled IN/OUT based on colocalization with ground truth
- `false` — single-channel, no labels

See `docs/extraction_pipeline.md` for full documentation.

## ML Pipeline

### Training

```bash
cd ML
python train.py -c config_train.yaml
sbatch submit_train.sh
```

### Cross-Validation

```bash
cd ML

# Single job
python crossval.py -c configs_cv/config_cv_notmirrored_minmax.yaml

# Submit all jobs for a protein pair
sbatch submit_cv_hthtl_vs_snap.sh     # HTHTL vs snap (9 configs)
sbatch submit_cv_hthtl_vs_htia.sh     # HTHTL vs HTIA (9 configs)
sbatch submit_cv_grx1_vs_k20ac.sh     # Grx1 vs K20Ac (9 configs)
```

CV configs follow the naming convention `{dataset}_{protein_pair}_{channel}`:
- **dataset**: `mirrored` (time-reversal augmentation on), `notmirrored` (off), `background` (background traces)
- **channel**: `minmax`, `zscored`, `both`

See `docs/train.md` and `docs/crossval.md` for full documentation.

## Vesicle Analysis

All scripts require the `picasso-env` environment.

```bash
cd VesicleAnalysis
conda activate picasso-env

# GT detection optimization — sweep gradient thresholds
python gradientsweep.py -c config.yaml --protein HTHTL

# Cargo exchange analysis
python cargo_exchange.py -c config.yaml

# Occupancy analysis
python occupation.py -c config.yaml

# Vesicle characterization (DLS / FCS)
python dls_analysis.py -c config.yaml
python fcs_analysis.py -c config.yaml
```

See `docs/analysis_scripts.md` for full documentation.

## Paper Figure Scripts

Publication-quality figures are in `assets/figures/`:

```bash
python assets/figures/figure_S3_dls.py -c assets/figures/configs/config_S3_dls.yaml
python assets/figures/figure_S4_nanofcm.py -c assets/figures/configs/config_S4_nanofcm.yaml
python assets/figures/figure_S5_cargo_exchange.py -c assets/figures/configs/config_S5_cargo_exchange.yaml
python assets/figures/figure_S6_occupation.py -c assets/figures/configs/config_S6_occupation.yaml
```

Colormaps and plotting standards are in `assets/colormaps/` and documented in `docs/plotting_standards.md`.
