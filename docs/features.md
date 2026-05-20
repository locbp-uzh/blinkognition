# Feature Extraction (features.py)

Extract photophysical blink feature distributions from protein traces classified by the ML model, and produce violin plots and CSVs.

## Overview

`Features/features.py` takes the `.npz` file produced by `train.py`'s MC Dropout evaluation and computes three per-trace photophysical features from GMM-segmented intensity traces. Traces are filtered by Wasserstein distance and all passing traces are analysed together in one output folder.

The script re-uses `Extraction/utils.gmm_classify_frames` for peak detection, so the same GMM segmentation logic is applied consistently throughout the pipeline.

## Quick Start

```bash
cd Features
python features.py -c config.yaml
```

## Features Computed

Three features are extracted per trace and arranged in a 1×3 panel:

| Panel | Feature | Unit | Description |
|---|---|---|---|
| A | Mean off-time | ms | Mean dark-interval duration per trace |
| B | Mean on-time | ms | Mean peak duration per trace |
| C | Duty cycle | fraction | On-frames / active-window frames per trace |

A trace contributes to mean off-time only if it has at least one dark interval (i.e. at least two peaks); it contributes to mean on-time if it has at least one peak.

The active window is defined as the span from the first frame of the first peak to the last frame of the last peak. Off-times are the gaps between consecutive peaks within that window.

## GMM Peak Detection

GMM is run on channel 0 (minmax-normalized) using the threshold `gmm_proba_threshold`. Peak intensities reported in intermediate outputs use channel 1 (z-scored). The `min_peak_width` parameter sets the minimum number of consecutive "on" frames required for a run to be counted as a peak (identical to the Extraction pipeline setting).

## Statistics

For binary comparisons (exactly two proteins), a Mann-Whitney U test is run on each of the three features. The test makes no distributional assumptions and is appropriate for the right-skewed, bounded distributions typical of blinking statistics. Raw p-values are corrected for multiple testing using Benjamini-Hochberg FDR (six tests when all three panels are considered together). The rank-biserial correlation r is reported alongside each test as an effect-size estimate independent of sample size. Corrected p-values and r values are printed to the log and displayed as significance brackets on the violin plots.

For three or more proteins no statistical annotation is drawn.

## Configuration File

```yaml
mcd_filter:
  npz_path: /path/to/MCD_results/traces_with_wasserstein.npz
  wasserstein_min: 0.7   # traces with WD >= this value are kept

gmm_proba_threshold: 0.9   # GMM posterior threshold for "on" frame classification
min_peak_width: 1          # minimum consecutive "on" frames to count as a peak
frame_interval_ms: 30.0    # camera frame interval in milliseconds
seed: 42                   # RNG seed for example trace selection
```

### Parameter Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mcd_filter.npz_path` | path | — | NPZ from `train.py` MC Dropout output |
| `mcd_filter.wasserstein_min` | float | 0.7 | Minimum WD score to include a trace |
| `gmm_proba_threshold` | float | 0.8 | GMM posterior probability threshold |
| `min_peak_width` | int | 1 | Minimum peak duration in frames |
| `frame_interval_ms` | float | 30.0 | Frame interval in milliseconds |
| `seed` | int | 42 | RNG seed for example trace sampling |

### Wasserstein Distance Threshold

The WD threshold trades off trace count against classification certainty:

| `wasserstein_min` | Approximate fraction retained | Effect |
|---|---|---|
| 0.5 | ~80% | Permissive; keeps uncertain traces |
| 0.7 | ~54% | Balanced |
| 0.9 | ~40% | Conservative; highest-certainty subset |

## Output Structure

```
Results/Features/
└── {proteins}_mcd_{timestamp}/
    ├── features/
    │   ├── plot.pdf                   # 1×3 violin panel (A–C)
    │   ├── data_panel_A.csv           # mean off-time, one column per protein
    │   ├── data_panel_B.csv           # mean on-time
    │   └── data_panel_C.csv           # duty cycle
    └── sample_traces/
        ├── plot_traces.pdf            # example traces windowed around highest peak
        └── data_traces.csv            # trace metadata and raw window values
```

Each `data_panel_X.csv` has one column per protein, one row per trace, NaN-padded to equal length when protein trace counts differ.

`sample_traces/plot_traces.pdf` shows four randomly selected traces per protein, each windowed around the highest peak (by mean minmax value) in a 1000-frame window. Detected peaks are shaded. This plot is for visual quality control.

## NPZ Format Expected

The input NPZ must contain:

| Key | Shape | Description |
|---|---|---|
| `traces` | (N, 2, T) | Channel 0: minmax; channel 1: z-scored |
| `labels` | (N,) | Integer class indices |
| `wasserstein_distances` | (N,) | MCD certainty score per trace |
| `class_names` | (C,) | Protein name strings |

This NPZ is produced automatically by `train.py` when `uncertainty.mc_dropout.enabled: true`.

## See Also

- `docs/train.md` — training pipeline and MC Dropout configuration
- `Extraction/utils.py` — `gmm_classify_frames` used for peak detection
