# Vesicle Analysis — Methods Description

This document describes two complementary quantitative analyses of vesicle–protein
interactions: (1) a kinetic cargo-exchange experiment that tracks fluorescently labeled
protein transfer between two distinct vesicle populations over time, and (2) a
photobleaching-based occupancy analysis that counts the absolute number of protein
molecules encapsulated per vesicle as a function of loading concentration.

---

## 1 — Cargo exchange analysis (`cargo_exchange.py`)

### 1.1 Experimental design

The cargo-exchange experiment tests whether proteins can transfer between two
physically distinct vesicle populations after they are brought into contact. Two
populations are distinguished by spectrally separated lipid dyes:

- Population A: Atto520-labeled vesicles (515 nm excitation, Channel C3) containing
  HaloTag-JN275 labeled protein (640 nm excitation, Channel C1).
- Population B: Atto425-labeled vesicles (405 nm excitation, Channel C4) containing
  unlabeled HaloTag protein.

Populations A and B are mixed and deposited on glass slides at four timepoints: 0 h
(on-slide control — populations placed in contact simultaneously with slide deposition),
1 h, 6 h, and 24 h of incubation before slide deposition. Two single-population controls
accompany the timeseries: Atto425 vesicles with unlabeled protein (to measure the baseline
rate of spurious protein-to-Atto425 co-localization), and Atto520 vesicles with labeled
protein (to establish a reference for maximum protein-to-Atto520 co-localization in the
absence of mixing). Each slide is imaged across multiple fields of view (FOVs), each recorded as a
multi-channel ND2 movie with four channels: 638/640 nm (C1, protein), 488 nm (unused),
515 nm (C3, Atto520 vesicles), and 405 nm (C4, Atto425 vesicles).

### 1.2 Localization and colocalization

The script operates in `mode: pipeline`, reading raw ND2 movies directly and running
Picasso MLE localization on each frame of each relevant channel (C1, C3, C4) before
colocalization. Multi-position ND2 files (containing several FOVs) are handled
transparently: each position is extracted and localized independently. Localization
uses the parameters recorded in the original Picasso YAML sidecar files: box size 7 px,
minimum net gradient 20 000, MLE fitting with convergence criterion ε = 0.001 and
maximum 1000 iterations, camera baseline 79 ADU, sensitivity 16.0 e⁻/ADU, gain 300.

For the colocalization step, all protein localizations and all vesicle localizations
from a given FOV (pooled across all frames) are treated as spatial point clouds.
For each protein localization, the nearest vesicle localization in each channel is found
via a KD-tree query [1]. Each protein localization is assigned to exactly one of four
mutually exclusive categories:

- **PV_A520**: within `max_dist` pixels (default 3.0) of an Atto520 (C3) localization but
  not of any Atto425 (C4) localization.
- **PV_A425**: within `max_dist` pixels of an Atto425 (C4) localization but not of any
  Atto520 (C3) localization.
- **PV_both**: within `max_dist` pixels of both an Atto520 and an Atto425 localization
  simultaneously.
- **P_free**: not within `max_dist` of either vesicle type.

By construction, PV_A520 + PV_A425 + PV_both + P_free = n_protein. Classification counts are
summed across all FOVs within a slide and expressed as percentages of the total protein
localization count for that slide.

Note that this analysis operates on raw Picasso localizations, not on linked or tracked
molecules. The same physical molecule contributes one localization per frame in which it
was detected, so the counts are weighted by the number of frames each molecule was
detected in. For the purpose of quantifying population-level transfer fractions, this
weighting is acceptable because it is applied uniformly across all slides and timepoints.

### 1.3 Aggregation and output

Results are aggregated per slide (summing counts across all FOVs) and then sorted by
timepoint. The analysis script writes two CSV files: one per replicate
(`cargo_exchange_{replicate}.csv`) and a combined file (`cargo_exchange_results.csv`)
with a `replicate` column for joint analysis. At t = 0, labeled protein is confined to
Atto520 vesicles (Population A), so pct_in_A520 is high and pct_in_A425 is near zero.
Cargo transfer is reported by the rise of pct_in_A425 over the 0–24 h window, as labeled
protein appears in Atto425 vesicles (Population B) following mixing.

Publication figures are produced by `assets/figures/Figure_S3/figure_S3_cargoexchange_quantitative.py`,
which calls `run_pipeline_mode` to generate the per-replicate CSVs from raw ND2 data and
then renders a four-segment stacked bar chart per replicate (Figure S3). The four segments
decompose the protein population into mutually exclusive fractions: Atto520 only, both
vesicle types simultaneously, Atto425 only, and free protein. A companion script
(`figure_S3_cargoexchange_fov.py`) produces a representative three-panel FOV image
(640 nm protein in magenta, 515 nm Atto520 in cyan, 405 nm Atto425 in blue) with protein
localization boxes overlaid.

---

## 2 — Vesicle occupation analysis (`occupation.py`)

### 2.1 Experimental context

To characterize how many protein molecules are encapsulated per vesicle as a function
of protein loading concentration, a photobleaching step-counting approach is used.
Vesicles are immobilized on a glass surface and imaged under continuous 640 nm
illumination. Each fluorescently labeled protein molecule undergoes irreversible
single-step photobleaching, producing a discrete downward step in the integrated vesicle
fluorescence trace. The number of steps observed in the trace is therefore equal to the
number of labeled protein molecules present in that vesicle at the start of imaging [2].

### 2.2 Trace preprocessing and step detection

Before step detection, traces are preprocessed per vesicle:

1. **Background subtraction.** The mean of the last 50 frames of the raw pixel-sum trace
   is subtracted from every frame (corresponding to the noise floor after all fluorophores
   have bleached).
2. **Per-pixel normalisation.** The background-subtracted trace is divided by the ROI area
   (boxsize² = 49 pixels), converting units to ADU per pixel per frame.

Photobleaching steps are then detected using the `quickpbsa` package (v2021.0.1), which
implements the Kalafut–Visscher (KV) algorithm [3]. The preliminary KV pass fits a
piecewise-constant model to the trace using a maximum-likelihood criterion, with the number
of steps determined by minimizing the Schwarz information criterion (SIC). A refinement
pass then resolves simultaneous multi-molecule bleaching events using a Bayesian posterior
criterion. Parameters: threshold = 75 ADU/pixel, percentile_step = 95,
length_laststep = 5 frames.

Each analyzed vesicle is assigned a quality flag:

- `flag = 1`: KV detected one or more clearly resolved photobleaching steps. The molecule
  count (occupation) is stored in the `fluors_kv` field of the result CSV. Only these
  vesicles are used for the occupation histogram.
- `flag = −1`: KV found zero steps, indicating either that all proteins were already
  bleached before imaging began, that no protein was encapsulated, or that the signal
  was too weak to resolve individual steps. These vesicles are excluded from the
  step-counting result.
- Other negative flags: additional quality-control rejections (e.g. `flag = −3` for
  incomplete bleaching, `flag = −7` when the refinement search space exceeds the
  combination cutoff).

The result CSV is produced directly by `occupation.py` running in `mode: pipeline`
(see `assets/figures/Figure_2C/config_2C_pipeline.yaml`). One row per vesicle per
descriptor type; the occupation value is read from the `fluors_kv` descriptor row,
column `0` (constant across all frame columns for a given vesicle).

### 2.3 Empty-vesicle handling

Vesicles that contain no protein are not detected in the 640 nm channel and therefore do
not appear in the KV result CSV at all. To correctly compute the full occupation
distribution (including the zero-protein fraction), these empty vesicles must be counted
separately. Their number is inferred as the difference between the total vesicle cluster
count from Picasso (read from `clustering_results.txt` in the Picasso output directory)
and the number of vesicles with `flag = 1`. Vesicle clusters are detected from the 488 nm
(lipid dye) channel by density-based clustering [4] applied to all localizations across
the full movie; each cluster represents one vesicle. The resulting empty-vesicle rows
(occupation = 0) are appended to the DataFrame before computing distributions.

The clustering results file occasionally contains duplicate entries (if the pipeline
was re-run on the same data), so the script reads only the first $N_\text{FOV}$ entries,
where $N_\text{FOV}$ is determined by counting the per-FOV trace pickle files in the
directory.

### 2.4 Output

`occupation.py` runs in pipeline mode only: it writes one result CSV per concentration
to the configured `pipeline_output_dir`. All downstream analysis and figure generation
is performed by the dedicated figure scripts (`figure_2C_occupation.py`,
`figure_S4_occupation.py`), which import the data-access helpers
(`load_result_csv`, `discover_result_csvs`, `load_concentration_series`,
`count_vesicle_clusters`, `add_empty_vesicles`) and the plotting standards
(`COLORS`, `FONTSIZE_*`, `apply_axis_standards`) directly from `occupation.py`.

> **Suggested figure:** Representative photobleaching trace of a single vesicle showing
> 2–3 discrete downward steps, with the KV piecewise-constant fit overlaid, to illustrate
> the step-counting method. Not currently generated — would require reading the raw trace
> data and plotting individual examples from the result CSV.

---

## Summary of configurable parameters

### Cargo exchange — pipeline mode (`mode: pipeline`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_dist` | 3.0 | Co-localization radius (pixels) |
| `channel_c1` | 0 | ND2 channel index for 640 nm protein |
| `channel_c3` | 2 | ND2 channel index for 515 nm Atto520 vesicles |
| `channel_c4` | 3 | ND2 channel index for 405 nm Atto425 vesicles |
| `box_size` | 7 | Localization box size (pixels) |
| `min_net_gradient` | 20000 | Picasso minimum net gradient threshold |
| `camera_baseline` | 79.0 | Camera baseline (ADU) |
| `camera_sensitivity` | 16.0 | Camera sensitivity (e⁻/ADU) |
| `camera_gain` | 300.0 | Camera EM gain |

### Occupation — pipeline mode (`mode: pipeline`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `boxsize` | 7 | ROI side length (pixels); per-pixel normalisation divides by boxsize² |
| `bg_frames` | 50 | Frames from the end of each trace used to estimate background |
| `kv_threshold` | 75.0 | quickpbsa KV threshold (ADU/pixel per bleaching step) |
| `kv_max_steps` | 100 | Maximum KV iterations (should exceed expected fluorophore count) |
| `percentile_step` | 95 | Upper percentile bound on single-fluorophore intensity (quickpbsa filter) |
| `length_laststep` | 5 | Minimum frames between the last two steps (quickpbsa filter) |
| `channel_640` | 640nm_TIRF2xLP_50pr | Substring identifying 640 nm trace pkl files |

---

## References

[1] Bentley JL. Multidimensional binary search trees used for associative searching.
*Communications of the ACM* 18, 509–517 (1975).

[2] Ulbrich MH, Isacoff EY. Subunit counting in membrane-bound proteins. *Nature Methods*
4, 319–321 (2007).

[3] Kalafut B, Visscher K. An objective, model-independent method for detection of
non-uniform steps in noisy signals. *Computer Physics Communications* 179, 716–723 (2008).

[4] Ester M, Kriegel H-P, Sander J, Xu X. A density-based algorithm for discovering
clusters in large spatial databases with noise. *Proceedings of the 2nd International
Conference on Knowledge Discovery and Data Mining (KDD)*, 226–231 (1996).
