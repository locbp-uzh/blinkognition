# Vesicle Analysis Scripts

Standalone scripts for vesicle characterization experiments. All scripts live in `VesicleAnalysis/`, read a YAML config file, and require the `picasso-env` environment.

```bash
conda activate picasso-env
cd VesicleAnalysis
```

---

## cargo_exchange.py

Analyzes cargo exchange between vesicles and a target compartment. For each slide, it localizes the ground-truth channel (Picasso), colocalizes protein localizations against vesicle positions, classifies each FOV, and computes per-slide exchange statistics.

```bash
python cargo_exchange.py -c config.yaml
```

Key config fields:

```yaml
proteins: [ProteinA, ProteinB]
input_folder: /path/to/movies
output_folder_base: /path/to/results
protein_channel: "640nm"
ground_truth_channel: "488nm"
boxsize: 9
max_distance_ground_truth: 2.5   # pixels
min_on_ground_truth: 3           # minimum localizations per cluster
```

Outputs: per-slide CSV with colocalization counts and exchange percentages, summary statistics, and bar plots.

---

## occupation.py

Computes the fraction of vesicles occupied by a labeled cargo across a concentration series. Integrates Picasso localization of vesicle positions with per-slide trace data to produce occupancy distributions.

```bash
python occupation.py -c config.yaml
```

Key config fields:

```yaml
proteins: [ProteinA]
input_folder: /path/to/movies
output_folder_base: /path/to/results
ground_truth_channel: "488nm"
concentration_series: [1, 2, 5, 10]
```

Outputs: occupancy distribution plots and CSV with mean/std occupancy per concentration.

---

## dls_analysis.py

Processes Dynamic Light Scattering (DLS) data to characterize vesicle size distributions. Reads Excel/CSV DLS exports, computes hydrodynamic diameter statistics, and produces size-distribution and stability plots.

```bash
python dls_analysis.py -c config.yaml
```

Key config fields:

```yaml
input_folder: /path/to/dls_files
output_folder: /path/to/results
samples:
  - name: SampleA
    file: sample_a.xlsx
```

Outputs: size-distribution overlay plot, stability plot (Z-average vs. time), and summary CSV.

---

## fcs_analysis.py

Analyzes Fluorescence Correlation Spectroscopy (FCS) data. Loads FCS files, gates scatter, computes fluorescence thresholds per dye, classifies quadrants, and computes mixing percentages.

```bash
python fcs_analysis.py -c config.yaml
```

Key config fields:

```yaml
input_folder: /path/to/fcs_files
output_folder: /path/to/results
dyes: [Alexa488, Cy5]
timepoints: [0h, 1h, 4h, 24h]
```

Outputs: quadrant scatter plots per timepoint, mixing percentage vs. time plot, and summary CSV.

---

## Shared utilities

`VesicleAnalysis/utils.py` provides helpers used by all four scripts: axis formatting, localization loading, colocalization, and colormap loading.
