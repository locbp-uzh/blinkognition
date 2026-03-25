# Analysis Scripts

This document describes two specialized analysis scripts for investigating ground truth (GT) localization and protein-to-GT correlation in dual-channel microscopy data.

## Overview

Both scripts are located in `Experiments/VesicleAnalysis/` and work with dual-channel ND2 microscopy data:
- **640nm channel**: Protein signal
- **488nm channel**: Ground truth (GT) signal

They use Picasso for localization and perform intensity-based analyses to understand signal characteristics.

## gradientsweep.py

Performs systematic gradient threshold sweeps for ground truth localization to understand how gradient selection affects intensity distributions.

### Purpose

Determines optimal gradient thresholds for GT detection by analyzing how different gradient values affect:
- Number of detections
- Intensity distributions
- Detection quality (signal vs. noise)

### Usage

```bash
cd Experiments/VesicleAnalysis
python gradientsweep.py -c config.yaml [--protein PROTEIN]
```

### Configuration

The config file should specify:
```yaml
proteins: [HTHTL, HTIA, SNAP]
input_folder: /path/to/movies
output_folder_base: /path/to/results
ground_truth_channel: "488nm"
boxsize: 9
n_frames_integrate: 10
n_workers: -1  # Use all CPUs

# Define gradients to test
gradient_list: [2000, 5000, 10000, 20000]  # Explicit list
# OR
gradient_sweep:
  min: 1000
  max: 20000
  step: 1000

# Clustering parameters
max_distance_ground_truth: 2.5  # pixels
min_on_ground_truth: 3          # minimum localizations per cluster
```

### Output

For each protein, creates:
- `localizations/`: HDF5 localization files for each gradient (e.g., `movie_grad5000_locs.hdf5`)
- `intensity_data/`: CSV summary with statistics for each gradient
- `plots/`: 4-panel comparison figure showing:
  1. Overlaid histograms (linear scale)
  2. Log-scale distributions
  3. Detection count vs. gradient
  4. Intensity statistics (median, P10, P90) vs. gradient

### Workflow

1. For each gradient value:
   - Run Picasso localization on 488nm channel
   - Cluster localizations (DBSCAN-like, 2.5px radius, min 3 locs)
   - Extract integrated intensities (9×9 pixels × 10 frames)
2. Compare distributions across gradients
3. Identify optimal gradient balancing detection count and intensity

### Parallelization

Supports multi-CPU processing:
- `n_workers: -1` - Use all available CPUs
- `n_workers: 1` - Sequential processing
- `n_workers: N` - Use N CPUs

## gt_protein_correlation.py

Tests whether protein signal (640nm) localizations correspond to vesicles with weak but above-background GT signal (488nm).

### Purpose

Addresses the question: Do protein-only localizations (detected in 640nm but not in 488nm with standard gradient) represent real vesicles with dim GT signal, or are they false positives?

### Hypothesis

If protein-only positions show elevated 488nm intensity compared to random background, they may represent real vesicles below the GT detection threshold.

### Usage

```bash
cd Experiments/VesicleAnalysis
python gt_protein_correlation.py -c config.yaml \
  [--protein PROTEIN] \
  [--protein-gradient 60000] \
  [--gt-gradient 5000]
```

### Configuration

Same as `analyze_gt_gradient.py`, but requires both channels:
```yaml
proteins: [HTHTL]
input_folder: /path/to/movies
output_folder_base: /path/to/results
protein_channel: "640nm"
ground_truth_channel: "488nm"
boxsize: 9
n_frames_integrate: 10
max_distance_ground_truth: 2.5
min_on_ground_truth: 3
```

### Workflow

1. **Localize both channels**:
   - 640nm (protein): High gradient (default: 60000)
   - 488nm (GT): Lower gradient (default: 5000)

2. **Cluster localizations**:
   - Find cluster centers for both channels
   - DBSCAN-like clustering (2.5px radius, min 3 locs)

3. **Filter protein positions**:
   - Remove protein clusters overlapping with GT clusters
   - Exclusion radius: `boxsize × √2 ≈ 13 pixels` for 9×9 boxes
   - Ensures protein-only positions don't co-localize with detected GT

4. **Extract 488nm intensities**:
   - At non-overlapping protein positions (9×9 pixels × 10 frames)
   - At random background positions (avoiding both GT and protein clusters)
   - Compare distributions

5. **Statistical testing**:
   - Independent samples t-test
   - Null hypothesis: No difference between protein positions and background

### Output

For each protein, creates:
- `localizations/`: HDF5 files for 640nm and 488nm localizations
- `data/`: CSV summary with statistics and t-test results
- `plots/`: Multiple visualization types

#### Per-Movie Plots
`plots/per_movie/` contains three plots for each movie:

1. **640nm_protein_rois.png**: Max projection with protein ROI overlays (yellow boxes)
2. **488nm_roi_overlays.png**: Max projection with three ROI types:
   - Green solid: Detected GT ROIs
   - Red solid: Protein-only positions
   - Cyan dotted: Random background ROIs
3. **488nm_line_profiles.png**: 4-panel figure with:
   - Panel 1: Max projection with numbered selected ROIs (blue=GT, yellow=protein, white=background)
   - Panels 2-4: Horizontal line profiles through center of each ROI type

#### Aggregate Plots
`plots/{protein}_protein_gt_correlation.png` - 4-panel comparison:
1. Overlaid histograms (linear scale)
2. Box plots with means
3. Log-scale distributions
4. Statistics table with t-test results

### Interpretation

Key metrics in the statistics table:
- **Mean/Median ratio**: How much higher protein positions are vs. background
- **t-test p-value**: Statistical significance (p < 0.05 indicates significant difference)
- **N**: Number of non-overlapping protein-only positions found

Expected results:
- **High ratio + significant p-value**: Protein-only positions likely represent real dim vesicles
- **Low ratio or non-significant**: Protein-only positions indistinguishable from background (possible false positives)

### Important Notes

1. **Overlap filtering is critical**: Without it, protein positions co-localized with GT would artificially inflate the protein-only intensity, defeating the purpose of the analysis.

2. **Exclusion radius calculation**: For square boxes of side `boxsize`, diagonal separation is `boxsize × √2`. This ensures complete spatial separation between ROIs.

3. **Sample size**: The number of non-overlapping protein-only positions may be small if most protein localizations co-localize with GT. Small sample sizes reduce statistical power.

4. **Line profiles**: Provide visual confirmation that selected ROIs represent the expected signal characteristics.

## Common Parameters

Both scripts share these parameters:

- `boxsize`: ROI size for intensity integration (typically 9 pixels)
- `n_frames_integrate`: Number of frames to integrate (typically 10)
- `max_distance_ground_truth`: Clustering radius in pixels (typically 2.5)
- `min_on_ground_truth`: Minimum localizations per cluster (typically 3)

## Dependencies

Both scripts require:
- Picasso (for localization)
- nd2 (for reading ND2 files)
- numpy, pandas, matplotlib, scipy
- YAML configuration file
- `utils.py` (from Extraction pipeline)

## Typical Workflow

1. **First**, run gradient sweep to find optimal GT threshold:
   ```bash
   python analyze_gt_gradient.py -c config.yaml --protein HTHTL
   ```

2. **Review** gradient comparison plots to select optimal threshold

3. **Then**, run correlation analysis with chosen gradient:
   ```bash
   python analyze_protein_to_gt_correlation.py -c config.yaml \
     --protein HTHTL --gt-gradient 5000
   ```

4. **Examine** line profile plots to visually confirm signal quality

5. **Interpret** t-test results to determine if protein-only positions are distinguishable from background
