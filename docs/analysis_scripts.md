# Analysis Scripts

## gradientsweep.py

Performs systematic gradient threshold sweeps for ground truth localization to understand how gradient selection affects intensity distributions.

Located in `VesicleAnalysis/`. Requires the `picasso-env` environment.

### Purpose

Determines optimal gradient thresholds for GT detection by analyzing how different gradient values affect:
- Number of detections
- Intensity distributions
- Detection quality (signal vs. noise)

### Usage

```bash
cd VesicleAnalysis
conda activate picasso-env
python gradientsweep.py -c config.yaml [--protein PROTEIN]
```

### Configuration

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
- `localizations/`: HDF5 localization files for each gradient
- `intensity_data/`: CSV summary with statistics for each gradient
- `plots/`: 4-panel comparison figure showing:
  1. Overlaid histograms (linear scale)
  2. Log-scale distributions
  3. Detection count vs. gradient
  4. Intensity statistics (median, P10, P90) vs. gradient

### Workflow

1. For each gradient value:
   - Run Picasso localization on the GT channel
   - Cluster localizations (2.5px radius, min 3 locs)
   - Extract integrated intensities (boxsize × n_frames_integrate)
2. Compare distributions across gradients
3. Identify optimal gradient balancing detection count and intensity quality

### Dependencies

- Picasso (for localization)
- nd2 (for reading ND2 files)
- numpy, pandas, matplotlib, scipy
- `utils.py` in VesicleAnalysis/
