# Plotting Standards

This document defines the visual standards for all plots and figures in this repository.

## Core Principles

1. **Font**: Arial (primary), Helvetica (macOS fallback), DejaVu Sans (Linux/HPC fallback), never smaller than 7 points
2. **Font weight**: Regular (no bold unless specifically requested)
3. **Axes**: Only draw left (y-axis) and bottom (x-axis) spines (remove top and right)
4. **Grid**: No grid unless specifically requested
5. **Colors**: Use only color-blind friendly palettes
6. **Capitalization**: Use sentence-style capitalization for all titles and axis labels (only first word capitalized), not headline-style (all major words capitalized)

## Implementation

### Font Configuration

```python
import matplotlib as mpl

# Set font to sans-serif with cross-platform fallbacks
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 9  # Base font size

# Font sizes for different elements
FONTSIZE_LABEL = 9      # Axis labels
FONTSIZE_TICK = 8       # Tick labels (minimum 7pt)
FONTSIZE_TITLE = 10     # Plot titles
FONTSIZE_LEGEND = 8     # Legend text
```

Note: Arial is the primary font (available on macOS and Windows). Helvetica is the macOS system fallback. DejaVu Sans is the Linux/HPC fallback (bundled with matplotlib).

### Color Palette

Use the Okabe-Ito color-blind friendly palette:

```python
COLORS = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
    'black': '#000000',
}
```

These colors are distinguishable for people with all common forms of color blindness.

### Axis Configuration

For each axis in your plot:

```python
# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Make left and bottom spines slightly thicker (10% increase)
ax.spines['left'].set_linewidth(1.1)
ax.spines['bottom'].set_linewidth(1.1)

# Set tick label size
ax.tick_params(labelsize=8)
```

### Labels and Titles

All titles and labels should use sentence-style capitalization:
- **Sentence style (correct)**: "Log-scale intensity distributions", "Detection count vs gradient"
- **Headline style (incorrect)**: "Log-Scale Intensity Distributions", "Detection Count vs Gradient"

```python
# Axis labels (sentence-style capitalization)
ax.set_xlabel('Time (s)', fontsize=9)  # Correct
ax.set_ylabel('Integrated intensity', fontsize=9)  # Correct

# Title with increased padding (50% more separation from axes)
ax.set_title('Log-scale intensity distributions', fontsize=10, pad=9)  # Correct
```

Examples of proper capitalization:
- "Median and percentiles vs gradient" ✓
- "Distribution of trace lengths" ✓
- "Signal intensity over time" ✓
- "Median And Percentiles Vs Gradient" ✗
- "Distribution Of Trace Lengths" ✗
- "Signal Intensity Over Time" ✗

### Figure Layout

For multi-panel figures, increase spacing between panels:

```python
# Double the default panel separation
plt.tight_layout(h_pad=3.0, w_pad=3.0)
```

### Legend

For regular plots (not on images):
```python
# Legend without frame
ax.legend(fontsize=8, frameon=False)
```

For plots on images (e.g., overlays on max projections):
```python
# Legend with white background for visibility on dark images
ax.legend(fontsize=8, frameon=True, facecolor='white',
          framealpha=0.9, edgecolor='none')
```

The white background ensures the legend is readable when plotted over grayscale microscopy images or other dark backgrounds.

## Complete Example Template

```python
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 9

# Color-blind friendly palette (Okabe-Ito)
COLORS = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
    'black': '#000000',
}

# Create figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot data (example)
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y, color=COLORS['blue'], linewidth=1.5, label='Sine wave')

# Configure axes
ax.set_xlabel('Time (s)', fontsize=9)
ax.set_ylabel('Amplitude', fontsize=9)
ax.set_title('Example Plot', fontsize=10, pad=9)
ax.legend(fontsize=8, frameon=False)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Thicken left and bottom spines
ax.spines['left'].set_linewidth(1.1)
ax.spines['bottom'].set_linewidth(1.1)

# Set tick label size
ax.tick_params(labelsize=8)

# Save figure
plt.tight_layout()
plt.savefig('output.png', dpi=300, bbox_inches='tight')
plt.close()
```

## Multi-Panel Figures

For figures with multiple subplots:

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Apply standards to each subplot
for ax in axes.flat:
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Thicken left and bottom spines
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)

    # Set tick label size
    ax.tick_params(labelsize=8)

    # Configure labels and title for each subplot
    ax.set_xlabel('X Label', fontsize=9)
    ax.set_ylabel('Y Label', fontsize=9)
    ax.set_title('Subplot Title', fontsize=10, pad=9)

# Increase spacing between panels
plt.tight_layout(h_pad=3.0, w_pad=3.0)
plt.savefig('multi_panel.png', dpi=300, bbox_inches='tight')
plt.close()
```

## Plot-Specific Guidelines

### Line Plots

```python
ax.plot(x, y, color=COLORS['blue'], linewidth=1.5, label='Data')
```

### Scatter Plots

```python
ax.scatter(x, y, s=30, alpha=0.6, color=COLORS['green'],
           edgecolors=COLORS['black'], linewidths=0.5)
```

### Bar Plots

```python
ax.bar(x_pos, values, yerr=errors, capsize=4,
       color=COLORS['blue'], alpha=0.8,
       edgecolor=COLORS['black'], linewidth=0.8)
```

### Box Plots

```python
bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                boxprops=dict(facecolor=COLORS['sky_blue'], alpha=0.6,
                             edgecolor=COLORS['black']),
                medianprops=dict(color=COLORS['vermillion'], linewidth=1.5),
                meanprops=dict(marker='o', markerfacecolor=COLORS['black'],
                              markeredgecolor=COLORS['black'], markersize=4),
                whiskerprops=dict(color=COLORS['black']),
                capprops=dict(color=COLORS['black']))
```

## Color Selection Guide

When choosing colors for multiple datasets:

1. **2 categories**: blue + orange
2. **3 categories**: blue + orange + green
3. **4 categories**: blue + orange + green + purple
4. **5+ categories**: Use all colors from palette, or use a continuous colormap

For emphasis:
- **Primary data**: blue or green
- **Secondary/comparison**: orange
- **Highlights**: vermillion or yellow
- **Neutral**: black or sky_blue

## ML Pipeline Helper Functions

The `ML/utils.py` module provides helper functions that automatically apply these standards.

### apply_axis_standards()

Applies standard axis formatting to any matplotlib axes:

```python
from utils import apply_axis_standards

fig, ax = plt.subplots()
# ... plot your data ...
apply_axis_standards(ax)  # Removes top/right spines, sets linewidths and tick sizes
```

### plot_losses()

Creates a standardized loss curve plot for training:

```python
from utils import plot_losses

# train_losses and val_losses are lists of loss values per epoch
plot_losses(train_losses, val_losses, save_path="loss_plot.png")
```

Features:
- Scatter plot with thin connecting lines
- Orange for training loss, sky blue for validation loss (Okabe-Ito palette)
- Automatic axis standards applied

### plot_confusion_matrix()

Creates standardized confusion matrices:

```python
from utils import plot_confusion_matrix

# From predictions
plot_confusion_matrix(y_true, y_pred, class_names=names, save_path="cm.png")
```

## Uncertainty Visualization

### Wasserstein Distance Interpretation

In MC dropout uncertainty quantification, the Wasserstein distance measures the separation between probability distributions of the top two predicted classes across multiple forward passes:

- **High Wasserstein distance** = **High certainty**: The model consistently assigns high probability to one class
- **Low Wasserstein distance** = **Low certainty**: Predictions are spread between classes

When plotting traces by certainty quartiles:
- Q1 (lowest Wasserstein) = Least certain predictions
- Q4 (highest Wasserstein) = Most certain predictions

## Output Organization

### Directory Structure

Every plot must be saved in its own directory with a descriptive name. Inside that directory:
- The plot file is named `plot.png` (or `plot.pdf`)
- Source data for each panel is saved as `data_panel_X.csv` where X identifies the panel

Example structure:
```
demo_plot_standards_comparison/
├── plot.png
├── data_panel_A.csv
├── data_panel_B.csv
├── data_panel_C.csv
└── data_panel_D.csv
```

### Implementation

```python
from pathlib import Path
import pandas as pd

# Create output directory with descriptive name
output_dir = Path('descriptive_plot_name')
output_dir.mkdir(parents=True, exist_ok=True)

# Save plot as "plot.png"
plt.savefig(output_dir / 'plot.png', dpi=300, bbox_inches='tight')

# Save source data for each panel
# Panel A example (line plot with two datasets)
df_panel_a = pd.DataFrame({
    'x': x_data,
    'y_dataset1': y1_data,
    'y_dataset2': y2_data,
})
df_panel_a.to_csv(output_dir / 'data_panel_A.csv', index=False)

# Panel B example (scatter plot)
df_panel_b = pd.DataFrame({
    'x': x_data,
    'y': y_data,
})
df_panel_b.to_csv(output_dir / 'data_panel_B.csv', index=False)
```

### Panel Naming Convention

Use clear, consistent identifiers for panels:
- Single panel: `data_panel_A.csv`
- Multi-panel figures: `data_panel_A.csv`, `data_panel_B.csv`, `data_panel_C.csv`, etc.
- Or use descriptive names: `data_panel_timeseries.csv`, `data_panel_histogram.csv`

### Data Format Guidelines

- Always use CSV format for data files
- Include descriptive column headers
- For multiple datasets in one panel, use separate columns
- For grouped data (e.g., box plots), use one column per group
- Pad with NaN if groups have different lengths

Example for box plot data:
```python
df_panel = pd.DataFrame({
    'Group_A': group_a_values,
    'Group_B': group_b_values,
    'Group_C': group_c_values,
})
df_panel.to_csv(output_dir / 'data_panel_D.csv', index=False)
```

### Output Resolution

Always save figures with high resolution:

```python
plt.savefig(output_dir / 'plot.png', dpi=300, bbox_inches='tight')
```

For publication, also save vector formats:

```python
plt.savefig(output_dir / 'plot.pdf', bbox_inches='tight')  # Vector format
plt.savefig(output_dir / 'plot.png', dpi=300, bbox_inches='tight')  # Raster format
```

## Testing Your Plot

A quick checklist:
- [ ] Font is Arial / Helvetica / DejaVu Sans (in priority order)
- [ ] No text smaller than 7pt
- [ ] No bold text (unless intentional)
- [ ] Top and right spines removed
- [ ] Left and bottom spines have linewidth=1.1
- [ ] No grid (unless needed)
- [ ] Colors are from Okabe-Ito palette
- [ ] Title has pad=9
- [ ] Multi-panel figures use h_pad=3.0, w_pad=3.0
- [ ] Saved at 300 dpi
- [ ] Plot saved in descriptive folder as `plot.png`
- [ ] Source data saved as `data_panel_X.csv` for each panel
- [ ] Titles and labels use sentence-style capitalization (not headline-style)
- [ ] For ML plots: consider using helper functions from `ML/utils.py`

## Reference

The helper functions in `ML/utils.py` (`plot_confusion_matrix`, `plot_confusion_matrix_with_std`, `plot_losses`) serve as practical examples of these standards applied to the ML pipeline.
