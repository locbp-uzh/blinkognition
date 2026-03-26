#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
DLS Stability Analysis — Vesicle Size Over Time

Reads Dynamic Light Scattering Excel files exported by the Anton Paar
Litesizer software and plots hydrodynamic diameter stability over time
for different labelling conditions and temperatures.

File naming convention:
  SEC column test_{condition}_{time}_{temperature}_dil_1e2.xlsx
  e.g.: SEC column test_VAtto520_1h_25C_dil_1e2.xlsx
        SEC column test_unlabelled_21h_30C_dil_1e2.xlsx
        SEC column test_PB_control.xlsx  (no time/temperature fields)

The Excel files are structured with a header block; the hydrodynamic
diameter (Z-average, intensity-weighted) is on row 6 (0-indexed), column 2.

Usage:
    python dls_analysis.py -c config_dls.yaml
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Plotting standards
# ---------------------------------------------------------------------------
COLORS = {
    'orange':    '#E69F00',
    'sky_blue':  '#56B4E9',
    'green':     '#009E73',
    'blue':      '#0072B2',
    'vermillion':'#D55E00',
    'pink':      '#CC79A7',
    'black':     '#000000',
}
FONTSIZE_LABEL  = 7
FONTSIZE_TICK   = 6
FONTSIZE_TITLE  = 7
FONTSIZE_LEGEND = 6

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         FONTSIZE_LABEL,
    'axes.titlesize':    FONTSIZE_TITLE,
    'axes.labelsize':    FONTSIZE_LABEL,
    'xtick.labelsize':   FONTSIZE_TICK,
    'ytick.labelsize':   FONTSIZE_TICK,
    'legend.fontsize':   FONTSIZE_LEGEND,
    'figure.dpi':        150,
    'savefig.dpi':       450,
    'savefig.bbox':      'tight',
    'pdf.fonttype':      42,
    'svg.fonttype':      'none',
})

def apply_axis_standards(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.tick_params(axis='both', which='major', labelsize=6, length=4, width=0.8, direction='out')
    ax.tick_params(axis='both', which='minor', length=2, width=0.6, direction='out')


# Rows (0-indexed) in the Excel sheet where key values live
_ROW_HYDRODYNAMIC_DIAMETER = 6   # col 2: value in µm
_ROW_PDI                   = 7   # polydispersity index (%)
_ROW_PEAK_INTENSITY        = 14  # col 2: intensity-weighted peak in µm

# Time label → hours mapping for sorting
_TIMEPOINT_HOURS: Dict[str, float] = {
    'pre_ext': -2.0,
    'post_ext': -1.0,
    'post_sec': 0.0,
    '1h': 1.0, '2h': 2.0, '4h': 4.0,
    '6h': 6.0, '21h': 21.0, '24h': 24.0,
}

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(path: Path) -> Optional[Dict[str, object]]:
    """
    Parse condition, timepoint, and temperature from a DLS Excel filename.

    Expected format:
      SEC column test_{condition}_{time}_{temperature}_dil_1e2.xlsx
    or special cases without time/temperature (e.g., PB_control).

    Returns dict with keys: condition, timepoint, timepoint_h, temperature,
    or None if the file should be skipped.
    """
    stem = path.stem  # remove .xlsx
    # Strip the leading prefix
    prefix = 'SEC column test_'
    if not stem.lower().startswith(prefix.lower()):
        return None
    rest = stem[len(prefix):]

    # Try to match: condition_time_temperature_dil
    m = re.match(
        r'^(?P<condition>.+?)_(?P<time>\d+h|pre_ext|post_ext|post_sec)'
        r'_(?P<temp>\d+C)(?:_dil.*)?$',
        rest, re.IGNORECASE,
    )
    if m:
        tp = m.group('time').lower()
        return {
            'condition':   m.group('condition'),
            'timepoint':   tp,
            'timepoint_h': _TIMEPOINT_HOURS.get(tp, float('nan')),
            'temperature': m.group('temp').upper(),
        }

    # Special label-only files (e.g., "PB_control")
    return {
        'condition':   rest,
        'timepoint':   'control',
        'timepoint_h': float('nan'),
        'temperature': 'unknown',
    }


# ---------------------------------------------------------------------------
# Excel reading
# ---------------------------------------------------------------------------

def read_dls_excel(path: Path) -> Optional[Dict[str, float]]:
    """
    Read a single DLS Excel file and extract key metrics.

    Args:
        path: Path to .xlsx file.

    Returns:
        Dict with 'hydrodynamic_diameter_nm', 'pdi', 'peak_intensity_nm',
        or None on failure.
    """
    try:
        df = pd.read_excel(str(path), sheet_name=0, header=None, engine='openpyxl')
    except Exception as e:
        logging.warning(f"Could not read {path.name}: {e}")
        return None

    try:
        hd_um  = float(df.iloc[_ROW_HYDRODYNAMIC_DIAMETER, 2])
        pdi    = float(df.iloc[_ROW_PDI, 2])
        pk_um  = float(df.iloc[_ROW_PEAK_INTENSITY, 2])
    except (IndexError, ValueError, TypeError) as e:
        logging.warning(f"Could not parse metrics from {path.name}: {e}")
        return None

    return {
        'hydrodynamic_diameter_nm': hd_um * 1000.0,
        'pdi': pdi,
        'peak_intensity_nm': pk_um * 1000.0,
    }


# ---------------------------------------------------------------------------
# Sample-level loading
# ---------------------------------------------------------------------------

def load_sample(
    sample_dir: Path,
    pattern: str = '*.xlsx',
) -> pd.DataFrame:
    """
    Load all DLS Excel files from a sample directory.

    Args:
        sample_dir: Directory containing .xlsx files.
        pattern: Glob pattern.

    Returns:
        DataFrame with one row per file, columns:
        condition, timepoint, timepoint_h, temperature,
        hydrodynamic_diameter_nm, pdi, peak_intensity_nm, file.
    """
    rows = []
    for f in sorted(sample_dir.rglob(pattern)):
        if f.name.startswith('._') or f.name.startswith('~$'):
            continue
        meta = parse_filename(f)
        if meta is None:
            logging.debug(f"Skipping {f.name} (no recognisable pattern)")
            continue
        metrics = read_dls_excel(f)
        if metrics is None:
            continue
        rows.append({**meta, **metrics, 'file': f.name})
        logging.debug(f"  {f.name}: d={metrics['hydrodynamic_diameter_nm']:.1f} nm")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('timepoint_h', na_position='first').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_stability(
    dfs_by_condition: Dict[str, pd.DataFrame],
    output_dir: Path,
    metric: str = 'hydrodynamic_diameter_nm',
    palette: Optional[List[str]] = None,
    temperatures: Optional[List[str]] = None,
) -> None:
    """
    Plot hydrodynamic diameter stability for all conditions and temperatures.

    Creates one figure per temperature showing all conditions on the same axes,
    plus a combined figure. Saves CSV with summary data.

    Args:
        dfs_by_condition: Dict mapping condition label (e.g. 'unlabelled',
                          'VAtto520') to DataFrame from load_sample().
        output_dir: Directory for output files.
        metric: Column to plot (default: hydrodynamic_diameter_nm).
        palette: Optional list of hex colours.
        temperatures: If given, only plot these temperatures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if palette is None:
        palette = [COLORS['blue'], COLORS['orange'], COLORS['green'],
                   COLORS['vermillion'], COLORS['pink']]

    # Collect all temperatures present
    all_temps = set()
    for df in dfs_by_condition.values():
        if not df.empty:
            all_temps.update(df['temperature'].unique())
    if temperatures:
        all_temps = all_temps & set(temperatures)
    all_temps = sorted(all_temps)

    # Save combined CSV
    combined = pd.concat(
        [df.assign(condition_label=cond) for cond, df in dfs_by_condition.items()],
        ignore_index=True,
    )
    combined.to_csv(output_dir / 'dls_stability_data.csv', index=False)

    ylabel = 'Hydrodynamic diameter (nm)' if metric == 'hydrodynamic_diameter_nm' else metric

    # One plot per temperature
    for temp in all_temps:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        for idx, (cond, df) in enumerate(dfs_by_condition.items()):
            sub = df[df['temperature'] == temp].copy()
            if sub.empty:
                continue
            # Mean and std per timepoint
            agg = sub.groupby('timepoint_h')[metric].agg(['mean', 'std']).reset_index()
            agg = agg.sort_values('timepoint_h')
            color = palette[idx % len(palette)]
            ax.errorbar(
                agg['timepoint_h'], agg['mean'],
                yerr=agg['std'].fillna(0),
                fmt='o-', color=color, lw=0.8, ms=3,
                capsize=2, elinewidth=0.8,
                label=cond,
            )

        ax.set_xlabel('Time (h)', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
        ax.set_title(f'{temp}', fontsize=FONTSIZE_TITLE, pad=9)
        apply_axis_standards(ax)
        ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False)
        fig.tight_layout()
        fname = f'dls_stability_{temp}.pdf'
        fig.savefig(output_dir / fname, dpi=450, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved {fname} → {output_dir}")

    # Combined plot (all temperatures, separate panels)
    if len(all_temps) > 1:
        fig, axes = plt.subplots(1, len(all_temps), figsize=(3.0 * len(all_temps), 2.5),
                                  sharey=True)
        if len(all_temps) == 1:
            axes = [axes]
        for ax, temp in zip(axes, all_temps):
            for idx, (cond, df) in enumerate(dfs_by_condition.items()):
                sub = df[df['temperature'] == temp]
                if sub.empty:
                    continue
                agg = sub.groupby('timepoint_h')[metric].agg(['mean', 'std']).reset_index()
                agg = agg.sort_values('timepoint_h')
                color = palette[idx % len(palette)]
                ax.errorbar(agg['timepoint_h'], agg['mean'], yerr=agg['std'].fillna(0),
                            fmt='o-', color=color, lw=0.8, ms=3,
                            capsize=2, elinewidth=0.8, label=cond)
            ax.set_title(temp, fontsize=FONTSIZE_TITLE, pad=9)
            ax.set_xlabel('Time (h)', fontsize=FONTSIZE_LABEL)
            apply_axis_standards(ax)
        axes[0].set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
        axes[0].legend(fontsize=FONTSIZE_LEGEND, frameon=False)
        fig.tight_layout(h_pad=3.0, w_pad=3.0)
        fig.savefig(output_dir / 'dls_stability_combined.pdf', dpi=450, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved dls_stability_combined.pdf → {output_dir}")


# ---------------------------------------------------------------------------
# Config and CLI
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str = 'INFO') -> None:
    logging.basicConfig(
        level=level.upper(),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stdout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='DLS vesicle size stability analysis')
    parser.add_argument('-c', '--config', required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get('log_level', 'INFO'))

    output_dir = Path(cfg['output_dir'])
    temperatures = cfg.get('temperatures', None)

    dfs_by_condition: Dict[str, pd.DataFrame] = {}
    for condition_name, sample_dir in cfg['samples'].items():
        logging.info(f"Loading DLS data: {condition_name} from {sample_dir}")
        df = load_sample(Path(sample_dir))
        logging.info(f"  → {len(df)} measurements")
        df.to_csv(output_dir / f'dls_{condition_name}.csv', index=False)
        dfs_by_condition[condition_name] = df

    plot_stability(
        dfs_by_condition,
        output_dir,
        temperatures=temperatures,
    )
    logging.info("Done.")


if __name__ == '__main__':
    main()
