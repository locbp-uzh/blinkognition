#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Vesicle Occupation Analysis — Proteins per Vesicle (Photobleaching)

Reads the Kalafut-Visscher (KV) step-detection result CSV files produced by
the photobleaching analysis pipeline and quantifies how many fluorescently
labelled proteins are found per vesicle as a function of protein loading
concentration.

Result CSV format (produced by make_ID_and_combine_multicolor_gt_photobleaching.py):
  First row: KV parameters dict (skipped)
  Second row onwards: CSV with columns including:
    crop_index, kv_time [s], kv_iter, laststep, sdev_laststep, bg, sdev_bg,
    flag, step2_time [s], sic_final, type, step2_time, 0..N (trace values)

  Key column: `type`  — number of fluorophores detected per vesicle cluster
  Key column: `flag`  — quality flag (0 = good, non-zero = rejected)

Usage:
    python occupation.py -c config_occupation.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

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


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_result_csv(path: Path, good_flag_only: bool = True) -> pd.DataFrame:
    """
    Load a KV step-detection result CSV.

    The first row of these files is a dict representation of KV parameters;
    the actual CSV starts on the second row.

    Args:
        path: Path to *_result.csv file.
        good_flag_only: If True, keep only rows where `flag == 0`.

    Returns:
        DataFrame with at minimum columns: crop_index, type, flag.
    """
    if not path.exists():
        logging.warning(f"Result CSV not found: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(str(path), skiprows=1)
    except Exception as e:
        logging.warning(f"Could not read {path.name}: {e}")
        return pd.DataFrame()

    if 'type' not in df.columns:
        logging.warning(f"No 'type' column in {path.name}")
        return pd.DataFrame()

    if good_flag_only and 'flag' in df.columns:
        n_before = len(df)
        df = df[df['flag'] == 0].copy()
        logging.debug(f"  {path.name}: {n_before} total, {len(df)} passing flag=0")
    return df


# ---------------------------------------------------------------------------
# Per-directory loading
# ---------------------------------------------------------------------------

def discover_result_csvs(traces_dir: Path) -> Dict[str, Path]:
    """
    Find all *_result.csv files in a TRACES directory.

    Returns dict mapping concentration label (e.g. '2uM') to Path.
    The concentration is parsed from the filename prefix before the first '_'.
    """
    result: Dict[str, Path] = {}
    for f in sorted(traces_dir.glob('*_result.csv')):
        if f.name.startswith('._'):
            continue
        # Expect format: {concentration}_{suffix}_result.csv
        conc = f.name.split('_')[0]
        result[conc] = f
        logging.debug(f"Found result CSV: {f.name} → {conc}")
    return result


def load_concentration_series(
    traces_dir: Path,
    concentrations: Optional[List[str]] = None,
    good_flag_only: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load all result CSVs from a TRACES directory.

    Args:
        traces_dir: Directory containing *_result.csv files.
        concentrations: If given, only load these concentration labels.
        good_flag_only: Pass only flag==0 rows.

    Returns:
        Dict mapping concentration label → DataFrame.
    """
    csvs = discover_result_csvs(traces_dir)
    if concentrations:
        csvs = {k: v for k, v in csvs.items() if k in concentrations}

    result: Dict[str, pd.DataFrame] = {}
    for conc, path in csvs.items():
        logging.info(f"Loading: {path.name}")
        df = load_result_csv(path, good_flag_only)
        if not df.empty:
            result[conc] = df
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def occupation_distribution(
    df: pd.DataFrame,
    max_n: int = 10,
) -> pd.Series:
    """
    Compute the distribution of proteins-per-vesicle.

    Args:
        df: DataFrame with a 'type' column.
        max_n: Group all counts ≥ max_n into a single ≥max_n bin.

    Returns:
        Series with index = number of proteins (0, 1, 2, ... ≥max_n)
        and values = fraction of vesicles (normalised to 1).
    """
    counts = df['type'].clip(upper=max_n).value_counts().sort_index()
    # Ensure all bins 0..max_n are present
    full_index = list(range(max_n + 1))
    counts = counts.reindex(full_index, fill_value=0)
    return counts / counts.sum()


def mean_occupation(df: pd.DataFrame) -> float:
    """Mean number of proteins per vesicle."""
    return float(df['type'].mean()) if not df.empty else 0.0


def poisson_expected(mean: float, max_n: int = 10) -> pd.Series:
    """
    Poisson distribution with given mean, for comparison to data.
    """
    from scipy.stats import poisson
    index = list(range(max_n + 1))
    probs = [poisson.pmf(k, mean) for k in index[:-1]]
    probs.append(1.0 - sum(probs))  # ≥max_n bin
    return pd.Series(probs, index=index)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _sort_concentrations(concs: List[str]) -> List[str]:
    """Sort concentration labels numerically (e.g. '2uM' < '4uM' < '8uM')."""
    def _key(c: str):
        import re
        m = re.search(r'(\d+\.?\d*)', c)
        return float(m.group(1)) if m else 0.0
    return sorted(concs, key=_key)


def plot_occupation_histograms(
    data_by_conc: Dict[str, pd.DataFrame],
    output_dir: Path,
    max_n: int = 10,
    show_poisson: bool = True,
    palette: Optional[List[str]] = None,
) -> None:
    """
    Bar charts of proteins-per-vesicle distribution for each concentration.

    Args:
        data_by_conc: Dict mapping concentration label → DataFrame.
        output_dir: Directory for output files.
        max_n: Maximum number of proteins to show as individual bars.
        show_poisson: Overlay Poisson fit.
        palette: Optional list of hex colours, one per concentration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if palette is None:
        palette = [COLORS['blue'], COLORS['sky_blue'], COLORS['green'],
                   COLORS['orange'], COLORS['vermillion']]

    concs = _sort_concentrations(list(data_by_conc.keys()))
    n_concs = len(concs)
    if n_concs == 0:
        logging.warning("No data to plot.")
        return

    fig, axes = plt.subplots(1, n_concs, figsize=(2.0 * n_concs, 2.5),
                              sharey=True)
    if n_concs == 1:
        axes = [axes]

    x_labels = [str(k) if k < max_n else f'≥{max_n}' for k in range(max_n + 1)]
    x = np.arange(max_n + 1)

    # Summary table
    summary_rows = []

    for ax, (conc, color) in zip(axes, zip(concs, palette)):
        df = data_by_conc[conc]
        dist = occupation_distribution(df, max_n)
        mu = mean_occupation(df)

        ax.bar(x, dist.values, color=color, edgecolor='black', linewidth=0.3,
               width=0.7, label='Data')

        if show_poisson:
            poiss = poisson_expected(mu, max_n)
            ax.step(x - 0.35, poiss.values, where='post',
                    color=COLORS['black'], lw=0.8, ls='--', label=f'Poisson (λ={mu:.2f})')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=FONTSIZE_TICK, rotation=45)
        ax.set_title(f'{conc}\nμ={mu:.2f}', fontsize=FONTSIZE_TITLE, pad=9)
        ax.set_xlabel('Proteins/vesicle', fontsize=FONTSIZE_LABEL)
        apply_axis_standards(ax)

        summary_rows.append({
            'concentration': conc,
            'n_vesicles':    len(df),
            'mean_occupation': mu,
            'pct_empty':     float(dist[0]) * 100,
            'pct_1':         float(dist[1]) * 100 if 1 in dist.index else 0.0,
            'pct_2plus':     float(dist[2:].sum()) * 100,
        })

    axes[0].set_ylabel('Fraction of vesicles', fontsize=FONTSIZE_LABEL)
    axes[-1].legend(fontsize=FONTSIZE_LEGEND, frameon=False)

    fig.tight_layout(h_pad=3.0, w_pad=3.0)
    fig.savefig(output_dir / 'occupation_histograms.pdf', dpi=450, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved occupation_histograms.pdf → {output_dir}")

    # Summary CSV
    pd.DataFrame(summary_rows).to_csv(output_dir / 'occupation_summary.csv', index=False)


def plot_mean_occupation(
    data_by_conc: Dict[str, pd.DataFrame],
    output_dir: Path,
    palette: Optional[List[str]] = None,
) -> None:
    """
    Plot mean proteins-per-vesicle vs. loading concentration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    concs = _sort_concentrations(list(data_by_conc.keys()))

    means = [mean_occupation(data_by_conc[c]) for c in concs]

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    x = np.arange(len(concs))
    ax.bar(x, means, color=COLORS['blue'], edgecolor='black', linewidth=0.5, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(concs, fontsize=FONTSIZE_TICK)
    ax.set_xlabel('Loading concentration', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Mean proteins per vesicle', fontsize=FONTSIZE_LABEL)
    apply_axis_standards(ax)
    fig.tight_layout()
    fig.savefig(output_dir / 'occupation_mean.pdf', dpi=450, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved occupation_mean.pdf → {output_dir}")


def plot_occupation_across_replicates(
    replicates: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: Path,
    max_n: int = 10,
) -> None:
    """
    Overlay multiple replicates on the same occupation plot.

    Args:
        replicates: Dict mapping replicate name → {concentration → DataFrame}.
        output_dir: Output directory.
        max_n: Max proteins to show individually.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get sorted concentrations from first replicate
    all_concs = set()
    for rep_data in replicates.values():
        all_concs.update(rep_data.keys())
    concs = _sort_concentrations(list(all_concs))

    palette = [COLORS['blue'], COLORS['sky_blue'], COLORS['orange'],
               COLORS['vermillion'], COLORS['pink']]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for idx, (rep_name, rep_data) in enumerate(replicates.items()):
        means = []
        for conc in concs:
            df = rep_data.get(conc, pd.DataFrame())
            means.append(mean_occupation(df))
        color = palette[idx % len(palette)]
        ax.plot(range(len(concs)), means, 'o-', color=color,
                lw=0.8, ms=3, label=rep_name)

    ax.set_xticks(range(len(concs)))
    ax.set_xticklabels(concs, fontsize=FONTSIZE_TICK)
    ax.set_xlabel('Loading concentration', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Mean proteins per vesicle', fontsize=FONTSIZE_LABEL)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False)
    apply_axis_standards(ax)
    fig.tight_layout()
    fig.savefig(output_dir / 'occupation_replicates.pdf', dpi=450, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved occupation_replicates.pdf → {output_dir}")


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
    parser = argparse.ArgumentParser(description='Vesicle occupation (photobleaching) analysis')
    parser.add_argument('-c', '--config', required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get('log_level', 'INFO'))

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_n = cfg.get('max_proteins', 10)

    replicates: Dict[str, Dict[str, pd.DataFrame]] = {}
    for rep_name, rep_cfg in cfg['replicates'].items():
        logging.info(f"=== Replicate: {rep_name} ===")
        traces_dir = Path(rep_cfg['traces_dir'])
        concentrations = rep_cfg.get('concentrations', None)
        rep_data = load_concentration_series(
            traces_dir,
            concentrations=concentrations,
            good_flag_only=cfg.get('good_flag_only', True),
        )
        replicates[rep_name] = rep_data

        # Per-replicate histograms
        rep_out = output_dir / rep_name
        plot_occupation_histograms(rep_data, rep_out, max_n=max_n)
        plot_mean_occupation(rep_data, rep_out)

    if len(replicates) > 1:
        plot_occupation_across_replicates(replicates, output_dir, max_n=max_n)

    logging.info("Done.")


if __name__ == '__main__':
    main()
