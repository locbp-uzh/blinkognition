#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
NanoFCM FCS Analysis — Vesicle Membrane Mixing

Reads .fcs files from NanoFCM experiments, applies scatter and fluorescence
gating, and quantifies the fraction of double-positive (membrane-mixed)
vesicles over time.

Channel mapping (NanoFCM N30E):
  SS-H   : side scatter (size proxy), used for vesicle-population gate
  FITC-H : Atto520 fluorescence (ex 488 nm)
  PC5-H  : Atto647N fluorescence (ex 638 nm)

Typical experiment layout per sample directory:
  Atto520_only_*.fcs        — single-colour control for FITC gate
  Atto647N_only_*.fcs       — single-colour control for PC5 gate
  Atto520_Atto647N_*.fcs    — double-positive reference
  PBS_blank_*.fcs            — blank (background)
  QC FL SiNPs_*.fcs          — instrument QC beads (skipped)
  2e3_0h_mixeddiluted_*.fcs  — mixed sample, 0 h (diluted before measurement)
  2e3_0h_mixedconc_*.fcs    — mixed sample, 0 h (concentrated)
  2e3_1h_mixed_*.fcs         — mixed sample, 1 h
  2e3_3h_mixed_*.fcs         — mixed sample, 3 h
  2e3_6h_mixed_*.fcs         — mixed sample, 6 h
  2e3_24h_mixed_*.fcs        — mixed sample, 24 h

Usage:
    python fcs_analysis.py -c config_fcs.yaml
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fcsparser
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Plotting standards (mirrors VesicleAnalysis/utils.py)
# ---------------------------------------------------------------------------
COLORS = {
    'orange':    '#E69F00',
    'sky_blue':  '#56B4E9',
    'green':     '#009E73',
    'yellow':    '#F0E442',
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
    'font.size':          FONTSIZE_LABEL,
    'axes.titlesize':     FONTSIZE_TITLE,
    'axes.labelsize':     FONTSIZE_LABEL,
    'xtick.labelsize':    FONTSIZE_TICK,
    'ytick.labelsize':    FONTSIZE_TICK,
    'legend.fontsize':    FONTSIZE_LEGEND,
    'figure.dpi':         150,
    'savefig.dpi':        450,
    'savefig.bbox':       'tight',
    'pdf.fonttype':       42,
    'svg.fonttype':       'none',
})

def apply_axis_standards(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.tick_params(axis='both', which='major', labelsize=6, length=4, width=0.8, direction='out')
    ax.tick_params(axis='both', which='minor', length=2, width=0.6, direction='out')


TIMEPOINT_ORDER = ['0h_dil', '0h_conc', '1h', '3h', '6h', '24h']
TIMEPOINT_LABELS = {
    '0h_dil': '0 h\n(dil)',
    '0h_conc': '0 h\n(conc)',
    '1h': '1 h',
    '3h': '3 h',
    '6h': '6 h',
    '24h': '24 h',
}

# ---------------------------------------------------------------------------
# FCS I/O
# ---------------------------------------------------------------------------

def load_fcs(path: Path) -> Tuple[dict, pd.DataFrame]:
    """Load a single FCS file; return (metadata, events DataFrame)."""
    meta, data = fcsparser.parse(str(path), reformat_meta=True)
    return meta, data


def load_all_fcs(
    directory: Path,
    pattern: str = "*.fcs",
    skip_prefixes: Tuple[str, ...] = ("._", "QC"),
) -> Dict[str, pd.DataFrame]:
    """
    Load all FCS files from a directory.

    Args:
        directory: Folder containing .fcs files.
        pattern: Glob pattern.
        skip_prefixes: Files whose names start with any of these are skipped.

    Returns:
        Dict mapping stem (filename without extension) to events DataFrame.
    """
    files = sorted(directory.glob(pattern))
    result: Dict[str, pd.DataFrame] = {}
    for f in files:
        if any(f.name.startswith(p) for p in skip_prefixes):
            logging.debug(f"Skipping {f.name}")
            continue
        try:
            _, data = load_fcs(f)
            result[f.stem] = data
            logging.debug(f"Loaded {f.name}: {len(data)} events")
        except Exception as e:
            logging.warning(f"Could not read {f.name}: {e}")
    return result


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def gate_scatter(
    df: pd.DataFrame,
    ss_channel: str = 'SS-H',
    min_percentile: float = 5.0,
    max_percentile: float = 99.5,
) -> pd.DataFrame:
    """
    Retain events within a percentile window on the scatter channel.
    Removes debris (low SS) and aggregates/large particles (high SS).

    Args:
        df: Events DataFrame.
        ss_channel: Name of the scatter channel column.
        min_percentile: Lower percentile cutoff.
        max_percentile: Upper percentile cutoff.

    Returns:
        Filtered DataFrame.
    """
    ss = df[ss_channel]
    lo, hi = np.percentile(ss, [min_percentile, max_percentile])
    mask = (ss >= lo) & (ss <= hi)
    return df[mask].copy()


def compute_fluorescence_threshold(
    control_dfs: List[pd.DataFrame],
    channel: str,
    percentile: float = 99.0,
    ss_channel: str = 'SS-H',
    ss_min_pct: float = 5.0,
    ss_max_pct: float = 99.5,
) -> float:
    """
    Compute the fluorescence threshold from single-colour control files.

    The threshold is the `percentile`-th percentile of the specified channel
    across all pooled control events (after scatter gating).

    Args:
        control_dfs: List of DataFrames from the negative-for-this-dye control.
                     E.g., Atto647N_only files when computing the FITC threshold.
        channel: Fluorescence channel to threshold (e.g. 'FITC-H').
        percentile: Percentile of the control distribution to use as cut-off.
        ss_channel: Scatter channel for pre-gating.
        ss_min_pct: Lower scatter percentile gate.
        ss_max_pct: Upper scatter percentile gate.

    Returns:
        Threshold value (float) in channel units.
    """
    pooled = []
    for df in control_dfs:
        gated = gate_scatter(df, ss_channel, ss_min_pct, ss_max_pct)
        pooled.append(gated[channel].values)
    all_values = np.concatenate(pooled)
    threshold = float(np.percentile(all_values, percentile))
    logging.info(f"  {channel} threshold ({percentile}th pct of controls): {threshold:.1f}")
    return threshold


def classify_quadrants(
    df: pd.DataFrame,
    fitc_threshold: float,
    pc5_threshold: float,
    fitc_channel: str = 'FITC-H',
    pc5_channel: str = 'PC5-H',
    ss_channel: str = 'SS-H',
    ss_min_pct: float = 5.0,
    ss_max_pct: float = 99.5,
) -> Dict[str, int]:
    """
    Classify gated events into four quadrants:
      Q1: FITC+ and PC5+  (double-positive → membrane mixing)
      Q2: FITC+ and PC5-  (Atto520-only vesicles)
      Q3: FITC- and PC5+  (Atto647N-only vesicles)
      Q4: FITC- and PC5-  (negative / unlabelled)

    Args:
        df: Events DataFrame.
        fitc_threshold: Threshold for FITC channel.
        pc5_threshold: Threshold for PC5 channel.
        fitc_channel: Column name for FITC-H.
        pc5_channel: Column name for PC5-H.
        ss_channel: Column for scatter gating.
        ss_min_pct: Lower scatter percentile.
        ss_max_pct: Upper scatter percentile.

    Returns:
        Dict with keys Q1, Q2, Q3, Q4 and total_gated.
    """
    gated = gate_scatter(df, ss_channel, ss_min_pct, ss_max_pct)
    fitc_pos = gated[fitc_channel] >= fitc_threshold
    pc5_pos  = gated[pc5_channel]  >= pc5_threshold

    counts = {
        'Q1_double_pos': int((fitc_pos  &  pc5_pos).sum()),   # both +
        'Q2_fitc_only':  int((fitc_pos  & ~pc5_pos).sum()),   # Atto520 only
        'Q3_pc5_only':   int((~fitc_pos &  pc5_pos).sum()),   # Atto647N only
        'Q4_negative':   int((~fitc_pos & ~pc5_pos).sum()),   # neither
        'total_gated':   len(gated),
    }
    return counts


def compute_mixing_pct(counts: Dict[str, int]) -> float:
    """
    Compute membrane-mixing percentage.

    Mixing % = Q1 / (Q1 + Q2 + Q3) * 100

    Excludes Q4 (unlabelled background).
    """
    labelled = counts['Q1_double_pos'] + counts['Q2_fitc_only'] + counts['Q3_pc5_only']
    if labelled == 0:
        return 0.0
    return 100.0 * counts['Q1_double_pos'] / labelled


# ---------------------------------------------------------------------------
# Filename → timepoint mapping
# ---------------------------------------------------------------------------

def parse_timepoint(stem: str) -> Optional[str]:
    """
    Extract a canonical timepoint label from an FCS file stem.

    Returns one of: '0h_dil', '0h_conc', '1h', '3h', '6h', '24h'
    or None if not a mixed-sample file.
    """
    s = stem.lower()
    if '0h' in s or '0_h' in s:
        if 'dil' in s:
            return '0h_dil'
        if 'conc' in s:
            return '0h_conc'
        return '0h_dil'  # fallback
    m = re.search(r'(\d+)\s*h', s)
    if m:
        h = m.group(1)
        if h in ('1', '3', '6', '24'):
            return f'{h}h'
    return None


def is_control_file(stem: str, dye: str) -> bool:
    """Return True if the file stem indicates a single-colour control for `dye`."""
    s = stem.lower()
    dye_l = dye.lower()
    return dye_l in s and 'only' in s


# ---------------------------------------------------------------------------
# Per-sample analysis
# ---------------------------------------------------------------------------

def analyze_sample(
    sample_dirs: List[Path],
    fitc_channel: str = 'FITC-H',
    pc5_channel: str = 'PC5-H',
    ss_channel: str = 'SS-H',
    ss_min_pct: float = 5.0,
    ss_max_pct: float = 99.5,
    control_percentile: float = 99.0,
) -> pd.DataFrame:
    """
    Analyse all FCS files from one biological sample (potentially spanning
    multiple acquisition days / subdirectories).

    Strategy:
    1. Pool all Atto647N_only controls → FITC threshold
       (defines Atto520-negative population; any FITC above this = Atto520+)
    2. Pool all Atto520_only controls → PC5 threshold
       (defines Atto647N-negative population)
    3. For each mixed-sample file, classify events and compute mixing %.

    Args:
        sample_dirs: List of directories containing FCS files for this sample
                     (one per acquisition day, if split across days).
        fitc_channel: Atto520 channel name.
        pc5_channel: Atto647N channel name.
        ss_channel: Scatter channel name.
        ss_min_pct: Lower scatter gate percentile.
        ss_max_pct: Upper scatter gate percentile.
        control_percentile: Percentile of single-colour controls used as gate.

    Returns:
        DataFrame with columns: timepoint, mixing_pct, Q1, Q2, Q3, Q4,
        total_gated, file, sample_dir.
    """
    # Collect all files across days
    all_files: Dict[str, Tuple[pd.DataFrame, Path]] = {}
    for d in sample_dirs:
        loaded = load_all_fcs(d)
        for stem, df in loaded.items():
            all_files[stem] = (df, d)

    # Build control pools
    fitc_controls, pc5_controls = [], []
    for stem, (df, _) in all_files.items():
        if is_control_file(stem, 'atto647n') or is_control_file(stem, 'pc5'):
            fitc_controls.append(df)   # Atto647N-only → sets FITC (Atto520) gate
        if is_control_file(stem, 'atto520') or is_control_file(stem, 'fitc'):
            pc5_controls.append(df)    # Atto520-only → sets PC5 (Atto647N) gate

    if not fitc_controls:
        logging.warning("No Atto647N-only control found; FITC threshold may be inaccurate")
        fitc_controls = list(df for df, _ in all_files.values())
    if not pc5_controls:
        logging.warning("No Atto520-only control found; PC5 threshold may be inaccurate")
        pc5_controls = list(df for df, _ in all_files.values())

    fitc_thresh = compute_fluorescence_threshold(
        [df for df, _ in all_files.items() if is_control_file(_, 'atto647n')]  # type: ignore
        if any(is_control_file(s, 'atto647n') for s in all_files)
        else [df for df, _ in [(df, d) for df, d in fitc_controls]],  # type: ignore
        channel=fitc_channel,
        percentile=control_percentile,
        ss_channel=ss_channel,
        ss_min_pct=ss_min_pct,
        ss_max_pct=ss_max_pct,
    ) if fitc_controls else 0.0

    # Recompute cleanly
    fitc_ctrl_dfs = [df for stem, (df, _) in all_files.items()
                     if is_control_file(stem, 'atto647n') or is_control_file(stem, 'pc5')]
    pc5_ctrl_dfs  = [df for stem, (df, _) in all_files.items()
                     if is_control_file(stem, 'atto520') or is_control_file(stem, 'fitc')]

    if not fitc_ctrl_dfs:
        fitc_ctrl_dfs = [df for df, _ in list(all_files.values())]
    if not pc5_ctrl_dfs:
        pc5_ctrl_dfs = [df for df, _ in list(all_files.values())]

    fitc_thresh = compute_fluorescence_threshold(
        fitc_ctrl_dfs, fitc_channel, control_percentile,
        ss_channel, ss_min_pct, ss_max_pct,
    )
    pc5_thresh = compute_fluorescence_threshold(
        pc5_ctrl_dfs, pc5_channel, control_percentile,
        ss_channel, ss_min_pct, ss_max_pct,
    )

    # Analyse mixed samples
    rows = []
    for stem, (df, src_dir) in all_files.items():
        tp = parse_timepoint(stem)
        if tp is None:
            continue  # not a mixed-sample file
        counts = classify_quadrants(
            df, fitc_thresh, pc5_thresh,
            fitc_channel, pc5_channel,
            ss_channel, ss_min_pct, ss_max_pct,
        )
        rows.append({
            'timepoint':    tp,
            'mixing_pct':   compute_mixing_pct(counts),
            'Q1_double_pos': counts['Q1_double_pos'],
            'Q2_fitc_only':  counts['Q2_fitc_only'],
            'Q3_pc5_only':   counts['Q3_pc5_only'],
            'Q4_negative':   counts['Q4_negative'],
            'total_gated':   counts['total_gated'],
            'fitc_threshold': fitc_thresh,
            'pc5_threshold':  pc5_thresh,
            'file':          stem,
            'directory':     str(src_dir),
        })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        # Sort by canonical timepoint order
        tp_order = {tp: i for i, tp in enumerate(TIMEPOINT_ORDER)}
        df_out['_order'] = df_out['timepoint'].map(tp_order)
        df_out = df_out.sort_values('_order').drop(columns='_order').reset_index(drop=True)
    return df_out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scatter_qc(
    df: pd.DataFrame,
    fitc_threshold: float,
    pc5_threshold: float,
    title: str = '',
    fitc_channel: str = 'FITC-H',
    pc5_channel: str = 'PC5-H',
    ss_channel: str = 'SS-H',
    ss_min_pct: float = 5.0,
    ss_max_pct: float = 99.5,
) -> plt.Figure:
    """
    QC scatter plot: FITC-H vs PC5-H with gate lines, coloured by quadrant.
    """
    gated = gate_scatter(df, ss_channel, ss_min_pct, ss_max_pct)
    fitc = gated[fitc_channel].values
    pc5  = gated[pc5_channel].values

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.scatter(fitc, pc5, s=0.3, alpha=0.4, c=COLORS['sky_blue'], rasterized=True)
    ax.axvline(fitc_threshold, color=COLORS['vermillion'], lw=0.8, ls='--')
    ax.axhline(pc5_threshold,  color=COLORS['orange'],    lw=0.8, ls='--')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(f'{fitc_channel} (Atto520)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(f'{pc5_channel} (Atto647N)', fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=9)
    apply_axis_standards(ax)
    fig.tight_layout()
    return fig


def plot_mixing_kinetics(
    results_by_sample: Dict[str, pd.DataFrame],
    output_dir: Path,
    palette: Optional[List[str]] = None,
) -> None:
    """
    Plot membrane-mixing percentage over time for multiple samples.

    Args:
        results_by_sample: Dict mapping sample name to DataFrame from analyze_sample().
        output_dir: Directory for PDF and CSV outputs.
        palette: Optional list of hex colours, one per sample.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if palette is None:
        palette = [COLORS['blue'], COLORS['sky_blue'], COLORS['orange']]

    # Aggregate per sample: mean mixing_pct per timepoint
    summary_frames = []
    for sample, df in results_by_sample.items():
        agg = df.groupby('timepoint')['mixing_pct'].agg(['mean', 'std']).reset_index()
        agg['sample'] = sample
        summary_frames.append(agg)
    summary = pd.concat(summary_frames, ignore_index=True)

    # Save CSV
    summary.to_csv(output_dir / 'mixing_kinetics_summary.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    for ax, ylim in zip(axes, [None, (0, 20)]):
        for idx, (sample, df_s) in enumerate(results_by_sample.items()):
            color = palette[idx % len(palette)]
            agg = df_s.groupby('timepoint', sort=False)['mixing_pct']
            means = agg.mean()
            # Reorder by canonical order
            ordered_tps = [tp for tp in TIMEPOINT_ORDER if tp in means.index]
            x = range(len(ordered_tps))
            y = [means[tp] for tp in ordered_tps]
            ax.plot(x, y, 'o--', color=color, lw=0.8, ms=3, label=sample)

        ax.set_xticks(range(len(TIMEPOINT_ORDER)))
        ax.set_xticklabels([TIMEPOINT_LABELS[tp] for tp in TIMEPOINT_ORDER],
                           fontsize=FONTSIZE_TICK)
        ax.set_xlabel('Timepoint', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Mixing (%)', fontsize=FONTSIZE_LABEL)
        if ylim:
            ax.set_ylim(*ylim)
        apply_axis_standards(ax)
        ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False)

    axes[0].set_title('Full scale', fontsize=FONTSIZE_TITLE, pad=9)
    axes[1].set_title('Zoomed (0–20%)', fontsize=FONTSIZE_TITLE, pad=9)
    fig.tight_layout(h_pad=3.0, w_pad=3.0)
    fig.savefig(output_dir / 'mixing_kinetics.pdf', dpi=450, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved mixing_kinetics.pdf → {output_dir}")


def plot_bar_chart(
    results_by_sample: Dict[str, pd.DataFrame],
    output_dir: Path,
    palette: Optional[List[str]] = None,
) -> None:
    """Bar chart of mean mixing % per timepoint, averaged across samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if palette is None:
        palette = [COLORS['blue'], COLORS['sky_blue'], COLORS['orange']]

    # Grand mean per timepoint
    all_rows = []
    for sample, df in results_by_sample.items():
        for _, row in df.iterrows():
            all_rows.append({'timepoint': row['timepoint'], 'mixing_pct': row['mixing_pct'],
                             'sample': sample})
    all_df = pd.DataFrame(all_rows)

    grand = all_df.groupby('timepoint')['mixing_pct'].agg(['mean', 'std']).reset_index()
    ordered = [tp for tp in TIMEPOINT_ORDER if tp in grand['timepoint'].values]
    grand = grand.set_index('timepoint').loc[ordered].reset_index()

    fig, ax = plt.subplots(figsize=(3, 2.5))
    x = np.arange(len(ordered))
    ax.bar(x, grand['mean'], yerr=grand['std'], width=0.6,
           color=COLORS['blue'], edgecolor='black', linewidth=0.5,
           error_kw=dict(elinewidth=0.8, capsize=2))
    ax.set_xticks(x)
    ax.set_xticklabels([TIMEPOINT_LABELS[tp] for tp in ordered], fontsize=FONTSIZE_TICK)
    ax.set_ylabel('Mixing (%)', fontsize=FONTSIZE_LABEL)
    apply_axis_standards(ax)
    fig.tight_layout()
    fig.savefig(output_dir / 'mixing_bar.pdf', dpi=450, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved mixing_bar.pdf → {output_dir}")


# ---------------------------------------------------------------------------
# Density scatter figure (Figure 2B)
# ---------------------------------------------------------------------------

def _load_cmap(name: str):
    lut = Path(__file__).parent.parent / "assets" / "colormaps" / f"{name}.txt"
    if lut.exists():
        return mcolors.LinearSegmentedColormap.from_list(name, np.loadtxt(lut))
    return mpl.colormaps.get_cmap("viridis")


def plot_nanofcm_scatter(
    sample_name: str,
    sample_dirs: List[Path],
    output_path: Path,
    fitc_channel: str = 'FITC-H',
    pc5_channel: str = 'PC5-H',
    ss_channel: str = 'SS-H',
    ss_min_pct: float = 5.0,
    ss_max_pct: float = 99.5,
    gridsize: int = 60,
    preferred_substring: Optional[str] = None,
    single_file: bool = False,
) -> Dict[str, dict]:
    """
    Four-panel hexbin density scatter figure for one sample:
    FITC-H (488 nm) vs PC5-H (640 nm) at 0 h (diluted), 1 h, 3 h, 24 h.

    Events are SS-H gated (percentile window applied per file), pooled
    across all replicate files for that timepoint, and displayed as a
    hexbin density map with the lapaz colormap.

    Args:
        preferred_substring: Prefer files whose stem contains this string
            (case-insensitive). If no match exists for a timepoint, falls
            back to all available files for that timepoint.
        single_file: If True, use only the first selected file per timepoint
            (after preferred_substring ordering). All available alternatives
            are recorded in gate_stats for documentation.

    Returns dict mapping timepoint → gate statistics including 'file_used'
    and 'files_available' keys when single_file=True.
    """
    _lapaz = _load_cmap("lapaz")
    CLIP_MIN = 1.0   # floor for log scale (instrument units)

    target_tps = ['0h_dil', '1h', '3h', '24h']
    tp_titles  = {'0h_dil': '0 h', '1h': '1 h', '3h': '3 h', '24h': '24 h'}

    # ── load all files across days ────────────────────────────────────────
    all_files: Dict[str, pd.DataFrame] = {}
    for d in sample_dirs:
        all_files.update(load_all_fcs(d))

    # ── collect & gate events per timepoint ──────────────────────────────
    pooled:     Dict[str, Optional[np.ndarray]] = {}
    gate_stats: Dict[str, dict] = {}

    for tp in target_tps:
        # Gather all candidate files for this timepoint
        all_tp: List[Tuple[str, pd.DataFrame]] = []
        for stem, df in all_files.items():
            if parse_timepoint(stem) != tp:
                continue
            if tp == '0h_dil' and 'dil' not in stem.lower():
                continue
            all_tp.append((stem, df))

        # Prefer files matching the substring; fall back to all if none match
        if preferred_substring:
            preferred = [(s, d) for s, d in all_tp
                         if preferred_substring.lower() in s.lower()]
            use_tp = preferred if preferred else all_tp
        else:
            use_tp = all_tp

        # Optionally restrict to a single file
        if single_file and use_tp:
            use_tp = use_tp[:1]

        fitc_vals, pc5_vals, ss_los, ss_his = [], [], [], []
        files_used: List[str] = []

        for stem, df in use_tp:
            ss = df[ss_channel]
            lo, hi = np.percentile(ss, [ss_min_pct, ss_max_pct])
            gated  = df[(ss >= lo) & (ss <= hi)]
            fitc_vals.append(gated[fitc_channel].values)
            pc5_vals.append(gated[pc5_channel].values)
            ss_los.append(lo); ss_his.append(hi)
            files_used.append(stem)
            logging.info(
                f"  {sample_name}/{stem}: {len(gated)}/{len(df)} events "
                f"| SS gate [{lo:.0f}, {hi:.0f}]"
            )

        if not fitc_vals:
            logging.warning(f"No '{tp}' files found for {sample_name}")
            pooled[tp] = None
            gate_stats[tp] = {}
            continue

        pooled[tp] = np.column_stack([
            np.concatenate(fitc_vals),
            np.concatenate(pc5_vals),
        ])
        gate_stats[tp] = {
            'n_files':         len(files_used),
            'n_events':        len(pooled[tp]),
            'ss_lo':           (min(ss_los), max(ss_los)),
            'ss_hi':           (min(ss_his), max(ss_his)),
            'files_used':      files_used,
            'files_available': [s for s, _ in all_tp],
        }

    # ── plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.5), sharey=True, sharex=True)

    Q_FITC, Q_PC5 = 30.0, 100.0   # quadrant thresholds (3×10¹, 10²)
    hb_list: list = []             # (ax, hb) for panels with data
    global_max = 1

    # First pass: render hexbins, collect global max count
    for ax, tp in zip(axes, target_tps):
        ax.set_title(tp_titles[tp], fontsize=FONTSIZE_TITLE, pad=5)
        ax.set_xlabel('488 nm (a.u.)', fontsize=FONTSIZE_LABEL)
        ax.set_xscale('log')
        ax.set_yscale('log')
        apply_axis_standards(ax)

        if pooled[tp] is None:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=FONTSIZE_TICK, color='grey')
            continue

        x = np.clip(pooled[tp][:, 0], CLIP_MIN, None)
        y = np.clip(pooled[tp][:, 1], CLIP_MIN, None)

        hb = ax.hexbin(
            x, y,
            xscale='log', yscale='log',
            gridsize=gridsize,
            cmap=_lapaz,
            mincnt=1,
            linewidths=0.0,
        )
        global_max = max(global_max, int(hb.get_array().max()))

        # Quadrant lines
        ax.axvline(Q_FITC, color='black', lw=0.525, ls='--', alpha=0.8)
        ax.axhline(Q_PC5,  color='black', lw=0.525, ls='--', alpha=0.8)

        # Upper-right quadrant percentage
        pct = 100.0 * np.sum((x >= Q_FITC) & (y >= Q_PC5)) / len(x)
        ax.text(0.97, 0.97, f"{pct:.1f}%", transform=ax.transAxes,
                ha='right', va='top', fontsize=FONTSIZE_TICK, color='black')

        hb_list.append((ax, hb))

    # Second pass: apply shared color scale and add colorbars
    for ax, hb in hb_list:
        hb.set_clim(vmin=1, vmax=global_max)
        cb = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.03)
        cb.set_label('Events', fontsize=FONTSIZE_LEGEND)
        cb.ax.tick_params(labelsize=FONTSIZE_TICK - 1, length=2, width=0.4)
        cb.outline.set_linewidth(0.4)

    axes[0].set_ylabel('640 nm (a.u.)', fontsize=FONTSIZE_LABEL)
    fig.suptitle(sample_name, fontsize=FONTSIZE_TITLE, y=1.01, fontweight='bold')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved {output_path.name} → {output_path.parent}")

    return gate_stats


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
    parser = argparse.ArgumentParser(description='NanoFCM vesicle mixing analysis')
    parser.add_argument('-c', '--config', required=True, type=Path,
                        help='Path to config_fcs.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get('log_level', 'INFO'))

    output_dir = Path(cfg['output_dir'])
    fitc_ch    = cfg.get('fitc_channel', 'FITC-H')
    pc5_ch     = cfg.get('pc5_channel', 'PC5-H')
    ss_ch      = cfg.get('ss_channel', 'SS-H')
    ss_min     = cfg.get('ss_gate_min_percentile', 5.0)
    ss_max     = cfg.get('ss_gate_max_percentile', 99.5)
    ctrl_pct   = cfg.get('control_percentile', 99.0)

    results_by_sample: Dict[str, pd.DataFrame] = {}
    for sample_name, dirs in cfg['samples'].items():
        logging.info(f"=== Analysing sample: {sample_name} ===")
        sample_dirs = [Path(d) for d in (dirs if isinstance(dirs, list) else [dirs])]
        df = analyze_sample(
            sample_dirs,
            fitc_channel=fitc_ch,
            pc5_channel=pc5_ch,
            ss_channel=ss_ch,
            ss_min_pct=ss_min,
            ss_max_pct=ss_max,
            control_percentile=ctrl_pct,
        )
        logging.info(f"  → {len(df)} timepoints analysed")
        df.to_csv(output_dir / f'{sample_name}_fcs_results.csv', index=False)
        results_by_sample[sample_name] = df

    plot_mixing_kinetics(results_by_sample, output_dir)
    plot_bar_chart(results_by_sample, output_dir)
    logging.info("Done.")


if __name__ == '__main__':
    main()
