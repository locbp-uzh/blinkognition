#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Cargo Exchange Analysis — Protein Transfer Kinetics

Two operating modes, selected by ``mode`` in the config file:

mode: analyze  (default)
    Reads pre-existing Picasso HDF5 localization files and quantifies what
    fraction of protein localisations co-localise with each vesicle population.

mode: pipeline
    Standalone end-to-end pipeline.  Reads multi-channel ND2 movies directly,
    runs Picasso MLE localization on each channel, then performs colocalization.
    Requires picasso-env with the nd2 and picasso packages.

Experimental design (Puentener 2026, Figure S5):
  Population A: Atto520 (C3, 515 nm) vesicles + HT-JN275 (C1, 640 nm)
  Population B: Atto425 (C4, 405 nm) vesicles + unlabelled HaloTag

  Slides:
    Slide_01_Mix_1h       → timepoint  1 h
    Slide_02_Mix_onslide  → timepoint  0 h (on-slide control)
    Slide_03_Atto425_HT   → control (Atto425 + unlabelled protein)
    Slide_04_Atto520_JN275→ control (Atto520 + JN275, no mixing)
    Slide_05_Mix_6h       → timepoint  6 h
    Slide_06_Mix_24h      → timepoint 24 h

Data structure (per slide, in the Picasso/ sub-directory):
  Picasso/
  └── Slide_01_Mix_1h/
      └── {movie_folder}/
          ├── {N}_C1_locs.hdf5  (640 nm, protein JN275)
          ├── {N}_C3_locs.hdf5  (515 nm, Atto520 vesicle)
          └── {N}_C4_locs.hdf5  (405 nm, Atto425 vesicle)

Colocalization strategy:
  For each FOV (set of matched C1/C3/C4 HDF5 files):
    1. Load all localisations across all frames for each channel.
    2. For each protein loc (C1), query the nearest vesicle loc in C3 and C4
       using a KD-tree.
    3. Classify:
       PV_A520: protein within max_dist of a C3 (Atto520) vesicle
       PV_A425: protein within max_dist of a C4 (Atto425) vesicle
       P_free:  protein not colocalized with either vesicle type
    4. Aggregate classification counts per slide, then per timepoint.

ND2 channel layout (confirmed from file metadata):
  Index 0: 638/640 nm  → C1 (protein JN275)
  Index 1: 488 nm      → unused
  Index 2: 515 nm      → C3 (Atto520 vesicles)
  Index 3: 405 nm      → C4 (Atto425 vesicles)

Localization parameters (from original Picasso YAML sidecar files):
  box_size: 7, min_net_gradient: 20000, method: mle (sigma), eps: 0.001,
  max_it: 1000, camera: baseline=79, sensitivity=16.0, gain=300

Usage:
    python cargo_exchange.py -c config.yaml           # analyze mode
    python cargo_exchange.py -c config_pipeline.yaml  # pipeline mode
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

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
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         FONTSIZE_LABEL,
    'axes.titlesize':    FONTSIZE_TITLE,
    'axes.labelsize':    FONTSIZE_LABEL,
    'xtick.labelsize':   FONTSIZE_TICK,
    'ytick.labelsize':   FONTSIZE_TICK,
    'legend.fontsize':   FONTSIZE_LEGEND,
    'axes.facecolor':    'white',
    'figure.facecolor':  'white',
    'axes.grid':         False,
    'text.color':        'black',
    'axes.labelcolor':   'black',
    'xtick.color':       'black',
    'ytick.color':       'black',
    'figure.dpi':        150,
    'savefig.dpi':       450,
    'savefig.bbox':      'tight',
    'pdf.fonttype':      42,
    'svg.fonttype':      'none',
})

def apply_axis_standards(ax: plt.Axes) -> None:
    ax.set_facecolor('white')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_color('black')
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICK, length=3, width=0.5,
                   direction='out', colors='black')
    ax.tick_params(axis='both', which='minor', length=2, width=0.5,
                   direction='out', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')


# Canonical timepoint ordering (label → hours for sorting)
SLIDE_TIMEPOINTS: Dict[str, Optional[float]] = {
    'Slide_02_Mix_onslide':   0.0,
    'Slide_01_Mix_1h':        1.0,
    'Slide_05_Mix_6h':        6.0,
    'Slide_06_Mix_24h':       24.0,
    'Slide_03_Atto425_HT':    None,   # control
    'Slide_04_Atto520_JN275': None,   # control
}

# ---------------------------------------------------------------------------
# HDF5 loading
# ---------------------------------------------------------------------------

def load_locs(hdf5_path: Path) -> np.ndarray:
    """
    Load Picasso localisations from an HDF5 file.

    Returns structured numpy array with fields including 'x', 'y', 'frame'.
    Returns empty array if file is missing or unreadable.
    """
    if not hdf5_path.exists():
        logging.debug(f"Missing HDF5: {hdf5_path.name}")
        return np.array([], dtype=[('x', 'f4'), ('y', 'f4'), ('frame', 'u4')])
    try:
        with h5py.File(str(hdf5_path), 'r') as f:
            return f['locs'][:]
    except Exception as e:
        logging.warning(f"Could not read {hdf5_path.name}: {e}")
        return np.array([], dtype=[('x', 'f4'), ('y', 'f4'), ('frame', 'u4')])


def locs_to_xy(locs: np.ndarray) -> np.ndarray:
    """Extract (x, y) positions as float32 array of shape (N, 2)."""
    if len(locs) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack([locs['x'], locs['y']]).astype(np.float32)


# ---------------------------------------------------------------------------
# Colocalization
# ---------------------------------------------------------------------------

def colocalize_protein_to_vesicles(
    protein_xy: np.ndarray,
    vesicle_xy: np.ndarray,
    max_dist: float,
) -> np.ndarray:
    """
    Boolean array: True for each protein loc that has a vesicle loc within
    max_dist pixels.

    Args:
        protein_xy: (N, 2) array of protein positions.
        vesicle_xy: (M, 2) array of vesicle positions.
        max_dist: Distance threshold in pixels.

    Returns:
        Boolean array of length N.
    """
    if len(protein_xy) == 0:
        return np.array([], dtype=bool)
    if len(vesicle_xy) == 0:
        return np.zeros(len(protein_xy), dtype=bool)
    tree = cKDTree(vesicle_xy)
    dists, _ = tree.query(protein_xy, k=1, workers=1)
    return dists <= max_dist


def classify_fov(
    c1_hdf5: Path,
    c3_hdf5: Path,
    c4_hdf5: Path,
    max_dist: float = 3.0,
) -> Dict[str, int]:
    """
    Classify protein localisations for one FOV (one set of three HDF5 files).

    Classification (mutually exclusive):
      PV_A520: protein within max_dist of Atto520 (C3) only
      PV_A425: protein within max_dist of Atto425 (C4) only
      PV_both: protein within max_dist of both vesicle types
      P_free:  protein not colocalized with any vesicle
      PV_A520 + PV_A425 + PV_both + P_free == n_protein
    """
    prot  = locs_to_xy(load_locs(c1_hdf5))
    ves_a = locs_to_xy(load_locs(c3_hdf5))  # Atto520 (C3, 515 nm)
    ves_b = locs_to_xy(load_locs(c4_hdf5))  # Atto425 (C4, 405 nm)

    in_a = colocalize_protein_to_vesicles(prot, ves_a, max_dist)
    in_b = colocalize_protein_to_vesicles(prot, ves_b, max_dist)

    return {
        'PV_A520':    int((in_a & ~in_b).sum()),
        'PV_A425':    int((in_b & ~in_a).sum()),
        'PV_both':    int((in_a &  in_b).sum()),
        'P_free':     int((~in_a & ~in_b).sum()),
        'n_protein':  len(prot),
        'n_ves_A520': len(ves_a),
        'n_ves_A425': len(ves_b),
    }


# ---------------------------------------------------------------------------
# Slide-level aggregation
# ---------------------------------------------------------------------------

def _discover_fov_triplets(
    picasso_slide_dir: Path,
    c_protein: str = 'C1',
    c_vesicle_a: str = 'C3',
    c_vesicle_b: str = 'C4',
) -> List[Tuple[Path, Path, Path]]:
    """
    Find all (C1, C3, C4) HDF5 triplets within a slide's Picasso output dir.

    Structure:
      picasso_slide_dir/
      └── {movie_folder}/
          ├── {N}_C1_locs.hdf5
          ├── {N}_C3_locs.hdf5
          └── {N}_C4_locs.hdf5
    """
    triplets = []
    for movie_dir in sorted(picasso_slide_dir.iterdir()):
        if not movie_dir.is_dir() or movie_dir.name.startswith('.'):
            continue
        c1_files = sorted(movie_dir.glob(f'*_{c_protein}_locs.hdf5'))
        for c1 in c1_files:
            prefix = c1.name.replace(f'_{c_protein}_locs.hdf5', '')
            c3 = movie_dir / f'{prefix}_{c_vesicle_a}_locs.hdf5'
            c4 = movie_dir / f'{prefix}_{c_vesicle_b}_locs.hdf5'
            triplets.append((c1, c3, c4))
    return triplets


def analyze_slide(
    picasso_slide_dir: Path,
    max_dist: float = 3.0,
    c_protein: str = 'C1',
    c_vesicle_a: str = 'C3',
    c_vesicle_b: str = 'C4',
) -> Dict[str, int]:
    """
    Aggregate classification counts across all FOVs in one slide.

    Returns dict with summed counts across all FOVs.
    """
    triplets = _discover_fov_triplets(
        picasso_slide_dir, c_protein, c_vesicle_a, c_vesicle_b,
    )
    logging.info(f"  {picasso_slide_dir.name}: {len(triplets)} FOV triplets")

    totals: Dict[str, int] = {
        'PV_A520': 0, 'PV_A425': 0, 'PV_both': 0, 'P_free': 0,
        'n_protein': 0, 'n_ves_A520': 0, 'n_ves_A425': 0,
        'n_fovs': len(triplets),
    }
    for c1, c3, c4 in triplets:
        counts = classify_fov(c1, c3, c4, max_dist)
        for k in ['PV_A520', 'PV_A425', 'PV_both', 'P_free', 'n_protein', 'n_ves_A520', 'n_ves_A425']:
            totals[k] += counts[k]

    return totals


# ---------------------------------------------------------------------------
# Experiment-level analysis
# ---------------------------------------------------------------------------

def analyze_experiment(
    picasso_base: Path,
    slide_names: Optional[List[str]] = None,
    slide_timepoint_map: Optional[Dict[str, Optional[float]]] = None,
    max_dist: float = 3.0,
    c_protein: str = 'C1',
    c_vesicle_a: str = 'C3',
    c_vesicle_b: str = 'C4',
) -> pd.DataFrame:
    """
    Analyse all slides within a Picasso base directory.

    Args:
        picasso_base: Directory containing Slide_*/ subdirectories.
        slide_names: If given, only process these slide names.
        slide_timepoint_map: Maps slide name → timepoint in hours (None = control).
                             Defaults to SLIDE_TIMEPOINTS.
        max_dist: Colocalization distance threshold (pixels).
        c_protein: HDF5 suffix for protein channel (default: 'C1').
        c_vesicle_a: HDF5 suffix for vesicle channel A (default: 'C3').
        c_vesicle_b: HDF5 suffix for vesicle channel B (default: 'C4').

    Returns:
        DataFrame with one row per slide, columns:
        slide, timepoint_h, is_control, PV_A520, PV_A425, P_free,
        n_protein, pct_in_A520, pct_in_A425, pct_free.
    """
    tp_map = slide_timepoint_map or SLIDE_TIMEPOINTS

    if slide_names is None:
        slide_dirs = sorted([
            d for d in picasso_base.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
    else:
        slide_dirs = [picasso_base / s for s in slide_names]

    rows = []
    for slide_dir in slide_dirs:
        slide_name = slide_dir.name
        logging.info(f"Analysing slide: {slide_name}")
        counts = analyze_slide(slide_dir, max_dist, c_protein, c_vesicle_a, c_vesicle_b)
        n = counts['n_protein']
        tp = tp_map.get(slide_name, float('nan'))
        rows.append({
            'slide':         slide_name,
            'timepoint_h':   tp,
            'is_control':    tp is None,
            'PV_A520':            counts['PV_A520'],
            'PV_A425':            counts['PV_A425'],
            'PV_both':            counts['PV_both'],
            'P_free':             counts['P_free'],
            'n_protein':          n,
            'n_ves_A520':         counts['n_ves_A520'],
            'n_ves_A425':         counts['n_ves_A425'],
            'n_fovs':             counts['n_fovs'],
            'pct_in_A520':        100.0 * counts['PV_A520'] / n if n > 0 else 0.0,
            'pct_in_A425':        100.0 * counts['PV_A425'] / n if n > 0 else 0.0,
            'pct_free':           100.0 * counts['P_free']  / n if n > 0 else 0.0,
            'pct_A425_also_A520': 100.0 * counts['PV_both'] / counts['PV_A425']
                                  if counts['PV_A425'] > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('timepoint_h', na_position='last').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pipeline mode — ND2 → localize → colocalize
# ---------------------------------------------------------------------------

def _localize_channel(
    frame: np.ndarray,
    box_size: int,
    min_net_gradient: float,
    camera_info: dict,
) -> np.ndarray:
    """
    Localize spots in a single 2D frame with Picasso MLE fitting.

    Returns (N, 2) float32 array of (x, y) pixel coordinates.
    """
    from picasso import localize as _loc
    movie = frame[np.newaxis, :, :]  # (1, H, W) required by identify/fit
    # threaded=False avoids a Picasso 0.8.8 bug where np.hstack is called on
    # an empty list when a frame contains no spots above the gradient threshold
    try:
        ids = _loc.identify(movie, min_net_gradient, box_size, threaded=False)
    except ValueError:
        return np.empty((0, 2), dtype=np.float32)
    if len(ids) == 0:
        return np.empty((0, 2), dtype=np.float32)
    locs = _loc.fit(movie, camera_info, ids, box_size, eps=0.001, max_it=1000)
    return np.column_stack([locs['x'], locs['y']]).astype(np.float32)


def _iter_nd2_positions(
    nd2_path: Path,
    ch_c1: int,
    ch_c3: int,
    ch_c4: int,
):
    """
    Yield (c1_frame, c3_frame, c4_frame) for every FOV in an ND2 file.

    Handles both single-position (C, Y, X) and multi-position (P, C, Y, X)
    files transparently by checking whether the 'P' dimension is present.
    """
    import nd2 as _nd2
    with _nd2.ND2File(str(nd2_path)) as f:
        arr = f.asarray()
        has_positions = 'P' in f.sizes
    if has_positions:
        for p in range(arr.shape[0]):
            yield arr[p, ch_c1], arr[p, ch_c3], arr[p, ch_c4]
    else:
        yield arr[ch_c1], arr[ch_c3], arr[ch_c4]


def analyze_slide_pipeline(
    nd2_slide_dir: Path,
    channel_c1: int = 0,
    channel_c3: int = 2,
    channel_c4: int = 3,
    box_size: int = 7,
    min_net_gradient: float = 20000,
    camera_info: Optional[dict] = None,
    max_dist: float = 3.0,
    nd2_pattern: str = 'Liposomes_*.nd2',
) -> Dict[str, int]:
    """
    Localize and colocalize all FOVs from ND2 files in one slide directory.

    Finds all ND2 files matching nd2_pattern, iterates over every position in
    each file, localizes the three relevant channels, and aggregates
    colocalization counts.  Returns the same dict structure as analyze_slide().
    """
    if camera_info is None:
        camera_info = {'Baseline': 79.0, 'Sensitivity': 16.0, 'Gain': 300.0}

    nd2_files = sorted(nd2_slide_dir.glob(nd2_pattern))
    logging.info(f"  {nd2_slide_dir.name}: {len(nd2_files)} ND2 file(s)")

    totals: Dict[str, int] = {
        'PV_A520': 0, 'PV_A425': 0, 'PV_both': 0, 'P_free': 0,
        'n_protein': 0, 'n_ves_A520': 0, 'n_ves_A425': 0, 'n_fovs': 0,
    }

    for nd2_path in nd2_files:
        for c1_frame, c3_frame, c4_frame in _iter_nd2_positions(
            nd2_path, channel_c1, channel_c3, channel_c4
        ):
            prot  = _localize_channel(c1_frame, box_size, min_net_gradient, camera_info)
            ves_a = _localize_channel(c3_frame, box_size, min_net_gradient, camera_info)
            ves_b = _localize_channel(c4_frame, box_size, min_net_gradient, camera_info)

            in_a = colocalize_protein_to_vesicles(prot, ves_a, max_dist)
            in_b = colocalize_protein_to_vesicles(prot, ves_b, max_dist)

            totals['PV_A520']    += int((in_a & ~in_b).sum())
            totals['PV_A425']    += int((in_b & ~in_a).sum())
            totals['PV_both']    += int((in_a &  in_b).sum())
            totals['P_free']     += int((~in_a & ~in_b).sum())
            totals['n_protein']  += len(prot)
            totals['n_ves_A520'] += len(ves_a)
            totals['n_ves_A425'] += len(ves_b)
            totals['n_fovs']     += 1

    return totals


def run_pipeline_mode(cfg: dict) -> None:
    """Run ND2 → localize → colocalize for all replicates in the config."""
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_dist           = float(cfg.get('max_dist', 3.0))
    channel_c1         = int(cfg.get('channel_c1', 0))
    channel_c3         = int(cfg.get('channel_c3', 2))
    channel_c4         = int(cfg.get('channel_c4', 3))
    box_size           = int(cfg.get('box_size', 7))
    min_net_gradient   = float(cfg.get('min_net_gradient', 20000))
    camera_info        = {
        'Baseline':    float(cfg.get('camera_baseline', 79.0)),
        'Sensitivity': float(cfg.get('camera_sensitivity', 16.0)),
        'Gain':        float(cfg.get('camera_gain', 300.0)),
    }

    results_by_replicate: Dict[str, pd.DataFrame] = {}
    for rep_name, rep_cfg in cfg['replicates'].items():
        logging.info(f"=== Replicate: {rep_name} ===")
        nd2_root    = Path(rep_cfg['nd2_root'])
        tp_map_raw  = rep_cfg.get('timepoints', {})
        tp_map      = {k: (None if v is None else float(v))
                       for k, v in tp_map_raw.items()}

        rows = []
        for slide_name, tp in tp_map.items():
            slide_dir = nd2_root / slide_name
            if not slide_dir.exists():
                logging.warning(f"  Slide dir not found: {slide_dir}")
                continue
            logging.info(f"Analysing slide: {slide_name}")
            counts = analyze_slide_pipeline(
                slide_dir,
                channel_c1=channel_c1, channel_c3=channel_c3, channel_c4=channel_c4,
                box_size=box_size, min_net_gradient=min_net_gradient,
                camera_info=camera_info, max_dist=max_dist,
            )
            n = counts['n_protein']
            rows.append({
                'slide':              slide_name,
                'timepoint_h':        tp,
                'is_control':         tp is None,
                'PV_A520':            counts['PV_A520'],
                'PV_A425':            counts['PV_A425'],
                'PV_both':            counts['PV_both'],
                'P_free':             counts['P_free'],
                'n_protein':          n,
                'n_ves_A520':         counts['n_ves_A520'],
                'n_ves_A425':         counts['n_ves_A425'],
                'n_fovs':             counts['n_fovs'],
                'pct_in_A520':        100.0 * counts['PV_A520'] / n if n > 0 else 0.0,
                'pct_in_A425':        100.0 * counts['PV_A425'] / n if n > 0 else 0.0,
                'pct_free':           100.0 * counts['P_free']  / n if n > 0 else 0.0,
                'pct_A425_also_A520': 100.0 * counts['PV_both'] / counts['PV_A425']
                                      if counts['PV_A425'] > 0 else 0.0,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('timepoint_h', na_position='last').reset_index(drop=True)
        df.to_csv(output_dir / f'cargo_exchange_{rep_name}.csv', index=False)
        results_by_replicate[rep_name] = df
        logging.info(f"Saved cargo_exchange_{rep_name}.csv")

    combined = pd.concat(
        [df.assign(replicate=rep) for rep, df in results_by_replicate.items()],
        ignore_index=True,
    )
    combined.to_csv(output_dir / 'cargo_exchange_results.csv', index=False)
    logging.info("Pipeline complete.")


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
    parser = argparse.ArgumentParser(description='Cargo exchange kinetics analysis')
    parser.add_argument('-c', '--config', required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get('log_level', 'INFO'))

    if cfg.get('mode') == 'pipeline':
        run_pipeline_mode(cfg)
        return

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_dist = cfg.get('max_dist', 3.0)

    results_by_replicate: Dict[str, pd.DataFrame] = {}
    for rep_name, rep_cfg in cfg['replicates'].items():
        logging.info(f"=== Replicate: {rep_name} ===")
        picasso_base = Path(rep_cfg['picasso_dir'])
        slide_names  = rep_cfg.get('slide_names', None)
        tp_map_raw   = rep_cfg.get('slide_timepoints', None)
        tp_map = {k: (None if v is None else float(v))
                  for k, v in tp_map_raw.items()} if tp_map_raw else None
        df = analyze_experiment(
            picasso_base,
            slide_names=slide_names,
            slide_timepoint_map=tp_map,
            max_dist=max_dist,
            c_protein=cfg.get('channel_protein', 'C1'),
            c_vesicle_a=cfg.get('channel_vesicle_a', 'C3'),
            c_vesicle_b=cfg.get('channel_vesicle_b', 'C4'),
        )
        df.to_csv(output_dir / f'cargo_exchange_{rep_name}.csv', index=False)
        results_by_replicate[rep_name] = df

    combined = pd.concat(
        [df.assign(replicate=rep) for rep, df in results_by_replicate.items()],
        ignore_index=True,
    )
    combined.to_csv(output_dir / 'cargo_exchange_results.csv', index=False)
    logging.info("Done.")


if __name__ == '__main__':
    main()
