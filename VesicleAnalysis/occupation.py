#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Vesicle Occupation Analysis — Proteins per Vesicle (Photobleaching)

Standalone end-to-end pipeline (``mode: pipeline``).  Reads all *_traces.pkl
files (per-FOV and combined) from Picasso slide directories, subtracts the
per-trace background (mean of the last bg_frames=50 frames), divides by
boxsize²=49, and runs quickpbsa (pbsa_file) to detect photobleaching steps.
Writes result CSV files consumed by figure_2C_occupation.py and
figure_S4_occupation.py.  Requires picasso-env with quickpbsa installed; no
ND2 movies needed.

Result CSV format (first row: quickpbsa params dict; remaining rows: one per
trace per type descriptor — crop_index, flag, type, 0..N-1 frame columns):
  Key column: ``type``  — row descriptor; fluors_kv = preliminary step count
  Key column: ``flag``  — quality flag (1 = good, -1 = no steps, -3/-7 = QC)
  Key column: ``0``     — for fluors_kv rows: number of detected steps

Usage:
    python occupation.py -c config_pipeline.yaml
"""
from __future__ import annotations

import argparse
import logging
import re
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
    Load a KV step-detection result CSV and return one row per vesicle.

    The first row of these files is a dict representation of KV parameters;
    the actual CSV starts on the second row.  Each vesicle contributes 7 rows
    with different `type` descriptors; the fluorophore count is stored in the
    `fluors_kv` row, column '0' (constant across all frame columns).

    Args:
        path: Path to *_result.csv file.
        good_flag_only: If True, keep only rows where ``flag == 1``.
            ``flag=1`` means KV clearly identified ≥1 photobleaching step.
            All other flag values are excluded; empty vesicles (no protein
            detected) must be added separately with ``add_empty_vesicles()``.

    Returns:
        DataFrame with columns: ``crop_index``, ``occupation``, ``flag``.
        One row per protein-associated vesicle with clearly detected steps.
    """
    if not path.exists():
        logging.warning(f"Result CSV not found: {path}")
        return pd.DataFrame()
    try:
        raw = pd.read_csv(str(path), skiprows=1)
    except Exception as e:
        logging.warning(f"Could not read {path.name}: {e}")
        return pd.DataFrame()

    if 'type' not in raw.columns:
        logging.warning(f"No 'type' column in {path.name}")
        return pd.DataFrame()

    # Extract one row per vesicle from the fluors_kv descriptor rows
    fk = raw[raw['type'] == 'fluors_kv'].copy()
    fk = fk.assign(occupation=fk['0'].astype(int))

    if good_flag_only and 'flag' in fk.columns:
        n_before = len(fk)
        # flag=1        : KV clearly detected ≥1 photobleaching step — keep
        # flag=-1       : KV found 0 steps (protein pre-bleached or no signal) — exclude
        # flag<-1       : other quality rejections — exclude
        # Empty vesicles (no protein detected at all) are added separately via
        # add_empty_vesicles() using total vesicle cluster counts from Picasso.
        fk = fk[fk['flag'] == 1].copy()
        logging.debug(f"  {path.name}: {n_before} vesicles, {len(fk)} with flag=1 (clear steps)")

    cols = ['crop_index', 'occupation', 'flag'] if 'flag' in fk.columns else ['crop_index', 'occupation']
    return fk[cols].reset_index(drop=True)


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
# Empty-vesicle handling
# ---------------------------------------------------------------------------

def count_vesicle_clusters(picasso_slide_dir: Path) -> int:
    """
    Return the total number of vesicle (488 nm) clusters found by Picasso
    for a given slide, as recorded in ``clustering_results.txt``.

    The clustering pipeline appends one entry per FOV each time it runs
    (sometimes 2-3×), so we read only the first ``n_fovs`` entries, where
    ``n_fovs`` is determined by the number of per-FOV 640 nm traces pkl files
    present in the directory.

    Args:
        picasso_slide_dir: Path to a single slide's Picasso output directory,
            e.g. ``Picasso_5000/Slide_02_4uM/``.

    Returns:
        Total vesicle cluster count, or 0 if the file is missing.
    """
    txt_path = picasso_slide_dir / 'clustering_results.txt'
    if not txt_path.exists():
        logging.warning(f"clustering_results.txt not found: {picasso_slide_dir}")
        return 0

    # Number of real per-FOV acquisitions = numbered traces pkls + 1 combined
    n_fovs = len([p for p in sorted(picasso_slide_dir.glob('*640nm*_traces.pkl'))
                  if re.search(r'_\d{4}_traces', p.name)]) + 1

    counts = list(map(int, re.findall(r'(\d+) clusters were found.*vesicle',
                                      txt_path.read_text())))
    n_use = min(n_fovs, len(counts))
    total = sum(counts[:n_use])
    logging.debug(f"  {picasso_slide_dir.name}: {total} vesicle clusters ({n_use} FOVs)")
    return total


def add_empty_vesicles(df: pd.DataFrame, n_total_vesicles: int) -> pd.DataFrame:
    """
    Augment an occupation DataFrame with occupation=0 rows for empty vesicles.

    Empty vesicles are those detected in the 488 nm channel that had no
    protein (640 nm) localisation within the co-localisation distance.
    Their count is ``n_total_vesicles − len(df)``.

    Args:
        df: DataFrame with an ``occupation`` column (one row per vesicle with
            clearly detected photobleaching steps, from ``load_result_csv``).
        n_total_vesicles: Total vesicle cluster count from Picasso
            (from ``count_vesicle_clusters``).

    Returns:
        Augmented DataFrame including empty-vesicle rows (occupation=0).
    """
    n_empty = max(0, n_total_vesicles - len(df))
    if n_empty == 0:
        logging.warning("n_total_vesicles ≤ len(df); no empty vesicles added.")
        return df
    empty = pd.DataFrame({'occupation': np.zeros(n_empty, dtype=int)})
    result = pd.concat([df[['occupation']], empty], ignore_index=True)
    logging.info(f"  Added {n_empty} empty vesicles → total {len(result)}")
    return result


# ---------------------------------------------------------------------------
# Pipeline mode — standalone trace extraction and KV step detection
# ---------------------------------------------------------------------------

def _read_pkl_compat(path: Path) -> pd.DataFrame:
    """
    Read a pandas 1.x DataFrame pickle file under pandas 2.x.

    Pandas 2.x changed BlockPlacement to require an explicit object instead
    of a raw slice.  This shim patches new_block around the unpickling call.
    """
    import pickle
    import pandas.core.internals.blocks as _blocks
    import pandas._libs.internals as _int

    _orig = _blocks.new_block

    def _compat(*args, **kwargs):
        if len(args) >= 2 and isinstance(args[1], slice):
            args = (args[0], _int.BlockPlacement(args[1])) + args[2:]
        elif 'placement' in kwargs and isinstance(kwargs['placement'], slice):
            kwargs['placement'] = _int.BlockPlacement(kwargs['placement'])
        return _orig(*args, **kwargs)

    _blocks.new_block = _compat
    try:
        with open(path, 'rb') as f:
            df = pickle.load(f)
    finally:
        _blocks.new_block = _orig
    return df


def extract_traces_for_slide(
    picasso_slide_dir: Path,
    channel_640: str,
) -> np.ndarray:
    """
    Collect all non-OUT traces from *_traces.pkl files for a slide.

    Reads all pkl files whose name contains ``channel_640`` and ends with
    ``_traces.pkl``, including both per-FOV (4-digit index) files and any
    combined per-slide pkl.  Traces labelled OUT are excluded; all others
    (IN and any other label) are kept — matching the original
    make_ID_and_combine behaviour of ``location_vesicle != 'OUT'``.

    Returns a 2-D float array of shape (n_traces, movie_length) with raw
    pixel-sum values (not background-subtracted).
    """
    in_traces: list = []
    pkl_files = sorted(
        p for p in picasso_slide_dir.glob(f'*{channel_640}*_traces.pkl')
        if not p.name.startswith('._')
    )
    if not pkl_files:
        logging.warning(f"No pkl files matching *{channel_640}*_traces.pkl "
                        f"in {picasso_slide_dir}")

    for p in pkl_files:
        try:
            df = _read_pkl_compat(p)
        except Exception as e:
            logging.warning(f"  Could not read {p.name}: {e}")
            continue
        non_out = df[df['location_vesicle'] != 'OUT']
        for _, row in non_out.iterrows():
            in_traces.append(np.asarray(row['trace'], dtype=float))
        logging.debug(f"  {p.name}: {len(non_out)} non-OUT traces")

    if not in_traces:
        logging.warning(f"No non-OUT traces found in {picasso_slide_dir}")
        return np.empty((0, 0))
    return np.array(in_traces)


def process_slide_pipeline(
    picasso_slide_dir: Path,
    output_csv: Path,
    channel_640: str,
    boxsize: int = 7,
    bg_frames: int = 50,
    kv_threshold: float = 75.0,
    max_steps: int = 100,
    percentile_step: int = 95,
    length_laststep: int = 5,
    concentration: str = '',
) -> None:
    """
    Full photobleaching step-detection pipeline for one slide using quickpbsa.

    Replicates the original make_ID_and_combine + Run_bleachstep_anal pipeline:
    1. Reads all *_traces.pkl files matching channel_640 (pandas 1.x compat).
    2. Keeps non-OUT traces (location_vesicle != 'OUT').
    3. Subtracts per-trace background (mean of last bg_frames frames).
    4. Drops traces with any raw pixel-sum of exactly zero.
    5. Divides by boxsize² (per-pixel normalisation).
    6. Writes normalised traces to a temp CSV and runs quickpbsa.pbsa_file.
    7. Copies the result CSV (written by quickpbsa) to output_csv.

    The result CSV format (parameter dict on line 1, column headers on line 2,
    data from line 3) matches what load_result_csv() expects.
    """
    import shutil
    import tempfile

    try:
        import quickpbsa as pbsa
    except ImportError:
        raise RuntimeError(
            "quickpbsa is required for pipeline mode. "
            "Install with: pip install quickpbsa"
        )

    logging.info(f"Pipeline: {picasso_slide_dir.name}  [{concentration}]")

    raw_traces = extract_traces_for_slide(picasso_slide_dir, channel_640)
    if raw_traces.shape[0] == 0:
        logging.warning("  No traces — skipping.")
        return

    N_traces, movie_len = raw_traces.shape
    logging.info(f"  {N_traces} non-OUT traces × {movie_len} frames")

    # Background-subtract: mean of last bg_frames raw frames, per trace
    bg = np.mean(raw_traces[:, -bg_frames:], axis=1, keepdims=True)
    traces_bg = raw_traces - bg

    # Drop traces with any raw pixel-sum of exactly zero (matches make_ID_and_combine)
    has_zero = np.any(raw_traces == 0, axis=1)
    n_zero = int(has_zero.sum())
    if n_zero:
        logging.info(f"  Dropped {n_zero} traces with zero raw pixel-sum values")
    traces_bg = traces_bg[~has_zero]

    # Per-pixel normalisation (divide by boxsize²)
    traces_norm = traces_bg / float(boxsize * boxsize)
    N_norm = traces_norm.shape[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        stem = f'{concentration}_traces_perpixel_transposed' if concentration else 'traces_perpixel_transposed'
        temp_csv = tmpdir_path / f'{stem}.csv'

        # Write CSV: one trace per row, columns '0'..'movie_len-1'
        cols = [str(i) for i in range(movie_len)]
        df_out = pd.DataFrame(traces_norm, columns=cols)
        df_out.to_csv(str(temp_csv), index=True)

        # Run quickpbsa (replicates pbsa_file call from Run_bleachstep_anal.ipynb)
        pbsa.pbsa_file(
            str(temp_csv),
            threshold=kv_threshold,
            maxiter=max_steps,
            outfolder=str(tmpdir_path),
            filter_optional={
                'percentile_step': percentile_step,
                'length_laststep': length_laststep,
            },
            num_cores=2,
        )

        result_file = tmpdir_path / f'{stem}_result.csv'
        if not result_file.exists():
            logging.error(f"  quickpbsa did not produce result file: {result_file}")
            return

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(result_file), str(output_csv))

    # Log summary
    try:
        raw = pd.read_csv(str(output_csv), skiprows=1)
        fk = raw[raw['type'] == 'fluors_kv']
        n_good = int((fk['flag'] == 1).sum())
    except Exception:
        n_good = -1
    logging.info(f"  Saved → {output_csv}  ({N_norm} traces, {n_good} flag=1)")


def run_pipeline_mode(cfg: dict) -> None:
    """Run the full pipeline for all concentrations specified in the config."""
    picasso_dir      = Path(cfg['picasso_dir'])
    output_dir       = Path(cfg.get('pipeline_output_dir',
                                    cfg.get('traces_dir',
                                    cfg.get('output_dir', '.'))))
    conc_slide       = cfg.get('conc_slide', {})
    channel_640      = cfg.get('channel_640', '640nm_TIRF2xLP_50pr')
    boxsize          = int(cfg.get('boxsize', 7))
    bg_frames        = int(cfg.get('bg_frames', 50))
    kv_threshold     = float(cfg.get('kv_threshold', 75.0))
    max_steps        = int(cfg.get('kv_max_steps', 100))
    percentile_step  = int(cfg.get('percentile_step', 95))
    length_laststep  = int(cfg.get('length_laststep', 5))

    output_dir.mkdir(parents=True, exist_ok=True)

    for conc, slide_name in conc_slide.items():
        slide_dir  = picasso_dir / slide_name
        output_csv = output_dir / f'{conc}_result.csv'
        process_slide_pipeline(
            slide_dir, output_csv,
            channel_640=channel_640,
            boxsize=boxsize,
            bg_frames=bg_frames,
            kv_threshold=kv_threshold,
            max_steps=max_steps,
            percentile_step=percentile_step,
            length_laststep=length_laststep,
            concentration=conc,
        )


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

    mode = cfg.get('mode', 'pipeline')

    if mode == 'pipeline':
        run_pipeline_mode(cfg)
        logging.info("Pipeline complete.")
        return

    logging.error(f"Unknown mode: '{mode}'. Only 'pipeline' is supported.")
    sys.exit(1)


if __name__ == '__main__':
    main()
