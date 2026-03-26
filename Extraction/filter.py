#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH), Andreas Biri (ETHZ).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Filter traces for machine learning.

This script:
- Loads z-scored trace matrices
- Rejects pure-noise traces via a GMM-separation SNR gate
- Filters traces by peak characteristics:
  - Number of peaks
  - Peak timing (first/last peak position)
  - Peak spacing (first-second peak distance)
- Saves filtered traces in multiple normalization formats

Note: time-reversal (mirror) augmentation is intentionally NOT done here.
It must be applied at the ML training stage (train.py / crossval.py) so it
is restricted to the training split and cannot leak into validation or test sets.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
import yaml
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils import setup_logging, gmm_classify_frames

# Suppress pandas deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Series.swapaxes.*")


def load_config(config_path: Path) -> dict:
    """Load and validate configuration file."""
    with config_path.open("r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    required = [
        "movie_length",
        "min_peak_number",
        "min_peak_width",
        "first_peak_time",
        "last_peak_time",
        "delta_first_second",
    ]

    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    return cfg


def load_run_info(run_folder: Path) -> dict:
    """Load run info (metadata) from run folder."""
    info_file = run_folder / "run_info.yaml"

    if not info_file.exists():
        raise FileNotFoundError(
            f"Run info not found: {info_file}\n"
            "Make sure you're pointing to a valid localization run folder."
        )

    with info_file.open("r") as f:
        info = yaml.load(f, Loader=yaml.SafeLoader)

    return info


def _filter_particle(particle, bg_rm_vals, params):
    """
    Decide whether a single trace passes the filtering criteria (GMM mode).
    Returns the particle id if it passes, else None.
    Runs as a top-level function so joblib can pickle it.
    """
    MIN_PEAK_WIDTH    = params["MIN_PEAK_WIDTH"]
    MIN_PEAK_NUMBER   = params["MIN_PEAK_NUMBER"]
    FIRST_PEAK_TIME   = params["FIRST_PEAK_TIME"]
    LAST_PEAK_TIME    = params["LAST_PEAK_TIME"]
    DELTA_FIRST_SECOND = params["DELTA_FIRST_SECOND"]
    GMM_PROBA_THRESHOLD = params["GMM_PROBA_THRESHOLD"]
    GMM_MIN_SEPARATION  = params["GMM_MIN_SEPARATION"]
    SNR_MIN_SEPARATION  = params["SNR_MIN_SEPARATION"]

    _, _, signal_mask, sep = gmm_classify_frames(
        bg_rm_vals, GMM_PROBA_THRESHOLD, GMM_MIN_SEPARATION, return_separation=True
    )

    # Explicit SNR gate: reject pure-noise traces before peak detection.
    # sep = (signal_mean - noise_mean) / noise_std from the GMM fit.
    if SNR_MIN_SEPARATION > 0 and sep < SNR_MIN_SEPARATION:
        return None

    runs, in_run = [], False
    for i, s in enumerate(signal_mask):
        if s and not in_run:
            run_start, in_run = i, True
        elif not s and in_run:
            runs.append((run_start, i - 1))
            in_run = False
    if in_run:
        runs.append((run_start, len(signal_mask) - 1))

    runs = [(l, r) for l, r in runs if (r - l + 1) >= MIN_PEAK_WIDTH]
    x_peaks = np.array([l + np.argmax(bg_rm_vals[l:r + 1]) for l, r in runs])
    n_peaks = len(x_peaks)

    if n_peaks <= MIN_PEAK_NUMBER:                           return None
    if x_peaks[0] >= FIRST_PEAK_TIME:                       return None
    if n_peaks < 2 or (x_peaks[1] - x_peaks[0]) >= DELTA_FIRST_SECOND: return None
    if x_peaks[-1] <= LAST_PEAK_TIME:                       return None

    return particle


def filter_and_augment_compound(
    compound: str,
    gt_label: str,
    cfg: dict,
    run_folder: Path,
) -> Tuple[int, int]:
    """
    Filter and optionally augment traces for one compound/label.

    Args:
        compound: Compound name.
        gt_label: Ground truth label ("IN", "OUT", or "" for no ground truth).
        cfg: Configuration dictionary.
        run_folder: Localization run folder.

    Returns:
        Tuple of (before_count, after_count)
    """
    MOVIE_LENGTH = int(cfg["movie_length"])
    MIN_PEAK_NUMBER = int(cfg["min_peak_number"])
    MIN_PEAK_WIDTH = int(cfg["min_peak_width"])
    FIRST_PEAK_TIME = int(cfg["first_peak_time"])
    LAST_PEAK_TIME = int(cfg["last_peak_time"])
    DELTA_FIRST_SECOND = int(cfg["delta_first_second"])
    GMM_PROBA_THRESHOLD = float(cfg.get("gmm_proba_threshold", 0.8))
    GMM_MIN_SEPARATION = float(cfg.get("gmm_min_separation", 3.0))
    SNR_MIN_SEPARATION = float(cfg.get("snr_min_separation", 3.0))
    method_tag = "gmm"

    # Determine file naming
    if gt_label:  # Has ground truth (IN or OUT)
        label_str = f"_{gt_label}"
        log_label = f"{compound}_{gt_label}"
    else:  # No ground truth
        label_str = ""
        log_label = compound

    # Load traces
    paths = {
        "raw":     run_folder / f"{compound}{label_str}_all_raw_traces.pkl",
        "bg_rm":   run_folder / f"{compound}{label_str}_{method_tag}_all_bg_rm_traces.pkl",
        "zscored": run_folder / f"{compound}{label_str}_{method_tag}_all_zscored_traces.pkl",
        "minmax":  run_folder / f"{compound}{label_str}_{method_tag}_all_minmax_traces.pkl",
    }

    # Check if files exist
    if not paths["zscored"].exists():
        logging.warning(f"No z-scored traces found for {log_label}")
        return 0, 0

    traces_raw = pd.read_pickle(paths["raw"])
    traces_bg_rm = pd.read_pickle(paths["bg_rm"])
    traces_zscored = pd.read_pickle(paths["zscored"])
    traces_minmax = pd.read_pickle(paths["minmax"])

    if traces_zscored.empty:
        logging.warning(f"Empty traces for {log_label}")
        # Write empty outputs
        empty = pd.DataFrame()
        suffixes = [
            "filtered_raw_traces",
            "filtered_bg_rm_traces",
            "filtered_zscored_traces",
            "filtered_minmax_traces",
        ]

        for suffix in suffixes:
            empty.to_pickle(run_folder / f"{compound}{label_str}_{method_tag}_{suffix}.pkl")

        return 0, 0

    before_count = traces_zscored.shape[1]

    # Remove traces with zeros (replace 0 with NaN, then drop)
    tz = traces_zscored.copy()
    tz.replace(0, np.nan, inplace=True)
    tz.dropna(axis="columns", inplace=True)

    n_workers = cfg.get("n_workers", -1)
    if n_workers is None:
        n_workers = -1

    # Filter traces by peak characteristics (parallel over particles)
    params = {
        "MIN_PEAK_WIDTH":      MIN_PEAK_WIDTH,
        "MIN_PEAK_NUMBER":     MIN_PEAK_NUMBER,
        "FIRST_PEAK_TIME":     FIRST_PEAK_TIME,
        "LAST_PEAK_TIME":      LAST_PEAK_TIME,
        "DELTA_FIRST_SECOND":  DELTA_FIRST_SECOND,
        "GMM_PROBA_THRESHOLD": GMM_PROBA_THRESHOLD,
        "GMM_MIN_SEPARATION":  GMM_MIN_SEPARATION,
        "SNR_MIN_SEPARATION":  SNR_MIN_SEPARATION,
    }

    raw_results = Parallel(n_jobs=n_workers)(
        delayed(_filter_particle)(
            particle,
            traces_bg_rm[particle].values.astype(np.float64),
            params,
        )
        for particle in tz.columns
    )

    selected_particles = [r for r in raw_results if r is not None]

    # Filter all trace sets
    filtered_raw = (
        traces_raw[selected_particles]
        if selected_particles and not traces_raw.empty
        else pd.DataFrame()
    )
    filtered_bg_rm = (
        traces_bg_rm[selected_particles]
        if selected_particles and not traces_bg_rm.empty
        else pd.DataFrame()
    )
    filtered_zscored = (
        tz[selected_particles] if selected_particles else pd.DataFrame()
    )
    filtered_minmax = (
        traces_minmax[selected_particles]
        if selected_particles and not traces_minmax.empty
        else pd.DataFrame()
    )

    # Save filtered traces
    filtered_raw.to_pickle(
        run_folder / f"{compound}{label_str}_{method_tag}_filtered_raw_traces.pkl"
    )
    filtered_bg_rm.to_pickle(
        run_folder / f"{compound}{label_str}_{method_tag}_filtered_bg_rm_traces.pkl"
    )
    filtered_zscored.to_pickle(
        run_folder / f"{compound}{label_str}_{method_tag}_filtered_zscored_traces.pkl"
    )
    filtered_minmax.to_pickle(
        run_folder / f"{compound}{label_str}_{method_tag}_filtered_minmax_traces.pkl"
    )

    after_count = filtered_raw.shape[1]

    return before_count, after_count


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Filter traces for machine learning"
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_path",
        required=True,
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "-r",
        "--run-folder",
        dest="run_folder",
        required=True,
        type=str,
        help="Path to localization run folder (e.g., Results/Extract/Grx1_K20Ac_grad640-12000_grad488-4000_001)",
    )
    args = parser.parse_args()

    # Setup
    setup_logging("INFO")
    cfg = load_config(Path(args.config_path))

    # Load run info
    run_folder = Path(args.run_folder).expanduser()
    run_info = load_run_info(run_folder)

    # Get proteins from run info
    compounds = list(run_info["input_data"]["proteins"])
    has_ground_truth = run_info["input_data"].get("has_ground_truth", True)

    logging.info(f"Run folder: {run_folder}")
    logging.info(f"Proteins: {', '.join(compounds)}")
    logging.info(f"Has ground truth: {has_ground_truth}")
    logging.info("Filtering traces by peak characteristics...")

    # Process each compound and ground_truth label
    results = {}

    for compound in compounds:
        logging.info(f"Processing compound: {compound}")

        if has_ground_truth:
            # Process IN and OUT separately
            gt_labels = ["IN", "OUT"]
        else:
            # Process all traces together (no label)
            gt_labels = [""]

        for gt_label in gt_labels:
            log_label = f"{compound}_{gt_label}" if gt_label else compound

            try:
                before, after = filter_and_augment_compound(
                    compound, gt_label, cfg, run_folder
                )

                results[(compound, gt_label)] = (before, after)

                if before > 0:
                    pct = (after / before) * 100 if before > 0 else 0
                    logging.info(
                        f"  {log_label}: {before} → {after} traces ({pct:.1f}% passed)"
                    )
                elif before == 0:
                    logging.warning(f"  {log_label}: No traces to filter (skipping)")

            except Exception as e:
                logging.error(f"  Error processing {log_label}: {e}")
                logging.warning(f"  Skipping {log_label} and continuing...")
                results[(compound, gt_label)] = (0, 0)
                continue

    # Summary
    logging.info("=" * 60)
    logging.info("Trace filtering complete")
    logging.info("=" * 60)

    logging.info("\nFiltering summary:")
    for (compound, gt_label), (before, after) in results.items():
        if before > 0:
            pct = (after / before) * 100
            log_label = f"{compound}_{gt_label}" if gt_label else compound
            logging.info(f"  {log_label}: {before} → {after} ({pct:.1f}%)")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        logging.error(f"Fatal error: {ex}")
        sys.exit(1)
