#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH), Andreas Biri (ETHZ).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Combine per-movie trace pickles by compound.

This script:
- Loads all per-movie trace files
- Groups by compound (extracted from filename)
- Assigns unique IDs to each trace
- Separates traces by ground_truth (IN vs OUT)
- Converts to matrices (rows=frames, cols=traces)
- Applies preprocessing (background removal, z-scoring, min-max)
- Saves compound-level matrices in multiple normalization formats
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import sys
import warnings
import yaml
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import (setup_logging, gmm_background_estimate,
                   apply_background_spatial_buffer, robust_background_filter)

# Suppress pandas deprecation warning triggered by sklearn scalers
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Series.swapaxes.*")


def load_config(config_path: Path) -> dict:
    """Load and validate configuration file."""
    with config_path.open("r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    required = [
        "movie_length",
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


def build_compound_patterns(compounds: List[str]) -> Dict[str, re.Pattern]:
    """
    Build regex patterns to match compounds in filenames.

    Matches compound inside filename only:
    - Bounded by start/underscore on left
    - Bounded by underscore/dash on right
    """
    return {
        c: re.compile(rf'(?:(?<=^)|(?<=_)){re.escape(c)}(?:(?=_)|(?=-))')
        for c in compounds
    }


def _load_trace_file(path: str, label: str, origin_key: str = "origin") -> pd.DataFrame:
    """Load a single per-movie trace pkl and tag it."""
    df = pd.read_pickle(path)
    df[origin_key] = path
    df["label"] = label
    return df


def _gmm_bg_col(arr: np.ndarray) -> float:
    """Wrapper so joblib can pickle gmm_background_estimate."""
    return gmm_background_estimate(arr)


def gmm_background_parallel(df_raw: pd.DataFrame, n_jobs: int) -> pd.Series:
    """Estimate GMM background for every column of df_raw in parallel."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(_gmm_bg_col)(df_raw[col].values) for col in df_raw.columns
    )
    return pd.Series(results, index=df_raw.columns)


def _protein_folder(run_folder: Path, gt_label: str) -> Path:
    """Return the ProteinTraces* root folder for a given GT label."""
    if gt_label == "IN":
        return run_folder / "ProteinTracesIN"
    elif gt_label == "OUT":
        return run_folder / "ProteinTracesOUT"
    else:
        return run_folder / "ProteinTraces"


def trace_dict_rearr(traces_df: pd.DataFrame, movie_length: int) -> pd.DataFrame:
    """
    Convert per-trace list format to matrix format.

    Args:
        traces_df: DataFrame with 'trace' column containing lists and 'uniqueID' column.
        movie_length: Expected trace length.

    Returns:
        DataFrame where each column is a trace, each row is a frame.
        Column names are uniqueIDs for traceability.
    """
    if traces_df.empty:
        return pd.DataFrame()

    def fix_len(t):
        """Pad or truncate trace to movie_length."""
        t = np.asarray(t)
        if t.size >= movie_length:
            return t[:movie_length]
        # Pad with NaN if too short
        out = np.full(movie_length, np.nan, dtype=float)
        out[:t.size] = t
        return out

    # Stack all traces as columns with uniqueID as column names
    arr = np.vstack([fix_len(t) for t in traces_df["trace"].tolist()]).T
    df = pd.DataFrame(arr, columns=traces_df["uniqueID"].tolist())

    # Remove traces with zeros (invalid data marker)
    df.replace(0, np.nan, inplace=True)
    df.dropna(axis="columns", inplace=True)

    return df


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Combine per-movie traces by compound and apply preprocessing"
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
        help="Path to localization run folder (e.g., Results/Extract/HaloD106_SNAPC148_grad640-12000_grad488-4000_001)",
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
    movie_length = int(cfg["movie_length"])
    gmm_proba_threshold = float(cfg.get("gmm_proba_threshold", 0.8))
    has_ground_truth = run_info["input_data"].get("has_ground_truth", True)
    n_workers = cfg.get("n_workers", -1)
    if n_workers is None:
        n_workers = -1

    logging.info(f"Run folder: {run_folder}")
    logging.info(f"Proteins: {', '.join(compounds)}")
    logging.info(f"Has ground truth: {has_ground_truth}")

    # Create output folder structure
    unique_ids_dir = run_folder / "UniqueIDs"
    unique_ids_dir.mkdir(exist_ok=True)
    bgf = run_folder / "BackgroundTraces"
    for sub in ["AllRaw", "AllBG", "AllNorm", "Filtered"]:
        (bgf / sub).mkdir(parents=True, exist_ok=True)
    if has_ground_truth:
        for label in ["IN", "OUT"]:
            for sub in ["AllRaw", "AllBG", "AllNorm", "Filtered"]:
                (run_folder / f"ProteinTraces{label}" / sub).mkdir(parents=True, exist_ok=True)
    else:
        for sub in ["AllRaw", "AllBG", "AllNorm", "Filtered"]:
            (run_folder / "ProteinTraces" / sub).mkdir(parents=True, exist_ok=True)

    # Load trace file list
    trace_list_path = run_folder / "FileLists" / "trace_file_list.pkl"
    if not trace_list_path.exists():
        logging.error(f"FileLists/trace_file_list.pkl not found in {run_folder}")
        sys.exit(1)

    with trace_list_path.open("rb") as f:
        trace_files_list = pickle.load(f)

    logging.info(f"Found {len(trace_files_list)} trace files to combine")

    # Build compound matchers
    patterns = build_compound_patterns(compounds)

    # Group traces by compound and save with unique IDs
    for compound in compounds:
        logging.info(f"Processing compound: {compound}")

        try:
            # Match files by compound (filename only, not full path)
            matched = [
                p for p in trace_files_list
                if patterns[compound].search(os.path.basename(p))
            ]

            if not matched:
                logging.warning(f"No trace files found for {compound}")
                # Create empty outputs
                if has_ground_truth:
                    for gt in ["IN", "OUT"]:
                        pf = _protein_folder(run_folder, gt)
                        pd.DataFrame().to_pickle(pf / "AllRaw"  / f"{compound}_{gt}_raw.pkl")
                        pd.DataFrame().to_pickle(pf / "AllBG"   / f"{compound}_{gt}_bg_rm.pkl")
                        pd.DataFrame().to_pickle(pf / "AllNorm" / f"{compound}_{gt}_zscored.pkl")
                        pd.DataFrame().to_pickle(pf / "AllNorm" / f"{compound}_{gt}_minmax.pkl")
                else:
                    pf = run_folder / "ProteinTraces"
                    pd.DataFrame().to_pickle(pf / "AllRaw"  / f"{compound}_raw.pkl")
                    pd.DataFrame().to_pickle(pf / "AllBG"   / f"{compound}_bg_rm.pkl")
                    pd.DataFrame().to_pickle(pf / "AllNorm" / f"{compound}_zscored.pkl")
                    pd.DataFrame().to_pickle(pf / "AllNorm" / f"{compound}_minmax.pkl")
                continue

            logging.info(f"  Found {len(matched)} files for {compound}")

            # Load and combine all traces for this compound (parallel I/O)
            dfs = Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(_load_trace_file)(f, compound) for f in matched
            )

            merged = pd.concat(dfs, ignore_index=True)

            # Add unique IDs
            merged = merged.reset_index(drop=True).reset_index().rename(
                columns={"index": "uniqueID"}
            )

            if has_ground_truth:
                # Separate by ground_truth and overlap status
                in_mask = (merged.get("ground_truth") == "IN") & (
                    merged.get("overlap") == "single"
                )
                out_mask = (merged.get("ground_truth") == "OUT") & (
                    merged.get("overlap") == "single"
                )

                real_trace = merged.loc[in_mask.fillna(False)].copy()
                out_trace = merged.loc[out_mask.fillna(False)].copy()

                # Save uniqueID files
                merged.to_pickle(unique_ids_dir / f"{compound}_uniqueID_all.pkl")
                real_trace.to_pickle(unique_ids_dir / f"{compound}_uniqueID_IN.pkl")
                out_trace.to_pickle(unique_ids_dir / f"{compound}_uniqueID_OUT.pkl")

                logging.info(
                    f"  IN traces: {len(real_trace)}, OUT traces: {len(out_trace)}"
                )

                # Convert to matrices (rows=frames, cols=traces) - WITH ground truth
                subsets_to_process = [("IN", real_trace), ("OUT", out_trace)]
            else:
                # No ground truth - only keep single (non-overlapping) traces
                single_mask = merged.get("overlap") == "single"
                single_trace = merged.loc[single_mask.fillna(False)].copy()

                # Save uniqueID files
                merged.to_pickle(unique_ids_dir / f"{compound}_uniqueID_all.pkl")
                single_trace.to_pickle(unique_ids_dir / f"{compound}_uniqueID_single.pkl")

                logging.info(f"  Single traces: {len(single_trace)}")

                # Convert to matrices (rows=frames, cols=traces) - WITHOUT ground truth
                subsets_to_process = [("", single_trace)]  # Empty label for no ground truth

            # Process each subset
            for gt_label, df_subset in subsets_to_process:
                # Raw traces
                df_raw = trace_dict_rearr(df_subset, movie_length)

                # Determine file naming and folder
                if gt_label:  # Has ground truth (IN or OUT)
                    label_str = f"_{gt_label}"
                    log_label = f"{compound}_{gt_label}"
                else:  # No ground truth
                    label_str = ""
                    log_label = compound
                pf = _protein_folder(run_folder, gt_label)

                if df_raw.empty:
                    logging.warning(f"  No valid traces for {log_label}")
                    # Save empty outputs
                    pd.DataFrame().to_pickle(pf / "AllRaw"  / f"{compound}{label_str}_raw.pkl")
                    pd.DataFrame().to_pickle(pf / "AllBG"   / f"{compound}{label_str}_bg_rm.pkl")
                    pd.DataFrame().to_pickle(pf / "AllNorm" / f"{compound}{label_str}_zscored.pkl")
                    pd.DataFrame().to_pickle(pf / "AllNorm" / f"{compound}{label_str}_minmax.pkl")
                    continue

                # Save raw traces
                df_raw.to_pickle(pf / "AllRaw" / f"{compound}{label_str}_raw.pkl")

                # Background removal (GMM)
                noise_avg = gmm_background_parallel(df_raw, n_workers)
                df_bg_rm = df_raw - noise_avg
                df_bg_rm.to_pickle(pf / "AllBG" / f"{compound}{label_str}_bg_rm.pkl")

                # Z-scoring
                scaler_z = StandardScaler()
                df_zscored = pd.DataFrame(
                    scaler_z.fit_transform(df_bg_rm),
                    columns=df_bg_rm.columns,
                )
                df_zscored.to_pickle(pf / "AllNorm" / f"{compound}{label_str}_zscored.pkl")

                # Min-max scaling
                scaler_mm = MinMaxScaler()
                df_minmax = pd.DataFrame(
                    scaler_mm.fit_transform(df_bg_rm),
                    columns=df_bg_rm.columns,
                )
                df_minmax.to_pickle(pf / "AllNorm" / f"{compound}{label_str}_minmax.pkl")

                logging.info(
                    f"  {log_label}: {df_raw.shape[1]} traces × {df_raw.shape[0]} frames"
                )

        except Exception as e:
            logging.error(f"Error processing {compound}: {e}")
            logging.warning(f"Creating empty outputs for {compound} and continuing...")
            # Create empty outputs for this compound
            if has_ground_truth:
                for gt in ["IN", "OUT"]:
                    pf = _protein_folder(run_folder, gt)
                    pd.DataFrame().to_pickle(pf / "AllRaw"  / f"{compound}_{gt}_raw.pkl")
                    pd.DataFrame().to_pickle(pf / "AllBG"   / f"{compound}_{gt}_bg_rm.pkl")
                    pd.DataFrame().to_pickle(pf / "AllNorm" / f"{compound}_{gt}_zscored.pkl")
                    pd.DataFrame().to_pickle(pf / "AllNorm" / f"{compound}_{gt}_minmax.pkl")
            else:
                pf = run_folder / "ProteinTraces"
                pd.DataFrame().to_pickle(pf / "AllRaw"  / f"{compound}_raw.pkl")
                pd.DataFrame().to_pickle(pf / "AllBG"   / f"{compound}_bg_rm.pkl")
                pd.DataFrame().to_pickle(pf / "AllNorm" / f"{compound}_zscored.pkl")
                pd.DataFrame().to_pickle(pf / "AllNorm" / f"{compound}_minmax.pkl")
            continue

    # =========================================================================
    # Process background traces
    # =========================================================================
    bg_list_path = run_folder / "FileLists" / "background_file_list.pkl"
    if bg_list_path.exists():
        logging.info("\nProcessing background traces...")

        with bg_list_path.open("rb") as f:
            bg_files_list = pickle.load(f)

        if bg_files_list:
            for compound in compounds:
                try:
                    # Match background files by compound
                    matched_bg = [
                        p for p in bg_files_list
                        if patterns[compound].search(os.path.basename(p))
                    ]

                    if not matched_bg:
                        logging.info(f"  No background files for {compound}")
                        continue

                    logging.info(f"  Found {len(matched_bg)} background files for {compound}")

                    # Load and combine all background traces (parallel I/O)
                    bg_dfs = Parallel(n_jobs=n_workers, prefer="threads")(
                        delayed(_load_trace_file)(f, compound, origin_key="file_origin")
                        for f in matched_bg
                    )

                    bg_merged = pd.concat(bg_dfs, ignore_index=True)

                    # Spatial buffer: remove background ROIs too close to protein ROIs
                    bg_buffer_px = float(cfg.get("background_buffer", 5.0))
                    if bg_buffer_px > 0:
                        bg_merged, _ = apply_background_spatial_buffer(bg_merged, bg_buffer_px)

                    # Add unique IDs
                    bg_merged = bg_merged.reset_index(drop=True).reset_index().rename(
                        columns={"index": "uniqueID"}
                    )

                    # Save uniqueID file
                    bg_merged.to_pickle(unique_ids_dir / f"{compound}_uniqueID_background.pkl")

                    # Convert to matrix format
                    df_bg_raw = trace_dict_rearr(bg_merged, movie_length)

                    if df_bg_raw.empty:
                        logging.warning(f"  No valid background traces for {compound}")
                        continue

                    # Save raw background traces
                    df_bg_raw.to_pickle(bgf / "AllRaw" / f"{compound}_background_raw.pkl")

                    # Background removal (GMM)
                    noise_avg = gmm_background_parallel(df_bg_raw, n_workers)
                    df_bg_bg_rm = df_bg_raw - noise_avg
                    df_bg_bg_rm.to_pickle(bgf / "AllBG" / f"{compound}_background_bg_rm.pkl")

                    # Z-scoring
                    scaler_z = StandardScaler()
                    df_bg_zscored = pd.DataFrame(
                        scaler_z.fit_transform(df_bg_bg_rm),
                        columns=df_bg_bg_rm.columns,
                    )
                    df_bg_zscored.to_pickle(bgf / "AllNorm" / f"{compound}_background_zscored.pkl")

                    # Min-max scaling
                    scaler_mm = MinMaxScaler()
                    df_bg_minmax = pd.DataFrame(
                        scaler_mm.fit_transform(df_bg_bg_rm),
                        columns=df_bg_bg_rm.columns,
                    )
                    df_bg_minmax.to_pickle(bgf / "AllNorm" / f"{compound}_background_minmax.pkl")

                    logging.info(
                        f"  {compound}_background: {df_bg_raw.shape[1]} traces × {df_bg_raw.shape[0]} frames"
                    )

                    # Filter background traces using robust outlier score on raw traces
                    bg_rob_threshold = float(cfg.get("background_rob_threshold", 8.0))
                    if bg_rob_threshold > 0:
                        df_bg_raw_filt, filter_stats = robust_background_filter(
                            df_bg_raw, bg_rob_threshold
                        )

                        if filter_stats["n_filtered"] > 0:
                            # Apply same column mask to all normalization versions
                            kept_cols = df_bg_raw_filt.columns
                            df_bg_bg_rm_filt   = df_bg_bg_rm[kept_cols]
                            df_bg_zscored_filt = df_bg_zscored[kept_cols]
                            df_bg_minmax_filt  = df_bg_minmax[kept_cols]

                            # Save filtered versions
                            df_bg_raw_filt.to_pickle(bgf / "Filtered" / f"{compound}_background_filtered_raw.pkl")
                            df_bg_bg_rm_filt.to_pickle(bgf / "Filtered" / f"{compound}_background_filtered_bg_rm.pkl")
                            df_bg_zscored_filt.to_pickle(bgf / "Filtered" / f"{compound}_background_filtered_zscored.pkl")
                            df_bg_minmax_filt.to_pickle(bgf / "Filtered" / f"{compound}_background_filtered_minmax.pkl")

                            logging.info(
                                f"  {compound}_background_filtered: {filter_stats['n_filtered']} traces "
                                f"({filter_stats['pct_removed']:.1f}% contaminated traces removed)"
                            )

                except Exception as e:
                    logging.error(f"Error processing background traces for {compound}: {e}")
                    continue
        else:
            logging.info("  No background trace files found")

    # Summary
    logging.info("=" * 60)
    logging.info("Trace combining complete")
    logging.info("=" * 60)

    # Report output files
    logging.info("\nOutput files created:")
    for compound in compounds:
        if has_ground_truth:
            for gt in ["IN", "OUT"]:
                raw_file = _protein_folder(run_folder, gt) / "AllRaw" / f"{compound}_{gt}_raw.pkl"
                if raw_file.exists():
                    df_check = pd.read_pickle(raw_file)
                    if not df_check.empty:
                        logging.info(f"  {compound}_{gt}: {df_check.shape[1]} traces")
        else:
            raw_file = run_folder / "ProteinTraces" / "AllRaw" / f"{compound}_raw.pkl"
            if raw_file.exists():
                df_check = pd.read_pickle(raw_file)
                if not df_check.empty:
                    logging.info(f"  {compound}: {df_check.shape[1]} traces")

        # Report background traces
        bg_raw_file = bgf / "AllRaw" / f"{compound}_background_raw.pkl"
        if bg_raw_file.exists():
            df_bg_check = pd.read_pickle(bg_raw_file)
            if not df_bg_check.empty:
                logging.info(f"  {compound}_background: {df_bg_check.shape[1]} traces")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        logging.error(f"Fatal error: {ex}")
        sys.exit(1)
