#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Parameter Finder - Bayesian optimization of gradient thresholds for Picasso localization.

This is Step 1 of the extraction pipeline and runs automatically via run_pipeline.py

This script:
- Discovers all protein/experiment combinations
- For each combo, selects a random test movie
- Runs Optuna optimization (default: 20 trials, 1000 frames)
- Optimizes gradient thresholds only (boxsize is fixed in config.yaml for consistent intensity integration)
- Saves optimized parameters directly in the data folder
- Parameters are used automatically by localize.py (Step 2)

Standalone usage:
    python paramfinder.py -c config.yaml

    # Optimize specific protein/experiment only
    python paramfinder.py -c config.yaml --protein HaloD106 --experiment Exp1

Integrated usage (recommended):
    python run_pipeline.py -c config.yaml  # Runs all steps including paramfinder
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import warnings
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd

# Set matplotlib to non-interactive backend for thread safety in parallel processing
import matplotlib
matplotlib.use('Agg')

from utils import (
    setup_logging,
    get_localization_method,
    get_picasso_python,
    get_fov_size,
    get_num_workers,
    discover_experiments,
    discover_protein_folders,
    find_nd2_files,
    pair_channel_files,
)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message=".*ND2File file not closed before garbage collection.*")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in cast")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Pandas doesn't allow columns to be created.*")


def load_config(config_path: Path) -> dict:
    """Load base configuration file."""
    with config_path.open("r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg


def define_search_space(trial: optuna.Trial, config: dict) -> dict:
    """
    Define Optuna search space for localization parameters.

    Args:
        trial: Optuna trial object.
        config: Configuration dictionary with search space settings.

    Returns:
        Dictionary of trial parameters.
    """
    # Get search space from config
    search_space = config.get("optimization", {}).get("search_space", {})
    has_ground_truth = config.get("has_ground_truth", True)

    # Note: boxsize is NOT optimized - it's fixed in config.yaml
    # This ensures consistent intensity integration across all traces
    params = {
        "gradient_protein": trial.suggest_int(
            "gradient_protein",
            search_space.get("gradient_protein_min", 10000),
            search_space.get("gradient_protein_max", 80000),
            step=search_space.get("gradient_protein_step", 10000),
        ),
    }

    # Only optimize ground truth gradient if we have ground truth
    if has_ground_truth:
        params["gradient_ground_truth"] = trial.suggest_int(
            "gradient_ground_truth",
            search_space.get("gradient_ground_truth_min", 1000),
            search_space.get("gradient_ground_truth_max", 10000),
            step=search_space.get("gradient_ground_truth_step", 1000),
        )

    return params


def run_localization_on_movie(
    nd2_file: Path,
    gradient: int,
    cfg: dict,
    picasso_python: str,
    method: str,
    temp_dir: Path,
) -> Path:
    """
    Run Picasso localization on a single ND2 file.

    Args:
        nd2_file: Path to ND2 file.
        gradient: Gradient threshold.
        cfg: Configuration with trial parameters.
        picasso_python: Path to Picasso Python.
        method: Localization method ('mle').
        temp_dir: Temporary directory for output.

    Returns:
        Path to localization HDF5 file.
    """
    # Copy ND2 file to temp dir (Picasso creates output next to input)
    temp_nd2 = temp_dir / nd2_file.name
    shutil.copy2(nd2_file, temp_nd2)

    # Clear macOS extended attributes from copied file to prevent deletion issues
    try:
        subprocess.run(
            ["xattr", "-c", str(temp_nd2)],
            check=False,
            capture_output=True,
        )
    except Exception:
        # Not critical if this fails
        pass

    cmd = [
        picasso_python,
        "-m",
        "picasso",
        "localize",
        temp_nd2.name,
        "-b", str(int(cfg["boxsize"])),
        "-g", str(int(gradient)),
        "-d", str(int(cfg["drift"])),
        "-bl", str(int(cfg["baseline"])),
        "-s", str(float(cfg["sensitivity"])),
        "-ga", str(int(cfg["gain"])),
        "-qe", str(float(cfg["quantum_efficiency"])),
        "-a", method,
    ]

    # Set environment for headless execution (HPC compatibility)
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["QT_DEBUG_PLUGINS"] = "0"
    env["MPLBACKEND"] = "Agg"
    # Force Qt to not use OpenGL
    env["QT_XCB_GL_INTEGRATION"] = "none"

    try:
        subprocess.run(
            cmd,
            cwd=str(temp_dir),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Localization failed: {e.stderr}")
        raise

    locs_file = temp_dir / f"{temp_nd2.stem}_locs.hdf5"
    return locs_file


def compute_ground_truth_intensities(
    movie_ground_truth: "np.ndarray",
    locs_file: Path,
    boxsize: int,
    n_frames: int = 10,
) -> "np.ndarray":
    """
    Compute integrated intensities for all detected ground truth localizations.

    Args:
        movie_ground_truth: Ground truth movie array.
        locs_file: Path to ground truth localization HDF5 file.
        boxsize: ROI box size.
        n_frames: Number of frames to integrate over (default: 10).

    Returns:
        Array of integrated intensities for each localization.
    """
    from picasso import io as pio, lib

    # Load raw localizations (not clustered)
    locs, info = pio.load_locs(str(locs_file))
    locs = lib.ensure_sanity(locs, info)

    if len(locs) == 0:
        return np.array([])

    integrated_intensities = []
    n_frames = min(n_frames, movie_ground_truth.shape[0])
    shift = boxsize / 2

    for loc in locs:
        # Get box coordinates (top-left corner)
        x = int(round(loc["x"] - shift))
        y = int(round(loc["y"] - shift))

        # Bounds check
        if x < 0 or y < 0:
            continue
        if x + boxsize > movie_ground_truth.shape[2] or y + boxsize > movie_ground_truth.shape[1]:
            continue

        # Extract ROI across first n_frames and integrate
        roi = movie_ground_truth[:n_frames, y:(y + boxsize), x:(x + boxsize)]
        integrated_intensity = np.sum(roi)
        integrated_intensities.append(integrated_intensity)

    return np.array(integrated_intensities)


def extract_traces_from_localizations(
    protein_nd2: Path,
    ground_truth_nd2: Path | None,
    locs_protein: Path,
    locs_ground_truth: Path | None,
    cfg: dict,
    temp_dir: Path,
) -> Tuple[pd.DataFrame, dict]:
    """
    Extract traces from localization files (simplified version of extract.py logic).

    Args:
        protein_nd2: Path to protein channel ND2 file.
        ground_truth_nd2: Path to ground truth channel ND2 file (None if no ground truth).
        locs_protein: Path to protein channel localizations.
        locs_ground_truth: Path to ground truth channel localizations (None if no ground truth).
        cfg: Configuration dictionary.
        temp_dir: Temporary directory.

    Returns:
        Tuple of (DataFrame with trace information, dict with QC visualization data).
    """
    import nd2
    from scipy import spatial
    from utils import (
        get_and_link_locs,
        get_and_cluster_locs,
        get_overlapping_rois,
        get_traces_new_incl_ov,
        colorize,
    )

    has_ground_truth = cfg.get("has_ground_truth", True)

    # Load protein movie fully first so max_proj covers all frames (Picasso localizes
    # all frames, so the max_proj must too — otherwise boxes for molecules that appear
    # after frame 1000 would fall on dark regions in the QC image).
    movie_protein_full = nd2.imread(str(protein_nd2))
    max_proj_protein = np.max(movie_protein_full, axis=0)
    movie_protein = movie_protein_full[:1000]  # Limit to 1000 frames for trace extraction
    del movie_protein_full

    movie_ground_truth = nd2.imread(str(ground_truth_nd2)) if has_ground_truth and ground_truth_nd2 else None

    # Max projections for visualization
    max_proj_ground_truth = np.max(movie_ground_truth, axis=0) if movie_ground_truth is not None else None

    # Create masks
    binary_mask = np.zeros(movie_protein.shape[1:])
    label_mask = np.zeros(movie_protein.shape[1:])

    # Process protein channel localizations
    x_pix, y_pix, x_pix_t, y_pix_t, n_rois = get_and_link_locs(
        str(locs_protein),
        box_size=int(cfg["boxsize"]),
        max_distance=float(cfg["max_distance"]),
        max_off_time=int(cfg["max_darktime"]),
    )

    if len(x_pix) == 0:
        return pd.DataFrame(), {}

    # Strip the dummy element at index 0 that get_and_link_locs prepends, so that
    # ROI indices are 0-based over real localizations only (no false box at (0,0)).
    x_pix_t = x_pix_t[1:]
    y_pix_t = y_pix_t[1:]
    n_rois = len(x_pix_t)

    # Find overlapping ROIs
    overlapping_rois, binary_mask, label_mask = get_overlapping_rois(
        x_pix_t,
        y_pix_t,
        binary_mask,
        label_mask,
        box_size=int(cfg["boxsize"]),
        overlap_threshold=int(cfg["overlap_threshold"]),
    )

    non_overlapping_rois = list(set(range(n_rois)).difference(overlapping_rois))

    # Extract traces
    df = get_traces_new_incl_ov(
        movie_protein,
        non_overlapping_rois,
        overlapping_rois,
        x_pix_t,
        y_pix_t,
        box_side_length=int(cfg["boxsize"]),
    )

    # Process ground truth localizations (only if has_ground_truth)
    ground_truth_positions = np.empty((0, 2))
    ground_truth_intensities = np.array([])

    if has_ground_truth and locs_ground_truth and locs_ground_truth.exists():
        try:
            _, _, x_gt_t, y_gt_t, _, n_ground_truth = get_and_cluster_locs(
                str(locs_ground_truth),
                box_size=int(cfg["boxsize"]),
                max_distance=float(cfg.get("max_distance_ground_truth", 2.5)),
                min_locs=int(cfg.get("min_on_ground_truth", 3)),
            )
            ground_truth_positions = np.stack((x_gt_t[1:], y_gt_t[1:]), axis=1)

            # Compute integrated intensities for all detected ground truth localizations
            ground_truth_intensities = compute_ground_truth_intensities(
                movie_ground_truth,
                locs_ground_truth,
                boxsize=int(cfg["boxsize"]),
                n_frames=10,
            )
        except Exception:
            pass

    # Label traces as IN/OUT (only if has_ground_truth)
    if has_ground_truth:
        if ground_truth_positions.size == 0:
            df["ground_truth"] = ["OUT"] * len(df)
        else:
            tree = spatial.KDTree(ground_truth_positions)
            max_dist = float(cfg.get("max_dist_closest_ground_truth", 2.0))

            gt_labels = []
            for x, y in zip(df["x"], df["y"]):
                dist, _ = tree.query((x, y))
                gt_labels.append("IN" if dist < max_dist else "OUT")

            df["ground_truth"] = gt_labels

    # Prepare QC visualization data
    qc_data = {
        "max_proj_protein": max_proj_protein,
        "max_proj_ground_truth": max_proj_ground_truth,
        "ground_truth_positions": ground_truth_positions,
        "ground_truth_intensities": ground_truth_intensities,
    }

    return df, qc_data


def draw_trial_qc(
    max_proj_protein: np.ndarray,
    max_proj_ground_truth: np.ndarray | None,
    df: pd.DataFrame,
    ground_truth_positions: np.ndarray,
    cfg: dict,
    output_path: Path,
    objective_value: float,
    num_single: int,
    num_IN: int,
    overlap_ratio: float,
) -> None:
    """
    Draw QC composite image for optimization trial.

    Args:
        max_proj_protein: Max projection of protein channel.
        max_proj_ground_truth: Max projection of ground truth channel (None if no ground truth).
        df: DataFrame with trace information.
        ground_truth_positions: Array of ground truth positions.
        cfg: Configuration dictionary.
        output_path: Path to save output image.
        objective_value: Computed objective score.
        num_single: Number of single traces.
        num_IN: Number of IN traces.
        overlap_ratio: Overlap ratio (0-1).
    """
    from matplotlib.figure import Figure
    from matplotlib import patches
    from matplotlib.lines import Line2D
    from utils import colorize

    # Validate input data
    if max_proj_protein is None or max_proj_protein.size == 0:
        logging.warning("Empty protein projection, skipping QC visualization")
        return

    has_ground_truth = cfg.get("has_ground_truth", True)

    # Create RGB composite: protein=red, GT=cyan
    try:
        im_red = colorize(max_proj_protein, (1, 0, 0))
        if max_proj_ground_truth is not None and max_proj_ground_truth.size > 0:
            im_cyan = colorize(max_proj_ground_truth, (0, 1, 1))
            im_comp = np.clip(im_red + im_cyan, 0, 1)
        else:
            im_comp = im_red
    except Exception as e:
        logging.warning(f"Failed to create composite image: {e}")
        return

    # Validate composite image
    if im_comp is None or im_comp.size == 0 or not np.isfinite(im_comp).all():
        logging.warning("Invalid composite image, skipping QC visualization")
        return

    try:
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(im_comp)
    except Exception as e:
        logging.warning(f"Failed to create figure or display image: {e}")
        return

    # Add title with objective score and metrics
    if has_ground_truth:
        title = (
            f"Objective: {objective_value:.1f}  |  "
            f"Single: {num_single}  |  "
            f"IN: {num_IN}  |  "
            f"Overlap: {overlap_ratio:.1%}\n"
            f"g_protein: {cfg.get('gradient_protein', 'N/A')}  |  "
            f"g_gt: {cfg.get('gradient_ground_truth', 'N/A')}  |  "
            f"box: {cfg['boxsize']}"
        )
    else:
        title = (
            f"Objective: {objective_value:.1f}  |  "
            f"Single: {num_single}  |  "
            f"Overlap: {overlap_ratio:.1%}\n"
            f"g_protein: {cfg.get('gradient_protein', 'N/A')}  |  "
            f"box: {cfg['boxsize']}"
        )
    ax.set_title(title, fontsize=7, pad=9)

    # Draw ground truth boxes (cyan, dotted) if applicable
    # x, y stored in df and ground_truth_positions are already top-left corner
    # coordinates (shifted by -box/2 from the Picasso centre in utils.py).
    # The -0.5 offset aligns to matplotlib's pixel-boundary convention: imshow
    # centres pixel i at coordinate i, so its left edge is at i - 0.5.
    box = int(cfg["boxsize"])
    if has_ground_truth and ground_truth_positions.size > 0:
        for x_gt, y_gt in ground_truth_positions:
            rect = patches.Rectangle(
                (x_gt - 0.5, y_gt - 0.5),
                box,
                box,
                linewidth=1,
                edgecolor="cyan",
                facecolor="none",
                linestyle=":",
            )
            ax.add_patch(rect)

    # Draw trace boxes
    if has_ground_truth:
        # With ground truth labels
        for x, y, gt, ov in zip(df["x"], df["y"], df["ground_truth"], df["overlap"]):
            if gt == "IN" and ov == "single":
                edge, style = "red", "-"
            elif gt == "IN" and ov == "overlapping":
                edge, style = "orange", "-"
            elif gt == "OUT" and ov == "single":
                edge, style = "grey", "-"
            else:
                continue  # OUT, overlapping: not plotted

            rect = patches.Rectangle(
                (x - 0.5, y - 0.5),
                box,
                box,
                linewidth=1,
                edgecolor=edge,
                facecolor="none",
                linestyle=style,
            )
            ax.add_patch(rect)
    else:
        # Without ground truth - just show single/overlapping
        for x, y, ov in zip(df["x"], df["y"], df["overlap"]):
            if ov == "single":
                edge, style = "red", "-"
            else:
                edge, style = "orange", "-"

            rect = patches.Rectangle(
                (x - 0.5, y - 0.5),
                box,
                box,
                linewidth=1,
                edgecolor=edge,
                facecolor="none",
                linestyle=style,
            )
            ax.add_patch(rect)

    # Add legend
    if has_ground_truth:
        legend_elements = [
            Line2D([0], [0], color="cyan",    linewidth=2, linestyle=":", label="Ground truth"),
            Line2D([0], [0], color="red", linewidth=2, linestyle="-", label="IN, single"),
            Line2D([0], [0], color="orange",  linewidth=2, linestyle="-", label="IN, overlapping"),
            Line2D([0], [0], color='#808080', linewidth=2, linestyle="-", label="OUT, single"),
        ]
    else:
        legend_elements = [
            Line2D([0], [0], color="red", linewidth=2, linestyle="-", label="Single"),
            Line2D([0], [0], color="orange",  linewidth=2, linestyle="-", label="Overlapping"),
        ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=6, framealpha=0.9)

    h, w = im_comp.shape[:2]

    try:
        ax.grid(False)
        fig.tight_layout()
        # Re-apply after tight_layout: on older matplotlib, tight_layout with
        # aspect='equal' can silently reset the axis limits.
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        fig.savefig(output_path, dpi=450, bbox_inches='tight')
    except Exception as e:
        logging.warning(f"Failed to save QC figure to {output_path}: {e}")


def compute_objective(df: pd.DataFrame, has_ground_truth: bool = True) -> float:
    """
    Compute objective metric from extracted traces.

    With ground truth (composite metric):
    - Base: Number of single (non-overlapping) traces
    - Bonus: Extra weight for IN traces
    - Penalty: Quadratic penalty for high overlap ratios

    Without ground truth (simplified metric):
    - Base: Number of single traces
    - Penalty: Quadratic penalty for high overlap ratios

    Args:
        df: DataFrame with trace information.
        has_ground_truth: Whether analysis includes ground truth labels.

    Returns:
        Objective value (higher is better).
    """
    if df.empty:
        return 0.0

    num_single = len(df[df["overlap"] == "single"])
    num_overlapping = len(df[df["overlap"] == "overlapping"])

    total = num_single + num_overlapping
    if total == 0:
        return 0.0

    overlap_ratio = num_overlapping / total

    if has_ground_truth:
        # Composite metric with IN bonus
        num_IN = len(df[(df["ground_truth"] == "IN") & (df["overlap"] == "single")])
        objective = (
            num_single
            * (1 + num_IN / max(num_single, 1))
            * (1 - overlap_ratio) ** 2
        )
    else:
        # Simplified metric without ground truth
        objective = num_single * (1 - overlap_ratio) ** 2

    return objective


def handle_remove_readonly(func, path, exc_info):
    """
    Error handler for shutil.rmtree to handle permission errors.

    On macOS, ND2 files can have special attributes that prevent deletion.
    This handler tries multiple strategies to remove stubborn files.

    Args:
        func: Function that raised the error (e.g., os.unlink).
        path: Path to the file that couldn't be removed.
        exc_info: Exception information.
    """
    try:
        # Strategy 1: Clear macOS extended attributes (quarantine, etc.)
        subprocess.run(
            ["xattr", "-c", path],
            check=False,
            capture_output=True,
        )

        # Strategy 2: Change file permissions to writable
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)

        # Retry the operation
        func(path)
    except Exception as e:
        # If all else fails, log and continue (don't block optimization)
        logging.debug(f"Could not remove {path}: {e}")


def evaluate_single_movie(
    movie_pair: Tuple[Path, Path | None],
    trial_params: dict,
    base_config: dict,
    picasso_python: str,
    method: str,
) -> Tuple[float, pd.DataFrame, dict]:
    """
    Evaluate a single movie pair with given parameters. Returns (score, df, qc_data).

    Args:
        movie_pair: (protein_nd2, ground_truth_nd2) paths.
        trial_params: Trial parameters (gradient values).
        base_config: Base configuration dict.
        picasso_python: Path to Picasso Python.
        method: Localization method.

    Returns:
        Tuple of (objective_value, DataFrame, qc_data dict).
    """
    cfg = base_config.copy()
    cfg.update(trial_params)
    has_ground_truth = base_config.get("has_ground_truth", True)

    temp_dir = Path(tempfile.mkdtemp(prefix="optuna_trial_"))

    try:
        protein_nd2, ground_truth_nd2 = movie_pair

        locs_protein = run_localization_on_movie(
            protein_nd2,
            trial_params["gradient_protein"],
            cfg,
            picasso_python,
            method,
            temp_dir,
        )

        locs_ground_truth = None
        if has_ground_truth and ground_truth_nd2:
            locs_ground_truth = run_localization_on_movie(
                ground_truth_nd2,
                trial_params["gradient_ground_truth"],
                cfg,
                picasso_python,
                method,
                temp_dir,
            )

        df, qc_data = extract_traces_from_localizations(
            protein_nd2,
            ground_truth_nd2,
            locs_protein,
            locs_ground_truth,
            cfg,
            temp_dir,
        )

        objective_value = compute_objective(df, has_ground_truth)
        return objective_value, df, qc_data

    except Exception as e:
        logging.warning(f"Evaluation failed for {movie_pair[0].name}: {e}")
        return 0.0, pd.DataFrame(), {}

    finally:
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
            except Exception as cleanup_error:
                logging.warning(f"Could not clean up temp directory {temp_dir}: {cleanup_error}")
                logging.info(f"Temp files left at: {temp_dir} (can be manually deleted later)")


def objective_function(
    trial: optuna.Trial,
    movie_pair: Tuple[Path, Path | None],
    base_config: dict,
    picasso_python: str,
    method: str,
    protein_path: Path,
) -> float:
    """
    Optuna objective function for per-protein optimization (legacy mode).

    Args:
        trial: Optuna trial object.
        movie_pair: (protein_nd2, ground_truth_nd2) paths (ground_truth_nd2 is None if no ground truth).
        base_config: Base configuration dict.
        picasso_python: Path to Picasso Python.
        method: Localization method.
        protein_path: Path to protein folder for QC output.

    Returns:
        Objective value (higher is better).
    """
    # Get trial parameters
    trial_params = define_search_space(trial, base_config)

    # Check if these exact parameters have been tried before
    study = trial.study
    for completed_trial in study.trials:
        if completed_trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if completed_trial.number == trial.number:
            continue

        # Compare parameters
        if completed_trial.params == trial_params:
            # Found duplicate - return cached result
            logging.info(
                f"  Trial {trial.number}: Duplicate of trial {completed_trial.number}, "
                f"returning cached objective: {completed_trial.value:.2f}"
            )

            # Copy user attributes
            for key, value in completed_trial.user_attrs.items():
                trial.set_user_attr(key, value)

            return completed_trial.value

    has_ground_truth = base_config.get("has_ground_truth", True)

    # Setup QC output with mode-specific subfolder
    mode_suffix = "gt" if has_ground_truth else "no_gt"
    qc_folder = protein_path / "visualQC" / mode_suffix
    qc_folder.mkdir(parents=True, exist_ok=True)

    # Generate QC filename with trial number and parameters
    if has_ground_truth:
        qc_filename = (
            f"trial_{trial.number:03d}_"
            f"gp-{trial_params['gradient_protein']}_"
            f"ggt-{trial_params['gradient_ground_truth']}_"
            f"box{base_config['boxsize']}.pdf"
        )
    else:
        qc_filename = (
            f"trial_{trial.number:03d}_"
            f"gp-{trial_params['gradient_protein']}_"
            f"box{base_config['boxsize']}.pdf"
        )
    qc_output_path = qc_folder / qc_filename

    # Evaluate movie
    objective_value, df, qc_data = evaluate_single_movie(
        movie_pair, trial_params, base_config, picasso_python, method,
    )

    # Log additional metrics
    if not df.empty:
        num_single = len(df[df["overlap"] == "single"])
        num_overlapping = len(df[df["overlap"] == "overlapping"])
        num_IN = 0
        if has_ground_truth and "ground_truth" in df.columns:
            num_IN = len(df[(df["ground_truth"] == "IN") & (df["overlap"] == "single")])
        total = num_single + num_overlapping
        overlap_ratio = num_overlapping / total if total > 0 else 0

        trial.set_user_attr("num_single", num_single)
        trial.set_user_attr("num_overlapping", num_overlapping)
        trial.set_user_attr("num_IN", num_IN)
        trial.set_user_attr("overlap_ratio", overlap_ratio)

        # Log ground truth intensity statistics (if available)
        if has_ground_truth and len(qc_data.get("ground_truth_intensities", [])) > 0:
            gt_intensities = qc_data["ground_truth_intensities"]
            logging.info(
                f"  Ground truth detections: {len(gt_intensities)} spots, "
                f"intensity range: {np.min(gt_intensities):.0f}-{np.max(gt_intensities):.0f} "
                f"({np.max(gt_intensities) / np.min(gt_intensities):.1f}x variation)"
            )

            gt_stats = {
                "n_detections": int(len(gt_intensities)),
                "intensity_min": float(np.min(gt_intensities)),
                "intensity_max": float(np.max(gt_intensities)),
                "intensity_mean": float(np.mean(gt_intensities)),
                "intensity_median": float(np.median(gt_intensities)),
                "intensity_range_fold": float(np.max(gt_intensities) / np.min(gt_intensities)),
                "intensity_p10": float(np.percentile(gt_intensities, 10)),
                "intensity_p90": float(np.percentile(gt_intensities, 90)),
            }
            trial.set_user_attr("ground_truth_intensity_stats", gt_stats)

        # Draw QC visualizations
        if qc_data:
            try:
                cfg = base_config.copy()
                cfg.update(trial_params)
                draw_trial_qc(
                    qc_data["max_proj_protein"],
                    qc_data["max_proj_ground_truth"],
                    df,
                    qc_data["ground_truth_positions"],
                    cfg,
                    qc_output_path,
                    objective_value,
                    num_single,
                    num_IN,
                    overlap_ratio,
                )
            except Exception as viz_error:
                logging.warning(f"QC visualization failed (trial still valid): {viz_error}")

    return objective_value


def pooled_objective_function(
    trial: optuna.Trial,
    protein_movie_pairs: List[Tuple[str, Tuple[Path, Path | None]]],
    base_config: dict,
    picasso_python: str,
    method: str,
    experiment_path: Path,
    aggregation: str = "mean",
) -> float:
    """
    Optuna objective function for per-experiment pooled optimization.

    Evaluates the same gradient on one movie from each protein in the experiment
    and returns an aggregated score.

    Args:
        trial: Optuna trial object.
        protein_movie_pairs: List of (protein_name, (protein_nd2, gt_nd2)) tuples.
        base_config: Base configuration dict.
        picasso_python: Path to Picasso Python.
        method: Localization method.
        experiment_path: Path to experiment folder for QC output.
        aggregation: How to aggregate scores ("mean" or "min").

    Returns:
        Aggregated objective value (higher is better).
    """
    trial_params = define_search_space(trial, base_config)

    # Check for duplicate parameters
    study = trial.study
    for completed_trial in study.trials:
        if completed_trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if completed_trial.number == trial.number:
            continue
        if completed_trial.params == trial_params:
            logging.info(
                f"  Trial {trial.number}: Duplicate of trial {completed_trial.number}, "
                f"returning cached objective: {completed_trial.value:.2f}"
            )
            for key, value in completed_trial.user_attrs.items():
                trial.set_user_attr(key, value)
            return completed_trial.value

    has_ground_truth = base_config.get("has_ground_truth", True)
    mode_suffix = "gt" if has_ground_truth else "no_gt"
    qc_folder = experiment_path / "visualQC" / mode_suffix
    qc_folder.mkdir(parents=True, exist_ok=True)

    per_protein_scores = {}

    for protein_name, movie_pair in protein_movie_pairs:
        score, df, qc_data = evaluate_single_movie(
            movie_pair, trial_params, base_config, picasso_python, method,
        )
        per_protein_scores[protein_name] = score

        # Store per-protein breakdown
        trial.set_user_attr(f"score_{protein_name}", float(score))

        # Store per-protein metrics
        if not df.empty:
            num_single = len(df[df["overlap"] == "single"])
            num_overlapping = len(df[df["overlap"] == "overlapping"])
            num_IN = 0
            if has_ground_truth and "ground_truth" in df.columns:
                num_IN = len(df[(df["ground_truth"] == "IN") & (df["overlap"] == "single")])
            total = num_single + num_overlapping
            overlap_ratio = num_overlapping / total if total > 0 else 0

            trial.set_user_attr(f"num_single_{protein_name}", num_single)
            trial.set_user_attr(f"num_IN_{protein_name}", num_IN)
            trial.set_user_attr(f"overlap_ratio_{protein_name}", float(overlap_ratio))

        # Draw per-protein QC image
        if not df.empty and qc_data:
            if has_ground_truth:
                qc_filename = (
                    f"trial_{trial.number:03d}_{protein_name}_"
                    f"gp-{trial_params['gradient_protein']}_"
                    f"ggt-{trial_params['gradient_ground_truth']}_"
                    f"box{base_config['boxsize']}.pdf"
                )
            else:
                qc_filename = (
                    f"trial_{trial.number:03d}_{protein_name}_"
                    f"gp-{trial_params['gradient_protein']}_"
                    f"box{base_config['boxsize']}.pdf"
                )

            try:
                cfg = base_config.copy()
                cfg.update(trial_params)
                num_single = len(df[df["overlap"] == "single"])
                num_overlapping = len(df[df["overlap"] == "overlapping"])
                num_IN = 0
                if has_ground_truth and "ground_truth" in df.columns:
                    num_IN = len(df[(df["ground_truth"] == "IN") & (df["overlap"] == "single")])
                total = num_single + num_overlapping
                overlap_ratio = num_overlapping / total if total > 0 else 0

                draw_trial_qc(
                    qc_data["max_proj_protein"],
                    qc_data["max_proj_ground_truth"],
                    df,
                    qc_data["ground_truth_positions"],
                    cfg,
                    qc_folder / qc_filename,
                    score,
                    num_single,
                    num_IN,
                    overlap_ratio,
                )
            except Exception as viz_error:
                logging.warning(f"QC visualization failed for {protein_name}: {viz_error}")

    scores = list(per_protein_scores.values())
    if not scores:
        return 0.0

    if aggregation == "min":
        aggregate = float(np.min(scores))
    else:
        aggregate = float(np.mean(scores))

    trial.set_user_attr("aggregation", aggregation)
    trial.set_user_attr("per_protein_scores", per_protein_scores)

    logging.info(
        f"  Trial {trial.number}: per-protein scores = "
        f"{', '.join(f'{p}={s:.1f}' for p, s in per_protein_scores.items())} "
        f"-> {aggregation}={aggregate:.1f}"
    )

    return aggregate


def optimize_experiment(
    experiment_path: Path,
    protein_folders: Dict[str, Path],
    base_config: dict,
    picasso_python: str,
    method: str,
    n_trials: int = 20,
    num_workers: int = 1,
    aggregation: str = "mean",
) -> dict:
    """
    Optimize localization parameters for an entire experiment using pooled scoring.

    Samples one movie per protein, evaluates each candidate gradient on all movies,
    and aggregates scores across proteins.

    Args:
        experiment_path: Path to experiment folder.
        protein_folders: Dict mapping protein name -> protein folder path.
        base_config: Base configuration.
        picasso_python: Path to Picasso Python.
        method: Localization method.
        n_trials: Number of optimization trials.
        num_workers: Number of parallel workers for optimization.
        aggregation: Score aggregation method ("mean" or "min").

    Returns:
        Dictionary with optimization results, or None if no movies found.
    """
    logging.info(f"Optimizing experiment: {experiment_path.name} (pooled across {len(protein_folders)} proteins)")

    has_ground_truth = base_config.get("has_ground_truth", True)
    protein_channel = base_config.get("protein_channel", "640")
    ground_truth_channel = base_config.get("ground_truth_channel", "488")

    # Sample one movie per protein
    protein_movie_pairs = []
    for protein, protein_path in protein_folders.items():
        nd2_files = find_nd2_files(protein_path)

        if has_ground_truth:
            pairs = pair_channel_files(nd2_files, protein_channel, ground_truth_channel)
        else:
            pairs = [(f, None) for f in nd2_files if protein_channel in f.name]

        if not pairs:
            logging.warning(f"  No movies found for {protein}, skipping this protein")
            continue

        selected = random.choice(pairs)
        protein_movie_pairs.append((protein, selected))
        logging.info(f"  {protein}: test movie = {selected[0].name}")

    if not protein_movie_pairs:
        logging.warning(f"  No movies found in any protein folder for {experiment_path.name}")
        return None

    # Auto-detect FOV size from first movie
    first_movie = protein_movie_pairs[0][1][0]
    try:
        fov_size = get_fov_size(first_movie)
        base_config["size_FOV"] = fov_size
        logging.info(f"  Auto-detected FOV size: {fov_size}x{fov_size} pixels")
    except Exception as e:
        logging.warning(f"  Could not auto-detect FOV size: {e}")

    # Create Optuna study at experiment level
    mode_suffix = "gt" if has_ground_truth else "no_gt"
    study_name = f"{experiment_path.name}_localization_{mode_suffix}"
    storage = f"sqlite:///{experiment_path}/optimization_{mode_suffix}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    logging.info(f"  Running {n_trials} trials (aggregation={aggregation})...")
    if num_workers > 1:
        logging.info(f"  Using {num_workers} parallel workers for optimization")
    logging.info(f"  QC images will be saved to: {experiment_path}/visualQC/{mode_suffix}/")

    study.optimize(
        lambda trial: pooled_objective_function(
            trial, protein_movie_pairs, base_config, picasso_python,
            method, experiment_path, aggregation,
        ),
        n_trials=n_trials,
        n_jobs=num_workers,
        timeout=3600,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial

    logging.info(f"  Best aggregate objective: {best_value:.2f}")
    logging.info(f"  Best parameters: {best_params}")

    # Log per-protein breakdown
    per_protein_scores = best_trial.user_attrs.get("per_protein_scores", {})
    for protein_name, score in per_protein_scores.items():
        logging.info(f"    {protein_name}: {score:.2f}")

    # Prepare results
    optimization_info = {
        "date": datetime.now().isoformat(),
        "experiment": experiment_path.name,
        "mode": "per_experiment",
        "aggregation": aggregation,
        "proteins_evaluated": [p for p, _ in protein_movie_pairs],
        "test_movies": {p: str(mp[0].name) for p, mp in protein_movie_pairs},
        "objective_function": "pooled_composite" if has_ground_truth else "pooled_simplified",
        "num_trials": n_trials,
        "best_aggregate_objective": float(best_value),
        "per_protein_scores": {k: float(v) for k, v in per_protein_scores.items()},
        "localization_method": method,
        "has_ground_truth": has_ground_truth,
    }

    optimized_parameters = {
        "gradient_protein": int(best_params["gradient_protein"]),
    }
    if has_ground_truth:
        optimized_parameters["gradient_ground_truth"] = int(best_params["gradient_ground_truth"])

    results = {
        "optimization_info": optimization_info,
        "optimized_parameters": optimized_parameters,
    }

    return results


def optimize_protein_experiment(
    protein: str,
    experiment_path: Path,
    protein_path: Path,
    base_config: dict,
    picasso_python: str,
    method: str,
    n_trials: int = 20,
    num_workers: int = 1,
) -> dict:
    """
    Optimize localization parameters for a specific protein/experiment.

    Args:
        protein: Protein name.
        experiment_path: Path to experiment folder.
        protein_path: Path to protein folder.
        base_config: Base configuration.
        picasso_python: Path to Picasso Python.
        method: Localization method.
        n_trials: Number of optimization trials.
        num_workers: Number of parallel workers for optimization.

    Returns:
        Dictionary with optimization results.
    """
    logging.info(f"Optimizing {experiment_path.name}/{protein}...")

    has_ground_truth = base_config.get("has_ground_truth", True)

    # Find all ND2 files
    nd2_files = find_nd2_files(protein_path)
    protein_channel = base_config.get("protein_channel", "640")
    ground_truth_channel = base_config.get("ground_truth_channel", "488")

    # Pair channels (or get single-channel files if no ground truth)
    if has_ground_truth:
        pairs = pair_channel_files(nd2_files, protein_channel, ground_truth_channel)
        if not pairs:
            logging.warning(f"  No paired movies found for {protein}, skipping")
            return None
    else:
        # Single-channel mode: use protein channel files only
        pairs = [(f, None) for f in nd2_files if protein_channel in f.name]
        if not pairs:
            logging.warning(f"  No {protein_channel}nm movies found for {protein}, skipping")
            return None

    # Select random test movie
    test_movie_pair = random.choice(pairs)
    logging.info(f"  Test movie: {test_movie_pair[0].name}")
    logging.info(f"  Found {len(pairs)} total movie pairs")

    # Auto-detect FOV size from test movie
    try:
        fov_size = get_fov_size(test_movie_pair[0])
        base_config["size_FOV"] = fov_size
        logging.info(f"  Auto-detected FOV size: {fov_size}x{fov_size} pixels")
    except Exception as e:
        logging.warning(f"  Could not auto-detect FOV size: {e}")
        if "size_FOV" in base_config:
            logging.info(f"  Using FOV size from config: {base_config['size_FOV']}")
        else:
            logging.warning(f"  No FOV size available, proceeding without it")

    # Create Optuna study with mode-specific naming
    mode_suffix = "gt" if has_ground_truth else "no_gt"
    study_name = f"{experiment_path.name}_{protein}_localization_{mode_suffix}"
    storage = f"sqlite:///{protein_path}/optimization_{mode_suffix}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    # Run optimization
    mode_suffix = "gt" if has_ground_truth else "no_gt"
    logging.info(f"  Running {n_trials} trials...")
    if num_workers > 1:
        logging.info(f"  Using {num_workers} parallel workers for optimization")
    logging.info(f"  QC images will be saved to: {protein_path}/visualQC/{mode_suffix}/")
    study.optimize(
        lambda trial: objective_function(
            trial, test_movie_pair, base_config, picasso_python, method, protein_path
        ),
        n_trials=n_trials,
        n_jobs=num_workers,  # Parallel optimization
        timeout=3600,  # 1 hour max
        show_progress_bar=False,
    )

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial

    logging.info(f"  Best objective: {best_value:.2f}")
    logging.info(f"  Best parameters: {best_params}")

    # Prepare results
    optimization_info = {
        "date": datetime.now().isoformat(),
        "experiment": experiment_path.name,
        "protein": protein,
        "test_movie_protein": str(test_movie_pair[0].name),
        "total_movie_pairs": len(pairs),
        "objective_function": "composite" if has_ground_truth else "simplified",
        "num_trials": n_trials,
        "best_objective_value": float(best_value),
        "localization_method": method,
        "has_ground_truth": has_ground_truth,
    }

    if has_ground_truth and test_movie_pair[1]:
        optimization_info["test_movie_ground_truth"] = str(test_movie_pair[1].name)

    optimized_parameters = {
        "gradient_protein": int(best_params["gradient_protein"]),
    }

    if has_ground_truth:
        optimized_parameters["gradient_ground_truth"] = int(best_params["gradient_ground_truth"])

    results = {
        "optimization_info": optimization_info,
        "optimized_parameters": optimized_parameters,
        "results_on_test_movie": {
            "num_single": int(best_trial.user_attrs.get("num_single", 0)),
            "num_overlapping": int(best_trial.user_attrs.get("num_overlapping", 0)),
            "num_IN": int(best_trial.user_attrs.get("num_IN", 0)),
            "overlap_ratio": float(best_trial.user_attrs.get("overlap_ratio", 0)),
        },
    }

    # Add ground truth intensity statistics if available
    if has_ground_truth:
        gt_stats = best_trial.user_attrs.get("ground_truth_intensity_stats")
        if gt_stats:
            results["ground_truth_detection_stats"] = gt_stats

    return results


def save_optimized_params(results: dict, save_path: Path) -> None:
    """
    Save optimized parameters with mode-specific naming.

    Args:
        results: Optimization results dictionary.
        save_path: Path to save location (experiment folder or protein folder).
    """
    has_ground_truth = results["optimization_info"].get("has_ground_truth", True)
    mode_suffix = "gt" if has_ground_truth else "no_gt"
    output_file = save_path / f"optimized_localization_params_{mode_suffix}.yaml"

    with output_file.open("w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    logging.info(f"  Saved parameters to: {output_file}")


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Optimize Picasso localization parameters per protein/experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--protein",
        type=str,
        help="Optimize specific protein only",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Optimize specific experiment only (requires --protein)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of optimization trials per protein/experiment (default: from config or 20)",
    )

    args = parser.parse_args()

    # Setup
    setup_logging("INFO")
    config_path = Path(args.config_path).resolve()

    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)

    # Get n_trials from config if not specified via CLI
    n_trials = args.n_trials
    if n_trials is None:
        n_trials = cfg.get("optimization", {}).get("n_trials", 20)

    # Get parallel processing configuration
    n_workers_config = cfg.get("n_workers", -1)
    num_workers = get_num_workers(n_workers_config)

    # Get paths
    input_root = Path(cfg["input_folder"]).expanduser()
    proteins = cfg["proteins"]

    # Get optimization mode and aggregation
    opt_cfg = cfg.get("optimization", {})
    opt_mode = opt_cfg.get("mode", "per_experiment")
    aggregation = opt_cfg.get("aggregation", "mean")

    logging.info("=" * 60)
    logging.info("Localization Parameter Optimization")
    logging.info("=" * 60)
    logging.info(f"Input folder: {input_root}")
    logging.info(f"Proteins: {', '.join(proteins)}")
    logging.info(f"Optimization mode: {opt_mode}")
    if opt_mode == "per_experiment":
        logging.info(f"Score aggregation: {aggregation}")
    logging.info(f"Trials per optimization: {n_trials}")
    logging.info(f"Frames per trial: 1000 (for speed)")

    # Log search space
    search_space = opt_cfg.get("search_space", {})
    has_ground_truth = cfg.get("has_ground_truth", True)
    logging.info(f"Search space:")
    logging.info(f"  gradient_protein: {search_space.get('gradient_protein_min', 10000)}-{search_space.get('gradient_protein_max', 80000)} (step {search_space.get('gradient_protein_step', 10000)})")
    if has_ground_truth:
        logging.info(f"  gradient_ground_truth: {search_space.get('gradient_ground_truth_min', 1000)}-{search_space.get('gradient_ground_truth_max', 10000)} (step {search_space.get('gradient_ground_truth_step', 1000)})")

    # Get localization method (always CPU-based MLE)
    method = get_localization_method()

    # Find Picasso
    try:
        picasso_python = get_picasso_python()
    except RuntimeError as e:
        logging.error(str(e))
        sys.exit(1)

    # Discover experiments
    experiments = discover_experiments(input_root)

    if not experiments:
        logging.error(f"No experiment folders found in {input_root}")
        sys.exit(1)

    logging.info(f"Found {len(experiments)} experiments")

    # Filter by user-specified experiment if provided
    if args.experiment:
        experiments = [e for e in experiments if e.name == args.experiment]
        if not experiments:
            logging.error(f"Experiment '{args.experiment}' not found")
            sys.exit(1)

    # Filter by user-specified protein if provided
    target_proteins = [args.protein] if args.protein else proteins

    total_optimized = 0
    total_skipped = 0

    if opt_mode == "per_experiment":
        # Pooled optimization: one gradient set per experiment
        for exp_path in experiments:
            logging.info(f"\nProcessing experiment: {exp_path.name}")

            protein_folders = discover_protein_folders(exp_path, target_proteins)
            if not protein_folders:
                logging.warning(f"  No protein folders found in {exp_path.name}")
                total_skipped += 1
                continue

            try:
                results = optimize_experiment(
                    exp_path,
                    protein_folders,
                    cfg,
                    picasso_python,
                    method,
                    n_trials,
                    num_workers,
                    aggregation,
                )

                if results:
                    save_optimized_params(results, exp_path)
                    total_optimized += 1
                else:
                    total_skipped += 1

            except Exception as e:
                logging.error(f"  Failed to optimize {exp_path.name}: {e}")
                total_skipped += 1
                continue

        # Summary
        logging.info("=" * 60)
        logging.info("Optimization Complete")
        logging.info("=" * 60)
        logging.info(f"Optimized: {total_optimized} experiments")
        logging.info(f"Skipped: {total_skipped} experiments")
        logging.info("")
        logging.info("Optimized parameters saved to experiment folders.")
        logging.info("Run localize.py to use these parameters automatically.")

    else:
        # Legacy per-protein optimization
        for exp_path in experiments:
            logging.info(f"\nProcessing experiment: {exp_path.name}")

            protein_folders = discover_protein_folders(exp_path, target_proteins)
            if not protein_folders:
                logging.warning(f"  No protein folders found in {exp_path.name}")
                continue

            for protein, protein_path in protein_folders.items():
                try:
                    results = optimize_protein_experiment(
                        protein,
                        exp_path,
                        protein_path,
                        cfg,
                        picasso_python,
                        method,
                        n_trials,
                        num_workers,
                    )

                    if results:
                        save_optimized_params(results, protein_path)
                        total_optimized += 1
                    else:
                        total_skipped += 1

                except Exception as e:
                    logging.error(f"  Failed to optimize {protein}: {e}")
                    total_skipped += 1
                    continue

        # Summary
        logging.info("=" * 60)
        logging.info("Optimization Complete")
        logging.info("=" * 60)
        logging.info(f"Optimized: {total_optimized} protein/experiment combinations")
        logging.info(f"Skipped: {total_skipped} combinations")
        logging.info("")
        logging.info("Optimized parameters saved to data folders.")
        logging.info("Run localize.py to use these parameters automatically.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("\nOptimization interrupted by user")
        sys.exit(1)
    except Exception as ex:
        logging.error(f"Fatal error: {ex}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
