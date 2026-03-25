#!/usr/bin/env python3
"""
Protein-to-Ground Truth Correlation Analysis

This script tests whether protein signal (640nm) localizations correspond to
vesicles with weak but above-background GT signal (488nm).

Workflow:
1. Localize in 640nm channel → cluster → get cluster centers
2. At each 640nm cluster center, extract 488nm intensity (9×9×10 voxel)
3. Find random boxes in 488nm that avoid any 488nm clusters (true background)
4. Compare: 488nm intensity at protein positions vs. random background
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import stats

from utils import (
    COLORS,
    FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_TITLE, FONTSIZE_LEGEND,
    apply_axis_standards,
    save_figure,
    load_config,
    setup_logging,
    get_picasso_python,
    get_localization_method,
    discover_all_data,
    get_num_workers,
)


def run_localization(
    nd2_file: Path,
    gradient: int,
    cfg: dict,
    picasso_python: str,
    output_dir: Path,
    channel_name: str,
) -> Path:
    """
    Run Picasso localization on specified channel.

    Args:
        nd2_file: Path to ND2 file.
        gradient: Gradient threshold.
        cfg: Configuration dictionary.
        picasso_python: Path to Picasso Python.
        output_dir: Directory for output files.
        channel_name: Channel identifier (e.g., "640" or "488").

    Returns:
        Path to localization HDF5 file.
    """
    # Skip macOS metadata files
    if nd2_file.name.startswith("._"):
        raise ValueError(f"Invalid file (macOS metadata): {nd2_file.name}")

    method = get_localization_method()

    # Use a temporary directory for Picasso processing
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir_path = Path(temp_dir_str)

        # Copy ND2 to temp directory (Picasso creates output next to input)
        temp_nd2 = temp_dir_path / nd2_file.name
        shutil.copy2(nd2_file, temp_nd2)

        # Clear macOS extended attributes
        try:
            subprocess.run(
                ["xattr", "-c", str(temp_nd2)],
                check=False,
                capture_output=True,
            )
        except Exception:
            pass

        # Picasso localize command
        cmd = [
            picasso_python,
            "-m",
            "picasso",
            "localize",
            str(temp_nd2),
            "-a", method,
            "-g", str(gradient),
            "-b", str(cfg["boxsize"]),
        ]

        # Set environment for headless execution (HPC compatibility)
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Localization failed: {result.stderr}")
            raise RuntimeError(f"Picasso localization failed with gradient={gradient}")

        # Picasso creates files with _locs.hdf5 and _locs.yaml suffixes
        locs_file = temp_nd2.parent / f"{temp_nd2.stem}_locs.hdf5"
        yaml_file = temp_nd2.parent / f"{temp_nd2.stem}_locs.yaml"

        if not locs_file.exists():
            raise RuntimeError(f"Expected localization file not created: {locs_file}")

        # Move both files to output directory with channel name
        final_locs_file = output_dir / f"{nd2_file.stem}_{channel_name}nm_locs.hdf5"
        final_yaml_file = output_dir / f"{nd2_file.stem}_{channel_name}nm_locs.yaml"

        shutil.move(locs_file, final_locs_file)
        if yaml_file.exists():
            shutil.move(yaml_file, final_yaml_file)

    return final_locs_file


def get_cluster_centers(locs_file: Path, cfg: dict) -> np.ndarray:
    """
    Get cluster centers from localizations.

    Args:
        locs_file: Path to localization HDF5 file.
        cfg: Configuration dictionary.

    Returns:
        Array of cluster centers with 'x' and 'y' fields.
    """
    from picasso import io as pio, lib, clusterer

    # Load localizations
    locs, info = pio.load_locs(str(locs_file))
    locs = lib.ensure_sanity(locs, info)

    if len(locs) == 0:
        return np.array([])

    # Cluster localizations
    cluster_locs = clusterer.cluster(
        locs,
        radius_xy=cfg["max_distance_ground_truth"],
        min_locs=cfg["min_on_ground_truth"],
        frame_analysis=False,
        radius_z=None,
        pixelsize=130,
    )
    cluster_locs = clusterer.find_cluster_centers(cluster_locs, pixelsize=None)

    return cluster_locs


def extract_intensity_at_positions(
    movie: np.ndarray,
    positions: np.ndarray,
    boxsize: int,
    n_frames: int,
) -> List[float]:
    """
    Extract integrated intensities at specified positions.

    Args:
        movie: Movie array (frames, height, width).
        positions: Array with 'x' and 'y' fields for each position.
        boxsize: ROI box size (pixels).
        n_frames: Number of frames to integrate.

    Returns:
        List of integrated intensities.
    """
    shift = boxsize / 2
    intensities = []
    n_frames = min(n_frames, movie.shape[0])

    for pos in positions:
        x = int(pos["x"] - shift)
        y = int(pos["y"] - shift)

        # Bounds check
        if x < 0 or y < 0:
            continue
        if x + boxsize > movie.shape[2] or y + boxsize > movie.shape[1]:
            continue

        # Extract ROI and integrate
        roi = movie[:n_frames, y:(y + boxsize), x:(x + boxsize)]
        integrated_intensity = np.sum(roi)
        intensities.append(integrated_intensity)

    return intensities


def find_random_background_positions(
    movie_shape: Tuple[int, int, int],
    gt_cluster_centers: np.ndarray,
    protein_cluster_centers: np.ndarray,
    boxsize: int,
    n_samples: int,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """
    Find random positions that don't overlap with GT clusters or protein positions.

    Args:
        movie_shape: Shape of movie (frames, height, width).
        gt_cluster_centers: Array of GT cluster centers with 'x' and 'y' fields.
        protein_cluster_centers: Array of protein cluster centers with 'x' and 'y' fields.
        boxsize: ROI box size (pixels).
        n_samples: Number of random samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of (x, y) tuples for random positions.
    """
    np.random.seed(seed)

    height, width = movie_shape[1], movie_shape[2]
    margin = boxsize

    # Exclusion radius: ensure no overlap with any ROI
    # For two boxes of size 'boxsize' centered at distance 'd':
    # - They touch (but don't overlap) if d = boxsize
    # - To ensure clear separation with no overlap, use d > boxsize
    # Add sqrt(2)*boxsize to ensure boxes separated even on diagonals
    exclusion_radius = int(np.ceil(boxsize * np.sqrt(2)))

    random_positions = []
    max_attempts = n_samples * 100  # Prevent infinite loop
    attempts = 0

    while len(random_positions) < n_samples and attempts < max_attempts:
        attempts += 1

        # Generate random position
        x = np.random.randint(margin, width - margin)
        y = np.random.randint(margin, height - margin)

        # Check if it overlaps with any GT cluster
        if len(gt_cluster_centers) > 0:
            distances = np.sqrt(
                (gt_cluster_centers["x"] - x)**2 +
                (gt_cluster_centers["y"] - y)**2
            )
            if np.min(distances) < exclusion_radius:
                continue

        # Check if it overlaps with any protein position
        if len(protein_cluster_centers) > 0:
            distances = np.sqrt(
                (protein_cluster_centers["x"] - x)**2 +
                (protein_cluster_centers["y"] - y)**2
            )
            if np.min(distances) < exclusion_radius:
                continue

        # Check if it overlaps with previously selected random positions
        if len(random_positions) > 0:
            prev_x = np.array([p[0] for p in random_positions])
            prev_y = np.array([p[1] for p in random_positions])
            distances = np.sqrt((prev_x - x)**2 + (prev_y - y)**2)
            if np.min(distances) < exclusion_radius:
                continue

        random_positions.append((x, y))

    if len(random_positions) < n_samples:
        logging.warning(
            f"Could only find {len(random_positions)} non-overlapping random positions "
            f"(requested {n_samples})"
        )

    return random_positions


def analyze_movie_pair_worker(
    protein_file: Path,
    gt_file: Path,
    cfg: dict,
    picasso_python: str,
    output_dir: Path,
    protein_gradient: int,
    gt_gradient: int,
) -> tuple[str, Dict[str, np.ndarray] | None]:
    """
    Worker function for parallel processing of movie pairs.

    Returns:
        Tuple of (movie_name, results_dict or None if failed)
    """
    try:
        results = analyze_movie_pair(
            protein_file, gt_file, cfg, picasso_python, output_dir,
            protein_gradient, gt_gradient
        )
        return (protein_file.name, results)
    except Exception as e:
        logging.error(f"  Failed to analyze {protein_file.name}: {e}")
        return (protein_file.name, None)


def analyze_movie_pair(
    protein_file: Path,
    gt_file: Path,
    cfg: dict,
    picasso_python: str,
    output_dir: Path,
    protein_gradient: int = 60000,
    gt_gradient: int = 5000,
) -> Dict[str, np.ndarray]:
    """
    Analyze protein-to-GT correlation for a movie pair.

    Args:
        protein_file: Path to 640nm ND2 file.
        gt_file: Path to 488nm ND2 file.
        cfg: Configuration dictionary.
        picasso_python: Path to Picasso Python.
        output_dir: Output directory.
        protein_gradient: Gradient for protein channel.
        gt_gradient: Gradient for GT channel.

    Returns:
        Dictionary with 'protein_positions' and 'background' intensity arrays.
    """
    import nd2

    logging.info(f"  Analyzing {protein_file.name}")

    # Load movies
    protein_movie = nd2.imread(str(protein_file))
    gt_movie = nd2.imread(str(gt_file))

    # Run localizations
    logging.info(f"    Localizing 640nm (gradient={protein_gradient})...")
    protein_locs_file = run_localization(
        protein_file, protein_gradient, cfg, picasso_python, output_dir, "640"
    )

    logging.info(f"    Localizing 488nm (gradient={gt_gradient})...")
    gt_locs_file = run_localization(
        gt_file, gt_gradient, cfg, picasso_python, output_dir, "488"
    )

    # Get cluster centers
    logging.info(f"    Clustering localizations...")
    protein_centers = get_cluster_centers(protein_locs_file, cfg)
    gt_centers = get_cluster_centers(gt_locs_file, cfg)

    logging.info(f"      → {len(protein_centers)} protein clusters")
    logging.info(f"      → {len(gt_centers)} GT clusters")

    if len(protein_centers) == 0:
        logging.warning(f"    No protein clusters found, skipping")
        return {"protein_positions": np.array([]), "background": np.array([])}

    # Filter protein centers to remove those that overlap with GT localizations
    # Use same exclusion radius as background selection
    exclusion_radius = int(np.ceil(cfg["boxsize"] * np.sqrt(2)))
    non_overlapping_protein_centers = []

    if len(gt_centers) > 0:
        for protein_center in protein_centers:
            # Calculate distances to all GT centers
            distances = np.sqrt(
                (gt_centers["x"] - protein_center["x"])**2 +
                (gt_centers["y"] - protein_center["y"])**2
            )
            # Keep only if it doesn't overlap with any GT
            if np.min(distances) >= exclusion_radius:
                non_overlapping_protein_centers.append(protein_center)
    else:
        # No GT centers, so all protein centers are non-overlapping
        non_overlapping_protein_centers = list(protein_centers)

    # Convert back to structured array
    if len(non_overlapping_protein_centers) > 0:
        protein_centers_filtered = np.array(
            non_overlapping_protein_centers,
            dtype=protein_centers.dtype
        )
    else:
        protein_centers_filtered = np.array([], dtype=protein_centers.dtype)

    logging.info(
        f"      → {len(protein_centers_filtered)} protein clusters after removing GT overlaps "
        f"({len(protein_centers) - len(protein_centers_filtered)} removed)"
    )

    if len(protein_centers_filtered) == 0:
        logging.warning(f"    No non-overlapping protein clusters found, skipping")
        return {"protein_positions": np.array([]), "background": np.array([])}

    # Extract 488nm intensities at protein positions (non-overlapping only)
    logging.info(f"    Extracting 488nm intensities at non-overlapping protein positions...")
    protein_position_intensities = extract_intensity_at_positions(
        gt_movie,
        protein_centers_filtered,
        cfg["boxsize"],
        cfg["n_frames_integrate"],
    )

    # Find random background positions (avoiding GT clusters and protein positions)
    logging.info(f"    Finding random background positions...")
    n_random = len(protein_centers_filtered)  # Same number as non-overlapping protein positions
    random_positions = find_random_background_positions(
        gt_movie.shape,
        gt_centers,
        protein_centers_filtered,
        cfg["boxsize"],
        n_random,
    )

    # Convert random positions to structured array format
    random_pos_array = np.zeros(
        len(random_positions),
        dtype=[("x", float), ("y", float)]
    )
    for i, (x, y) in enumerate(random_positions):
        random_pos_array[i]["x"] = x
        random_pos_array[i]["y"] = y

    # Extract 488nm intensities at random background positions
    logging.info(f"    Extracting 488nm intensities at background positions...")
    background_intensities = extract_intensity_at_positions(
        gt_movie,
        random_pos_array,
        cfg["boxsize"],
        cfg["n_frames_integrate"],
    )

    logging.info(
        f"      → Protein positions: mean={np.mean(protein_position_intensities):.0f}, "
        f"median={np.median(protein_position_intensities):.0f}"
    )
    logging.info(
        f"      → Background: mean={np.mean(background_intensities):.0f}, "
        f"median={np.median(background_intensities):.0f}"
    )

    # Create visualization plots
    logging.info(f"    Creating visualization plots...")

    # Create per-movie plots directory
    plots_dir = output_dir / "plots" / "per_movie"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get base name for output files
    base_name = protein_file.stem.replace("_640nm", "")

    # Plot 640nm max projection with protein ROIs (show all, not just filtered)
    plot_640nm_path = plots_dir / f"{base_name}_640nm_protein_rois.pdf"
    plot_640nm_max_projection(
        protein_movie,
        protein_centers,
        cfg["boxsize"],
        plot_640nm_path,
        base_name,
    )

    # Plot 488nm max projection with all ROI types (use filtered protein centers)
    plot_488nm_path = plots_dir / f"{base_name}_488nm_roi_overlays.pdf"
    plot_488nm_max_projection(
        gt_movie,
        gt_centers,
        protein_centers_filtered,
        random_positions,
        cfg["boxsize"],
        plot_488nm_path,
        base_name,
    )

    # Plot 488nm max projection with line profiles (use filtered protein centers)
    plot_line_profiles_path = plots_dir / f"{base_name}_488nm_line_profiles.pdf"
    plot_488nm_with_line_profiles(
        gt_movie,
        gt_centers,
        protein_centers_filtered,
        random_positions,
        cfg["boxsize"],
        plot_line_profiles_path,
        base_name,
    )

    logging.info(f"      → Saved plots to {plots_dir}/")

    return {
        "protein_positions": np.array(protein_position_intensities),
        "background": np.array(background_intensities),
    }


def plot_640nm_max_projection(
    movie: np.ndarray,
    protein_centers: np.ndarray,
    boxsize: int,
    output_path: Path,
    movie_name: str,
) -> None:
    """
    Plot 640nm max projection with protein ROI overlays.

    Args:
        movie: 640nm movie array (frames, height, width).
        protein_centers: Array of protein cluster centers with 'x' and 'y' fields.
        boxsize: ROI box size.
        output_path: Path to save the plot.
        movie_name: Name of the movie for title.
    """
    # Compute max projection
    max_proj = np.max(movie, axis=0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(max_proj, cmap="gray")

    # Draw protein ROI boxes (yellow - color-blind friendly)
    for center in protein_centers:
        x = center["x"] - boxsize / 2
        y = center["y"] - boxsize / 2
        rect = patches.Rectangle(
            (x, y),
            boxsize,
            boxsize,
            linewidth=1.5,
            edgecolor=COLORS['yellow'],
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.set_title(
        f"640nm max projection - protein ROIs",
        fontsize=FONTSIZE_TITLE,
        pad=9,
    )
    ax.axis("off")

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['yellow'], linewidth=2, label=f"Protein ROIs (n={len(protein_centers)})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=FONTSIZE_LEGEND,
              frameon=False)

    save_figure(plt.gcf(), output_path.parent, output_path.name)
    plt.close("all")


def plot_488nm_max_projection(
    movie: np.ndarray,
    gt_centers: np.ndarray,
    protein_centers: np.ndarray,
    background_positions: List[Tuple[float, float]],
    boxsize: int,
    output_path: Path,
    movie_name: str,
) -> None:
    """
    Plot 488nm max projection with three types of ROI overlays.

    Args:
        movie: 488nm movie array (frames, height, width).
        gt_centers: Array of GT cluster centers (detected in 488nm).
        protein_centers: Array of protein cluster centers (from 640nm).
        background_positions: List of (x, y) tuples for random background positions.
        boxsize: ROI box size.
        output_path: Path to save the plot.
        movie_name: Name of the movie for title.
    """
    # Compute max projection
    max_proj = np.max(movie, axis=0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(max_proj, cmap="gray")

    # Draw GT detected ROI boxes (solid blue - color-blind friendly)
    for center in gt_centers:
        x = center["x"] - boxsize / 2
        y = center["y"] - boxsize / 2
        rect = patches.Rectangle(
            (x, y),
            boxsize,
            boxsize,
            linewidth=1.5,
            edgecolor=COLORS['blue'],
            facecolor="none",
            linestyle="-",
        )
        ax.add_patch(rect)

    # Draw protein position ROI boxes (solid yellow - color-blind friendly)
    for center in protein_centers:
        x = center["x"] - boxsize / 2
        y = center["y"] - boxsize / 2
        rect = patches.Rectangle(
            (x, y),
            boxsize,
            boxsize,
            linewidth=1.5,
            edgecolor=COLORS['yellow'],
            facecolor="none",
            linestyle="-",
        )
        ax.add_patch(rect)

    # Draw background ROI boxes (dotted cyan)
    for x_center, y_center in background_positions:
        x = x_center - boxsize / 2
        y = y_center - boxsize / 2
        rect = patches.Rectangle(
            (x, y),
            boxsize,
            boxsize,
            linewidth=1.5,
            edgecolor="cyan",
            facecolor="none",
            linestyle=":",
        )
        ax.add_patch(rect)

    ax.set_title(
        f"488nm max projection - ROI overlays",
        fontsize=FONTSIZE_TITLE,
        pad=9,
    )
    ax.axis("off")

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['blue'], linewidth=2, linestyle="-",
               label=f"Detected GT ROIs (n={len(gt_centers)})"),
        Line2D([0], [0], color=COLORS['yellow'], linewidth=2, linestyle="-",
               label=f"Protein positions (n={len(protein_centers)})"),
        Line2D([0], [0], color='cyan', linewidth=2, linestyle=":",
               label=f"Background ROIs (n={len(background_positions)})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=FONTSIZE_LEGEND,
              frameon=False)

    save_figure(plt.gcf(), output_path.parent, output_path.name)
    plt.close("all")


def roi_touches_edge(
    center_x: float,
    center_y: float,
    boxsize: int,
    image_shape: Tuple[int, int],
) -> bool:
    """
    Check if an ROI touches the edge of the field of view.

    Args:
        center_x: X coordinate of ROI center.
        center_y: Y coordinate of ROI center.
        boxsize: Size of the ROI box (square).
        image_shape: Shape of the image (height, width).

    Returns:
        True if the ROI touches any edge, False otherwise.
    """
    height, width = image_shape
    half_box = boxsize / 2

    # Check all four edges
    if center_x - half_box < 0:  # Left edge
        return True
    if center_x + half_box >= width:  # Right edge
        return True
    if center_y - half_box < 0:  # Top edge
        return True
    if center_y + half_box >= height:  # Bottom edge
        return True

    return False


def extract_line_profile(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    boxsize: int,
) -> np.ndarray:
    """
    Extract horizontal line profile through the center of a box.

    Args:
        image: 2D image array.
        center_x: X coordinate of box center.
        center_y: Y coordinate of box center.
        boxsize: Size of the box.

    Returns:
        1D array of intensities along the horizontal line through center.
    """
    # Extract horizontal line through center
    y_line = int(center_y)
    x_start = int(center_x - boxsize / 2)
    x_end = int(center_x + boxsize / 2)

    # Ensure within bounds
    x_start = max(0, x_start)
    x_end = min(image.shape[1], x_end)

    line_profile = image[y_line, x_start:x_end]
    return line_profile


def plot_488nm_with_line_profiles(
    movie: np.ndarray,
    gt_centers: np.ndarray,
    protein_centers: np.ndarray,
    background_positions: List[Tuple[float, float]],
    boxsize: int,
    output_path: Path,
    movie_name: str,
) -> None:
    """
    Plot 488nm max projection with line profiles for selected ROIs.

    Creates a 4-panel figure:
    - Top: Max projection with numbered selected ROIs
    - Bottom 3: Line profiles through GT, protein, and background boxes

    Args:
        movie: 488nm movie array (frames, height, width).
        gt_centers: Array of GT cluster centers (detected in 488nm).
        protein_centers: Array of protein cluster centers (from 640nm).
        background_positions: List of (x, y) tuples for random background positions.
        boxsize: ROI box size.
        output_path: Path to save the plot.
        movie_name: Name of the movie for title.
    """
    # Compute max projection
    max_proj = np.max(movie, axis=0)
    image_shape = max_proj.shape

    # Select representative examples (one of each type)
    # Avoid ROIs that touch the edge of the FOV
    selected_gt = None
    if len(gt_centers) > 0:
        for center in gt_centers:
            if not roi_touches_edge(center["x"], center["y"], boxsize, image_shape):
                selected_gt = center
                break

    # Select a protein position that does NOT overlap with any GT localization
    # AND does not touch the edge of the FOV
    # Use same exclusion radius as background selection
    exclusion_radius = int(np.ceil(boxsize * np.sqrt(2)))
    selected_protein = None

    if len(protein_centers) > 0 and len(gt_centers) > 0:
        for protein_center in protein_centers:
            # Skip if touches edge
            if roi_touches_edge(protein_center["x"], protein_center["y"], boxsize, image_shape):
                continue

            # Calculate distances to all GT centers
            distances = np.sqrt(
                (gt_centers["x"] - protein_center["x"])**2 +
                (gt_centers["y"] - protein_center["y"])**2
            )
            # If this protein position doesn't overlap with any GT, select it
            if np.min(distances) >= exclusion_radius:
                selected_protein = protein_center
                break
    elif len(protein_centers) > 0:
        # No GT centers, so any protein center is fine (as long as not touching edge)
        for protein_center in protein_centers:
            if not roi_touches_edge(protein_center["x"], protein_center["y"], boxsize, image_shape):
                selected_protein = protein_center
                break

    # Select background position that doesn't touch edge
    selected_background = None
    if len(background_positions) > 0:
        for bg_pos in background_positions:
            if not roi_touches_edge(bg_pos[0], bg_pos[1], boxsize, image_shape):
                selected_background = bg_pos
                break

    # Create figure with 4 panels in one row
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # First panel: Max projection - show ONLY the 3 selected ROIs
    ax_img = axes[0]
    ax_img.imshow(max_proj, cmap="gray")

    # Draw only the selected GT ROI (blue - color-blind friendly)
    if selected_gt is not None:
        x = selected_gt["x"] - boxsize / 2
        y = selected_gt["y"] - boxsize / 2

        rect = patches.Rectangle(
            (x, y),
            boxsize,
            boxsize,
            linewidth=1.25,
            edgecolor=COLORS['blue'],
            facecolor="none",
            linestyle="-",
        )
        ax_img.add_patch(rect)

        # Add number outside box
        ax_img.text(
            selected_gt["x"] - boxsize / 2 - 1.5,
            selected_gt["y"] - boxsize / 2 - 1.5,
            "1",
            color=COLORS['blue'], fontsize=FONTSIZE_TICK, fontweight="bold",
            ha="right", va="bottom",
        )

    # Draw only the selected protein ROI (yellow - color-blind friendly)
    if selected_protein is not None:
        x = selected_protein["x"] - boxsize / 2
        y = selected_protein["y"] - boxsize / 2

        rect = patches.Rectangle(
            (x, y),
            boxsize,
            boxsize,
            linewidth=1.25,
            edgecolor=COLORS['yellow'],
            facecolor="none",
            linestyle="-",
        )
        ax_img.add_patch(rect)

        # Add number outside box
        ax_img.text(
            selected_protein["x"] - boxsize / 2 - 1.5,
            selected_protein["y"] - boxsize / 2 - 1.5,
            "2",
            color=COLORS['yellow'], fontsize=FONTSIZE_TICK, fontweight="bold",
            ha="right", va="bottom",
        )

    # Draw only the selected background ROI (cyan - color-blind friendly)
    if selected_background is not None:
        x_center, y_center = selected_background
        x = x_center - boxsize / 2
        y = y_center - boxsize / 2

        rect = patches.Rectangle(
            (x, y),
            boxsize,
            boxsize,
            linewidth=1.25,
            edgecolor='cyan',
            facecolor="none",
            linestyle=":",
        )
        ax_img.add_patch(rect)

        # Add number outside box
        ax_img.text(
            x_center - boxsize / 2 - 1.5,
            y_center - boxsize / 2 - 1.5,
            "3",
            color='cyan', fontsize=FONTSIZE_TICK, fontweight="bold",
            ha="right", va="bottom",
        )

    # Add legend to show ROI types
    from matplotlib.lines import Line2D
    legend_elements = []
    if selected_gt is not None:
        legend_elements.append(Line2D([0], [0], color=COLORS['blue'], lw=1.25, label='GT ROI'))
    if selected_protein is not None:
        legend_elements.append(Line2D([0], [0], color=COLORS['yellow'], lw=1.25, label='Protein ROI'))
    if selected_background is not None:
        legend_elements.append(Line2D([0], [0], color='cyan', lw=1.25, linestyle=':', label='Background ROI'))

    if legend_elements:
        ax_img.legend(handles=legend_elements, loc='upper right', fontsize=FONTSIZE_LEGEND,
                      frameon=False)

    ax_img.set_title(
        f"488nm max projection",
        fontsize=FONTSIZE_TITLE,
        pad=9,
    )
    ax_img.axis("off")

    # Remaining 3 panels: Line profiles
    # Match colors to the ROI box colors (blue, yellow, cyan - color-blind friendly)
    profiles = [
        (selected_gt, "1. GT ROI", COLORS['blue']),
        (selected_protein, "2. Protein ROI", COLORS['yellow']),
        (selected_background, "3. Background ROI", 'cyan'),
    ]

    # Extract all profiles first to determine max intensity
    profile_data = []
    max_intensity = 0

    for idx, (roi_data, title, color) in enumerate(profiles):
        if roi_data is not None:
            if idx == 2:  # Background (tuple)
                cx, cy = roi_data
            else:  # GT or protein (structured array)
                cx, cy = roi_data["x"], roi_data["y"]

            profile = extract_line_profile(max_proj, cx, cy, boxsize)
            profile_data.append(profile)
            max_intensity = max(max_intensity, np.max(profile))
        else:
            profile_data.append(None)

    # Add some headroom to max intensity
    y_max = max_intensity * 1.1

    # Plot all profiles with same y-axis scale
    for idx, (roi_data, title, color) in enumerate(profiles):
        ax = axes[idx + 1]
        profile = profile_data[idx]

        if profile is not None:
            ax.plot(profile, color=color, linewidth=1.5)
            ax.set_xlabel("Pixel position", fontsize=FONTSIZE_LABEL)
            ax.set_ylabel("Intensity", fontsize=FONTSIZE_LABEL)
            ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=9)
            ax.set_ylim(0, y_max)
            apply_axis_standards(ax)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                   fontsize=FONTSIZE_LABEL, color='#808080', transform=ax.transAxes)
            ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=9)
            apply_axis_standards(ax)

    save_figure(plt.gcf(), output_path.parent, output_path.name)
    plt.close("all")


def plot_comparison(
    all_results: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
    protein: str,
):
    """
    Plot comparison of 488nm intensities at protein positions vs. background.

    Args:
        all_results: Nested dict {movie_name: {"protein_positions": array, "background": array}}
        output_dir: Base output directory.
        protein: Protein name.
    """
    # Aggregate all data
    all_protein_intensities = []
    all_background_intensities = []

    for movie_results in all_results.values():
        all_protein_intensities.extend(movie_results["protein_positions"])
        all_background_intensities.extend(movie_results["background"])

    all_protein_intensities = np.array(all_protein_intensities)
    all_background_intensities = np.array(all_background_intensities)

    # Check if we have enough data
    if len(all_protein_intensities) == 0:
        logging.warning(f"No protein position data available for {protein} (all positions overlapped with GT)")
        return

    if len(all_background_intensities) == 0:
        logging.warning(f"No background data available for {protein}")
        return

    # Create output directory with descriptive name
    plot_dir = output_dir / f"{protein}_protein_gt_comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overlaid histograms (top-left) - Panel A
    ax = axes[0, 0]
    ax.hist(all_protein_intensities, bins=50, alpha=0.6, color=COLORS['vermillion'],
            label=f'Protein positions (n={len(all_protein_intensities)})', edgecolor='none')
    ax.hist(all_background_intensities, bins=50, alpha=0.6, color=COLORS['blue'],
            label=f'Background (n={len(all_background_intensities)})', edgecolor='none')
    ax.set_xlabel('488nm integrated intensity', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Count', fontsize=FONTSIZE_LABEL)
    ax.set_title('Intensity distributions', fontsize=FONTSIZE_TITLE, pad=9)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False)
    apply_axis_standards(ax)

    # 2. Box plot comparison (top-right) - Panel B
    ax = axes[0, 1]
    data_to_plot = [all_protein_intensities, all_background_intensities]
    bp = ax.boxplot(data_to_plot, labels=['Protein\npositions', 'Background'],
                    patch_artist=True, showmeans=True,
                    boxprops=dict(facecolor=COLORS['sky_blue'], alpha=0.6, edgecolor=COLORS['black']),
                    medianprops=dict(color=COLORS['vermillion'], linewidth=1.5),
                    meanprops=dict(marker='o', markerfacecolor=COLORS['black'],
                                  markeredgecolor=COLORS['black'], markersize=4),
                    whiskerprops=dict(color=COLORS['black']),
                    capprops=dict(color=COLORS['black']))
    ax.set_ylabel('488nm integrated intensity', fontsize=FONTSIZE_LABEL)
    ax.set_title('Distribution comparison', fontsize=FONTSIZE_TITLE, pad=9)
    apply_axis_standards(ax)

    # 3. Log-scale histograms (bottom-left) - Panel C
    ax = axes[1, 0]
    if len(all_protein_intensities) > 10 and len(all_background_intensities) > 10:
        min_val = max(1, min(all_protein_intensities.min(), all_background_intensities.min()))
        max_val = max(all_protein_intensities.max(), all_background_intensities.max())
        log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
        ax.hist(all_protein_intensities, bins=log_bins, alpha=0.6, color=COLORS['vermillion'],
                label='Protein positions', edgecolor='none')
        ax.hist(all_background_intensities, bins=log_bins, alpha=0.6, color=COLORS['blue'],
                label='Background', edgecolor='none')
    ax.set_xscale('log')
    ax.set_xlabel('488nm integrated intensity (log scale)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Count', fontsize=FONTSIZE_LABEL)
    ax.set_title('Log-scale distributions', fontsize=FONTSIZE_TITLE, pad=9)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False)
    apply_axis_standards(ax)

    # 4. Statistics table (bottom-right) - Panel D
    ax = axes[1, 1]
    ax.axis('off')

    # Perform t-test (two-sided, independent samples)
    t_statistic, p_value = stats.ttest_ind(all_protein_intensities, all_background_intensities)

    # Format p-value
    if p_value < 0.001:
        p_value_str = "< 0.001"
    else:
        p_value_str = f"{p_value:.3f}"

    # Determine significance
    if p_value < 0.05:
        significance = "Yes (p < 0.05)"
    else:
        significance = "No (p ≥ 0.05)"

    # Calculate statistics
    stats_data = [
        ['Metric', 'Protein positions', 'Background', 'Ratio'],
        ['Mean', f'{np.mean(all_protein_intensities):.0f}',
         f'{np.mean(all_background_intensities):.0f}',
         f'{np.mean(all_protein_intensities) / np.mean(all_background_intensities):.2f}'],
        ['Median', f'{np.median(all_protein_intensities):.0f}',
         f'{np.median(all_background_intensities):.0f}',
         f'{np.median(all_protein_intensities) / np.median(all_background_intensities):.2f}'],
        ['Std dev', f'{np.std(all_protein_intensities):.0f}',
         f'{np.std(all_background_intensities):.0f}', '—'],
        ['P10', f'{np.percentile(all_protein_intensities, 10):.0f}',
         f'{np.percentile(all_background_intensities, 10):.0f}', '—'],
        ['P90', f'{np.percentile(all_protein_intensities, 90):.0f}',
         f'{np.percentile(all_background_intensities, 90):.0f}', '—'],
        ['N', f'{len(all_protein_intensities)}',
         f'{len(all_background_intensities)}', '—'],
        ['', '', '', ''],  # Empty row for spacing
        ['t-test p-value', p_value_str, '', ''],
        ['Significant?', significance, '', ''],
    ]

    table = ax.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Statistical summary', fontsize=FONTSIZE_TITLE, pad=20)

    # Save figure
    save_figure(plt.gcf(), plot_dir)
    plt.close()

    # Save source data for each panel
    # Panel A & C: Histogram data (same data, just different bins)
    df_panel_a = pd.DataFrame({
        'protein_positions': all_protein_intensities,
        'background': np.pad(all_background_intensities,
                            (0, len(all_protein_intensities) - len(all_background_intensities)),
                            constant_values=np.nan) if len(all_background_intensities) < len(all_protein_intensities)
                       else all_background_intensities[:len(all_protein_intensities)]
    })
    df_panel_a.to_csv(plot_dir / 'data_panel_A.csv', index=False)
    df_panel_a.to_csv(plot_dir / 'data_panel_C.csv', index=False)  # Same data for log-scale

    # Panel B: Box plot data (same as Panel A, just different representation)
    df_panel_b = df_panel_a.copy()
    df_panel_b.to_csv(plot_dir / 'data_panel_B.csv', index=False)

    # Panel D: Statistics summary
    df_panel_d = pd.DataFrame(stats_data[1:], columns=stats_data[0])
    df_panel_d.to_csv(plot_dir / 'data_panel_D.csv', index=False)

    logging.info(f"Saved comparison plot and data to: {plot_dir}")


def save_summary_csv(
    all_results: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
):
    """
    Save summary statistics to CSV.

    Args:
        all_results: Nested dict {movie_name: {"protein_positions": array, "background": array}}
        output_path: Path to save CSV.
    """
    # Aggregate data
    all_protein = []
    all_background = []
    for results in all_results.values():
        all_protein.extend(results["protein_positions"])
        all_background.extend(results["background"])

    all_protein = np.array(all_protein)
    all_background = np.array(all_background)

    # Check if we have enough data
    if len(all_protein) == 0 or len(all_background) == 0:
        logging.warning("Insufficient data to generate summary CSV (empty arrays)")
        return

    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(all_protein, all_background)

    # Calculate statistics
    summary = {
        'condition': ['Protein_Positions', 'Background'],
        'n': [len(all_protein), len(all_background)],
        'mean': [np.mean(all_protein), np.mean(all_background)],
        'median': [np.median(all_protein), np.median(all_background)],
        'std': [np.std(all_protein), np.std(all_background)],
        'min': [np.min(all_protein), np.min(all_background)],
        'max': [np.max(all_protein), np.max(all_background)],
        'p10': [np.percentile(all_protein, 10), np.percentile(all_background, 10)],
        'p25': [np.percentile(all_protein, 25), np.percentile(all_background, 25)],
        'p75': [np.percentile(all_protein, 75), np.percentile(all_background, 75)],
        'p90': [np.percentile(all_protein, 90), np.percentile(all_background, 90)],
    }

    df = pd.DataFrame(summary)

    # Add t-test results as a comment in the CSV
    with open(output_path, 'w') as f:
        f.write(f"# t-test: t-statistic={t_statistic:.4f}, p-value={p_value:.6f}\n")
        df.to_csv(f, index=False)

    logging.info(f"Saved summary CSV: {output_path}")
    logging.info(f"  t-test p-value: {p_value:.6f} (t={t_statistic:.4f})")


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze protein-to-GT correlation"
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
        default=None,
        help="Analyze specific protein only (optional)",
    )
    parser.add_argument(
        "--protein-gradient",
        type=int,
        default=60000,
        help="Gradient for 640nm protein channel (default: 60000)",
    )
    parser.add_argument(
        "--gt-gradient",
        type=int,
        default=5000,
        help="Gradient for 488nm GT channel (default: 5000)",
    )
    args = parser.parse_args()

    # Setup
    setup_logging("INFO")
    cfg = load_config(Path(args.config_path))

    # Set defaults if not in config
    if "n_frames_integrate" not in cfg:
        cfg["n_frames_integrate"] = 10
        logging.info("n_frames_integrate not in config, using default: 10")
    if "boxsize" not in cfg:
        cfg["boxsize"] = 9
        logging.info("boxsize not in config, using default: 9")

    # Get Picasso Python
    picasso_python = get_picasso_python()

    # Get proteins
    proteins = cfg["proteins"]
    if args.protein:
        if args.protein in proteins:
            proteins = [args.protein]
            logging.info(f"Filtering to protein: {args.protein}")
        else:
            logging.error(f"Protein '{args.protein}' not found in config")
            sys.exit(1)

    logging.info("=" * 60)
    logging.info("Protein-to-Ground Truth Correlation Analysis")
    logging.info("=" * 60)
    logging.info(f"Proteins: {proteins}")
    logging.info(f"Protein channel (640nm) gradient: {args.protein_gradient}")
    logging.info(f"GT channel (488nm) gradient: {args.gt_gradient}")
    logging.info(f"Integration frames: {cfg['n_frames_integrate']}")
    logging.info("")

    # Discover data
    input_root = Path(cfg["input_folder"]).expanduser()
    protein_data = discover_all_data(
        input_root,
        proteins,
        protein_channel=cfg["protein_channel"],
        ground_truth_channel=cfg["ground_truth_channel"],
    )

    # Process each protein
    for protein in proteins:
        if protein not in protein_data or not protein_data[protein]:
            logging.warning(f"No data found for {protein}")
            continue

        logging.info("")
        logging.info("=" * 60)
        logging.info(f"Analyzing {protein}")
        logging.info("=" * 60)

        # Create output directories
        output_base = Path(cfg["output_folder_base"]).expanduser() / f"{protein}_protein_gt_correlation"
        locs_dir = output_base / "localizations"
        data_dir = output_base / "data"
        plots_dir = output_base / "plots"

        for d in [locs_dir, data_dir, plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Get number of workers
        n_workers_config = cfg.get("n_workers", -1)
        num_workers = get_num_workers(n_workers_config)

        # Process all movie pairs
        all_results = {}

        if num_workers == 1:
            # Sequential processing
            logging.info(f"  Sequential processing of {len(protein_data[protein])} movies")
            for protein_file, gt_file in protein_data[protein]:
                try:
                    movie_results = analyze_movie_pair(
                        protein_file,
                        gt_file,
                        cfg,
                        picasso_python,
                        locs_dir,
                        args.protein_gradient,
                        args.gt_gradient,
                    )
                    all_results[protein_file.name] = movie_results
                except Exception as e:
                    logging.error(f"  Failed to analyze {protein_file.name}: {e}")
                    continue
        else:
            # Parallel processing
            import multiprocessing as mp
            logging.info(f"  Parallel processing with {num_workers} workers")

            # Prepare work items
            work_items = [
                (protein_file, gt_file, cfg, picasso_python, locs_dir,
                 args.protein_gradient, args.gt_gradient)
                for protein_file, gt_file in protein_data[protein]
            ]

            # Process in parallel
            with mp.Pool(processes=num_workers) as pool:
                results_list = pool.starmap(analyze_movie_pair_worker, work_items)

            # Aggregate results
            for movie_name, movie_results in results_list:
                if movie_results is not None:
                    all_results[movie_name] = movie_results

        if not all_results:
            logging.warning(f"No results for {protein}")
            continue

        # Generate comparison plot
        plot_comparison(all_results, plots_dir, protein)

        # Save summary CSV
        csv_path = data_dir / f"{protein}_correlation_summary.csv"
        save_summary_csv(all_results, csv_path)

        logging.info("")
        logging.info(f"Results saved to: {output_base}")

    logging.info("")
    logging.info("=" * 60)
    logging.info("Analysis complete!")
    logging.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        logging.error(f"Fatal error: {ex}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
