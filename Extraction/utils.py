#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH), Andreas Biri (ETHZ).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Utility functions for trace extraction pipeline.

This module contains all helper functions for:
- Logging and setup
- Picasso installation detection
- Data discovery (experiment/protein/file organization)
- Picasso processing (localization, clustering, trace extraction)
- Parallel processing helpers
"""
from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Set matplotlib to non-interactive backend for thread safety in parallel processing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches

# Plotting standards
_FONTSIZE_LABEL  = 7
_FONTSIZE_TICK   = 6
_FONTSIZE_TITLE  = 7
_FONTSIZE_LEGEND = 6
_COLORS = {
    'orange':    '#E69F00',
    'sky_blue':  '#56B4E9',
    'green':     '#009E73',
    'yellow':    '#F0E442',
    'blue':      '#0072B2',
    'vermillion':'#D55E00',
    'pink':      '#CC79A7',
    'black':     '#000000',
}
matplotlib.rcParams['font.family']     = 'sans-serif'
# DejaVu Sans is bundled with every matplotlib installation; listing it first
# ensures identical font metrics (and therefore identical tight_layout margins)
# on all platforms.  Helvetica/Arial are kept as fallbacks for systems that
# have them and want them for other purposes, but will not be used here.
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
matplotlib.rcParams['font.size']       = _FONTSIZE_LABEL
matplotlib.rcParams['axes.titlesize']  = _FONTSIZE_TITLE
matplotlib.rcParams['axes.labelsize']  = _FONTSIZE_LABEL
matplotlib.rcParams['xtick.labelsize'] = _FONTSIZE_TICK
matplotlib.rcParams['ytick.labelsize'] = _FONTSIZE_TICK
matplotlib.rcParams['legend.fontsize'] = _FONTSIZE_LEGEND
matplotlib.rcParams['figure.dpi']      = 100   # fixes tight_layout margin arithmetic
matplotlib.rcParams['savefig.dpi']     = 450
matplotlib.rcParams['savefig.bbox']    = 'tight'
matplotlib.rcParams['axes.grid']       = False
matplotlib.rcParams['pdf.fonttype']    = 42


# =============================================================================
# Logging and Setup
# =============================================================================


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with consistent formatting."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level.upper())
    sh.setFormatter(fmt)
    root.addHandler(sh)


# =============================================================================
# Localization Method
# =============================================================================


def get_localization_method() -> str:
    """
    Get the localization method for Picasso.

    Returns:
        'mle' (maximum likelihood estimation on CPU).
    """
    logging.info("Using CPU-based localization (mle)")
    return "mle"


def get_num_workers(n_workers: int | None) -> int:
    """
    Determine the number of worker processes to use for parallelization.

    Args:
        n_workers: Number of workers from config.
                   None or -1 means use all CPUs.
                   1 means sequential processing.
                   N > 1 means use N workers.

    Returns:
        Number of worker processes to use.
    """
    import multiprocessing as mp

    if n_workers is None or n_workers == -1:
        # Use all available CPUs
        num_cpus = mp.cpu_count()
        logging.info(f"Using all available CPUs: {num_cpus} workers")
        return num_cpus
    elif n_workers == 1:
        logging.info("Sequential processing (no parallelization)")
        return 1
    elif n_workers > 1:
        num_cpus = mp.cpu_count()
        if n_workers > num_cpus:
            logging.warning(
                f"Requested {n_workers} workers but only {num_cpus} CPUs available. "
                f"Using {num_cpus} workers."
            )
            return num_cpus
        logging.info(f"Using {n_workers} parallel workers")
        return n_workers
    else:
        logging.warning(f"Invalid n_workers value: {n_workers}. Using 1 worker.")
        return 1


def get_picasso_python() -> str:
    """
    Get the path to the Python interpreter that has Picasso installed.

    Checks:
    1. Current environment
    2. picasso-env conda/mamba environment
    3. System python

    Returns:
        Path to Python executable with Picasso.

    Raises:
        RuntimeError if Picasso is not found.
    """
    # Try current Python first
    candidates = [
        sys.executable,
        # Mamba/Miniforge environments
        os.path.expanduser("~/mambaforge/envs/picasso-env/bin/python"),
        os.path.expanduser("~/miniforge3/envs/picasso-env/bin/python"),
        # Conda/Anaconda environments
        os.path.expanduser("~/opt/anaconda3/envs/picasso-env/bin/python"),
        os.path.expanduser("~/anaconda3/envs/picasso-env/bin/python"),
        os.path.expanduser("~/miniconda3/envs/picasso-env/bin/python"),
        "/opt/anaconda3/envs/picasso-env/bin/python",
        "/usr/local/anaconda3/envs/picasso-env/bin/python",
        # System Python
        "python3",
        "python",
    ]

    for python_exe in candidates:
        if (
            not python_exe
            or not os.path.exists(python_exe)
            if os.path.isabs(python_exe)
            else False
        ):
            continue

        try:
            # Test if picasso is available
            result = subprocess.run(
                [python_exe, "-m", "picasso", "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            if result.returncode == 0:
                logging.info(f"Found Picasso in: {python_exe}")
                return python_exe
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    raise RuntimeError(
        "Picasso not found. Please install with: pip install picassosr\n"
        "Or create picasso-env with: mamba env create -f picasso-env.yaml"
    )


def get_fov_size(nd2_file: Path) -> int:
    """
    Automatically detect FOV size from ND2 file.

    Reads the image dimensions from an ND2 file and returns the FOV size
    (assumes square FOV, returns width).

    Args:
        nd2_file: Path to ND2 file.

    Returns:
        FOV size in pixels (width of the image).

    Raises:
        RuntimeError if file cannot be read or dimensions cannot be determined.
    """
    try:
        import nd2

        # Read just the metadata (don't load the full movie)
        with nd2.ND2File(nd2_file) as f:
            # Get image shape: typically (frames, height, width) or (height, width)
            shape = f.shape

            # Handle different shape formats
            if len(shape) >= 2:
                # Get height and width (last two dimensions)
                height, width = shape[-2], shape[-1]

                # Sanity check: verify it's approximately square
                if abs(height - width) > 10:
                    logging.warning(
                        f"FOV is not square: {width}x{height} pixels. Using width: {width}"
                    )

                logging.info(f"Detected FOV size: {width}x{height} pixels")
                return int(width)
            else:
                raise RuntimeError(f"Unexpected image shape: {shape}")

    except ImportError:
        raise RuntimeError(
            "nd2 package not found. Install with: pip install nd2"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to read FOV size from {nd2_file}: {e}"
        )


# =============================================================================
# Legacy File Discovery (for backward compatibility)
# =============================================================================


def get_path_list(root_folder: str, dirs_not_to_visit: Sequence[str]) -> list[str]:
    """
    Search all ND2 files inside a folder, skipping specified directories.

    Note: This is legacy code kept for backward compatibility.
    New code should use discover_all_data() instead.

    Args:
        root_folder: Root directory to search.
        dirs_not_to_visit: List of directory names to skip.

    Returns:
        List of ND2 file paths.
    """
    path_list = []
    dir_path = os.path.join(root_folder, "**", "*.nd2")

    for file_path in glob.glob(dir_path, recursive=True):
        # Skip macOS metadata files (._filename)
        if os.path.basename(file_path).startswith("._"):
            continue
        # Check if any excluded directory is in the path
        if any(excluded in file_path for excluded in dirs_not_to_visit):
            continue
        path_list.append(file_path)

    return path_list


def rel_under(root: Path, path: Path) -> Path:
    """
    Get relative path of 'path' under 'root'.

    Args:
        root: Root directory.
        path: Path to make relative (can be file or directory).

    Returns:
        Relative path from root to path.
    """
    root = root.resolve()
    path = path.resolve()
    if path.is_file():
        path = path.parent
    return Path(os.path.relpath(path, root))


# =============================================================================
# Structured Data Discovery
# =============================================================================


def discover_experiments(input_folder: Path) -> List[Path]:
    """
    Discover all experiment folders in input_folder.

    Expected structure:
        input_folder/
        ├── Exp1/
        ├── Exp2/
        └── ...

    Args:
        input_folder: Root directory containing experiments.

    Returns:
        List of experiment directory paths.
    """
    experiments = []

    if not input_folder.exists():
        logging.warning(f"Input folder does not exist: {input_folder}")
        return experiments

    for item in input_folder.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            experiments.append(item)

    return sorted(experiments)


def discover_protein_folders(
    experiment_folder: Path, proteins: List[str]
) -> Dict[str, Path]:
    """
    Find protein folders within an experiment.

    Expected structure:
        experiment_folder/
        ├── Grx1/
        ├── K20Ac/
        └── ...

    Args:
        experiment_folder: Path to experiment directory.
        proteins: List of protein names to look for.

    Returns:
        Dictionary mapping protein name to folder path.
    """
    protein_folders = {}

    for protein in proteins:
        protein_path = experiment_folder / protein
        if protein_path.exists() and protein_path.is_dir():
            protein_folders[protein] = protein_path
        else:
            logging.debug(
                f"Protein folder not found: {protein} in {experiment_folder.name}"
            )

    return protein_folders


def find_nd2_files(folder: Path) -> List[Path]:
    """
    Find all ND2 files in a folder (non-recursive).

    Args:
        folder: Directory to search.

    Returns:
        List of ND2 file paths.
    """
    nd2_files = []

    for item in folder.iterdir():
        # Skip macOS metadata files (._filename)
        if item.name.startswith("._"):
            continue
        if item.is_file() and item.suffix.lower() == ".nd2":
            nd2_files.append(item)

    return sorted(nd2_files)


def pair_channel_files(
    nd2_files: List[Path],
    protein_channel: str | None = "640",
    ground_truth_channel: str | None = "488",
) -> List[Tuple[Path, Path | None]]:
    """
    Pair protein and ground truth channel files.

    Matching strategy:
    1. Extract protein name (prefix before first underscore)
    2. Extract file number (e.g., _0001.nd2) if present, or use "NONE" for unnumbered files
    3. Match files with same protein and file number (or both unnumbered)

    Special modes:
    - If ground_truth_channel is None: protein-only mode, returns list of (protein_file, None) tuples
    - If protein_channel is None: GT-only mode, returns list of (None, gt_file) tuples

    Args:
        nd2_files: List of ND2 files to pair.
        protein_channel: Token identifying protein channel (default: "640"). None for GT-only mode.
        ground_truth_channel: Token identifying ground truth channel (default: "488"). None for protein-only mode.

    Returns:
        List of (protein_file, ground_truth_file) tuples.
    """
    # GT-only mode: return GT files paired with None
    if protein_channel is None:
        if ground_truth_channel is None:
            logging.error("Both protein_channel and ground_truth_channel cannot be None")
            return []

        ground_truth_files = [f for f in nd2_files if ground_truth_channel in f.name]

        if not ground_truth_files:
            logging.warning(f"No {ground_truth_channel}nm files found")
            return []

        return [(None, gtf) for gtf in ground_truth_files]

    # Separate by channel
    protein_files = [f for f in nd2_files if protein_channel in f.name]

    if not protein_files:
        logging.warning(f"No {protein_channel}nm files found")
        return []

    # Protein-only mode: return protein files paired with None
    if ground_truth_channel is None:
        return [(pf, None) for pf in protein_files]

    ground_truth_files = [f for f in nd2_files if ground_truth_channel in f.name]

    if not ground_truth_files:
        logging.warning(f"No {ground_truth_channel}nm files found")
        return []

    # Pattern to extract file number: Protein_...anything..._XXXX.nd2
    number_pattern = re.compile(r"_(\d{4})\.nd2$")

    pairs = []

    # Create mapping of (protein, file_number) to ground truth files
    ground_truth_map = {}
    for gtf in ground_truth_files:
        # Extract protein name (everything before first underscore)
        protein = gtf.name.split("_")[0]

        # Extract file number (use "NONE" for files without sequential numbers)
        m = number_pattern.search(gtf.name)
        file_num = m.group(1) if m else "NONE"
        ground_truth_map[(protein, file_num)] = gtf

    # Match protein files to ground truth files
    for pf in protein_files:
        protein = pf.name.split("_")[0]
        m = number_pattern.search(pf.name)
        file_num = m.group(1) if m else "NONE"
        key = (protein, file_num)

        if key in ground_truth_map:
            pairs.append((pf, ground_truth_map[key]))
        else:
            logging.warning(
                f"No matching ground truth file for {pf.name} (protein={protein}, num={file_num})"
            )

    return pairs


def discover_all_data(
    input_folder: Path,
    proteins: List[str],
    protein_channel: str = "640",
    ground_truth_channel: str | None = "488",
) -> Dict[str, List[Tuple[Path, Path | None]]]:
    """
    Discover all experiment/protein data with structured organization.

    Expected structure:
        input_folder/
        ├── Exp1/
        │   ├── Grx1/
        │   │   ├── Grx1_640nm_..._0001.nd2
        │   │   ├── Grx1_488nm_..._0001.nd2  (optional, if has ground truth)
        │   │   └── ...
        │   └── K20Ac/
        │       └── ...
        └── Exp2/
            └── ...

    Args:
        input_folder: Root directory containing experiments.
        proteins: List of protein names to process.
        protein_channel: Token for protein channel (default: "640").
        ground_truth_channel: Token for ground truth channel (default: "488"). None for single-channel mode.

    Returns:
        Dictionary mapping protein name to list of (protein_file, ground_truth_file) pairs.
        ground_truth_file is None if single-channel mode.
        Each pair includes the full path from any experiment.
    """
    protein_data = {protein: [] for protein in proteins}

    # Discover experiments
    experiments = discover_experiments(input_folder)

    if not experiments:
        logging.warning(f"No experiment folders found in {input_folder}")
        return protein_data

    logging.info(f"Found {len(experiments)} experiments: {[e.name for e in experiments]}")

    # Process each experiment
    for exp in experiments:
        logging.info(f"Processing experiment: {exp.name}")

        # Find protein folders in this experiment
        protein_folders = discover_protein_folders(exp, proteins)

        if not protein_folders:
            logging.warning(f"  No protein folders found in {exp.name}")
            continue

        logging.info(f"  Found proteins: {list(protein_folders.keys())}")

        # Process each protein
        for protein, protein_path in protein_folders.items():
            # Find ND2 files
            nd2_files = find_nd2_files(protein_path)

            if not nd2_files:
                logging.warning(f"    {protein}: No ND2 files found")
                continue

            # Pair channels
            pairs = pair_channel_files(nd2_files, protein_channel, ground_truth_channel)

            if pairs:
                protein_data[protein].extend(pairs)
                if ground_truth_channel is None:
                    logging.info(f"    {protein}: Found {len(pairs)} movies")
                else:
                    logging.info(f"    {protein}: Found {len(pairs)} movie pairs")
            else:
                logging.warning(f"    {protein}: Could not find/pair files")

    return protein_data


def print_discovery_summary(protein_data: Dict[str, List[Tuple[Path, Path | None]]]) -> None:
    """
    Print a summary of discovered data.

    Args:
        protein_data: Output from discover_all_data.
    """
    logging.info("=" * 60)
    logging.info("Data Discovery Summary")
    logging.info("=" * 60)

    total_items = 0
    has_ground_truth = False

    for protein, items in protein_data.items():
        if items:
            # Check if we have ground truth (any non-None second element)
            if any(gt_file is not None for _, gt_file in items):
                has_ground_truth = True
                logging.info(f"{protein}: {len(items)} movie pairs")
            else:
                logging.info(f"{protein}: {len(items)} movies")
            total_items += len(items)
        else:
            logging.info(f"{protein}: No data found")

    if has_ground_truth:
        logging.info(f"\nTotal: {total_items} movie pairs across all proteins")
    else:
        logging.info(f"\nTotal: {total_items} movies across all proteins")
    logging.info("=" * 60)


# =============================================================================
# Picasso Localization Processing
# =============================================================================


def get_and_link_locs(
    loc_path: str,
    box_size: int,
    max_distance: float,
    max_off_time: int,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", int]:
    """
    Load and link localizations across frames for protein signals.

    Args:
        loc_path: Path to HDF5 localization file.
        box_size: Box size for ROI.
        max_distance: Maximum distance (pixels) to link localizations.
        max_off_time: Maximum dark time (frames) for linking.

    Returns:
        Tuple of (x_pix, y_pix, x_pix_transl, y_pix_transl, number_rois)
    """
    # Lazy import of Picasso
    import numpy as np
    from picasso import io as pio, lib, postprocess

    # Load localizations
    locs, info = pio.load_locs(loc_path)

    # Filter invalid values
    locs = lib.ensure_sanity(locs, info)

    # Link localizations across frames
    linked_locs = postprocess.link(
        locs, info, r_max=max_distance, max_dark_time=max_off_time
    )

    # Get coordinates
    x_pix = linked_locs["x"]
    y_pix = linked_locs["y"]

    # Translate to box coordinates (top-left corner)
    # Use round (not floor) so the box is centred as closely as possible on the
    # sub-pixel localisation; floor can shift the centre by up to ~1.5 pixels.
    shift = box_size / 2
    x_pix_transl = x_pix - shift
    y_pix_transl = y_pix - shift

    # Add dummy 0 for indexing (original code convention)
    x_pix_transl = np.concatenate([[0], x_pix_transl])
    y_pix_transl = np.concatenate([[0], y_pix_transl])

    # Filter out any remaining NaN/inf values before casting
    valid_mask = np.isfinite(x_pix_transl) & np.isfinite(y_pix_transl)
    if not valid_mask.all():
        # Replace invalid values with 0 (will be handled by downstream filtering)
        x_pix_transl = np.where(valid_mask, x_pix_transl, 0)
        y_pix_transl = np.where(valid_mask, y_pix_transl, 0)

    # Convert to integers
    x_pix_transl = np.round(x_pix_transl).astype(int)
    y_pix_transl = np.round(y_pix_transl).astype(int)

    number_rois = len(x_pix_transl)

    return x_pix, y_pix, x_pix_transl, y_pix_transl, number_rois


def get_and_cluster_locs(
    loc_path: str,
    box_size: int,
    max_distance: float,
    min_locs: int,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", int, int]:
    """
    Load and cluster localizations for vesicle detection.

    Args:
        loc_path: Path to HDF5 localization file.
        box_size: Box size for ROI.
        max_distance: Clustering radius (pixels).
        min_locs: Minimum localizations per cluster.

    Returns:
        Tuple of (x_pix, y_pix, x_pix_transl, y_pix_transl, number_rois, cluster_count)
    """
    # Lazy import of Picasso
    import numpy as np
    from picasso import clusterer, io as pio, lib

    # Load localizations
    locs, info = pio.load_locs(loc_path)

    # Filter invalid values
    locs = lib.ensure_sanity(locs, info)

    # Cluster localizations (pixelsize=130nm hardcoded as in original)
    cluster_locs = clusterer.cluster(
        locs,
        radius_xy=max_distance,
        min_locs=min_locs,
        frame_analysis=False,
        radius_z=None,
        pixelsize=130,
    )
    cluster_locs = clusterer.find_cluster_centers(cluster_locs, pixelsize=None)

    cluster_count = len(cluster_locs["x"])

    # Get coordinates
    x_pix = cluster_locs["x"]
    y_pix = cluster_locs["y"]

    # Translate to box coordinates
    # Use round (not floor) so the box is centred as closely as possible on the
    # sub-pixel localisation; floor can shift the centre by up to ~1.5 pixels.
    shift = box_size / 2
    x_pix_transl = x_pix - shift
    y_pix_transl = y_pix - shift

    # Add dummy 0 for indexing
    x_pix_transl = np.concatenate([[0], x_pix_transl])
    y_pix_transl = np.concatenate([[0], y_pix_transl])

    # Filter out any remaining NaN/inf values before casting
    valid_mask = np.isfinite(x_pix_transl) & np.isfinite(y_pix_transl)
    if not valid_mask.all():
        # Replace invalid values with 0 (will be handled by downstream filtering)
        x_pix_transl = np.where(valid_mask, x_pix_transl, 0)
        y_pix_transl = np.where(valid_mask, y_pix_transl, 0)

    # Convert to integers
    x_pix_transl = np.round(x_pix_transl).astype(int)
    y_pix_transl = np.round(y_pix_transl).astype(int)

    number_rois = len(x_pix_transl)

    return x_pix, y_pix, x_pix_transl, y_pix_transl, number_rois, cluster_count


def get_overlapping_rois(
    x: "np.ndarray",
    y: "np.ndarray",
    binary: "np.ndarray",
    labels: "np.ndarray",
    box_size: int,
    overlap_threshold: int,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """
    Identify overlapping ROIs.

    Args:
        x: X coordinates (top-left corner).
        y: Y coordinates (top-left corner).
        binary: Binary mask (modified in place).
        labels: Label mask (modified in place).
        box_size: ROI box size.
        overlap_threshold: Minimum overlapping pixels to flag.

    Returns:
        Tuple of (overlapping_roi_indices, binary_mask, label_mask)
    """
    import numpy as np

    number_rois = len(x)
    overlap_rois = []

    for i in range(number_rois):
        # Get current ROI region
        curr_mask = binary[y[i] : (y[i] + box_size), x[i] : (x[i] + box_size)]
        curr_roi = labels[y[i] : (y[i] + box_size), x[i] : (x[i] + box_size)]
        curr_mask = curr_mask.astype(int)

        # Find overlapping labels
        overlapping_labels = np.extract(curr_mask, curr_roi)
        overlap_unique, counts = np.unique(overlapping_labels, return_counts=True)

        if overlapping_labels.any() != 0:
            for j in range(len(counts)):
                if counts[j] > overlap_threshold:
                    overlap_rois.append(int(overlap_unique[j]))
                    overlap_rois.append(int(i))

        # Mark this ROI in masks
        binary[y[i] : (y[i] + box_size), x[i] : (x[i] + box_size)] = 1
        labels[y[i] : (y[i] + box_size), x[i] : (x[i] + box_size)] = i

    # Get unique overlapping ROI indices
    overlap_final = np.unique(np.array(overlap_rois))

    return overlap_final, binary, labels


def get_traces_new_incl_ov(
    movie: "np.ndarray",
    not_overlapping_rois: list,
    overlapping_rois: list,
    x_coord: "np.ndarray",
    y_coord: "np.ndarray",
    box_side_length: int,
) -> "pd.DataFrame":
    """
    Extract intensity traces from movie for both overlapping and non-overlapping ROIs.

    Args:
        movie: Movie array (frames, height, width).
        not_overlapping_rois: List of non-overlapping ROI indices.
        overlapping_rois: List of overlapping ROI indices.
        x_coord: X coordinates (top-left corner).
        y_coord: Y coordinates (top-left corner).
        box_side_length: ROI box size.

    Returns:
        DataFrame with columns: [x, y, trace, overlap]
    """
    import numpy as np
    import pandas as pd

    particle_dict = {
        "x": [],
        "y": [],
        "trace": [],
        "overlap": [],
    }

    # Extract traces from non-overlapping ROIs
    for roi in not_overlapping_rois:
        y_end = y_coord[roi] + box_side_length
        x_end = x_coord[roi] + box_side_length

        # Sum intensity over box for each frame
        trace = np.sum(
            np.sum(
                movie[:, y_coord[roi] : y_end, x_coord[roi] : x_end],
                axis=1,
            ),
            axis=1,
        )

        particle_dict["x"].append(x_coord[roi])
        particle_dict["y"].append(y_coord[roi])
        particle_dict["trace"].append(trace)
        particle_dict["overlap"].append("single")

    # Mark overlapping ROIs but don't extract traces (save space)
    for roi in overlapping_rois:
        particle_dict["x"].append(x_coord[roi])
        particle_dict["y"].append(y_coord[roi])
        particle_dict["trace"].append([])  # Empty trace
        particle_dict["overlap"].append("overlapping")

    return pd.DataFrame(particle_dict)


def sample_background_positions(
    binary_mask: "np.ndarray",
    boxsize: int,
    n_samples: int,
    min_edge_distance: int = 5,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Sample random background ROI positions from unoccupied pixels.

    Args:
        binary_mask: Binary mask where 1 = occupied by protein ROI.
        boxsize: ROI box size.
        n_samples: Number of background positions to sample.
        min_edge_distance: Minimum distance from image edges.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (x_positions, y_positions) as lists of top-left coordinates.
    """
    import numpy as np

    np.random.seed(seed)

    height, width = binary_mask.shape

    # Create mask of valid positions (not occupied and not near edges)
    # A position is valid if the entire box fits and doesn't overlap with any ROI
    valid_mask = np.zeros((height, width), dtype=bool)

    # For each potential top-left position, check if box is valid
    for y in range(min_edge_distance, height - boxsize - min_edge_distance):
        for x in range(min_edge_distance, width - boxsize - min_edge_distance):
            # Check if this box region is entirely unoccupied
            box_region = binary_mask[y:y+boxsize, x:x+boxsize]
            if box_region.sum() == 0:  # No overlap with protein ROIs
                valid_mask[y, x] = True

    # Get all valid positions
    valid_y, valid_x = np.where(valid_mask)

    if len(valid_x) == 0:
        logging.warning("No valid background positions found")
        return [], []

    # Sample positions (with minimum spacing to avoid overlapping background ROIs)
    sampled_x = []
    sampled_y = []

    # Create a temporary mask to track sampled background ROIs
    sampled_mask = np.zeros_like(binary_mask)

    # Shuffle valid positions
    indices = np.random.permutation(len(valid_x))

    for idx in indices:
        if len(sampled_x) >= n_samples:
            break

        x, y = valid_x[idx], valid_y[idx]

        # Check if this position overlaps with already sampled background ROIs
        box_region = sampled_mask[y:y+boxsize, x:x+boxsize]
        if box_region.sum() == 0:
            sampled_x.append(x)
            sampled_y.append(y)
            sampled_mask[y:y+boxsize, x:x+boxsize] = 1

    if len(sampled_x) < n_samples:
        logging.warning(f"Only found {len(sampled_x)} valid background positions (requested {n_samples})")

    return sampled_x, sampled_y


def extract_background_traces(
    movie: "np.ndarray",
    binary_mask: "np.ndarray",
    boxsize: int,
    n_traces: int = 100,
    min_edge_distance: int = 5,
    seed: int = 42,
) -> "pd.DataFrame":
    """
    Extract background intensity traces from regions without protein localizations.

    Args:
        movie: Movie array (frames, height, width).
        binary_mask: Binary mask where 1 = occupied by protein ROI.
        boxsize: ROI box size (same as protein ROIs).
        n_traces: Number of background traces to extract.
        min_edge_distance: Minimum distance from image edges.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: [x, y, trace, overlap, origin]
        where origin="background" for all traces.
    """
    import numpy as np
    import pandas as pd

    # Sample background positions
    bg_x, bg_y = sample_background_positions(
        binary_mask, boxsize, n_traces, min_edge_distance, seed
    )

    if len(bg_x) == 0:
        return pd.DataFrame(columns=["x", "y", "trace", "overlap", "origin"])

    # Extract traces
    particle_dict = {
        "x": [],
        "y": [],
        "trace": [],
        "overlap": [],
        "origin": [],
    }

    for x, y in zip(bg_x, bg_y):
        # Sum intensity over box for each frame (same as protein traces)
        trace = np.sum(
            np.sum(
                movie[:, y:y+boxsize, x:x+boxsize],
                axis=1,
            ),
            axis=1,
        )

        particle_dict["x"].append(x)
        particle_dict["y"].append(y)
        particle_dict["trace"].append(trace)
        particle_dict["overlap"].append("single")
        particle_dict["origin"].append("background")

    return pd.DataFrame(particle_dict)


def apply_background_spatial_buffer(
    bg_merged: "pd.DataFrame",
    buffer_px: float,
) -> Tuple["pd.DataFrame", dict]:
    """
    Remove background traces whose ROI center is within buffer_px pixels of any
    protein ROI from the same movie.

    Requires bg_merged to have columns: x, y, file_origin.
    The protein trace file for each movie is derived by replacing
    '_background_traces.pkl' with '_traces.pkl' in file_origin.
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import KDTree

    drop_indices: list = []

    for file_origin, grp in bg_merged.groupby("file_origin"):
        prot_file = str(file_origin).replace("_background_traces.pkl", "_traces.pkl")
        if not Path(prot_file).exists():
            logging.warning(f"  Protein trace file not found for spatial buffer: {prot_file}")
            continue

        prot_df = pd.read_pickle(prot_file)
        if len(prot_df) == 0:
            continue

        bg_xy   = np.column_stack([grp["x"].values, grp["y"].values])
        prot_xy = np.column_stack([prot_df["x"].values, prot_df["y"].values])
        tree = KDTree(prot_xy)
        dists, _ = tree.query(bg_xy)
        drop_indices.extend(grp.index[dists <= buffer_px].tolist())

    n_original = len(bg_merged)
    filtered = bg_merged.drop(index=drop_indices).reset_index(drop=True)
    n_removed = len(drop_indices)
    pct = 100 * n_removed / max(1, n_original)

    logging.info(f"  Spatial buffer ({buffer_px:.0f}px): "
                 f"removed {n_removed}/{n_original} background traces ({pct:.1f}%)")

    return filtered, {
        "n_original": n_original,
        "n_filtered": len(filtered),
        "n_removed": n_removed,
        "pct_removed": pct,
    }


def robust_background_filter(
    df_raw: "pd.DataFrame",
    rob_threshold: float,
) -> Tuple["pd.DataFrame", dict]:
    """
    Filter background traces using the robust outlier score: (max - median) / MAD.

    Applied to raw (non-normalized) traces. Catches both long binding events
    (which inflate the max) and brief spikes (which don't form a second mode
    but still push max well above the bulk).

    For Gaussian noise with n=6000 the expected score is ~5.6.
    Default threshold 8 rejects only genuine outlier traces.
    """
    import numpy as np

    traces = df_raw.values  # (timesteps, n_traces)
    n_traces = traces.shape[1]

    if n_traces == 0:
        return df_raw, {"n_original": 0, "n_filtered": 0, "n_removed": 0, "pct_removed": 0.0}

    meds = np.median(traces, axis=0)
    mads = np.median(np.abs(traces - meds), axis=0)
    mads = np.where(mads > 1e-10, mads, 1e-10)
    rob_scores = (traces.max(axis=0) - meds) / mads

    keep_mask = rob_scores <= rob_threshold
    kept_cols = [col for i, col in enumerate(df_raw.columns) if keep_mask[i]]
    filtered = df_raw[kept_cols]

    n_removed = int((~keep_mask).sum())
    pct = 100 * n_removed / n_traces

    logging.info(f"  Robust outlier filter (threshold={rob_threshold:.1f}): "
                 f"removed {n_removed}/{n_traces} traces ({pct:.1f}%)")

    return filtered, {
        "n_original": n_traces,
        "n_filtered": int(keep_mask.sum()),
        "n_removed": n_removed,
        "pct_removed": pct,
        "rob_scores": rob_scores,
        "rob_threshold": rob_threshold,
    }


# =============================================================================
# Visualization and QC Functions
# =============================================================================


def colorize(
    im: "np.ndarray",
    color: Tuple[float, float, float],
    clip_percentile: float = 0.1
) -> "np.ndarray":
    """
    Convert grayscale image to RGB with color mapping.

    Args:
        im: Grayscale image (2D or 3D with single channel).
        color: RGB color tuple (e.g., (1, 0, 0) for red).
        clip_percentile: Percentile for contrast clipping.

    Returns:
        RGB image (H, W, 3) with color mapping applied.
    """
    import numpy as np

    if im.ndim > 2 and im.shape[2] != 1:
        raise ValueError("Expect single-channel image")

    # Contrast adjustment
    im_scaled = im.astype(np.float32) - np.percentile(im, clip_percentile)
    denom = max(np.percentile(im_scaled, 100 - clip_percentile), 1e-9)
    im_scaled = np.clip(im_scaled / denom, 0, 1)

    # Apply color
    im_scaled = np.atleast_3d(im_scaled)
    color = np.asarray(color).reshape((1, 1, -1))

    return im_scaled * color


def draw_max_projection(
    x: "np.ndarray",
    y: "np.ndarray",
    boxsize: int,
    overlap: "np.ndarray",
    output_file: str,
    max_projection: "np.ndarray",
    bg_x: "np.ndarray | None" = None,
    bg_y: "np.ndarray | None" = None,
) -> None:
    """
    Draw max projection image with ROI overlays.

    Creates a figure showing:
    - Grayscale max projection
    - Green boxes for single (non-overlapping) protein ROIs
    - Vermillion boxes for overlapping protein ROIs
    - Sky blue boxes for background ROIs (if provided)

    Args:
        x: X coordinates of protein ROIs (top-left corner).
        y: Y coordinates of protein ROIs (top-left corner).
        boxsize: ROI box size.
        overlap: Array of overlapping ROI indices.
        output_file: Path to save output (will create folder with plot.png + data CSV).
        max_projection: Max projection image array.
        bg_x: X coordinates of background ROIs (optional).
        bg_y: Y coordinates of background ROIs (optional).
    """
    import numpy as np
    import matplotlib as mpl
    from pathlib import Path

    COLORS = _COLORS

    number_rois = len(x)
    overlapping_rois = list(overlap)

    # Create output directory
    output_path = Path(output_file)
    output_dir = output_path.parent / output_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    h_im, w_im = max_projection.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(max_projection, cmap="gray")
    # Lock axis limits to the image extent so that aspect='equal' + tight_layout()
    # cannot silently expand the data range and shift the image relative to boxes.
    ax.set_xlim(-0.5, w_im - 0.5)
    ax.set_ylim(h_im - 0.5, -0.5)

    # Draw protein ROI boxes
    # Offset by -0.5 because imshow places pixel centers at integer coordinates,
    # so a box at integer (x, y) would start at the pixel center, not its edge.
    single_count = 0
    overlap_count = 0
    for i in range(number_rois):
        if i in overlapping_rois:
            # Overlapping ROIs: vermillion boxes
            rect = patches.Rectangle(
                (x[i] - 0.5, y[i] - 0.5),
                boxsize,
                boxsize,
                linewidth=1,
                edgecolor=COLORS['vermillion'],
                facecolor="none",
            )
            ax.add_patch(rect)
            overlap_count += 1
        else:
            # Single ROIs: green boxes
            rect = patches.Rectangle(
                (x[i] - 0.5, y[i] - 0.5),
                boxsize,
                boxsize,
                linewidth=1,
                edgecolor=COLORS['green'],
                facecolor="none",
            )
            ax.add_patch(rect)
            single_count += 1

    # Draw background ROI boxes (if provided)
    bg_count = 0
    if bg_x is not None and bg_y is not None and len(bg_x) > 0:
        for i in range(len(bg_x)):
            rect = patches.Rectangle(
                (bg_x[i] - 0.5, bg_y[i] - 0.5),
                boxsize,
                boxsize,
                linewidth=1,
                edgecolor=COLORS['sky_blue'],
                facecolor="none",
                linestyle='--',
            )
            ax.add_patch(rect)
            bg_count += 1

    # Build legend (white background for visibility on dark images)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['green'], linewidth=2,
               label=f"Single protein ROI (n={single_count})"),
        Line2D([0], [0], color=COLORS['vermillion'], linewidth=2,
               label=f"Overlapping ROI (n={overlap_count})"),
    ]
    if bg_count > 0:
        legend_elements.append(
            Line2D([0], [0], color=COLORS['sky_blue'], linewidth=2, linestyle='--',
                   label=f"Background ROI (n={bg_count})")
        )

    ax.legend(handles=legend_elements, loc="upper right", fontsize=_FONTSIZE_LEGEND,
              frameon=False)

    ax.set_xlabel("X (pixels)", fontsize=_FONTSIZE_LABEL)
    ax.set_ylabel("Y (pixels)", fontsize=_FONTSIZE_LABEL)
    ax.set_title("Max projection with ROI overlays", fontsize=_FONTSIZE_TITLE, pad=9)
    ax.tick_params(labelsize=_FONTSIZE_TICK)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_dir / "plot.pdf", dpi=450, bbox_inches="tight")
    plt.close("all")

    # Save ROI data as CSV
    import pandas as pd
    roi_data = {
        'roi_type': [],
        'x': [],
        'y': [],
    }
    for i in range(number_rois):
        roi_type = 'overlapping' if i in overlapping_rois else 'single'
        roi_data['roi_type'].append(roi_type)
        roi_data['x'].append(x[i])
        roi_data['y'].append(y[i])

    if bg_x is not None and bg_y is not None:
        for i in range(len(bg_x)):
            roi_data['roi_type'].append('background')
            roi_data['x'].append(bg_x[i])
            roi_data['y'].append(bg_y[i])

    pd.DataFrame(roi_data).to_csv(output_dir / "data_rois.csv", index=False)


def gmm_background_estimate(trace: "np.ndarray") -> float:
    """
    Fit a 2-component GMM to a raw intensity trace and return the noise component mean.
    Falls back to mean of the last 500 frames if fitting fails.
    """
    import numpy as np
    from sklearn.mixture import GaussianMixture

    X = trace.reshape(-1, 1).astype(np.float64)
    try:
        gmm = GaussianMixture(n_components=2, n_init=5, random_state=0)
        gmm.fit(X)
        order = np.argsort(gmm.means_.ravel())
        return float(gmm.means_.ravel()[order[0]])
    except Exception:
        return float(trace[-500:].mean())


def gmm_classify_frames(
    trace: "np.ndarray",
    proba_threshold: float = 0.8,
    min_separation: float = 0.0,
    return_separation: bool = False,
) -> "tuple":
    """
    Fit a 2-component GMM to a bg_rm trace and classify each frame by posterior probability.

    Args:
        trace: Background-removed intensity trace.
        proba_threshold: Posterior probability threshold for signal classification.
        min_separation: Minimum (signal_mean - noise_mean) / noise_std required to accept
            signal frames. Traces below this threshold are returned as all-noise, preventing
            unimodal noise distributions from being artificially split into signal runs.
            0.0 disables the check (default, backwards compatible).
        return_separation: If True, append the separation value as the last element of the
            returned tuple, giving (noise_mask, overlap_mask, signal_mask, separation).

    Returns:
        (noise_mask, overlap_mask, signal_mask) boolean arrays, or
        (noise_mask, overlap_mask, signal_mask, separation) if return_separation=True.
        Falls back to all-noise (separation=0.0) if fitting fails.
    """
    import numpy as np
    from sklearn.mixture import GaussianMixture

    X = trace.reshape(-1, 1).astype(np.float64)
    all_noise = np.ones(len(trace), dtype=bool)
    all_false = ~all_noise

    try:
        gmm = GaussianMixture(n_components=2, n_init=5, random_state=0)
        gmm.fit(X)
        order = np.argsort(gmm.means_.ravel())
        means = gmm.means_.ravel()
        noise_std = np.sqrt(max(float(gmm.covariances_[order[0]].ravel()[0]), 1e-12))
        separation = (means[order[1]] - means[order[0]]) / noise_std

        if min_separation > 0 and separation < min_separation:
            if return_separation:
                return all_noise, all_false, all_false, separation
            return all_noise, all_false, all_false

        p_signal = gmm.predict_proba(X)[:, order[1]]
        noise_mask   = p_signal <  (1 - proba_threshold)
        overlap_mask = (p_signal >= (1 - proba_threshold)) & (p_signal < proba_threshold)
        signal_mask  = p_signal >= proba_threshold
        if return_separation:
            return noise_mask, overlap_mask, signal_mask, separation
        return noise_mask, overlap_mask, signal_mask
    except Exception:
        if return_separation:
            return all_noise, all_false, all_false, 0.0
        return all_noise, all_false, all_false
