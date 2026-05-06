#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH), Andreas Biri (ETHZ).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Shared utility functions for vesicle analysis scripts.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re

import matplotlib as mpl
import yaml

# =============================================================================
# Matplotlib Configuration
# =============================================================================

# Color-blind friendly palette (Okabe-Ito)
COLORS = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'pink': '#CC79A7',
    'black': '#000000',
}

# Font size hierarchy
FONTSIZE_LABEL  = 7
FONTSIZE_TICK   = 6
FONTSIZE_TITLE  = 7
FONTSIZE_LEGEND = 6


def apply_rcparams():
    """Apply global matplotlib rcParams to match locbplots standards."""
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    mpl.rcParams['font.size']          = FONTSIZE_LABEL
    mpl.rcParams['axes.titlesize']     = FONTSIZE_TITLE
    mpl.rcParams['axes.labelsize']     = FONTSIZE_LABEL
    mpl.rcParams['xtick.labelsize']    = FONTSIZE_TICK
    mpl.rcParams['ytick.labelsize']    = FONTSIZE_TICK
    mpl.rcParams['legend.fontsize']    = FONTSIZE_LEGEND
    mpl.rcParams['figure.dpi']         = 150
    mpl.rcParams['savefig.dpi']        = 450
    mpl.rcParams['savefig.bbox']       = 'tight'
    mpl.rcParams['lines.linewidth']    = 1.2
    mpl.rcParams['patch.linewidth']    = 0.8
    mpl.rcParams['axes.grid']          = False
    mpl.rcParams['pdf.fonttype']       = 42
    mpl.rcParams['svg.fonttype']       = 'none'


apply_rcparams()


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: Path) -> dict:
    """Load configuration file."""
    with config_path.open("r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


# =============================================================================
# Logging
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
# Plotting
# =============================================================================

def apply_axis_standards(ax):
    """Apply standard axis formatting for all plots."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICK,
                   length=4, width=0.8, direction='out')
    ax.tick_params(axis='both', which='minor', length=2, width=0.6, direction='out')


def save_figure(fig, output_dir, filename='plot.pdf', dpi=450):
    """Save figure as PDF at publication quality."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(h_pad=3, w_pad=3)
    fig.savefig(out / (Path(filename).stem + '.pdf'), dpi=dpi, bbox_inches='tight')


# =============================================================================
# Picasso Integration
# =============================================================================

def get_picasso_python() -> str:
    """
    Get the path to the Python interpreter that has Picasso installed.

    Checks:
    1. picasso-env conda/mamba environment (prioritize this)
    2. Current environment
    3. System python

    Returns:
        Path to Python executable with Picasso.

    Raises:
        RuntimeError if Picasso is not found.
    """
    # Try picasso-env first (most reliable)
    candidates = [
        # Mamba/Miniforge environments
        os.path.expanduser("~/mambaforge/envs/picasso-env/bin/python"),
        os.path.expanduser("~/miniforge3/envs/picasso-env/bin/python"),
        # Conda/Anaconda environments
        os.path.expanduser("~/opt/anaconda3/envs/picasso-env/bin/python"),
        os.path.expanduser("~/anaconda3/envs/picasso-env/bin/python"),
        os.path.expanduser("~/miniconda3/envs/picasso-env/bin/python"),
        "/opt/anaconda3/envs/picasso-env/bin/python",
        "/anaconda3/envs/picasso-env/bin/python",
        # Current Python
        sys.executable,
        # System python
        "/usr/bin/python3",
        "/usr/local/bin/python3",
    ]

    for python_path in candidates:
        try:
            result = subprocess.run(
                [python_path, "-c", "import picasso"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                logging.info(f"Found Picasso in: {python_path}")
                return python_path
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    raise RuntimeError(
        "Picasso not found. Please ensure it's installed:\n"
        "  mamba env create -f picasso-env.yaml\n"
        "  conda activate picasso-env"
    )


def get_localization_method() -> str:
    """
    Get the localization method for Picasso.

    Returns:
        'mle' (maximum likelihood estimation on CPU).
    """
    logging.info("Using CPU-based localization (mle)")
    return "mle"


# =============================================================================
# Parallel Processing
# =============================================================================

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
        logging.info("Sequential processing (n_workers=1)")
        return 1
    else:
        logging.info(f"Using {n_workers} worker processes")
        return n_workers


# =============================================================================
# Data Discovery
# =============================================================================

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
        │   ├── Protein1/
        │   │   ├── Protein1_640nm_..._0001.nd2
        │   │   ├── Protein1_488nm_..._0001.nd2  (optional, if has ground truth)
        │   │   └── ...
        │   └── Protein2/
        │       └── ...
        └── Exp2/
            └── ...

    Args:
        input_folder: Root data folder containing experiments.
        proteins: List of protein names to search for.
        protein_channel: Channel name for protein (default: "640").
        ground_truth_channel: Channel name for ground truth (default: "488").
                              Set to None for single-channel analysis.

    Returns:
        Dict mapping protein names to list of (protein_file, gt_file) tuples.
        gt_file will be None if ground_truth_channel is None.
    """
    data: Dict[str, List[Tuple[Path, Path | None]]] = {p: [] for p in proteins}

    # Find all experiment folders
    exp_folders = sorted([
        d for d in input_folder.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    logging.info(f"Found {len(exp_folders)} experiments: {[e.name for e in exp_folders]}")

    for exp_folder in exp_folders:
        logging.info(f"Processing experiment: {exp_folder.name}")

        # Find protein folders within experiment
        protein_folders = [
            d for d in exp_folder.iterdir()
            if d.is_dir() and d.name in proteins and not d.name.startswith(".")
        ]

        if not protein_folders:
            logging.warning(f"  No protein folders found in {exp_folder.name}")
            continue

        logging.info(f"  Found proteins: {[p.name for p in protein_folders]}")

        for protein_folder in protein_folders:
            protein = protein_folder.name

            # Handle GT-only mode (protein_channel=None)
            if protein_channel is None and ground_truth_channel is not None:
                # Find all GT files directly
                gt_files_found = sorted(
                    protein_folder.glob(f"*{ground_truth_channel}nm*.nd2")
                )
                for gt_file in gt_files_found:
                    # Skip macOS metadata files
                    if gt_file.name.startswith("._"):
                        continue
                    data[protein].append((None, gt_file))
            else:
                # Normal mode: find protein files and optionally pair with GT
                protein_files = sorted(
                    protein_folder.glob(f"*{protein_channel}nm*.nd2")
                )

                if ground_truth_channel is not None:
                    # Find all GT files
                    gt_files = sorted(
                        protein_folder.glob(f"*{ground_truth_channel}nm*.nd2")
                    )

                    # Create mapping of frame number to GT file
                    # Extract frame number from filename (e.g., _0001.nd2 -> "0001")
                    gt_map = {}
                    for gt_file in gt_files:
                        if gt_file.name.startswith("._"):
                            continue
                        # Extract frame number: find _XXXX.nd2 pattern at end
                        match = re.search(r'_(\d{4})\.nd2$', gt_file.name)
                        if match:
                            frame_num = match.group(1)
                            gt_map[frame_num] = gt_file

                for protein_file in protein_files:
                    # Skip macOS metadata files
                    if protein_file.name.startswith("._"):
                        continue

                    # Find corresponding GT file if needed
                    gt_file = None
                    if ground_truth_channel is not None:
                        # Extract frame number from protein filename
                        match = re.search(r'_(\d{4})\.nd2$', protein_file.name)
                        if match:
                            frame_num = match.group(1)
                            gt_file = gt_map.get(frame_num)

                            if gt_file is None:
                                logging.warning(
                                    f"    Missing GT file for {protein_file.name} (frame {frame_num})"
                                )
                                continue
                        else:
                            logging.warning(
                                f"    Could not extract frame number from {protein_file.name}"
                            )
                            continue

                    data[protein].append((protein_file, gt_file))

            logging.info(f"    {protein}: Found {len(data[protein])} movie pairs")

    return data
