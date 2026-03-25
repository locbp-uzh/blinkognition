#!/usr/bin/env python3
"""
Ground Truth Gradient Sweep Analysis

This script performs a systematic sweep of gradient thresholds for ground truth
localization to understand how gradient selection affects intensity distributions.

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
) -> Path:
    """
    Run Picasso localization on ground truth channel.

    Args:
        nd2_file: Path to ND2 file.
        gradient: Gradient threshold.
        cfg: Configuration dictionary.
        picasso_python: Path to Picasso Python.
        output_dir: Directory for output files.

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
            "-a", method,  # Algorithm flag is -a
            "-g", str(gradient),
            "-b", str(cfg["boxsize"]),  # Box side length flag is -b
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

        # Move both files to output directory with gradient in name
        final_locs_file = output_dir / f"{nd2_file.stem}_grad{gradient}_locs.hdf5"
        final_yaml_file = output_dir / f"{nd2_file.stem}_grad{gradient}_locs.yaml"

        shutil.move(locs_file, final_locs_file)
        if yaml_file.exists():
            shutil.move(yaml_file, final_yaml_file)

    return final_locs_file


def compute_intensities_from_locs(
    nd2_file: Path,
    locs_file: Path,
    cfg: dict,
) -> np.ndarray:
    """
    Compute integrated intensities from localizations.

    Args:
        nd2_file: Path to ND2 file.
        locs_file: Path to localization HDF5 file.
        cfg: Configuration dictionary.

    Returns:
        Array of integrated intensities.
    """
    import nd2
    from picasso import io as pio, lib, clusterer

    # Load movie
    movie = nd2.imread(str(nd2_file))
    n_frames = min(cfg["n_frames_integrate"], movie.shape[0])

    # Load localizations
    locs, info = pio.load_locs(str(locs_file))
    locs = lib.ensure_sanity(locs, info)

    if len(locs) == 0:
        return np.array([])

    # Cluster localizations to identify unique spots
    cluster_locs = clusterer.cluster(
        locs,
        radius_xy=cfg["max_distance_ground_truth"],
        min_locs=cfg["min_on_ground_truth"],
        frame_analysis=False,
        radius_z=None,
        pixelsize=130,
    )
    cluster_locs = clusterer.find_cluster_centers(cluster_locs, pixelsize=None)

    if len(cluster_locs) == 0:
        return np.array([])

    # Compute integrated intensities
    boxsize = cfg["boxsize"]
    shift = boxsize / 2
    intensities = []

    for loc in cluster_locs:
        x = int(loc["x"] - shift)
        y = int(loc["y"] - shift)

        # Bounds check
        if x < 0 or y < 0:
            continue
        if x + boxsize > movie.shape[2] or y + boxsize > movie.shape[1]:
            continue

        # Extract ROI and integrate
        roi = movie[:n_frames, y:(y + boxsize), x:(x + boxsize)]
        integrated_intensity = np.sum(roi)
        intensities.append(integrated_intensity)

    return np.array(intensities)


def gradient_sweep_worker(args: tuple) -> tuple[int, np.ndarray]:
    """
    Worker function for parallel gradient sweep processing.

    Args:
        args: Tuple of (nd2_file, gradient, cfg, picasso_python, output_dir)

    Returns:
        Tuple of (gradient, intensities array)
    """
    nd2_file, gradient, cfg, picasso_python, output_dir = args

    try:
        # Run localization
        locs_file = run_localization(
            nd2_file, gradient, cfg, picasso_python, output_dir
        )

        # Compute intensities
        intensities = compute_intensities_from_locs(nd2_file, locs_file, cfg)

        return (gradient, intensities)

    except Exception as e:
        logging.warning(f"Failed gradient {gradient} for {nd2_file.name}: {e}")
        return (gradient, np.array([]))


def analyze_movie_gradient_sweep(
    nd2_file: Path,
    gradients: List[int],
    cfg: dict,
    picasso_python: str,
    output_dir: Path,
    num_workers: int = -1,
) -> Dict[int, np.ndarray]:
    """
    Analyze a single movie across gradient sweep.

    Args:
        nd2_file: Path to ground truth ND2 file.
        gradients: List of gradient values to test.
        cfg: Configuration dictionary.
        picasso_python: Path to Picasso Python.
        output_dir: Output directory.
        num_workers: Number of parallel workers (-1 = all CPUs, 1 = sequential).

    Returns:
        Dictionary mapping gradient to intensity array.
    """
    results = {}

    logging.info(f"  Analyzing {nd2_file.name} with {len(gradients)} gradients")

    # Prepare work items for parallel processing
    work_items = [
        (nd2_file, gradient, cfg, picasso_python, output_dir)
        for gradient in gradients
    ]

    # Process gradients (parallel or sequential based on num_workers)
    if num_workers == 1:
        # Sequential processing
        logging.info(f"    Sequential processing of {len(work_items)} gradients")
        for work_item in work_items:
            gradient = work_item[1]
            logging.info(f"    Gradient {gradient}...")
            grad_val, intensities = gradient_sweep_worker(work_item)
            results[grad_val] = intensities
            logging.info(f"      → {len(intensities)} detections")
    else:
        # Parallel processing
        import multiprocessing as mp
        logging.info(f"    Parallel processing with {num_workers} workers")

        with mp.Pool(processes=num_workers) as pool:
            gradient_results = pool.map(gradient_sweep_worker, work_items)

        # Aggregate results
        for gradient, intensities in gradient_results:
            results[gradient] = intensities
            logging.info(f"    Gradient {gradient} → {len(intensities)} detections")

    return results


def plot_gradient_comparison(
    all_results: Dict[str, Dict[int, np.ndarray]],
    gradients: List[int],
    output_dir: Path,
    protein: str,
):
    """
    Plot comparison of intensity distributions across gradients.

    Args:
        all_results: Nested dict {movie_name: {gradient: intensities}}
        gradients: List of gradient values.
        output_dir: Base output directory for this protein.
        protein: Protein name.
    """
    # Aggregate all movies for each gradient
    gradient_data = {}
    for gradient in gradients:
        all_intensities = []
        for movie_results in all_results.values():
            if gradient in movie_results:
                all_intensities.extend(movie_results[gradient])
        gradient_data[gradient] = np.array(all_intensities)

    # Create output directory with descriptive name
    plot_dir = output_dir / f"{protein}_gradient_comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Generate colors for gradients (use subset of Okabe-Ito palette cycling)
    color_list = [COLORS['blue'], COLORS['orange'], COLORS['green'],
                  COLORS['pink'], COLORS['vermillion'], COLORS['sky_blue']]
    gradient_colors = [color_list[i % len(color_list)] for i in range(len(gradients))]

    # Create figure with 3 panels in single row
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Log-scale intensity distributions
    ax = axes[0]
    for idx, gradient in enumerate(gradients):
        intensities = gradient_data[gradient]
        if len(intensities) > 0 and len(intensities) > 10:
            log_bins = np.logspace(np.log10(max(intensities.min(), 1)),
                                   np.log10(intensities.max()), 50)
            ax.hist(intensities, bins=log_bins, alpha=0.6, color=gradient_colors[idx],
                   label=f'grad={gradient}', edgecolor='none')

    ax.set_xscale('log')
    ax.set_xlabel('Integrated Intensity (log scale)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Count', fontsize=FONTSIZE_LABEL)
    ax.set_title('Log-scale intensity distributions', fontsize=FONTSIZE_TITLE, pad=9)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc='upper right')
    apply_axis_standards(ax)

    # Panel B: Detection count vs gradient
    ax = axes[1]
    detection_counts = [len(gradient_data[g]) for g in gradients]
    ax.plot(gradients, detection_counts, 'o-', linewidth=1.5, markersize=6, color=COLORS['blue'])

    ax.set_xlabel('Gradient Threshold', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Total Detections', fontsize=FONTSIZE_LABEL)
    ax.set_title('Detection count vs gradient', fontsize=FONTSIZE_TITLE, pad=9)
    apply_axis_standards(ax)

    # Add annotations
    for g, count in zip(gradients, detection_counts):
        ax.text(g, count + max(detection_counts)*0.02, f'{count}',
               ha='center', fontsize=FONTSIZE_TICK)

    # Panel C: Median and percentiles vs gradient
    ax = axes[2]

    medians = []
    p10s = []
    p90s = []

    for gradient in gradients:
        intensities = gradient_data[gradient]
        if len(intensities) > 0:
            medians.append(np.median(intensities))
            p10s.append(np.percentile(intensities, 10))
            p90s.append(np.percentile(intensities, 90))
        else:
            medians.append(np.nan)
            p10s.append(np.nan)
            p90s.append(np.nan)

    ax.plot(gradients, medians, 'o-', linewidth=1.5, markersize=6, label='Median', color=COLORS['green'])
    ax.plot(gradients, p10s, 's-', linewidth=1.5, markersize=5, label='P10', color=COLORS['orange'], alpha=0.7)
    ax.plot(gradients, p90s, '^-', linewidth=1.5, markersize=5, label='P90', color=COLORS['vermillion'], alpha=0.7)

    ax.set_xlabel('Gradient Threshold', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Intensity', fontsize=FONTSIZE_LABEL)
    ax.set_title('Median and percentiles vs gradient', fontsize=FONTSIZE_TITLE, pad=9)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False)
    apply_axis_standards(ax)

    # Save figure
    save_figure(plt.gcf(), plot_dir)
    plt.close()

    # Save source data for each panel
    # Panel A: Histogram data for each gradient (save raw intensities)
    panel_a_data = {}
    for gradient in gradients:
        intensities = gradient_data[gradient]
        # Pad to same length with NaN
        panel_a_data[f'grad_{gradient}'] = intensities

    # Find max length
    max_len = max(len(v) for v in panel_a_data.values())
    for key in panel_a_data.keys():
        arr = panel_a_data[key]
        if len(arr) < max_len:
            panel_a_data[key] = np.pad(arr, (0, max_len - len(arr)),
                                       constant_values=np.nan)
    df_panel_a = pd.DataFrame(panel_a_data)
    df_panel_a.to_csv(plot_dir / 'data_panel_A.csv', index=False)

    # Panel B: Detection count data
    df_panel_b = pd.DataFrame({
        'gradient': gradients,
        'detection_count': detection_counts,
    })
    df_panel_b.to_csv(plot_dir / 'data_panel_B.csv', index=False)

    # Panel C: Statistics data
    df_panel_c = pd.DataFrame({
        'gradient': gradients,
        'median': medians,
        'p10': p10s,
        'p90': p90s,
    })
    df_panel_c.to_csv(plot_dir / 'data_panel_C.csv', index=False)

    logging.info(f"Saved gradient comparison plot and data to: {plot_dir}")


def save_summary_csv(
    all_results: Dict[str, Dict[int, np.ndarray]],
    gradients: List[int],
    output_path: Path,
):
    """
    Save summary statistics to CSV.

    Args:
        all_results: Nested dict {movie_name: {gradient: intensities}}
        gradients: List of gradient values.
        output_path: Path to save CSV.
    """
    rows = []

    for gradient in gradients:
        # Aggregate across all movies
        all_intensities = []
        for movie_results in all_results.values():
            if gradient in movie_results:
                all_intensities.extend(movie_results[gradient])

        intensities = np.array(all_intensities)

        if len(intensities) > 0:
            rows.append({
                'gradient': gradient,
                'n_detections': len(intensities),
                'mean': np.mean(intensities),
                'median': np.median(intensities),
                'std': np.std(intensities),
                'min': np.min(intensities),
                'max': np.max(intensities),
                'p10': np.percentile(intensities, 10),
                'p25': np.percentile(intensities, 25),
                'p75': np.percentile(intensities, 75),
                'p90': np.percentile(intensities, 90),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved summary CSV: {output_path}")


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Ground truth gradient sweep analysis"
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
    args = parser.parse_args()

    # Setup
    setup_logging("INFO")
    cfg = load_config(Path(args.config_path))

    # Get Picasso Python
    picasso_python = get_picasso_python()

    # Get parallel processing configuration
    n_workers_config = cfg.get("n_workers", -1)
    num_workers = get_num_workers(n_workers_config)

    # Get proteins
    proteins = cfg["proteins"]
    if args.protein:
        if args.protein in proteins:
            proteins = [args.protein]
            logging.info(f"Filtering to protein: {args.protein}")
        else:
            logging.error(f"Protein '{args.protein}' not found in config")
            sys.exit(1)

    # Get gradient sweep
    if "gradient_list" in cfg:
        gradients = cfg["gradient_list"]
    else:
        sweep_cfg = cfg["gradient_sweep"]
        gradients = list(range(sweep_cfg["min"], sweep_cfg["max"] + 1, sweep_cfg["step"]))

    logging.info("=" * 60)
    logging.info("Ground Truth Gradient Sweep Analysis")
    logging.info("=" * 60)
    logging.info(f"Proteins: {proteins}")
    logging.info(f"Gradients to test: {gradients}")
    logging.info(f"Integration frames: {cfg['n_frames_integrate']}")
    logging.info("")

    # Discover ground truth data
    input_root = Path(cfg["input_folder"]).expanduser()
    protein_data = discover_all_data(
        input_root,
        proteins,
        protein_channel=None,  # Don't look for protein channel
        ground_truth_channel=cfg["ground_truth_channel"],
    )

    # Extract only ground truth files
    gt_files = {}
    for protein, pairs in protein_data.items():
        # pairs is [(None, gt_file)] in GT-only mode
        # Filter out macOS metadata files (._filename)
        gt_files[protein] = [
            gt_file for pf, gt_file in pairs
            if gt_file is not None and not gt_file.name.startswith("._")
        ]

    # Print summary (adapt for GT-only mode)
    logging.info("=" * 60)
    logging.info("Data Discovery Summary (GT-only)")
    logging.info("=" * 60)
    total_items = 0
    for protein, files in gt_files.items():
        if files:
            logging.info(f"{protein}: {len(files)} ground truth movies")
            total_items += len(files)
        else:
            logging.info(f"{protein}: No data found")
    logging.info(f"\nTotal: {total_items} ground truth movies across all proteins")
    logging.info("=" * 60)

    # Process each protein
    for protein in proteins:
        if protein not in gt_files or not gt_files[protein]:
            logging.warning(f"No ground truth files found for {protein}")
            continue

        logging.info("")
        logging.info("=" * 60)
        logging.info(f"Analyzing {protein}")
        logging.info("=" * 60)

        # Create output directories
        output_base = Path(cfg["output_folder_base"]).expanduser() / protein
        locs_dir = output_base / "localizations"
        data_dir = output_base / "intensity_data"
        plots_dir = output_base / "plots"

        for d in [locs_dir, data_dir, plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Process all movies for this protein
        all_results = {}

        for nd2_file in gt_files[protein]:
            movie_results = analyze_movie_gradient_sweep(
                nd2_file, gradients, cfg, picasso_python, locs_dir, num_workers
            )
            all_results[nd2_file.name] = movie_results

        # Generate comparison plots
        plot_gradient_comparison(all_results, gradients, plots_dir, protein)

        # Save summary CSV
        csv_path = data_dir / f"{protein}_gradient_summary.csv"
        save_summary_csv(all_results, gradients, csv_path)

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
