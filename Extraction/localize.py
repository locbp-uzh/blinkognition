#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH), Andreas Biri (ETHZ).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Execute Picasso localization on ND2 movies using CPU-based MLE method.

This script:
- Uses Picasso's MLE (maximum likelihood estimation) localization method
- Supports multi-channel imaging with channel-specific gradients
- Processes ND2 files recursively while skipping specified directories
- Parallel processing across multiple CPUs
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import warnings
import yaml
from datetime import datetime
from pathlib import Path

from utils import (
    setup_logging,
    get_localization_method,
    get_picasso_python,
    get_fov_size,
    get_num_workers,
    discover_experiments,
    discover_protein_folders,
    find_nd2_files,
)

# Suppress known benign warnings from Picasso's internal code:
# 1. "invalid value encountered in sqrt": Occurs during CRLB calculation for poor-quality
#    localizations. These are automatically filtered by Picasso's ensure_sanity() later.
# 2. "ND2File not closed": Picasso's internal ND2 file handling doesn't use context managers.
#    Files are still properly closed by Python's garbage collector.
# 3. "Series.swapaxes deprecated": Pandas deprecation triggered by sklearn scalers on Series.
#    Will be fixed in future sklearn/pandas versions.
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message=".*ND2File file not closed before garbage collection.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Series.swapaxes.*")


def load_config(config_path: Path) -> dict:
    """Load and validate configuration file."""
    with config_path.open("r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    required = [
        "input_folder",
        "proteins",
        "boxsize",
        "gradient_protein",
        "drift",
        "baseline",
        "sensitivity",
        "gain",
        "quantum_efficiency",
    ]

    # Add ground_truth gradient to required only if has_ground_truth is True
    if cfg.get("has_ground_truth", True):
        required.append("gradient_ground_truth")

    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    # Backward compatibility: compounds is alias for proteins
    if "compounds" in cfg and "proteins" not in cfg:
        cfg["proteins"] = cfg["compounds"]

    return cfg


def load_optimized_params(
    protein_folder: Path,
    has_ground_truth: bool,
    experiment_folder: Path | None = None,
) -> dict | None:
    """
    Load optimized localization parameters.

    Checks in order:
    1. Experiment-level params (from per-experiment pooled optimization)
    2. Protein-level params (from legacy per-protein optimization)

    Args:
        protein_folder: Path to protein folder (e.g., /path/to/Exp1/HaloD106).
        has_ground_truth: Whether to load params for ground truth mode or no ground truth mode.
        experiment_folder: Path to experiment folder (e.g., /path/to/Exp1). If None, skips experiment-level check.

    Returns:
        Dictionary with optimized parameters, or None if not found.
    """
    mode_suffix = "gt" if has_ground_truth else "no_gt"

    # Check experiment-level params first (per-experiment optimization)
    if experiment_folder is not None:
        exp_params_file = experiment_folder / f"optimized_localization_params_{mode_suffix}.yaml"
        if exp_params_file.exists():
            try:
                with exp_params_file.open("r") as f:
                    data = yaml.load(f, Loader=yaml.SafeLoader)
                opt_params = data.get("optimized_parameters", {})
                if opt_params:
                    logging.info(f"  Using experiment-level optimized parameters from: {exp_params_file.name}")
                    return opt_params
            except Exception as e:
                logging.warning(f"  Failed to load experiment-level params from {exp_params_file}: {e}")

    # Fall back to protein-level params (legacy per-protein optimization)
    params_file = protein_folder / f"optimized_localization_params_{mode_suffix}.yaml"
    if not params_file.exists():
        return None

    try:
        with params_file.open("r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        opt_params = data.get("optimized_parameters", {})
        if opt_params:
            logging.info(f"  Using protein-level optimized parameters from: {params_file.name}")
            return opt_params
        else:
            return None
    except Exception as e:
        logging.warning(f"  Failed to load optimized params from {params_file}: {e}")
        return None


def detect_channel_gradient(file_name: str, cfg: dict) -> int:
    """
    Detect the appropriate gradient based on channel in filename.

    Args:
        file_name: Name of the ND2 file.
        cfg: Configuration dictionary with gradient values.

    Returns:
        Gradient value for the detected channel.
    """
    protein_channel = cfg.get("protein_channel", "640")
    ground_truth_channel = cfg.get("ground_truth_channel", "488")

    # Check which channel this file belongs to
    if protein_channel in file_name:
        return int(cfg.get("gradient_protein", 10000))
    elif ground_truth_channel in file_name:
        return int(cfg.get("gradient_ground_truth", cfg.get("gradient_405", 1000)))
    elif "405" in file_name:
        return int(cfg.get("gradient_405", cfg.get("gradient_ground_truth", 1000)))
    else:
        return int(cfg.get("gradient", 10000))


def generate_run_folder_name(proteins: list, cfg: dict, use_optimized: bool = False) -> str:
    """
    Generate run folder name based on proteins and gradients.

    Format (with ground truth, optimized): {proteins}_optparam_{number}
    Format (with ground truth, config): {proteins}_gradP-{value}_gradGT-{value}_{number}
    Format (no ground truth, optimized): {proteins}_optparam_{number}
    Format (no ground truth, config): {proteins}_gradP-{value}_{number}
    Example: HaloD106_SNAPC148_optparam_001 or HaloD106_SNAPC148_gradP-12000_gradGT-4000_001

    Args:
        proteins: List of protein names.
        cfg: Configuration dictionary.
        use_optimized: If True, use "optparam" instead of specific gradient values.

    Returns:
        Base folder name (without run number).
    """
    protein_str = "_".join(sorted(proteins))

    if use_optimized:
        return f"{protein_str}_optparam"

    grad_protein = int(cfg.get("gradient_protein", 10000))
    has_ground_truth = cfg.get("has_ground_truth", True)

    if has_ground_truth:
        grad_ground_truth = int(cfg.get("gradient_ground_truth", 1000))
        return f"{protein_str}_gradP-{grad_protein}_gradGT-{grad_ground_truth}"
    else:
        return f"{protein_str}_gradP-{grad_protein}"


def find_next_run_number(base_folder: Path, base_name: str) -> int:
    """
    Find the next available run number.

    Args:
        base_folder: Base output folder (e.g., Results/Extract).
        base_name: Base run folder name without number.

    Returns:
        Next available run number.
    """
    if not base_folder.exists():
        return 1

    # Find existing runs with this base name
    pattern = f"{base_name}_*"
    existing = list(base_folder.glob(pattern))

    if not existing:
        return 1

    # Extract numbers from existing runs
    numbers = []
    for folder in existing:
        parts = folder.name.split("_")
        if parts and parts[-1].isdigit():
            numbers.append(int(parts[-1]))

    return max(numbers) + 1 if numbers else 1


def save_protein_localization_params(
    protein_path: Path,
    protein: str,
    experiment: str,
    protein_cfg: dict,
    method: str,
    has_ground_truth: bool,
    source: str,
) -> None:
    """
    Save protein-specific localization parameters to protein folder.

    Args:
        protein_path: Path to protein folder.
        protein: Protein name.
        experiment: Experiment name.
        protein_cfg: Configuration dictionary with protein-specific params.
        method: Localization method used.
        has_ground_truth: Whether analysis includes ground truth channel.
        source: Source of parameters ("optimized" or "config_defaults").
    """
    mode_suffix = "gt" if has_ground_truth else "no_gt"

    localization_parameters = {
        "boxsize": int(protein_cfg["boxsize"]),
        "gradient_protein": int(protein_cfg.get("gradient_protein", 10000)),
        "drift": int(protein_cfg["drift"]),
        "baseline": int(protein_cfg["baseline"]),
        "sensitivity": float(protein_cfg["sensitivity"]),
        "gain": int(protein_cfg["gain"]),
        "quantum_efficiency": float(protein_cfg["quantum_efficiency"]),
    }

    if has_ground_truth:
        localization_parameters["gradient_ground_truth"] = int(
            protein_cfg.get("gradient_ground_truth", 1000)
        )

    params = {
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "protein": protein,
            "experiment": experiment,
            "parameter_source": source,
            "analysis_mode": mode_suffix,
        },
        "localization_parameters": localization_parameters,
    }

    params_file = protein_path / f"used_localization_params_{mode_suffix}.yaml"
    with params_file.open("w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)


def run_picasso_localize(
    nd2_path: Path,
    gradient: int,
    method: str,
    cfg: dict,
    picasso_python: str,
) -> None:
    """
    Run Picasso localize on a single ND2 file.

    Args:
        nd2_path: Path to ND2 file.
        gradient: Gradient threshold for localization.
        method: Localization method ('mle').
        cfg: Configuration dictionary.
        picasso_python: Path to Python with Picasso installed.
    """
    cmd = [
        picasso_python,
        "-m",
        "picasso",
        "localize",
        nd2_path.name,  # Use basename; cwd set to parent directory
        "-b", str(int(cfg["boxsize"])),
        "-g", str(int(gradient)),
        "-d", str(int(cfg["drift"])),
        "-bl", str(int(cfg["baseline"])),
        "-s", str(float(cfg["sensitivity"])),
        "-ga", str(int(cfg["gain"])),
        "-qe", str(float(cfg["quantum_efficiency"])),
        "-a", method,
    ]

    logging.info(f"Localizing: {nd2_path.name}")
    logging.debug(f"Command: {' '.join(cmd)}")

    # Set environment for headless execution (HPC compatibility)
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(nd2_path.parent),
            env=env,
            check=True,
            capture_output=False,  # Stream output to console
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Picasso localization failed for {nd2_path}: {e}")


def localize_worker(args: tuple) -> tuple[Path, bool, str]:
    """
    Worker function for parallel localization of ND2 files.

    Args:
        args: Tuple of (nd2_file, gradient, method, protein_cfg, picasso_python)

    Returns:
        Tuple of (nd2_file, success, error_message)
    """
    nd2_file, gradient, method, protein_cfg, picasso_python = args

    try:
        run_picasso_localize(nd2_file, gradient, method, protein_cfg, picasso_python)
        return (nd2_file, True, "")
    except Exception as e:
        return (nd2_file, False, str(e))


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Execute Picasso localization using CPU-based MLE method"
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
        "--experiment",
        type=str,
        default=None,
        help="Process only this experiment (e.g., 'Exp1'). If not specified, processes all experiments.",
    )
    parser.add_argument(
        "--protein",
        type=str,
        default=None,
        help="Process only this protein (e.g., 'HaloD106'). If not specified, processes all proteins from config.",
    )
    args = parser.parse_args()

    # Setup
    setup_logging("INFO")
    cfg = load_config(Path(args.config_path))

    # Get paths
    input_root = Path(cfg["input_folder"]).expanduser()
    output_base = Path(cfg.get("output_folder_base", "Results/Extract")).expanduser()
    proteins = cfg["proteins"]
    protein_channel = cfg.get("protein_channel", "640")
    ground_truth_channel = cfg.get("ground_truth_channel", "488")
    has_ground_truth = cfg.get("has_ground_truth", True)

    # Filter to specific protein if requested (for debugging)
    if args.protein:
        if args.protein in proteins:
            proteins = [args.protein]
            logging.info(f"Filtering to protein: {args.protein}")
        else:
            logging.error(f"Protein '{args.protein}' not found in config. Available: {proteins}")
            sys.exit(1)

    # Check if optimized parameters exist
    logging.info("Checking for optimized parameters...")
    experiments = discover_experiments(input_root)

    # Filter to specific experiment if requested (for debugging)
    if args.experiment:
        experiments = [e for e in experiments if e.name == args.experiment]
        if not experiments:
            logging.error(f"Experiment '{args.experiment}' not found in {input_root}")
            sys.exit(1)
        logging.info(f"Filtering to experiment: {args.experiment}")

    has_optimized_params = False

    for exp_path in experiments:
        protein_folders = discover_protein_folders(exp_path, proteins)
        for protein, protein_path in protein_folders.items():
            opt_params = load_optimized_params(protein_path, has_ground_truth, experiment_folder=exp_path)
            if opt_params:
                has_optimized_params = True
                break
        if has_optimized_params:
            break

    # Generate run folder name
    if has_optimized_params:
        logging.info("Optimized parameters found - using 'optparam' in folder name")
        run_base_name = generate_run_folder_name(proteins, cfg, use_optimized=True)
    else:
        logging.info("No optimized parameters found - using config gradient values in folder name")
        run_base_name = generate_run_folder_name(proteins, cfg, use_optimized=False)
    run_number = find_next_run_number(output_base, run_base_name)
    run_folder_name = f"{run_base_name}_{run_number:03d}"
    output_folder = output_base / run_folder_name

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output folder: {output_folder}")

    # Get parallel processing configuration
    n_workers_config = cfg.get("n_workers", -1)
    num_workers = get_num_workers(n_workers_config)

    # Get localization method (always CPU-based MLE)
    method = get_localization_method()

    # Save run info (metadata only, no parameters since they vary by protein)
    run_info = {
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "method": method,
        },
        "input_data": {
            "input_folder": str(cfg["input_folder"]),
            "proteins": cfg["proteins"],
            "protein_channel": cfg.get("protein_channel", "640"),
            "has_ground_truth": has_ground_truth,
        },
    }

    if has_ground_truth:
        run_info["input_data"]["ground_truth_channel"] = cfg.get("ground_truth_channel", "488")

    run_info_file = output_folder / "run_info.yaml"
    with run_info_file.open("w") as f:
        yaml.dump(run_info, f, default_flow_style=False, sort_keys=False)
    run_info_file.chmod(0o644)  # ensure readable by all pipeline steps regardless of umask

    logging.info("Run metadata saved (protein-specific parameters saved in each protein folder)")

    # Find Picasso Python
    try:
        picasso_python = get_picasso_python()
    except RuntimeError as e:
        logging.error(str(e))
        sys.exit(1)

    # Process experiments (already discovered above)
    if not experiments:
        logging.error(f"No experiments found in {input_root}")
        return

    logging.info(f"\nProcessing {len(experiments)} experiments")

    # Process each experiment
    success_count = 0
    fail_count = 0
    processed_files = []

    for exp_path in experiments:
        logging.info(f"\nProcessing experiment: {exp_path.name}")

        # Find protein folders
        protein_folders = discover_protein_folders(exp_path, proteins)

        if not protein_folders:
            logging.warning(f"  No protein folders found in {exp_path.name}")
            continue

        # Process each protein
        for protein, protein_path in protein_folders.items():
            logging.info(f"  Processing protein: {protein}")

            # Try to load optimized parameters (experiment-level first, then protein-level)
            opt_params = load_optimized_params(protein_path, has_ground_truth, experiment_folder=exp_path)

            # Create working config for this protein/experiment
            protein_cfg = cfg.copy()

            if opt_params:
                # Use optimized gradient parameters (boxsize always from config)
                protein_cfg["gradient_protein"] = opt_params.get("gradient_protein", cfg.get("gradient_protein", 10000))
                if has_ground_truth:
                    protein_cfg["gradient_ground_truth"] = opt_params.get("gradient_ground_truth", cfg.get("gradient_ground_truth", 1000))
                logging.info(f"    gradient_protein: {protein_cfg['gradient_protein']} (optimized)")
                if has_ground_truth:
                    logging.info(f"    gradient_ground_truth: {protein_cfg['gradient_ground_truth']} (optimized)")
                logging.info(f"    boxsize: {protein_cfg['boxsize']} (from config)")
                param_source = "optimized"
            else:
                logging.info(f"    Using default parameters from config.yaml")
                logging.info(f"    gradient_protein: {protein_cfg['gradient_protein']}")
                if has_ground_truth:
                    logging.info(f"    gradient_ground_truth: {protein_cfg.get('gradient_ground_truth', 'N/A')}")
                logging.info(f"    boxsize: {protein_cfg['boxsize']}")
                param_source = "config_defaults"

            # Save protein-specific parameters used for this localization run
            save_protein_localization_params(
                protein_path=protein_path,
                protein=protein,
                experiment=exp_path.name,
                protein_cfg=protein_cfg,
                method=method,
                has_ground_truth=has_ground_truth,
                source=param_source,
            )

            # Find ND2 files in this protein folder
            nd2_files = find_nd2_files(protein_path)

            if not nd2_files:
                logging.warning(f"    No ND2 files found")
                continue

            # Filter files based on has_ground_truth
            if has_ground_truth:
                # Process both protein and ground truth channels
                nd2_files_to_process = sorted(set(nd2_files))
            else:
                # Process only protein channel files
                nd2_files_to_process = sorted(set([f for f in nd2_files if protein_channel in f.name]))

            if not nd2_files_to_process:
                logging.warning(f"    No ND2 files to process (check channel filters)")
                continue

            # Auto-detect FOV size from first file
            try:
                fov_size = get_fov_size(nd2_files_to_process[0])
                protein_cfg["size_FOV"] = fov_size
                logging.info(f"    Auto-detected FOV size: {fov_size}x{fov_size} pixels")
            except Exception as e:
                logging.warning(f"    Could not auto-detect FOV size: {e}")
                if "size_FOV" in cfg:
                    protein_cfg["size_FOV"] = cfg["size_FOV"]
                    logging.info(f"    Using FOV size from config: {cfg['size_FOV']}")
                else:
                    logging.warning(f"    No FOV size available, proceeding without it")

            logging.info(f"    Found {len(nd2_files_to_process)} ND2 files to process")

            # Prepare work items for parallel/sequential processing
            work_items = [
                (nd2_file, detect_channel_gradient(nd2_file.name, protein_cfg), method, protein_cfg, picasso_python)
                for nd2_file in nd2_files_to_process
            ]

            # Process files (parallel or sequential based on num_workers)
            if num_workers == 1:
                # Sequential processing
                for work_item in work_items:
                    nd2_file = work_item[0]
                    result = localize_worker(work_item)
                    nd2_file, success, error_msg = result
                    if success:
                        processed_files.append(nd2_file)
                        success_count += 1
                    else:
                        logging.error(f"    Failed to process {nd2_file.name}: {error_msg}")
                        fail_count += 1
            else:
                # Parallel processing
                import multiprocessing as mp
                with mp.Pool(processes=num_workers) as pool:
                    results = pool.map(localize_worker, work_items)

                # Process results
                for nd2_file, success, error_msg in results:
                    if success:
                        processed_files.append(nd2_file)
                        success_count += 1
                    else:
                        logging.error(f"    Failed to process {nd2_file.name}: {error_msg}")
                        fail_count += 1

    # Copy localization files to output folder
    logging.info("\nCopying localization files to output folder...")
    copied_count = 0

    for nd2_file in processed_files:
        # Localization files are next to ND2 files
        locs_hdf5 = nd2_file.parent / f"{nd2_file.stem}_locs.hdf5"
        locs_yaml = nd2_file.parent / f"{nd2_file.stem}_locs.yaml"

        # Create output directory structure mirroring input
        relative_path = nd2_file.parent.relative_to(input_root)
        output_dir = output_folder / relative_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        if locs_hdf5.exists():
            shutil.copy2(locs_hdf5, output_dir / locs_hdf5.name)
            copied_count += 1
        if locs_yaml.exists():
            shutil.copy2(locs_yaml, output_dir / locs_yaml.name)

    logging.info(f"Copied {copied_count} localization files")

    # Summary
    logging.info("=" * 60)
    logging.info(f"Localization complete: {success_count} succeeded, {fail_count} failed")
    logging.info(f"Results saved to: {output_folder}")
    logging.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        logging.error(f"Fatal error: {ex}")
        sys.exit(1)
