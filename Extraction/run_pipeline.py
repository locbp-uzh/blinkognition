#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH), Andreas Biri (ETHZ).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Orchestrate the complete extraction pipeline.

Pipeline steps:
  1. paramfinder: Optimize localization parameters per protein/experiment
  2. localize: Run Picasso localization with optimized parameters
  3. extract: Extract intensity traces from localizations
  4. combine: Combine traces by compound and apply preprocessing
  5. filter: Filter traces by quality and optionally augment
  6. diagnose: Analyze trace quality and generate diagnostic reports

Usage:
  # Full pipeline from scratch (includes parameter optimization)
  python run_pipeline.py -c config.yaml

  # Resume from a specific step
  python run_pipeline.py -c config.yaml --start-from extract -r Results/Extract/Grx1_K20Ac_grad640-12000_grad488-4000_001

  # Skip parameter optimization (use existing or config defaults)
  python run_pipeline.py -c config.yaml --start-from localize
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import yaml
from pathlib import Path


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def find_picasso_python() -> str:
    """
    Find the picasso-env Python interpreter.

    Returns:
        Path to picasso-env Python.

    Raises:
        RuntimeError if not found.
    """
    # First check if we're already running in picasso-env
    current_python = sys.executable
    if "picasso-env" in current_python:
        logging.info(f"Already running in picasso-env: {current_python}")
        return current_python

    # Try to find picasso-env using conda/mamba
    for cmd in ["mamba", "conda"]:
        try:
            result = subprocess.run(
                [cmd, "env", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "picasso-env" in line:
                        # Extract path from line (format: "name    /path/to/env")
                        parts = line.split()
                        if len(parts) >= 2:
                            env_path = Path(parts[-1]) / "bin" / "python"
                            if env_path.exists():
                                logging.info(f"Found picasso-env via {cmd}: {env_path}")
                                return str(env_path)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # Try common installation locations
    candidates = [
        Path.home() / "mambaforge/envs/picasso-env/bin/python",
        Path.home() / "miniforge3/envs/picasso-env/bin/python",
        Path.home() / "opt/anaconda3/envs/picasso-env/bin/python",
        Path.home() / "anaconda3/envs/picasso-env/bin/python",
        Path.home() / "miniconda3/envs/picasso-env/bin/python",
    ]

    for candidate in candidates:
        if candidate.exists():
            logging.info(f"Found picasso-env Python: {candidate}")
            return str(candidate)

    raise RuntimeError(
        "Could not find picasso-env Python interpreter.\n"
        "Please ensure picasso-env is installed.\n"
        "Tip: Try running this script from within the picasso-env environment."
    )


def load_config(config_path: Path) -> dict:
    """Load configuration file."""
    with config_path.open("r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg


def parse_run_folder_from_output(output: str) -> str | None:
    """
    Parse run folder name from localize.py output.

    Args:
        output: stdout/stderr from localize.py.

    Returns:
        Run folder path if found, None otherwise.
    """
    # Look for patterns like: "Output folder: Results/Extract/Grx1_K20Ac_grad640-12000_grad488-4000_001"
    pattern = r"Output folder:\s+(.+?)(?:\n|$)"
    match = re.search(pattern, output)
    if match:
        return match.group(1).strip()

    # Alternative: look for "Results saved to: ..."
    pattern = r"Results saved to:\s+(.+?)(?:\n|$)"
    match = re.search(pattern, output)
    if match:
        return match.group(1).strip()

    return None


def run_step(
    step_name: str,
    script_path: Path,
    config_path: Path,
    python_path: str,
    run_folder: Path | None = None,
    experiment: str | None = None,
    protein: str | None = None,
) -> tuple[bool, str]:
    """
    Run a single pipeline step.

    Args:
        step_name: Name of the step (for logging).
        script_path: Path to the Python script.
        config_path: Path to config.yaml.
        python_path: Path to Python interpreter.
        run_folder: Run folder (required for extract/combine/filter).
        experiment: Filter to specific experiment (only for paramfinder/localize).
        protein: Filter to specific protein (only for paramfinder/localize/diagnose).

    Returns:
        Tuple of (success: bool, output: str).
    """
    logging.info("=" * 60)
    logging.info(f"Running step: {step_name}")
    logging.info("=" * 60)

    cmd = [python_path, str(script_path), "-c", str(config_path)]

    # Add run folder for steps that need it
    if run_folder:
        cmd.extend(["-r", str(run_folder)])

    # Add filtering options for steps that support them
    if step_name in ["paramfinder", "localize"]:
        if experiment:
            cmd.extend(["--experiment", experiment])
        if protein:
            cmd.extend(["--protein", protein])
    elif step_name == "diagnose":
        if protein:
            cmd.extend(["--protein", protein])

    logging.info(f"Command: {' '.join(cmd)}\n")

    try:
        # For localize step, capture output to parse run folder
        # For other steps, stream output in real-time
        if step_name == "localize":
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            output = result.stdout + result.stderr
            print(output)
        else:
            # Stream output in real-time
            result = subprocess.run(
                cmd,
                check=False,
            )
            output = ""  # No output captured for non-localize steps

        if result.returncode != 0:
            logging.error(f"Step '{step_name}' failed with exit code {result.returncode}")
            return False, output

        logging.info(f"Step '{step_name}' completed successfully\n")
        return True, output

    except Exception as e:
        logging.error(f"Error running step '{step_name}': {e}")
        return False, str(e)


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Orchestrate the complete extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline from scratch (includes parameter optimization)
  python run_pipeline.py -c config.yaml

  # Skip parameter optimization (use existing optimized params or config defaults)
  python run_pipeline.py -c config.yaml --start-from localize

  # Resume from extract step
  python run_pipeline.py -c config.yaml --start-from extract -r Results/Extract/Grx1_K20Ac_grad640-12000_grad488-4000_001

  # Resume from combine step
  python run_pipeline.py -c config.yaml --start-from combine -r Results/Extract/Grx1_K20Ac_grad640-12000_grad488-4000_001

  # Run only diagnostics on existing filtered data
  python run_pipeline.py -c config.yaml --start-from diagnose -r Results/Extract/Grx1_K20Ac_grad640-12000_grad488-4000_001

  # Debug: Process only one protein in one experiment (useful for testing)
  python run_pipeline.py -c config.yaml --experiment Exp1 --protein Grx1

  # Debug: Process only one protein across all experiments
  python run_pipeline.py -c config.yaml --protein Grx1
        """,
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
        "--start-from",
        dest="start_from",
        type=str,
        choices=["paramfinder", "localize", "extract", "combine", "filter", "diagnose"],
        help="Step to start from (overrides config.yaml)",
    )
    parser.add_argument(
        "-r",
        "--run-folder",
        dest="run_folder",
        type=str,
        help="Run folder to resume from (overrides config.yaml, required if start_from != localize)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Process only this experiment (e.g., 'Exp1'). Only affects paramfinder/localize steps.",
    )
    parser.add_argument(
        "--protein",
        type=str,
        default=None,
        help="Process only this protein (e.g., 'Grx1'). Only affects paramfinder/localize steps.",
    )

    args = parser.parse_args()

    # Setup
    setup_logging()
    config_path = Path(args.config_path).resolve()

    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)

    # Determine start point (CLI overrides config)
    start_from = args.start_from if args.start_from else cfg.get("start_from", "paramfinder")
    run_folder_str = args.run_folder if args.run_folder else cfg.get("resume_run_folder")

    # Validate inputs
    if start_from not in ["paramfinder", "localize"] and not run_folder_str:
        logging.error(
            f"--run-folder/-r is required when starting from '{start_from}'\n"
            "Alternatively, set 'resume_run_folder' in config.yaml"
        )
        sys.exit(1)

    run_folder = Path(run_folder_str).resolve() if run_folder_str else None

    # Find picasso-env Python
    try:
        python_path = find_picasso_python()
    except RuntimeError as e:
        logging.error(str(e))
        sys.exit(1)

    # Define pipeline steps
    script_dir = config_path.parent
    steps = [
        ("paramfinder", script_dir / "paramfinder.py", False),  # (name, script, needs_run_folder)
        ("localize", script_dir / "localize.py", False),
        ("extract", script_dir / "extract.py", True),
        ("combine", script_dir / "combine.py", True),
        ("filter", script_dir / "filter.py", True),
        ("diagnose", script_dir / "diagnose.py", True),
    ]

    # Find starting index
    step_names = [s[0] for s in steps]
    try:
        start_idx = step_names.index(start_from)
    except ValueError:
        logging.error(f"Invalid start_from value: {start_from}")
        sys.exit(1)

    logging.info(f"Starting pipeline from step: {start_from}")
    if run_folder:
        logging.info(f"Using run folder: {run_folder}")
    if args.experiment or args.protein:
        filters = []
        if args.experiment:
            filters.append(f"experiment={args.experiment}")
        if args.protein:
            filters.append(f"protein={args.protein}")
        logging.info(f"Filtering enabled: {', '.join(filters)} (affects paramfinder/localize steps)")

    # Execute pipeline steps
    for i in range(start_idx, len(steps)):
        step_name, script_path, needs_run_folder = steps[i]

        # Skip diagnose step if disabled in config
        if step_name == "diagnose":
            diag_cfg = cfg.get("diagnostics", {})
            if not diag_cfg.get("enabled", True):
                logging.info("Skipping diagnose step (disabled in config)")
                continue

        # Determine run folder for this step
        step_run_folder = run_folder if needs_run_folder else None

        # Run the step
        success, output = run_step(
            step_name,
            script_path,
            config_path,
            python_path,
            step_run_folder,
            args.experiment,
            args.protein,
        )

        if not success:
            logging.error(f"Pipeline failed at step: {step_name}")
            sys.exit(1)

        # Parse run folder from localize.py output
        if step_name == "localize":
            parsed_run_folder = parse_run_folder_from_output(output)
            if parsed_run_folder:
                run_folder = Path(parsed_run_folder).resolve()
                logging.info(f"Detected run folder: {run_folder}")
            else:
                logging.error("Could not parse run folder from localize.py output")
                sys.exit(1)

    # Success!
    logging.info("=" * 60)
    logging.info("Pipeline completed successfully!")
    logging.info("=" * 60)
    if run_folder:
        logging.info(f"Results saved to: {run_folder}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as ex:
        logging.error(f"Fatal error: {ex}")
        sys.exit(1)
