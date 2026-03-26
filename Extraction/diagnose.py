#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Diagnostic tool for visualizing example protein and background traces.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal

from joblib import Parallel, delayed

from utils import setup_logging, gmm_classify_frames

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Plotting Standards (see docs/plotting_standards.md)
# =============================================================================
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.titlesize'] = 7
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 450
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['lines.linewidth'] = 1.2
mpl.rcParams['patch.linewidth'] = 0.8
mpl.rcParams['axes.grid'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# Font sizes
FONTSIZE_LABEL = 7
FONTSIZE_TICK = 6
FONTSIZE_TITLE = 7
FONTSIZE_LEGEND = 6

# Okabe-Ito color-blind friendly palette
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


def apply_axis_standards(ax):
    """Apply standard axis formatting per plotting standards."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICK,
                   length=4, width=0.8, direction='out')
    ax.tick_params(axis='both', which='minor', length=2, width=0.6, direction='out')


def load_config(config_path: Path) -> dict:
    """Load configuration file."""
    with config_path.open("r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def load_run_info(run_folder: Path) -> dict:
    """Load run info from run folder."""
    info_file = run_folder / "run_info.yaml"
    if not info_file.exists():
        raise FileNotFoundError(f"Run info not found: {info_file}")
    with info_file.open("r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def extract_experiment_from_path(origin_path: str, run_folder: Path) -> str:
    """
    Extract experiment name from origin path.

    Path format: .../run_folder/Exp1/Protein/movie_traces.pkl
    Returns experiment name (e.g., "Exp1")
    """
    try:
        path = Path(origin_path)
        parts = path.parts

        # Find run folder in path
        run_folder_name = run_folder.name
        for i, part in enumerate(parts):
            if part == run_folder_name and i < len(parts) - 2:
                # Next part should be experiment name
                return parts[i + 1]

        # Fallback: look for pattern like "Exp*" or similar
        for part in parts:
            if part.startswith("Exp") or part.startswith("exp"):
                return part

        return "Unknown"
    except Exception:
        return "Unknown"


def _analyze_particle_worker(particle, trace_vals, bg_rm_vals, cfg, experiment):
    """Compute metrics for one particle. Top-level function for joblib pickling."""
    trace = pd.Series(trace_vals)
    x_peaks, gmm_snr = detect_peaks_gmm(bg_rm_vals, cfg)
    metrics = compute_trace_metrics(trace, cfg, x_peaks=x_peaks, gmm_snr=gmm_snr)
    metrics["particle"] = particle
    metrics["threshold"] = np.nan
    metrics["experiment"] = experiment
    return metrics


def _analyze_bg_particle_worker(particle, trace_vals, bg_rm_vals):
    """Compute basic metrics for one background particle. Top-level for joblib pickling."""
    trace = pd.Series(trace_vals)
    mean_signal = trace.mean()
    std_signal = trace.std()
    max_intensity = trace.max()
    min_intensity = trace.min()

    _, _, _, separation = gmm_classify_frames(
        bg_rm_vals.astype(np.float64), proba_threshold=0.8, return_separation=True
    )
    snr = separation
    peaks = signal.find_peaks(trace, height=trace.std() * 0.5, width=2)
    n_peaks = len(peaks[0])
    return {
        "particle": particle,
        "mean_signal": mean_signal,
        "std_signal": std_signal,
        "max_intensity": max_intensity,
        "min_intensity": min_intensity,
        "snr": snr,
        "n_peaks": n_peaks,
    }


def detect_peaks_gmm(bg_rm_trace: np.ndarray, cfg: dict) -> tuple:
    """
    Detect peaks using GMM posterior probabilities, mirroring filter.py logic.

    Returns (x_peaks, gmm_snr) where:
        x_peaks: array of peak frame indices (argmax of bg_rm within each signal run).
        gmm_snr: GMM component separation = (signal_mean - noise_mean) / noise_std.
            Low values (~1-3) indicate noise; high values indicate real signal.
    """
    MIN_PEAK_WIDTH = int(cfg["min_peak_width"])
    GMM_PROBA_THRESHOLD = float(cfg.get("gmm_proba_threshold", 0.8))
    GMM_MIN_SEPARATION = float(cfg.get("gmm_min_separation", 3.0))
    SNR_MIN_SEPARATION = float(cfg.get("snr_min_separation", 3.0))

    _, _, signal_mask, separation = gmm_classify_frames(
        bg_rm_trace, GMM_PROBA_THRESHOLD, GMM_MIN_SEPARATION, return_separation=True
    )

    # Mirror the explicit SNR gate in filter.py
    if SNR_MIN_SEPARATION > 0 and separation < SNR_MIN_SEPARATION:
        return np.array([]), separation

    runs = []
    in_run = False
    for i, s in enumerate(signal_mask):
        if s and not in_run:
            run_start = i
            in_run = True
        elif not s and in_run:
            runs.append((run_start, i - 1))
            in_run = False
    if in_run:
        runs.append((run_start, len(signal_mask) - 1))

    runs = [(l, r) for l, r in runs if (r - l + 1) >= MIN_PEAK_WIDTH]
    x_peaks = np.array([l + np.argmax(bg_rm_trace[l:r + 1]) for l, r in runs])
    return x_peaks, separation


def compute_trace_metrics(
    trace: pd.Series,
    cfg: dict,
    x_peaks: np.ndarray,
    gmm_snr: float,
) -> Dict:
    """
    Compute quality metrics for a single trace.

    Returns dict with:
        - snr: GMM component separation (signal_mean - noise_mean) / noise_std
        - n_peaks: Number of peaks detected
        - first_peak_time: Frame of first peak
        - last_peak_time: Frame of last peak
        - delta_first_second: Distance between first two peaks
        - mean_signal: Mean intensity
        - std_signal: Std of intensity
        - max_intensity: Maximum intensity
        - pass_filter: Whether trace passes all filter criteria
        - fail_reasons: List of reasons why trace failed (if applicable)
    """
    MIN_PEAK_NUMBER = int(cfg["min_peak_number"])
    FIRST_PEAK_TIME = int(cfg["first_peak_time"])
    LAST_PEAK_TIME = int(cfg["last_peak_time"])
    DELTA_FIRST_SECOND = int(cfg["delta_first_second"])

    # Basic statistics
    mean_signal = trace.mean()
    std_signal = trace.std()
    max_intensity = trace.max()

    snr = gmm_snr
    n_peaks = len(x_peaks)

    # Peak timing
    first_peak_time = x_peaks[0] if n_peaks > 0 else -1
    last_peak_time = x_peaks[-1] if n_peaks > 0 else -1
    delta_first_second = (x_peaks[1] - x_peaks[0]) if n_peaks >= 2 else -1

    # Check filtering criteria
    fail_reasons = []

    if n_peaks <= MIN_PEAK_NUMBER:
        fail_reasons.append(f"n_peaks={n_peaks} <= {MIN_PEAK_NUMBER}")
    if n_peaks == 0:
        fail_reasons.append("n_peaks=0")
    if n_peaks > 0 and first_peak_time >= FIRST_PEAK_TIME:
        fail_reasons.append(f"first_peak={first_peak_time} >= {FIRST_PEAK_TIME}")
    if n_peaks < 2:
        fail_reasons.append(f"n_peaks={n_peaks} < 2 (cannot check delta)")
    elif delta_first_second >= DELTA_FIRST_SECOND:
        fail_reasons.append(f"delta_1-2={delta_first_second} >= {DELTA_FIRST_SECOND}")
    if n_peaks > 0 and last_peak_time <= LAST_PEAK_TIME:
        fail_reasons.append(f"last_peak={last_peak_time} <= {LAST_PEAK_TIME}")

    pass_filter = len(fail_reasons) == 0

    return {
        "snr": snr,
        "n_peaks": n_peaks,
        "first_peak_time": first_peak_time,
        "last_peak_time": last_peak_time,
        "delta_first_second": delta_first_second,
        "mean_signal": mean_signal,
        "std_signal": std_signal,
        "max_intensity": max_intensity,
        "pass_filter": pass_filter,
        "fail_reasons": fail_reasons,
        "x_peaks": x_peaks,
    }


def analyze_traces(
    compound: str,
    gt_label: str,
    cfg: dict,
    run_folder: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze all traces for a compound/label and compute metrics.

    Returns:
        metrics_df: DataFrame with one row per trace (includes experiment info)
        traces: The original z-scored traces
    """
    method_tag = "gmm"

    # Determine file naming
    if gt_label:
        label_str = f"_{gt_label}"
        log_label = f"{compound}_{gt_label}"
    else:
        label_str = ""
        log_label = compound

    # Load z-scored traces
    zscored_path = run_folder / f"{compound}{label_str}_{method_tag}_all_zscored_traces.pkl"

    if not zscored_path.exists():
        logging.warning(f"No z-scored traces found for {log_label}")
        return pd.DataFrame(), pd.DataFrame()

    traces = pd.read_pickle(zscored_path)

    if traces.empty:
        logging.warning(f"Empty traces for {log_label}")
        return pd.DataFrame(), pd.DataFrame()

    # Remove traces with zeros
    traces = traces.copy()
    traces.replace(0, np.nan, inplace=True)
    traces.dropna(axis="columns", inplace=True)

    # Load uniqueID file to get experiment information
    uniqueID_path = run_folder / f"_{compound}_uniqueID_{gt_label if gt_label else 'all'}.pkl"
    experiment_map = {}

    if uniqueID_path.exists():
        try:
            uniqueID_df = pd.read_pickle(uniqueID_path)
            # Extract experiment for each uniqueID
            uniqueID_df['experiment'] = uniqueID_df['origin'].apply(
                lambda x: extract_experiment_from_path(x, run_folder)
            )
            # Create mapping: uniqueID -> experiment
            experiment_map = dict(zip(uniqueID_df['uniqueID'], uniqueID_df['experiment']))
        except Exception as e:
            logging.warning(f"Could not load experiment info from {uniqueID_path}: {e}")

    # Load bg_rm traces for GMM peak detection
    bg_rm_path = run_folder / f"{compound}{label_str}_{method_tag}_all_bg_rm_traces.pkl"
    if not bg_rm_path.exists():
        logging.warning(f"bg_rm traces not found for {log_label}")
        return pd.DataFrame(), pd.DataFrame()
    traces_bg_rm = pd.read_pickle(bg_rm_path)

    # Only process particles present in both traces and bg_rm
    common_particles = [p for p in traces.columns if p in traces_bg_rm.columns]
    if len(common_particles) < len(traces.columns):
        logging.warning(f"  {len(traces.columns) - len(common_particles)} particles missing bg_rm, skipping")

    # Compute metrics for each trace (parallel over particles)
    n_workers = cfg.get("n_workers", -1)
    if n_workers is None:
        n_workers = -1

    metrics_list = Parallel(n_jobs=n_workers)(
        delayed(_analyze_particle_worker)(
            particle,
            traces[particle].values,
            traces_bg_rm[particle].values.astype(np.float64),
            cfg,
            experiment_map.get(particle, "Unknown"),
        )
        for particle in common_particles
    )

    metrics_df = pd.DataFrame(metrics_list)

    return metrics_df, traces


def plot_trace_examples(
    metrics_df: pd.DataFrame,
    traces: pd.DataFrame,
    cfg: dict,
    output_path: Path,
    n_examples: int = 12,
):
    """
    Plot example traces showing passed and failed cases.
    """
    # Select examples
    passed = metrics_df[metrics_df["pass_filter"]]
    failed = metrics_df[~metrics_df["pass_filter"]]

    n_pass = min(n_examples // 2, len(passed))
    n_fail = min(n_examples // 2, len(failed))

    # Sample traces
    pass_samples = passed.sample(n=n_pass) if n_pass > 0 else pd.DataFrame()
    fail_samples = failed.sample(n=n_fail) if n_fail > 0 else pd.DataFrame()

    samples = pd.concat([pass_samples, fail_samples])

    if len(samples) == 0:
        logging.warning("No traces to plot")
        return

    # Create output directory
    output_dir = output_path.parent / output_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create grid
    n_cols = 3
    n_rows = int(np.ceil(len(samples) / n_cols))

    fig = plt.figure(figsize=(15, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    trace_data = []
    for idx, (_, row) in enumerate(samples.iterrows()):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])

        particle = row["particle"]
        trace = traces[particle]

        # Plot trace
        status = "PASS" if row["pass_filter"] else "FAIL"
        trace_color = COLORS['green'] if row["pass_filter"] else COLORS['vermillion']
        ax.plot(trace.values, linewidth=0.8, alpha=0.8, color=trace_color)

        threshold = row.get("threshold", np.nan)
        if not (isinstance(threshold, float) and np.isnan(threshold)):
            ax.axhline(threshold, color=COLORS['orange'], linestyle="--",
                       linewidth=0.8, alpha=0.6, label="Threshold")

        # Mark peaks — use stored x_peaks (works for both GMM and threshold modes)
        x_peaks_row = row.get("x_peaks")
        if x_peaks_row is not None and len(x_peaks_row) > 0:
            ax.plot(x_peaks_row, trace.iloc[x_peaks_row], "x",
                    color=COLORS['vermillion'], markersize=5)

        # Title with status and key metrics (sentence-style capitalization)
        title = (
            f"{status}: {particle}\n"
            f"SNR={row['snr']:.1f}, peaks={row['n_peaks']}, "
            f"first={row['first_peak_time']}, last={row['last_peak_time']}"
        )
        ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=9, color=trace_color)

        # Add fail reasons if applicable
        if not row["pass_filter"] and row["fail_reasons"]:
            reasons_text = "\n".join(row["fail_reasons"][:3])
            ax.text(
                0.02, 0.98, reasons_text,
                transform=ax.transAxes,
                fontsize=FONTSIZE_TICK,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3)
            )

        ax.set_xlabel("Frame", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Z-scored intensity", fontsize=FONTSIZE_LABEL)
        ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
        apply_axis_standards(ax)

        # Collect data for CSV
        trace_data.append({
            'particle': particle,
            'status': status,
            'snr': row['snr'],
            'n_peaks': row['n_peaks'],
            'first_peak_time': row['first_peak_time'],
            'last_peak_time': row['last_peak_time'],
        })

    plt.tight_layout(h_pad=3.0, w_pad=3.0)
    plt.savefig(output_dir / "plot.pdf", dpi=450, bbox_inches="tight")
    plt.close()

    # Save trace metadata
    pd.DataFrame(trace_data).to_csv(output_dir / "data_panel_traces.csv", index=False)

    logging.info(f"Saved trace examples: {output_dir}")


def analyze_background_traces(
    compound: str,
    cfg: dict,
    run_folder: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze background traces for a compound.

    Background traces have no filtering - they are used for comparison.

    Returns:
        metrics_df: DataFrame with basic metrics per trace
        traces: The z-scored background traces
    """
    # Load z-scored background traces
    zscored_path = run_folder / f"{compound}_background_gmm_all_zscored_traces.pkl"

    if not zscored_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    traces = pd.read_pickle(zscored_path)

    if traces.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Remove traces with zeros
    traces = traces.copy()
    traces.replace(0, np.nan, inplace=True)
    traces.dropna(axis="columns", inplace=True)

    if traces.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Load bg_rm traces for GMM-based SNR
    bg_rm_path = run_folder / f"{compound}_background_gmm_all_bg_rm_traces.pkl"
    bg_rm_traces = pd.read_pickle(bg_rm_path) if bg_rm_path.exists() else None

    # Compute basic metrics for each trace (parallel over particles, no filtering)
    n_workers = cfg.get("n_workers", -1)
    if n_workers is None:
        n_workers = -1

    metrics_list = Parallel(n_jobs=n_workers)(
        delayed(_analyze_bg_particle_worker)(
            particle,
            traces[particle].values,
            bg_rm_traces[particle].values.astype(np.float64)
            if bg_rm_traces is not None and particle in bg_rm_traces.columns
            else np.zeros(len(traces[particle])),
        )
        for particle in traces.columns
    )

    metrics_df = pd.DataFrame(metrics_list)

    return metrics_df, traces


def plot_background_examples(
    bg_traces: pd.DataFrame,
    output_path: Path,
    compound: str,
    n_examples: int = 12,
    bg_rm_traces: pd.DataFrame = None,
    cfg: dict = None,
):
    """
    Plot example background traces.
    """
    if bg_traces.empty:
        logging.warning(f"No background traces to plot for {compound}")
        return

    # Create output directory
    output_dir = output_path.parent / output_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample traces
    n_samples = min(n_examples, len(bg_traces.columns))
    sample_cols = np.random.choice(bg_traces.columns, size=n_samples, replace=False)

    # Create grid
    n_cols = 3
    n_rows = int(np.ceil(n_samples / n_cols))

    fig = plt.figure(figsize=(15, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    trace_data = []
    for idx, particle in enumerate(sample_cols):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])

        trace = bg_traces[particle]

        # Plot trace
        ax.plot(trace.values, linewidth=0.8, alpha=0.8, color=COLORS['sky_blue'])

        # Compute SNR: GMM separation if bg_rm available, else last-50-frames ratio
        if bg_rm_traces is not None and particle in bg_rm_traces.columns:
            proba_thr = float(cfg.get("gmm_proba_threshold", 0.8)) if cfg else 0.8
            _, _, _, snr = gmm_classify_frames(
                bg_rm_traces[particle].values.astype(np.float64),
                proba_threshold=proba_thr,
                return_separation=True,
            )
            peaks = signal.find_peaks(trace, height=trace.std() * 0.5, width=2)
        else:
            snr = trace.std() / trace.iloc[-50:].std() if trace.iloc[-50:].std() > 0 else 0
            peaks = signal.find_peaks(trace, height=trace.iloc[-50:].std() * 3, width=2)
        n_peaks = len(peaks[0])

        title = (
            f"Background: {particle}\n"
            f"SNR={snr:.1f}, peaks={n_peaks}, "
            f"mean={trace.mean():.2f}, std={trace.std():.2f}"
        )
        ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=9)

        ax.set_xlabel("Frame", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Z-scored intensity", fontsize=FONTSIZE_LABEL)
        apply_axis_standards(ax)

        trace_data.append({
            'particle': particle,
            'snr': snr,
            'n_peaks': n_peaks,
            'mean': trace.mean(),
            'std': trace.std(),
        })

    fig.suptitle(f"Background trace examples - {compound}", fontsize=FONTSIZE_TITLE + 2, y=1.02)
    plt.tight_layout(h_pad=3.0, w_pad=3.0)
    plt.savefig(output_dir / "plot.pdf", dpi=450, bbox_inches="tight")
    plt.close()

    pd.DataFrame(trace_data).to_csv(output_dir / "data_panel_traces.csv", index=False)
    logging.info(f"Saved background trace examples: {output_dir}")


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Diagnostic tool for trace quality analysis"
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
        help="Path to localization run folder",
    )
    parser.add_argument(
        "--protein",
        type=str,
        default=None,
        help="Analyze specific protein only (optional, overrides config)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=None,
        help="Number of example traces to plot (optional, overrides config)",
    )
    args = parser.parse_args()

    # Setup
    setup_logging("INFO")
    cfg = load_config(Path(args.config_path))

    # Load run info
    run_folder = Path(args.run_folder).expanduser()
    run_info = load_run_info(run_folder)

    # Get proteins and input folder
    proteins = list(run_info["input_data"]["proteins"])
    has_ground_truth = run_info["input_data"].get("has_ground_truth", True)

    # Get diagnostic settings from config or CLI
    diag_cfg = cfg.get("diagnostics", {})
    n_examples = args.n_examples if args.n_examples is not None else diag_cfg.get("n_examples", 12)
    config_proteins = diag_cfg.get("proteins")

    # Filter to specific protein (CLI overrides config)
    if args.protein:
        if args.protein in proteins:
            proteins = [args.protein]
        else:
            logging.error(f"Protein '{args.protein}' not found in run info")
            sys.exit(1)
    elif config_proteins:
        # Use proteins from config if specified
        config_proteins_list = config_proteins if isinstance(config_proteins, list) else [config_proteins]
        valid_proteins = [p for p in config_proteins_list if p in proteins]
        if valid_proteins:
            proteins = valid_proteins
        else:
            logging.warning(f"No valid proteins found in config diagnostics.proteins, analyzing all")
    # Otherwise analyze all proteins

    logging.info(f"Run folder: {run_folder}")
    logging.info(f"Analyzing proteins: {', '.join(proteins)}")
    logging.info(f"Has ground truth: {has_ground_truth}")
    logging.info(f"Number of example traces to plot: {n_examples}")
    logging.info("")

    # Create diagnostics output folder
    diag_folder = run_folder / "diagnostics"
    diag_folder.mkdir(exist_ok=True)

    # Process each compound: plot protein trace examples
    for compound in proteins:
        logging.info(f"Analyzing compound: {compound}")

        if has_ground_truth:
            gt_labels = ["IN", "OUT"]
        else:
            gt_labels = [""]

        for gt_label in gt_labels:
            log_label = f"{compound}_{gt_label}" if gt_label else compound

            try:
                # Analyze traces
                metrics_df, traces = analyze_traces(compound, gt_label, cfg, run_folder)

                if metrics_df.empty:
                    logging.warning(f"No traces found for {log_label}, skipping")
                    continue

                # Plot trace examples
                label_suffix = f"_{gt_label}" if gt_label else ""
                examples_path = diag_folder / f"{compound}{label_suffix}_trace_examples.png"
                plot_trace_examples(metrics_df, traces, cfg, examples_path, n_examples)

                logging.info("")

            except Exception as e:
                logging.error(f"Error analyzing {log_label}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # =========================================================================
    # Background Trace Analysis
    # =========================================================================
    logging.info("")
    logging.info("=" * 70)
    logging.info("BACKGROUND TRACE ANALYSIS")
    logging.info("=" * 70)

    for compound in proteins:
        try:
            # Check if background traces exist for this compound
            bg_zscored_path = run_folder / f"{compound}_background_gmm_all_zscored_traces.pkl"

            if not bg_zscored_path.exists():
                logging.info(f"No background traces found for {compound}, skipping")
                continue

            logging.info(f"Analyzing background traces for {compound}...")

            # Analyze background traces
            bg_metrics_df, bg_traces = analyze_background_traces(compound, cfg, run_folder)

            if bg_metrics_df.empty:
                logging.warning(f"Empty background traces for {compound}")
                continue

            logging.info(f"  Found {len(bg_metrics_df)} background traces")

            # Plot background trace examples
            bg_examples_path = diag_folder / f"{compound}_background_trace_examples.png"
            _bg_rm_path = run_folder / f"{compound}_background_gmm_all_bg_rm_traces.pkl"
            _bg_rm_traces = pd.read_pickle(_bg_rm_path) if _bg_rm_path.exists() else None
            plot_background_examples(bg_traces, bg_examples_path, compound, n_examples,
                                     bg_rm_traces=_bg_rm_traces, cfg=cfg)

        except Exception as e:
            logging.error(f"Error analyzing background traces for {compound}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logging.info("")
    logging.info("=" * 70)
    logging.info(f"Diagnostic analysis complete. Results in: {diag_folder}")
    logging.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        logging.error(f"Fatal error: {ex}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
