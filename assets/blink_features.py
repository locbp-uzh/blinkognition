#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Sonnet 4.6)
# __copyright__ = "Copyright 2026, UZH, Switzerland"

"""
Extract and plot blink feature distributions from MCD-filtered protein traces.

Traces are loaded directly from the traces_with_wasserstein.npz produced by a
training run.  No pkl file handling or fingerprint matching is required.

NPZ layout expected:
    traces               (N, 2, T)  channel 0 = minmax, channel 1 = zscored
    labels               (N,)       integer class indices
    wasserstein_distances (N,)      MCD certainty score per trace (0–1)
    class_names          (C,)       protein name strings

GMM peak detection uses:
    channel 0 (minmax) as input to gmm_classify_frames
    channel 1 (zscored) for per-peak mean intensity reporting

All traces that pass the WD threshold are analysed.  No trace is dropped based
on its peak properties.

Ten features are computed across two rows:

  Row 1 — per-peak distributions:
  A) Peak duration  — consecutive "on" frames per blink event, converted to ms
  B) Off-time       — dark-interval duration between consecutive peaks, converted to ms
  C) Peak intensity — mean z-scored signal within each blink event
  D) Duty cycle     — fraction of trace frames in the "on" state (per trace)
  E) Blinking rate  — peaks per second (per trace)

  Row 2 — per-trace kinetic summaries:
  F) Blinks per trace  — raw blink count
  G) Mean on-time      — mean peak duration per trace, converted to ms
  H) Mean off-time     — mean dark-interval per trace, converted to ms
  I) CV on-times       — coefficient of variation of peak durations per trace
  J) CV off-times      — coefficient of variation of off-times per trace

Usage:
    conda activate blink2env
    python assets/blink_features.py -c assets/blink_features_config_hthtl_htia_mcd.yaml

Output (in output_path/{proteins}_mcd_{timestamp}/):
    plot.pdf          — 2×5 violin figure
    data_panel_A.csv  — peak durations (ms), one column per protein
    data_panel_B.csv  — off-times (ms), one column per protein
    data_panel_C.csv  — peak mean intensities (z-score), one column per protein
    data_panel_D.csv  — per-trace duty cycles, one column per protein
    data_panel_E.csv  — per-trace blinking rates (peaks s⁻¹), one column per protein
    data_panel_F.csv  — blinks per trace, one column per protein
    data_panel_G.csv  — mean on-times (ms), one column per protein
    data_panel_H.csv  — mean off-times (ms), one column per protein
    data_panel_I.csv  — CV on-times, one column per protein
    data_panel_J.csv  — CV off-times, one column per protein
"""

import sys
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Extraction"))
from utils import gmm_classify_frames  # noqa: E402

# ── Plotting standards ──────────────────────────────────────────────────────────
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
mpl.rcParams['font.size'] = 9
mpl.rcParams['pdf.fonttype'] = 42

FONTSIZE_LABEL = 9
FONTSIZE_TICK  = 8
FONTSIZE_TITLE = 10

COLORS = [
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # green
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # purple
    '#000000',  # black
]
# ───────────────────────────────────────────────────────────────────────────────


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _extract_runs(signal_mask: np.ndarray, min_peak_width: int) -> list:
    """Return list of (start, end) inclusive run intervals from a boolean mask."""
    runs, in_run = [], False
    run_start = 0
    for i, s in enumerate(signal_mask):
        if s and not in_run:
            run_start, in_run = i, True
        elif not s and in_run:
            runs.append((run_start, i - 1))
            in_run = False
    if in_run:
        runs.append((run_start, len(signal_mask) - 1))
    return [(l, r) for l, r in runs if (r - l + 1) >= min_peak_width]


def _trace_features(
    minmax_vals: np.ndarray,
    zscored_vals: np.ndarray,
    gmm_proba: float,
    min_peak_width: int,
    frame_interval_ms: float,
) -> tuple[list, list, float, list, float]:
    """
    Compute blink features for a single trace.

    GMM is run on the minmax channel (min_separation=0 — no trace-level gating).
    Peak intensities are reported from the zscored channel.

    Returns:
        durations        — list of per-peak durations in frames
        mean_intensities — list of per-peak mean z-scored intensities
        duty_cycle       — fraction of frames classified as "on"
        off_times        — list of dark-interval durations in frames
        blinking_rate    — peaks per second
    """
    _, _, signal_mask = gmm_classify_frames(minmax_vals, gmm_proba, 0.0)

    runs             = _extract_runs(signal_mask, min_peak_width)
    durations        = [r - l + 1 for l, r in runs]
    mean_intensities = [float(np.mean(zscored_vals[l:r + 1])) for l, r in runs]
    duty_cycle       = float(np.sum(signal_mask) / len(signal_mask))
    off_times        = [runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)]
    movie_length_s   = len(minmax_vals) * frame_interval_ms / 1000.0
    blinking_rate    = len(runs) / movie_length_s if movie_length_s > 0 else 0.0
    return durations, mean_intensities, duty_cycle, off_times, blinking_rate


def process_protein(
    label: str,
    traces_array: np.ndarray,
    gmm_proba: float,
    min_peak_width: int,
    frame_interval_ms: float,
) -> list[dict]:
    """
    Extract blink features for one protein from a pre-filtered trace array.

    Args:
        label:         Protein name (for logging).
        traces_array:  (N, 2, T) — channel 0 = minmax, channel 1 = zscored.
        gmm_proba:     GMM posterior probability threshold.
        min_peak_width: Minimum run length (frames) to count as a peak.
        frame_interval_ms: Frame duration in milliseconds.

    Returns a list of per-trace dicts with keys:
        durations, intensities, duty_cycle, off_times, blinking_rate,
        n_blinks, mean_on, mean_off, cv_on, cv_off
    """
    nan = float('nan')
    traces = []
    for i in range(len(traces_array)):
        minmax_vals  = traces_array[i, 0, :]
        zscored_vals = traces_array[i, 1, :]
        dur, inten, dc, off, rate = _trace_features(
            minmax_vals, zscored_vals, gmm_proba, min_peak_width, frame_interval_ms,
        )
        n = len(dur)
        traces.append({
            'durations':     dur,
            'intensities':   inten,
            'duty_cycle':    dc,
            'off_times':     off,
            'blinking_rate': rate,
            'n_blinks':      n,
            'mean_on':       float(np.mean(dur))               if n >= 1       else nan,
            'mean_off':      float(np.mean(off))               if len(off) >= 1 else nan,
            'cv_on':         float(np.std(dur) / np.mean(dur)) if n >= 2       else nan,
            'cv_off':        float(np.std(off) / np.mean(off)) if len(off) >= 2 else nan,
        })

    logging.info("  %-12s  %d traces processed", label, len(traces_array))
    return traces


def _flatten(traces_by_protein: dict) -> tuple[
    dict, dict, dict, dict, dict, dict, dict, dict, dict, dict
]:
    """Flatten per-trace dicts into protein-level lists for plotting."""
    durations_all, intensities_all, duty_cycles_all = {}, {}, {}
    off_times_all, blinking_rates_all = {}, {}
    n_blinks_all, mean_on_all, mean_off_all, cv_on_all, cv_off_all = {}, {}, {}, {}, {}
    for protein, traces in traces_by_protein.items():
        durations_all[protein]      = [d for t in traces for d in t['durations']]
        intensities_all[protein]    = [i for t in traces for i in t['intensities']]
        duty_cycles_all[protein]    = [t['duty_cycle']    for t in traces]
        off_times_all[protein]      = [o for t in traces for o in t['off_times']]
        blinking_rates_all[protein] = [t['blinking_rate'] for t in traces]
        n_blinks_all[protein]       = [t['n_blinks']      for t in traces]
        mean_on_all[protein]        = [t['mean_on']       for t in traces]
        mean_off_all[protein]       = [t['mean_off']      for t in traces]
        cv_on_all[protein]          = [t['cv_on']         for t in traces]
        cv_off_all[protein]         = [t['cv_off']        for t in traces]
        logging.info(
            "  %-12s  %d traces  %d peaks",
            protein, len(traces), len(durations_all[protein]),
        )
    return (durations_all, intensities_all, duty_cycles_all, off_times_all, blinking_rates_all,
            n_blinks_all, mean_on_all, mean_off_all, cv_on_all, cv_off_all)


def _draw_violin_panel(
    ax: plt.Axes,
    data_per_protein: dict,
    proteins: list,
    ylabel: str,
    title: str,
) -> None:
    """Draw a single violin panel. NaN values are silently dropped."""
    for i, prot in enumerate(proteins):
        values = [v for v in data_per_protein[prot] if v == v]
        if len(values) < 2:
            logging.warning("Too few data points for %s, skipping violin", prot)
            continue
        parts = ax.violinplot([values], positions=[i], showmedians=True, widths=0.65)
        color = COLORS[i % len(COLORS)]
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('none')
        for key in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
            if key in parts:
                parts[key].set_color('#333333')
                parts[key].set_linewidth(0.9)

    ax.set_xticks(range(len(proteins)))
    ax.set_xticklabels(proteins, fontsize=FONTSIZE_TICK, rotation=30, ha='right')
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.tick_params(labelsize=FONTSIZE_TICK)


def _save_csv(data_per_protein: dict, output_dir: Path, filename: str) -> None:
    """Save NaN-padded CSV with one column per protein."""
    max_len = max((len(v) for v in data_per_protein.values()), default=0)
    padded = {
        prot: list(vals) + [float('nan')] * (max_len - len(vals))
        for prot, vals in data_per_protein.items()
    }
    pd.DataFrame(padded).to_csv(output_dir / filename, index=False)


PROTEIN_COLORS = {
    'HTHTL': '#0072B2',
    'HTIA':  '#D55E00',
}


def _plot_example_traces(
    traces_by_protein_arrays: dict,
    proteins: list,
    gmm_proba: float,
    min_peak_width: int,
    output_dir: Path,
    rng: np.random.Generator,
    n_traces: int = 4,
    n_frames: int = 1000,
) -> None:
    """
    Plot the first n_frames of n_traces randomly selected traces per protein,
    with GMM-detected ON regions shaded.  Saved as plot_traces.pdf.
    """
    ncols = n_traces
    nrows = len(proteins)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 2.0),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.15)

    records = []

    for row, protein in enumerate(proteins):
        arr   = traces_by_protein_arrays[protein]          # (N, 2, T)
        color = PROTEIN_COLORS.get(protein, COLORS[row % len(COLORS)])
        idx   = rng.choice(len(arr), size=min(n_traces, len(arr)), replace=False)

        for col, i in enumerate(idx):
            ax          = axes[row][col]
            minmax_vals = arr[i, 0, :n_frames]
            frames      = np.arange(len(minmax_vals))

            _, _, signal_mask = gmm_classify_frames(minmax_vals, gmm_proba, 0.0)
            runs    = _extract_runs(signal_mask, min_peak_width)
            n_peaks = len(runs)

            ax.plot(frames, minmax_vals, '-', color=color, linewidth=0.6, rasterized=True)
            for r_s, r_e in runs:
                ax.axvspan(r_s, r_e + 1, color=color, alpha=0.25, linewidth=0)

            ax.set_title(f'{n_peaks} peaks', fontsize=FONTSIZE_TICK)
            ax.set_yticks([])
            ax.set_xticks([0, n_frames // 2, n_frames])
            ax.tick_params(labelsize=FONTSIZE_TICK - 1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            if col == 0:
                ax.set_ylabel(protein, fontsize=FONTSIZE_LABEL, fontweight='bold', color=color)
            if row == nrows - 1:
                ax.set_xlabel('Frame', fontsize=FONTSIZE_TICK)

            records.append({
                'protein':        protein,
                'trace_idx':      int(i),
                'n_peaks_shown':  n_peaks,
                'n_frames_shown': n_frames,
                'minmax_values':  arr[i, 0, :n_frames].tolist(),
                'zscored_values': arr[i, 1, :n_frames].tolist(),
            })

        # hide unused axes if fewer traces than n_traces
        for col in range(len(idx), ncols):
            axes[row][col].set_visible(False)

    fig.suptitle(
        f'Example traces — first {n_frames} frames  |  gmm_proba={gmm_proba}  min_width={min_peak_width}',
        fontsize=8, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_dir / 'plot_traces.pdf', dpi=450, bbox_inches='tight')
    plt.close(fig)
    logging.info("Saved plot_traces.pdf")

    pd.DataFrame(records).to_csv(output_dir / 'data_traces.csv', index=False)
    logging.info("Saved data_traces.csv")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Extract blink feature distributions from MCD-filtered protein traces."
    )
    parser.add_argument("-c", "--config", required=True,
                        help="Path to blink_features config yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_root       = Path(cfg["output_path"])
    frame_interval_ms = float(cfg.get("frame_interval_ms", 30.0))
    gmm_proba         = float(cfg.get("gmm_proba_threshold", 0.8))
    min_peak_width    = int(cfg.get("min_peak_width", 1))
    seed              = int(cfg.get("seed", 42))

    rng      = np.random.default_rng(seed)

    mcd_cfg  = cfg["mcd_filter"]
    npz_path = Path(mcd_cfg["npz_path"])
    wd_min   = float(mcd_cfg.get("wasserstein_min", 0.7))

    logging.info("NPZ            : %s", npz_path)
    logging.info("WD threshold   : %.2f", wd_min)
    logging.info("GMM proba      : %.2f", gmm_proba)
    logging.info("Min peak width : %d frames", min_peak_width)
    logging.info("Frame interval : %.1f ms", frame_interval_ms)

    # ── Load NPZ ────────────────────────────────────────────────────────────────
    data        = np.load(npz_path, allow_pickle=True)
    all_traces  = data['traces']                            # (N, 2, T)
    labels      = data['labels']                            # (N,)
    wd          = data['wasserstein_distances']             # (N,)
    class_names = [str(c) for c in data['class_names']]    # list of protein names

    keep_mask = wd >= wd_min
    logging.info("WD filter: %d / %d traces pass WD≥%.2f",
                 keep_mask.sum(), len(wd), wd_min)

    # ── Group by protein ────────────────────────────────────────────────────────
    proteins = class_names
    traces_by_protein_arrays = {}
    for cls_idx, name in enumerate(proteins):
        mask = (labels == cls_idx) & keep_mask
        traces_by_protein_arrays[name] = all_traces[mask]  # (N_cls, 2, T)
        logging.info("  %-12s  %d traces selected", name, mask.sum())

    # ── Output folders ──────────────────────────────────────────────────────────
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name    = "_".join(proteins) + "_mcd_" + timestamp
    output_dir     = output_root / folder_name
    features_dir   = output_dir / "features"
    traces_dir     = output_dir / "sample_traces"
    features_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Output         : %s", output_dir)

    # ── Extract features ────────────────────────────────────────────────────────
    traces_by_protein = {}
    for name, arr in traces_by_protein_arrays.items():
        traces_by_protein[name] = process_protein(
            name, arr, gmm_proba, min_peak_width, frame_interval_ms,
        )

    # ── Flatten ─────────────────────────────────────────────────────────────────
    logging.info("Flattening features …")
    (durations_all, intensities_all, duty_cycles_all, off_times_all, blinking_rates_all,
     n_blinks_all, mean_on_all, mean_off_all, cv_on_all, cv_off_all) = \
        _flatten(traces_by_protein)

    # Convert frame-based features to ms
    def _to_ms(d):
        return {p: [v * frame_interval_ms for v in vals] for p, vals in d.items()}

    def _to_ms_nan(d):
        return {p: [v * frame_interval_ms if v == v else v for v in vals]
                for p, vals in d.items()}

    durations_all = _to_ms(durations_all)
    off_times_all = _to_ms(off_times_all)
    mean_on_all   = _to_ms_nan(mean_on_all)
    mean_off_all  = _to_ms_nan(mean_off_all)

    # ── Figure: 2 rows × 5 columns ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    _draw_violin_panel(axes[0, 0], durations_all,      proteins, "Peak duration (ms)",          "Peak duration")
    _draw_violin_panel(axes[0, 1], off_times_all,      proteins, "Off-time (ms)",               "Off-time")
    _draw_violin_panel(axes[0, 2], intensities_all,    proteins, "Mean peak intensity (z-score)","Peak intensity")
    _draw_violin_panel(axes[0, 3], duty_cycles_all,    proteins, "Duty cycle",                  "Duty cycle")
    _draw_violin_panel(axes[0, 4], blinking_rates_all, proteins, "Blinking rate (peaks s\u207b\u00b9)", "Blinking rate")

    _draw_violin_panel(axes[1, 0], n_blinks_all,  proteins, "Blinks per trace",    "Blinks per trace")
    _draw_violin_panel(axes[1, 1], mean_on_all,   proteins, "Mean on-time (ms)",   "Mean on-time")
    _draw_violin_panel(axes[1, 2], mean_off_all,  proteins, "Mean off-time (ms)",  "Mean off-time")
    _draw_violin_panel(axes[1, 3], cv_on_all,     proteins, "CV on-times",         "CV on-times")
    _draw_violin_panel(axes[1, 4], cv_off_all,    proteins, "CV off-times",        "CV off-times")

    plt.tight_layout(h_pad=3.0, w_pad=3.0)
    fig.savefig(features_dir / "plot.pdf", bbox_inches='tight')
    plt.close(fig)
    logging.info("Saved features/plot.pdf")

    # ── CSVs → features/ ────────────────────────────────────────────────────────
    _save_csv(durations_all,      features_dir, "data_panel_A.csv")
    _save_csv(off_times_all,      features_dir, "data_panel_B.csv")
    _save_csv(intensities_all,    features_dir, "data_panel_C.csv")
    _save_csv(duty_cycles_all,    features_dir, "data_panel_D.csv")
    _save_csv(blinking_rates_all, features_dir, "data_panel_E.csv")
    _save_csv(n_blinks_all,       features_dir, "data_panel_F.csv")
    _save_csv(mean_on_all,        features_dir, "data_panel_G.csv")
    _save_csv(mean_off_all,       features_dir, "data_panel_H.csv")
    _save_csv(cv_on_all,          features_dir, "data_panel_I.csv")
    _save_csv(cv_off_all,         features_dir, "data_panel_J.csv")
    logging.info("Saved features/data_panel_A–J.csv")

    # ── Example traces → sample_traces/ ─────────────────────────────────────────
    _plot_example_traces(
        traces_by_protein_arrays, proteins,
        gmm_proba, min_peak_width, traces_dir, rng,
    )


if __name__ == "__main__":
    main()
