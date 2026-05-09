#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from __future__ import annotations
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Sonnet 4.6)
# __copyright__ = "Copyright 2026, UZH, Switzerland"

"""
Extract and plot blink feature distributions from protein traces.

Two analysis modes are supported (set via `analysis_mode` in the YAML config):

  per_experiment (default)
    All traces from the NPZ are used without any Wasserstein filter.
    Each experiment (date-prefixed acquisition folder) gets its own output
    folder with a separate 2×3 violin figure and CSVs.  Requires `uid_dir`
    pointing to the UniqueIDs PKL folder so traces can be mapped to their
    source experiment.  Optionally restrict to a subset via `experiments:`.

  pooled
    Traces are filtered by Wasserstein distance (mcd_filter.wasserstein_min)
    and all passing traces are analysed together in one figure.

NPZ layout expected:
    traces               (N, 2, T)  channel 0 = minmax, channel 1 = zscored
    labels               (N,)       integer class indices
    wasserstein_distances (N,)      MCD certainty score per trace (0–1)
    class_names          (C,)       protein name strings
    unique_ids           (N,)       integer per-protein trace indices

GMM peak detection uses:
    channel 0 (minmax) as input to gmm_classify_frames
    channel 1 (zscored) for per-peak mean intensity reporting

Three per-trace features are computed in a 1×3 panel:

  A) Mean off-time  — mean dark-interval per trace, converted to ms
  B) Mean on-time   — mean peak duration per trace, converted to ms
  C) Duty cycle     — fraction of active window in the "on" state

Usage:
    # pooled (WD-filtered, used in the paper):
    python features.py -c config.yaml

    # per-experiment (one output per acquisition date):
    python features.py -c config.yaml   # with analysis_mode: per_experiment

Output (in output_path/):
  per_experiment mode → {proteins}_exp_{timestamp}/{exp_name}/features/
  pooled mode        → {proteins}_mcd_{timestamp}/features/  +  sample_traces/
"""

import sys
import argparse
import logging
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Extraction"))
from utils import gmm_classify_frames  # noqa: E402

# ── Plotting standards ──────────────────────────────────────────────────────────
FONTSIZE_LABEL  = 18
FONTSIZE_TICK   = 16
FONTSIZE_TITLE  = 18
FONTSIZE_LEGEND = 16

mpl.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':        FONTSIZE_LABEL,
    'axes.titlesize':   FONTSIZE_TITLE,
    'axes.labelsize':   FONTSIZE_LABEL,
    'xtick.labelsize':  FONTSIZE_TICK,
    'ytick.labelsize':  FONTSIZE_TICK,
    'legend.fontsize':  FONTSIZE_LEGEND,
    'figure.dpi':       150,
    'savefig.dpi':      450,
    'savefig.bbox':     'tight',
    'lines.linewidth':  1.2,
    'patch.linewidth':  0.8,
    'axes.grid':        False,
    'pdf.fonttype':     42,
    'svg.fonttype':     'none',
})

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
) -> tuple[list, list, float, list]:
    """
    Compute blink features for a single trace.

    GMM is run on the minmax channel (min_separation=0 — no trace-level gating).
    Peak intensities are reported from the zscored channel.

    Returns:
        durations        — list of per-peak durations in frames
        mean_intensities — list of per-peak mean z-scored intensities
        duty_cycle       — fraction of active window in the "on" state
        off_times        — list of dark-interval durations in frames
    """
    _, _, signal_mask = gmm_classify_frames(minmax_vals, gmm_proba, 0.0)

    runs             = _extract_runs(signal_mask, min_peak_width)
    durations        = [r - l + 1 for l, r in runs]
    mean_intensities = [float(np.mean(zscored_vals[l:r + 1])) for l, r in runs]
    off_times            = [runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)]
    active_window_frames = runs[-1][1] - runs[0][0] + 1
    duty_cycle           = float(np.sum(signal_mask) / active_window_frames)
    return durations, mean_intensities, duty_cycle, off_times


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
        durations, intensities, duty_cycle, off_times, mean_on, mean_off
    """
    nan = float('nan')
    traces = []
    for i in range(len(traces_array)):
        minmax_vals  = traces_array[i, 0, :]
        zscored_vals = traces_array[i, 1, :]
        dur, inten, dc, off = _trace_features(
            minmax_vals, zscored_vals, gmm_proba, min_peak_width, frame_interval_ms,
        )
        traces.append({
            'durations':   dur,
            'intensities': inten,
            'duty_cycle':  dc,
            'off_times':   off,
            'mean_on':     float(np.mean(dur)) if len(dur) >= 1  else nan,
            'mean_off':    float(np.mean(off)) if len(off) >= 1  else nan,
        })

    logging.info("  %-12s  %d traces processed", label, len(traces_array))
    return traces


def _flatten(traces_by_protein: dict) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Flatten per-trace dicts into protein-level lists for plotting."""
    durations_all, intensities_all, duty_cycles_all = {}, {}, {}
    off_times_all, mean_on_all, mean_off_all = {}, {}, {}
    for protein, traces in traces_by_protein.items():
        durations_all[protein]   = [d for t in traces for d in t['durations']]
        intensities_all[protein] = [i for t in traces for i in t['intensities']]
        duty_cycles_all[protein] = [t['duty_cycle'] for t in traces]
        off_times_all[protein]   = [o for t in traces for o in t['off_times']]
        mean_on_all[protein]     = [t['mean_on']    for t in traces]
        mean_off_all[protein]    = [t['mean_off']   for t in traces]
        logging.info(
            "  %-12s  %d traces  %d peaks",
            protein, len(traces), len(durations_all[protein]),
        )
    return durations_all, intensities_all, duty_cycles_all, off_times_all, mean_on_all, mean_off_all


def _mannwhitney_stats(vals1: list, vals2: list) -> tuple[float, float]:
    """Two-sided Mann-Whitney U + rank-biserial correlation r."""
    v1 = np.array([v for v in vals1 if v == v])
    v2 = np.array([v for v in vals2 if v == v])
    if len(v1) < 2 or len(v2) < 2:
        return float('nan'), float('nan')
    res = stats.mannwhitneyu(v1, v2, alternative='two-sided')
    r   = 1.0 - (2.0 * res.statistic) / (len(v1) * len(v2))
    return float(res.pvalue), float(r)


def _bh_correct(pvals: list) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    m   = len(pvals)
    arr = np.array(pvals, dtype=float)
    order = np.argsort(arr)
    adj = arr[order] * m / (np.arange(m) + 1.0)
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    p_adj = np.empty(m)
    p_adj[order] = np.minimum(adj, 1.0)
    return p_adj


def _fmt_pval(p: float) -> str:
    if np.isnan(p): return ''
    return f'p = {p:.2e}'


def _draw_violin_panel(
    ax: plt.Axes,
    data_per_protein: dict,
    proteins: list,
    ylabel: str,
    title: str,
    coverage: float = 0.80,
    p_adj: float = None,
    effect_r: float = None,
) -> None:
    """Draw a single violin panel. NaN values are silently dropped.

    Y-axis is clipped to the union of per-class [q_lo, q_hi] percentile ranges
    (where q_lo = (1-coverage)/2, q_hi = 1 - q_lo), so that at least `coverage`
    fraction of each class's data is visible.  Outliers remain in the violin KDE.
    """
    q_lo = (1.0 - coverage) / 2.0 * 100   # e.g. 10.0 for coverage=0.80
    q_hi = 100.0 - q_lo                    # e.g. 90.0

    all_lo, all_hi = [], []
    valid_data = {}

    for prot in proteins:
        values = [v for v in data_per_protein[prot] if v == v]
        valid_data[prot] = values
        if len(values) >= 2:
            all_lo.append(float(np.percentile(values, q_lo)))
            all_hi.append(float(np.percentile(values, q_hi)))

    for i, prot in enumerate(proteins):
        values = valid_data[prot]
        if len(values) < 2:
            logging.warning("Too few data points for %s, skipping violin", prot)
            continue
        parts = ax.violinplot([values], positions=[i], showmeans=False,
                              showmedians=False, widths=0.65)
        color = PROTEIN_COLORS.get(prot, COLORS[i % len(COLORS)])
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('none')
        for key in ('cbars', 'cmins', 'cmaxes'):
            if key in parts:
                parts[key].set_color('#333333')
                parts[key].set_linewidth(0.9)
        # Draw mean as a horizontal line spanning the violin width
        m = float(np.mean(values))
        ax.hlines(m, i - 0.3, i + 0.3, colors='#333333', linewidths=1.2)

    # Set y-limits to the union of per-class coverage ranges + 5% padding
    if all_lo and all_hi:
        ymin, ymax = min(all_lo), max(all_hi)
        pad = (ymax - ymin) * 0.05 if ymax > ymin else abs(ymax) * 0.05 + 1e-9
        ax.set_ylim(ymin - pad, ymax + pad)


    ax.set_xticks(range(len(proteins)))
    ax.set_xticklabels(proteins, fontsize=FONTSIZE_TICK, rotation=30, ha='right')
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICK,
                   length=4, width=0.8, direction='out')
    ax.tick_params(axis='both', which='minor', length=2, width=0.6, direction='out')

    # Significance bracket (only drawn when exactly 2 proteins)
    if p_adj is not None and len(proteins) == 2:
        ylo, yhi = ax.get_ylim()
        span  = yhi - ylo
        bkt_y = yhi + span * 0.04
        tip   = span * 0.025
        ax.plot([0, 0, 1, 1], [bkt_y - tip, bkt_y, bkt_y, bkt_y - tip],
                color='#333333', linewidth=0.8, clip_on=False)
        label = _fmt_pval(p_adj)
        if effect_r is not None and not np.isnan(effect_r):
            label += f'\nr={effect_r:.2f}'
        ax.text(0.5, bkt_y + span * 0.02, label,
                ha='center', va='bottom', fontsize=FONTSIZE_TICK - 1,
                color='#333333', clip_on=False)
        ax.set_ylim(ylo, bkt_y + span * 0.18)


def _save_csv(data_per_protein: dict, output_dir: Path, filename: str) -> None:
    """Save NaN-padded CSV with one column per protein."""
    max_len = max((len(v) for v in data_per_protein.values()), default=0)
    padded = {
        prot: list(vals) + [float('nan')] * (max_len - len(vals))
        for prot, vals in data_per_protein.items()
    }
    pd.DataFrame(padded).to_csv(output_dir / filename, index=False)


PROTEIN_COLORS = {
    'HTHTL': '#0072B2',  # blue
    'HTIA':  '#E69F00',  # orange
    'Grx1':  '#009E73',  # green
    'snap':  '#CC79A7',  # pink/purple
}


def _plot_example_traces(
    traces_by_protein_arrays: dict,
    proteins: list,
    gmm_proba: float,
    min_peak_width: int,
    output_dir: Path,
    rng: np.random.Generator,
    n_traces: int = 4,
    half_window: int = 500,
) -> None:
    """
    Plot n_traces randomly selected traces per protein, windowed around the highest peak.

    For each trace:
      1. GMM is run on the full minmax trace.
      2. All peaks are detected; the highest (by mean minmax value) is selected.
      3. A window of 2*half_window frames is placed so that the highest peak falls
         inside it while the window stays within trace boundaries — no NaN padding.
         If the peak is closer than half_window to one edge, the window is shifted
         so it starts/ends at that edge and the slack is taken up on the other side.
      4. GMM peak shading is recomputed on the extracted window for display.

    Saved as plot_traces.pdf + data_traces.csv.
    """
    n_frames = 2 * half_window
    ncols = n_traces
    nrows = len(proteins)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 2.0),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.15)

    records = []

    for row, protein in enumerate(proteins):
        arr   = traces_by_protein_arrays[protein]          # (N, 2, T)
        T     = arr.shape[2]
        color = PROTEIN_COLORS.get(protein, COLORS[row % len(COLORS)])
        idx   = rng.choice(len(arr), size=min(n_traces, len(arr)), replace=False)

        for col, i in enumerate(idx):
            ax = axes[row][col]

            # 1. GMM on full trace
            full_minmax = arr[i, 0, :]
            _, _, signal_mask_full = gmm_classify_frames(full_minmax, gmm_proba, 0.0)
            runs_full = _extract_runs(signal_mask_full, min_peak_width)
            n_peaks_total = len(runs_full)

            # 2. Find highest peak by mean minmax value; fall back to trace midpoint
            if runs_full:
                peak_means = [float(np.mean(full_minmax[l:r + 1])) for l, r in runs_full]
                best = runs_full[int(np.argmax(peak_means))]
                peak_centre = (best[0] + best[1]) // 2
            else:
                peak_centre = T // 2

            # 3. Place a window of n_frames that contains the peak and stays in bounds.
            #    Start from an ideal centred position then clamp to [0, T).
            win_len = min(n_frames, T)
            t_start = peak_centre - half_window
            t_start = max(t_start, 0)
            t_start = min(t_start, T - win_len)
            t_end   = t_start + win_len

            frames      = np.arange(t_start, t_end)
            minmax_win  = arr[i, 0, t_start:t_end]
            zscored_win = arr[i, 1, t_start:t_end]

            # 4. Recompute GMM on the window for shading
            _, _, sm_win = gmm_classify_frames(minmax_win, gmm_proba, 0.0)
            runs_win = _extract_runs(sm_win, min_peak_width)

            ax.plot(frames, minmax_win, '-', color=color, linewidth=0.6, rasterized=True)
            for r_s, r_e in runs_win:
                ax.axvspan(frames[r_s], frames[r_e] + 1, color=color, alpha=0.25, linewidth=0)

            ax.set_title(f'{n_peaks_total} peaks total', fontsize=FONTSIZE_TICK)
            ax.set_yticks([])
            ax.set_xticks([frames[0], frames[len(frames) // 2], frames[-1]])
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
                'n_peaks_total':  n_peaks_total,
                'window_start':   int(t_start),
                'window_end':     int(t_end),
                'n_frames_shown': int(win_len),
                'minmax_values':  minmax_win.tolist(),
                'zscored_values': zscored_win.tolist(),
            })

        # hide unused axes if fewer traces than n_traces
        for col in range(len(idx), ncols):
            axes[row][col].set_visible(False)

    fig.suptitle(
        f'Example traces — {n_frames} frames around highest peak  |  gmm_proba={gmm_proba}  min_width={min_peak_width}',
        fontsize=8, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_dir / 'plot_traces.pdf', dpi=450, bbox_inches='tight')
    plt.close(fig)
    logging.info("Saved plot_traces.pdf")

    pd.DataFrame(records).to_csv(output_dir / 'data_traces.csv', index=False)
    logging.info("Saved data_traces.csv")


def _map_traces_to_experiments(
    ids: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    uid_dir: Path,
) -> list[str]:
    """
    Map each NPZ trace to its source experiment folder name.

    Loads {protein}_uniqueID_all.pkl for each protein class, reads the
    'origin' column (file path), and extracts the date-prefixed experiment
    folder (e.g. '20231130_SP_Exp1').  The lookup key is (protein, uniqueID)
    because IDs are assigned independently per protein class.

    Returns a list of experiment names, one per trace, in NPZ order.
    Traces whose (protein, uniqueID) pair is not found receive 'UNKNOWN'.
    """
    def _exp_from_origin(origin: str) -> str:
        for part in str(origin).replace('\\', '/').split('/'):
            if len(part) >= 8 and part[:8].isdigit():
                return part
        return 'UNKNOWN'

    uid_to_exp: dict[tuple, str] = {}
    for cls_idx, prot in enumerate(class_names):
        pkl_path = uid_dir / f"{prot}_uniqueID_all.pkl"
        if not pkl_path.exists():
            logging.warning("UniqueID PKL not found: %s", pkl_path)
            continue
        with open(pkl_path, 'rb') as fh:
            df = pickle.load(fh)
        for _, row in df.iterrows():
            uid_to_exp[(prot, int(row['uniqueID']))] = _exp_from_origin(row['origin'])
        logging.info("Loaded %s (%d entries)", pkl_path.name, len(df))

    return [
        uid_to_exp.get((class_names[labels[i]], int(ids[i])), 'UNKNOWN')
        for i in range(len(ids))
    ]


def _run_feature_analysis(
    traces_by_protein_arrays: dict,
    proteins: list,
    frame_interval_ms: float,
    gmm_proba: float,
    min_peak_width: int,
    output_dir: Path,
    title: str = '',
) -> None:
    """
    Extract features, run statistics, save violin plot and CSVs.

    output_dir is expected to already contain a 'features/' subdirectory
    (the caller is responsible for creating it).
    """
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Extract features
    traces_by_protein = {
        name: process_protein(name, arr, gmm_proba, min_peak_width, frame_interval_ms)
        for name, arr in traces_by_protein_arrays.items()
    }

    # Flatten
    logging.info("Flattening features …")
    (_, _, duty_cycles_all, _, mean_on_all, mean_off_all) = _flatten(traces_by_protein)

    def _to_ms_nan(d):
        return {p: [v * frame_interval_ms if v == v else v for v in vals]
                for p, vals in d.items()}

    mean_on_all  = _to_ms_nan(mean_on_all)
    mean_off_all = _to_ms_nan(mean_off_all)

    # Statistics: Mann-Whitney U + BH-FDR (only for exactly 2 proteins)
    panel_data = [mean_off_all, mean_on_all, duty_cycles_all]
    if len(proteins) == 2:
        raw_stats = [_mannwhitney_stats(d[proteins[0]], d[proteins[1]]) for d in panel_data]
        p_adj_all = _bh_correct([s[0] for s in raw_stats])
        logging.info("Mann-Whitney U (BH-corrected p-values):")
        for lbl, (p_raw, r), p_adj in zip(
            ["Mean off-time", "Mean on-time", "Duty cycle"],
            raw_stats, p_adj_all,
        ):
            logging.info("  %-16s  p_raw=%.2e  p_adj=%.2e  r=%.3f",
                         lbl, p_raw, p_adj, r)
    else:
        raw_stats = [(None, None)] * 3
        p_adj_all = [None] * 3

    # Figure: 1 row x 3 columns — scale width with number of proteins
    fig_w = max(11, 5.5 * len(proteins))
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 4))
    if title:
        fig.suptitle(title, fontsize=FONTSIZE_TITLE, y=1.01)

    _draw_violin_panel(axes[0], mean_off_all,    proteins, "Mean off-time (ms)", "Mean off-time", p_adj=p_adj_all[0], effect_r=raw_stats[0][1])
    _draw_violin_panel(axes[1], mean_on_all,     proteins, "Mean on-time (ms)",  "Mean on-time",  p_adj=p_adj_all[1], effect_r=raw_stats[1][1])
    _draw_violin_panel(axes[2], duty_cycles_all, proteins, "Duty cycle",         "Duty cycle",    p_adj=p_adj_all[2], effect_r=raw_stats[2][1])

    plt.tight_layout(h_pad=3.0, w_pad=3.0)
    fig.savefig(features_dir / "plot.pdf", bbox_inches='tight')
    plt.close(fig)
    logging.info("Saved %s/plot.pdf", features_dir.name)

    _save_csv(mean_off_all,       features_dir, "data_panel_A.csv")
    _save_csv(mean_on_all,        features_dir, "data_panel_B.csv")
    _save_csv(duty_cycles_all, features_dir, "data_panel_C.csv")
    logging.info("Saved features/data_panel_A–C.csv")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Extract blink feature distributions from MCD-filtered protein traces."
    )
    parser.add_argument("-c", "--config", required=True,
                        help="Path to blink_features config yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    _default_output = Path(__file__).parent.parent / "Results" / "Features"
    output_root       = Path(cfg.get("output_path", str(_default_output)))
    frame_interval_ms = float(cfg.get("frame_interval_ms", 30.0))
    gmm_proba         = float(cfg.get("gmm_proba_threshold", 0.8))
    min_peak_width    = int(cfg.get("min_peak_width", 1))
    seed              = int(cfg.get("seed", 42))
    analysis_mode     = cfg.get("analysis_mode", "per_experiment")

    rng = np.random.default_rng(seed)

    mcd_cfg  = cfg.get("mcd_filter", {})
    npz_path = Path(mcd_cfg["npz_path"])
    wd_min   = float(mcd_cfg.get("wasserstein_min", 0.7))

    logging.info("NPZ            : %s", npz_path)
    logging.info("Analysis mode  : %s", analysis_mode)
    logging.info("GMM proba      : %.2f", gmm_proba)
    logging.info("Min peak width : %d frames", min_peak_width)
    logging.info("Frame interval : %.1f ms", frame_interval_ms)

    # ── Load NPZ ────────────────────────────────────────────────────────────────
    data        = np.load(npz_path, allow_pickle=True)
    all_traces  = data['traces']
    labels      = data['labels']
    wd          = data['wasserstein_distances']
    ids         = data['unique_ids'] if 'unique_ids' in data else None
    class_names = [str(c) for c in data['class_names']]
    proteins    = class_names
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Per-experiment mode ─────────────────────────────────────────────────────
    if analysis_mode == "per_experiment":
        if ids is None:
            raise ValueError(
                "per_experiment mode requires 'unique_ids' in the NPZ file. "
                "This NPZ was generated without unique_ids — use analysis_mode: pooled instead."
            )
        uid_dir     = Path(cfg["uid_dir"])
        exp_filter  = cfg.get("experiments", None)  # None = run all found

        logging.info("UID dir        : %s", uid_dir)
        trace_exps = _map_traces_to_experiments(ids, labels, class_names, uid_dir)

        # Determine which experiments to process
        all_exps = sorted(set(e for e in trace_exps if e != 'UNKNOWN'))
        if exp_filter:
            run_exps = [e for e in exp_filter if e in all_exps]
            missing  = [e for e in exp_filter if e not in all_exps]
            if missing:
                logging.warning("Experiments not found in data: %s", missing)
        else:
            run_exps = all_exps
        logging.info("Experiments    : %s", run_exps)

        run_dir = output_root / ("_".join(proteins) + "_exp_" + timestamp)
        logging.info("Output root    : %s", run_dir)

        for exp in run_exps:
            logging.info("\n── %s ──────────────────────────────────────", exp)
            exp_arrays: dict[str, list] = {p: [] for p in proteins}
            for i in range(len(all_traces)):
                if trace_exps[i] == exp:
                    exp_arrays[class_names[labels[i]]].append(all_traces[i])
            traces_by_protein_arrays = {
                p: np.stack(arr) if arr else np.empty((0, 2, all_traces.shape[2]))
                for p, arr in exp_arrays.items()
            }
            for p, arr in traces_by_protein_arrays.items():
                logging.info("  %-12s  %d traces", p, len(arr))
            _run_feature_analysis(
                traces_by_protein_arrays, proteins,
                frame_interval_ms, gmm_proba, min_peak_width,
                output_dir=run_dir / exp,
                title=exp,
            )

    # ── Pooled mode ─────────────────────────────────────────────────────────────
    else:
        logging.info("WD threshold   : %.2f", wd_min)
        keep_mask = wd >= wd_min
        logging.info("WD filter: %d / %d traces pass WD≥%.2f",
                     keep_mask.sum(), len(wd), wd_min)

        traces_by_protein_arrays = {}
        for cls_idx, name in enumerate(proteins):
            mask = (labels == cls_idx) & keep_mask
            traces_by_protein_arrays[name] = all_traces[mask]
            logging.info("  %-12s  %d traces selected", name, mask.sum())

        run_dir = output_root / ("_".join(proteins) + "_mcd_" + timestamp)
        logging.info("Output         : %s", run_dir)

        _run_feature_analysis(
            traces_by_protein_arrays, proteins,
            frame_interval_ms, gmm_proba, min_peak_width,
            output_dir=run_dir,
        )

        # Sample traces (pooled mode only)
        traces_dir = run_dir / "sample_traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        _plot_example_traces(
            traces_by_protein_arrays, proteins,
            gmm_proba, min_peak_width, traces_dir, rng,
        )


if __name__ == "__main__":
    main()
