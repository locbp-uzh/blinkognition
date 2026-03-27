#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Sonnet 4.6)
# __copyright__ = "Copyright 2026, UZH, Switzerland"

"""
Extract and plot blink feature distributions from filtered protein traces.

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

Peak detection reuses the same GMM classification as the extraction pipeline
(gmm_classify_frames from Extraction/utils.py); parameters should match the
values used during filtering.

Optionally, an MCD (Monte Carlo Dropout) filter can be applied using a
pre-computed traces_with_wasserstein.npz from a training run.  Traces that
were classified with low Wasserstein certainty are excluded.  Only proteins
whose names match the NPZ class_names are filtered; others are sampled
normally.

Usage:
    python assets/blink_features.py -c assets/blink_features_config.yaml

Output (in output_path/{proteins}_{timestamp}/):
    plot.pdf          — 5-panel violin figure
    data_panel_A.csv  — peak durations (ms), one column per protein
    data_panel_B.csv  — off-times (ms), one column per protein
    data_panel_C.csv  — peak mean intensities (z-score), one column per protein
    data_panel_D.csv  — per-trace duty cycles, one column per protein
    data_panel_E.csv  — per-trace blinking rates (peaks s⁻¹), one column per protein
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

# Import GMM classifier from the extraction pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Extraction"))
from utils import gmm_classify_frames  # noqa: E402

# ── Plotting standards ─────────────────────────────────────────────────────────
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
mpl.rcParams['font.size'] = 9
mpl.rcParams['pdf.fonttype'] = 42

FONTSIZE_LABEL  = 9
FONTSIZE_TICK   = 8
FONTSIZE_TITLE  = 10

# Okabe-Ito color-blind friendly palette (yellow omitted for white backgrounds)
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


def get_protein_files(traces_folder: Path) -> dict:
    """
    Discover (bg_rm, zscored) pkl pairs in *traces_folder*.

    Returns an ordered dict {protein_label: (bg_rm_path, zscored_path)}.
    The protein label is the filename stem with '_filtered_bg_rm' stripped and
    any trailing '_IN' / '_OUT' removed (folder context already encodes the class).
    """
    bg_rm_files = sorted(traces_folder.glob("*_filtered_bg_rm.pkl"))
    result = {}
    for bg_path in bg_rm_files:
        label = bg_path.name.replace("_filtered_bg_rm.pkl", "")
        label = label.removesuffix("_IN").removesuffix("_OUT")
        zs_path = traces_folder / bg_path.name.replace("_bg_rm.pkl", "_zscored.pkl")
        if not zs_path.exists():
            logging.warning("No matching zscored file for %s — skipping", bg_path.name)
            continue
        result[label] = (bg_path, zs_path)
    return dict(sorted(result.items()))


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
    bg_rm_vals: np.ndarray,
    zscored_vals: np.ndarray,
    gmm_proba: float,
    snr_min: float,
    min_peak_width: int,
    frame_interval_ms: float,
) -> tuple[list, list, float, list, float]:
    """
    Compute features for a single trace.

    Returns:
        durations        — list of per-peak durations in frames
        mean_intensities — list of per-peak mean z-scored intensities
        duty_cycle       — fraction of frames classified as "on"
        off_times        — list of dark-interval durations in frames (gaps between peaks)
        blinking_rate    — peaks per second
    """
    _, _, signal_mask, sep = gmm_classify_frames(
        bg_rm_vals, gmm_proba, snr_min, return_separation=True
    )
    if snr_min > 0 and sep < snr_min:
        return [], [], 0.0, [], 0.0

    runs = _extract_runs(signal_mask, min_peak_width)
    durations        = [r - l + 1 for l, r in runs]
    mean_intensities = [float(np.mean(zscored_vals[l : r + 1])) for l, r in runs]
    duty_cycle       = float(np.sum(signal_mask) / len(signal_mask))
    off_times        = [runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)]
    movie_length_s   = len(bg_rm_vals) * frame_interval_ms / 1000.0
    blinking_rate    = len(runs) / movie_length_s if movie_length_s > 0 else 0.0
    return durations, mean_intensities, duty_cycle, off_times, blinking_rate


def process_protein(
    label: str,
    bg_rm_path: Path,
    zscored_path: Path,
    n_traces: int,
    gmm_proba: float,
    snr_min: float,
    min_peak_width: int,
    frame_interval_ms: float,
    rng: np.random.Generator,
    force_cols: list | None = None,
) -> list[dict]:
    """
    Load traces and extract features for one protein.

    When force_cols is given (e.g. from MCD filtering), those exact columns are
    used instead of random sampling.  Otherwise up to n_traces columns are drawn
    at random.

    Returns a list of per-trace dicts with keys:
        'durations', 'intensities', 'duty_cycle', 'off_times', 'blinking_rate'
    """
    bg_rm_df   = pd.read_pickle(bg_rm_path)
    zscored_df = pd.read_pickle(zscored_path)

    if force_cols is not None:
        available = [c for c in force_cols if c in bg_rm_df.columns and c in zscored_df.columns]
        sampled   = available
    else:
        cols    = bg_rm_df.columns.intersection(zscored_df.columns)
        n       = min(n_traces, len(cols))
        sampled = rng.choice(cols, size=n, replace=False)

    nan = float('nan')
    traces = []
    for col in sampled:
        dur, inten, dc, off, rate = _trace_features(
            bg_rm_df[col].values,
            zscored_df[col].values,
            gmm_proba, snr_min, min_peak_width, frame_interval_ms,
        )
        n   = len(dur)
        traces.append({
            'durations':     dur,
            'intensities':   inten,
            'duty_cycle':    dc,
            'off_times':     off,
            'blinking_rate': rate,
            'n_blinks':      n,
            'mean_on':       float(np.mean(dur))            if n >= 1 else nan,
            'mean_off':      float(np.mean(off))            if len(off) >= 1 else nan,
            'cv_on':         float(np.std(dur) / np.mean(dur))   if n >= 2 else nan,
            'cv_off':        float(np.std(off) / np.mean(off))   if len(off) >= 2 else nan,
        })

    logging.info("  %-12s  %d traces processed", label, len(sampled))
    return traces


def _build_npz_col_lookup(
    npz_path: Path,
    traces_folder: Path,
    protein_files: dict,
    wd_min: float,
) -> dict:
    """
    Build a per-protein dict of pkl column names that pass the MCD certainty threshold.

    Matches NPZ traces to pkl columns by comparing the first 20 float32 values of
    the minmax channel (channel 0).  Returns only traces with WD >= wd_min.

    Returns:
        {protein_label: list_of_col_names}  — only for proteins present in the NPZ.
        Proteins absent from the NPZ are not included in the result.
    """
    data       = np.load(npz_path, allow_pickle=True)
    traces_np  = data['traces']          # (N, C, T), channel 0 = minmax
    wd         = data['wasserstein_distances']
    class_names = [str(c) for c in data['class_names']]

    # Keep only high-certainty traces
    keep_mask = wd >= wd_min
    logging.info(
        "MCD NPZ: %d/%d traces pass WD≥%.2f",
        keep_mask.sum(), len(wd), wd_min,
    )

    # Group NPZ trace arrays by class
    labels = data['labels']
    npz_by_class = {}
    for cls_idx, cls_name in enumerate(class_names):
        cls_mask = (labels == cls_idx) & keep_mask
        npz_by_class[cls_name] = traces_np[cls_mask, 0, :]  # (N_cls, T) minmax channel

    # Match each protein to a class name
    result = {}
    for protein, (bg_path, _) in protein_files.items():
        # protein label already has _IN/_OUT stripped; class_names in NPZ should match
        cls_name = next((c for c in class_names if c.upper() == protein.upper()), None)
        if cls_name is None:
            continue  # protein not in the NPZ

        mm_path = traces_folder / bg_path.name.replace("_bg_rm.pkl", "_minmax.pkl")
        if not mm_path.exists():
            logging.warning("  %-12s  minmax pkl not found — cannot match NPZ", protein)
            continue

        mm_df = pd.read_pickle(mm_path)

        # Build fingerprint lookup: first-20-values tuple → column name
        fp_to_col = {}
        for col in mm_df.columns:
            key = tuple(mm_df[col].values[:20].astype(np.float32))
            fp_to_col[key] = col

        # Match NPZ traces
        matched_cols = []
        n_no_match = 0
        for npz_trace in npz_by_class[cls_name]:
            key = tuple(npz_trace[:20].astype(np.float32))
            col = fp_to_col.get(key)
            if col is not None:
                matched_cols.append(col)
            else:
                n_no_match += 1

        if n_no_match:
            logging.warning(
                "  %-12s  %d/%d NPZ traces had no match in pkl (different dataset?)",
                protein, n_no_match, len(npz_by_class[cls_name]),
            )
        logging.info(
            "  %-12s  %d high-certainty traces matched from NPZ",
            protein, len(matched_cols),
        )
        result[protein] = matched_cols

    return result


def _filter_traces(
    traces_by_protein: dict,
    max_duration_ms: float,
    frame_interval_ms: float,
    intensity_percentile: float,
) -> dict:
    """
    Remove entire traces that contain any outlier peak.

    A trace is removed if any of its peaks has:
      - duration  > max_duration_ms (durations stored in frames, threshold converted here), OR
      - mean intensity > global *intensity_percentile*-th percentile

    The intensity threshold is computed from all peak intensities across all
    proteins before any filtering, so it is not affected by the duration cut.
    """
    max_duration_frames = max_duration_ms / frame_interval_ms

    # Compute global intensity threshold across all proteins
    all_intensities = [
        v
        for traces in traces_by_protein.values()
        for t in traces
        for v in t['intensities']
        if np.isfinite(v)
    ]
    intensity_threshold = (
        np.percentile(all_intensities, intensity_percentile)
        if all_intensities else np.inf
    )
    logging.info(
        "Intensity threshold (%.0fth pct): %.3f z-score",
        intensity_percentile, intensity_threshold,
    )
    logging.info(
        "Max peak duration threshold: %.0f ms (%.1f frames)",
        max_duration_ms, max_duration_frames,
    )

    filtered = {}
    for protein, traces in traces_by_protein.items():
        kept = []
        for t in traces:
            if t['durations'] and max(t['durations']) > max_duration_frames:
                continue
            if t['intensities'] and max(t['intensities']) > intensity_threshold:
                continue
            kept.append(t)
        n_removed = len(traces) - len(kept)
        if n_removed:
            logging.info("  %-12s  removed %d / %d traces", protein, n_removed, len(traces))
        filtered[protein] = kept
    return filtered


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
            "  %-12s  %d traces kept  %d peaks",
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
        values = [v for v in data_per_protein[prot] if v == v]  # drop NaN
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


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Extract blink feature distributions from filtered protein traces."
    )
    parser.add_argument("-c", "--config", required=True,
                        help="Path to blink_features_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    traces_folder     = Path(cfg["traces_folder"])
    n_traces          = int(cfg.get("n_traces", 500))
    output_dir        = Path(cfg["output_path"])
    frame_interval_ms = float(cfg.get("frame_interval_ms", 30.0))
    gmm_proba              = float(cfg.get("gmm_proba_threshold", 0.6))
    snr_min                = float(cfg.get("snr_min_separation", 2.0))
    min_peak_width         = int(cfg.get("min_peak_width", 1))
    seed                   = int(cfg.get("seed", 42))
    max_peak_duration_ms   = float(cfg.get("max_peak_duration_s", 10.0)) * 1000
    intensity_percentile   = float(cfg.get("intensity_clip_percentile", 99))
    mcd_cfg                = cfg.get("mcd_filter")

    rng = np.random.default_rng(seed)

    logging.info("Traces folder  : %s", traces_folder)
    logging.info("N traces       : %d", n_traces)
    logging.info("Frame interval : %.1f ms", frame_interval_ms)

    protein_files = get_protein_files(traces_folder)
    if not protein_files:
        logging.error("No trace files found in %s", traces_folder)
        sys.exit(1)

    proteins = list(protein_files.keys())
    logging.info("Proteins       : %s", proteins)

    # Build output subfolder: {output_path}/{protein1_protein2_...}_{YYYYMMDD_HHMMSS}
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = "_".join(proteins) + "_" + timestamp
    output_dir  = Path(cfg["output_path"]) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Output         : %s", output_dir)

    # ── Optional MCD lookup (must happen before pass 1) ─────────────────────
    mcd_cols_by_protein = {}
    if mcd_cfg:
        npz_path = Path(mcd_cfg['npz_path'])
        wd_min   = float(mcd_cfg.get('wasserstein_min', 0.7))
        logging.info("Loading MCD NPZ: %s  (WD_min=%.2f)", npz_path, wd_min)
        mcd_cols_by_protein = _build_npz_col_lookup(
            npz_path, traces_folder, protein_files, wd_min
        )
        # Restrict to only proteins present in the NPZ
        protein_files = {k: v for k, v in protein_files.items() if k in mcd_cols_by_protein}
        proteins = list(protein_files.keys())
        logging.info("MCD filter active — proteins restricted to: %s", proteins)
        folder_name = "_".join(proteins) + "_mcd_" + timestamp
        output_dir  = Path(cfg["output_path"]) / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Output         : %s", output_dir)

    # ── Pass 1: extract raw per-trace features ───────────────────────────────
    traces_by_protein = {}
    for label, (bg_path, zs_path) in protein_files.items():
        force_cols = mcd_cols_by_protein.get(label)  # None → random sampling
        traces_by_protein[label] = process_protein(
            label, bg_path, zs_path,
            n_traces, gmm_proba, snr_min, min_peak_width, frame_interval_ms, rng,
            force_cols=force_cols,
        )

    # ── Pass 2: filter outlier traces, then flatten ──────────────────────────
    logging.info("Filtering outlier traces …")
    traces_by_protein = _filter_traces(
        traces_by_protein, max_peak_duration_ms, frame_interval_ms, intensity_percentile
    )
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

    # ── Figure: 2 rows × 5 columns ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    # Row 1 — per-peak distributions
    _draw_violin_panel(
        axes[0, 0], durations_all, proteins,
        ylabel="Peak duration (ms)",
        title="Peak duration",
    )
    _draw_violin_panel(
        axes[0, 1], off_times_all, proteins,
        ylabel="Off-time (ms)",
        title="Off-time",
    )
    _draw_violin_panel(
        axes[0, 2], intensities_all, proteins,
        ylabel="Mean peak intensity (z-score)",
        title="Peak intensity",
    )
    _draw_violin_panel(
        axes[0, 3], duty_cycles_all, proteins,
        ylabel="Duty cycle",
        title="Duty cycle",
    )
    _draw_violin_panel(
        axes[0, 4], blinking_rates_all, proteins,
        ylabel="Blinking rate (peaks s\u207b\u00b9)",
        title="Blinking rate",
    )

    # Row 2 — per-trace kinetic summaries
    _draw_violin_panel(
        axes[1, 0], n_blinks_all, proteins,
        ylabel="Blinks per trace",
        title="Blinks per trace",
    )
    _draw_violin_panel(
        axes[1, 1], mean_on_all, proteins,
        ylabel="Mean on-time (ms)",
        title="Mean on-time",
    )
    _draw_violin_panel(
        axes[1, 2], mean_off_all, proteins,
        ylabel="Mean off-time (ms)",
        title="Mean off-time",
    )
    _draw_violin_panel(
        axes[1, 3], cv_on_all, proteins,
        ylabel="CV on-times",
        title="CV on-times",
    )
    _draw_violin_panel(
        axes[1, 4], cv_off_all, proteins,
        ylabel="CV off-times",
        title="CV off-times",
    )

    plt.tight_layout(h_pad=3.0, w_pad=3.0)
    fig.savefig(output_dir / "plot.pdf", bbox_inches='tight')
    plt.close(fig)
    logging.info("Saved plot.pdf")

    # ── CSVs ────────────────────────────────────────────────────────────────
    _save_csv(durations_all,      output_dir, "data_panel_A.csv")
    _save_csv(off_times_all,      output_dir, "data_panel_B.csv")
    _save_csv(intensities_all,    output_dir, "data_panel_C.csv")
    _save_csv(duty_cycles_all,    output_dir, "data_panel_D.csv")
    _save_csv(blinking_rates_all, output_dir, "data_panel_E.csv")
    _save_csv(n_blinks_all,       output_dir, "data_panel_F.csv")
    _save_csv(mean_on_all,        output_dir, "data_panel_G.csv")
    _save_csv(mean_off_all,       output_dir, "data_panel_H.csv")
    _save_csv(cv_on_all,          output_dir, "data_panel_I.csv")
    _save_csv(cv_off_all,         output_dir, "data_panel_J.csv")
    logging.info("Saved data_panel_A–J.csv")


if __name__ == "__main__":
    main()
