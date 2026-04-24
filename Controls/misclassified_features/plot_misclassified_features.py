#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Sonnet 4.6)
# __copyright__ = "Copyright 2026, UZH, Switzerland"

"""
Plot blinking features of high-confidence misclassified K20Ac traces.

Filters traces_with_wasserstein.npz for:
  - Wasserstein distance >= wasserstein_min
  - true label == K20Ac   (labels == 1)
  - predicted label == Grx1 (predictions == 0)

Compares those traces against the correctly classified high-WD Grx1 and K20Ac
populations in a 2×3 violin panel (same six features as blink_features.py pooled mode).

Usage:
    cd Controls/misclassified_features
    conda run -n blink2env python plot_misclassified_features.py -c config_misclassified.yaml
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
from scipy import stats

# --- path setup ---
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "Extraction"))

from utils import gmm_classify_frames  # noqa

# ── Plotting standards (match blink_features.py) ────────────────────────────
FONTSIZE_LABEL  = 18
FONTSIZE_TICK   = 16
FONTSIZE_TITLE  = 18
FONTSIZE_LEGEND = 16

mpl.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':        FONTSIZE_LABEL,
    'axes.titlesize':   FONTSIZE_TITLE,
    'axes.labelsize':   FONTSIZE_LABEL,
    'xtick.labelsize':  FONTSIZE_TICK,
    'ytick.labelsize':  FONTSIZE_TICK,
    'figure.dpi':       150,
    'savefig.dpi':      450,
    'savefig.bbox':     'tight',
    'pdf.fonttype':     42,
})

# Three groups: correctly classified Grx1 (WD≥thr), correctly classified K20Ac, misclassified K20Ac
GROUP_LABELS = ["Grx1\n(correct)", "K20Ac\n(correct)", "K20Ac→Grx1\n(misclassified)"]
GROUP_COLORS = ["#009E73", "#E69F00", "#D55E00"]


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _extract_runs(signal_mask, min_peak_width):
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


def compute_trace_features(minmax_vals, zscored_vals, gmm_proba, min_peak_width, frame_interval_ms):
    """Return dict of six blink features for one trace; None if fewer than 2 peaks."""
    _, _, signal_mask = gmm_classify_frames(minmax_vals, float(gmm_proba), 0.0)
    runs = _extract_runs(signal_mask, min_peak_width)
    if len(runs) < 2:
        return None
    durations = [r - l + 1 for l, r in runs]
    off_times = [runs[i+1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)]
    active_frames = runs[-1][1] - runs[0][0] + 1
    active_s      = active_frames * frame_interval_ms / 1000.0

    return {
        "mean_off_ms":    float(np.mean(off_times))               * frame_interval_ms,
        "mean_on_ms":     float(np.mean(durations))               * frame_interval_ms,
        "blinking_rate":  len(runs) / active_s,
        "duty_cycle":     float(np.sum(signal_mask) / active_frames),
        "cv_on":          float(np.std(durations) / np.mean(durations)) if len(durations) >= 2 else float('nan'),
        "cv_off":         float(np.std(off_times)  / np.mean(off_times))  if len(off_times)  >= 2 else float('nan'),
    }


def process_group(traces_2ch, gmm_proba, min_peak_width, frame_interval_ms, label):
    """Run feature extraction on an (N, 2, T) array; return list of feature dicts."""
    results = []
    for i in range(len(traces_2ch)):
        feat = compute_trace_features(
            traces_2ch[i, 0, :],   # minmax
            traces_2ch[i, 1, :],   # zscored
            gmm_proba, min_peak_width, frame_interval_ms,
        )
        if feat is not None:
            results.append(feat)
    logging.info("  %-30s  %d / %d traces had ≥2 peaks", label, len(results), len(traces_2ch))
    return results


def _mannwhitney(vals1, vals2):
    v1 = np.array([v for v in vals1 if np.isfinite(v)])
    v2 = np.array([v for v in vals2 if np.isfinite(v)])
    if len(v1) < 2 or len(v2) < 2:
        return float('nan'), float('nan')
    res = stats.mannwhitneyu(v1, v2, alternative='two-sided')
    r   = 1.0 - (2.0 * res.statistic) / (len(v1) * len(v2))
    return float(res.pvalue), float(r)


def _bh_correct(pvals):
    m   = len(pvals)
    arr = np.array(pvals, dtype=float)
    order = np.argsort(arr)
    adj   = arr[order] * m / (np.arange(m) + 1.0)
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i+1])
    p_adj = np.empty(m)
    p_adj[order] = np.minimum(adj, 1.0)
    return p_adj


def draw_violin(ax, data_per_group, group_labels, colors, ylabel, title, coverage=0.80):
    q_lo = (1 - coverage) / 2 * 100
    q_hi = 100 - q_lo
    all_lo, all_hi = [], []

    for i, (grp, color) in enumerate(zip(group_labels, colors)):
        values = [v for v in data_per_group[grp] if np.isfinite(v)]
        if len(values) < 2:
            continue
        all_lo.append(float(np.percentile(values, q_lo)))
        all_hi.append(float(np.percentile(values, q_hi)))
        parts = ax.violinplot([values], positions=[i], showmeans=False,
                              showmedians=False, widths=0.65)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('none')
        for key in ('cbars', 'cmins', 'cmaxes'):
            if key in parts:
                parts[key].set_color('#333333')
                parts[key].set_linewidth(0.9)
        m = float(np.mean(values))
        ax.hlines(m, i - 0.3, i + 0.3, colors='#333333', linewidths=1.2)

    if all_lo and all_hi:
        ymin, ymax = min(all_lo), max(all_hi)
        pad = (ymax - ymin) * 0.05 if ymax > ymin else abs(ymax) * 0.05 + 1e-9
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xticks(range(len(group_labels)))
    ax.set_xticklabels(group_labels, fontsize=FONTSIZE_TICK - 2)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICK, length=4, width=0.8)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    cfg              = load_config(args.config)
    npz_path         = cfg["npz_path"]
    wd_min           = float(cfg.get("wasserstein_min", 0.61))
    output_root      = cfg.get("output_root", "../../Results/Controls/MisclassifiedFeatures")
    gmm_proba        = float(cfg.get("gmm", {}).get("proba", 0.7))
    min_peak_width   = int(cfg.get("gmm", {}).get("min_peak_width", 3))
    frame_interval   = float(cfg.get("gmm", {}).get("frame_interval_ms", 30.0))

    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(output_root, f"{ts}_misclassified_features")
    feat_dir = os.path.join(out_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    # Load NPZ
    logging.info("Loading NPZ: %s", npz_path)
    d = np.load(npz_path)
    traces       = d["traces"]              # (N, 2, T)
    labels       = d["labels"]             # (N,)
    predictions  = d["predictions"]        # (N,)
    wd           = d["wasserstein_distances"]
    class_names  = list(d["class_names"])  # ['Grx1', 'K20Ac']

    grx1_idx  = class_names.index("Grx1")
    k20ac_idx = class_names.index("K20Ac")

    high_wd = wd >= wd_min

    mask_grx1_correct   = high_wd & (labels == grx1_idx)  & (predictions == grx1_idx)
    mask_k20ac_correct  = high_wd & (labels == k20ac_idx) & (predictions == k20ac_idx)
    mask_misclassified  = high_wd & (labels == k20ac_idx) & (predictions == grx1_idx)

    logging.info("WD ≥ %.2f — correctly classified Grx1: %d, K20Ac: %d, misclassified K20Ac→Grx1: %d",
                 wd_min, mask_grx1_correct.sum(), mask_k20ac_correct.sum(), mask_misclassified.sum())

    if mask_misclassified.sum() == 0:
        print("No misclassified K20Ac traces found with WD ≥ %.2f — nothing to plot." % wd_min)
        return

    # Feature extraction for each group
    feats_grx1  = process_group(traces[mask_grx1_correct],  gmm_proba, min_peak_width, frame_interval, "Grx1 (correct)")
    feats_k20ac = process_group(traces[mask_k20ac_correct], gmm_proba, min_peak_width, frame_interval, "K20Ac (correct)")
    feats_misc  = process_group(traces[mask_misclassified], gmm_proba, min_peak_width, frame_interval, "K20Ac→Grx1 (misclassified)")

    feature_keys = ["mean_off_ms", "mean_on_ms", "blinking_rate", "duty_cycle", "cv_on", "cv_off"]
    ylabels      = ["Mean off-time (ms)", "Mean on-time (ms)",
                    "Blinking rate (s⁻¹)", "Duty cycle", "CV on-times", "CV off-times"]
    titles       = ["A) Mean off-time", "B) Mean on-time",
                    "C) Blinking rate", "D) Duty cycle",
                    "E) CV on-times", "F) CV off-times"]

    def _vals(feats, key):
        return [f[key] for f in feats if np.isfinite(f[key])]

    data_per_group = {
        GROUP_LABELS[0]: {},
        GROUP_LABELS[1]: {},
        GROUP_LABELS[2]: {},
    }
    for key in feature_keys:
        data_per_group[GROUP_LABELS[0]][key] = _vals(feats_grx1,  key)
        data_per_group[GROUP_LABELS[1]][key] = _vals(feats_k20ac, key)
        data_per_group[GROUP_LABELS[2]][key] = _vals(feats_misc,  key)

    # Mann-Whitney U: misclassified vs each correctly classified group
    pvals_vs_grx1, pvals_vs_k20ac = [], []
    effect_vs_grx1, effect_vs_k20ac = [], []
    for key in feature_keys:
        p1, r1 = _mannwhitney(data_per_group[GROUP_LABELS[0]][key], data_per_group[GROUP_LABELS[2]][key])
        p2, r2 = _mannwhitney(data_per_group[GROUP_LABELS[1]][key], data_per_group[GROUP_LABELS[2]][key])
        pvals_vs_grx1.append(p1); effect_vs_grx1.append(r1)
        pvals_vs_k20ac.append(p2); effect_vs_k20ac.append(r2)

    finite_vs_grx1 = [p for p in pvals_vs_grx1 if np.isfinite(p)]
    finite_vs_k20ac = [p for p in pvals_vs_k20ac if np.isfinite(p)]
    padj_vs_grx1  = _bh_correct(finite_vs_grx1)  if finite_vs_grx1  else []
    padj_vs_k20ac = _bh_correct(finite_vs_k20ac) if finite_vs_k20ac else []

    # 2×3 violin panel
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for idx, (key, ylabel, title) in enumerate(zip(feature_keys, ylabels, titles)):
        ax = axes_flat[idx]
        group_data = {grp: data_per_group[grp][key] for grp in GROUP_LABELS}
        draw_violin(ax, group_data, GROUP_LABELS, GROUP_COLORS, ylabel, title)

    plt.suptitle(f"Blinking features — high-WD (≥{wd_min}) traces",
                 fontsize=FONTSIZE_TITLE + 2, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(feat_dir, "blinking_features_misclassified.pdf"))
    plt.close()

    # Save CSVs
    for key, title in zip(feature_keys, titles):
        data = {
            "Grx1_correct":          data_per_group[GROUP_LABELS[0]][key],
            "K20Ac_correct":         data_per_group[GROUP_LABELS[1]][key],
            "K20Ac_misclassified":   data_per_group[GROUP_LABELS[2]][key],
        }
        max_len = max(len(v) for v in data.values())
        padded = {k: v + [float('nan')] * (max_len - len(v)) for k, v in data.items()}
        pd.DataFrame(padded).to_csv(
            os.path.join(feat_dir, f"{key}.csv"), index=False
        )

    # Stats summary
    stats_rows = []
    for i, key in enumerate(feature_keys):
        stats_rows.append({
            "feature": key,
            "n_grx1_correct":       len(data_per_group[GROUP_LABELS[0]][key]),
            "n_k20ac_correct":      len(data_per_group[GROUP_LABELS[1]][key]),
            "n_misclassified":      len(data_per_group[GROUP_LABELS[2]][key]),
            "p_vs_grx1":            pvals_vs_grx1[i],
            "p_vs_k20ac":           pvals_vs_k20ac[i],
            "padj_vs_grx1":         padj_vs_grx1[i]  if i < len(padj_vs_grx1)  else float('nan'),
            "padj_vs_k20ac":        padj_vs_k20ac[i] if i < len(padj_vs_k20ac) else float('nan'),
            "effect_r_vs_grx1":     effect_vs_grx1[i],
            "effect_r_vs_k20ac":    effect_vs_k20ac[i],
        })
    pd.DataFrame(stats_rows).to_csv(os.path.join(feat_dir, "stats_summary.csv"), index=False)

    print(f"\nDone. Plots and CSVs saved to: {feat_dir}")
    print(f"  Misclassified K20Ac→Grx1 traces processed: {len(feats_misc)}")


if __name__ == "__main__":
    main()
