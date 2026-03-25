#!/usr/bin/env python3
"""
publication_replots.py

Standalone replotting script for publication-quality figures derived from
existing crossval results. Does not import from or modify any core pipeline scripts.

Run from the repo root or assets/ directory:
    python assets/publication_replots.py
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).parent.parent
CMAP_DIR     = REPO_ROOT / "assets" / "colormaps"
RESULTS_ROOT = Path("/Users/privera/Desktop/Puentener2026_ML")

# ---------------------------------------------------------------------------
# Plotting standards (mirrors docs/plotting_standards.md)
# ---------------------------------------------------------------------------
mpl.rcParams['font.family']     = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.size']       = 7

FONTSIZE_LABEL  = 7
FONTSIZE_TICK   = 6
FONTSIZE_TITLE  = 7
FONTSIZE_LEGEND = 6

COLORS = {
    'orange':    '#E69F00',
    'sky_blue':  '#56B4E9',
    'green':     '#009E73',
    'yellow':    '#F0E442',
    'blue':      '#0072B2',
    'vermillion':'#D55E00',
    'pink':      '#CC79A7',
    'black':     '#000000',
}

MODEL_DISPLAY_NAMES = {
    "orig_conv_gru": "CNN-GRU",
    "resnet1d":      "ResNet",
    "tcn":           "TCN",
}


def _load_cmap(name: str) -> mcolors.LinearSegmentedColormap:
    lut = CMAP_DIR / f"{name}.txt"
    if lut.exists():
        return mcolors.LinearSegmentedColormap.from_list(name, np.loadtxt(lut))
    return mpl.colormaps.get_cmap("viridis")


_lipari_cmap = _load_cmap("lipari")


# ---------------------------------------------------------------------------
# Core heatmap function (self-contained, supports fixed vmin/vmax)
# ---------------------------------------------------------------------------
def plot_heatmap(
    results_df: pd.DataFrame,
    output_path: Path,
    metric: str = "auc_mean",
    title: str = "AUC vs augmentation factor",
    vmin: float = None,
    vmax: float = None,
):
    """
    Plot an augmentation sweep heatmap and save to output_path.

    Args:
        results_df:  DataFrame with columns model, aug_factor, {metric}, {metric_std}.
        output_path: Full path for the output PDF.
        metric:      Column name for the mean values to plot.
        title:       Plot title.
        vmin/vmax:   Color scale limits; if None, derived from the data.
    """
    std_metric = metric.replace("_mean", "_std")

    pivot_mean = results_df.pivot(index="model", columns="aug_factor", values=metric)
    pivot_std  = results_df.pivot(index="model", columns="aug_factor", values=std_metric)

    pivot_mean = pivot_mean[sorted(pivot_mean.columns)]
    pivot_std  = pivot_std[sorted(pivot_std.columns)]

    # Apply display names
    pivot_mean.index = [MODEL_DISPLAY_NAMES.get(m, m) for m in pivot_mean.index]
    pivot_std.index  = pivot_mean.index

    n_models = len(pivot_mean.index)
    n_augs   = len(pivot_mean.columns)

    data_vmin = pivot_mean.values.min() if vmin is None else vmin
    data_vmax = pivot_mean.values.max() if vmax is None else vmax
    midpoint  = (data_vmin + data_vmax) / 2

    fig, ax = plt.subplots(figsize=(max(3.5, n_augs * 1.2), max(2.0, n_models * 0.7)))

    im = ax.imshow(
        pivot_mean.values,
        cmap=_lipari_cmap.reversed(),
        aspect='auto',
        vmin=data_vmin,
        vmax=data_vmax,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    metric_label = {
        "auc_mean":             "AUC",
        "bal_acc_mean":         "Balanced accuracy (%)",
        "time_per_epoch_mean":  "Time per epoch (s)",
    }.get(metric, metric)
    cb.set_label(metric_label, fontsize=FONTSIZE_LEGEND)
    cb.ax.tick_params(labelsize=FONTSIZE_TICK, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)

    ax.set_xticks(range(n_augs))
    ax.set_xticklabels(
        [f"{int(c)}x" if c > 0 else "None" for c in pivot_mean.columns],
        fontsize=FONTSIZE_TICK,
    )
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(pivot_mean.index, fontsize=FONTSIZE_TICK, rotation=90, va='center')

    for i in range(n_models):
        for j in range(n_augs):
            mean_val = pivot_mean.values[i, j]
            std_val  = pivot_std.values[i, j]
            if np.isfinite(mean_val):
                text_color = 'white' if mean_val > midpoint else 'black'
                ax.text(
                    j, i, f"{mean_val:.3f}\n±{std_val:.3f}",
                    ha="center", va="center",
                    color=text_color, fontsize=FONTSIZE_TICK, linespacing=1.3,
                )

    ax.set_xlabel("Augmentation factor", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Model", fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=5)
    ax.tick_params(axis='both', which='major', length=2, width=0.4, direction='out')
    ax.tick_params(axis='both', which='minor', length=1, width=0.3, direction='out')

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Figure 1 — Normalization channel comparison (mirrored, HTHTL vs SNAP)
# ---------------------------------------------------------------------------
def fig_normalization_comparison():
    """
    Replot augmentation sweep heatmaps for the three mirrored HTHTL/SNAP runs
    on a shared color scale, saved as heatmap_normalizationcomparison.pdf in
    each run's augmentation_sweep/ directory.
    """
    runs = {
        "minmax only":   "2026-03-18_04-31-01_mirrored_minmax_4fold_hthtl_snap",
        "z-scored only": "2026-03-18_04-31-37_mirrored_zscored_4fold_hthtl_snap",
        "both channels": "2026-03-17_17-13-41_mirrored_both_4fold_hthtl_snap",
    }

    # Load all data and compute global color scale
    dataframes = {}
    all_auc = []
    for label, run_name in runs.items():
        csv = RESULTS_ROOT / run_name / "augmentation_sweep" / "data.csv"
        df = pd.read_csv(csv)
        dataframes[label] = (run_name, df)
        all_auc.extend(df["auc_mean"].tolist())

    global_vmin = min(all_auc)
    global_vmax = max(all_auc)
    print(f"Global AUC range: {global_vmin:.4f} – {global_vmax:.4f}")

    for label, (run_name, df) in dataframes.items():
        out = RESULTS_ROOT / run_name / "augmentation_sweep" / "heatmap_normalizationcomparison.pdf"
        plot_heatmap(
            df,
            output_path=out,
            metric="auc_mean",
            title=f"AUC vs augmentation factor ({label})",
            vmin=global_vmin,
            vmax=global_vmax,
        )
        print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Normalization channel comparison (not mirrored, HTHTL vs SNAP)
#             Same color scale as the mirrored comparison (0.6906 – 0.9170)
# ---------------------------------------------------------------------------
MIRRORED_VMIN = 0.6906
MIRRORED_VMAX = 0.9170

def fig_normalization_comparison_notmirrored():
    """
    Replot augmentation sweep heatmaps for the three not-mirrored HTHTL/SNAP runs
    on the same color scale as the mirrored normalization comparison, saved as
    heatmap_normalizationcomparison.pdf in each run's augmentation_sweep/ directory.
    """
    runs = {
        "minmax only":   "2026-03-18_04-32-45_notmirrored_minmax_4fold_hthtl_snap",
        "z-scored only": "2026-03-18_05-04-43_notmirrored_zscored_4fold_hthtl_snap",
        "both channels": "2026-03-18_04-32-48_notmirrored_both_4fold_hthtl_snap",
    }

    for label, run_name in runs.items():
        csv = RESULTS_ROOT / run_name / "augmentation_sweep" / "data.csv"
        df = pd.read_csv(csv)
        out = RESULTS_ROOT / run_name / "augmentation_sweep" / "heatmap_normalizationcomparison.pdf"
        plot_heatmap(
            df,
            output_path=out,
            metric="auc_mean",
            title=f"AUC vs augmentation factor ({label})",
            vmin=MIRRORED_VMIN,
            vmax=MIRRORED_VMAX,
        )
        print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure — Normalization channel comparison (background, HTHTL vs SNAP)
#           Shared color scale computed from the three background runs.
# ---------------------------------------------------------------------------
def fig_normalization_comparison_background():
    """
    Replot augmentation sweep heatmaps for the three background HTHTL/SNAP runs
    on a shared color scale derived from their AUC range, saved as
    heatmap_normalizationcomparison.pdf in each run's augmentation_sweep/ directory.
    """
    runs = {
        "minmax only":   "2026-03-18_09-46-28_background_minmax_4fold_hthtl_snap",
        "z-scored only": "2026-03-18_10-02-16_background_zscored_4fold_hthtl_snap",
        "both channels": "2026-03-18_07-02-37_background_both_4fold_hthtl_snap",
    }

    # Load all data and compute global color scale
    dataframes = {}
    all_auc = []
    for label, run_name in runs.items():
        csv = RESULTS_ROOT / run_name / "augmentation_sweep" / "data.csv"
        df = pd.read_csv(csv)
        dataframes[label] = (run_name, df)
        all_auc.extend(df["auc_mean"].tolist())

    global_vmin = min(all_auc)
    global_vmax = max(all_auc)
    print(f"Global AUC range (background): {global_vmin:.4f} – {global_vmax:.4f}")

    for label, (run_name, df) in dataframes.items():
        out = RESULTS_ROOT / run_name / "augmentation_sweep" / "heatmap_normalizationcomparison.pdf"
        plot_heatmap(
            df,
            output_path=out,
            metric="auc_mean",
            title=f"AUC vs augmentation factor ({label})",
            vmin=global_vmin,
            vmax=global_vmax,
        )
        print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3 — Training speed: time per epoch by model, A100 vs H100
#             Two-panel bar chart with broken y-axis (shared scale).
#             Source: noaug runs — mirrored_zscored (A100) and mirrored_minmax (H100).
# ---------------------------------------------------------------------------
def fig_epoch_times():
    """
    Two side-by-side panels (A100 left, H100 right), each a bar chart of
    time per epoch (noaug) for CNN-GRU, ResNet and TCN on a log y-axis.
    Shared scale across both GPU panels for direct visual comparison.
    Output: RESULTS_ROOT/epoch_time_comparison/plot.pdf + data_panel_{A,B}.csv
    """
    MODEL_ORDER  = ["orig_conv_gru", "resnet1d", "tcn"]
    MODEL_LABELS = [MODEL_DISPLAY_NAMES[m] for m in MODEL_ORDER]
    MODEL_COLORS = [COLORS['orange'], COLORS['blue'], COLORS['green']]

    sources = {
        "A100": "2026-03-18_04-31-37_mirrored_zscored_4fold_hthtl_snap",
        "H100": "2026-03-18_04-31-01_mirrored_minmax_4fold_hthtl_snap",
    }

    # Load noaug fold-aggregated metrics
    panel_data = {}
    for gpu, run_name in sources.items():
        df    = pd.read_csv(RESULTS_ROOT / run_name / "augmentation_sweep" / "data.csv")
        noaug = df[df.aug_label == "noaug"].set_index("model")
        panel_data[gpu] = {
            "means": [noaug.loc[m, "time_per_epoch_mean"] for m in MODEL_ORDER],
            "stds":  [noaug.loc[m, "time_per_epoch_std"]  for m in MODEL_ORDER],
        }

    x         = np.arange(len(MODEL_ORDER))
    bar_width = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0), sharey=True)

    for ax, gpu in zip(axes, ["A100", "H100"]):
        means = panel_data[gpu]["means"]
        stds  = panel_data[gpu]["stds"]

        for i, (mean, std, color) in enumerate(zip(means, stds, MODEL_COLORS)):
            ax.bar(
                i, mean, yerr=std, capsize=3,
                color=color, alpha=0.85,
                edgecolor=COLORS['black'], linewidth=0.6,
                error_kw=dict(linewidth=0.8, capthick=0.8),
                width=bar_width,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, fontsize=FONTSIZE_TICK)
        ax.set_title(gpu, fontsize=FONTSIZE_TITLE, pad=5)
        ax.tick_params(which='major', labelsize=FONTSIZE_TICK, length=4, width=0.8, direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.1)
        ax.spines['bottom'].set_linewidth(1.1)

    axes[0].set_ylabel("Time per epoch (s)", fontsize=FONTSIZE_LABEL)

    # ---- Save ----
    output_dir = RESULTS_ROOT / "epoch_time_comparison"
    output_dir.mkdir(exist_ok=True)
    plt.tight_layout(w_pad=3.0)
    plt.savefig(output_dir / "plot.pdf", dpi=450, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/plot.pdf")

    # Source data
    for gpu, run_name in sources.items():
        df    = pd.read_csv(RESULTS_ROOT / run_name / "augmentation_sweep" / "data.csv")
        noaug = df[df.aug_label == "noaug"][['model', 'time_per_epoch_mean', 'time_per_epoch_std']].copy()
        noaug['gpu'] = gpu
        panel_id = 'A' if gpu == 'A100' else 'B'
        noaug.to_csv(output_dir / f"data_panel_{panel_id}.csv", index=False)


# ---------------------------------------------------------------------------
# Figure 4 — Replot all confusion matrices with Helvetica 12 pt
#             Overwrites existing confmat PDFs in every experiment folder.
# ---------------------------------------------------------------------------
_davos_r = _load_cmap("davos").reversed()


def _plot_confmat_pub(
    mean_cm: np.ndarray,
    std_cm: np.ndarray,
    class_names: list,
    save_path: Path,
    title: str,
):
    """
    Confusion matrix styled for publication: Helvetica 12 pt, davos_r colormap,
    vmin=0 / vmax=1.  Reproduces the layout of plot_confusion_matrix_with_std
    from utils.py with the font override applied.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Local font override — Helvetica 20 pt
    with mpl.rc_context({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':       20,
    }):
        n = len(class_names)
        fig, ax = plt.subplots(figsize=(7, 7))

        im = ax.imshow(mean_cm, cmap=_davos_r, vmin=0.0, vmax=1.0)

        for i in range(n):
            for j in range(n):
                color = "white" if mean_cm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{mean_cm[i, j]:.3f}\n±{std_cm[i, j]:.3f}",
                        ha="center", va="center",
                        color=color, fontsize=20, linespacing=1.3)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(class_names, fontsize=20)
        ax.set_yticklabels(class_names, fontsize=20)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=20, length=2, width=0.4)
        cb.outline.set_linewidth(0.4)

        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Predicted label", fontsize=20)
        ax.set_ylabel("True label", fontsize=20)
        ax.tick_params(axis="both", which='major', labelsize=20, length=2, width=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
        ax.grid(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=450, bbox_inches='tight')
        plt.close()


def replot_confmats():
    """
    Replot all confusion matrices for every experiment in RESULTS_ROOT using
    Helvetica 12 pt, overwriting the original PDFs.
    """
    def slug(s):
        return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")

    total = 0
    for run_dir in sorted(RESULTS_ROOT.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "slurm":
            continue

        cfg_file = run_dir / "config_snapshot.json"
        if not cfg_file.exists():
            continue
        n_splits = json.load(open(cfg_file)).get("cv", {}).get("n_splits", 4)
        aug_dir  = run_dir / "augmentation_sweep"

        for csv_file, filtered in [
            (aug_dir / "cv_confmat.csv",          False),
            (aug_dir / "cv_confmat_filtered.csv", True),
        ]:
            if not csv_file.exists():
                continue
            df = pd.read_csv(csv_file)

            for (model_name, aug_label), grp in df.groupby(["model", "aug_label"]):
                class_names = sorted(grp["row_cls"].unique().tolist())
                n = len(class_names)
                mean_cm = np.zeros((n, n))
                std_cm  = np.zeros((n, n))
                for _, row in grp.iterrows():
                    i = class_names.index(row["row_cls"])
                    j = class_names.index(row["col_cls"])
                    mean_cm[i, j] = row["mean"]
                    std_cm[i, j]  = row["std"]

                suffix = "_filtered" if filtered else ""
                out    = aug_dir / f"confmat_{slug(model_name)}_{aug_label}{suffix}.pdf"
                tag    = "filtered" if filtered else ""
                title  = (f"Confusion matrix - {model_name} "
                          f"({aug_label}, {n_splits}-fold CV"
                          + (", filtered)" if filtered else ")"))

                _plot_confmat_pub(mean_cm, std_cm, class_names, out, title)
                total += 1

        print(f"  {run_dir.name}: done")

    print(f"\nReplotted {total} confusion matrices.")


# ---------------------------------------------------------------------------
# Entry point — run all publication figures
# ---------------------------------------------------------------------------
# Figure — HTHTL trace examples (5×5 grid, randomly selected)
# ---------------------------------------------------------------------------
def fig_trace_examples(pkl_name="HTHTL_IN_gmm_all_minmax_traces.pkl", seed=42):
    """
    Load HTHTL minmax traces, randomly select 25, and plot them in a 5×5 grid.
    Each panel is titled with the movie ID and (x, y) pixel coordinates.
    Also saves a lookup CSV with the full origin path for each trace.
    Output: trace_vs_background/<stem>_examples.pdf
             trace_vs_background/<stem>_examples_lookup.csv
    """
    import pickle, re

    stem      = Path(pkl_name).stem
    pkl_path  = RESULTS_ROOT / "trace_vs_background" / pkl_name
    uid_path  = RESULTS_ROOT / "trace_vs_background" / "_HTHTL_uniqueID_IN.pkl"
    out_path  = RESULTS_ROOT / "trace_vs_background" / f"{stem}_examples.pdf"
    csv_path  = RESULTS_ROOT / "trace_vs_background" / f"{stem}_examples_lookup.csv"

    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    with open(uid_path, "rb") as f:
        df_uid = pickle.load(f).set_index("uniqueID")

    # Same selection as before
    rng = np.random.default_rng(seed)
    selected = rng.choice(df.columns, size=25, replace=False)
    frames = df.index.to_numpy()

    # Build lookup: uniqueID → x, y, experiment date, movie number, full origin
    def _parse_origin(origin):
        p = Path(origin)
        stem = p.stem  # e.g. HTHTL_640nm_..._0018_traces
        # Extract trailing movie number (e.g. _0018_traces → 0018, or no number)
        m = re.search(r'_(\d{4})_traces$', stem)
        movie_num = m.group(1) if m else "0000"
        # Extract experiment date from path parts
        parts = p.parts
        date_part = next((pt for pt in parts if re.match(r'\d{8}_', pt)), "unknown")
        return date_part, movie_num

    records = []
    for uid in selected:
        row = df_uid.loc[uid]
        date_part, movie_num = _parse_origin(row["origin"])
        records.append({
            "uniqueID":    uid,
            "x":           row["x"],
            "y":           row["y"],
            "experiment":  date_part,
            "movie_num":   movie_num,
            "origin":      row["origin"],
        })
    lookup = pd.DataFrame(records)
    lookup.to_csv(csv_path, index=False)
    print(f"Saved lookup: {csv_path}")

    # Plot
    fig, axes = plt.subplots(5, 5, figsize=(9, 7.5), sharey=True)
    fig.subplots_adjust(hspace=0.55, wspace=0.15, left=0.07, right=0.98, top=0.95, bottom=0.06)

    for ax, rec in zip(axes.flat, records):
        trace = df[rec["uniqueID"]].to_numpy()
        ax.plot(frames, trace, lw=0.6, color=COLORS["blue"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"{rec['experiment'][:8]}  #{rec['movie_num']}\n({rec['x']}, {rec['y']})",
            fontsize=4.5, pad=2,
        )
        ax.tick_params(axis="both", labelsize=FONTSIZE_TICK, length=2, width=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.5)

    fig.text(0.5,  0.005, "Frame", ha="center", fontsize=FONTSIZE_LABEL)
    fig.text(0.01, 0.5,   "Intensity (a.u.)", va="center", rotation="vertical", fontsize=FONTSIZE_LABEL)

    plt.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure — Data augmentation demonstration, 6-panel (trace 74444)
#   Row 1: original (0–700) | time-reversed (0–700) | zoom overlay (330–400)
#   Row 2: +noise (0–700)   | +time warp (0–700)    | +jitter (0–700)
#
# Augmentation parameters from config_cv.yaml: sigma=0.05 for all three.
# Time reversal from filter.py mirror_trace; time warp from ML/utils.py apply_augmentation.
# ---------------------------------------------------------------------------
def fig_augmentation_demo(
    uid=74444,
    frame_end=700,
    win_start=330,
    win_end=400,
    pkl_name="HTHTL_IN_gmm_filtered_minmax_traces.pkl",
    seed=0,
):
    """
    Six-panel figure demonstrating data augmentation on a single trace.
    Output: trace_vs_background/augmentation_demo.pdf
    """
    import pickle
    from scipy.interpolate import interp1d

    pkl_path = RESULTS_ROOT / "trace_vs_background" / pkl_name
    out_path  = RESULTS_ROOT / "trace_vs_background" / "augmentation_demo.pdf"

    with open(pkl_path, "rb") as f:
        df = pickle.load(f)

    rng   = np.random.default_rng(seed)
    trace = df[uid].to_numpy()
    T     = len(trace)

    noise_sigma      = 0.05
    magnitude_jitter = 0.05
    time_warp_sigma  = 0.05

    # -- Augmented versions (full trace, each applied independently to original) --

    # Time reversal: reverse signal region 0–frame_end (mirror_trace from filter.py)
    mirrored = trace.copy()
    mirrored[0:frame_end] = mirrored[0:frame_end][::-1]

    # Gaussian noise
    noisy = trace.copy()
    noisy = noisy + rng.normal(0, noise_sigma * np.std(noisy), T)

    # Time warping (apply_augmentation from ML/utils.py)
    warp_factor = float(np.clip(1.0 + rng.normal(0, time_warp_sigma), 0.95, 1.05))
    warped_indices = np.linspace(0, T - 1, int(T * warp_factor))
    f_warp = interp1d(np.arange(T), trace, kind='linear',
                      bounds_error=False, fill_value='extrapolate')
    warped_trace = f_warp(warped_indices)
    f_resamp = interp1d(np.arange(len(warped_trace)), warped_trace, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    warped = f_resamp(np.linspace(0, len(warped_trace) - 1, T))

    # Magnitude jitter
    jitter_f = float(np.clip(1.0 + rng.normal(0, magnitude_jitter), 0.95, 1.05))
    jittered = trace.copy() * jitter_f

    # -- Colours --------------------------------------------------------------
    C_ORIG   = COLORS["black"]
    C_NOISE  = COLORS["orange"]
    C_WARP   = COLORS["vermillion"]
    C_JITTER = COLORS["blue"]

    # -- Layout ---------------------------------------------------------------
    t_full = np.arange(frame_end)
    t_zoom = np.arange(win_start, win_end)

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 4.2))
    fig.subplots_adjust(hspace=0.42, wspace=0.30,
                        left=0.07, right=0.98, top=0.93, bottom=0.13)

    def _style(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_linewidth(0.5)
        ax.tick_params(axis="both", labelsize=FONTSIZE_TICK, length=2, width=0.4)

    lw_full = 0.7
    lw_zoom = 1.0
    ylim    = (-0.05, 1.10)

    # Row 1 col 0 — original
    ax = axes[0, 0]
    ax.plot(t_full, trace[:frame_end], lw=lw_full, color=C_ORIG)
    ax.set_title("Original", fontsize=FONTSIZE_TITLE, pad=3)
    ax.set_ylim(ylim); _style(ax)

    # Row 1 col 1 — time-reversed
    ax = axes[0, 1]
    ax.plot(t_full, mirrored[:frame_end], lw=lw_full, color=C_ORIG)
    ax.set_title("Time reversal", fontsize=FONTSIZE_TITLE, pad=3)
    ax.set_ylim(ylim); _style(ax)

    # Row 1 col 2 — zoom overlay
    ax = axes[0, 2]
    ax.plot(t_zoom, trace[win_start:win_end],    lw=lw_zoom, color=C_ORIG,   label="Original", zorder=4)
    ax.plot(t_zoom, noisy[win_start:win_end],    lw=lw_zoom, color=C_NOISE,  label="Noise",    zorder=3)
    ax.plot(t_zoom, warped[win_start:win_end],   lw=lw_zoom, color=C_WARP,   label="Warp",     zorder=2)
    ax.plot(t_zoom, jittered[win_start:win_end], lw=lw_zoom, color=C_JITTER, label="Jitter",   zorder=1)
    ax.set_title(f"Zoom (frames {win_start}–{win_end})", fontsize=FONTSIZE_TITLE, pad=3)
    ax.set_ylim(ylim); _style(ax)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")

    # Row 2 col 0 — noise
    ax = axes[1, 0]
    ax.plot(t_full, noisy[:frame_end], lw=lw_full, color=C_NOISE)
    ax.set_title("Gaussian noise", fontsize=FONTSIZE_TITLE, pad=3)
    ax.set_ylim(ylim); _style(ax)

    # Row 2 col 1 — time warp
    ax = axes[1, 1]
    ax.plot(t_full, warped[:frame_end], lw=lw_full, color=C_WARP)
    ax.set_title("Time warping", fontsize=FONTSIZE_TITLE, pad=3)
    ax.set_ylim(ylim); _style(ax)

    # Row 2 col 2 — jitter
    ax = axes[1, 2]
    ax.plot(t_full, jittered[:frame_end], lw=lw_full, color=C_JITTER)
    ax.set_title("Magnitude jitter", fontsize=FONTSIZE_TITLE, pad=3)
    ax.set_ylim(ylim); _style(ax)

    # Axis labels
    for ax in axes[1]:
        ax.set_xlabel("Frame", fontsize=FONTSIZE_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel("Intensity (a.u.)", fontsize=FONTSIZE_LABEL)

    plt.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig_normalization_comparison()
    fig_normalization_comparison_notmirrored()
    fig_normalization_comparison_background()
    fig_epoch_times()
    replot_confmats()
