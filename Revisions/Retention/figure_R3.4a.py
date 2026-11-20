#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Figure R3.4a (first-round review, Reviewer 3, point 4, first half): how many traces the uncertainty
(Wasserstein distance, WD) filter keeps, and what accuracy it buys.

Rows a-c: for the cross-validation runs behind Figs. 4C, 5B and 6B (4 folds each), the percentage of
  test traces kept as a function of the WD threshold, for protein traces and for the matching
  background-trace control; the paper's threshold (0.7) is marked.
Rows d-f: balanced accuracy against the percentage of protein test traces kept (the same sweeps), with
  each fold's value at WD = 0.7.
Panels g-h: the final models behind Figs. 5D and 6C (single train/validation/test split), percentage
  of test traces kept per protein against the WD threshold.

Every number is read from the paper's published outputs; nothing is retrained.
  - Cross-validation stores the sweep on 50 thresholds per fold (0 to the fold's largest WD), not at
    0.7. The kept fraction can only fall as the threshold rises, so the grid points on either side
    of 0.7 bound the true value; the single value is the straight-line interpolation between them
    (bounds in R3.4a_wd07_by_fold.csv). Folds whose largest WD is below 0.7 keep nothing (0 %).
  - Test folds are balanced (the larger class subsampled to the smaller), so kept fractions are of a
    balanced test set and pooled over both proteins. Per-protein values exist only for the final
    models, whose per-trace WD is stored.
  - Fold summaries: mean +/- SD with SD over the 4 folds dividing by 4 (population SD), as in the
    paper's figures.
  - Approximate yield per construct = quality-filter pass rate (step 0 of the review) x the pair's
    mean kept fraction at WD 0.7. Approximate because the kept fraction is pooled over both proteins
    and measured on balanced test folds.
  No statistical tests.

Usage:
    python figure_R3.4a.py
        Replot from the CSVs next to this script (works in the figure folder as distributed).
    python figure_R3.4a.py --build --general-data <GeneralData> --step0 <PaperData check run> [--out <dir>]
        Rebuild the CSVs from the published outputs, replot, and copy this script into --out.
        <GeneralData>: the Zenodo GeneralData folder (MLresults/). <PaperData check run>: the review's
        step 0 output (funnel.csv, rois.csv; blinkognition Revisions/PaperData/check_data.py).

Outputs (in --out, default: this script's folder): R3.4a.pdf, R3.4a_panels_a-c.csv,
R3.4a_panels_d-f.csv, R3.4a_panels_g-h.csv, R3.4a_wd07_by_fold.csv, R3.4a_table_cv.csv,
R3.4a_table_final.csv, R3.4a_table_funnel.csv, R3.4a_caption.txt, R3.4a_provenance.yaml (build only),
figure_R3.4a.py (build only).
"""

from __future__ import annotations

# Standard library
import argparse
import hashlib
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# --- Constants ---
FIG = "R3.4a"
WD_PAPER = 0.7                       # the paper's cross-validation threshold (and Figs. 5D, 6C)
FINAL_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)

# Okabe-Ito; construct colors as in Figs. 5D and 6C (Features/features.py PROTEIN_COLORS)
COLORS = {"black": "#000000", "sky_blue": "#56B4E9", "blue": "#0072B2", "orange": "#E69F00",
          "green": "#009E73", "pink": "#CC79A7", "vermillion": "#D55E00"}
PROTEIN_COLORS = {"HaloD106": COLORS["blue"], "HaloK117": COLORS["orange"], "SNAPC148": COLORS["pink"],
                  "scGrx1": COLORS["green"], "scGrx1AcK20": COLORS["vermillion"]}
LABELS = {"HaloD106": "Halo-D106", "HaloK117": "Halo-K117", "SNAPC148": "SNAP-C148",
          "scGrx1": "scGrx1", "scGrx1AcK20": "scGrx1-AcK20"}
COLOR_PROTEIN_TRACES = COLORS["black"]
COLOR_BACKGROUND_TRACES = COLORS["sky_blue"]

# The cross-validation runs behind the main-figure confusion matrices (matched to the published
# numbers): min-max traces, time reversal, no further augmentation; background controls without
# time reversal (the only background runs).
PAIRS = [
    {"pair": "HaloD106_SNAPC148", "figure": "4C", "model": "orig_conv_gru", "model_label": "CNN-GRU",
     "proteins": ("HaloD106", "SNAPC148"), "final": None},
    {"pair": "HaloD106_HaloK117", "figure": "5B", "model": "orig_conv_gru", "model_label": "CNN-GRU",
     "proteins": ("HaloD106", "HaloK117"),
     "final": ("5D", "2026-05-19_08-21-58_orig_conv_gru_mirrored_IN_training_HaloD106_HaloK117")},
    {"pair": "scGrx1_scGrx1AcK20", "figure": "6B", "model": "tcn", "model_label": "TCN",
     "proteins": ("scGrx1", "scGrx1AcK20"),
     "final": ("6C", "2026-05-19_10-14-25_tcn_mirrored_IN_training_scGrx1_scGrx1AcK20")},
]
RUN_GLOB = {"protein": "*_protein_mirrored_*_minmax_4fold_*", "background": "*_background_notmirrored_*_minmax_4fold_*"}
AUG_FACTOR = 0
FINAL_CLASS_KEYS = {"HTHTL": "HaloD106", "HTIA": "HaloK117", "Grx1": "scGrx1", "K20Ac": "scGrx1AcK20"}

FIG_WIDTH_IN = 180 / 25.4            # double column
FIG_HEIGHT_IN = 6.3
LW_FOLD = 0.8
PANEL_LETTERS = "abcdefgh"
TEXT_CLEAR_YMAX = 0.80               # the WD 0.7 line in panels a-c stops below the kept-percentage label
TEXT_CLEAR_YMIN = 0.28               # and in panels g-h starts above it
ACC_YLIM = (45, 100)                 # balanced accuracy is 50 % by definition when one class is left


# --- Style ---

def apply_style() -> None:
    """The paper's figure style (locbplots): Helvetica, else Arial; 7 pt text, 6 pt ticks and legend."""
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Microsoft/Arial.ttf"):
        if Path(path).exists():
            fm.fontManager.addfont(path)
            break
    available = {f.name for f in fm.fontManager.ttflist}
    sans = ["Helvetica", "Arial", "DejaVu Sans"] if "Helvetica" in available else ["Arial", "DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": sans, "font.size": 7,
        "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "lines.linewidth": 1.2, "axes.grid": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": "tight",
    })


def axis_standards(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", which="major", labelsize=6, length=4, width=0.8, direction="out")


# --- Build: published outputs to plotted data ---

def _one(paths: list[Path], what: str) -> Path:
    paths = [p for p in paths if not p.name.startswith("._")]   # macOS AppleDouble files on exFAT
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one {what}, found {len(paths)}: {paths}")
    return paths[0]


def _at_threshold(g: pd.DataFrame, t: float) -> dict:
    """Kept fraction and balanced accuracy at threshold t from one fold's grid sweep:
    bracketing grid points and linear interpolation between them."""
    g = g.sort_values("wd_thresh")
    below, above = g[g["wd_thresh"] <= t].iloc[-1], g[g["wd_thresh"] > t]
    out = {"wd_grid_below": below["wd_thresh"], "kept_pct_upper_bound": 100 * below["kept_frac"],
           "bal_acc_at_grid_below": below["bal_acc"], "max_wd": g["wd_thresh"].max()}
    if above.empty:                   # largest WD below t: nothing kept at t
        out.update(wd_grid_above=np.nan, kept_pct_lower_bound=0.0, kept_pct=0.0, bal_acc_kept=np.nan)
        return out
    above = above.iloc[0]
    w = (t - below["wd_thresh"]) / (above["wd_thresh"] - below["wd_thresh"])
    out.update(wd_grid_above=above["wd_thresh"], kept_pct_lower_bound=100 * above["kept_frac"],
               kept_pct=100 * ((1 - w) * below["kept_frac"] + w * above["kept_frac"]),
               bal_acc_kept=(1 - w) * below["bal_acc"] + w * above["bal_acc"])
    return out


def build_cv(general_data: Path, inputs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long-format sweeps (panels a-f) and per-fold values at WD 0.7."""
    sweeps, at07 = [], []
    for p in PAIRS:
        for traces, pattern in RUN_GLOB.items():
            run = _one(sorted((general_data / "MLresults" / "CrossValidation" / p["pair"]).glob(pattern)),
                       f"{traces} run of {p['pair']}")
            sw_p, fm_p = run / "MCD_results" / "cv_wd_sweep.csv", run / "cv_folds_metrics.csv"
            inputs.extend([sw_p, fm_p])
            sw = pd.read_csv(sw_p)
            sw = sw[(sw["model"] == p["model"]) & (sw["aug_factor"] == AUG_FACTOR)]
            folds = pd.read_csv(fm_p)
            folds = folds[(folds["model"] == p["model"]) & (folds["aug_factor"] == AUG_FACTOR)].set_index("fold")
            if sw.empty or folds.empty:
                raise ValueError(f"no {p['model']} aug {AUG_FACTOR} rows in {run}")
            base = {"pair": p["pair"], "figure": p["figure"], "model": p["model_label"], "traces": traces,
                    "run": run.name}
            for fold, g in sw.groupby("fold"):
                g = g.sort_values("wd_thresh")
                sweeps.append(pd.DataFrame({**base, "fold": int(fold), "wd_threshold": g["wd_thresh"].values,
                                            "kept_pct": 100 * g["kept_frac"].values,
                                            "balanced_accuracy_pct": g["bal_acc"].values}))
                row = {**base, "fold": int(fold), "n_test_traces": int(folds.loc[fold, "n_val"]),
                       "bal_acc_all": g.iloc[0]["bal_acc"], **_at_threshold(g, WD_PAPER)}
                at07.append(row)
    return pd.concat(sweeps, ignore_index=True), pd.DataFrame(at07)


def build_final(general_data: Path, inputs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-protein kept percentage against the WD threshold for the final models (panels g-h), and
    the values at WD 0.7."""
    curves, table = [], []
    for p in PAIRS:
        if p["final"] is None:
            continue
        fig, name = p["final"]
        npz_p = general_data / "MLresults" / "FinalModels" / name / "MCD_results" / "traces_with_wasserstein.npz"
        inputs.append(npz_p)
        z = np.load(npz_p)
        y, yhat, wd = z["labels"], z["predictions"], z["wasserstein_distances"]
        names = [FINAL_CLASS_KEYS[str(c)] for c in z["class_names"]]
        for k, prot in enumerate(names):
            m = y == k
            kept = np.array([(wd[m] >= t).mean() for t in FINAL_GRID]) * 100
            curves.append(pd.DataFrame({"pair": p["pair"], "figure": fig, "model": p["model_label"],
                                        "protein": prot, "wd_threshold": FINAL_GRID, "kept_pct": kept}))
            keep = m & (wd >= WD_PAPER)
            table.append({"pair": p["pair"], "figure": fig, "model": p["model_label"], "final_model": name,
                          "protein": prot, "n_test_traces": int(m.sum()), "n_kept_wd07": int(keep.sum()),
                          "kept_pct_wd07": 100 * keep.mean() / m.mean(),
                          "n_kept_wd07_correct": int((keep & (yhat == y)).sum()),
                          "recall_kept_wd07_pct": 100 * (keep & (yhat == y)).sum() / max(keep.sum(), 1),
                          "recall_all_pct": 100 * ((yhat == y) & m).sum() / m.sum()})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(table)


def summarize_cv(at07: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- population SD over folds, per pair and trace type."""
    rows = []
    for (pair, traces), g in at07.groupby(["pair", "traces"], sort=False):
        r = {"pair": pair, "figure": g["figure"].iloc[0], "model": g["model"].iloc[0], "traces": traces,
             "n_folds": len(g), "n_test_traces_per_fold": "/".join(str(n) for n in g["n_test_traces"])}
        for col, name in (("kept_pct", "kept_pct_wd07"), ("kept_pct_lower_bound", "kept_pct_wd07_lower_bound"),
                          ("kept_pct_upper_bound", "kept_pct_wd07_upper_bound"), ("bal_acc_all", "bal_acc_all_pct"),
                          ("bal_acc_kept", "bal_acc_kept_wd07_pct")):
            v = g[col].astype(float)
            r[f"{name}_mean"] = v.mean()
            r[f"{name}_sd"] = v.std(ddof=0)
        rows.append(r)
    return pd.DataFrame(rows)


def build_funnel(step0: Path, cv_table: pd.DataFrame, final_table: pd.DataFrame, inputs: list[Path]) -> pd.DataFrame:
    """Per construct and pair: ROIs, single ROIs inside vesicles, valid traces, quality-filtered traces,
    confidence retention, approximate yield."""
    f_p, r_p = step0 / "funnel.csv", step0 / "rois.csv"
    inputs.extend([f_p, r_p])
    funnel = pd.read_csv(f_p)
    funnel = funnel[funnel["where"] == "IN"].set_index("construct")
    rois = pd.read_csv(r_p).groupby(["construct", "overlap", "ground_truth"])["n_rois"].sum()
    prot = cv_table[cv_table["traces"] == "protein"].set_index("pair")
    rows = []
    for p in PAIRS:
        for c in p["proteins"]:
            n_single_in = int(rois.loc[(c, "single", "IN")])
            if n_single_in != int(funnel.loc[c, "n_ids"]):
                raise ValueError(f"{c}: single IN ROIs {n_single_in} != IN IDs {funnel.loc[c, 'n_ids']}")
            q = funnel.loc[c, "n_filtered"] / funnel.loc[c, "n_unfiltered"]
            kept = prot.loc[p["pair"], "kept_pct_wd07_mean"]
            fin = final_table[(final_table["pair"] == p["pair"]) & (final_table["protein"] == c)]
            rows.append({
                "construct": c, "pair": p["pair"], "figure": p["figure"],
                "rois_all": int(rois.loc[c].sum()),
                "rois_overlapping": int(rois.loc[(c, "overlapping")].sum()),
                "rois_single_out": int(rois.loc[(c, "single", "OUT")]),
                "rois_single_in": n_single_in,
                "traces_valid_in": int(funnel.loc[c, "n_unfiltered"]),
                "traces_quality_filtered_in": int(funnel.loc[c, "n_filtered"]),
                "quality_filter_pass_pct": 100 * q,
                "cv_kept_pct_wd07_pair_mean": kept,
                "approx_yield_pct": q * kept,
                "final_model_kept_pct_wd07": float(fin["kept_pct_wd07"].iloc[0]) if len(fin) else np.nan,
            })
    return pd.DataFrame(rows)


# --- Plot ---

def _panel_letter(ax, letter: str) -> None:
    ax.text(-0.28, 1.12, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left")


def plot(out: Path) -> Path:
    sweeps = pd.read_csv(out / f"{FIG}_panels_a-c.csv")
    at07 = pd.read_csv(out / f"{FIG}_wd07_by_fold.csv")
    cv = pd.read_csv(out / f"{FIG}_table_cv.csv")
    final = pd.read_csv(out / f"{FIG}_panels_g-h.csv")
    ftab = pd.read_csv(out / f"{FIG}_table_final.csv")

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    gs = fig.add_gridspec(3, 3, hspace=0.75, wspace=0.42, left=0.08, right=0.98, top=0.93, bottom=0.07)
    letters = iter(PANEL_LETTERS)

    for j, p in enumerate(PAIRS):
        a = fig.add_subplot(gs[0, j])
        for traces, color, ls in (("background", COLOR_BACKGROUND_TRACES, "--"), ("protein", COLOR_PROTEIN_TRACES, "-")):
            s = sweeps[(sweeps["pair"] == p["pair"]) & (sweeps["traces"] == traces)]
            for _, g in s.groupby("fold"):
                a.plot(g["wd_threshold"], g["kept_pct"], color=color, ls=ls, lw=LW_FOLD)
        a.axvline(WD_PAPER, ymax=TEXT_CLEAR_YMAX, color=COLORS["black"], ls=":", lw=0.8)
        o = at07[(at07["pair"] == p["pair"]) & (at07["traces"] == "protein")]
        a.plot(np.full(len(o), WD_PAPER), o["kept_pct"], ls="none", marker="o", ms=3.5, mfc="white",
               mec=COLOR_PROTEIN_TRACES, mew=0.8, zorder=3)
        a.set(xlim=(0, 1), ylim=(0, 102), xlabel="Wasserstein distance threshold", ylabel="Test traces kept (%)")
        names = " vs ".join(LABELS[c] for c in p["proteins"])
        a.set_title(f"{names}\n(Fig. {p['figure']}, {p['model_label']}, 4 folds)", pad=4)
        r_p = cv[(cv["pair"] == p["pair"]) & (cv["traces"] == "protein")].iloc[0]
        r_b = cv[(cv["pair"] == p["pair"]) & (cv["traces"] == "background")].iloc[0]
        a.text(0.99, 0.99, f"Protein {r_p['kept_pct_wd07_mean']:.0f} ± {r_p['kept_pct_wd07_sd']:.0f} %\n"
               f"Background {r_b['kept_pct_wd07_mean']:.1f} ± {r_b['kept_pct_wd07_sd']:.1f} %",
               transform=a.transAxes, fontsize=6, color="black", va="top", ha="right")
        axis_standards(a)
        _panel_letter(a, next(letters))

    for j, p in enumerate(PAIRS):
        b = fig.add_subplot(gs[1, j])
        s = sweeps[(sweeps["pair"] == p["pair"]) & (sweeps["traces"] == "protein")]
        for _, g in s.groupby("fold"):
            b.plot(g["kept_pct"], g["balanced_accuracy_pct"], color=COLOR_PROTEIN_TRACES, lw=LW_FOLD)
        o = at07[(at07["pair"] == p["pair"]) & (at07["traces"] == "protein")]
        b.plot(o["kept_pct"], o["bal_acc_kept"], ls="none", marker="o", ms=3.5, mfc="white",
               mec=COLOR_PROTEIN_TRACES, mew=0.8, zorder=3)
        b.set(xlim=(0, 102), ylim=ACC_YLIM, xlabel="Test traces kept (%)", ylabel="Balanced accuracy (%)")
        b.set_yticks(np.arange(50, 101, 10))
        r = cv[(cv["pair"] == p["pair"]) & (cv["traces"] == "protein")].iloc[0]
        b.text(0.97, 0.04, f"All traces: {r['bal_acc_all_pct_mean']:.1f} ± {r['bal_acc_all_pct_sd']:.1f} %\n"
               f"WD ≥ 0.7: {r['bal_acc_kept_wd07_pct_mean']:.1f} ± {r['bal_acc_kept_wd07_pct_sd']:.1f} %",
               transform=b.transAxes, fontsize=6, color="black", va="bottom", ha="right")
        axis_standards(b)
        _panel_letter(b, next(letters))

    leg = fig.add_subplot(gs[2, 0])
    leg.axis("off")
    handles = [
        Line2D([], [], color=COLOR_PROTEIN_TRACES, lw=LW_FOLD, label="Protein test traces (one line per fold)"),
        Line2D([], [], color=COLOR_BACKGROUND_TRACES, lw=LW_FOLD, ls="--", label="Background test traces (one line per fold)"),
        Line2D([], [], color=COLORS["black"], lw=0.8, ls=":", label="WD = 0.7 (paper's threshold)"),
        Line2D([], [], ls="none", marker="o", ms=3.5, mfc="white", mec=COLOR_PROTEIN_TRACES, mew=0.8,
               label="Fold value at WD = 0.7"),
    ] + [Line2D([], [], color=PROTEIN_COLORS[c], lw=1.0, label=f"{LABELS[c]} (final model)")
         for c in ("HaloD106", "HaloK117", "scGrx1", "scGrx1AcK20")]
    leg.legend(handles=handles, loc="center left", frameon=False, handlelength=2.2, borderaxespad=0)

    for j, p in enumerate(PAIRS):
        if p["final"] is None:
            continue
        c_ax = fig.add_subplot(gs[2, j])
        f = final[final["pair"] == p["pair"]]
        lines = []
        for prot in p["proteins"]:
            g = f[f["protein"] == prot]
            c_ax.plot(g["wd_threshold"], g["kept_pct"], color=PROTEIN_COLORS[prot], lw=1.0)
            t = ftab[(ftab["pair"] == p["pair"]) & (ftab["protein"] == prot)].iloc[0]
            lines.append(f"{LABELS[prot]} {t['kept_pct_wd07']:.0f} % ({t['n_kept_wd07']}/{t['n_test_traces']})")
        c_ax.axvline(WD_PAPER, ymin=TEXT_CLEAR_YMIN, color=COLORS["black"], ls=":", lw=0.8)
        c_ax.set(xlim=(0, 1), ylim=(0, 102), xlabel="Wasserstein distance threshold", ylabel="Test traces kept (%)")
        c_ax.set_title(f"Final {p['model_label']} model (Fig. {p['final'][0]}), test split", pad=4)
        c_ax.text(0.03, 0.04, "Kept at WD 0.7:\n" + "\n".join(lines), transform=c_ax.transAxes, fontsize=6,
                  color="black", va="bottom", ha="left")
        axis_standards(c_ax)
        _panel_letter(c_ax, next(letters))

    pdf = out / f"{FIG}.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


# --- Caption and provenance ---

CAPTION = (
    "Figure R3.4a. Fraction of single-molecule traces kept by the uncertainty filter. "
    "a-c) Percentage of test traces with a Wasserstein distance (WD) at or above each threshold in the "
    "four-fold cross-validations behind Figures 4C (Halo-D106 vs SNAP-C148, CNN-GRU), 5B (Halo-D106 vs "
    "Halo-K117, CNN-GRU) and 6B (scGrx1 vs scGrx1-AcK20, TCN), for protein traces (solid) and the "
    "corresponding background-trace controls (dashed); one line per fold; dotted line, WD = 0.7, the "
    "threshold used in the paper; circles, each fold's protein traces at WD = 0.7; numbers, percentage "
    "kept at WD = 0.7. d-f) Balanced accuracy against the percentage of protein test traces kept, for the "
    "same sweeps; circles, each fold at WD = 0.7; numbers, balanced accuracy on all test traces and on "
    "those kept at WD = 0.7. g, h) Percentage of test traces kept per protein for the final models behind "
    "Figures 5D and 6C (single test split); numbers, kept at WD = 0.7. Test folds were balanced "
    "by subsampling the larger class; the test traces of a fold are distinct molecules, and folds are "
    "disjoint. Kept percentages at WD = 0.7 are mean ± SD over the four folds (SD with n in the "
    "denominator), interpolated between the stored threshold grid points that bracket 0.7. No statistical "
    "comparisons were performed."
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_provenance(out: Path, args: argparse.Namespace, inputs: list[Path]) -> None:
    import yaml
    here = Path(__file__).resolve()
    git = lambda *a: subprocess.run(["git", "-C", str(here.parent), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {
        "figure": FIG, "created": datetime.now().isoformat(timespec="seconds"), "script": str(here),
        "script_sha256": _sha256(here), "git_commit": git("rev-parse", "HEAD") or None,
        "git_dirty": bool(git("status", "--porcelain", "--", str(here))),
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "general_data": str(args.general_data), "step0_run": str(args.step0),
        "inputs": {str(p): _sha256(p) for p in inputs},
    }
    with open(out / f"{FIG}_provenance.yaml", "w") as f:
        yaml.safe_dump(prov, f, sort_keys=False, allow_unicode=True)


# --- Main ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="rebuild the CSVs from the published outputs")
    ap.add_argument("--general-data", type=Path, help="Zenodo GeneralData folder (with MLresults/)")
    ap.add_argument("--step0", type=Path, help="review step 0 run folder (funnel.csv, rois.csv)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    apply_style()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.build:
        if not (args.general_data and args.step0):
            ap.error("--build needs --general-data and --step0")
        inputs: list[Path] = []
        sweeps, at07 = build_cv(args.general_data, inputs)
        final_curves, final_table = build_final(args.general_data, inputs)
        cv_table = summarize_cv(at07)
        funnel = build_funnel(args.step0, cv_table, final_table, inputs)
        sweeps.to_csv(out / f"{FIG}_panels_a-c.csv", index=False)
        sweeps[sweeps["traces"] == "protein"].to_csv(out / f"{FIG}_panels_d-f.csv", index=False)
        final_curves.to_csv(out / f"{FIG}_panels_g-h.csv", index=False)
        at07.to_csv(out / f"{FIG}_wd07_by_fold.csv", index=False)
        cv_table.to_csv(out / f"{FIG}_table_cv.csv", index=False)
        final_table.to_csv(out / f"{FIG}_table_final.csv", index=False)
        funnel.to_csv(out / f"{FIG}_table_funnel.csv", index=False)
        (out / f"{FIG}_caption.txt").write_text(CAPTION + "\n")
        if Path(__file__).resolve() != (out / Path(__file__).name).resolve():
            shutil.copy(Path(__file__).resolve(), out / Path(__file__).name)
        write_provenance(out, args, inputs)

    pdf = plot(out)
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
