#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Table R3.4a (first-round review, Reviewer 3, point 4, first half): the percentage of the data retained by the
quality filter and by the uncertainty (Wasserstein distance, WD) filter, with the accuracy before and after.

Trimmed to the request (user, 2026-11-20): one table instead of the first version's figure (threshold sweeps and
trade-off curves; revisions_figures/not_in_revision/R3.4a_full_20261120 and git history, commit
e4dc7fb). Rows: per comparison of the paper (Figs. 4C, 5B, 6B) and protein.
  - Single molecules inside vesicles and those passing the quality filter: the published extraction (step 0).
  - Kept by the uncertainty filter at the paper's threshold (WD >= 0.7): four-fold cross-validation behind Figs.
    4C, 5B, 6B (mean +/- SD over folds, pooled over both proteins of the balanced test folds; the stored 50-point
    threshold grid interpolated at 0.7), and per protein for the final models behind Figs. 5D and 6C.
  - Approximate overall yield = quality-filter pass rate x cross-validation kept fraction.
  - Balanced accuracy in cross-validation on all test traces and on those kept (mean +/- SD over folds).
Every number is read from the paper's published outputs; nothing is retrained. No statistical tests.

Usage:
    python figure_R3.4a.py
        Re-render the table from the CSVs next to this script (works in the figure folder as distributed).
    python figure_R3.4a.py --build --general-data <GeneralData> --step0 <PaperData check run> [--out <dir>]
        Rebuild the CSVs from the published outputs, render, and copy this script into --out.

Outputs: R3.4a.pdf (the table, rendered), R3.4a_table.csv (the table), R3.4a_table_cv.csv, R3.4a_table_final.csv,
R3.4a_table_funnel.csv, R3.4a_wd07_by_fold.csv (sources), R3.4a_caption.txt, R3.4a_provenance.yaml,
figure_R3.4a.py.
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


# --- Table ---

COLUMNS = ["Comparison", "Protein", "Single molecules in vesicles", "Passing quality filter",
           "Kept at WD >= 0.7, cross-validation", "Kept at WD >= 0.7, final model", "Approximate overall yield",
           "Balanced accuracy, all", "Balanced accuracy, kept"]


def compose(out: Path) -> pd.DataFrame:
    cv = pd.read_csv(out / f"{FIG}_table_cv.csv")
    cv = cv[cv["traces"] == "protein"].set_index("pair")
    fun = pd.read_csv(out / f"{FIG}_table_funnel.csv")
    rows = []
    for r in fun.itertuples():
        c = cv.loc[r.pair]
        rows.append({
            "Comparison": f"Fig. {r.figure} ({c['model']})", "Protein": LABELS[r.construct],
            "Single molecules in vesicles": f"{int(r.rois_single_in)}",
            "Passing quality filter": f"{int(r.traces_quality_filtered_in)} ({r.quality_filter_pass_pct:.0f} %)",
            "Kept at WD >= 0.7, cross-validation": f"{c['kept_pct_wd07_mean']:.0f} ± {c['kept_pct_wd07_sd']:.0f} %",
            "Kept at WD >= 0.7, final model": ("-" if pd.isna(r.final_model_kept_pct_wd07)
                                                else f"{r.final_model_kept_pct_wd07:.0f} %"),
            "Approximate overall yield": f"{r.approx_yield_pct:.0f} %",
            "Balanced accuracy, all": f"{c['bal_acc_all_pct_mean']:.1f} ± {c['bal_acc_all_pct_sd']:.1f} %",
            "Balanced accuracy, kept": f"{c['bal_acc_kept_wd07_pct_mean']:.1f} ± {c['bal_acc_kept_wd07_pct_sd']:.1f} %"})
    return pd.DataFrame(rows, columns=COLUMNS)


def plot(out: Path) -> Path:
    t = pd.read_csv(out / f"{FIG}_table.csv", dtype=str).fillna("-")
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 0.35 + 0.2 * (len(t) + 2)))
    ax.axis("off")
    headers = ["Comparison", "Protein", "Single molecules\nin vesicles", "Passing\nquality filter",
               "Kept at\nWD ≥ 0.7\n(cross-validation)", "Kept at\nWD ≥ 0.7\n(final model)", "Approximate\noverall yield",
               "Balanced\naccuracy\n(all)", "Balanced\naccuracy\n(kept)"]
    widths = [0.13, 0.10, 0.10, 0.10, 0.12, 0.11, 0.10, 0.12, 0.12]
    tab = ax.table(cellText=t.values, colLabels=headers, colWidths=widths, loc="center", cellLoc="center",
                   colLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(6)
    tab.scale(1.0, 1.35)
    for (i, j), cell in tab.get_celld().items():
        cell.set_linewidth(0.0 if i > 0 else 0.6)
        cell.visible_edges = "B" if i == 0 else ""
        if i == 0:
            cell.set_height(cell.get_height() * 2.8)
            cell.set_text_props(fontweight="bold")
    path = out / f"{FIG}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# --- Caption and provenance ---

CAPTION = (
    "Table R3.4a. Data retained by the quality filter and by the uncertainty filter. Single molecules inside "
    "vesicles: single regions of interest of the published extraction within 4 pixels of a vesicle; passing "
    "quality filter: traces used for classification. Kept at WD >= 0.7: traces whose Monte Carlo dropout "
    "prediction has a Wasserstein distance of at least 0.7 (the threshold of the paper); cross-validation, four "
    "folds behind Figures 4C, 5B and 6B, mean ± SD over folds, pooled over the two proteins of the balanced test "
    "folds; final model, the models behind Figures 5D and 6C, per protein (single test split). Approximate overall "
    "yield: quality-filter pass rate x cross-validation kept fraction. Balanced accuracy in cross-validation on all "
    "test traces and on the kept ones, mean ± SD over folds. All numbers from the published outputs; nothing was "
    "retrained."
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
        at07.to_csv(out / f"{FIG}_wd07_by_fold.csv", index=False)
        cv_table.to_csv(out / f"{FIG}_table_cv.csv", index=False)
        final_table.to_csv(out / f"{FIG}_table_final.csv", index=False)
        funnel.to_csv(out / f"{FIG}_table_funnel.csv", index=False)
        compose(out).to_csv(out / f"{FIG}_table.csv", index=False)
        (out / f"{FIG}_caption.txt").write_text(CAPTION + "\n")
        if Path(__file__).resolve() != (out / Path(__file__).name).resolve():
            shutil.copy(Path(__file__).resolve(), out / Path(__file__).name)
        write_provenance(out, args, inputs)

    pdf = plot(out)
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
