#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Figures R3.2a-e (first-round review, Reviewer 3, point 2): raw physical features of every filtered single-
molecule trace inside vesicles, before deep learning.

Features per molecule (R3.3a event table; the paper's ON classification): median detected photons per ON
event, mean ON time, mean OFF time, duty cycle (34.49 ms per frame, full 6000-frame traces).

R3.2a  Per protein pair (rows: Halo-D106 vs SNAP-C148, Halo-D106 vs Halo-K117, scGrx1 vs scGrx1-AcK20):
       a-c) PCA of the log10, standardized features: density contours per protein enclosing 50 % and 90 % of
            the molecules, feature directions (loadings x 2); adjusted Rand index of an unsupervised
            two-component Gaussian mixture against the proteins.
       d-f) Mean ON time against photons per event, same contours (log axes).
       g-i) AUC of a logistic regression on the four features (4-fold cross-validation, seed 840410; bars:
            all sessions pooled, mean +/- SD over folds; points: within single sessions) against the
            networks behind Figs. 4C, 5B, 6B (same cross-validation design).
       j-l) Rank-biserial effect size per feature (r > 0: the pair's first protein tends to larger values),
            one point per session in which both proteins were measured, bar: all sessions pooled.
R3.2b-d  Corner plots of the four features for each pair: per-protein distributions (diagonal) and
       density contours (50 % and 90 %) for every pair of features.
R3.2e  Mean ON time per molecule on all filtered traces against the traces shown in Figs. 5D and 6C (final
       models' test traces with Wasserstein distance >= 0.7, grouped by true identity), recomputed here on
       6000 frames at 34.49 ms.
Boxes: median and quartiles, whiskers 1.5 x IQR. Descriptive; no hypothesis tests.

Usage:
    python figure_R3.2.py
        Replot from the CSVs next to this script (works in the figure folder as distributed).
    python figure_R3.2.py --build --analysis <analysis run> --general-data <GeneralData> [--out <dir>]
        From the blinkognition repo: collect the CSVs from Revisions/PhysicalFeatures/features_analysis.py's
        run and the final models' MC-dropout outputs, replot, copy this script into --out.
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
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

# --- Constants ---
FIG = "R3.2"
PAIRS = [("HaloD106", "SNAPC148"), ("HaloD106", "HaloK117"), ("scGrx1", "scGrx1AcK20")]
LABELS = {"HaloD106": "Halo-D106", "HaloK117": "Halo-K117", "SNAPC148": "SNAP-C148",
          "scGrx1": "scGrx1", "scGrx1AcK20": "scGrx1-AcK20"}
PROTEIN_COLORS = {"HaloD106": "#0072B2", "HaloK117": "#E69F00", "SNAPC148": "#CC79A7",
                  "scGrx1": "#009E73", "scGrx1AcK20": "#D55E00"}
BLACK, WHITE = "#000000", "#FFFFFF"
FEATURES = ["photons_per_event", "mean_on_ms", "mean_off_ms", "duty_cycle"]
FEATURE_LABELS = {"photons_per_event": "Photons per event", "mean_on_ms": "Mean ON (ms)",
                  "mean_off_ms": "Mean OFF (ms)", "duty_cycle": "Duty cycle"}
FEATURE_SHORT = {"photons_per_event": "Photons", "mean_on_ms": "ON", "mean_off_ms": "OFF", "duty_cycle": "Duty"}
FINAL_MODELS = {"HaloD106_HaloK117": ("2026-05-19_08-21-58_orig_conv_gru_mirrored_IN_training_HaloD106_HaloK117",
                                      {"HTHTL": "HaloD106", "HTIA": "HaloK117"}, "5D"),
                "scGrx1_scGrx1AcK20": ("2026-05-19_10-14-25_tcn_mirrored_IN_training_scGrx1_scGrx1AcK20",
                                       {"Grx1": "scGrx1", "K20Ac": "scGrx1AcK20"}, "6C")}
WD_MIN = 0.7
HDR_LEVELS = (0.9, 0.5)              # contours enclosing 90 % and 50 % of the molecules
KDE_GRID = 90
LOADING_SCALE = 2.0
JITTER = 0.30
RNG_SEED = 42
MM = 1 / 25.4


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
    ax.tick_params(axis="both", which="minor", length=2, width=0.6, direction="out")


def pair_key(p: tuple[str, str]) -> str:
    return f"{p[0]}_{p[1]}"


# --- Build ---

def build(analysis: Path, general_data: Path, out: Path) -> list[Path]:
    files = {k: analysis / f"{k}.csv" for k in ("features", "pca_scores", "pca_summary", "gmm", "gmm_summary",
                                                "lr_folds", "lr_summary", "lr_coefficients", "effect_sizes",
                                                "dl_reference")}
    feats = pd.read_csv(files["features"])
    scores = pd.read_csv(files["pca_scores"])
    gmm = pd.read_csv(files["gmm"])
    rows = []
    for p in PAIRS:
        k = pair_key(p)
        d = feats[feats["construct"].isin(p)].merge(scores[scores["pair"] == k][["construct", "uniqueID", "pc1", "pc2"]],
                                                    on=["construct", "uniqueID"], validate="one_to_one")
        d = d.merge(gmm[gmm["pair"] == k][["construct", "uniqueID", "cluster"]], on=["construct", "uniqueID"],
                    validate="one_to_one")
        d["pair"] = k
        d["in_figure_set"] = False
        if k in FINAL_MODELS:
            model, keymap, _ = FINAL_MODELS[k]
            npz = general_data / "MLresults" / "FinalModels" / model / "MCD_results" / "traces_with_wasserstein.npz"
            files[f"npz_{k}"] = npz
            z = np.load(npz)
            names = [keymap[str(c)] for c in z["class_names"]]
            sel = pd.DataFrame({"construct": [names[int(l)] for l in z["labels"]], "uniqueID": z["unique_ids"],
                                "wd": z["wasserstein_distances"], "in_test": True})
            d = d.merge(sel, on=["construct", "uniqueID"], how="left")
            d["in_test"] = d["in_test"].fillna(False).astype(bool)
            d["in_figure_set"] = d["in_test"] & (d["wd"] >= WD_MIN)
        rows.append(d)
    mol = pd.concat(rows, ignore_index=True)
    mol.to_csv(out / f"{FIG}_molecules.csv", index=False, float_format="%.6g")
    for k in ("pca_summary", "gmm_summary", "lr_folds", "lr_summary", "lr_coefficients", "effect_sizes", "dl_reference"):
        pd.read_csv(files[k]).to_csv(out / f"{FIG}_{k}.csv", index=False, float_format="%.5f")
    sub = []
    for k, g in mol[mol["pair"].isin(FINAL_MODELS)].groupby("pair"):
        for c, h in g.groupby("construct"):
            for name, s in (("all filtered traces", h), ("final-model test traces", h[h["in_test"]]),
                            (f"Fig. {FINAL_MODELS[k][2]} set (WD >= {WD_MIN})", h[h["in_figure_set"]])):
                v = s["mean_on_ms"]
                sub.append({"pair": k, "construct": c, "set": name, "n_molecules": len(v), "mean_on_ms_median": v.median(),
                            "mean_on_ms_q25": v.quantile(0.25), "mean_on_ms_q75": v.quantile(0.75),
                            "mean_on_ms_mean": v.mean(), "mean_on_ms_sd": v.std(ddof=0)})
    pd.DataFrame(sub).to_csv(out / f"{FIG}_table_on_time_subsets.csv", index=False, float_format="%.3f")
    return list(files.values()) + [analysis / "manifest.yaml"]


# --- Plot helpers ---

def _letter(ax, letter: str, x: float = -0.32, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def _hdr_contours(ax, x: np.ndarray, y: np.ndarray, color: str, xlim, ylim, rng: np.random.Generator,
                  max_points: int = 6000) -> None:
    """Contours of a Gaussian KDE enclosing HDR_LEVELS of the points (x, y in plotting coordinates)."""
    if x.size > max_points:
        i = rng.choice(x.size, max_points, replace=False)
        x, y = x[i], y[i]
    kde = gaussian_kde(np.vstack([x, y]))
    gx, gy = np.meshgrid(np.linspace(*xlim, KDE_GRID), np.linspace(*ylim, KDE_GRID))
    dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    at_points = kde(np.vstack([x, y]))
    levels = sorted(np.quantile(at_points, [1 - q for q in HDR_LEVELS]))
    ax.contour(gx, gy, dens, levels=levels, colors=[color], linewidths=[0.8, 1.3])


def _log_limits(v: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    lo, hi = np.log10(np.percentile(v, [0.5, 99.5]))
    r = hi - lo
    return lo - pad * r, hi + pad * r


def _fmt_tick(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:g}k"                           # compact: 1k, 3k, 10k (narrow panels)
    return f"{v:g}"


def _set_log_ticks(ax, axis: str, lim: tuple[float, float], max_ticks: int = 5) -> None:
    """Ticks at 1-2-5 (else 1-3, else 1) multiples within lim (log10 units), labeled in data units."""
    for mults in ((1, 2, 5), (1, 3), (1,)):
        vals = [m * 10.0 ** e for e in range(int(np.floor(lim[0])) - 1, int(np.ceil(lim[1])) + 1) for m in mults]
        ticks = [np.log10(v) for v in vals if lim[0] <= np.log10(v) <= lim[1]]
        if len(ticks) <= max_ticks:
            break
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks, [_fmt_tick(10 ** t) for t in ticks])


# --- Plot ---

def plot_overview(out: Path, rng: np.random.Generator) -> Path:
    mol = pd.read_csv(out / f"{FIG}_molecules.csv")
    pca = pd.read_csv(out / f"{FIG}_pca_summary.csv")
    gmm = pd.read_csv(out / f"{FIG}_gmm_summary.csv").set_index("pair")
    lr = pd.read_csv(out / f"{FIG}_lr_summary.csv")
    dl = pd.read_csv(out / f"{FIG}_dl_reference.csv").set_index("pair")
    eff = pd.read_csv(out / f"{FIG}_effect_sizes.csv")

    fig = plt.figure(figsize=(180 * MM, 150 * MM))
    gs = fig.add_gridspec(3, 4, hspace=0.66, wspace=0.62, left=0.07, right=0.98, top=0.95, bottom=0.15)
    for i, p in enumerate(PAIRS):
        k = pair_key(p)
        d = mol[mol["pair"] == k]
        # PCA
        a = fig.add_subplot(gs[i, 0])
        xl, yl = np.percentile(d["pc1"], [0.5, 99.5]) + np.array([-0.5, 0.5]), np.percentile(d["pc2"], [0.5, 99.5]) + np.array([-0.5, 0.5])
        for c in p:
            g = d[d["construct"] == c]
            _hdr_contours(a, g["pc1"].to_numpy(), g["pc2"].to_numpy(), PROTEIN_COLORS[c], xl, yl, rng)
        ld = pca[pca["pair"] == k].set_index("component")
        vecs = {f: LOADING_SCALE * np.array([ld.loc["PC1", f"loading_{f}"], ld.loc["PC2", f"loading_{f}"]])
                for f in FEATURES}
        for v in vecs.values():
            a.annotate("", xy=tuple(v), xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 0.7, "color": BLACK})
        groups: list[list[str]] = []                     # features whose arrows point within 20 degrees share a label
        for f, v in vecs.items():
            ang = np.degrees(np.arctan2(v[1], v[0]))
            for g in groups:
                ref = vecs[g[0]]
                if abs((ang - np.degrees(np.arctan2(ref[1], ref[0])) + 180) % 360 - 180) < 20:
                    g.append(f)
                    break
            else:
                groups.append([f])
        for g in groups:
            v = np.mean([vecs[f] for f in g], axis=0)
            u = v / np.linalg.norm(v)
            pos = v + 0.45 * u
            a.text(pos[0], pos[1], "/".join(FEATURE_SHORT[f] for f in g), fontsize=5.5, color=BLACK,
                   ha="center", va="center")
        ev = ld["explained_variance_ratio"]
        a.set(xlim=xl, ylim=yl, xlabel=f"PC1 ({100 * ev['PC1']:.0f} %)", ylabel=f"PC2 ({100 * ev['PC2']:.0f} %)")
        a.text(0.02, 0.98, f"ARI {gmm.loc[k, 'adjusted_rand_index']:.2f}", transform=a.transAxes, fontsize=6,
               color=BLACK, va="top", ha="left")
        a.set_title(f"{LABELS[p[0]]} vs {LABELS[p[1]]}", loc="left", pad=3)
        axis_standards(a)
        _letter(a, "abc"[i])
        # ON vs photons
        b = fig.add_subplot(gs[i, 1])
        lx, ly = np.log10(d["photons_per_event"].to_numpy()), np.log10(d["mean_on_ms"].to_numpy())
        xl2, yl2 = _log_limits(d["photons_per_event"].to_numpy()), _log_limits(d["mean_on_ms"].to_numpy())
        for c in p:
            m = (d["construct"] == c).to_numpy()
            _hdr_contours(b, lx[m], ly[m], PROTEIN_COLORS[c], xl2, yl2, rng)
        b.set(xlim=xl2, ylim=yl2, xlabel="Photons per event", ylabel="Mean ON (ms)")
        _set_log_ticks(b, "x", xl2)
        _set_log_ticks(b, "y", yl2)
        axis_standards(b)
        _letter(b, "def"[i])
        # AUC
        c_ax = fig.add_subplot(gs[i, 2])
        pooled = lr[(lr["pair"] == k) & (lr["scope"] == "pooled")].iloc[0]
        ses = lr[(lr["pair"] == k) & (lr["scope"] == "session")]
        c_ax.bar(0, pooled["auc_mean"], yerr=pooled["auc_sd"], width=0.6, color=WHITE, edgecolor=BLACK, lw=0.8,
                 error_kw={"lw": 0.8, "capsize": 2})
        c_ax.plot(1 + rng.uniform(-0.12, 0.12, len(ses)), ses["auc_mean"], ls="none", marker="o", ms=3.2,
                  mfc=WHITE, mec=BLACK, mew=0.8)
        c_ax.bar(2, dl.loc[k, "auc_mean"], yerr=dl.loc[k, "auc_sd"], width=0.6, color="#BBBBBB", edgecolor=BLACK,
                 lw=0.8, error_kw={"lw": 0.8, "capsize": 2})
        c_ax.axhline(0.5, color=BLACK, ls=":", lw=0.8)
        c_ax.set_xticks([0, 1, 2], ["Features, pooled", "Features, per day", f"Network (Fig. {dl.loc[k, 'figure']})"],
                        rotation=35, ha="right")
        c_ax.set(ylim=(0.4, 1.0), ylabel="AUC", xlim=(-0.6, 2.6))
        axis_standards(c_ax)
        _letter(c_ax, "ghi"[i])
        # effect sizes
        e_ax = fig.add_subplot(gs[i, 3])
        e = eff[eff["pair"] == k]
        for j, f in enumerate(FEATURES):
            pooled_r = e[(e["scope"] == "pooled") & (e["feature"] == f)]["rank_biserial_r"].iloc[0]
            e_ax.bar(j, pooled_r, width=0.6, color=WHITE, edgecolor=BLACK, lw=0.8)
            s = e[(e["scope"] == "session") & (e["feature"] == f)]["rank_biserial_r"].to_numpy()
            e_ax.plot(j + rng.uniform(-0.15, 0.15, s.size), s, ls="none", marker="o", ms=2.6, mfc=BLACK, mec=BLACK)
        e_ax.axhline(0, color=BLACK, lw=0.6)
        e_ax.set_xticks(range(len(FEATURES)), [FEATURE_SHORT[f] for f in FEATURES])
        e_ax.set(ylim=(-1, 1), ylabel=f"Effect size r\n(+: {LABELS[p[0]]} larger)")
        axis_standards(e_ax)
        _letter(e_ax, "jkl"[i])

    handles = [Line2D([], [], color=PROTEIN_COLORS[c], lw=1.2, label=LABELS[c])
               for c in ("HaloD106", "SNAPC148", "HaloK117", "scGrx1", "scGrx1AcK20")]
    handles += [Patch(facecolor=WHITE, edgecolor=BLACK, lw=0.8, label="Features, all sessions (± SD over folds)"),
                Line2D([], [], ls="none", marker="o", ms=3.2, mfc=WHITE, mec=BLACK, mew=0.8, label="Features, one session"),
                Patch(facecolor="#BBBBBB", edgecolor=BLACK, lw=0.8, label="Network (± SD over folds)"),
                Line2D([], [], ls="none", marker="o", ms=2.6, mfc=BLACK, mec=BLACK, label="Effect size, one session")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False,
               handlelength=1.6, fontsize=5.5, columnspacing=1.0)
    pdf = out / f"{FIG}a.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


def plot_corner(out: Path, p: tuple[str, str], suffix: str, rng: np.random.Generator) -> Path:
    mol = pd.read_csv(out / f"{FIG}_molecules.csv")
    d = mol[mol["pair"] == pair_key(p)]
    logs = {f: np.log10(d[f].to_numpy()) for f in FEATURES}
    lims = {f: _log_limits(d[f].to_numpy()) for f in FEATURES}
    n = len(FEATURES)
    fig, axes = plt.subplots(n, n, figsize=(120 * MM, 120 * MM))
    fig.subplots_adjust(hspace=0.15, wspace=0.15, left=0.13, right=0.98, top=0.93, bottom=0.12)
    for i, fy in enumerate(FEATURES):
        for j, fx in enumerate(FEATURES):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                edges = np.linspace(*lims[fx], 40)
                for c in p:
                    m = (d["construct"] == c).to_numpy()
                    ax.hist(logs[fx][m], bins=edges, density=True, histtype="step", color=PROTEIN_COLORS[c], lw=1.0)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
            else:
                for c in p:
                    m = (d["construct"] == c).to_numpy()
                    _hdr_contours(ax, logs[fx][m], logs[fy][m], PROTEIN_COLORS[c], lims[fx], lims[fy], rng, 4000)
                ax.set_ylim(lims[fy])
                if j == 0:
                    _set_log_ticks(ax, "y", lims[fy])
                    ax.set_ylabel(FEATURE_LABELS[fy])
                else:
                    ax.set_yticklabels([])
            ax.set_xlim(lims[fx])
            axis_standards(ax)
            if i == j:
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", left=False)
            if i == n - 1:
                _set_log_ticks(ax, "x", lims[fx])
                ax.set_xlabel(FEATURE_LABELS[fx])
            else:
                ax.set_xticklabels([])
    n_mol = d["construct"].value_counts()
    fig.legend(handles=[Line2D([], [], color=PROTEIN_COLORS[c], lw=1.2, label=f"{LABELS[c]} (n = {n_mol[c]})") for c in p]
               + [Line2D([], [], color=BLACK, lw=1.3, label="50 % of molecules"),
                  Line2D([], [], color=BLACK, lw=0.8, label="90 % of molecules")],
               loc="upper right", bbox_to_anchor=(0.98, 0.93), frameon=False, fontsize=6)
    pdf = out / f"{FIG}{suffix}.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


def plot_subsets(out: Path, rng: np.random.Generator) -> Path:
    mol = pd.read_csv(out / f"{FIG}_molecules.csv")
    fig, axes = plt.subplots(1, 2, figsize=(120 * MM, 60 * MM))
    fig.subplots_adjust(wspace=0.45, left=0.11, right=0.98, top=0.88, bottom=0.30)
    for ax, (k, (_, _, fig_id)), letter in zip(axes, FINAL_MODELS.items(), "ab"):
        d = mol[mol["pair"] == k]
        p = next(pp for pp in PAIRS if pair_key(pp) == k)
        groups, colors, ticks = [], [], []
        for set_name, mask in (("All traces", np.ones(len(d), bool)), (f"Fig. {fig_id} set", d["in_figure_set"].to_numpy(bool))):
            for c in p:
                v = d.loc[mask & (d["construct"] == c).to_numpy(), "mean_on_ms"].to_numpy()
                groups.append(v)
                colors.append(PROTEIN_COLORS[c])
                ticks.append(f"{LABELS[c]}\n{set_name} (n = {v.size})")
        for i, (v, col) in enumerate(zip(groups, colors)):
            ax.scatter(i + rng.uniform(-JITTER, JITTER, v.size), v, s=0.8 if v.size > 500 else 2.5, color=col,
                       alpha=0.4, lw=0, rasterized=True, zorder=1)
        ax.boxplot(groups, positions=range(len(groups)), widths=0.55, whis=1.5, showfliers=False, zorder=2,
                   medianprops={"color": BLACK, "lw": 1.0}, boxprops={"facecolor": "none", "lw": 0.8},
                   whiskerprops={"lw": 0.8}, capprops={"lw": 0.8}, patch_artist=True)
        ax.set(yscale="log", ylabel="Mean ON (ms)", xlim=(-0.6, len(groups) - 0.4))
        ax.set_xticks(range(len(groups)), ticks, rotation=40, ha="right", fontsize=5.5)
        ax.set_title(f"{LABELS[p[0]]} vs {LABELS[p[1]]}", loc="left", pad=3)
        axis_standards(ax)
        _letter(ax, letter, -0.28, 1.04)
    pdf = out / f"{FIG}e.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


# --- Captions and provenance ---

def captions(out: Path) -> str:
    mol = pd.read_csv(out / f"{FIG}_molecules.csv")
    n = {k: ", ".join(f"{LABELS[c]} {v}" for c, v in g["construct"].value_counts().items())
         for k, g in mol.groupby("pair")}
    nsets = mol[mol["in_figure_set"]].groupby(["pair", "construct"]).size()
    common = ("Features per molecule from every filtered single-molecule trace inside vesicles: median detected "
              "photons per ON event, mean ON time, mean OFF time and duty cycle (ON frames as classified for "
              "Figures 5D and 6C; 6000 frames, 34.49 ms per frame). Detected photons: background-removed 5 x 5 pixel sum x 0.062 photons per ADU, measured from the camera noise (mean-variance analysis of the raw movies), no quantum-efficiency correction.")
    return "\n\n".join([
        "Figure R3.2a. Physical features of single molecules before deep learning. " + common +
        " Rows, Halo-D106 vs SNAP-C148, Halo-D106 vs Halo-K117 and scGrx1 vs scGrx1-AcK20 (n: " +
        "; ".join(n[pair_key(p)] for p in PAIRS) + " molecules). a-c) First two principal components of the "
        "log-transformed, standardized features; contours enclose 50 % (thick) and 90 % (thin) of each "
        "protein's molecules; arrows, feature directions; ARI, adjusted Rand index between an unsupervised "
        "two-component Gaussian mixture and the protein identity (0, no relation; 1, identical). d-f) Mean ON "
        "time against photons per event, same contours. g-i) Area under the ROC curve of a logistic regression "
        "on the four features, four-fold stratified cross-validation with class-balanced test folds (bars, all "
        "sessions, mean ± SD over folds; points, single sessions in which both proteins were measured with at "
        "least 20 molecules each) and of the networks of Figures 4C, 5B and 6B in the same design (gray, mean "
        "± SD over folds); dotted line, chance. j-l) Rank-biserial effect size per feature (positive: the first "
        "protein of the row tends to larger values); bars, all sessions; points, single sessions. Molecules are "
        "distinct. No statistical tests were performed.",
        *[f"Figure R3.2{s}. Corner plot of the four features, {LABELS[p[0]]} vs {LABELS[p[1]]}. " + common +
          " Diagonal, per-protein distributions (log scale); below the diagonal, contours enclosing 50 % (thick) "
          f"and 90 % (thin) of each protein's molecules (n: {n[pair_key(p)]}). No statistical tests were performed."
          for s, p in zip("bcd", PAIRS)],
        "Figure R3.2e. Mean ON time per molecule for all filtered traces and for the traces shown in Figures 5D "
        "and 6C (test traces of the final models with Wasserstein distance >= 0.7, grouped by true identity; n: " +
        ", ".join(f"{LABELS[c]} {v}" for (_, c), v in nsets.items()) + "), recomputed here on 6000 frames at "
        "34.49 ms per frame (the published panels used the first 4000 frames and 30 ms). Points, molecules; "
        "boxes, median and quartiles; whiskers, 1.5 x IQR. No statistical tests were performed.",
    ])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_provenance(out: Path, inputs: list[Path]) -> None:
    import yaml
    here = Path(__file__).resolve()
    git = lambda *a: subprocess.run(["git", "-C", str(here.parent), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {
        "figure": FIG, "created": datetime.now().isoformat(timespec="seconds"), "script": str(here),
        "script_sha256": _sha256(here), "git_commit": git("rev-parse", "HEAD") or None,
        "git_dirty": bool(git("status", "--porcelain", "--", str(here.parent))),
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "inputs": {str(p): _sha256(p) for p in inputs},
    }
    with open(out / f"{FIG}_provenance.yaml", "w") as f:
        yaml.safe_dump(prov, f, sort_keys=False, allow_unicode=True)


# --- Main ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--analysis", type=Path, help="features_analysis.py run folder")
    ap.add_argument("--general-data", type=Path, help="Zenodo GeneralData folder (final models)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.build:
        if not (args.analysis and args.general_data):
            ap.error("--build needs --analysis and --general-data")
        inputs = build(args.analysis, args.general_data, out)
        (out / f"{FIG}_captions.txt").write_text(captions(out) + "\n")
        if Path(__file__).resolve() != (out / Path(__file__).name).resolve():
            shutil.copy(Path(__file__).resolve(), out / Path(__file__).name)
        write_provenance(out, inputs)
    apply_style()
    rng = np.random.default_rng(RNG_SEED)
    pdfs = [plot_overview(out, rng)] + [plot_corner(out, p, s, rng) for p, s in zip(PAIRS, "bcd")] + [plot_subsets(out, rng)]
    for p in pdfs:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
