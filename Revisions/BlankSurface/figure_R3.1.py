#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Figure R3.1 (first-round review, Reviewer 3, point 1): blank coverslips and empty vesicles through the paper's
extraction and quality filter, against protein slides imaged at the same 640 nm power and against the paper.

Slides (Revisions/BlankSurface/config.yaml): coated coverslips without vesicles (DFK786, DFK787), empty vesicles
(ATTO520, the paper's vesicle dye; ATTO390 twice; ATTO425), and protein in vesicles at the same power (DFK788:
SNAP-tag in ATTO520 vesicles, HaloTag7 in ATTO390 vesicles). 6000 frames at 640 nm, 34.49 ms per frame, 13-15 mW
(the paper: 20-21 mW). Extraction with fixed protein gradients 10000 (main; the paper's most permissive),
20000 and 40000.

R3.1a.pdf
  a-h) One field per slide, 640 nm maximum projection over 6000 frames, one common gray scale (ADU; limits:
       lowest 1st and highest 99.9th percentile of the eight fields shown). The field shown is the slide's field
       with the median number of detected emitters at gradient 10000 (not chosen by eye). Circles: single
       emitters (ROIs) the extraction kept; squares: traces that passed the quality filter.
  i)   Detected emitters per field (linked localization groups = extraction ROIs), gradient 10000.
  j)   Vesicles per field (vesicle-channel clusters, as the extraction finds them), slides and the paper's movies.
  k)   Traces passing the quality filter per field, inside and outside vesicles, at the three gradients (points:
       fields at 10000; open markers: means at 10000, 20000, 40000).
  l)   Traces inside vesicles passing the quality filter per field: slides (as k) and the paper's movies per
       construct (points: movies; line: mean).
R3.1b.pdf
  a)   Every trace from a blank or empty-vesicle slide that passed the quality filter at gradient 10000, and one
       random passing trace inside vesicles from each same-power protein slide (seed 42); detected photons per
       frame (0.062 photons per ADU), one common scale.
  b)   Median photons per ON event against number of ON events per trace (the paper's ON classification without
       the separation minimum, as R3.3), every passing trace of the blank and empty-vesicle slides at any gradient
       (an emitter found at two gradients counts once) and of the same-power protein slides.
  c)   The paper's final models on those traces: share kept at the paper's Wasserstein threshold (0.7), and the
       class called among the kept.
Descriptive; no statistical tests.

Usage:
    python figure_R3.1.py
        Replot from the CSVs next to this script (works in the figure folder as distributed).
    python figure_R3.1.py --build [--out <dir>] [--count ... --projections ... --classify ... --filtered ...
                                   --edge-groups ... --step0 ...]
        From the blinkognition repo: build the CSVs from the analysis runs, replot, copy this script into --out.
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
from matplotlib.patches import Rectangle  # noqa: E402

# --- Constants ---
FIG = "R3.1"
SLIDES = [("DFK786", "Blank", "Blank coverslip 1"), ("DFK787", "Blank", "Blank coverslip 2"),
          ("DFK786", "EmptyATTO520", "Empty ATTO520 vesicles"), ("DFK786", "EmptyATTO390", "Empty ATTO390 vesicles 1"),
          ("DFK787", "EmptyATTO390", "Empty ATTO390 vesicles 2"), ("DFK787", "EmptyATTO425", "Empty ATTO425 vesicles"),
          ("DFK788", "SNAP520", "SNAP-tag, same power"), ("DFK788", "HT390", "HaloTag7, same power")]
SHORT = {"Blank coverslip 1": "Blank 1", "Blank coverslip 2": "Blank 2", "Empty ATTO520 vesicles": "Empty 520",
         "Empty ATTO390 vesicles 1": "Empty 390 (1)", "Empty ATTO390 vesicles 2": "Empty 390 (2)",
         "Empty ATTO425 vesicles": "Empty 425", "SNAP-tag, same power": "SNAP-tag", "HaloTag7, same power": "HaloTag7"}
ROLE_STYLE = {"blank": {"color": "#000000", "fill": False}, "empty_vesicles": {"color": "#56B4E9", "fill": True},
              "protein": {"color": "#000000", "fill": True}}
CONSTRUCTS = ["HaloD106", "SNAPC148", "HaloK117", "scGrx1", "scGrx1AcK20"]
LABELS = {"HaloD106": "Halo-D106", "HaloK117": "Halo-K117", "SNAPC148": "SNAP-C148",
          "scGrx1": "scGrx1", "scGrx1AcK20": "scGrx1-AcK20"}
PROTEIN_COLORS = {"HaloD106": "#0072B2", "HaloK117": "#E69F00", "SNAPC148": "#CC79A7",
                  "scGrx1": "#009E73", "scGrx1AcK20": "#D55E00"}
MODEL_LABELS = {"HaloD106_HaloK117": "Halo-D106 vs Halo-K117 (CNN-GRU)",
                "scGrx1_scGrx1AcK20": "scGrx1 vs scGrx1-AcK20 (TCN)"}
GRADIENTS = [10000, 20000, 40000]
GRAD_MARKERS = {10000: "o", 20000: "s", 40000: "^"}
MAIN_GRADIENT = 10000
BOX = 5
PIXEL_NM = 130
SCALE_BAR_UM = 5
FRAME_S = 0.03449
PAPERDATA_CONFIG = "Revisions/PaperData/config.yaml"   # camera.photons_per_adu (build only)
JITTER = 0.30
RNG_SEED = 42
FIG_WIDTH_IN = 180 / 25.4

R = "Results/Revisions"
DEFAULTS = {"count": f"{R}/BlankSurface/s3it/count/run_003", "projections": f"{R}/BlankSurface/s3it/projections/run_001",
            "classify": f"{R}/BlankSurface/classify/run_001", "filtered": f"{R}/BlankSurface/s3it/filtered",
            "edge_groups": f"{R}/PaperData/s3it/edge_groups/run_001", "step0": f"{R}/PaperData/check/run_003"}


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


def _letter(ax, letter: str, x: float = -0.30, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


# --- Build ---

def _slide_label(session: str, slide: str) -> str:
    return {(s, n): lab for s, n, lab in SLIDES}[(session, slide)]


def build(repo: Path, a: argparse.Namespace, out: Path) -> list[Path]:
    import yaml
    p = {k: repo / getattr(a, k) for k in DEFAULTS}
    with open(repo / PAPERDATA_CONFIG) as fh:
        pcfg = yaml.safe_load(fh)
    ph_per_adu = float(pcfg["camera"]["photons_per_adu"])
    inputs = []

    # fields (all gradients)
    f = pd.read_csv(p["count"] / "fov.csv")
    inputs.append(p["count"] / "fov.csv")
    f["label"] = [_slide_label(s, n) for s, n in zip(f["session"], f["slide"])]
    cols = ["gradient", "session", "slide", "label", "role", "fov", "n_locs", "n_linked", "n_linked_all",
            "n_vesicles", "in_area_fraction", "n_single", "n_single_in", "n_single_out", "n_pass_in", "n_pass_out"]
    f[cols].to_csv(out / f"{FIG}a_fields.csv", index=False, float_format="%.5g")

    # fields shown: median emitters at the main gradient
    m = f[f["gradient"] == MAIN_GRADIENT]
    shown = []
    for s, n, lab in SLIDES:
        g = m[(m["session"] == s) & (m["slide"] == n)].sort_values(["n_linked", "fov"]).reset_index(drop=True)
        shown.append(g.iloc[(len(g) - 1) // 2][["session", "slide", "role", "fov", "n_linked", "n_vesicles"]]
                     .to_dict() | {"label": lab})
    shown = pd.DataFrame(shown)
    z = np.load(p["projections"] / "images.npz")
    inputs.append(p["projections"] / "images.npz")
    imgs = [z[f"{r.session}|{r.slide}|{int(r.fov):04d}|max640"] for r in shown.itertuples()]
    shown["vmin"] = min(float(np.percentile(im, 1)) for im in imgs)
    shown["vmax"] = max(float(np.percentile(im, 99.9)) for im in imgs)
    for i, im in enumerate(imgs):
        np.savetxt(out / f"{FIG}a_image_{i + 1}.csv", im, delimiter=",", fmt="%.1f")
    shown.to_csv(out / f"{FIG}a_fields_shown.csv", index=False, float_format="%.6g")

    # traces: ROI marks for the shown fields, and every passing trace
    tr = pd.read_csv(p["count"] / "traces.csv")
    inputs.append(p["count"] / "traces.csv")
    tr["label"] = [_slide_label(s, n) for s, n in zip(tr["session"], tr["slide"])]
    marks = []
    for i, r in enumerate(shown.itertuples()):
        g = tr[(tr["gradient"] == MAIN_GRADIENT) & (tr["session"] == r.session) & (tr["slide"] == r.slide)
               & (tr["fov"] == r.fov)]
        for t in g.itertuples():
            marks.append({"panel": i + 1, "x": t.x, "y": t.y, "where": t.where, "passed": bool(t.passed_pipeline)})
    pd.DataFrame(marks, columns=["panel", "x", "y", "where", "passed"]).to_csv(out / f"{FIG}a_rois_shown.csv", index=False)

    # the paper: vesicles and passing IN traces per movie
    em = pd.read_csv(p["edge_groups"] / "movies.csv")
    inputs.append(p["edge_groups"] / "movies.csv")
    cam = pd.read_csv(p["step0"] / "camera.csv")
    cam["movie"] = [Path(x).stem for x in cam["file"]]
    key2name = {v: k for k, v in pcfg["constructs"].items()}
    cam["construct"] = cam["folder"].map(key2name)
    rows = []
    for c in CONSTRUCTS:
        idx = pd.read_csv(p["step0"] / f"index_IN_{c}.csv")
        inputs.append(p["step0"] / f"index_IN_{c}.csv")
        npass = idx[idx["passed_filter"]].groupby(["session", "movie"]).size()
        for r in cam[cam["construct"] == c].itertuples():
            rows.append({"construct": c, "session": r.session, "movie": r.movie,
                         "n_pass_in": int(npass.get((r.session, r.movie), 0))})
    paper = pd.DataFrame(rows).merge(em[["session", "movie", "n_vesicles"]], on=["session", "movie"], how="left")
    paper.to_csv(out / f"{FIG}a_paper_movies.csv", index=False)

    # R3.1b: passing traces of the blank / empty slides (unique emitters) and the protein slides
    ps = tr[tr["passed_pipeline"]].copy()
    ps["impurity"] = ps["slide"].isin(["Blank", "EmptyATTO520", "EmptyATTO390", "EmptyATTO425"])
    ps = ps.sort_values("gradient")
    ps["emitter"] = ps["session"] + "|" + ps["slide"] + "|" + ps["fov"].astype(str) + "|" + ps["x"].astype(str) + "|" + ps["y"].astype(str)
    ps["first_gradient"] = ps.groupby("emitter")["gradient"].transform("min")
    uniq = ps.drop_duplicates("emitter")
    keep = ["emitter", "gradient", "gt", "session", "slide", "label", "where", "uniqueID", "fov", "x", "y", "impurity",
            "separation", "n_events_free", "frac_events_1frame_free", "median_photons_per_event_free"]
    uniq[keep].to_csv(out / f"{FIG}b_passing_traces.csv", index=False, float_format="%.5g")

    # example traces (gradient 10000): every impurity trace, one random protein trace per protein slide
    ex = ps[(ps["gradient"] == MAIN_GRADIENT)]
    rng = np.random.default_rng(RNG_SEED)
    sel = [r for r in ex[ex["impurity"]].sort_values(["slide", "session", "where", "uniqueID"]).itertuples()]
    for s, n, _ in SLIDES[6:]:
        g = ex[(ex["session"] == s) & (ex["slide"] == n) & (ex["where"] == "IN")].sort_values("uniqueID")
        sel.append(list(g.itertuples())[int(rng.integers(len(g)))])
    series, meta = {}, []
    for k, r in enumerate(sel):
        fp = p["filtered"] / f"gt{r.gt}_g{r.gradient}" / f"ProteinTraces{r.where}" / "Filtered" / f"{r.slide}_{r.where}_filtered_bg_rm.pkl"
        inputs.append(fp)
        bg = pd.read_pickle(fp)[r.uniqueID].to_numpy(float)
        series[f"trace_{k + 1}_photons"] = bg * ph_per_adu
        meta.append({"trace": k + 1, "session": r.session, "slide": r.slide, "label": r.label, "where": r.where,
                     "gradient": r.gradient, "uniqueID": r.uniqueID, "fov": r.fov, "x": r.x, "y": r.y,
                     "impurity": r.impurity})
    T = len(next(iter(series.values())))
    pd.DataFrame({"time_s": np.arange(T) * FRAME_S, **series}).to_csv(out / f"{FIG}b_traces.csv", index=False,
                                                                       float_format="%.2f")
    pd.DataFrame(meta).to_csv(out / f"{FIG}b_traces_selection.csv", index=False)

    # classification of the unique passing emitters
    if not (p["classify"] / "predictions.csv").exists():
        print(f"no {p['classify'] / 'predictions.csv'}: panel R3.1b c left out")
        return inputs
    pr = pd.read_csv(p["classify"] / "predictions.csv")
    inputs.append(p["classify"] / "predictions.csv")
    pr["emitter"] = pr["session"] + "|" + pr["slide"] + "|" + pr["fov"].astype(str) + "|" + pr["x"].astype(str) + "|" + pr["y"].astype(str)
    pr = pr.sort_values("gradient").drop_duplicates(["model", "emitter"])
    pr["impurity"] = pr["slide"].isin(["Blank", "EmptyATTO520", "EmptyATTO390", "EmptyATTO425"])
    pr[["model", "emitter", "gradient", "session", "slide", "where", "uniqueID", "impurity", "prediction",
        "wasserstein", "kept"]].to_csv(out / f"{FIG}b_classification.csv", index=False, float_format="%.5g")
    return inputs


# --- Plot ---

def _strip(ax, x0: float, vals: np.ndarray, color: str, fill: bool, rng, size: float = 6) -> None:
    xs = x0 + rng.uniform(-JITTER, JITTER, len(vals))
    ax.scatter(xs, vals, s=size, facecolors=color if fill else "none", edgecolors=color, linewidths=0.5, zorder=2)


def plot_a(out: Path) -> Path:
    apply_style()
    shown = pd.read_csv(out / f"{FIG}a_fields_shown.csv")
    marks = pd.read_csv(out / f"{FIG}a_rois_shown.csv")
    fields = pd.read_csv(out / f"{FIG}a_fields.csv")
    paper = pd.read_csv(out / f"{FIG}a_paper_movies.csv")
    vmin, vmax = float(shown["vmin"].iloc[0]), float(shown["vmax"].iloc[0])
    fig = plt.figure(figsize=(FIG_WIDTH_IN, 230 / 25.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 2.0], hspace=0.16)
    top = outer[0].subgridspec(2, 4, hspace=0.18, wspace=0.10)
    bottom = outer[1].subgridspec(2, 2, hspace=0.85, wspace=0.28)
    bottom = [bottom[0, 0], bottom[0, 1], bottom[1, 0], bottom[1, 1]]
    im_h = None
    for i, r in enumerate(shown.itertuples()):
        ax = fig.add_subplot(top[i // 4, i % 4])
        img = np.loadtxt(out / f"{FIG}a_image_{i + 1}.csv", delimiter=",")
        im_h = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        for t in marks[marks["panel"] == i + 1].itertuples():
            cx, cy = t.x + (BOX - 1) / 2, t.y + (BOX - 1) / 2
            if t.passed:
                ax.add_patch(Rectangle((t.x - 0.5 - 2, t.y - 0.5 - 2), BOX + 4, BOX + 4, fill=False, lw=0.8,
                                       ec="#F0E442"))
            else:
                ax.add_patch(plt.Circle((cx, cy), 4, fill=False, lw=0.5, ec="#56B4E9"))
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        ne = int(r.n_linked)
        ax.set_title(f"{r.label}\n{r.session}, field {int(r.fov)}: {ne} emitter{'' if ne == 1 else 's'}", fontsize=6,
                     pad=2)
        _letter(ax, "abcdefgh"[i], x=-0.08, y=1.10)
        if i == 0:
            L = SCALE_BAR_UM * 1000 / PIXEL_NM
            ax.plot([230 - 12 - L, 230 - 12], [230 - 12, 230 - 12], color="white", lw=1.5)
            ax.text(230 - 12 - L / 2, 230 - 18, f"{SCALE_BAR_UM} µm", color="white", ha="center", va="bottom", fontsize=6)
        if i == 7:
            cax = ax.inset_axes([1.04, 0.0, 0.05, 1.0])
    cb = fig.colorbar(im_h, cax=cax)
    cb.set_label("640 nm max. projection (ADU)", fontsize=6)
    cb.ax.tick_params(labelsize=6, width=0.6, length=2)

    rng = np.random.default_rng(RNG_SEED)
    labels = [lab for _, _, lab in SLIDES]
    xs = np.arange(len(SLIDES))
    m = fields[fields["gradient"] == MAIN_GRADIENT]

    def by_slide(df, s, n):
        return df[(df["session"] == s) & (df["slide"] == n)]

    # i) emitters per field
    ax = fig.add_subplot(bottom[0])
    for x, (s, n, lab) in zip(xs, SLIDES):
        g = by_slide(m, s, n)
        st = ROLE_STYLE[g["role"].iloc[0]]
        _strip(ax, x, g["n_linked"].to_numpy(float), st["color"], st["fill"], rng)
        ax.plot([x - 0.35, x + 0.35], [g["n_linked"].median()] * 2, color="#000000", lw=1.0, zorder=3)
    ax.set_yscale("symlog", linthresh=1)
    ax.set(ylabel="Detected emitters per field", ylim=(-0.3, None))
    ax.set_xticks(xs, [SHORT[l] for l in labels], rotation=60, ha="right")
    axis_standards(ax)
    _letter(ax, "i", x=-0.16)

    # j) vesicles per field, slides and paper
    ax = fig.add_subplot(bottom[1])
    for x, (s, n, lab) in zip(xs, SLIDES):
        g = by_slide(m, s, n)
        st = ROLE_STYLE[g["role"].iloc[0]]
        _strip(ax, x, g["n_vesicles"].to_numpy(float), st["color"], st["fill"], rng)
        ax.plot([x - 0.35, x + 0.35], [g["n_vesicles"].median()] * 2, color="#000000", lw=1.0, zorder=3)
    px = len(SLIDES) + 0.8 + np.arange(len(CONSTRUCTS))
    for x, c in zip(px, CONSTRUCTS):
        v = paper.loc[paper["construct"] == c, "n_vesicles"].dropna().to_numpy(float)
        _strip(ax, x, v, PROTEIN_COLORS[c], True, rng, size=1.5)
        ax.plot([x - 0.35, x + 0.35], [np.median(v)] * 2, color="#000000", lw=1.0, zorder=3)
    ax.set_yscale("symlog", linthresh=10)
    ax.set(ylabel="Vesicles per field", ylim=(-1, None))
    ax.set_xticks(list(xs) + list(px), [SHORT[l] for l in labels] + [LABELS[c] for c in CONSTRUCTS],
                  rotation=60, ha="right")
    axis_standards(ax)
    _letter(ax, "j", x=-0.16)

    # k) passing traces per field, all positions, three gradients
    ax = fig.add_subplot(bottom[2])
    for x, (s, n, lab) in zip(xs, SLIDES):
        g = by_slide(m, s, n)
        st = ROLE_STYLE[g["role"].iloc[0]]
        _strip(ax, x - 0.12, (g["n_pass_in"] + g["n_pass_out"]).to_numpy(float), st["color"], st["fill"], rng)
        for k, gr in enumerate(GRADIENTS):
            gg = by_slide(fields[fields["gradient"] == gr], s, n)
            ax.scatter(x + 0.28, (gg["n_pass_in"] + gg["n_pass_out"]).mean(), marker=GRAD_MARKERS[gr], s=10,
                       facecolors="none", edgecolors="#000000", linewidths=0.6, zorder=3)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set(ylabel="Passing traces per field", ylim=(-0.02, None))
    ax.set_xticks(xs, [SHORT[l] for l in labels], rotation=60, ha="right")
    axis_standards(ax)
    _letter(ax, "k", x=-0.16)

    # l) passing IN traces per field, slides and paper
    ax = fig.add_subplot(bottom[3])
    for x, (s, n, lab) in zip(xs, SLIDES):
        g = by_slide(m, s, n)
        st = ROLE_STYLE[g["role"].iloc[0]]
        if g["role"].iloc[0] == "blank":
            continue
        _strip(ax, x - 0.12, g["n_pass_in"].to_numpy(float), st["color"], st["fill"], rng)
        for gr in GRADIENTS:
            gg = by_slide(fields[fields["gradient"] == gr], s, n)
            ax.scatter(x + 0.28, gg["n_pass_in"].mean(), marker=GRAD_MARKERS[gr], s=10, facecolors="none",
                       edgecolors="#000000", linewidths=0.6, zorder=3)
    for x, c in zip(px, CONSTRUCTS):
        v = paper.loc[paper["construct"] == c, "n_pass_in"].to_numpy(float)
        _strip(ax, x, v, PROTEIN_COLORS[c], True, rng, size=1.5)
        ax.plot([x - 0.35, x + 0.35], [v.mean()] * 2, color="#000000", lw=1.0, zorder=3)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set(ylabel="Passing traces inside\nvesicles per field", ylim=(-0.02, None))
    keep = [i for i, (s, n, _) in enumerate(SLIDES) if n != "Blank"]
    ax.set_xticks([xs[i] for i in keep] + list(px), [SHORT[labels[i]] for i in keep] + [LABELS[c] for c in CONSTRUCTS],
                  rotation=60, ha="right")
    axis_standards(ax)
    _letter(ax, "l", x=-0.16)

    handles = [Line2D([], [], ls="none", marker="o", mfc="none", mec="#000000", ms=4, label="Blank coverslip, field"),
               Line2D([], [], ls="none", marker="o", mfc="#56B4E9", mec="#56B4E9", ms=4, label="Empty vesicles, field"),
               Line2D([], [], ls="none", marker="o", mfc="#000000", mec="#000000", ms=4, label="Protein, same power, field"),
               Line2D([], [], color="#000000", lw=1.0, label="Median (i, j) / mean (l, paper)"),
               *[Line2D([], [], ls="none", marker=GRAD_MARKERS[g], mfc="none", mec="#000000", ms=4,
                        label=f"Mean, gradient {g}") for g in GRADIENTS],
               Line2D([], [], ls="none", marker="o", mfc="none", mec="#56B4E9", ms=5, label="Emitter (a-h)"),
               Line2D([], [], ls="none", marker="s", mfc="none", mec="#F0E442", ms=5, label="Passed the filter (a-h)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    path = out / f"{FIG}a.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_b(out: Path) -> Path:
    apply_style()
    tr = pd.read_csv(out / f"{FIG}b_traces.csv")
    sel = pd.read_csv(out / f"{FIG}b_traces_selection.csv")
    pt = pd.read_csv(out / f"{FIG}b_passing_traces.csv")
    clp = out / f"{FIG}b_classification.csv"
    cl = pd.read_csv(clp) if clp.exists() else pd.DataFrame(columns=["model", "impurity", "kept", "prediction"])
    n = len(sel)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, (25 + 13 * n) / 25.4))
    gs = fig.add_gridspec(n, 2, width_ratios=[2.2, 1], hspace=0.55, wspace=0.35)
    cols = [f"trace_{k}_photons" for k in sel["trace"]]
    lo = 500 * np.floor(tr[cols].min().min() / 500)
    hi = 500 * np.ceil(tr[cols].max().max() / 500)
    for k, r in enumerate(sel.itertuples()):
        ax = fig.add_subplot(gs[k, 0])
        color = "#56B4E9" if r.impurity else "#000000"
        ax.plot(tr["time_s"], tr[f"trace_{r.trace}_photons"], color=color, lw=0.4)
        ax.set(xlim=(0, tr["time_s"].iloc[-1]), ylim=(lo, hi))
        where = "inside vesicle" if r.where == "IN" else "outside vesicles"
        ax.set_title(f"{r.label}, {r.session} field {int(r.fov)}, {where}", loc="left", fontsize=6, pad=1.5)
        axis_standards(ax)
        if k < n - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time (s)")
        if k == n // 2:
            ax.set_ylabel("Photons per frame")
        if k == 0:
            _letter(ax, "a", x=-0.10, y=1.15)
    # b) properties
    right = gs[:, 1].subgridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.95)
    ax = fig.add_subplot(right[0])
    prot = pt[~pt["impurity"]]
    imp = pt[pt["impurity"]]
    ax.scatter(prot["n_events_free"], prot["median_photons_per_event_free"], s=4, color="#BBBBBB", lw=0,
               label=f"Protein, same power (n = {len(prot)})")
    for where, mk in (("IN", "o"), ("OUT", "D")):
        g = imp[imp["where"] == where]
        ax.scatter(g["n_events_free"], g["median_photons_per_event_free"], s=12, marker=mk, facecolors="#56B4E9",
                   edgecolors="#000000", linewidths=0.4,
                   label=f"Blank or empty, {'inside' if where == 'IN' else 'outside'} vesicles (n = {len(g)})")
    ax.set(xscale="log", yscale="log", xlabel="ON events per trace", ylabel="Photons per ON event (median)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.05, 1.02), handletextpad=0.2, borderaxespad=0)
    axis_standards(ax)
    _letter(ax, "b", x=-0.32, y=1.25)
    # c) classification
    ax = fig.add_subplot(right[1])
    models = list(MODEL_LABELS)
    width = 0.36
    for j, imp_flag in enumerate((True, False)):
        vals, ns = [], []
        for mdl in models:
            g = cl[(cl["model"] == mdl) & (cl["impurity"] == imp_flag)]
            vals.append(100 * g["kept"].mean() if len(g) else np.nan)
            ns.append(len(g))
        x = np.arange(len(models)) + (j - 0.5) * width
        ax.bar(x, vals, width=width, color="#56B4E9" if imp_flag else "#BBBBBB", edgecolor="#000000", lw=0.6,
               label="Blank or empty slides" if imp_flag else "Protein, same power")
        for xx, v, nn in zip(x, vals, ns):
            ax.text(xx, (0 if np.isnan(v) else v) + 2, f"n={nn}", ha="center", va="bottom", fontsize=5)
    ax.set_xticks(np.arange(len(models)), ["Halo-D106 vs\nHalo-K117", "scGrx1 vs\nscGrx1-AcK20"])
    ax.set(ylabel="Kept at WD ≥ 0.7 (%)", ylim=(0, 100))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1)
    axis_standards(ax)
    _letter(ax, "c", x=-0.32, y=1.05)
    path = out / f"{FIG}b.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot(out: Path) -> list[Path]:
    return [plot_a(out), plot_b(out)]


# --- Caption ---

def caption(out: Path) -> str:
    fields = pd.read_csv(out / f"{FIG}a_fields.csv")
    m = fields[fields["gradient"] == MAIN_GRADIENT]
    nf = m.groupby(["session", "slide"]).size()
    nfs = "; ".join(f"{lab} {nf[(s, n)]}" for s, n, lab in SLIDES)
    paper = pd.read_csv(out / f"{FIG}a_paper_movies.csv")
    npm = ", ".join(f"{LABELS[c]} {int((paper['construct'] == c).sum())}" for c in CONSTRUCTS)
    pt = pd.read_csv(out / f"{FIG}b_passing_traces.csv")
    clp = out / f"{FIG}b_classification.csv"
    cl = pd.read_csv(clp) if clp.exists() else pd.DataFrame(columns=["model", "impurity", "kept", "prediction"])
    return "\n\n".join([
        "Figure R3.1a. Blank coverslips and empty vesicles through the extraction and quality filter. Coverslips "
        "coated as in all experiments (PLL-PEG with 2 % biotin-PLL-PEG, NeutrAvidin) and imaged without vesicles "
        "(blank), with vesicles without protein (empty; ATTO520, the vesicle dye of the paper, ATTO390 or ATTO425 "
        "lipids), or with protein in vesicles imaged at the same 640 nm power on another day (SNAP-tag, HaloTag7; "
        "13-15 mW, against 20-21 mW in the paper's sessions). 6000 frames at 640 nm (30 ms exposure, 34.49 ms per "
        "frame); vesicle snapshots at 488 nm (ATTO520) or 405 nm (ATTO390, ATTO425). The paper's extraction and "
        "quality filter, unchanged, with fixed protein-channel spot-detection gradients of 10000 (the paper's most "
        "permissive setting) and, in k and l, also 20000 and 40000. a-h) One field per slide: maximum projection "
        "of the 640 nm movie, one common gray scale; the field shown is the one with the median number of detected "
        "emitters at gradient 10000. Circles, emitters the extraction kept (single regions of interest); squares, "
        "traces that passed the quality filter. Scale bar, 5 µm. i) Detected emitters per field (linked "
        "localizations, gradient 10000). j) Vesicles per field, slides and the paper's movies. k) Traces passing "
        "the quality filter per field, inside and outside vesicles; points, fields at gradient 10000; open "
        "markers, means at the three gradients. l) Traces inside vesicles passing the quality filter per field: "
        "slides (as in k) and the paper's movies (points; line, mean). Fields: " + nfs + ". Paper movies: " + npm +
        ". Fields with no localization at all are included as zero. No statistical comparisons were performed.",
        "Figure R3.1b. Every trace from the blank and empty-vesicle slides that passed the quality filter. a) "
        "Detected photons per frame (background-removed 5 x 5 pixel sum x 0.062 photons per ADU, measured from the "
        "camera noise) of each such trace at gradient 10000 (blue) and of one randomly chosen passing trace inside "
        "vesicles from each same-power protein slide (black). b) Median detected photons per ON event against the "
        "number of ON events per trace (ON frames as classified for Figures 5D and 6C), every passing trace at any "
        f"gradient (an emitter found at several gradients counted once; blank or empty slides n = "
        f"{int(pt['impurity'].sum())}, protein slides n = {int((~pt['impurity']).sum())}). c) The final models of "
        "Figures 5D and 6C on the same traces (first 4000 frames, as trained): percentage kept at the paper's "
        "Wasserstein-distance threshold of 0.7 (n = traces). No statistical comparisons were performed."
        + (f" Calls among kept blank or empty traces: " + "; ".join(
            f"{MODEL_LABELS[mdl]}: " + (", ".join(f"{LABELS.get(k, k)} {v}" for k, v in
                                                  cl[(cl['model'] == mdl) & cl['impurity'] & cl['kept']]['prediction']
                                                  .value_counts().items()) or "none")
            for mdl in MODEL_LABELS) + "." if len(cl) else ""),
    ])


# --- Provenance ---

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
        "inputs": {str(p): _sha256(p) for p in sorted(set(inputs))},
    }
    with open(out / f"{FIG}_provenance.yaml", "w") as f:
        yaml.safe_dump(prov, f, sort_keys=False, allow_unicode=True)


# --- Main ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, default=v, help=f"repo-relative run folder (default {v})")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.build:
        repo = Path(__file__).resolve().parents[2]
        inputs = build(repo, args, out)
        (out / f"{FIG}_caption.txt").write_text(caption(out) + "\n")
        if Path(__file__).resolve() != (out / Path(__file__).name).resolve():
            shutil.copy(Path(__file__).resolve(), out / Path(__file__).name)
        write_provenance(out, inputs)
    for p in plot(out):
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
