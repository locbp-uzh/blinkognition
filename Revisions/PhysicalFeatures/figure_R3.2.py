#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Figure R3.2 (first-round review, Reviewer 3, point 2): a clustering plot of measurable blinking properties of
every filtered single-molecule trace inside vesicles, before deep learning.

Trimmed to the request (user, 2026-11-20): one clustering plot per protein pair. The features are the kinetic
properties the paper uses: mean ON time, mean OFF time and duty cycle per molecule (R3.3a event table; the
paper's ON classification on the full 6000 frames, 34.49 ms per frame). Absolute photon numbers are not used to
separate proteins (user, 2026-11-20): they carry day- and slide-specific information (laser power, focus,
slide), which is why the traces are normalized; photon numbers are shown in R3.3. The full version (feature-only
classifier against the networks, effect sizes, corner plots, Fig. 6C subsets) is kept under
revisions_figures/not_in_revision/R3.2_full_20261120 and in git history (commit afaf3f5).

a-c) Per pair (Halo-D106 vs SNAP-C148, Halo-D106 vs Halo-K117, scGrx1 vs scGrx1-AcK20): first two principal
     components of the log10-transformed features, standardized within the pair; density contours per protein
     enclosing 50 % and 90 % of its molecules; arrows, feature directions (loadings x 2); ARI, adjusted Rand
     index between an unsupervised two-component Gaussian mixture on the same features and the protein identity.
Descriptive; no hypothesis tests.

Usage:
    python figure_R3.2.py
        Replot from the CSVs next to this script (works in the figure folder as distributed).
    python figure_R3.2.py --build --analysis <features_analysis.py run> [--out <dir>]
        From the blinkognition repo: collect the CSVs from the analysis run, replot, copy this script into --out.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

FIG = "R3.2"
PAIRS = [("HaloD106", "SNAPC148"), ("HaloD106", "HaloK117"), ("scGrx1", "scGrx1AcK20")]
LABELS = {"HaloD106": "Halo-D106", "HaloK117": "Halo-K117", "SNAPC148": "SNAP-C148",
          "scGrx1": "scGrx1", "scGrx1AcK20": "scGrx1-AcK20"}
PROTEIN_COLORS = {"HaloD106": "#0072B2", "HaloK117": "#E69F00", "SNAPC148": "#CC79A7",
                  "scGrx1": "#009E73", "scGrx1AcK20": "#D55E00"}
BLACK = "#000000"
FEATURES = ["mean_on_ms", "mean_off_ms", "duty_cycle"]
FEATURE_SHORT = {"mean_on_ms": "ON", "mean_off_ms": "OFF", "duty_cycle": "Duty"}
HDR_LEVELS = (0.9, 0.5)              # contours enclosing 90 % and 50 % of the molecules
KDE_GRID = 90
LOADING_SCALE = 2.0
RNG_SEED = 42
MM = 1 / 25.4


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


def build(analysis: Path, out: Path) -> list[Path]:
    files = {k: analysis / f"{k}.csv" for k in ("features", "pca_scores", "pca_summary", "gmm", "gmm_summary")}
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
        rows.append(d)
    pd.concat(rows, ignore_index=True).to_csv(out / f"{FIG}_molecules.csv", index=False, float_format="%.6g")
    for k in ("pca_summary", "gmm_summary"):
        pd.read_csv(files[k]).to_csv(out / f"{FIG}_{k}.csv", index=False, float_format="%.5f")
    return list(files.values()) + [analysis / "manifest.yaml"]


def _hdr_contours(ax, x: np.ndarray, y: np.ndarray, color: str, xlim, ylim, rng: np.random.Generator,
                  max_points: int = 6000) -> None:
    """Contours of a Gaussian KDE enclosing HDR_LEVELS of the points."""
    if x.size > max_points:
        i = rng.choice(x.size, max_points, replace=False)
        x, y = x[i], y[i]
    kde = gaussian_kde(np.vstack([x, y]))
    gx, gy = np.meshgrid(np.linspace(*xlim, KDE_GRID), np.linspace(*ylim, KDE_GRID))
    dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    levels = sorted(np.quantile(kde(np.vstack([x, y])), [1 - q for q in HDR_LEVELS]))
    ax.contour(gx, gy, dens, levels=levels, colors=[color], linewidths=[0.8, 1.3])


def plot(out: Path) -> Path:
    apply_style()
    rng = np.random.default_rng(RNG_SEED)
    mol = pd.read_csv(out / f"{FIG}_molecules.csv")
    pca = pd.read_csv(out / f"{FIG}_pca_summary.csv")
    gmm = pd.read_csv(out / f"{FIG}_gmm_summary.csv").set_index("pair")
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM), gridspec_kw={"wspace": 0.38})
    for i, (a, p) in enumerate(zip(axes, PAIRS)):
        k = pair_key(p)
        d = mol[mol["pair"] == k]
        xl = np.percentile(d["pc1"], [0.5, 99.5]) + np.array([-0.5, 0.5])
        yl = np.percentile(d["pc2"], [0.5, 99.5]) + np.array([-0.5, 0.5])
        for c in p:
            g = d[d["construct"] == c]
            _hdr_contours(a, g["pc1"].to_numpy(), g["pc2"].to_numpy(), PROTEIN_COLORS[c], xl, yl, rng)
        ld = pca[pca["pair"] == k].set_index("component")
        vecs = {f: LOADING_SCALE * np.array([ld.loc["PC1", f"loading_{f}"], ld.loc["PC2", f"loading_{f}"]])
                for f in FEATURES}
        for f, v in vecs.items():
            a.annotate("", xy=tuple(v), xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 0.7, "color": BLACK})
            u = v / np.linalg.norm(v)
            a.text(*(v + 0.45 * u), FEATURE_SHORT[f], fontsize=5.5, color=BLACK, ha="center", va="center")
        ev = ld["explained_variance_ratio"]
        a.set(xlim=xl, ylim=yl, xlabel=f"PC1 ({100 * ev['PC1']:.0f} %)", ylabel=f"PC2 ({100 * ev['PC2']:.0f} %)")
        a.text(0.02, 0.98, f"ARI {gmm.loc[k, 'adjusted_rand_index']:.2f}", transform=a.transAxes, fontsize=6,
               color=BLACK, va="top", ha="left")
        a.set_title(f"{LABELS[p[0]]} vs {LABELS[p[1]]}", loc="left", pad=3)
        axis_standards(a)
        a.text(-0.22, 1.06, "abc"[i], transform=a.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
    handles = [Line2D([], [], color=PROTEIN_COLORS[c], lw=1.2, label=LABELS[c])
               for c in ("HaloD106", "SNAPC148", "HaloK117", "scGrx1", "scGrx1AcK20")]
    handles += [Line2D([], [], color=BLACK, lw=1.3, label="50 % of molecules"),
                Line2D([], [], color=BLACK, lw=0.8, label="90 % of molecules")]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
    path = out / f"{FIG}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def caption(out: Path) -> str:
    mol = pd.read_csv(out / f"{FIG}_molecules.csv")
    n = {k: ", ".join(f"{LABELS[c]} {v}" for c, v in g["construct"].value_counts().items()) for k, g in mol.groupby("pair")}
    return (
        "Figure R3.2. Measurable blinking properties of single molecules before deep learning. For every filtered "
        "single-molecule trace inside vesicles, mean ON time, mean OFF time and duty cycle (ON frames as classified "
        "for Figures 5D and 6C; 6000 frames, 34.49 ms per frame). a) Halo-D106 vs SNAP-C148, b) Halo-D106 vs "
        "Halo-K117, c) scGrx1 vs scGrx1-AcK20: first two principal components of the log-transformed, standardized "
        "features; contours enclose 50 % (thick) and 90 % (thin) of each protein's molecules; arrows, feature "
        "directions; ARI, adjusted Rand index between an unsupervised two-component Gaussian mixture and the "
        "protein identity (0, no relation; 1, identical). n: " + "; ".join(n[pair_key(p)] for p in PAIRS) +
        " molecules. Absolute photon numbers are not used, as they depend on excitation power, focus and slide "
        "(photon numbers per protein: Figure R3.3). No statistical tests were performed."
    )


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
    prov = {"figure": FIG, "created": datetime.now().isoformat(timespec="seconds"), "script": str(here),
            "script_sha256": _sha256(here), "git_commit": git("rev-parse", "HEAD") or None,
            "git_dirty": bool(git("status", "--porcelain", "--", str(here.parent))),
            "python": platform.python_version(),
            "packages": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
            "inputs": {str(p): _sha256(p) for p in inputs}}
    with open(out / f"{FIG}_provenance.yaml", "w") as f:
        yaml.safe_dump(prov, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--analysis", type=Path, help="features_analysis.py run folder")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.build:
        if not args.analysis:
            ap.error("--build needs --analysis")
        inputs = build(args.analysis, out)
        (out / f"{FIG}_caption.txt").write_text(caption(out) + "\n")
        if Path(__file__).resolve() != (out / Path(__file__).name).resolve():
            shutil.copy(Path(__file__).resolve(), out / Path(__file__).name)
        write_provenance(out, inputs)
    print(f"Saved: {plot(out)}")


if __name__ == "__main__":
    main()
