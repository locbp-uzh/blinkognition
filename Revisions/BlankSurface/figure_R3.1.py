#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Figure R3.1 (first-round review, Reviewer 3, point 1): fully prepared coverslips imaged without any sample.

Trimmed to the request (user, 2026-11-20): the two blank coverslips (coated as in all experiments, no vesicles,
no protein; DFK786, DFK787) and, for scale, protein in vesicles imaged at the same 640 nm power (DFK788, SNAP-tag
in ATTO520 vesicles). The full version (empty vesicles, counts against the paper, passing traces) is kept under
revisions_figures/not_in_revision/R3.1_full_20261120 and in git history (commit 4ddac80).

a-c) One field per slide, 640 nm maximum projection over 6000 frames, one common gray scale (ADU; lowest 1st and
     highest 99.9th percentile of the fields shown). The field shown is the slide's field with the median number of
     detected emitters at gradient 10000 (not chosen by eye). Circles: emitters the paper's extraction kept
     (single ROIs); squares: traces that passed the quality filter. The caption gives the counts over all fields
     at the three gradients (10000, the paper's most permissive, 20000, 40000).
Descriptive; no statistical tests.

Usage:
    python figure_R3.1.py
        Replot from the CSVs next to this script (works in the figure folder as distributed).
    python figure_R3.1.py --build [--out <dir>] [--count ... --projections ...]
        From the blinkognition repo: build the CSVs from the analysis runs, replot, copy this script into --out.
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
from matplotlib.patches import Rectangle  # noqa: E402

FIG = "R3.1"
SLIDES = [("DFK786", "Blank", "Blank coverslip 1"), ("DFK787", "Blank", "Blank coverslip 2"),
          ("DFK788", "SNAP520", "SNAP-tag in vesicles, same power")]
GRADIENTS = [10000, 20000, 40000]
MAIN_GRADIENT = 10000
BOX = 5
PIXEL_NM = 130
SCALE_BAR_UM = 5
FIG_WIDTH_IN = 180 / 25.4
R = "Results/Revisions"
DEFAULTS = {"count": f"{R}/BlankSurface/s3it/count/run_003", "projections": f"{R}/BlankSurface/s3it/projections/run_001"}


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


def _letter(ax, letter: str, x: float = -0.06, y: float = 1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def build(repo: Path, a: argparse.Namespace, out: Path) -> list[Path]:
    p = {k: repo / getattr(a, k) for k in DEFAULTS}
    inputs = [p["count"] / "fov.csv", p["count"] / "traces.csv", p["projections"] / "images.npz"]
    f = pd.read_csv(inputs[0])
    used = {(s, n): lab for s, n, lab in SLIDES}
    f = f[[(s, n) in used for s, n in zip(f["session"], f["slide"])]].copy()
    f["label"] = [used[(s, n)] for s, n in zip(f["session"], f["slide"])]
    f[["gradient", "session", "slide", "label", "fov", "n_locs", "n_linked", "n_single", "n_single_in",
       "n_single_out", "n_pass_in", "n_pass_out"]].to_csv(out / f"{FIG}_fields.csv", index=False)
    m = f[f["gradient"] == MAIN_GRADIENT]
    shown = []
    for s, n, lab in SLIDES:
        g = m[(m["session"] == s) & (m["slide"] == n)].sort_values(["n_linked", "fov"]).reset_index(drop=True)
        shown.append(g.iloc[(len(g) - 1) // 2][["session", "slide", "fov", "n_linked"]].to_dict() | {"label": lab})
    shown = pd.DataFrame(shown)
    z = np.load(inputs[2])
    imgs = [z[f"{r.session}|{r.slide}|{int(r.fov):04d}|max640"] for r in shown.itertuples()]
    shown["vmin"] = min(float(np.percentile(im, 1)) for im in imgs)
    shown["vmax"] = max(float(np.percentile(im, 99.9)) for im in imgs)
    for i, im in enumerate(imgs):
        np.savetxt(out / f"{FIG}_image_{'abc'[i]}.csv", im, delimiter=",", fmt="%.1f")
    shown.to_csv(out / f"{FIG}_fields_shown.csv", index=False, float_format="%.6g")
    tr = pd.read_csv(inputs[1])
    marks = []
    for i, r in enumerate(shown.itertuples()):
        g = tr[(tr["gradient"] == MAIN_GRADIENT) & (tr["session"] == r.session) & (tr["slide"] == r.slide)
               & (tr["fov"] == r.fov)]
        marks += [{"panel": "abc"[i], "x": t.x, "y": t.y, "passed": bool(t.passed_pipeline)} for t in g.itertuples()]
    pd.DataFrame(marks, columns=["panel", "x", "y", "passed"]).to_csv(out / f"{FIG}_rois_shown.csv", index=False)
    return inputs


def plot(out: Path) -> Path:
    apply_style()
    shown = pd.read_csv(out / f"{FIG}_fields_shown.csv")
    marks = pd.read_csv(out / f"{FIG}_rois_shown.csv")
    vmin, vmax = float(shown["vmin"].iloc[0]), float(shown["vmax"].iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 72 / 25.4), gridspec_kw={"wspace": 0.08})
    im_h = None
    for i, (ax, r) in enumerate(zip(axes, shown.itertuples())):
        img = np.loadtxt(out / f"{FIG}_image_{'abc'[i]}.csv", delimiter=",")
        im_h = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        for t in marks[marks["panel"] == "abc"[i]].itertuples():
            if t.passed:
                ax.add_patch(Rectangle((t.x - 2.5, t.y - 2.5), BOX + 4, BOX + 4, fill=False, lw=0.8, ec="#F0E442"))
            else:
                ax.add_patch(plt.Circle((t.x + (BOX - 1) / 2, t.y + (BOX - 1) / 2), 4, fill=False, lw=0.5,
                                        ec="#56B4E9"))
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        ne = int(r.n_linked)
        ax.set_title(f"{r.label}\n{r.session}, field {int(r.fov)}: {ne} emitter{'' if ne == 1 else 's'}", fontsize=6,
                     pad=2)
        _letter(ax, "abc"[i])
        if i == 0:
            L = SCALE_BAR_UM * 1000 / PIXEL_NM
            ax.plot([230 - 12 - L, 230 - 12], [230 - 12, 230 - 12], color="white", lw=1.5)
            ax.text(230 - 12 - L / 2, 230 - 18, f"{SCALE_BAR_UM} µm", color="white", ha="center", va="bottom", fontsize=6)
    cax = axes[-1].inset_axes([1.04, 0.0, 0.05, 1.0])
    cb = fig.colorbar(im_h, cax=cax)
    cb.set_label("640 nm max. projection (ADU)", fontsize=6)
    cb.ax.tick_params(labelsize=6, width=0.6, length=2)
    handles = [Line2D([], [], ls="none", marker="o", mfc="none", mec="#56B4E9", ms=5, label="Detected emitter"),
               Line2D([], [], ls="none", marker="s", mfc="none", mec="#F0E442", ms=5, label="Passed the quality filter")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
    path = out / f"{FIG}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def caption(out: Path) -> str:
    f = pd.read_csv(out / f"{FIG}_fields.csv")
    m = f[f["gradient"] == MAIN_GRADIENT]
    blank = m[m["slide"] == "Blank"]
    ref = m[m["slide"] == "SNAP520"]
    pass_blank = {g: int(f.loc[(f["gradient"] == g) & (f["slide"] == "Blank"), ["n_pass_in", "n_pass_out"]].sum().sum())
                  for g in GRADIENTS}
    return (
        "Figure R3.1. Coverslips prepared as for all measurements (PLL-PEG with 2 % biotin-PLL-PEG, NeutrAvidin) "
        "and imaged without any sample (a, b; no vesicles, no protein), and, for comparison, SNAP-tag in vesicles "
        "imaged at the same 640 nm power (c). One field per slide: maximum projection of the 640 nm movie (6000 "
        "frames, 30 ms exposure, 34.49 ms per frame, 13-15 mW), one common gray scale; the field shown is the one "
        "with the median number of detected emitters. Circles, emitters detected by the unchanged extraction "
        "(spot-detection gradient 10000, the most permissive setting of the paper); squares, traces that passed "
        f"the quality filter. Scale bar, 5 µm. Over all {len(blank)} fields of the blank coverslips: median "
        f"{blank['n_locs'].median():g} localizations per field, {int((blank['n_linked'] == 0).sum())} fields without "
        f"any detected emitter, {int(blank['n_linked'].sum())} emitters in total, and "
        f"{pass_blank[10000]}, {pass_blank[20000]} and {pass_blank[40000]} traces passing the quality filter at "
        f"gradients 10000, 20000 and 40000. SNAP-tag slide ({len(ref)} fields): {ref['n_linked'].mean():.1f} "
        f"emitters and {(ref['n_pass_in'] + ref['n_pass_out']).mean():.1f} passing traces per field. No statistical "
        "comparisons were performed."
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
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", dest=k, default=v, help=f"repo-relative run folder (default {v})")
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
    print(f"Saved: {plot(out)}")


if __name__ == "__main__":
    main()
