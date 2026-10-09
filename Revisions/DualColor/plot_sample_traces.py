#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Sample traces of the dual-color datasets: for each dataset and slide type (single-protein or
mixed), N randomly chosen traces of each protein, as the models see them (filtered traces,
5 x 5 box sum, GMM background removed, per-trace min-max), against time (34.49 ms per frame,
ND2 timestamps of the 640 movies).

Single-protein slides: the models' training pool (train_pure.py): filtered traces within the
slide's radius (4 px) of a vesicle carrying the slide's dye; the class is the slide's
protein. Mixed slides: the test traces, classed IN_ATTO390 / IN_ATTO520 (one single-dye
vesicle within 2 px); the class is the vesicle's protein. Traces are drawn at random
(seeded) from all slides of the type; each panel names its slide, FOV and uniqueID.

Plot standards: the lab's (locbplots; here through Extraction/utils.py, which sets the same
rcParams and Okabe-Ito palette): 7 / 6 pt, no top or right spines, no grid, units on axes.
Colors: the dyes' (base.yaml): HT / ATTO390 blue, SNAP / ATTO520 orange.

Outputs (Results/Revisions/SampleTraces/): <dataset>_<pure|mixed>.pdf, with source data
<dataset>_<pure|mixed>_traces.csv (time and one column per plotted trace) and
<dataset>_<pure|mixed>_selection.csv (which traces), caption.txt, a copy of this script.

Usage (from the repo root):
    python Revisions/DualColor/plot_sample_traces.py [--n 8] [--seed 42]
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "ML"))
from dc import REPO_ROOT  # noqa: E402
from train_pure import load_dataset  # noqa: E402
from utils import apply_axis_standards  # noqa: E402  (ML/utils.py, the locbplots axis style)

# Extraction/utils.py (lab rcParams and Okabe-Ito colors) under its own name: `utils` is ML/utils.py here
_spec = importlib.util.spec_from_file_location("extraction_utils", HERE.parents[1] / "Extraction" / "utils.py")
xu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xu)

# --- Constants ---
DATASETS = ["DFK785", "DFK788", "DFK789"]
FRAME_S = 0.03449                      # s per frame (30 ms exposure + readout; ND2 timestamps)
CLASS_DYE = {"HT": "ATTO390", "SNAP": "ATTO520"}
SLIDE_TYPES = {"pure": "single-protein slides", "mixed": "mixed slides"}
FIG_WIDTH_IN = 180 / 25.4              # double column
ROW_HEIGHT_IN = 0.72
LINE_WIDTH = 0.4
OUT_DIR = REPO_ROOT / "Results" / "Revisions" / "SampleTraces"


def pick(meta: pd.DataFrame, slide_type: str, n: int, rng: np.random.Generator) -> dict[str, pd.Index]:
    """Row indices of n random traces per class for one slide type."""
    if slide_type == "pure":
        pool = meta[(meta["slide_type"] == "pure") & meta["in_vesicle"] & meta["own_dye"]]
    else:
        pool = meta[(meta["slide_type"] == "mixed") & meta["true_class"].isin(CLASS_DYE)]
    out = {}
    for c in CLASS_DYE:
        idx = pool.index[pool["true_class"] == c].to_numpy()
        out[c] = pd.Index(np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False)))
    return out


def plot(ds: str, slide_type: str, meta: pd.DataFrame, X: np.ndarray, chosen: dict, dyes: dict) -> plt.Figure:
    n = max(len(v) for v in chosen.values())
    t = np.arange(X.shape[-1]) * FRAME_S
    fig, axes = plt.subplots(n, 2, figsize=(FIG_WIDTH_IN, ROW_HEIGHT_IN * n + 0.7), sharex=True, sharey=True,
                             squeeze=False)
    for j, c in enumerate(CLASS_DYE):
        color = xu._COLORS[dyes[CLASS_DYE[c]]["color"]]
        axes[0, j].text(0.5, 1.55, f"{dyes[CLASS_DYE[c]]['protein']} ({c}), {CLASS_DYE[c]} vesicles",
                        transform=axes[0, j].transAxes, ha="center", va="bottom", fontsize=7, color="black")
        for i in range(n):
            ax = axes[i, j]
            if i < len(chosen[c]):
                r = meta.loc[chosen[c][i]]
                ax.plot(t, X[chosen[c][i], 0], color=color, lw=LINE_WIDTH)
                ax.set_title(f"{r['slide']}  {r['fov']}  #{r['uniqueID']}", fontsize=6, color="black", pad=2,
                             loc="left")
            else:
                ax.set_visible(False)
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 1])
            ax.set_xlim(0, t[-1])
            apply_axis_standards(ax)
        axes[n - 1, j].set_xlabel("Time (s)")
    fig.supylabel("Normalized intensity (a.u.)", fontsize=7)
    fig.suptitle(f"{ds}, {SLIDE_TYPES[slide_type]}", fontsize=7, y=1.0)
    fig.tight_layout(h_pad=0.6, w_pad=2)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8, help="traces per protein and panel column")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    dyes = yaml.safe_load(open(HERE / "base.yaml"))["dyes"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        meta, X, info = load_dataset(ds, {})
        for st in SLIDE_TYPES:
            chosen = pick(meta, st, a.n, rng)
            fig = plot(ds, st, meta, X, chosen, dyes)
            stem = f"{ds}_{st}"
            fig.savefig(OUT_DIR / f"{stem}.pdf")
            plt.close(fig)
            sel = pd.concat([meta.loc[v].assign(panel_class=c, panel_row=np.arange(len(v)) + 1)
                             for c, v in chosen.items()])
            sel["trace_assignment_run"] = info["trace_assignment_run"]
            sel["extraction_run"] = info["extraction_run"]
            sel.to_csv(OUT_DIR / f"{stem}_selection.csv", index=False)
            src = pd.DataFrame({"time_s": np.arange(X.shape[-1]) * FRAME_S})
            for c, v in chosen.items():
                for i in v:
                    r = meta.loc[i]
                    src[f"{c}|{r['slide']}|{r['fov']}|{r['uniqueID']}"] = X[i, 0]
            src.to_csv(OUT_DIR / f"{stem}_traces.csv", index=False)
            print(f"{stem}: " + ", ".join(f"{c} {len(v)}" for c, v in chosen.items()))
    (OUT_DIR / "caption.txt").write_text(
        f"Sample single-molecule fluorescence traces of HT7-HMSiR-HTL (HT) and SNAP-HMSiR-IA (SNAP) "
        f"encapsulated in ATTO390- and ATTO520-labeled vesicles, respectively, for each dual-color dataset "
        f"(DFK785, DFK788, DFK789) on single-protein and mixed slides. Each panel shows one randomly chosen "
        f"filtered trace (5 x 5 pixel sum, background removed, min-max normalized; 640 nm excitation, "
        f"{FRAME_S * 1000:.2f} ms per frame, {X.shape[-1]} frames) with its slide, field of view and trace "
        f"identifier; {a.n} traces per protein. On mixed slides the protein is assigned from the single-dye "
        f"vesicle within 2 pixels of the trace. No statistical comparisons were performed.\n")
    shutil.copy(__file__, OUT_DIR / Path(__file__).name)
    print(f"Done: {OUT_DIR}")


if __name__ == "__main__":
    main()
