#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Limits of detection of two cells side by side, from a readout folder of points 16-18 (lod.csv, lod_paired.csv,
model_check.csv when present): SNAP in HT and HT in SNAP on the designs with about 16 molecules per FOV (16/1 to
500/24), 95 % FOV-cluster bootstrap intervals, the perfect-classifier limit 1 - 0.05^(1/n) (black dotted), and
open symbols where point 15's model check failed for that cell at that number of FOVs. No computation beyond
reading the readout's tables.

    python Revisions/DualColor/dfk851_lod_compare.py --cells C P --readout Results/Revisions/DualColor/dfk851_b/pool7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dfk851_ft_summary as F  # noqa: E402
import dfk851_lod as L  # noqa: E402
from dc import REPO_ROOT  # noqa: E402

COLOR = {"C": "orange", "B": "blue", "FT head": "green", "FT full": "vermillion", "P": "pink", "M": "sky_blue",
         "CC": "green", "S": "sky_blue"}   # as in the point 16-18 figures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs=2, default=["C", "P"])
    ap.add_argument("--readout", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/dfk851_b/pool7")
    a = ap.parse_args()
    c1, c2 = a.cells
    lod = pd.read_csv(a.readout / "lod.csv")
    pair = pd.read_csv(a.readout / "lod_paired.csv")
    chk = pd.read_csv(a.readout / "model_check.csv") if (a.readout / "model_check.csv").exists() else None
    valid = (chk.groupby(["model", "k"])["pass"].all().to_dict() if chk is not None else {})

    sys.path.insert(0, str(F.LOCBPLOTS))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.plot_standards import COLORS, apply_axis_standards, apply_rcparams, save_figure, save_source_data
    apply_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    rows, notes = [], []
    for ax, (key, ylab, letter) in zip(axes, (("lod_snap", "SNAP in HT (SNAP share)", "a"),
                                             ("lod_ht", "HT in SNAP (HT share)", "b"))):
        nn = np.geomspace(L.DIAG[0][0], L.DIAG[-1][0], 200)
        ax.plot(nn, L.ideal_lod(nn), color=COLORS["black"], lw=1.0, ls=":", zorder=1)
        for j, c in enumerate((c1, c2)):
            g = lod[(lod["cell"] == c) & lod["metric"].str.startswith(f"{key}@")].copy()
            g["n"] = g["metric"].str.extract(r"@(\d+)/").astype(int)[0]
            g["k"] = g["metric"].str.extract(r"/(\d+)$").astype(int)[0]
            g = g.sort_values("n")
            g["validated"] = [valid.get((c, k), np.nan) for k in g["k"]]
            x = g["n"].to_numpy() * (1 + 0.035 * (j - 0.5))
            y, lo, hi = g["value"].to_numpy(), g["lo"].to_numpy(), np.minimum(g["hi"].to_numpy(), 1.0)
            col = COLORS[COLOR.get(c, "black")]
            ax.plot(x, y, color=col, lw=1.1)
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=col, elinewidth=0.7, capsize=0)
            ok = g["validated"].fillna(True).astype(bool).to_numpy()
            ax.scatter(x[ok], y[ok], s=16, color=col, zorder=3, edgecolor="none")
            ax.scatter(x[~ok], y[~ok], s=16, facecolor="white", edgecolor=col, linewidth=0.9, zorder=3)
            rows.append(g.assign(direction=key))
        d = pair[(pair["pair"] == f"{c2} - {c1}") & pair["metric"].str.startswith(f"{key}@")]
        if len(d):
            notes.append(f"{'SNAP in HT' if key == 'lod_snap' else 'HT in SNAP'}: " + ", ".join(
                f"{m.split('@')[1]} {v:+.3f} [{lo_:+.3f}, {hi_:+.3f}]" for m, v, lo_, hi_ in
                sorted(zip(d["metric"], d["diff"], d["lo"], d["hi"]), key=lambda t: int(t[0].split("@")[1].split("/")[0]))))
        ax.set_xscale("log")
        ax.set_xticks([n for n, _ in L.DIAG])
        ax.set_xticklabels([f"{n}/{k}" for n, k in L.DIAG])
        ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.set_xlabel("Molecules / FOVs")
        ax.set_ylabel("Limit of detection\n(share of the minority protein)" if letter == "a" else "")
        ax.set_title(f"Limit of detection, {ylab.split(' (')[0]}", color="black")
        ax.text(-0.18 if letter == "a" else -0.08, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
        apply_axis_standards(ax)
    handles = [Line2D([], [], color=COLORS[COLOR.get(c, "black")], lw=1.1, marker="o", ms=3.5, label=F.LABELS.get(c, c))
               for c in (c1, c2)]
    handles += [Line2D([], [], color=COLORS["black"], lw=1.0, ls=":", label="perfect classifier, 1 - 0.05^(1/n)"),
                Line2D([], [], ls="none", marker="o", ms=3.5, mfc="white", mec=COLORS["black"],
                       label="model check failed at this number of FOVs")]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=2, fontsize=6, bbox_to_anchor=(0.5, -0.12))
    out = a.readout / "figures"
    name = f"lod_{c1}_vs_{c2}".replace(" ", "_")
    save_figure(fig, out, f"{name}.png")
    plt.close(fig)
    save_source_data(pd.concat(rows), out, f"{name}_data.csv")
    (out / f"{name}_caption.txt").write_text(
        f"Limits of detection on DFK851's mixed slide for {F.LABELS.get(c1, c1)} and {F.LABELS.get(c2, c2)}. (a) SNAP "
        f"in HT; (b) HT in SNAP: the smallest minority share detected with 5 % false positives against a blank of the "
        f"same design and 5 % false negatives (CLSI EP17, normal theory), for designs with about 16 molecules per FOV, "
        f"from a two-component variance model (FOV and within-FOV variance of the ensemble's SNAP probability) fitted "
        f"to n = 1828 single-dye traces in n = 112 FOVs of one slide. Bars, 95 % FOV-cluster bootstrap intervals "
        f"(2000 resamples); open symbols, numbers of FOVs at which the model failed its check on real samples (point "
        f"15's rule; at one FOV the check is decided by a few of 112 FOVs and is noisy); black dotted line, the limit of "
        f"a perfect classifier, 1 - 0.05^(1/n). Paired differences {c2} - {c1} with 95 % intervals: "
        + "; ".join(notes) + ".\n")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
