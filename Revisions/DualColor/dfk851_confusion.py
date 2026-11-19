#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Confusion matrix of an ensemble's kept traces on DFK851's mixed slide (slide 3, single-dye traces), in the
paper's style (ML/utils.plot_confusion_matrix: rows = true class, normalized per row, reversed Crameri davos,
0-1) with locbplots' fonts and output conventions; counts in every cell.

Ensemble as points 13-18: p = mean of the runs' MC-mean p_SNAP, call SNAP at p >= 0.5; kept = mean Wasserstein
certainty > mean of the runs' thresholds (the user's rule, point 13). Cells and runs are dfk851_ft_summary.py's
(--cell C, B, FT head, FT full, P, M, CC, S). The balanced accuracy of the matrix is checked against the readout's
ensembles.csv when that file is given (--check).

Outputs (<out>): <name>.png / .pdf, <name>_matrix.csv (counts and row fractions), <name>_traces.csv (source data:
one row per trace), <name>_caption.txt.

    python Revisions/DualColor/dfk851_confusion.py --cell P \
        --out Results/Revisions/DualColor/dfk851_b/pool7/figures \
        --check Results/Revisions/DualColor/dfk851_b/pool7/ensembles.csv
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
from dc import REPO_ROOT  # noqa: E402

F.PATTERN.update({"P": "pool7_valfov_s*.yaml", "M": "pool7_valmix_s*.yaml", "CC": "ccont_s*.yaml", "S": "size_s*.yaml"})
CLASSES = ("HT", "SNAP")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="P")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/dfk851_b/pool7/figures")
    ap.add_argument("--check", type=Path, help="the readout's ensembles.csv: the matrix's balanced accuracy must match")
    a = ap.parse_args()
    runs, mixed, _, _ = F.load_cell(a.cell, REPO_ROOT / "Results/Revisions/DualColor/dfk851_b/c_transfer")
    tr = runs[runs["trained"]]
    e = (mixed[mixed["run_name"].isin(tr["run_name"])].groupby(F.COLS, as_index=False)
         .agg(p_SNAP=("p_SNAP", "mean"), wasserstein=("wasserstein", "mean"), n_runs=("run_name", "nunique")))
    if (e["n_runs"] != len(tr)).any():
        raise ValueError("a trace is not predicted by every run")
    thr = float(tr["threshold"].mean())
    e["kept"] = e["wasserstein"] > thr
    e["predicted"] = np.where(e["p_SNAP"] >= 0.5, "SNAP", "HT")
    k = e[e["kept"]]
    counts = pd.crosstab(pd.Categorical(k["true_class"], CLASSES), pd.Categorical(k["predicted"], CLASSES), dropna=False)
    counts.index.name, counts.columns.name = "true_class", "predicted"
    frac = counts.div(counts.sum(axis=1), axis=0)
    ba = float(np.mean(np.diag(frac.to_numpy())))
    n_all = e.groupby("true_class").size()
    if a.check:
        ref = pd.read_csv(a.check).query("cell == @a.cell").set_index("metric")["value"]
        if abs(ref["ba_kept"] - ba) > 1e-9 or abs(ref["kept_frac"] - e["kept"].mean()) > 1e-9:
            raise ValueError(f"matrix BA {ba} / kept {e['kept'].mean()} differ from {a.check}: {ref['ba_kept']}, {ref['kept_frac']}")
        print(f"checked against {a.check}: BA kept {ba:.4f}, kept fraction {e['kept'].mean():.4f}")

    sys.path.insert(0, str(F.LOCBPLOTS))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from scripts.plot_standards import apply_rcparams, save_figure, save_source_data
    apply_rcparams()
    davos_r = ListedColormap(np.loadtxt(F.LOCBPLOTS / "ScientificColourMaps8" / "davos" / "davos.txt")).reversed()
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    im = ax.imshow(frac.to_numpy(), cmap=davos_r, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            v = frac.iat[i, j]
            ax.text(j, i, f"{v:.2f}\n(n = {counts.iat[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(CLASSES)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Fraction of the true class")
    cb.outline.set_linewidth(0.4)
    name = f"confusion_kept_{a.cell.replace(' ', '_')}"
    ax.set_title(f"{F.LABELS.get(a.cell, a.cell)}\nkept {len(k)} of {len(e)} traces ({e['kept'].mean():.0%}), "
                 f"balanced accuracy {ba:.3f}", fontsize=7, color="black")
    save_figure(fig, a.out, f"{name}.png")
    plt.close(fig)
    m = counts.stack().rename("n").reset_index().merge(frac.stack().rename("row_fraction").reset_index(),
                                                        on=["true_class", "predicted"])
    m.to_csv(a.out / f"{name}_matrix.csv", index=False)
    save_source_data(e[F.COLS + ["p_SNAP", "wasserstein", "kept", "predicted"]], a.out, f"{name}_traces.csv")
    (a.out / f"{name}_caption.txt").write_text(
        f"Confusion matrix of the traces kept by the uncertainty filter on DFK851's mixed slide, for the ensemble of "
        f"{len(tr)} models of cell {a.cell} ({F.LABELS.get(a.cell, a.cell)}). Rows, true class (single-dye vesicle "
        f"within 2 px); columns, predicted class (ensemble mean p_SNAP >= 0.5); colour and upper number, fraction of "
        f"the row; lower number, traces. Kept: ensemble mean Wasserstein certainty above the mean of the models' "
        f"validation thresholds ({thr:.3f}); {len(k)} of {len(e)} single-dye traces kept "
        f"(HT {int((k['true_class'] == 'HT').sum())} of {int(n_all['HT'])}, SNAP {int((k['true_class'] == 'SNAP').sum())} "
        f"of {int(n_all['SNAP'])}) from 118 fields of view of one slide; balanced accuracy {ba:.3f}. No statistical "
        f"test was performed.\n")
    print(counts.to_string(), "\n", frac.round(3).to_string(), f"\nBA {ba:.4f}; kept {len(k)} / {len(e)}; threshold {thr:.4f}")


if __name__ == "__main__":
    main()
