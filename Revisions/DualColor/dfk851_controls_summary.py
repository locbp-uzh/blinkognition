#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Readouts of TRAINING_NOTES.md point 18 (the controls of points 16 and 17).

Cells: C (r6_pooled), CC (ccont_s<seed>: C continued at the lower learning rate on its own data), FT full (point
16), S (size_s<seed>: point 17's P with DFK851's pool cut to 841 traces; seven seeds), P (point 17).
Pairs: CC - C, FT full - CC, S - C, P - S, and for reference FT full - C, P - C.

1. DFK851's mixed slide: dfk851_ft_summary.py's readouts (point 16/17; ensembles, paired with point 11's rule,
   limits of detection, counting line) for these cells (<out>/); S's six-seed ensemble (P's seeds) in <out>/six_seeds/.
2. The six older sessions' mixed slides (older.csv, older_paired.csv): per session the ensemble AUC on all its
   labeled mixed traces (as round 6), and the mean over the six sessions; FOV-cluster bootstrap (FOVs resampled
   within each slide, 2000 draws, the same draws for every cell); point 11's rule on the six-session mean (margin
   = 2 x the larger SD over the cells' runs of their per-run six-session means).

    python Revisions/DualColor/dfk851_controls_summary.py [--n-boot 2000] [--out <dir>]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dfk851_ft_summary as F  # noqa: E402
from dc import REPO_ROOT  # noqa: E402
from round6_summary import boot_groups, draw, rows_of  # noqa: E402

F.CELLS = ("C", "CC", "FT full", "S", "P")
F.PATTERN.update({"CC": "ccont_s*.yaml", "S": "size_s*.yaml", "P": "pool7_valfov_s*.yaml"})
F.PAIRS = (("CC", "C"), ("FT full", "CC"), ("S", "C"), ("P", "S"), ("FT full", "C"), ("P", "C"))
F.COLOR_KEYS = ("orange", "green", "vermillion", "sky_blue", "pink")
F.LABELS.update({"CC": "C continued at the lower rate (own data)", "S": "7 sessions, DFK851 cut to 841 traces"})
F.FIG_NAME = "controls"
F.CAPTION_LEAD = ("Controls of the fine-tune and of the seven-session pool on DFK851's mixed slide: C, C continued at "
                  "the fine-tune's lower learning rate on its own data (CC), C fine-tuned on DFK851's single-protein "
                  "slides (FT full), the seven-session pool with DFK851's pool cut to 841 traces (S) and in full (P); "
                  "five models per ensemble for C, CC and FT full, seven for S, six for P.")


def older_sessions(cells, n_boot: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-session and six-session-mean ensemble AUC on the older sessions' labeled mixed traces, paired bootstrap."""
    ens, per_run = None, {}
    for c in F.CELLS:
        runs, _, other, _ = cells[c]
        tr = runs.loc[runs["trained"], "run_name"]
        o = other[other["run_name"].isin(tr) & other["true_class"].isin(["HT", "SNAP"])]
        g = o.groupby(F.COLS, as_index=False).agg(**{f"p_{c}": ("p_SNAP", "mean"), f"n_{c}": ("run_name", "nunique")})
        if (g[f"n_{c}"] != len(tr)).any():
            raise ValueError(f"{c}: an older mixed trace is not predicted by every run")
        ens = g.drop(columns=[f"n_{c}"]) if ens is None else ens.merge(g.drop(columns=[f"n_{c}"]), on=F.COLS, validate="one_to_one")
        y = (o["true_class"] == "SNAP").astype(int)
        per_run[c] = [np.mean([roc_auc_score(y[(o["run_name"] == rn) & (o["dataset"] == ds)],
                                             o.loc[(o["run_name"] == rn) & (o["dataset"] == ds), "p_SNAP"])
                               for ds in sorted(o["dataset"].unique())]) for rn in tr]
    if any(len(cells[c][2][cells[c][2]["true_class"].isin(["HT", "SNAP"])].drop_duplicates(F.COLS)) != len(ens) for c in F.CELLS):
        raise ValueError("the cells cover different older mixed traces")
    ens = ens.reset_index(drop=True)
    y = (ens["true_class"] == "SNAP").astype(int).to_numpy()
    dss = sorted(ens["dataset"].unique())
    dmask = {ds: (ens["dataset"] == ds).to_numpy() for ds in dss}

    def stats(ix: np.ndarray) -> dict:
        out = {}
        for c in F.CELLS:
            p = ens[f"p_{c}"].to_numpy()[ix]
            yy = y[ix]
            per = [roc_auc_score(yy[dmask[ds][ix]], p[dmask[ds][ix]]) for ds in dss]
            out.update({(c, ds): a for ds, a in zip(dss, per)})
            out[(c, "mean of six")] = float(np.mean(per))
        return out

    pt = stats(np.arange(len(ens)))
    groups = boot_groups(ens)
    boot = []
    for _ in range(n_boot):
        ix, _ = rows_of(groups, draw(groups, rng))
        boot.append(stats(ix))
    rows = []
    for (c, u), v in pt.items():
        lo, hi = np.percentile([b[(c, u)] for b in boot], [2.5, 97.5])
        rows.append({"cell": c, "unit": u, "auc": v, "lo": lo, "hi": hi,
                     "n_HT": int(((ens["dataset"] == u) & (y == 0)).sum()) if u in dss else int((y == 0).sum()),
                     "n_SNAP": int(((ens["dataset"] == u) & (y == 1)).sum()) if u in dss else int((y == 1).sum())})
    sd = {c: float(np.std(per_run[c], ddof=1)) for c in F.CELLS}
    prow = []
    for a_, b_ in F.PAIRS:
        for u in dss + ["mean of six"]:
            d = np.array([b[(a_, u)] - b[(b_, u)] for b in boot])
            lo, hi = np.percentile(d, [2.5, 97.5])
            diff = pt[(a_, u)] - pt[(b_, u)]
            row = {"pair": f"{a_} - {b_}", "unit": u, "diff": diff, "lo": lo, "hi": hi}
            if u == "mean of six":
                margin = 2 * max(sd[a_], sd[b_])
                row.update({"seed_sd_a": sd[a_], "seed_sd_b": sd[b_], "margin": margin,
                            "read_as_real": bool((lo > 0 or hi < 0) and abs(diff) > margin)})
            prow.append(row)
    return pd.DataFrame(rows), pd.DataFrame(prow)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/dfk851_b/controls")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    res = F.main(["--out", str(a.out), "--n-boot", str(a.n_boot)] + (["--no-figures"] if a.no_figures else []))
    if not a.no_figures:
        shutil.copy(__file__, a.out / "figures")
    cells = {c: F.load_cell(c, REPO_ROOT / "Results/Revisions/DualColor/dfk851_b/c_transfer") for c in F.CELLS}
    older, older_p = older_sessions(cells, a.n_boot, np.random.default_rng([F.SEED, 18]))
    older.to_csv(a.out / "older.csv", index=False)
    older_p.to_csv(a.out / "older_paired.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(older.pivot(index="unit", columns="cell", values="auc")[list(F.CELLS)].to_string())
        print(older_p[older_p["unit"] == "mean of six"].to_string(index=False))
    # S with P's six seeds (840410, 1-5; the glob [1-58] skips seed 6)
    F.PATTERN["S"] = "size_s[1-58]*.yaml"
    print("\nS with six seeds:")
    F.main(["--out", str(a.out / "six_seeds"), "--n-boot", str(a.n_boot), "--no-figures"])


if __name__ == "__main__":
    main()
