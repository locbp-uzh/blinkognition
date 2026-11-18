#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Readouts of TRAINING_NOTES.md point 17 (seven-session pool, validated on the older sessions' mixed slides) on the
single-dye traces of DFK851's mixed slide (slide 3): dfk851_ft_summary.py's readouts (point 16) with the cells

  C        r6_pooled_s<seed>, applied unchanged (point 13)
  FT full  ft_full_DFK851_s<seed>, C fine-tuned on DFK851's single-protein slides (point 16)
  P        pool7_valfov_s<seed>: seven sessions' single-protein pools, validation 20 % of their FOVs (as C);
           M and P have six seeds (840410, 1-5; the amendment of point 17), their five-seed readouts in five_seeds/
  M        pool7_valmix_s<seed>: the same pools, all trained on; validation = the older sessions' labeled mixed
           traces (DFK785 slide 4 excluded): checkpoint, Wasserstein threshold, equal-recall cut and the counting
           constants a, b come from mixed slides

and the pairs M - P, P - C, M - C, M - FT full, P - FT full (point 11's rule for the AUC and balanced
accuracies). lod.csv also holds the counting line (count_offset = what an all-HT sample reads with the cell's
validation constants, count_slope; 1 and 0 = unbiased) with FOV-bootstrap intervals. Added here: point 15's model
check for each cell (model_check.csv; the point-16 readout did not run it). retention.csv drops M: its labeled
older mixed traces are its validation set, not a test.

    python Revisions/DualColor/dfk851_pool7_summary.py [--n-boot 2000] [--out <dir>]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dfk851_ft_summary as F  # noqa: E402
import dfk851_lod as L  # noqa: E402
from dc import REPO_ROOT  # noqa: E402

F.CELLS = ("C", "FT full", "P", "M")
F.PATTERN.update({"P": "pool7_valfov_s*.yaml", "M": "pool7_valmix_s*.yaml"})
F.PAIRS = (("M", "P"), ("P", "C"), ("M", "C"), ("M", "FT full"), ("P", "FT full"))
F.COLOR_KEYS = ("orange", "vermillion", "pink", "sky_blue")   # C and FT full as in point 16's figure
F.FIG_NAME = "pool7"
F.CAPTION_LEAD = ("Models trained on the single-protein slides of seven sessions (DFK851 included) and validated on "
                  "held-out single-protein FOVs (P) or on the six older sessions' mixed slides (M), compared with C "
                  "(six other sessions) and C fine-tuned on DFK851's single-protein slides (all weights), on DFK851's "
                  "mixed slide; five models per ensemble for C and FT full, six for P and M (five-seed ensembles of P "
                  "and M: five_seeds/).")


def model_check(e: pd.DataFrame, rng) -> pd.DataFrame:
    """Point 15's check of the variance model on real whole-FOV samples, for each cell's ensemble p."""
    mol, inc, fov, cls = F.molecules(e)
    D = {"fov": fov, "cls": cls, "P": mol[[f"p_{c}" for c in F.CELLS]].to_numpy(), "J": len(inc),
         "members": [[np.flatnonzero((fov == j) & (cls == c)) for c in (0, 1)] for j in range(len(inc))]}
    L.MODELS = F.CELLS
    cnt, S, Q = L.sufficient(D)
    cps = {m: L.components(cnt, S, Q, mi) for mi, m in enumerate(F.CELLS)}
    return L.model_check(D, cps, rng)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/dfk851_b/pool7")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    res = F.main(["--out", str(a.out), "--n-boot", str(a.n_boot)] + (["--no-figures"] if a.no_figures else []))
    ret = pd.read_csv(a.out / "retention.csv")
    ret.drop(columns=[c for c in ("M",) if c in ret]).to_csv(a.out / "retention.csv", index=False)
    if not a.no_figures:   # F.figures copies dfk851_ft_summary.py; this script's cells and pairs are set here
        shutil.copy(__file__, a.out / "figures")
    chk = model_check(res["e"], np.random.default_rng([F.SEED, 17]))
    chk.to_csv(a.out / "model_check.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(chk.groupby(["model", "k"])["pass"].all().unstack().to_string())
    # the amendment of point 17: M and P have six seeds, C and FT full five; the same readouts with M's and P's
    # five pre-registered seeds (840410, 1-4; the glob [1-48] skips seed 5) in <out>/five_seeds
    F.PATTERN.update({"P": "pool7_valfov_s[1-48]*.yaml", "M": "pool7_valmix_s[1-48]*.yaml"})
    print("\nFive-seed ensembles of M and P:")
    F.main(["--out", str(a.out / "five_seeds"), "--n-boot", str(a.n_boot), "--no-figures"])
    r5 = pd.read_csv(a.out / "five_seeds" / "retention.csv")   # as above: M's older mixed traces are its validation
    r5.drop(columns=[c for c in ("M",) if c in r5]).to_csv(a.out / "five_seeds" / "retention.csv", index=False)


if __name__ == "__main__":
    main()
