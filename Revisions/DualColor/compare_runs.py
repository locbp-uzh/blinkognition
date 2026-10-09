#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Compare train_pure.py models from their evaluation/metrics.csv (evaluate.py) and apply the
selection rule fixed in TRAINING_NOTES.md: within each setup, the run with the highest
validation AUC (MC mean, pooled validation set) is the chosen one. The test numbers of all
runs are shown; the choice never looks at them.

Setup and factor are parsed from the run name r2_<setup>_aug<factor>.

Usage:
    python Revisions/DualColor/compare_runs.py "Results/Revisions/DualColor/models/*_r2_*" [--out file.csv]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def ci(r: pd.Series, k: str) -> str:
    return f"{r[k]:.2f} [{r[k + '_lo']:.2f}-{r[k + '_hi']:.2f}]" if pd.notna(r.get(k)) else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = []
    for run in sorted(glob.glob(str(REPO_ROOT / a.pattern))):
        run = Path(run)
        mfile = run / "evaluation" / "metrics.csv"
        if not mfile.exists():
            continue
        cfg = json.load(open(run / "config_full.json"))
        name = cfg["config"]["run_name"]
        m = re.match(r"r\d+_(.+)_aug(\d+)$", name)
        setup, factor = (m.group(1), int(m.group(2))) if m else (name, None)
        met = pd.read_csv(mfile)
        pooled = met[(met["dataset"] == "pooled") & (met["subset"] == "all")].set_index("set")
        row = {"setup": setup, "aug_factor": factor, "run": run.name, "epochs": cfg["epochs_run"],
               "threshold": round(cfg["wasserstein_threshold"], 3)}
        for s, tag in (("val", "val"), ("test_mixed", "mixed"), ("holdout_pure", "holdout")):
            if s in pooled.index:
                r = pooled.loc[s]
                row[f"{tag}_auc"] = r["auc"]
                row[f"{tag}_AUC"] = ci(r, "auc")
                row[f"{tag}_BA"] = ci(r, "balanced_accuracy")
                row[f"{tag}_kept"] = round(r["kept_frac"], 2)
                row[f"{tag}_BA_kept"] = ci(r, "kept_balanced_accuracy")
        rows.append(row)
    d = pd.DataFrame(rows).sort_values(["setup", "aug_factor"])
    d["chosen_on_val"] = d.groupby("setup")["val_auc"].transform("max") == d["val_auc"]
    if a.out:
        d.to_csv(a.out, index=False)
    cols = [c for c in ["setup", "aug_factor", "epochs", "chosen_on_val", "val_AUC", "mixed_AUC", "mixed_BA",
                        "mixed_kept", "mixed_BA_kept", "holdout_AUC", "holdout_BA", "holdout_kept",
                        "holdout_BA_kept"] if c in d]
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(d[cols].to_string(index=False))


if __name__ == "__main__":
    main()
