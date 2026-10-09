#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Round 4 summary (TRAINING_NOTES.md point 8): the vesicle-table pool against the paper-IN pool,
five seeds each, from the evaluate.py outputs of the latest run of each name (the table pool's
seed 840410 is r2_all_aug0).

Tables (printed, and written to <out>/r4_*.csv):
- runs     per run: pool, seed, training and validation set sizes (from data_split.csv),
           epochs, validation AUC, mixed-slide AUC pooled and per dataset with FOV-bootstrap
           intervals, kept fraction and balanced accuracy of the kept traces
- paired   per seed: mixed-slide AUC of each pool and the difference (paper-IN minus table),
           pooled and per dataset; then the mean difference over the seeds with a 95 % t
           interval (n - 1 degrees of freedom) from the seed-to-seed spread
- decision the rule of point 8: the paper-IN pool is the default unless the interval of the
           pooled mean difference lies entirely below 0

Usage:
    python Revisions/DualColor/round4_summary.py [--models Results/Revisions/DualColor/models] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = [840410, 1, 2, 3, 4]
DATASETS = ["DFK785", "DFK788", "DFK789"]


def latest(models: Path, name: str) -> Path | None:
    runs = sorted(p for p in models.glob(f"*_{name}") if p.is_dir() and (p / "evaluation" / "metrics.csv").exists())
    return runs[-1] if runs else None


def run_name(pool: str, seed: int) -> str:
    return "r2_all_aug0" if (pool == "table" and seed == 840410) else f"r4_{pool}_s{seed}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/models")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/comparisons")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pool in ["table", "paperin"]:
        for seed in SEEDS:
            run = latest(a.models, run_name(pool, seed))
            if run is None:
                continue
            m = pd.read_csv(run / "evaluation" / "metrics.csv")
            m = m[m["subset"] == "all"].set_index(["set", "dataset"])
            sp = pd.read_csv(run / "data_split.csv", usecols=["set", "true_class"])
            cnt = sp.groupby(["set", "true_class"]).size()
            full = pd.read_json(run / "config_full.json", typ="series")
            r = {"pool": pool, "seed": seed, "run": run.name, "epochs": int(full["epochs_run"]),
                 "train_HT": int(cnt.get(("train", "HT"), 0)), "train_SNAP": int(cnt.get(("train", "SNAP"), 0)),
                 "val_HT": int(cnt.get(("val", "HT"), 0)), "val_SNAP": int(cnt.get(("val", "SNAP"), 0)),
                 "val_auc": m.loc[("val", "pooled"), "auc"]}
            for ds in ["pooled"] + DATASETS:
                x = m.loc[("test_mixed", ds)]
                r[f"mixed_auc_{ds}"] = x["auc"]
                if ds == "pooled":
                    r["mixed_auc_lo"], r["mixed_auc_hi"] = x["auc_lo"], x["auc_hi"]
                    r["mixed_kept"], r["mixed_ba_kept"] = x["kept_frac"], x["kept_balanced_accuracy"]
            rows.append(r)
    runs = pd.DataFrame(rows)
    tables = {"runs": runs}
    if len(runs) and set(runs["pool"]) == {"table", "paperin"}:
        w = runs.pivot(index="seed", columns="pool")
        paired = pd.DataFrame(index=w.index)
        for ds in ["pooled"] + DATASETS:
            c = f"mixed_auc_{ds}"
            paired[f"table_{ds}"] = w[(c, "table")]
            paired[f"paperin_{ds}"] = w[(c, "paperin")]
            paired[f"diff_{ds}"] = w[(c, "paperin")] - w[(c, "table")]
        paired = paired.dropna().reset_index()
        summ = []
        for ds in ["pooled"] + DATASETS:
            d = paired[f"diff_{ds}"].to_numpy()
            n = len(d)
            half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            summ.append({"dataset": ds, "n_seeds": n, "table_mean": paired[f"table_{ds}"].mean(),
                         "paperin_mean": paired[f"paperin_{ds}"].mean(), "mean_diff": d.mean(),
                         "diff_lo": d.mean() - half, "diff_hi": d.mean() + half, "diff_sd": d.std(ddof=1)})
        summ = pd.DataFrame(summ)
        tables["paired"] = paired
        tables["paired_summary"] = summ
        p = summ.set_index("dataset").loc["pooled"]
        verdict = ("table pool kept (interval of the mean difference below 0)" if p["diff_hi"] < 0
                   else "paper-IN pool adopted (not shown to be worse)")
        tables["decision"] = pd.DataFrame([{"n_seeds": int(p["n_seeds"]), "mean_diff": p["mean_diff"],
                                            "interval": f"[{p['diff_lo']:.3f}, {p['diff_hi']:.3f}]",
                                            "decision": verdict}])
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        for k, t in tables.items():
            t.to_csv(a.out / f"r4_{k}.csv", index=False)
            print(f"\n## {k}\n{t.to_string(index=False)}")


if __name__ == "__main__":
    main()
