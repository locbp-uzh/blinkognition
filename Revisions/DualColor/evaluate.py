#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Metrics of a train_pure.py model on its validation, mixed-slide test and holdout traces.

Per set and subset (pooled and per dataset):
    n, AUC of p_SNAP (threshold-free ranking), balanced accuracy and recall per class at the
    argmax, kept fraction at the model's fixed Wasserstein threshold, balanced accuracy of the
    kept traces. Intervals: 95 % percentile cluster bootstrap over FOVs (FOVs resampled with
    replacement within each set, n_boot times), since traces of one FOV are not independent.

Mixed-slide subsets (labels from the vesicle; truth only for single-dye vesicles within 2 px):
    all          every trace classed IN_ATTO390 / IN_ATTO520
    clean        the vesicle has no other vesicle within occupancy.crowding_radius_px (5 px)
    515-confirmed  SNAP: the vesicle's 515/488 ratio lies in label_check.snap_515_488 and its
                 488 SNR >= label_check.min_z488 (real ATTO520 vesicles: about 3.0; objects
                 mislabeled ATTO520 on the HT slides: 0.2-0.4); HT: the vesicle's 515 SNR is
                 below label_check.ht_max_z515 (no ATTO520 contribution)

Outputs: <model run>/evaluation/metrics.csv, predictions_annotated.csv, log.

Usage:
    python Revisions/DualColor/evaluate.py Results/Revisions/DualColor/models/<run> [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc import REPO_ROOT, load_config, run_dir  # noqa: E402

LABEL_CHECK = {"snap_515_488": [2.0, 4.5], "min_z488": 10.0, "ht_max_z515": 5.0}


def annotate(pred: pd.DataFrame, crowd_px: float, check: dict) -> pd.DataFrame:
    """Join each trace's vesicle (labels.csv of the dataset's unmixing run) for the subsets."""
    parts = []
    for ds, g in pred.groupby("dataset"):
        cfg = load_config(ds)
        lab = pd.read_csv(run_dir(cfg, "unmixing") / "labels.csv", dtype={"slide": str, "fov": str})
        cols = ["slide", "fov", "vesicle_id", "nn_dist_px", "z_488", "z_515", "F_488", "F_515"]
        g = g.merge(lab[cols], on=["slide", "fov", "vesicle_id"], how="left")
        parts.append(g)
    p = pd.concat(parts, ignore_index=True)
    p["clean"] = p["nn_dist_px"] >= crowd_px
    ratio = p["F_515"] / p["F_488"]
    lo, hi = check["snap_515_488"]
    snap_ok = (p["vesicle_label"] == "ATTO520") & ratio.between(lo, hi) & (p["z_488"] >= check["min_z488"])
    ht_ok = (p["vesicle_label"] == "ATTO390") & (p["z_515"] < check["ht_max_z515"])
    p["confirmed_515"] = snap_ok | ht_ok
    return p


def metrics(g: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    y = (g["true_class"] == "SNAP").astype(int).to_numpy()
    p = g["p_SNAP"].to_numpy()
    pred = (g["prediction"] == "SNAP").astype(int).to_numpy()
    kept = g["kept"].astype(bool).to_numpy()

    def stats(idx):
        yy, pp, pr, kk = y[idx], p[idx], pred[idx], kept[idx]
        out = {}
        out["auc"] = roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan
        rec = [np.mean(pr[yy == c] == c) if (yy == c).any() else np.nan for c in (0, 1)]
        out["recall_HT"], out["recall_SNAP"] = rec
        out["balanced_accuracy"] = np.nanmean(rec) if not np.all(np.isnan(rec)) else np.nan
        out["kept_frac"] = kk.mean() if len(kk) else np.nan
        rk = [np.mean(pr[kk & (yy == c)] == c) if (kk & (yy == c)).any() else np.nan for c in (0, 1)]
        out["kept_recall_HT"], out["kept_recall_SNAP"] = rk
        out["kept_balanced_accuracy"] = np.nanmean(rk) if not np.all(np.isnan(rk)) else np.nan
        return out

    point = stats(np.arange(len(g)))
    fov = (g["dataset"] + "/" + g["slide"] + "/" + g["fov"]).to_numpy()
    groups = {f: np.flatnonzero(fov == f) for f in np.unique(fov)}
    keys = list(groups)
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        boot.append(stats(np.concatenate([groups[keys[k]] for k in pick])))
    boot = pd.DataFrame(boot)
    row = {"n": len(g), "n_HT": int((y == 0).sum()), "n_SNAP": int((y == 1).sum()), "n_fovs": len(keys)}
    for k, v in point.items():
        row[k] = v
        row[f"{k}_lo"], row[f"{k}_hi"] = (np.nanpercentile(boot[k], [2.5, 97.5]) if boot[k].notna().any()
                                          else (np.nan, np.nan))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_run", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=840410)
    a = ap.parse_args()
    run = a.model_run if a.model_run.is_absolute() else REPO_ROOT / a.model_run
    out = run / "evaluation"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)
    base = yaml.safe_load(open(HERE / "base.yaml"))
    pred = pd.read_csv(run / "predictions.csv", dtype={"slide": str, "fov": str, "true_class": str})
    pred = annotate(pred, float(base["occupancy"]["crowding_radius_px"]), LABEL_CHECK)
    pred.to_csv(out / "predictions_annotated.csv", index=False)

    rows = []
    lab = pred[pred["true_class"].isin(["HT", "SNAP"])]
    for s, gs in lab.groupby("set"):
        subsets = {"all": gs}
        if s == "test_mixed":
            subsets |= {"clean": gs[gs["clean"]], "515-confirmed": gs[gs["confirmed_515"]],
                        "clean and 515-confirmed": gs[gs["clean"] & gs["confirmed_515"]]}
        for sub, g in subsets.items():
            if len(g):
                rows.append({"set": s, "subset": sub, "dataset": "pooled", **metrics(g, a.n_boot, rng)})
                for ds, gd in g.groupby("dataset"):
                    rows.append({"set": s, "subset": sub, "dataset": ds, **metrics(gd, a.n_boot, rng)})
    m = pd.DataFrame(rows)
    m.to_csv(out / "metrics.csv", index=False)
    cols = ["set", "subset", "dataset", "n_HT", "n_SNAP", "n_fovs", "auc", "auc_lo", "auc_hi", "balanced_accuracy",
            "balanced_accuracy_lo", "balanced_accuracy_hi", "recall_HT", "recall_SNAP", "kept_frac",
            "kept_balanced_accuracy"]
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        txt = m[cols].to_string(index=False)
    print(txt)
    (out / "metrics.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
