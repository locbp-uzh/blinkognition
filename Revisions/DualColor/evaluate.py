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

auc_within_dataset: AUC over HT-SNAP pairs of the same dataset only; the pooled AUC also ranks
pairs across datasets, so a score that only tells the datasets apart moves it when the class
mix differs by dataset (it does: the training set is 64 / 32 / 40 % SNAP).

Background-trained models (train_pure.py trace_source: background) have no vesicle: their sets
(val, val_unused) get no subsets; slide_pairs.csv gives the AUC between every two single-protein
slides of a dataset on the held-out background (val_unused), same-protein pairs as the reference.

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
    if "vesicle_id" not in pred.columns:      # background traces (trace_source: background): no vesicle
        return pred.assign(clean=False, confirmed_515=False)
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


def within_dataset_auc(y: np.ndarray, p: np.ndarray, ds: np.ndarray) -> float:
    """AUC over the HT-SNAP pairs of the same dataset only (the per-dataset AUCs weighted by their
    pair counts): a score that only tells the datasets apart gets 0.5, whatever the class mix."""
    num = den = 0.0
    for d in np.unique(ds):
        m = ds == d
        n1 = int(y[m].sum())
        n0 = int(m.sum()) - n1
        if n1 and n0:
            num += roc_auc_score(y[m], p[m]) * n1 * n0
            den += n1 * n0
    return num / den if den else np.nan


def slide_pairs(pred: pd.DataFrame, score: str = "p_SNAP", n_boot: int = 2000,
                rng: np.random.Generator | None = None) -> pd.DataFrame:
    """AUC of `score` between every two single-protein slides of a dataset, with 95 % intervals from
    a FOV-cluster bootstrap within each slide.

    Cross-protein pairs are oriented HT slide (a) vs SNAP slide (b): AUC > 0.5 = higher score on the
    SNAP slide. Same-protein pairs (a = the earlier slide) measure how much two slides differ without
    a protein difference, the reference for the cross-protein ones. separation = |AUC - 0.5|.
    """
    rng = rng or np.random.default_rng(840410)
    lab = pred[pred["true_class"].isin(["HT", "SNAP"])]
    rows = []
    for ds, g in lab.groupby("dataset"):
        cls = g.groupby("slide")["true_class"].first()
        slides = sorted(cls.index)
        for i, s1 in enumerate(slides):
            for s2 in slides[i + 1:]:
                a, b = (s2, s1) if (cls[s1], cls[s2]) == ("SNAP", "HT") else (s1, s2)
                ga, gb = g[g["slide"] == a], g[g["slide"] == b]
                sa, sb = ga[score].to_numpy(), gb[score].to_numpy()
                fa = {f: np.flatnonzero(ga["fov"].to_numpy() == f) for f in ga["fov"].unique()}
                fb = {f: np.flatnonzero(gb["fov"].to_numpy() == f) for f in gb["fov"].unique()}

                def auc(xa, xb):
                    return roc_auc_score(np.r_[np.zeros(len(xa)), np.ones(len(xb))], np.r_[xa, xb])

                boot = []
                ka, kb = list(fa), list(fb)
                for _ in range(n_boot):
                    ia = np.concatenate([fa[ka[k]] for k in rng.integers(0, len(ka), len(ka))])
                    ib = np.concatenate([fb[kb[k]] for k in rng.integers(0, len(kb), len(kb))])
                    boot.append(auc(sa[ia], sb[ib]))
                v = auc(sa, sb)
                rows.append({"dataset": ds, "slide_a": a, "class_a": cls[a], "slide_b": b, "class_b": cls[b],
                             "pair": "same protein" if cls[a] == cls[b] else "HT vs SNAP",
                             "n_a": len(sa), "n_b": len(sb), "fovs_a": len(ka), "fovs_b": len(kb),
                             "auc": v, "auc_lo": np.percentile(boot, 2.5), "auc_hi": np.percentile(boot, 97.5),
                             "separation": abs(v - 0.5)})
    return pd.DataFrame(rows)


def metrics(g: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    y = (g["true_class"] == "SNAP").astype(int).to_numpy()
    dsa = g["dataset"].to_numpy()
    p = g["p_SNAP"].to_numpy()
    pred = (g["prediction"] == "SNAP").astype(int).to_numpy()
    kept = g["kept"].astype(bool).to_numpy()

    def stats(idx):
        yy, pp, pr, kk = y[idx], p[idx], pred[idx], kept[idx]
        out = {}
        out["auc"] = roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan
        out["auc_within_dataset"] = (out["auc"] if len(np.unique(dsa[idx])) == 1
                                     else within_dataset_auc(yy, pp, dsa[idx]))
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
    if "vesicle_id" not in pred.columns:      # background-trained model: slide pairs on held-out background
        held = pred[pred["set"] == "val_unused"]
        if len(held):
            sp = slide_pairs(held, "p_SNAP", a.n_boot, rng)
            sp.to_csv(out / "slide_pairs.csv", index=False)
            with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
                print("Slide pairs (val_unused background):\n" + sp.to_string(index=False))
    cols = ["set", "subset", "dataset", "n_HT", "n_SNAP", "n_fovs", "auc", "auc_lo", "auc_hi", "auc_within_dataset",
            "balanced_accuracy",
            "balanced_accuracy_lo", "balanced_accuracy_hi", "recall_HT", "recall_SNAP", "kept_frac",
            "kept_balanced_accuracy"]
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        txt = m[cols].to_string(index=False)
    print(txt)
    (out / "metrics.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
