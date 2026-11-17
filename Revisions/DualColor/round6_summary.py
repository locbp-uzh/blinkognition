#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Round 6 readouts (TRAINING_NOTES.md point 11) from the runs' predictions.csv.

Runs: the latest Results/Revisions/DualColor/models/*_<name> with a predictions.csv for every
config in ml/round6/ (A r6_loeo_<DS>_s<seed>, B r6_within_<DS>_s<seed>, C r6_pooled_s<seed>).
Test traces: set test_mixed with a class (single-dye vesicle within 2 px). Units: each mixed
slide, each session (its mixed slides together), all, and all without DFK785 slide 4.

Per run and unit (runs.csv): AUC of p_SNAP; within-FOV AUC (HT-SNAP pairs of one FOV only, the
per-FOV AUCs weighted by their pair counts); recalls and balanced accuracy at p_SNAP 0.5 (the
argmax); balanced accuracy at the run's equal-recall threshold t_eer (the p_SNAP cut at which HT
and SNAP recall are equal on the run's own balanced validation set); kept fraction and balanced
accuracy of the kept traces (Wasserstein threshold); trained = training loss below 0.65 at some
epoch (TRAINING_NOTES.md point 10).

Per design cell and unit (cells.csv; A: left-out session, B: session, C): mean and SD over all runs
and over the trained runs (TRAINING_NOTES.md point 10); 'primary' marks the design's own test. A run
without a log is reported and counted as trained, as in kfold_summary.py. Ensembles (ensembles.csv): mean p_SNAP over the trained seeds of a cell, on every
unit: AUC and within-FOV AUC with 95 % FOV-cluster bootstrap intervals (FOVs resampled with
replacement within each slide), balanced accuracy at 0.5 and at the median of the members' t_eer.
Main table (main.csv): session x {A, B, C, r5b} on the session's mixed slides; design differences
(paired.csv): C - A, B - A, C - B per session, the two ensembles on the same resampled FOVs, with
the seed SD of both cells (trained runs, that unit) and read_as_real (point 11: the interval
excludes 0 and the difference exceeds twice the larger seed SD). The r5b
column is the round-5b ensemble (15 k-fold models, trained on DFK785/788/789 only): MC-mean
predictions on the old mixed slides, the deterministic transfer check (transfer_check.py scores)
on the new ones; AUC only. B's transfer matrix (transfer_B.csv): ensemble AUC of each session's
B models on every session's mixed slides.

    python Revisions/DualColor/round6_summary.py [--n-boot 2000] [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc import REPO_ROOT  # noqa: E402

SESSIONS = ["DFK785", "DFK788", "DFK789", "DFK842", "DFK843", "DFK846"]
PLATEAU_LOSS = 0.65
NAME = re.compile(r"^r6_(?:(?P<pooled>pooled)|(?P<design>loeo|within)_(?P<ds>DFK\d+))_s(?P<seed>\d+)$")
KEY = ["dataset", "extraction_key", "uniqueID"]


def latest(models: Path, name: str) -> Path | None:
    runs = sorted(p for p in models.glob(f"*_{name}") if p.is_dir() and (p / "predictions.csv").exists())
    return runs[-1] if runs else None


def eer_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """The p_SNAP cut t (calls SNAP at p >= t) with HT and SNAP recall closest to equal; ties: the
    middle of the tied range. Candidate cuts lie between consecutive distinct scores."""
    s = np.unique(p)
    cuts = np.concatenate([[s[0] - 1e-9], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-9]])
    gap = np.array([abs(np.mean(p[y == 0] < t) - np.mean(p[y == 1] >= t)) for t in cuts])
    best = np.flatnonzero(gap == gap.min())
    return float(np.median(cuts[best]))


def within_fov_auc(y: np.ndarray, p: np.ndarray, fov: np.ndarray) -> float:
    num = den = 0.0
    for f in np.unique(fov):
        m = fov == f
        n1 = int(y[m].sum())
        n0 = int(m.sum()) - n1
        if n1 and n0:
            num += roc_auc_score(y[m], p[m]) * n1 * n0
            den += n1 * n0
    return num / den if den else np.nan


def point(y, p, fov, t=0.5) -> dict:
    rec = [np.mean(p[y == 0] < t) if (y == 0).any() else np.nan, np.mean(p[y == 1] >= t) if (y == 1).any() else np.nan]
    return {"auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            "auc_within_fov": within_fov_auc(y, p, fov), "recall_HT": rec[0], "recall_SNAP": rec[1],
            "ba": np.nanmean(rec) if not np.all(np.isnan(rec)) else np.nan}


def units(mixed: pd.DataFrame) -> dict:
    """name -> boolean mask over the mixed rows."""
    out = {}
    for (ds, sl), _ in mixed.groupby(["dataset", "slide"]):
        out[f"{ds} {sl}"] = (mixed["dataset"] == ds) & (mixed["slide"] == sl)
    for ds in mixed["dataset"].unique():
        if mixed.loc[mixed["dataset"] == ds, "slide"].nunique() > 1:
            out[ds] = mixed["dataset"] == ds
    out["all"] = pd.Series(True, index=mixed.index)
    out["all without DFK785 slide4"] = ~((mixed["dataset"] == "DFK785") & (mixed["slide"] == "slide4"))
    return out


def session_unit(ds: str, names: list[str]) -> str:
    return ds if ds in names else next(n for n in names if n.startswith(ds + " "))


def boot_groups(df: pd.DataFrame) -> list[list[np.ndarray]]:
    """Per (dataset, slide), the row positions of df (0 .. len - 1) of each FOV. Labels of the reset
    index, not groupby positions, which count from 0 inside every slide."""
    d = df.reset_index(drop=True)
    return [[np.asarray(ix) for ix in g.groupby("fov").groups.values()] for _, g in d.groupby(["dataset", "slide"])]


def resample(groups, rng) -> tuple[np.ndarray, np.ndarray]:
    """One FOV-cluster bootstrap draw: FOVs with replacement within each slide. Returns the row
    positions and a copy id per row, so that a FOV drawn k times counts as k separate FOVs in the
    within-FOV AUC (merged, its pairs would count k squared times)."""
    rows, ids, k = [], [], 0
    for fovs in groups:
        for i in rng.integers(0, len(fovs), len(fovs)):
            rows.append(fovs[i])
            ids.append(np.full(len(fovs[i]), k))
            k += 1
    return np.concatenate(rows), np.concatenate(ids)


def load(models: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    info, preds = [], []
    for cfg in sorted((HERE / "ml/round6").glob("r6_*.yaml")):
        name = cfg.stem
        m = NAME.match(name)
        run = latest(models, name)
        if run is None:
            print(f"missing: {name}")
            continue
        design = "C" if m["pooled"] else {"loeo": "A", "within": "B"}[m["design"]]
        cell = "pooled" if design == "C" else m["ds"]
        log = (run / f"{name}.log").read_text() if (run / f"{name}.log").exists() else None
        if log is None:
            print(f"{run.name}: no run log; counted as trained (as kfold_summary.py), check by hand")
        losses = [float(x) for x in re.findall(r"^Epoch \d+ \S+ train loss: ([0-9.]+)", log or "", re.M)]
        full = json.load(open(run / "config_full.json"))
        pr = pd.read_csv(run / "predictions.csv", dtype={"slide": str, "fov": str, "true_class": str,
                                                          "uniqueID": str, "extraction_key": str}, low_memory=False)
        pr = pr[pr["true_class"].isin(["HT", "SNAP"])]
        val = pr[pr["set"] == "val"]
        t = eer_threshold((val["true_class"] == "SNAP").to_numpy(int), val["p_SNAP"].to_numpy())
        info.append({"run_name": name, "run": run.name, "design": design, "cell": cell, "seed": int(m["seed"]),
                     "epochs": int(full["epochs_run"]), "best_val_auc": full.get("best_val_auc"),
                     "trained": log is None or (bool(losses) and min(losses) < PLATEAU_LOSS), "t_eer": t,
                     "n_val": len(val), "wasserstein_threshold": float(full["wasserstein_threshold"])})
        mx = pr[pr["set"] == "test_mixed"][KEY + ["slide", "fov", "true_class", "p_SNAP", "kept"]]
        preds.append(mx.assign(run_name=name))
    return pd.DataFrame(info), pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def run_table(info: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in info.itertuples():
        g = preds[preds["run_name"] == r.run_name].reset_index(drop=True)
        for u, m in units(g).items():
            h = g[m.to_numpy()]
            y, p = (h["true_class"] == "SNAP").to_numpy(int), h["p_SNAP"].to_numpy()
            fov = (h["dataset"] + "/" + h["slide"] + "/" + h["fov"]).to_numpy()
            k = h["kept"].astype(bool).to_numpy()
            pt, pe = point(y, p, fov), point(y, p, fov, r.t_eer)
            pk = point(y[k], p[k], fov[k]) if k.any() else {"ba": np.nan}
            rows.append({"run_name": r.run_name, "design": r.design, "cell": r.cell, "seed": r.seed, "trained": r.trained,
                         "unit": u, "n_HT": int((y == 0).sum()), "n_SNAP": int((y == 1).sum()), **pt,
                         "ba_eer": pe["ba"], "recall_HT_eer": pe["recall_HT"], "recall_SNAP_eer": pe["recall_SNAP"],
                         "kept_frac": k.mean(), "ba_kept": pk["ba"]})
    return pd.DataFrame(rows)


def primary_unit(design: str, cell: str, names: list[str]) -> list[str]:
    return [session_unit(cell, names)] if design in "AB" else ["all", "all without DFK785 slide4"]


def ensemble_scores(info: pd.DataFrame, preds: pd.DataFrame) -> dict:
    """(design, cell) -> (mixed rows with the mean p_SNAP of the trained seeds, median member t_eer, n)."""
    out = {}
    for (d, c), g in info[info["trained"]].groupby(["design", "cell"]):
        p = preds[preds["run_name"].isin(g["run_name"])]
        e = (p.groupby(KEY + ["slide", "fov", "true_class"], as_index=False)["p_SNAP"].mean())
        if p.groupby(KEY).size().nunique() != 1:
            raise ValueError(f"{d} {c}: the members predict different mixed traces")
        out[(d, c)] = (e, float(g["t_eer"].median()), len(g))
    return out


def ensemble_table(ens: dict, n_boot: int, rng) -> pd.DataFrame:
    rows = []
    for (d, c), (e, t, n) in ens.items():
        for u, m in units(e).items():
            h = e[m.to_numpy()].reset_index(drop=True)
            y, p = (h["true_class"] == "SNAP").to_numpy(int), h["p_SNAP"].to_numpy()
            fov = (h["dataset"] + "/" + h["slide"] + "/" + h["fov"]).to_numpy()
            pt, pe = point(y, p, fov), point(y, p, fov, t)
            groups, bs = boot_groups(h), []
            for _ in range(n_boot):
                ix, copy_id = resample(groups, rng)
                bs.append((roc_auc_score(y[ix], p[ix]) if len(np.unique(y[ix])) == 2 else np.nan,
                           within_fov_auc(y[ix], p[ix], copy_id)))
            bs = np.array(bs)
            rows.append({"design": d, "cell": c, "n_models": n, "unit": u, "n_HT": int((y == 0).sum()),
                         "n_SNAP": int((y == 1).sum()), "auc": pt["auc"],
                         "auc_lo": np.nanpercentile(bs[:, 0], 2.5), "auc_hi": np.nanpercentile(bs[:, 0], 97.5),
                         "auc_within_fov": pt["auc_within_fov"], "auc_within_fov_lo": np.nanpercentile(bs[:, 1], 2.5),
                         "auc_within_fov_hi": np.nanpercentile(bs[:, 1], 97.5), "ba": pt["ba"], "t_eer": t,
                         "ba_eer": pe["ba"], "recall_HT_eer": pe["recall_HT"], "recall_SNAP_eer": pe["recall_SNAP"]})
    return pd.DataFrame(rows)


def r5b_reference(models: Path) -> pd.DataFrame | None:
    """Round-5b ensemble scores on all eight mixed slides (old: MC mean of the 15 runs; new: transfer check)."""
    parts = []
    for p in sorted(models.glob("*_r5b_kfold_p*_f*")):
        if (p / "predictions.csv").exists():
            pr = pd.read_csv(p / "predictions.csv", dtype={"slide": str, "fov": str, "true_class": str, "uniqueID": str,
                                                            "extraction_key": str}, low_memory=False)
            parts.append(pr[(pr["set"] == "test_mixed") & pr["true_class"].isin(["HT", "SNAP"])]
                         [KEY + ["slide", "fov", "true_class", "p_SNAP"]])
    if not parts:
        return None
    old = pd.concat(parts).groupby(KEY + ["slide", "fov", "true_class"], as_index=False)["p_SNAP"].mean()
    new = []
    chk = REPO_ROOT / "Results/Revisions/DualColor/checks/20261116_new_sessions_first_look"
    for ds in ("DFK842", "DFK843", "DFK846"):
        f = chk / f"transfer_{ds}" / "scores.csv"
        if f.exists():
            s = pd.read_csv(f, dtype={"slide": str, "fov": str, "true_class": str, "uniqueID": str, "extraction_key": str})
            new.append(s)
    if new:       # long format, one row per (trace, model): the ensemble is the mean over the 15 models
        s = pd.concat(new)
        s = s[(s["slide_type"] == "mixed") & s["true_class"].isin(["HT", "SNAP"])]
        if s.groupby(KEY)["model"].nunique().nunique() != 1:
            raise ValueError("transfer scores: traces scored by different numbers of models")
        s = s.groupby(KEY + ["slide", "fov", "true_class"], as_index=False)["p_SNAP"].mean()
        old = pd.concat([old, s], ignore_index=True)
    return old


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--models", default=str(REPO_ROOT / "Results/Revisions/DualColor/models"))
    ap.add_argument("--out", default=str(REPO_ROOT / "Results/Revisions/DualColor/round6"))
    a = ap.parse_args()
    rng = np.random.default_rng(20261117)
    models, out = Path(a.models), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info, preds = load(models)
    info.to_csv(out / "run_info.csv", index=False)
    print(info.groupby("design")[["trained"]].agg(["sum", "count"]).to_string())
    runs = run_table(info, preds)
    runs.to_csv(out / "runs.csv", index=False)
    names = list(units(preds[preds["run_name"] == info["run_name"].iloc[0]]).keys())
    cells = []
    for (d, c), g in runs.groupby(["design", "cell"]):
        for u in sorted(g["unit"].unique()):
            for subset, h in (("all runs", g[g["unit"] == u]), ("trained runs", g[(g["unit"] == u) & g["trained"]])):
                cells.append({"design": d, "cell": c, "unit": u, "primary": u in primary_unit(d, c, names),
                              "runs": subset, "n": len(h),
                              **{f"{k}_{s}": getattr(h[k], s)() for k in ("auc", "auc_within_fov", "ba", "ba_eer", "ba_kept")
                                 for s in ("mean", "std")}})
    cells = pd.DataFrame(cells)
    cells.to_csv(out / "cells.csv", index=False)

    ens = ensemble_scores(info, preds)
    et = ensemble_table(ens, a.n_boot, rng)
    et.to_csv(out / "ensembles.csv", index=False)
    ref = r5b_reference(models)
    if ref is not None:
        ens[("r5b", "old sessions")] = (ref, np.nan, 15)

    # Main table and paired differences, per session on its own mixed slides
    main_rows, paired = [], []
    for ds in SESSIONS:
        u = session_unit(ds, names)
        row = {"session": ds, "unit": u}
        cand = {"A": ("A", ds), "B": ("B", ds), "C": ("C", "pooled"), "r5b": ("r5b", "old sessions")}
        sub = {}
        for col, key in cand.items():
            if key not in ens:
                continue
            e = ens[key][0]
            h = e[units(e)[u].to_numpy()].reset_index(drop=True) if u in units(e) else None
            if h is None or h.empty:
                continue
            sub[col] = h
            y = (h["true_class"] == "SNAP").to_numpy(int)
            row[f"{col}_auc"] = roc_auc_score(y, h["p_SNAP"])
            if col != "r5b":
                r = et[(et["design"] == key[0]) & (et["cell"] == key[1]) & (et["unit"] == u)].iloc[0]
                row[f"{col}_auc_lo"], row[f"{col}_auc_hi"], row[f"{col}_ba_eer"] = r["auc_lo"], r["auc_hi"], r["ba_eer"]
        main_rows.append(row)
        for x, z in (("C", "A"), ("B", "A"), ("C", "B")):
            if x in sub and z in sub:
                m = sub[x].merge(sub[z], on=KEY + ["slide", "fov", "true_class"], suffixes=("_x", "_z"), validate="one_to_one")
                if len(m) != len(sub[x]) or len(m) != len(sub[z]):
                    raise ValueError(f"{ds}: {x} and {z} ensembles cover different mixed traces")
                y = (m["true_class"] == "SNAP").to_numpy(int)
                d0 = roc_auc_score(y, m["p_SNAP_x"]) - roc_auc_score(y, m["p_SNAP_z"])
                groups, bs = boot_groups(m), []
                for _ in range(a.n_boot):
                    ix, _ = resample(groups, rng)
                    if len(np.unique(y[ix])) == 2:
                        bs.append(roc_auc_score(y[ix], m["p_SNAP_x"].to_numpy()[ix]) -
                                  roc_auc_score(y[ix], m["p_SNAP_z"].to_numpy()[ix]))
                sd = {}
                for col in (x, z):
                    dd, cc = cand[col]
                    r = cells[(cells["design"] == dd) & (cells["cell"] == cc) & (cells["unit"] == u)
                              & (cells["runs"] == "trained runs")]
                    sd[col] = float(r["auc_std"].iloc[0]) if len(r) else np.nan
                lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
                big = max(sd[x], sd[z]) if np.isfinite(sd[x]) and np.isfinite(sd[z]) else np.nan
                paired.append({"session": ds, "unit": u, "diff": f"{x} - {z}", "auc_diff": d0, "lo": lo, "hi": hi,
                               "seed_sd_x": sd[x], "seed_sd_z": sd[z],
                               # point 11: real only if the interval excludes 0 and |diff| > 2 x the larger seed SD;
                               # undefined (None) when a cell has fewer than two trained seeds
                               "read_as_real": (bool((lo > 0 or hi < 0) and abs(d0) > 2 * big)
                                                if np.isfinite(big) else None)})
    mt, pt = pd.DataFrame(main_rows), pd.DataFrame(paired)
    mt.to_csv(out / "main.csv", index=False)
    pt.to_csv(out / "paired.csv", index=False)

    # B transfer matrix: each session's within-session ensemble on every session's mixed slides
    tm = et[et["design"] == "B"]
    tm = tm[tm["unit"].isin([session_unit(ds, names) for ds in SESSIONS])]
    tm = tm.assign(test=tm["unit"].str.split(" ").str[0]).pivot(index="cell", columns="test", values="auc")
    tm.to_csv(out / "transfer_B.csv")

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print("\nPrimary tests, mean (SD) over seeds:\n" + cells[cells["primary"]][
            ["design", "cell", "unit", "runs", "n", "auc_mean", "auc_std", "ba_eer_mean", "ba_eer_std"]].to_string(index=False))
        print("\nMain table (ensembles on each session's mixed slides):\n" + mt.to_string(index=False))
        print("\nPaired AUC differences (same mixed traces, FOV bootstrap):\n" + pt.to_string(index=False))
        print("\nB transfer matrix (rows: training session, columns: mixed slides):\n" + tm.to_string())
    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
