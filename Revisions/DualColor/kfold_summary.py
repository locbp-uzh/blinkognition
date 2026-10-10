#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Round 5 summary (TRAINING_NOTES.md point 9): the FOV-grouped repeated k-fold on the paper-IN
pool, from the evaluate.py outputs of the latest run of each name r5_kfold_p<seed>_f<fold>.

Tables (printed, and written to <out>/r5_*.csv):
- runs          per run: repeat, fold, epochs, set sizes, threshold; AUC on its test fold
                (oof_auc_within: within datasets at the test fold's own HT x SNAP pair
                weights, DFK785 about 0.8; oof_auc_mixed_weights: the per-dataset AUCs at the
                mixed set's pair weights, the counterpart of mixed_auc_within) and on the mixed
                slides (pooled, within datasets, per dataset); mixed balanced accuracy all /
                kept and kept fraction
- oof           per repeat, the five test folds together (checked: every trace of the pool in
                data_split.csv exactly once, and nothing else):
                evaluate.py metrics (AUC pooled and within datasets, balanced accuracy all /
                kept, kept fraction, 95 % FOV-cluster bootstrap intervals), pooled and per
                dataset; each trace scored by the model that did not see its FOV, at that
                model's threshold
- oof_slide_pairs per repeat: AUC between every two single-protein slides of a dataset on the
                out-of-fold scores (HT vs SNAP slides; same-protein pairs as the reference)
- mixed_summary per mixed-slide quantity: mean and SD over the runs (the same 412 traces, so
                training variance alone), and the SD of the 3 repeat means (2 degrees of
                freedom: reported, not read)
- ensembles     mean p_SNAP over the five fold models of a repeat, and over all runs: mixed
                AUC and balanced accuracy at the argmax, with FOV-cluster bootstrap intervals
- gap           the held-out-FOV vs mixed-slide gap (TRAINING_NOTES.md point 9c): per dataset,
                per mixed slide and overall, the mean over the models of d = (AUC on the model's
                test fold, per dataset) - (AUC on the mixed slides of that dataset, or on one
                mixed slide), the same model on both sides. Overall = the per-dataset means
                weighted by the mixed set's HT x SNAP pair counts (auc_test_fold_mean and
                auc_mixed_mean of that row at the same weights). Sensitivity row: the overall d
                without the mixed slides in SENSITIVITY_DROP (weights recomputed). Interval: 95 %
                FOV-cluster bootstrap with the models fixed, pool FOVs and mixed FOVs resampled
                with replacement within their slide, the same draw for every model (a FOV's
                multiplicity enters the AUCs as a sample weight). sd_over_models holds training
                variance and the differences between the models' test folds;
                sd_of_repeat_means (2 degrees of freedom) is reported, not read
- gap_runs      per model and unit (dataset or mixed slide): the two AUCs and d

Usage (after evaluate.py on every run):
    python Revisions/DualColor/kfold_summary.py [--models DIR] [--out DIR] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc import REPO_ROOT  # noqa: E402
from evaluate import metrics, slide_pairs  # noqa: E402

PARTITION_SEEDS = [840410, 1, 2]
N_FOLDS = 5
DATASETS = ["DFK785", "DFK788", "DFK789"]
COLS = ["dataset", "slide", "fov", "extraction_key", "uniqueID", "true_class", "p_SNAP", "prediction", "kept"]
KEY = ["dataset", "extraction_key", "uniqueID"]
POOL_SETS = ["train", "val", "val_unused", "test_oof"]
# Declared in point 9: the gap is also reported without this mixed slide (at chance for every model so far:
# round-4 AUC 0.44-0.55, against about 0.74 on DFK785 slide 3)
SENSITIVITY_DROP = ["DFK785 slide4"]


def latest(models: Path, name: str) -> Path | None:
    runs = sorted(p for p in models.glob(f"*_{name}")
                  if p.is_dir() and (p / "evaluation" / "predictions_annotated.csv").exists())
    return runs[-1] if runs else None


def load_runs(models: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per-run info, the labeled test_oof + test_mixed predictions of every run, and the pool keys
    (data_split.csv) per repeat."""
    info, preds, pools = [], [], {}
    for p in PARTITION_SEEDS:
        for k in range(N_FOLDS):
            name = f"r5_kfold_p{p}_f{k}"
            run = latest(models, name)
            if run is None:
                print(f"missing: {name}")
                continue
            full = json.load(open(run / "config_full.json"))
            kf = full["config"]["kfold"]
            if (int(kf["partition_seed"]), int(kf["fold"]), int(kf["n_folds"])) != (p, k, N_FOLDS):
                raise ValueError(f"{run}: kfold block {kf} does not match its name")
            split = pd.read_csv(run / "data_split.csv", usecols=KEY + ["set"], dtype={"uniqueID": str})
            pool = set(map(tuple, split.loc[split["set"].isin(POOL_SETS), KEY].values))
            if pools.setdefault(p, pool) != pool:
                raise ValueError(f"{run}: training pool differs from the other runs of repeat {p}")
            sp = split["set"].value_counts()
            info.append({"repeat": p, "fold": k, "run": run.name, "epochs": int(full["epochs_run"]),
                         "threshold": float(full["wasserstein_threshold"]),
                         **{f"n_{s}": int(sp.get(s, 0)) for s in ("train", "val", "val_unused", "test_oof")}})
            pr = pd.read_csv(run / "evaluation" / "predictions_annotated.csv",
                             dtype={"slide": str, "fov": str, "true_class": str, "uniqueID": str}, low_memory=False)
            pr = pr[pr["set"].isin(["test_oof", "test_mixed"]) & pr["true_class"].isin(["HT", "SNAP"])]
            preds.append(pr[["set"] + COLS].assign(repeat=p, fold=k))
    return pd.DataFrame(info), pd.concat(preds, ignore_index=True), pools


def mixed_weights(pred: pd.DataFrame) -> pd.Series:
    """Per dataset: the share of the mixed set's HT x SNAP pairs (each mixed trace once)."""
    mix = pred[pred["set"] == "test_mixed"].drop_duplicates(KEY)
    pc = mix.groupby("dataset")["true_class"].agg(lambda t: int((t == "SNAP").sum()) * int((t == "HT").sum()))
    return pc / pc.sum()


def run_metrics(models: Path, info: pd.DataFrame, w: pd.Series) -> pd.DataFrame:
    rows = []
    for _, r in info.iterrows():
        m = pd.read_csv(models / r["run"] / "evaluation" / "metrics.csv")
        m = m[m["subset"] == "all"].set_index(["set", "dataset"])
        x = {"repeat": r["repeat"], "fold": r["fold"],
             "oof_auc_within": m.loc[("test_oof", "pooled"), "auc_within_dataset"]}
        mp = m.loc[("test_mixed", "pooled")]
        x |= {"mixed_auc": mp["auc"], "mixed_auc_within": mp["auc_within_dataset"],
              "mixed_ba_all": mp["balanced_accuracy"], "mixed_kept_frac": mp["kept_frac"],
              "mixed_ba_kept": mp["kept_balanced_accuracy"]}
        for ds in DATASETS:
            x[f"oof_auc_{ds}"] = m.loc[("test_oof", ds), "auc"]
            x[f"mixed_auc_{ds}"] = m.loc[("test_mixed", ds), "auc"]
        x["oof_auc_mixed_weights"] = sum(w[ds] * x[f"oof_auc_{ds}"] for ds in w.index)
        rows.append(x)
    return info.merge(pd.DataFrame(rows), on=["repeat", "fold"])


def oof_tables(pred: pd.DataFrame, pools: dict, n_boot: int, rng: np.random.Generator
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pairs = [], []
    for p, g in pred[pred["set"] == "test_oof"].groupby("repeat"):
        dup = g.duplicated(KEY).sum()
        keys = set(map(tuple, g[KEY].values))
        if dup or g["fold"].nunique() != N_FOLDS or keys != pools[p]:
            raise ValueError(f"repeat {p}: {dup} traces in more than one test fold, {g['fold'].nunique()} folds, "
                             f"{len(pools[p] - keys)} pool traces in no test fold, {len(keys - pools[p])} not in the pool")
        rows.append({"repeat": p, "dataset": "pooled", **metrics(g, n_boot, rng)})
        for ds, gd in g.groupby("dataset"):
            rows.append({"repeat": p, "dataset": ds, **metrics(gd, n_boot, rng)})
        pairs.append(slide_pairs(g, "p_SNAP", n_boot, rng).assign(repeat=p))
    return pd.DataFrame(rows), pd.concat(pairs, ignore_index=True)


def ensembles(pred: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    mix = pred[pred["set"] == "test_mixed"]
    key = ["dataset", "slide", "fov", "extraction_key", "uniqueID", "true_class"]
    rows = []
    for label, g in [(f"repeat {p}", g) for p, g in mix.groupby("repeat")] + [("all runs", mix)]:
        e = g.groupby(key, as_index=False).agg(p_SNAP=("p_SNAP", "mean"), n_models=("p_SNAP", "size"))
        e["prediction"] = np.where(e["p_SNAP"] > 0.5, "SNAP", "HT")
        e["kept"] = False
        for ds, gd in [("pooled", e)] + list(e.groupby("dataset")):
            m = metrics(gd, n_boot, rng)
            rows.append({"ensemble": label, "n_models": int(e["n_models"].min()), "dataset": ds,
                         **{c: m[c] for c in ("n", "n_fovs", "auc", "auc_lo", "auc_hi", "auc_within_dataset",
                                               "auc_within_dataset_lo", "auc_within_dataset_hi", "recall_HT",
                                               "recall_SNAP", "balanced_accuracy", "balanced_accuracy_lo",
                                               "balanced_accuracy_hi")}})
    return pd.DataFrame(rows)


def mixed_summary(runs: pd.DataFrame) -> pd.DataFrame:
    q = ["mixed_auc", "mixed_auc_within"] + [f"mixed_auc_{d}" for d in DATASETS] + \
        ["mixed_ba_all", "mixed_kept_frac", "mixed_ba_kept", "oof_auc_within", "oof_auc_mixed_weights"]
    rep = runs.groupby("repeat")[q].mean()
    return pd.DataFrame({"quantity": q, "n_runs": len(runs), "mean": runs[q].mean().to_numpy(),
                         "sd_over_runs": runs[q].std(ddof=1).to_numpy(), "min": runs[q].min().to_numpy(),
                         "max": runs[q].max().to_numpy(), "sd_of_repeat_means": rep.std(ddof=1).to_numpy()})


def gap(pred: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """d per model and unit (test-fold AUC of the dataset - mixed AUC of the dataset or of one mixed
    slide), its means and their FOV-cluster bootstrap."""
    pred = pred.assign(fov_key=pred["dataset"] + "/" + pred["slide"] + "/" + pred["fov"],
                       side=np.where(pred["set"] == "test_oof", "pool", "mixed"),
                       y=(pred["true_class"] == "SNAP").astype(int))
    w = mixed_weights(pred).reindex(DATASETS)
    # the FOVs to resample: per side and slide (the pool FOVs are the same in every repeat)
    fovs = pred.drop_duplicates(["side", "fov_key"])[["side", "dataset", "slide", "fov_key"]].reset_index(drop=True)
    code = pd.Series(fovs.index, index=fovs["fov_key"] + "|" + fovs["side"])
    strata = [g.index.to_numpy() for _, g in fovs.groupby(["side", "dataset", "slide"])]
    pred = pred.assign(fov_code=code.loc[pred["fov_key"] + "|" + pred["side"]].to_numpy())
    mixed = pred["side"] == "mixed"
    groups = ([((p, k, ds, side), g) for (p, k, ds, side), g in pred.groupby(["repeat", "fold", "dataset", "side"])]
              + [((p, k, f"{ds} {sl}", "mixed"), g)
                 for (p, k, ds, sl), g in pred[mixed].groupby(["repeat", "fold", "dataset", "slide"])])
    cells = [(key, g["fov_code"].to_numpy(), g["y"].to_numpy(), g["p_SNAP"].to_numpy()) for key, g in groups]
    units = DATASETS + sorted(u for u in {key[2] for key, *_ in cells} if " " in u)

    def per_model(mult: np.ndarray | None) -> pd.DataFrame:
        auc = {}
        for key, fc, y, s in cells:
            if mult is None:
                auc[key] = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
                continue
            sw = mult[fc]
            ok = sw > 0
            auc[key] = roc_auc_score(y[ok], s[ok], sample_weight=sw[ok]) if len(np.unique(y[ok])) == 2 else np.nan
        a = pd.Series(auc).rename_axis(["repeat", "fold", "unit", "side"]).unstack("side")
        pool = a["pool"].dropna()
        a["pool"] = [pool.get((r, k, u.split(" ")[0]), np.nan) for r, k, u in a.index]
        return a.assign(d=a["pool"] - a["mixed"]).reset_index()

    def summarize(pm: pd.DataFrame) -> pd.Series:
        m = pm.groupby("unit")["d"].mean().reindex(units)
        md = m.reindex(DATASETS)
        return pd.concat([m, pd.Series({"overall": float((md * w).sum()) if md.notna().all() else np.nan})])

    point_pm = per_model(None)
    point = summarize(point_pm)
    boot, n_nan = [], 0
    for _ in range(n_boot):
        mult = np.zeros(len(fovs))
        for f in strata:
            np.add.at(mult, rng.choice(f, size=len(f), replace=True), 1)
        pm = per_model(mult)
        n_nan += int(pm["d"].isna().sum())
        boot.append(summarize(pm))
    boot = pd.DataFrame(boot)

    def interval(u: str) -> tuple[float, float]:
        b = boot[u].dropna()
        return (np.percentile(b, 2.5), np.percentile(b, 97.5)) if len(b) else (np.nan, np.nan)

    rows = []
    for u in units:
        g = point_pm[point_pm["unit"] == u]
        lo, hi = interval(u)
        rows.append({"unit": u, "kind": "dataset" if u in DATASETS else "mixed slide",
                     "weight": float(w[u]) if u in DATASETS else np.nan, "n_models": int(g["d"].notna().sum()),
                     "auc_test_fold_mean": g["pool"].mean(), "auc_mixed_mean": g["mixed"].mean(), "d_mean": point[u],
                     "d_lo": lo, "d_hi": hi, "sd_over_models": g["d"].std(ddof=1),
                     "sd_of_repeat_means": g.groupby("repeat")["d"].mean().std(ddof=1)})
    ds_pm = point_pm[point_pm["unit"].isin(DATASETS)].assign(wt=lambda t: t["unit"].map(w))
    per = ds_pm.assign(wd=ds_pm["d"] * ds_pm["wt"], wp=ds_pm["pool"] * ds_pm["wt"], wm=ds_pm["mixed"] * ds_pm["wt"]
                       ).groupby(["repeat", "fold"])[["wd", "wp", "wm"]].sum()
    lo, hi = interval("overall")
    rows.append({"unit": "overall", "kind": "overall", "weight": 1.0, "n_models": len(per),
                 "auc_test_fold_mean": per["wp"].mean(), "auc_mixed_mean": per["wm"].mean(), "d_mean": point["overall"],
                 "d_lo": lo, "d_hi": hi, "sd_over_models": per["wd"].std(ddof=1),
                 "sd_of_repeat_means": per.groupby("repeat")["wd"].mean().std(ddof=1)})
    if n_nan:
        print(f"gap bootstrap: {n_nan} model x unit AUCs undefined (one class drawn out) over {n_boot} draws; "
              f"those draws average the other models")
    return pd.DataFrame(rows), point_pm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/models")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/comparisons")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=840410)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    info, pred, pools = load_runs(a.models)
    runs = run_metrics(a.models, info, mixed_weights(pred))
    tables = {"runs": runs}
    tables["oof"], tables["oof_slide_pairs"] = oof_tables(pred, pools, a.n_boot, rng)
    tables["mixed_summary"] = mixed_summary(runs)
    tables["ensembles"] = ensembles(pred, a.n_boot, rng)
    g, tables["gap_runs"] = gap(pred, a.n_boot, rng)
    drop = (pred["set"] == "test_mixed") & (pred["dataset"] + " " + pred["slide"]).isin(SENSITIVITY_DROP)
    sens, _ = gap(pred[~drop], a.n_boot, rng)
    sens = sens[sens["unit"] == "overall"].assign(unit=f"overall without {', '.join(SENSITIVITY_DROP)}",
                                                  kind="sensitivity")
    tables["gap"] = pd.concat([g, sens], ignore_index=True)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_rows", 300,
                           "display.float_format", "{:.3f}".format):
        for k, t in tables.items():
            t.to_csv(a.out / f"r5_{k}.csv", index=False)
            print(f"\n## {k}\n{t.to_string(index=False)}")


if __name__ == "__main__":
    main()
