#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Are the traces the paper's IN rule adds protein-like? A check without training (round 4,
TRAINING_NOTES.md point 8).

The paper-IN pool keeps most of the vesicle-table pool and adds traces the table rejected
(border band, dim or crowded vesicles, other objects). Each vesicle-table model (r2_all_aug0 and
r4_table_s1..s4) never saw the FOVs of its own validation side; on the single-protein slides of
those FOVs, its scores rank the slide's protein (truth = the slide's protein) for:
- table traces: the traces both pools contain (the model's own validation material);
- added traces: in the paper-IN pool, not in the table pool.
If the added traces are protein traces like the others, their AUC is about the same; if they are
mostly something else (not in vesicles, noise), it drops toward 0.5. Scores: one deterministic
pass (every dropout off; ML/classify.deterministic_probs), on the CPU. AUC per dataset and pooled
within datasets, with 95 % FOV-cluster bootstrap intervals, per model and averaged over models.

Outputs (Results/Revisions/DualColor/checks/20261009_round4_added_traces/): scores.csv, auc.csv.

Usage (from the repo root):
    python Revisions/DualColor/added_traces_check.py
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "ML"))
from dc import REPO_ROOT  # noqa: E402
from evaluate import within_dataset_auc  # noqa: E402
from train_pure import CLASSES, load_dataset  # noqa: E402
from classify import deterministic_probs  # noqa: E402  (ML/classify.py)
from train import build_model_from_config  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

MODELS = REPO_ROOT / "Results/Revisions/DualColor/models"
TABLE_RUNS = ["r2_all_aug0"] + [f"r4_table_s{k}" for k in (1, 2, 3, 4)]
OUT = REPO_ROOT / "Results/Revisions/DualColor/checks/20261009_round4_added_traces"
N_BOOT = 2000


def latest(name: str) -> Path:
    return sorted(p for p in MODELS.glob(f"*_{name}") if p.is_dir())[-1]


def auc_boot(g: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float, float, float]:
    y = (g["true_class"] == "SNAP").astype(int).to_numpy()
    p, ds = g["p_SNAP"].to_numpy(), g["dataset"].to_numpy()
    point = within_dataset_auc(y, p, ds)
    fov = (g["dataset"] + "/" + g["slide"] + "/" + g["fov"]).to_numpy()
    groups = [np.flatnonzero(fov == f) for f in np.unique(fov)]
    boot = []
    for _ in range(N_BOOT):
        i = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        boot.append(within_dataset_auc(y[i], p[i], ds[i]))
    boot = np.array(boot, dtype=float)
    return point, np.nanpercentile(boot, 2.5), np.nanpercentile(boot, 97.5), len(groups)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mc = {"train_label_filter": "paper_in"}
    loaded = [load_dataset(ds, mc) for ds in ("DFK785", "DFK788", "DFK789")]
    meta = pd.concat([x[0] for x in loaded], ignore_index=True)
    X = np.concatenate([x[1] for x in loaded], axis=0)
    pure = meta["slide_type"] == "pure"
    table = pure & meta["in_vesicle"] & meta["own_dye"]
    paper = pure & meta["paper_in"]
    meta["group"] = np.where(table & paper, "table", np.where(paper & ~table, "added", np.where(table, "table only", "")))
    print(meta[pure].groupby(["dataset", "group"]).size().unstack(fill_value=0).to_string())

    rows = []
    for name in TABLE_RUNS:
        run = latest(name)
        split = pd.read_csv(run / "data_split.csv", dtype={"slide": str, "fov": str})
        val_fovs = set(map(tuple, split.loc[split["set"].isin(["val", "val_unused"]), ["dataset", "slide", "fov"]].values))
        train_fovs = set(map(tuple, split.loc[split["set"] == "train", ["dataset", "slide", "fov"]].values))
        key = list(zip(meta["dataset"], meta["slide"], meta["fov"]))
        sel = (meta["group"].isin(["table", "added"]) & pd.Series([k in val_fovs and k not in train_fovs for k in key])).to_numpy()
        full = json.load(open(run / "config_full.json"))
        mcfg = full["config"].get("model", {"name": "orig_conv_gru"})
        model = build_model_from_config(model_name=mcfg["name"], in_channels=1, num_classes=len(CLASSES),
                                        dropout=mcfg.get("dropout"), model_kwargs=mcfg.get("kwargs", {}), device="cpu")
        model.load_state_dict(torch.load(run / "best_model.pth", map_location="cpu", weights_only=True))
        probs = deterministic_probs(model, torch.from_numpy(X[sel]).float(), "cpu", lambda: nullcontext())
        rows.append(meta.loc[sel, ["dataset", "slide", "fov", "extraction_key", "uniqueID", "true_class", "group"]]
                    .assign(model=name, p_SNAP=probs[:, CLASSES.index("SNAP")]))
        print(f"{name}: {int(sel.sum())} traces in its validation FOVs", flush=True)
    sc = pd.concat(rows, ignore_index=True)
    sc.to_csv(OUT / "scores.csv", index=False)

    rng = np.random.default_rng(840410)
    res = []
    for (model, grp), g in sc.groupby(["model", "group"]):
        for ds, gd in [("within datasets", g)] + list(g.groupby("dataset")):
            if gd["true_class"].nunique() < 2:
                continue
            a, lo, hi, nf = auc_boot(gd, rng)
            res.append({"model": model, "group": grp, "dataset": ds, "n_HT": int((gd.true_class == "HT").sum()),
                        "n_SNAP": int((gd.true_class == "SNAP").sum()), "n_fovs": nf, "auc": a, "auc_lo": lo, "auc_hi": hi})
    res = pd.DataFrame(res)
    mean = res.groupby(["group", "dataset"])[["auc"]].agg(["mean", "min", "max"]).round(3)
    mean.columns = ["auc_mean_over_models", "auc_min", "auc_max"]
    res.to_csv(OUT / "auc.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))
        print(mean.to_string())
    print(f"Done: {OUT}")


if __name__ == "__main__":
    main()
