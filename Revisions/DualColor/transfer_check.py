#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Apply already trained models, unchanged, to the traces of other datasets: does a classifier
trained on some sessions separate HT and SNAP in a session it never saw? No training here.

Models: the latest run of each name <prefix>_kfold_p<seed>_f<fold> (default r5b, the 15 round-5b
CNN-GRU models, TRAINING_NOTES.md points 9-10), one deterministic pass each (every dropout off;
ML/classify.deterministic_probs), and their mean (the ensemble).
Traces (train_pure.load_dataset, minmax, filtered): on the single-protein slides the vesicle
table's pool (own dye within the slide's match radius), truth = the slide's protein; on the
mixed slides the traces of single-dye vesicles within 2 px, truth = the vesicle's dye. Both sets
are new to the models when the datasets were not in their training.

Readout per dataset and set (single-protein / mixed), and per mixed slide: n, AUC of p_SNAP for
each model and for the ensemble, with a 95 % FOV-cluster bootstrap interval for the ensemble
(FOVs resampled within slide); balanced accuracy at the ensemble's argmax.

Outputs (<out>): scores.csv (per trace and model), auc.csv; printed.

Usage (from the repo root, blink2env):
    python Revisions/DualColor/transfer_check.py DFK842 DFK843 DFK846 --out Results/Revisions/DualColor/checks/<folder>
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "ML"))
from dc import REPO_ROOT  # noqa: E402
from train_pure import CLASSES, load_dataset  # noqa: E402
from classify import deterministic_probs  # noqa: E402  (ML/classify.py)
from train import build_model_from_config  # noqa: E402

MODELS = REPO_ROOT / "Results/Revisions/DualColor/models"


def model_runs(prefix: str) -> list[Path]:
    runs = []
    for p in (840410, 1, 2):
        for k in range(5):
            found = sorted(r for r in MODELS.glob(f"*_{prefix}_kfold_p{p}_f{k}") if r.is_dir() and (r / "best_model.pth").exists())
            if found:
                runs.append(found[-1])
    return runs


def load_model(run: Path):
    full = json.load(open(run / "config_full.json"))
    mcfg = full["config"].get("model", {"name": "orig_conv_gru"})
    model = build_model_from_config(model_name=mcfg["name"], in_channels=1, num_classes=len(CLASSES),
                                    dropout=mcfg.get("dropout"), model_kwargs=mcfg.get("kwargs", {}), device="cpu")
    model.load_state_dict(torch.load(run / "best_model.pth", map_location="cpu", weights_only=True))
    return model


def auc_boot(g: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    y = (g["true_class"] == "SNAP").to_numpy(int)
    p = g["p_SNAP"].to_numpy()
    point = roc_auc_score(y, p)
    # .groups (labels of the reset index = row positions of g), not .indices (positions inside each
    # slide): fixed 2026-11-17; before, units of several slides (the single-protein slides) resampled
    # the wrong rows. Single-slide units (each new session's mixed slide) were unaffected.
    idx_by_slide = {s: [np.asarray(ix) for ix in gs.groupby("fov").groups.values()]
                    for s, gs in g.reset_index(drop=True).groupby("slide")}
    boot = []
    for _ in range(n_boot):
        pick = np.concatenate([np.concatenate([fovs[i] for i in rng.integers(0, len(fovs), len(fovs))])
                               for fovs in idx_by_slide.values()])
        if len(np.unique(y[pick])) == 2:
            boot.append(roc_auc_score(y[pick], p[pick]))
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--prefix", default="r5b")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    runs = model_runs(a.prefix)
    print(f"{len(runs)} models ({a.prefix})")
    frames = []
    for ds in a.datasets:
        meta, X, info = load_dataset(ds, {"train_label_filter": "own_dye"})
        pure = (meta["slide_type"] == "pure") & meta["in_vesicle"] & meta["own_dye"]
        mixed = (meta["slide_type"] == "mixed") & meta["true_class"].isin(CLASSES)
        sel = (pure | mixed).to_numpy()
        m = meta.loc[sel, ["dataset", "slide", "fov", "extraction_key", "uniqueID", "slide_type", "true_class"]].reset_index(drop=True)
        print(f"{ds}: {int(pure.sum())} single-protein-slide traces, {int(mixed.sum())} labeled mixed-slide traces", flush=True)
        Xt = torch.from_numpy(X[sel]).float()
        for run in runs:
            probs = deterministic_probs(load_model(run), Xt, "cpu", lambda: nullcontext())
            frames.append(m.assign(model=run.name.split("_", 2)[2], p_SNAP=probs[:, CLASSES.index("SNAP")]))
    sc = pd.concat(frames, ignore_index=True)
    sc.to_csv(a.out / "scores.csv", index=False)

    rng = np.random.default_rng(840410)
    ens = sc.groupby(["dataset", "slide", "fov", "extraction_key", "uniqueID", "slide_type", "true_class"],
                     as_index=False)["p_SNAP"].mean()
    rows = []
    units = [(ds, st, None) for ds in a.datasets for st in ("pure", "mixed")] + \
            [(ds, "mixed", s) for ds in a.datasets for s in sorted(ens.loc[(ens.dataset == ds) & (ens.slide_type == "mixed"), "slide"].unique())]
    for ds, st, slide in units:
        e = ens[(ens.dataset == ds) & (ens.slide_type == st) & ((ens.slide == slide) if slide else True)]
        if e["true_class"].nunique() < 2:
            continue
        pt, lo, hi = auc_boot(e, rng, a.n_boot)
        per = sc[(sc.dataset == ds) & (sc.slide_type == st) & ((sc.slide == slide) if slide else True)].groupby("model").apply(
            lambda g: roc_auc_score((g["true_class"] == "SNAP").astype(int), g["p_SNAP"]), include_groups=False)
        pred_snap = e["p_SNAP"] > 0.5
        rec = [((~pred_snap) & (e.true_class == "HT")).sum() / (e.true_class == "HT").sum(),
               (pred_snap & (e.true_class == "SNAP")).sum() / (e.true_class == "SNAP").sum()]
        rows.append({"dataset": ds, "set": "single-protein slides" if st == "pure" else "mixed slides",
                     "slide": slide or "all", "n_HT": int((e.true_class == "HT").sum()), "n_SNAP": int((e.true_class == "SNAP").sum()),
                     "n_fovs": int(e.groupby(["slide", "fov"]).ngroups), "auc_ensemble": pt, "auc_lo": lo, "auc_hi": hi,
                     "auc_models_mean": float(per.mean()), "auc_models_min": float(per.min()), "auc_models_max": float(per.max()),
                     "recall_HT": rec[0], "recall_SNAP": rec[1], "balanced_accuracy": float(np.mean(rec))})
    t = pd.DataFrame(rows)
    t.to_csv(a.out / "auc.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
