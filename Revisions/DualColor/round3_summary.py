#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Round 3 summary (TRAINING_NOTES.md point 5): the three architectures and their controls, from
the evaluate.py and classify_background.py outputs of the latest run of each name.

Tables (printed, and written to <out>/r3_*.csv):
- models       protein models (CNN-GRU = r2_all_aug0, r3_resnet1d_protein, r3_tcn_protein):
               validation and mixed-slide AUC, per dataset and pooled, with FOV bootstrap
               intervals; mixed-slide within-dataset AUC
- scrambled    the five permutations per architecture: validation AUC (against the scrambled
               labels) and mixed-slide AUC (against the true labels), each run and their
               mean / min / max; the real model's mixed AUC next to them
- background   background-trained models: AUC on the held-out background of the validation
               FOVs (val_unused), per dataset and within datasets; validation AUC
- inference    background inference of each protein model (deterministic pass, FOVs not
               trained on): per-dataset and within-dataset AUC, fraction called SNAP
- pairs        slide pairs of both background controls: per dataset, AUC and |AUC - 0.5| of
               HT-vs-SNAP slide pairs next to the same-protein pairs (the reference)

Usage:
    python Revisions/DualColor/round3_summary.py [--models Results/Revisions/DualColor/models] [--out DIR]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCH = {"gru": "CNN-GRU", "resnet1d": "ResNet1D", "tcn": "TCN"}


def latest(models: Path, name: str) -> Path | None:
    runs = sorted(p for p in models.glob(f"*_{name}") if p.is_dir())
    return runs[-1] if runs else None


def ci(r, k: str = "auc") -> str:
    return f"{r[k]:.2f} [{r[k + '_lo']:.2f}-{r[k + '_hi']:.2f}]"


def metrics_of(run: Path) -> pd.DataFrame | None:
    f = run / "evaluation" / "metrics.csv"
    return pd.read_csv(f) if f.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/models")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/comparisons")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    protein = {"gru": "r2_all_aug0", "resnet1d": "r3_resnet1d_protein", "tcn": "r3_tcn_protein"}
    tables = {}

    rows = []
    for arch, name in protein.items():
        run = latest(a.models, name)
        m = metrics_of(run) if run else None
        if m is None:
            continue
        m = m[m["subset"] == "all"]
        for _, r in m.iterrows():
            rows.append({"model": ARCH[arch], "run": run.name, "set": r["set"], "dataset": r["dataset"],
                         "n_HT": r["n_HT"], "n_SNAP": r["n_SNAP"], "AUC": ci(r),
                         "AUC_within_dataset": f"{r['auc_within_dataset']:.2f}", "BA": f"{r['balanced_accuracy']:.2f}",
                         "kept": f"{r['kept_frac']:.2f}", "BA_kept": f"{r['kept_balanced_accuracy']:.2f}"})
    tables["models"] = pd.DataFrame(rows)

    rows = []
    for arch in ARCH:
        for k in range(1, 6):
            run = latest(a.models, f"r3_{arch}_scrambled_s{k}")
            m = metrics_of(run) if run else None
            if m is None:
                continue
            pooled = m[(m["dataset"] == "pooled") & (m["subset"] == "all")].set_index("set")
            rows.append({"model": ARCH[arch], "permutation": k, "run": run.name,
                         "val_auc": pooled.loc["val", "auc"], "mixed_auc": pooled.loc["test_mixed", "auc"],
                         "mixed_auc_within_dataset": pooled.loc["test_mixed", "auc_within_dataset"]})
    sc = pd.DataFrame(rows)
    if len(sc):
        agg = sc.groupby("model")[["val_auc", "mixed_auc"]].agg(["mean", "min", "max"]).round(3)
        agg.columns = [f"{c}_{s}" for c, s in agg.columns]
        real = tables["models"]
        if len(real):
            r = real[(real["set"] == "test_mixed") & (real["dataset"] == "pooled")].set_index("model")["AUC"]
            agg["real_model_mixed_AUC"] = agg.index.map(r)
        tables["scrambled_runs"] = sc
        tables["scrambled"] = agg.reset_index()

    rows, pairs = [], []
    for arch in ARCH:
        run = latest(a.models, f"r3_{arch}_background")
        m = metrics_of(run) if run else None
        if m is None:
            continue
        for _, r in m[m["subset"] == "all"].iterrows():
            rows.append({"model": ARCH[arch], "set": r["set"], "dataset": r["dataset"], "n_HT": r["n_HT"],
                         "n_SNAP": r["n_SNAP"], "AUC": ci(r), "AUC_within_dataset": f"{r['auc_within_dataset']:.2f}",
                         "separation": f"{abs(r['auc'] - 0.5):.2f}"})
        f = run / "evaluation" / "slide_pairs.csv"
        if f.exists():
            pairs.append(pd.read_csv(f).assign(model=ARCH[arch], control="background training (val_unused)"))
    tables["background"] = pd.DataFrame(rows)

    rows = []
    for arch, name in protein.items():
        run = latest(a.models, name)
        bi = run / "background_inference" if run else None
        if not bi or not (bi / "metrics.csv").exists():
            continue
        m = pd.read_csv(bi / "metrics.csv")
        m = m[(m["variant"] == "deterministic") & (m["fovs"] == "untrained_fovs")]
        ps = pd.read_csv(bi / "per_slide.csv")
        for _, r in m.iterrows():
            sl = ps[(ps["slide_type"] == "pure") & ((ps["dataset"] == r["dataset"]) | (r["dataset"] == "pooled"))]
            rows.append({"model": ARCH[arch], "dataset": r["dataset"], "n_HT": r["n_HT"], "n_SNAP": r["n_SNAP"],
                         "AUC": ci(r), "AUC_within_dataset": f"{r['auc_within_dataset']:.2f}",
                         "separation": f"{abs(r['auc'] - 0.5):.2f}",
                         "called_SNAP": f"{(sl['called_SNAP_det'] * sl['n']).sum() / sl['n'].sum():.3f}",
                         "mean_p_SNAP": f"{(sl['mean_p_SNAP_det'] * sl['n']).sum() / sl['n'].sum():.3f}"})
        sp = pd.read_csv(bi / "slide_pairs.csv")
        pairs.append(sp[(sp["variant"] == "deterministic") & (sp["fovs"] == "untrained_fovs")]
                     .assign(model=ARCH[arch], control="background inference (untrained FOVs)"))
    tables["inference"] = pd.DataFrame(rows)
    if pairs:
        p = pd.concat(pairs, ignore_index=True)
        p["AUC"] = p.apply(ci, axis=1)
        tables["pairs"] = p[["control", "model", "dataset", "pair", "slide_a", "class_a", "slide_b", "class_b",
                             "n_a", "n_b", "AUC", "separation"]].round(3)

    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 40):
        for k, t in tables.items():
            if len(t):
                t.to_csv(a.out / f"r3_{k}.csv", index=False)
                print(f"\n## {k}\n{t.to_string(index=False)}")


if __name__ == "__main__":
    main()
