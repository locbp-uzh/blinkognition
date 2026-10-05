#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Check ML/classify.py against the training run's own test set.

export   writes the test traces saved by ML/train.py (MCD_results/traces_with_wasserstein.npz)
         as trace pickles that classify.py reads (test<Class>_filtered_minmax.pkl, columns =
         UniqueIDs) and the training's predictions as training_reference.csv
compare  joins a classify.py run on those pickles with training_reference.csv: prediction
         agreement, Wasserstein correlation and signed shift, kept fraction and filtered
         accuracy at the training threshold, for both

Usage (from ML/):
    python ../Revisions/DFK785/ml/testset_check.py export  --model-dir <train run> --out <dir>
    python ../Revisions/DFK785/ml/testset_check.py compare --ref <dir>/training_reference.csv --run <classify run>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def export(model_dir: Path, out: Path) -> None:
    d = np.load(model_dir / "MCD_results" / "traces_with_wasserstein.npz", allow_pickle=True)
    names = [str(c) for c in d["class_names"]]
    uid = d["unique_ids"]
    out.mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(names):
        m = d["labels"] == k
        pd.DataFrame(d["traces"][m, 0, :].T, columns=uid[m]).to_pickle(out / f"test{name}_filtered_minmax.pkl")
    pd.DataFrame({"uniqueID": uid, "label": [names[k] for k in d["labels"]],
                  "train_prediction": [names[k] for k in d["predictions"]],
                  "train_wasserstein": d["wasserstein_distances"]}).to_csv(out / "training_reference.csv", index=False)
    print(f"Exported {len(uid)} test traces ({dict(zip(names, np.bincount(d['labels'])))}) to {out}")


def compare(ref: Path, run: Path) -> None:
    r = pd.read_csv(ref)
    p = pd.read_csv(run / "predictions.csv")
    with open(run / "config_summary.json") as f:
        thr = float(json.load(f)["wasserstein_threshold"])
    m = p.merge(r, on=["uniqueID", "label"])
    same = m["prediction"] == m["train_prediction"]
    d = (m["wasserstein"] - m["train_wasserstein"])[same]
    out = {"n": len(m), "prediction_agreement": same.mean(),
           "wasserstein_r": np.corrcoef(m["wasserstein"], m["train_wasserstein"])[0, 1],
           "wasserstein_shift_mean": d.mean(), "wasserstein_shift_se": d.std() / np.sqrt(len(d))}
    for tag, wd, pr in (("train", m["train_wasserstein"], m["train_prediction"]),
                        ("classify", m["wasserstein"], m["prediction"])):
        k = wd > thr
        out[f"{tag}_initial_accuracy"] = (pr == m["label"]).mean()
        out[f"{tag}_kept_frac"] = k.mean()
        out[f"{tag}_filtered_accuracy"] = (pr[k] == m["label"][k]).mean()
    s = pd.Series(out)
    print(s.to_string())
    s.to_csv(run / "testset_check.csv", header=["value"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--model-dir", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--ref", type=Path, required=True)
    c.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    export(a.model_dir, a.out) if a.cmd == "export" else compare(a.ref, a.run)


if __name__ == "__main__":
    main()
