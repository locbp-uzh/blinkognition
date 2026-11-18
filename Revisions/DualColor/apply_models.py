#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Trained train_pure.py runs applied, unchanged, to the mixed-slide traces of datasets they were not
tested on (TRAINING_NOTES.md point 13: the round-6 C models on DFK851). No training.

The prediction is train_pure.py's own (train_pure.predict: MC dropout, n_mc passes as the run's config,
mean p_SNAP, Wasserstein distance, kept at the run's threshold.json), on the traces train_pure.py would
use as 'test_mixed' (train_pure.load_dataset, the run's input_channel; mixed slides; every filtered
trace, the class where a single-dye vesicle lies within 2 px). transfer_check.py differs: one
deterministic pass, no threshold. Torch is seeded with the run's seed before each run's pass.

Output: <out>/<run_name>.csv with the columns of predictions.csv (set = test_mixed) and the run folder.

    python Revisions/DualColor/apply_models.py --runs r6_pooled_s840410 r6_pooled_s1 ... \
        --datasets DFK851 --ta-runs DFK851=run_001 --out Results/Revisions/DualColor/dfk851/c_transfer
    # check: on a dataset the runs did predict, compare with their own predictions.csv
    python Revisions/DualColor/apply_models.py --runs ... --datasets DFK846 --out <dir> --compare
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
from train_pure import CLASSES, load_dataset, predict  # noqa: E402
from train import build_model_from_config  # noqa: E402
from utils import detect_accelerator  # noqa: E402  (ML/utils.py)

MODELS = REPO_ROOT / "Results/Revisions/DualColor/models"


def latest(name: str) -> Path:
    runs = sorted(p for p in MODELS.glob(f"*_{name}") if p.is_dir() and (p / "best_model.pth").exists())
    if not runs:
        raise SystemExit(f"no trained run named {name} in {MODELS}")
    return runs[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run names (the latest run folder of each)")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--ta-runs", nargs="*", default=[], help="DS=run_NNN for datasets the run configs do not pin")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", action="store_true", help="compare with the runs' own test_mixed predictions")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    extra_ta = dict(s.split("=", 1) for s in a.ta_runs)
    accel = detect_accelerator()
    device = accel["device"]
    print(f"Device {accel['name']} ({accel['type']})")
    rows = []
    for name in a.runs:
        run = latest(name)
        full = json.load(open(run / "config_full.json"))
        mc = dict(full["config"])
        if mc.get("precision", "fp32") != "fp32":
            raise SystemExit(f"{name}: only fp32 runs are supported")
        mc["trace_assignment_runs"] = {**(mc.get("trace_assignment_runs") or {}), **extra_ta}
        thr = json.load(open(run / "threshold.json"))["wasserstein_threshold"]
        mcfg = mc.get("model", {"name": "orig_conv_gru"})
        model = build_model_from_config(model_name=mcfg["name"], in_channels=1, num_classes=len(CLASSES),
                                        dropout=mcfg.get("dropout"), model_kwargs=mcfg.get("kwargs", {}), device=device)
        model.load_state_dict(torch.load(run / "best_model.pth", map_location=device, weights_only=True))
        n_mc = int(mc.get("mc_dropout", {}).get("n_mc", 100))
        parts = []
        for ds in a.datasets:
            meta, X, info = load_dataset(ds, {k: mc[k] for k in ("trace_assignment_runs", "input_channel") if k in mc})
            idx = meta.index[meta["slide_type"] == "mixed"].to_numpy()
            torch.manual_seed(int(mc.get("seed", 840410)))
            p, _ = predict(model, X[idx], n_mc, device, lambda: nullcontext(), thr)
            parts.append(meta.loc[idx].reset_index(drop=True).assign(set="test_mixed").join(p))
            print(f"{name} on {ds}: {len(idx)} mixed traces ({info['trace_assignment_run']})", flush=True)
        df = pd.concat(parts, ignore_index=True).assign(run=run.name)
        df.to_csv(a.out / f"{name}.csv", index=False)
        if a.compare:
            own = pd.read_csv(run / "predictions.csv", dtype={"slide": str, "fov": str, "uniqueID": str,
                                                             "extraction_key": str, "true_class": str}, low_memory=False)
            own = own[(own["set"] == "test_mixed") & own["dataset"].isin(a.datasets)]
            m = df.astype({"uniqueID": str, "extraction_key": str}).merge(
                own, on=["dataset", "extraction_key", "uniqueID"], suffixes=("", "_own"), validate="one_to_one")
            lab = m["true_class"].isin(CLASSES)
            y = (m.loc[lab, "true_class"] == "SNAP").astype(int)
            rows.append({"run": name, "n_here": len(df), "n_own": len(own), "n_matched": len(m),
                         "corr_p_SNAP": np.corrcoef(m["p_SNAP"], m["p_SNAP_own"])[0, 1],
                         "max_abs_diff": float((m["p_SNAP"] - m["p_SNAP_own"]).abs().max()),
                         "auc_here": roc_auc_score(y, m.loc[lab, "p_SNAP"]), "auc_own": roc_auc_score(y, m.loc[lab, "p_SNAP_own"]),
                         "kept_agree": float((m["kept"] == m["kept_own"]).mean())})
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(a.out / "compare_own.csv", index=False)
        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            print(t.to_string(index=False))


if __name__ == "__main__":
    main()
