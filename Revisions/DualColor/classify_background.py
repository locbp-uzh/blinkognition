#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Control: a model trained on protein traces (train_pure.py) classifies background traces.

The paper's "noise classification with pre-trained model" (SI; Figure S24B): background
traces from the same acquisitions, labeled with the protein of their slide, are classified
by the trained model, inference only, with one deterministic forward pass (every dropout
off); the paper read it as passed when "classification collapses into a single class".
Here the deterministic pass is the primary readout; MC dropout (n_mc passes, Wasserstein
uncertainty, kept at the model's validation threshold) is added unless --n-mc 0.

Every filtered background trace (background.py, gap >= 3 px to every protein box) of the
model's datasets is classified. Readouts, on the single-protein slides (truth = the slide's
protein):
- Collapse: fraction called SNAP and mean p_SNAP per slide (per_slide.csv; mixed slides too).
- AUC of p_SNAP, SNAP-slide vs HT-slide background, per dataset and within datasets
  (evaluate.within_dataset_auc), read two-sided (|AUC - 0.5|) with 95 % FOV-cluster bootstrap
  intervals. Slides differ in their background even when they carry the same protein, so 0.5
  is not the expected value: the reference is the same-protein slide pairs (slide_pairs.csv,
  evaluate.slide_pairs). A model that responds to something protein-specific in the
  background separates HT from SNAP slides more than slides of one protein.
- By FOV side: the model trained on the protein traces of its training FOVs, whose OFF frames
  carry those FOVs' background, so the metrics are given for 'untrained_fovs' (FOVs none of
  whose protein traces were trained on: validation FOVs and FOVs outside the training pool)
  as the primary readout, 'training_fovs' and 'all'.

Outputs (<model run>/background_inference/): predictions.csv, inputs.json and, unless
--n-boot 0, metrics.csv, slide_pairs.csv, per_slide.csv, metrics.txt. The bootstrap takes
minutes of CPU, so on Daint the GPU part runs with --n-boot 0 and the summary locally with
--summarize-only.

Usage (from the repo root):
    python Revisions/DualColor/classify_background.py Results/Revisions/DualColor/models/<run> [--n-mc N] [--n-boot 0]
    python Revisions/DualColor/classify_background.py Results/Revisions/DualColor/models/<run> --summarize-only
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "ML"))
from background import load_background  # noqa: E402
from dc import REPO_ROOT  # noqa: E402
from evaluate import metrics, slide_pairs  # noqa: E402
from train_pure import CLASSES, predict  # noqa: E402
from classify import deterministic_probs  # noqa: E402  (ML/classify.py)
from train import build_model_from_config  # noqa: E402
from utils import detect_accelerator  # noqa: E402


def classify_background(model, datasets: list[str], thr: float, n_mc: int, device, autocast_ctx,
                        ta_runs: dict | None = None, min_gap_px: int = 3) -> tuple[pd.DataFrame, list[dict]]:
    """Background traces of `datasets` with deterministic and (n_mc > 0) MC predictions."""
    parts, infos = [], []
    for ds in datasets:
        meta, X, info = load_background(ds, (ta_runs or {}).get(ds), min_gap_px)
        Xt = torch.from_numpy(X).float()
        det = deterministic_probs(model, Xt, device, autocast_ctx)
        p = pd.DataFrame({"p_SNAP_det": det[:, CLASSES.index("SNAP")],
                          "prediction_det": [CLASSES[k] for k in det.argmax(axis=1)]})
        if n_mc > 0:
            p = p.join(predict(model, X, n_mc, device, autocast_ctx, thr)[0])
        parts.append(meta.join(p))
        infos.append(info)
        print(f"{ds}: {len(meta)} background traces classified", flush=True)
    return pd.concat(parts, ignore_index=True), infos


def fov_sides(pred: pd.DataFrame, data_split: pd.DataFrame) -> pd.Series:
    """'training_fovs' if any protein trace of the FOV was a training trace of the model, else
    'untrained_fovs'."""
    trained = set(map(tuple, data_split.loc[data_split["set"] == "train", ["dataset", "slide", "fov"]].values))
    key = zip(pred["dataset"], pred["slide"], pred["fov"])
    return pd.Series(["training_fovs" if k in trained else "untrained_fovs" for k in key], index=pred.index)


def summarize(pred: pd.DataFrame, out: Path, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lab = pred[pred["true_class"].isin(CLASSES)]
    variants = {"deterministic": lab.assign(p_SNAP=lab["p_SNAP_det"], prediction=lab["prediction_det"], kept=True)}
    if "p_SNAP" in lab:
        variants["mc"] = lab
    rows, pairs = [], []
    for variant, gv in variants.items():
        for side, g in (("untrained_fovs", gv[gv["fov_side"] == "untrained_fovs"]),
                        ("training_fovs", gv[gv["fov_side"] == "training_fovs"]), ("all", gv)):
            if not len(g):
                continue
            rows.append({"variant": variant, "fovs": side, "dataset": "pooled", **metrics(g, n_boot, rng)})
            for ds, gd in g.groupby("dataset"):
                rows.append({"variant": variant, "fovs": side, "dataset": ds, **metrics(gd, n_boot, rng)})
            if side != "training_fovs":
                pairs.append(slide_pairs(g, "p_SNAP", n_boot, rng).assign(variant=variant, fovs=side))
    m = pd.DataFrame(rows)
    m.to_csv(out / "metrics.csv", index=False)
    sp = pd.concat(pairs, ignore_index=True)
    sp.to_csv(out / "slide_pairs.csv", index=False)
    agg = {"slide_class": ("true_class", "first"), "n": ("uniqueID", "size"), "n_fovs": ("fov", "nunique"),
           "mean_p_SNAP_det": ("p_SNAP_det", "mean"), "called_SNAP_det": ("called_SNAP_det", "mean")}
    if "p_SNAP" in pred:
        agg |= {"mean_p_SNAP": ("p_SNAP", "mean"), "called_SNAP": ("called_SNAP", "mean"), "kept": ("kept", "mean")}
    per_slide = (pred.assign(called_SNAP_det=pred["prediction_det"] == "SNAP",
                             called_SNAP=(pred["prediction"] == "SNAP") if "prediction" in pred else False)
                 .groupby(["dataset", "slide", "slide_type"], dropna=False).agg(**agg).reset_index())
    per_slide.to_csv(out / "per_slide.csv", index=False)
    cols = ["variant", "fovs", "dataset", "n_HT", "n_SNAP", "n_fovs", "auc", "auc_lo", "auc_hi", "auc_within_dataset",
            "auc_within_dataset_lo", "auc_within_dataset_hi", "recall_HT", "recall_SNAP", "kept_frac"]
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        txt = "\n\n".join([m[cols].to_string(index=False), sp.to_string(index=False), per_slide.to_string(index=False)])
    (out / "metrics.txt").write_text(txt + "\n")
    print(txt, flush=True)
    return m


def classify_run(run: Path, n_mc: int | None = None, n_boot: int = 2000, seed: int = 840410) -> None:
    """The control on a finished train_pure.py run; writes <run>/background_inference/."""
    full = json.load(open(run / "config_full.json"))
    mc = full["config"]
    if mc.get("trace_source", "protein") != "protein":
        raise SystemExit(f"{run.name} was trained on {mc['trace_source']} traces; this control is for protein models")
    if full.get("precision", "fp32") != "fp32":
        raise SystemExit(f"precision {full['precision']}: only fp32 models are handled here")
    thr = float(json.load(open(run / "threshold.json"))["wasserstein_threshold"])
    accel = detect_accelerator()
    device = accel["device"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    autocast_ctx = lambda: nullcontext()  # noqa: E731
    mcfg = mc.get("model", {"name": "orig_conv_gru"})
    model = build_model_from_config(model_name=mcfg["name"], in_channels=1, num_classes=len(CLASSES),
                                    dropout=mcfg.get("dropout"), model_kwargs=mcfg.get("kwargs", {}), device=device)
    model.load_state_dict(torch.load(run / "best_model.pth", map_location=device, weights_only=True))
    datasets = [i["dataset"] for i in full["inputs"]]
    ta_runs = {i["dataset"]: Path(i["trace_assignment_run"]).name for i in full["inputs"]}
    n_mc = int(mc.get("mc_dropout", {}).get("n_mc", 100)) if n_mc is None else int(n_mc)
    print(f"Model {run.name} ({mcfg['name']}), threshold {thr:.3f}, n_mc {n_mc}, datasets {datasets}, "
          f"device {accel['name']}", flush=True)
    out = run / "background_inference"
    out.mkdir(exist_ok=True)
    pred, infos = classify_background(model, datasets, thr, n_mc, device, autocast_ctx, ta_runs)
    split = pd.read_csv(run / "data_split.csv", dtype={"slide": str, "fov": str})
    pred["fov_side"] = fov_sides(pred, split)
    pred.to_csv(out / "predictions.csv", index=False)
    with open(out / "inputs.json", "w") as f:
        json.dump({"model_run": str(run.relative_to(REPO_ROOT)) if run.is_relative_to(REPO_ROOT) else str(run),
                   "n_mc": n_mc, "wasserstein_threshold": thr, "background_inputs": infos}, f, indent=2)
    if n_boot > 0:
        summarize(pred, out, n_boot, seed)
    print(f"Done: {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_run", type=Path)
    ap.add_argument("--n-mc", type=int, default=None, help="MC passes (default: the model's n_mc; 0 = deterministic only)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=840410)
    ap.add_argument("--summarize-only", action="store_true", help="summarize an existing predictions.csv")
    a = ap.parse_args()
    run = a.model_run if a.model_run.is_absolute() else REPO_ROOT / a.model_run
    if a.summarize_only:
        out = run / "background_inference"
        pred = pd.read_csv(out / "predictions.csv", dtype={"slide": str, "fov": str, "true_class": str})
        summarize(pred, out, a.n_boot, a.seed)
    else:
        classify_run(run, a.n_mc, a.n_boot, a.seed)


if __name__ == "__main__":
    main()
