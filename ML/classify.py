#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright__ = "Copyright 2026, UZH, Switzerland"

"""
Classify new traces with a trained model (inference only, no retraining).

Loads best_model.pth and config_full.json of a run of ML/train.py, rebuilds the model
exactly as train.py did (build_model_from_config), and for every trace set in the
config runs a deterministic pass and MC Dropout (n_mc passes, as in training). The
prediction is the argmax of the MC-mean probabilities and the uncertainty is the
Wasserstein distance between the predicted class's MC distribution and the closest
other class's (the definition of utils.evaluate_uncertainty_filtered). A trace is kept
if its distance exceeds the threshold.

The threshold is fixed BEFORE looking at the new data: by default the one the training
run auto-selected on its own test set (MCD_results/mc_dropout_metrics.json,
selected_threshold), or a number given in the config. It is never re-tuned on the
traces being classified, because on new data with uncertain or no labels that would
fit the threshold to those labels.

Trace sets are folders of filtered trace pickles as written by the extraction pipeline
(one column per trace, column name = UniqueID), found with utils.discover_protein_files
by key prefix and channel. A set may carry a true label (a class of the model); sets
with a label get accuracy, confusion matrices and the diagnostic Wasserstein sweep.

Outputs (<output_root>/<timestamp>_<run_name>/):
    predictions.csv        one row per trace: set, uniqueID, label, deterministic and MC-mean
                           probabilities, MC SD, prediction, Wasserstein distance, kept
    summary.csv            per set: traces, kept, predicted fractions (all and kept); with a
                           label, accuracy with a Wilson 95 % interval
    labeled/               pooled labeled sets: confusion matrices (all, kept at the fixed
                           threshold) and the Wasserstein sweep (diagnostic, threshold fixed)
    config_full.json, config_summary.json, <run_name>.log

Usage:
    cd ML
    python classify.py -c config_classify.yaml
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (  # noqa: E402
    _load_col_names,
    batch_size_hint,
    detect_accelerator,
    discover_protein_files,
    estimate_batch_size,
    evaluate_uncertainty_filtered,
    load_series,
    mc_dropout_predict,
    plot_confusion_matrix,
    setup_precision_and_flags,
)
from train import build_model_from_config, set_seed  # noqa: E402
from models import LockedDropout, MCDropout  # noqa: E402


class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()


def _slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s or "run").strip("_")


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k / n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def wasserstein_uncertainty(probs_mc, pred):
    """Per trace: distance between the MC distribution of the predicted class and the closest other class."""
    n_mc, N, C = probs_mc.shape
    out = np.empty(N)
    for i in range(N):
        other = [wasserstein_distance(probs_mc[:, i, pred[i]], probs_mc[:, i, c]) for c in range(C) if c != pred[i]]
        out[i] = min(other) if other else 0.0
    return out


def deterministic_probs(model, X_t, device, autocast_ctx, batch=256):
    """Softmax with every dropout off. The zoo's MCDropout ignores eval() and LockedDropout follows
    mc_eval, so both are switched off here for the pass and restored afterwards."""
    saved = [(m, m.p) for m in model.modules() if isinstance(m, MCDropout)]
    locked = [(m, m.mc_eval) for m in model.modules() if isinstance(m, LockedDropout)]
    for m, _ in saved:
        m.p = 0.0
    for m, _ in locked:
        m.mc_eval = False
    model.eval()
    try:
        out = []
        with torch.no_grad():
            for i in range(0, len(X_t), batch):
                with autocast_ctx():
                    out.append(torch.softmax(model(X_t[i:i + batch].to(device)), dim=1).float().cpu().numpy())
        return np.concatenate(out)
    finally:
        for m, p in saved:
            m.p = p
        for m, v in locked:
            m.mc_eval = v


def training_length(model_dir, cfg):
    """Frames per trace the model was trained on: config n_frames, else the training log's 'Dataset loaded' line."""
    if cfg.get("n_frames"):
        return int(cfg["n_frames"])
    for log in Path(model_dir).glob("*.log"):
        m = re.search(r"Dataset loaded: \d+ samples, \d+ channels, (\d+) timesteps", log.read_text(errors="ignore"))
        if m:
            return int(m.group(1))
    raise ValueError(f"Training length not found in the logs of {model_dir}; set n_frames in the config")


def load_set(spec, channels, T_model):
    """X (N, C, T) and UniqueIDs of one trace set."""
    files = discover_protein_files(spec["path"], spec["key"], channels)
    arrs = [load_series(f) for f in files]
    if len({a.shape for a in arrs}) != 1:
        raise ValueError(f"{spec['name']}: shape mismatch across channels {[a.shape for a in arrs]}")
    X = np.stack(arrs, axis=1).astype(np.float32)
    uids = _load_col_names(files[0])
    if X.shape[2] < T_model:
        raise ValueError(f"{spec['name']}: traces have {X.shape[2]} frames, the model was trained on {T_model}")
    if X.shape[2] > T_model:
        print(f"  {spec['name']}: trimming {X.shape[2]} -> {T_model} frames (training length)")
        X = X[:, :, :T_model]
    if not np.isfinite(X).all():
        raise ValueError(f"{spec['name']}: non-finite values in traces")
    return X, uids, files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_dir = cfg["model_dir"]
    sets = cfg["trace_sets"]
    output_root = cfg.get("output_root", "../Results/Classify")
    run_name = cfg.get("run_name", "classify")
    sys_cfg = cfg.get("system", {})
    seed = int(sys_cfg.get("seed", 840410))
    mcd_cfg = cfg.get("mc_dropout", {})

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_root, f"{ts}_{_slug(run_name)}")
    os.makedirs(run_dir, exist_ok=True)
    log_file = open(os.path.join(run_dir, f"{_slug(run_name)}.log"), "w")
    sys.stdout = sys.stderr = Tee(sys.__stdout__, log_file)
    print(f"Output directory: {run_dir}")
    with open(os.path.join(run_dir, "config_full.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    set_seed(seed)

    # The trained model, rebuilt as train.py built it
    with open(os.path.join(model_dir, "config_full.json")) as f:
        train_cfg = json.load(f)
    model_cfg = train_cfg.get("model", {})
    model_name = model_cfg.get("name", "orig_conv_gru")
    dataset = train_cfg["data"]["dataset"]
    class_names = list(dataset.keys())
    first = dataset[class_names[0]]
    channels = cfg.get("channels") or (first.get("channels") if isinstance(first, dict) else None)
    if not channels:
        raise ValueError("The training config does not list channels; give them in the classify config")
    user_kwargs = dict(model_cfg.get("kwargs", {}))
    user_kwargs.update(model_cfg.get("per_model", {}).get(model_name, {}))

    accel = detect_accelerator()
    device = accel["device"]
    _, autocast_ctx, _ = setup_precision_and_flags(accel)
    print(f"Device: {accel['name']} ({accel['type']})")
    model = build_model_from_config(model_name=model_name, in_channels=len(channels), num_classes=len(class_names),
                                    dropout=model_cfg.get("dropout"), model_kwargs=user_kwargs, device=device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pth"), map_location=device,
                                     weights_only=True))
    model.eval()
    print(f"Model: {model_name} from {model_dir}; classes {class_names}; channels {channels}")

    T_model = training_length(model_dir, cfg)

    thr_cfg = mcd_cfg.get("wasserstein_threshold", "model")
    if thr_cfg == "model":
        with open(os.path.join(model_dir, "MCD_results", "mc_dropout_metrics.json")) as f:
            threshold = float(json.load(f)["selected_threshold"])
        thr_source = "training run (auto-selected on its test set)"
    else:
        threshold, thr_source = float(thr_cfg), "config"
    n_mc = int(mcd_cfg.get("n_mc", 100))
    print(f"Wasserstein threshold {threshold:.4f} from {thr_source}; n_mc {n_mc}")

    batch_size = int(train_cfg.get("optimization", {}).get("batch_size", 64))
    bs_hint = batch_size_hint(batch_size, accel)
    mc_bs = estimate_batch_size(model_name=model_name) if (accel["type"] == "cuda" and bs_hint is None) \
        else (bs_hint or batch_size)

    rows, summ = [], []
    lab_probs, lab_y = [], []
    for spec in sets:
        X, uids, files = load_set(spec, channels, T_model)
        print(f"\n{spec['name']}: {X.shape[0]} traces from {[os.path.basename(f) for f in files]}")
        X_t = torch.from_numpy(X)
        p_det = deterministic_probs(model, X_t, device, autocast_ctx)
        with torch.no_grad():
            probs_mc = mc_dropout_predict(model, X_t, n_mc=n_mc, batch_size=mc_bs, device=device,
                                          autocast_ctx=autocast_ctx)
        p_mean, p_sd = probs_mc.mean(axis=0), probs_mc.std(axis=0)
        pred = p_mean.argmax(axis=1)
        wd = wasserstein_uncertainty(probs_mc, pred)
        kept = wd > threshold
        label = spec.get("label")
        if label is not None and label not in class_names:
            raise ValueError(f"{spec['name']}: label {label} is not a model class {class_names}")

        df = pd.DataFrame({"set": spec["name"], "uniqueID": uids, "label": label or "",
                           "prediction": [class_names[k] for k in pred], "wasserstein": wd, "kept": kept})
        for k, c in enumerate(class_names):
            df[f"p_det_{c}"] = p_det[:, k]
            df[f"p_mc_{c}"] = p_mean[:, k]
            df[f"p_mc_sd_{c}"] = p_sd[:, k]
        rows.append(df)

        s = {"set": spec["name"], "label": label or "", "n": len(df), "n_kept": int(kept.sum()),
             "kept_frac": kept.mean()}
        for c in class_names:
            s[f"frac_pred_{c}"] = (df["prediction"] == c).mean()
            s[f"frac_pred_{c}_kept"] = (df.loc[kept, "prediction"] == c).mean() if kept.any() else np.nan
        if label is not None:
            for tag, m in (("", np.ones(len(df), bool)), ("_kept", kept)):
                k, n = int((df.loc[m, "prediction"] == label).sum()), int(m.sum())
                lo, hi = wilson(k, n)
                s[f"accuracy{tag}"], s[f"ci95_lo{tag}"], s[f"ci95_hi{tag}"] = (k / n if n else np.nan), lo, hi
            lab_probs.append(probs_mc)
            lab_y.append(np.full(len(df), class_names.index(label)))
        summ.append(s)
        print(pd.DataFrame([s]).T.to_string(header=False))

    pred_df = pd.concat(rows, ignore_index=True)
    pred_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    if lab_y:
        lab_dir = os.path.join(run_dir, "labeled")
        y = np.concatenate(lab_y)
        P = np.concatenate(lab_probs, axis=1)
        pr = P.mean(axis=0).argmax(axis=1)
        plot_confusion_matrix(y, pr, class_names=class_names, normalize="true",
                              save_path=os.path.join(lab_dir, "confusion_matrix"),
                              title="Confusion matrix (all traces)", figsize=(6, 6))
        mcd = evaluate_uncertainty_filtered(P, y_true=y, threshold=threshold, class_names=class_names,
                                            show_plots=False, save_dir=lab_dir, trace_loss=100.0)
        with open(os.path.join(lab_dir, "mc_dropout_metrics.json"), "w") as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else
                           (float(v) if isinstance(v, (np.floating, np.integer)) else v))
                       for k, v in mcd.items()}, f, indent=2)

    with open(os.path.join(run_dir, "config_summary.json"), "w") as f:
        json.dump({"run_dir": run_dir, "model_dir": model_dir, "model": model_name, "classes": class_names,
                   "channels": channels, "n_frames": T_model, "wasserstein_threshold": threshold,
                   "threshold_source": thr_source, "n_mc": n_mc, "seed": seed, "device": str(device),
                   "sets": {s["set"]: s["n"] for s in summ}}, f, indent=2)
    print(f"\nDone. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
