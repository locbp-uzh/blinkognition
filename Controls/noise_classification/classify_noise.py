#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Sonnet 4.6)
# __copyright__ = "Copyright 2026, UZH, Switzerland"

"""
Classify background (noise) traces using a pre-trained TCN model.

Loads the model saved in pretrained_model_dir, runs a deterministic forward
pass and MC Dropout (100 passes) on background traces from both Grx1 and K20Ac.
Produces the same rich inference outputs as ML/train.py.

Outputs (inside a timestamped run directory):
  - predicted_class_distribution.pdf   — mean predicted probability per background protein
  - confusion_matrix/                  — standard confusion matrix from evaluate_model
  - test_metrics.json                  — full per-class metrics from evaluate_model
  - MCD_results/                       — full evaluate_uncertainty_filtered outputs
      wasserstein_histogram.pdf, wd_sweep.pdf, traces_with_wasserstein.npz
  - mc_dropout_metrics.json
  - config_full.json                   — this run's config
  - config_summary.json
  - <run_name>.log                     — full stdout log

Usage:
    cd Controls/noise_classification
    conda run -n blink2env python classify_noise.py -c config_noise.yaml
"""

import sys
import os
import json
import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- path setup ---
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "ML"))

from utils import (
    discover_protein_files,
    load_series,
    mc_dropout_predict,
    evaluate_model,
    evaluate_uncertainty_filtered,
    detect_accelerator,
    setup_precision_and_flags,
    dataloader_kwargs_for,
    batch_size_hint,
    estimate_batch_size,
    COLORS,
    FONTSIZE_LABEL,
    FONTSIZE_TICK,
    FONTSIZE_TITLE,
    apply_axis_standards,
)
from models import build_model as build_from_zoo

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': FONTSIZE_LABEL,
    'pdf.fonttype': 42,
    'savefig.dpi': 450,
    'savefig.bbox': 'tight',
})


class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s or "run").strip("_")


def load_background_dataset(background_path, proteins, channels):
    """Load background traces for each protein; returns X (N,C,T), y (N,), class_names."""
    X_list, y_list = [], []
    min_T = None

    for prot in proteins:
        files = discover_protein_files(background_path, prot, channels)
        arrs  = [load_series(f) for f in files]
        t = arrs[0].shape[-1]
        min_T = t if min_T is None else min(min_T, t)

    for class_idx, prot in enumerate(proteins):
        files = discover_protein_files(background_path, prot, channels)
        arrs  = [load_series(f)[:, :min_T] for f in files]
        shapes = [a.shape for a in arrs]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(f"Shape mismatch for {prot}: {shapes}")
        X_i = np.stack(arrs, axis=1)          # (N, C, T)
        y_i = np.full(X_i.shape[0], class_idx, dtype=np.int64)
        X_list.append(X_i)
        y_list.append(y_i)
        print(f"  {prot}: {X_i.shape[0]} background traces")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y, proteins


def plot_prediction_distribution(probs_mean, y_true, class_names, save_path):
    """Bar chart: mean predicted probability per true background label (noise-specific plot)."""
    fig, axes = plt.subplots(1, len(class_names),
                             figsize=(3.5 * len(class_names), 3.0), squeeze=False)
    color_list = list(COLORS.values())

    for col, true_cls in enumerate(class_names):
        ax = axes[0][col]
        mask = y_true == col
        probs_sub = probs_mean[mask]
        means = probs_sub.mean(axis=0)
        sems  = probs_sub.std(axis=0) / np.sqrt(max(len(probs_sub), 1))
        ax.bar(class_names, means,
               color=[color_list[i % len(color_list)] for i in range(len(class_names))],
               width=0.5, alpha=0.85)
        ax.errorbar(range(len(class_names)), means, yerr=sems, fmt='none',
                    color='#333333', capsize=3, linewidth=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean predicted probability", fontsize=FONTSIZE_LABEL)
        ax.set_title(f"Background: {true_cls}\n(n={mask.sum()})", fontsize=FONTSIZE_TITLE)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=FONTSIZE_TICK)
        apply_axis_standards(ax)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_dir    = cfg["pretrained_model_dir"]
    bg_path      = cfg["background_traces_path"]
    proteins     = cfg["proteins"]
    channels     = cfg["channels"]
    output_root  = cfg.get("output_root", "../../Results/Controls/NoiseClassification")
    seed         = int(cfg.get("system", {}).get("seed", 840410))
    verbose      = bool(cfg.get("system", {}).get("verbose", False))
    num_workers  = int(cfg.get("system", {}).get("num_workers", 0))
    mcd_cfg      = cfg.get("mc_dropout", {})
    n_mc         = int(mcd_cfg.get("n_mc", 100))
    trace_loss   = float(mcd_cfg.get("trace_loss", 50.0))
    wd_threshold = mcd_cfg.get("wasserstein_threshold")
    if wd_threshold is not None:
        wd_threshold = float(wd_threshold)

    # Output directory and logging
    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_root, f"{ts}_noise_classification")
    mcd_dir = os.path.join(run_dir, "MCD_results")
    conf_matrix_dir = os.path.join(run_dir, "confusion_matrix")
    for d in (run_dir, mcd_dir, conf_matrix_dir):
        os.makedirs(d, exist_ok=True)

    log_path = os.path.join(run_dir, f"noise_classification_{_slug('_'.join(proteins))}.log")
    log_file = open(log_path, "w")
    sys.stdout = sys.stderr = Tee(sys.__stdout__, log_file)
    print(f"Logging to: {log_path}")

    with open(os.path.join(run_dir, "config_full.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Output directory: {run_dir}")
    print(f"Pre-trained model: {model_dir}")
    print("*** NOISE CLASSIFICATION CONTROL — inference on background traces ***")

    # Load training config to reconstruct model
    config_path = os.path.join(model_dir, "config_full.json")
    with open(config_path) as f:
        train_cfg = json.load(f)

    train_data_cfg  = train_cfg.get("data", {})
    train_model_cfg = train_cfg.get("model", {})
    train_dataset   = train_data_cfg.get("dataset", {})
    class_names     = list(train_dataset.keys())
    in_channels     = len(channels)
    num_classes     = len(class_names)
    model_name      = train_model_cfg.get("name", "tcn")

    print(f"Model: {model_name}, classes: {class_names}, in_channels: {in_channels}")

    # Try to load optimal_threshold from the training run
    optimal_threshold = None
    for cand in ("test_metrics.json",):
        cand_path = os.path.join(model_dir, cand)
        if os.path.exists(cand_path):
            with open(cand_path) as f:
                saved = json.load(f)
            optimal_threshold = saved.get("optimal_threshold")
            if optimal_threshold is not None:
                print(f"Loaded optimal_threshold={optimal_threshold:.4f} from {cand}")
            break

    # Accelerator
    accel = detect_accelerator()
    device = accel["device"]
    print(f"Device: {accel['name']} ({accel['type']})")
    _, autocast_ctx, _ = setup_precision_and_flags(accel)
    dl_kwargs = dataloader_kwargs_for(accel)

    # Build model and load weights
    build_kwargs = {"in_channels": in_channels, "num_classes": num_classes}
    if model_name.lower() == "tcn":
        build_kwargs.pop("in_channels", None)
        build_kwargs["num_inputs"] = in_channels
    model = build_from_zoo(model_name, **build_kwargs).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pth"), map_location=device))
    model.eval()
    print("Model loaded.")

    # Load background traces and build DataLoader
    print("\nLoading background traces...")
    X, y_true, _ = load_background_dataset(bg_path, proteins, channels)
    print(f"Total background traces: {X.shape[0]}  shape: {X.shape}")

    X_t   = torch.from_numpy(X).float()
    y_t   = torch.from_numpy(y_true).long()
    batch_size = int(train_cfg.get("optimization", {}).get("batch_size", 64))
    extra = {"persistent_workers": True} if num_workers > 0 else {}
    bg_loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, **extra,
    )

    # evaluate_model: confusion matrix + full per-class metrics
    print("\nEvaluating on background traces (deterministic)...")
    test_metrics = evaluate_model(
        model, bg_loader, device,
        label_names=class_names,
        save_path=conf_matrix_dir,
        autocast_ctx=autocast_ctx,
        optimal_threshold=optimal_threshold,
    )

    metrics_json = {}
    for k, v in test_metrics.items():
        if isinstance(v, (float, int)):
            metrics_json[k] = v if not (isinstance(v, float) and np.isnan(v)) else None
        elif isinstance(v, (np.floating, np.integer)):
            metrics_json[k] = float(v) if not np.isnan(v) else None
        elif isinstance(v, dict):
            metrics_json[k] = {str(kk): float(vv) if not np.isnan(vv) else None
                               for kk, vv in v.items()}
        else:
            metrics_json[k] = v
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Test metrics saved.")

    # MC Dropout
    print(f"\nRunning MC Dropout ({n_mc} passes)...")
    bs_hint = batch_size_hint(batch_size, accel)
    est_bs  = estimate_batch_size(model_name=model_name) if (accel["type"] == "cuda" and bs_hint is None) else (bs_hint or batch_size)

    with torch.no_grad():
        probs_mc = mc_dropout_predict(
            model, X_t,
            n_mc=n_mc,
            batch_size=est_bs,
            device=device,
            autocast_ctx=autocast_ctx,
        )

    probs_mean = probs_mc.mean(axis=0)

    # Noise-specific plot: predicted class distribution per background protein
    plot_prediction_distribution(
        probs_mean, y_true, class_names,
        os.path.join(run_dir, "predicted_class_distribution.pdf"),
    )

    # Background traces have no UniqueIDs — fill with -1
    unique_ids_bg = np.full(len(X_t), -1, dtype=np.int64)

    mcd_metrics = evaluate_uncertainty_filtered(
        probs_mc,
        y_true=y_true,
        threshold=wd_threshold,
        class_names=class_names,
        show_plots=False,
        save_dir=mcd_dir,
        trace_loss=trace_loss,
        verbose=verbose,
        optimal_threshold=optimal_threshold,
        traces=X_t,
        unique_ids=unique_ids_bg,
    )

    mcd_out = {}
    for k, v in mcd_metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            mcd_out[k] = float(v) if not np.isnan(v) else None
        elif isinstance(v, (np.ndarray, list)):
            mcd_out[k] = v.tolist() if isinstance(v, np.ndarray) else v
        else:
            mcd_out[k] = v
    with open(os.path.join(mcd_dir, "mc_dropout_metrics.json"), "w") as f:
        json.dump(mcd_out, f, indent=2)
    print("MC Dropout evaluation saved.")

    with open(os.path.join(run_dir, "config_summary.json"), "w") as f:
        json.dump({
            "run_dir": run_dir,
            "model": model_name,
            "pretrained_model_dir": model_dir,
            "control": "noise_classification",
            "num_classes": num_classes,
            "in_channels": in_channels,
            "n_background_traces": int(len(y_true)),
            "device": str(device),
        }, f, indent=2)

    print(f"\nDone. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
