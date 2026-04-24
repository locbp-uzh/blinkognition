#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Sonnet 4.6)
# __copyright__ = "Copyright 2026, UZH, Switzerland"

"""
Random Forest classifier trained on six handcrafted blinking features extracted
from all Grx1 and K20Ac zscored traces.

Features per trace (computed via GMM ON/OFF state detection):
  1. Mean ON time
  2. Mean OFF time
  3. Std of ON times
  4. Std of OFF times
  5. Total ON time  (active window)
  6. Total OFF time (active window)

Outputs:
  - features.csv              (all traces × features + protein label)
  - confusion_matrix.pdf
  - feature_importances.pdf
  - metrics.json

Usage:
    cd Controls/random_forest
    conda run -n blink2env python rf_classifier.py -c config_rf.yaml
"""

import sys
import os
import json
import argparse
import logging
import importlib.util as _ilu
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)

# --- path setup: ML/utils via normal import, Extraction/utils via importlib ---
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "ML"))

from utils import (   # ML/utils
    discover_protein_files, load_series,
    COLORS, FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_TITLE, FONTSIZE_LEGEND,
    apply_axis_standards, plot_confusion_matrix_with_std,
)

# Load Extraction/utils explicitly to avoid shadowing ML/utils
_ext_spec = _ilu.spec_from_file_location("extraction_utils", str(_root / "Extraction" / "utils.py"))
_ext_utils = _ilu.module_from_spec(_ext_spec)
_ext_spec.loader.exec_module(_ext_utils)
gmm_classify_frames = _ext_utils.gmm_classify_frames

mpl.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':        FONTSIZE_LABEL,
    'pdf.fonttype':     42,
    'savefig.dpi':      450,
    'savefig.bbox':     'tight',
})

FEATURE_NAMES = ["mean_on_ms", "mean_off_ms", "std_on_ms", "std_off_ms",
                 "total_on_ms", "total_off_ms"]
FEATURE_LABELS = {
    "mean_on_ms":   "Mean ON time (ms)",
    "mean_off_ms":  "Mean OFF time (ms)",
    "std_on_ms":    "Std ON time (ms)",
    "std_off_ms":   "Std OFF time (ms)",
    "total_on_ms":  "Total ON time (ms)",
    "total_off_ms": "Total OFF time (ms)",
}


class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _extract_runs(signal_mask, min_peak_width):
    runs, in_run, start = [], False, 0
    for i, s in enumerate(signal_mask):
        if s and not in_run:
            start, in_run = i, True
        elif not s and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(signal_mask) - 1))
    return [(l, r) for l, r in runs if (r - l + 1) >= min_peak_width]


def compute_rf_features(trace_1d, gmm_proba, min_peak_width, frame_interval_ms):
    """Six features from a 1-D zscored trace; returns None if fewer than 2 peaks."""
    _, _, signal_mask = gmm_classify_frames(trace_1d, float(gmm_proba), 0.0)
    runs = _extract_runs(signal_mask, min_peak_width)
    if len(runs) < 2:
        return None

    dur = np.array([r - l + 1 for l, r in runs], dtype=float) * frame_interval_ms
    off = np.array([runs[i+1][0] - runs[i][1] - 1 for i in range(len(runs)-1)],
                   dtype=float) * frame_interval_ms

    return {
        "mean_on_ms":   float(np.mean(dur)),
        "mean_off_ms":  float(np.mean(off)),
        "std_on_ms":    float(np.std(dur, ddof=1)) if len(dur) >= 2 else 0.0,
        "std_off_ms":   float(np.std(off, ddof=1)) if len(off) >= 2 else 0.0,
        "total_on_ms":  float(np.sum(dur)),
        "total_off_ms": float(np.sum(off)),
    }


def extract_all_features(traces_path, proteins, channel, gmm_proba, min_pk, frame_ms):
    rows = []
    for class_idx, protein in enumerate(proteins):
        logging.info("Processing %s ...", protein)
        files = discover_protein_files(traces_path, protein, [channel])
        arr   = load_series(files[0])   # (n_traces, T)
        skipped = 0
        for i in range(arr.shape[0]):
            feat = compute_rf_features(arr[i], gmm_proba, min_pk, frame_ms)
            if feat is None:
                skipped += 1
                continue
            feat["protein"]   = protein
            feat["class_idx"] = class_idx
            rows.append(feat)
        kept = arr.shape[0] - skipped
        logging.info("  %s: %d / %d traces kept (≥2 peaks)", protein, kept, arr.shape[0])
    return pd.DataFrame(rows)


def plot_confusion(y_true, y_pred, class_names, save_path):
    cm  = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize="true")
    std = np.zeros_like(cm)
    plot_confusion_matrix_with_std(cm, std, class_names,
                                   save_path=save_path,
                                   title="Random Forest — confusion matrix (test set)")


def plot_importances(importances, feature_names, save_path):
    order  = np.argsort(importances)
    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order]
    vals   = importances[order]
    color_list = list(COLORS.values())

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(range(len(vals)), vals,
            color=[color_list[i % len(color_list)] for i in range(len(vals))],
            alpha=0.85)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=FONTSIZE_TICK)
    ax.set_xlabel("Mean decrease in impurity", fontsize=FONTSIZE_LABEL)
    ax.set_title("Random Forest — feature importances", fontsize=FONTSIZE_TITLE)
    apply_axis_standards(ax)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    cfg         = load_config(args.config)
    traces_path = cfg["traces_path"]
    proteins    = cfg["proteins"]
    channel     = cfg.get("channel", "zscored")  # zscored for GMM by design; minmax is reserved for the TCN
    output_root = cfg.get("output_root", "../../Results/Controls/RandomForest")
    gmm_proba   = float(cfg.get("gmm", {}).get("proba", 0.7))
    min_pk      = int(cfg.get("gmm", {}).get("min_peak_width", 3))
    frame_ms    = float(cfg.get("gmm", {}).get("frame_interval_ms", 30.0))
    rf_cfg      = cfg.get("random_forest", {})
    n_trees     = int(rf_cfg.get("n_estimators", 500))
    test_size   = float(rf_cfg.get("test_size", 0.20))
    seed        = int(rf_cfg.get("seed", 840410))

    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(output_root, f"{ts}_random_forest")
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, f"random_forest_{'_'.join(proteins)}.log")
    log_file = open(log_path, "w")
    sys.stdout = sys.stderr = Tee(sys.__stdout__, log_file)
    print(f"Logging to: {log_path}")
    print(f"Output: {out_dir}")

    print(f"\nExtracting features from '{channel}' channel...")
    df = extract_all_features(traces_path, proteins, channel, gmm_proba, min_pk, frame_ms)
    print(f"Traces with ≥2 peaks: {len(df)}")
    df.to_csv(os.path.join(out_dir, "features.csv"), index=False)

    X_feat = df[FEATURE_NAMES].values
    y      = df["class_idx"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=test_size, stratify=y, random_state=seed,
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    print(f"Training RandomForest ({n_trees} trees)...")
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    bal_acc = balanced_accuracy_score(y_test, y_pred) * 100.0
    auc     = roc_auc_score(y_test, y_proba)

    print("\n=== Test set results ===")
    print(classification_report(y_test, y_pred, target_names=proteins))
    print(f"Balanced accuracy: {bal_acc:.1f}%")
    print(f"AUC:               {auc:.3f}")

    plot_confusion(y_test, y_pred, proteins,
                   os.path.join(out_dir, "confusion_matrix.pdf"))
    plot_importances(rf.feature_importances_, FEATURE_NAMES,
                     os.path.join(out_dir, "feature_importances.pdf"))

    metrics = {
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "n_traces_total": int(len(df)),
        "proteins": proteins, "channel": channel,
        "feature_names": FEATURE_NAMES,
        "n_estimators": n_trees, "seed": seed,
        "balanced_accuracy_pct": float(bal_acc),
        "auc": float(auc),
        "feature_importances": {
            FEATURE_NAMES[i]: float(rf.feature_importances_[i])
            for i in range(len(FEATURE_NAMES))
        },
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
