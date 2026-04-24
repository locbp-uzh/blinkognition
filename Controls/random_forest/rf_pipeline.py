#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Sonnet 4.6)
# __copyright__ = "Copyright 2026, UZH, Switzerland"

"""
Random Forest pipeline: feature extraction → classifier → label-scramble control → LOO CV.

Features are extracted once and reused across all three analyses. Requires uid_path
in the config for the experiment-based LOO CV and feature distribution plots; if
omitted, only the classifier and scramble sections run.

Outputs (under {output_root}/{ts}_rf_pipeline_{proteins}/):
  features_with_exp.csv
  rf_pipeline_{proteins}.log
  classifier/
    confusion_matrix.pdf
    feature_importances.pdf
    metrics.json
    margin_results/
      margin_histogram.pdf        margin_sweep.pdf
      confusion_matrix_filtered.pdf
      traces_with_margin.npz      margin_metrics.json
  scrambled/
    metrics.json
  loocv/                          (only when uid_path is set)
    feature_distributions.pdf
    per_fold/{exp}/
      metrics.json
      confusion_matrix.pdf
    aggregated/
      loocv_metrics.json
      confusion_matrix_mean_std.pdf

Usage:
    cd Controls/random_forest
    conda run -n blink2env python rf_pipeline.py -c config_rf.yaml
"""

import sys
import os
import json
import argparse
import logging
import importlib.util as _ilu
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "ML"))

from utils import (
    discover_protein_files, load_series, _load_col_names,
    COLORS, FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_TITLE, FONTSIZE_LEGEND,
    apply_axis_standards, plot_confusion_matrix_with_std, _load_cmap,
)

_ext_spec = _ilu.spec_from_file_location("extraction_utils", str(_root / "Extraction" / "utils.py"))
_ext_utils = _ilu.module_from_spec(_ext_spec)
_ext_spec.loader.exec_module(_ext_utils)
gmm_classify_frames = _ext_utils.gmm_classify_frames

_lapaz_cmap = _load_cmap("lapaz")

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


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (shared across all three sections)
# ─────────────────────────────────────────────────────────────────────────────

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


def build_uid_to_exp(uid_path, proteins):
    uid_to_exp = {}
    for protein in proteins:
        pkl = os.path.join(uid_path, f"{protein}_uniqueID_IN.pkl")
        if not os.path.exists(pkl):
            logging.warning("UID file not found: %s", pkl)
            continue
        df = pd.read_pickle(pkl)
        for _, row in df.iterrows():
            uid    = int(row["uniqueID"])
            origin = str(row.get("origin", ""))
            exp    = Path(origin).parent.parent.name if origin else "UNKNOWN"
            uid_to_exp[uid] = exp if exp else "UNKNOWN"
    return uid_to_exp


def extract_features(traces_path, uid_path, proteins, channel, gmm_proba, min_pk, frame_ms):
    """Extract 6 features per trace in parallel. Attaches experiment label if uid_path given."""
    uid_to_exp = build_uid_to_exp(uid_path, proteins) if uid_path else {}
    if uid_to_exp:
        print(f"UID→experiment mapping: {len(uid_to_exp)} entries")

    rows = []
    for class_idx, protein in enumerate(proteins):
        logging.info("Processing %s ...", protein)
        files = discover_protein_files(traces_path, protein, [channel])
        arr   = load_series(files[0])

        uids = _load_col_names(files[0]) if uid_to_exp else None
        if uids is not None and len(uids) != arr.shape[0]:
            raise ValueError(
                f"{protein}: mismatch between trace count ({arr.shape[0]}) "
                f"and UniqueID count ({len(uids)})"
            )

        results = Parallel(n_jobs=-1)(
            delayed(compute_rf_features)(arr[i], gmm_proba, min_pk, frame_ms)
            for i in range(arr.shape[0])
        )
        skipped = 0
        exp_counts = defaultdict(int)
        for i, feat in enumerate(results):
            if feat is None:
                skipped += 1
                continue
            exp = uid_to_exp.get(int(uids[i]), "UNKNOWN") if uids is not None else "UNKNOWN"
            feat["protein"]    = protein
            feat["class_idx"]  = class_idx
            feat["experiment"] = exp
            rows.append(feat)
            exp_counts[exp] += 1

        kept = arr.shape[0] - skipped
        logging.info("  %s: %d / %d traces kept. Exp breakdown: %s",
                     protein, kept, arr.shape[0], dict(exp_counts))

    df = pd.DataFrame(rows)
    if uid_to_exp:
        unknown = (df["experiment"] == "UNKNOWN").sum()
        if unknown:
            print(f"Warning: {unknown} traces have an empty origin path "
                  f"and will be excluded from LOO folds.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def undersample_to_minority(X, y, seed):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    n = counts.min()
    idx = np.concatenate([
        rng.choice(np.where(y == c)[0], size=n, replace=False) for c in classes
    ])
    rng.shuffle(idx)
    return X[idx], y[idx]


def plot_confusion(y_true, y_pred, class_names, save_path, title):
    cm  = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize="true")
    plot_confusion_matrix_with_std(cm, np.zeros_like(cm), class_names,
                                   save_path=save_path, title=title)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Classifier
# ─────────────────────────────────────────────────────────────────────────────

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


def plot_margin_histogram(margins, proteins, save_path):
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.hist(margins, bins=40, color=list(COLORS.values())[0],
            alpha=0.8, edgecolor='white', linewidth=0.4)
    ax.set_xlabel(f"Vote margin  |p({proteins[0]}) − p({proteins[1]})|",
                  fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
    ax.set_title("RF uncertainty — vote margin distribution", fontsize=FONTSIZE_TITLE)
    apply_axis_standards(ax)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_margin_sweep(thresholds, bal_accs, kept_fracs, save_path):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    norm = mcolors.Normalize(vmin=thresholds.min(), vmax=thresholds.max())
    sc = ax.scatter(
        np.array(kept_fracs) * 100.0, bal_accs,
        c=thresholds, cmap=_lapaz_cmap, norm=norm, s=18, zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.04)
    cb.set_label("Margin threshold", fontsize=FONTSIZE_LEGEND)
    cb.ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.set_xlim(105, -5)
    ax.set_xlabel("Retained traces (%)", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Balanced accuracy (%)", fontsize=FONTSIZE_LABEL)
    ax.set_title("Margin sweep — RF confidence filtering", fontsize=FONTSIZE_TITLE)
    apply_axis_standards(ax)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def margin_sweep(y_true, y_pred, margins, trace_loss, save_dir, proteins):
    N = len(margins)
    thresholds  = np.linspace(0.0, 1.0, 50)
    bal_accs, kept_fracs = [], []
    best_thr, best_acc   = 0.0, -np.inf

    for thr in thresholds:
        mask = margins > thr
        kept = int(mask.sum())
        removed_pct = (1.0 - kept / N) * 100.0
        if kept == 0 or len(np.unique(y_true[mask])) < 2:
            bal_accs.append(np.nan)
            kept_fracs.append(kept / N)
            continue
        acc = balanced_accuracy_score(y_true[mask], y_pred[mask]) * 100.0
        bal_accs.append(acc)
        kept_fracs.append(kept / N)
        if removed_pct <= trace_loss and acc > best_acc:
            best_acc, best_thr = acc, float(thr)

    mask_sel    = margins > best_thr
    removed_pct = (1.0 - mask_sel.sum() / N) * 100.0

    print(f"\n=== Margin filtering (trace_loss ≤ {trace_loss:.0f}%) ===")
    print(f"Selected margin threshold: {best_thr:.3f} → "
          f"keeps {mask_sel.sum()}/{N} traces ({removed_pct:.1f}% removed)")
    if mask_sel.sum() > 0 and len(np.unique(y_true[mask_sel])) >= 2:
        print(f"Filtered balanced accuracy: {best_acc:.1f}%")
        print(classification_report(y_true[mask_sel], y_pred[mask_sel],
                                    target_names=proteins))

    plot_margin_histogram(margins, proteins, os.path.join(save_dir, "margin_histogram.pdf"))
    plot_margin_sweep(thresholds, bal_accs, kept_fracs,
                      os.path.join(save_dir, "margin_sweep.pdf"))
    plot_confusion(y_true[mask_sel], y_pred[mask_sel], proteins,
                   os.path.join(save_dir, "confusion_matrix_filtered.pdf"),
                   title=f"RF — confusion matrix (margin > {best_thr:.3f})")
    np.savez(
        os.path.join(save_dir, "traces_with_margin.npz"),
        features=np.array([]),
        labels=y_true, predictions=y_pred, margins=margins,
        class_names=np.array(proteins),
    )
    metrics = {
        "initial_accuracy":    float(balanced_accuracy_score(y_true, y_pred) * 100),
        "filtered_accuracy":   float(best_acc) if best_acc > -np.inf else None,
        "selected_threshold":  float(best_thr),
        "removed_percent":     float(removed_pct),
        "n_total":             int(N),
        "n_retained":          int(mask_sel.sum()),
        "thresholds":          thresholds.tolist(),
        "balanced_accuracies": [float(a) if not np.isnan(a) else None for a in bal_accs],
        "kept_fractions":      [float(f) for f in kept_fracs],
    }
    with open(os.path.join(save_dir, "margin_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def run_classifier(df, proteins, n_trees, test_size, seed, trace_loss, out_dir):
    print("\n" + "=" * 60)
    print("SECTION 1 — Random Forest classifier")
    print("=" * 60)

    margin_dir = os.path.join(out_dir, "margin_results")
    os.makedirs(margin_dir, exist_ok=True)

    X_feat = df[FEATURE_NAMES].values
    y      = df["class_idx"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=test_size, stratify=y, random_state=seed,
    )
    print(f"Before undersampling — Train: {len(X_train)}, Test: {len(X_test)}")
    X_train, y_train = undersample_to_minority(X_train, y_train, seed)
    X_test,  y_test  = undersample_to_minority(X_test,  y_test,  seed)
    print(f"After undersampling  — Train: {len(X_train)}, Test: {len(X_test)}")

    print(f"\nTraining RandomForest ({n_trees} trees)...")
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)

    proba_test = rf.predict_proba(X_test)[:, 1]
    y_pred     = (proba_test >= 0.5).astype(int)
    auc        = roc_auc_score(y_test, proba_test)
    bal_acc    = balanced_accuracy_score(y_test, y_pred) * 100.0

    print("\n=== Test set results ===")
    print(classification_report(y_test, y_pred, target_names=proteins))
    print(f"Balanced accuracy: {bal_acc:.1f}%")
    print(f"AUC:               {auc:.3f}")

    plot_confusion(y_test, y_pred, proteins,
                   os.path.join(out_dir, "confusion_matrix.pdf"),
                   title="Random Forest — confusion matrix (test set)")
    plot_importances(rf.feature_importances_, FEATURE_NAMES,
                     os.path.join(out_dir, "feature_importances.pdf"))

    proba_all = rf.predict_proba(X_test)
    margins   = np.abs(proba_all[:, 0] - proba_all[:, 1])
    print(f"\nMargin stats: mean={margins.mean():.3f}, "
          f"median={np.median(margins):.3f}, min={margins.min():.3f}")
    margin_metrics = margin_sweep(y_test, y_pred, margins, trace_loss, margin_dir, proteins)

    metrics = {
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "n_traces_total": int(len(df)),
        "proteins": proteins, "channel": "zscored",
        "feature_names": FEATURE_NAMES,
        "n_estimators": n_trees, "seed": seed, "balancing": "undersample_to_minority",
        "auc": float(auc),
        "balanced_accuracy_pct": float(bal_acc),
        "margin_filtered_accuracy_pct": margin_metrics["filtered_accuracy"],
        "margin_selected_threshold":    margin_metrics["selected_threshold"],
        "margin_removed_percent":       margin_metrics["removed_percent"],
        "feature_importances": {
            FEATURE_NAMES[i]: float(rf.feature_importances_[i])
            for i in range(len(FEATURE_NAMES))
        },
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Label-scrambling control
# ─────────────────────────────────────────────────────────────────────────────

def run_scrambled(df, proteins, n_trees, test_size, seed, out_dir):
    print("\n" + "=" * 60)
    print("SECTION 2 — Label-scrambling control")
    print("=" * 60)
    print("*** Labels randomly shuffled — expected balanced accuracy ≈ 50% ***")

    X_feat = df[FEATURE_NAMES].values
    y      = df["class_idx"].values.copy()

    rng = np.random.default_rng(seed)
    rng.shuffle(y)
    print(f"Class distribution after scrambling: "
          f"{ {proteins[i]: int((y == i).sum()) for i in range(len(proteins))} }")

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=test_size, stratify=y, random_state=seed,
    )
    X_train, y_train = undersample_to_minority(X_train, y_train, seed)
    X_test,  y_test  = undersample_to_minority(X_test,  y_test,  seed)
    print(f"Train: {len(X_train)}, Test: {len(X_test)} (after undersampling)")

    rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)

    proba_test = rf.predict_proba(X_test)[:, 1]
    y_pred     = (proba_test >= 0.5).astype(int)
    bal_acc    = balanced_accuracy_score(y_test, y_pred) * 100.0
    auc        = roc_auc_score(y_test, proba_test)

    print("\n=== Test set results (scrambled labels) ===")
    print(classification_report(y_test, y_pred, target_names=proteins))
    print(f"Balanced accuracy: {bal_acc:.1f}%  (expected ≈ 50%)")
    print(f"AUC:               {auc:.3f}  (expected ≈ 0.50)")

    plot_confusion(y_test, y_pred, proteins,
                   os.path.join(out_dir, "confusion_matrix.pdf"),
                   title="RF scrambled — confusion matrix (test set)")

    metrics = {
        "scrambled": True,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "n_traces_total": int(len(df)),
        "proteins": proteins, "n_estimators": n_trees, "seed": seed,
        "balanced_accuracy_pct": float(bal_acc),
        "auc": float(auc),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Experiment-based LOO CV
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_distributions(df, proteins, save_path):
    experiments   = sorted([e for e in df["experiment"].unique() if e != "UNKNOWN"])
    n_exp         = len(experiments)
    color_list    = list(COLORS.values())
    protein_colors = {p: color_list[i % len(color_list)] for i, p in enumerate(proteins)}
    offsets       = np.linspace(-0.2, 0.2, len(proteins))

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    for ax_idx, feat in enumerate(FEATURE_NAMES):
        ax = axes[ax_idx]
        for p_idx, protein in enumerate(proteins):
            sub = df[df["protein"] == protein]
            positions, data = [], []
            for e_idx, exp in enumerate(experiments):
                vals = sub[sub["experiment"] == exp][feat].values
                if len(vals) == 0:
                    continue
                positions.append(e_idx + offsets[p_idx])
                data.append(vals)
            if not data:
                continue
            parts = ax.violinplot(data, positions=positions,
                                  widths=0.35, showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(protein_colors[protein])
                pc.set_alpha(0.65)
            parts["cmedians"].set_color(protein_colors[protein])
            parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks(range(n_exp))
        ax.set_xticklabels(experiments, fontsize=FONTSIZE_TICK, rotation=30, ha="right")
        ax.set_ylabel(FEATURE_LABELS[feat], fontsize=FONTSIZE_LABEL)
        apply_axis_standards(ax)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=protein_colors[p], label=p, alpha=0.75) for p in proteins]
    fig.legend(handles=handles, loc="upper right", fontsize=FONTSIZE_LEGEND,
               frameon=False, ncol=len(proteins))
    fig.suptitle("Feature distributions by experiment and protein", fontsize=FONTSIZE_TITLE)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print("Feature distribution plot saved.")


def run_loocv(df, proteins, n_trees, seed, trace_loss, out_dir):
    experiments  = sorted([e for e in df["experiment"].unique() if e != "UNKNOWN"])
    n_classes    = len(proteins)
    per_fold_dir = os.path.join(out_dir, "per_fold")
    os.makedirs(per_fold_dir, exist_ok=True)

    fold_metrics, cms, cms_margin = [], [], []

    for test_exp in experiments:
        fold_dir = os.path.join(per_fold_dir, test_exp)
        os.makedirs(fold_dir, exist_ok=True)

        df_test  = df[df["experiment"] == test_exp]
        df_train = df[df["experiment"] != test_exp]

        X_train = df_train[FEATURE_NAMES].values
        y_train = df_train["class_idx"].values
        X_test  = df_test[FEATURE_NAMES].values
        y_test  = df_test["class_idx"].values

        if len(np.unique(y_train)) < n_classes or len(np.unique(y_test)) < n_classes:
            print(f"  [{test_exp}] Skipping — not all classes present in train or test.")
            continue

        X_train, y_train = undersample_to_minority(X_train, y_train, seed)
        X_test,  y_test  = undersample_to_minority(X_test,  y_test,  seed)

        rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)

        proba  = rf.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)
        bal_acc = balanced_accuracy_score(y_test, y_pred) * 100.0
        auc     = roc_auc_score(y_test, proba)

        print(f"  [{test_exp}] train={len(X_train)}, test={len(X_test)} | "
              f"bal_acc={bal_acc:.1f}%  AUC={auc:.3f}")

        cm = confusion_matrix(y_test, y_pred, labels=range(n_classes), normalize="true")
        cms.append(cm)

        plot_confusion_matrix_with_std(
            cm, np.zeros_like(cm), proteins,
            save_path=os.path.join(fold_dir, "confusion_matrix.pdf"),
            title=f"RF LOO — test={test_exp}",
        )

        # Margin results for this fold
        proba_all = rf.predict_proba(X_test)
        margins   = np.abs(proba_all[:, 0] - proba_all[:, 1])
        margin_dir = os.path.join(fold_dir, "margin_results")
        os.makedirs(margin_dir, exist_ok=True)
        margin_m = margin_sweep(y_test, y_pred, margins, trace_loss, margin_dir, proteins)

        best_thr  = margin_m["selected_threshold"]
        mask_sel  = margins > best_thr
        if mask_sel.sum() > 0 and len(np.unique(y_test[mask_sel])) >= n_classes:
            cm_m = confusion_matrix(y_test[mask_sel], y_pred[mask_sel],
                                    labels=range(n_classes), normalize="true")
            cms_margin.append(cm_m)

        fold_m = {
            "test_experiment": test_exp,
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "balanced_accuracy_pct": float(bal_acc),
            "auc": float(auc),
            "margin_filtered_accuracy_pct": margin_m["filtered_accuracy"],
            "margin_selected_threshold":    margin_m["selected_threshold"],
            "margin_removed_percent":       margin_m["removed_percent"],
        }
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(fold_m, f, indent=2)
        fold_metrics.append(fold_m)

    if not fold_metrics:
        print("No valid folds — cannot aggregate.")
        return

    bal_accs        = [m["balanced_accuracy_pct"] for m in fold_metrics]
    aucs            = [m["auc"] for m in fold_metrics]
    margin_accs     = [m["margin_filtered_accuracy_pct"] for m in fold_metrics
                       if m["margin_filtered_accuracy_pct"] is not None]
    cm_mean         = np.mean(cms, axis=0)
    cm_std          = np.std(cms, axis=0)

    print(f"\nLOOCV summary ({len(fold_metrics)} folds):")
    print(f"  Balanced accuracy:         {np.mean(bal_accs):.1f}% ± {np.std(bal_accs):.1f}%")
    print(f"  AUC:                       {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    if margin_accs:
        print(f"  Margin-filtered accuracy:  {np.mean(margin_accs):.1f}% ± {np.std(margin_accs):.1f}%")

    agg_dir        = os.path.join(out_dir, "aggregated")
    agg_margin_dir = os.path.join(agg_dir, "margin_results")
    os.makedirs(agg_dir, exist_ok=True)
    os.makedirs(agg_margin_dir, exist_ok=True)

    plot_confusion_matrix_with_std(
        cm_mean, cm_std, proteins,
        save_path=os.path.join(agg_dir, "confusion_matrix_mean_std.pdf"),
        title="RF LOO — aggregated confusion matrix (mean ± std)",
    )

    if cms_margin:
        cm_m_mean = np.mean(cms_margin, axis=0)
        cm_m_std  = np.std(cms_margin, axis=0)
        plot_confusion_matrix_with_std(
            cm_m_mean, cm_m_std, proteins,
            save_path=os.path.join(agg_margin_dir, "confusion_matrix_margin_mean_std.pdf"),
            title="RF LOO — aggregated margin-filtered CM (mean ± std)",
        )

    agg_metrics = {
        "n_folds": len(fold_metrics),
        "experiments": [m["test_experiment"] for m in fold_metrics],
        "per_fold_balanced_accuracy_pct": bal_accs,
        "per_fold_auc": aucs,
        "mean_balanced_accuracy_pct": float(np.mean(bal_accs)),
        "std_balanced_accuracy_pct":  float(np.std(bal_accs)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc":  float(np.std(aucs)),
        "mean_margin_filtered_accuracy_pct": float(np.mean(margin_accs)) if margin_accs else None,
        "std_margin_filtered_accuracy_pct":  float(np.std(margin_accs))  if margin_accs else None,
        "per_fold": fold_metrics,
    }
    with open(os.path.join(agg_dir, "loocv_metrics.json"), "w") as f:
        json.dump(agg_metrics, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    cfg         = load_config(args.config)
    traces_path = cfg["traces_path"]
    uid_path    = cfg.get("uid_path", None)
    proteins    = cfg["proteins"]
    channel     = cfg.get("channel", "zscored")
    output_root = cfg.get("output_root", "../../Results/Controls/RandomForest")
    gmm_proba   = float(cfg.get("gmm", {}).get("proba", 0.7))
    min_pk      = int(cfg.get("gmm", {}).get("min_peak_width", 3))
    frame_ms    = float(cfg.get("gmm", {}).get("frame_interval_ms", 30.0))
    rf_cfg      = cfg.get("random_forest", {})
    n_trees     = int(rf_cfg.get("n_estimators", 500))
    test_size   = float(rf_cfg.get("test_size", 0.20))
    seed        = int(rf_cfg.get("seed", 840410))
    trace_loss  = float(rf_cfg.get("trace_loss", 50.0))

    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_root, f"{ts}_rf_pipeline_{'_'.join(proteins)}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, f"rf_pipeline_{'_'.join(proteins)}.log")
    log_file = open(log_path, "w")
    sys.stdout = sys.stderr = Tee(sys.__stdout__, log_file)
    print(f"Output: {run_dir}")

    # ── Feature extraction (done once) ───────────────────────────────────────
    print(f"\nExtracting features from '{channel}' channel...")
    df = extract_features(traces_path, uid_path, proteins, channel,
                          gmm_proba, min_pk, frame_ms)
    print(f"Traces with ≥2 peaks: {len(df)}")
    df.to_csv(os.path.join(run_dir, "features_with_exp.csv"), index=False)

    if uid_path:
        experiments = sorted([e for e in df["experiment"].unique() if e != "UNKNOWN"])
        print(f"\nExperiments found: {experiments}")
        for exp in experiments:
            sub = df[df["experiment"] == exp]
            counts = {p: int((sub["protein"] == p).sum()) for p in proteins}
            print(f"  {exp}: {counts}")

    # ── Three analyses on the same feature table ──────────────────────────────
    clf_dir      = os.path.join(run_dir, "classifier")
    scramble_dir = os.path.join(run_dir, "scrambled")
    os.makedirs(clf_dir, exist_ok=True)
    os.makedirs(scramble_dir, exist_ok=True)

    run_classifier(df, proteins, n_trees, test_size, seed, trace_loss, clf_dir)
    run_scrambled(df, proteins, n_trees, test_size, seed, scramble_dir)

    if uid_path:
        loocv_dir = os.path.join(run_dir, "loocv")
        os.makedirs(loocv_dir, exist_ok=True)
        print("\n" + "=" * 60)
        print("SECTION 3 — Experiment-based LOO cross-validation")
        print("=" * 60)
        print("\nPlotting feature distributions...")
        plot_feature_distributions(
            df, proteins,
            save_path=os.path.join(loocv_dir, "feature_distributions.pdf"),
        )
        print("\nRunning LOO CV...")
        run_loocv(df, proteins, n_trees, seed, trace_loss, loocv_dir)
    else:
        print("\nSkipping LOO CV (no uid_path in config).")

    print(f"\nDone. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
