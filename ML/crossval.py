#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH), Andreas Biri (ETHZ) and Roman Briskine (UZH).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
crossval.py

Definitions first: cross-validation (CV), area under the receiver operating characteristic curve (AUC),
Monte Carlo (MC), comma-separated values (CSV), temporal convolutional network (TCN),
convolutional gated recurrent unit (ConvGRU), Wasserstein distance (WD).

Purpose:
- Run K-fold CV to compare multiple models registered in your model zoo.
- Record per-fold metrics before MC dropout.
- Run MC dropout and sweep WD thresholds; select the best threshold subject to at most 50% sample loss.
- Aggregate confusion matrices (mean and standard deviation) before and after MC dropout.
- Write all results to CSV files for downstream analysis.

Usage:
    python crossval.py -c config_cv.yaml --models tcn,resnet1d,orig_conv_gru --k 5
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
import json
import time
import functools
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import torch.multiprocessing as mp

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

import yaml

# Project utilities
from utils import (
    build_dataset,
    build_dataset_from_keys,
    mc_dropout_predict,
    plot_confusion_matrix_with_std,
    plot_losses,
    optimize_binary_threshold,
    optimize_multiclass_thresholds,
    apply_threshold,
    apply_axis_standards,
    apply_augmentation,
    COLORS,
    FONTSIZE_LABEL,
    FONTSIZE_TICK,
    FONTSIZE_TITLE,
    FONTSIZE_LEGEND,
    _load_cmap,
)
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

# Matplotlib configuration (consistent with plotting standards)
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 7
from utils import (
    detect_accelerator,
    setup_precision_and_flags,
    dataloader_kwargs_for,
    maybe_compile,
    batch_size_hint,
)
from models import build_model as zoo_build


_lipari_cmap = _load_cmap("lipari")
_lapaz_cmap  = _load_cmap("lapaz")


# ---------------------------- helpers ---------------------------- #

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()


def _slug(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")



def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_run_dir(output_root: str, run_name: str, dataset_keys, tag: str = "cv") -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = f"{ts}_{_slug(run_name)}_{tag}_{_slug('_'.join(dataset_keys))}"
    path = os.path.join(output_root, d)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "MCD_results"), exist_ok=True)
    return path


def get_dataset(cfg: dict) -> dict:
    data_cfg = cfg.get("data", {})
    if "datasets" in data_cfg and data_cfg.get("use"):
        name = data_cfg["use"]
        datasets = data_cfg["datasets"]
        if name not in datasets:
            raise KeyError(f"data.use='{name}' not found in data.datasets")
        return datasets[name]
    if "dataset" in data_cfg:
        return data_cfg["dataset"]
    raise KeyError("No dataset specified. Provide data.dataset or data.datasets+data.use in config.")


def stratified_kfold_indices(y: np.ndarray, n_splits: int, shuffle: bool, seed: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros_like(y), y), start=1):
        yield fold, train_idx, val_idx


def to_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)
    # Expect X as (N, C, T) or (N, T); enforce channel-first tensor
    if X.ndim == 2:
        X = X[:, None, :]
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    return TensorDataset(X_t, y_t)


def _balance_dataset(X, y, random_seed=42):
    """
    Balance a dataset by subsampling majority classes to match the minority class size.

    Args:
        X: Data array of shape (N, ...)
        y: Labels array of shape (N,)
        random_seed: Random seed for reproducibility

    Returns:
        X_balanced, y_balanced: Balanced arrays
    """
    rng = np.random.default_rng(seed=random_seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    balanced_indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > min_count:
            selected = rng.choice(cls_indices, size=min_count, replace=False)
        else:
            selected = cls_indices
        balanced_indices.extend(selected)

    balanced_indices = np.array(balanced_indices)
    rng.shuffle(balanced_indices)

    return X[balanced_indices], y[balanced_indices]


def build_dataloaders_from_indices(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    balance_train: bool,
    balance_test: bool,
    dl_kwargs: dict,
    random_seed: int = 42,
):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Balance validation/test set if requested
    if balance_test:
        original_val_size = len(y_val)
        X_val, y_val = _balance_dataset(X_val, y_val, random_seed=random_seed)
        from collections import Counter
        print(f"  Balanced val/test set: {original_val_size} -> {len(y_val)} samples")
        print(f"    Val set class distribution: {Counter(y_val.tolist())}")

    train_ds = to_tensor_dataset(X_train, y_train)
    val_ds   = to_tensor_dataset(X_val,   y_val)

    extra = {}
    if dl_kwargs.get("num_workers", 0) > 0 and "persistent_workers" not in dl_kwargs:
        extra["persistent_workers"] = True

    if balance_train:
        counts = np.bincount(y_train)
        counts[counts == 0] = 1
        class_weights = 1.0 / counts
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **dl_kwargs, **extra)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs, **extra)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, **dl_kwargs, **extra)
    return train_loader, val_loader, train_ds, val_ds


def eval_on_loader(model, loader, device, autocast_ctx):
    """Return y_true, probs, avg_loss."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    y_true, probs = [], []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast_ctx():
                logits = model(xb)
                loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)
            p = F.softmax(logits, dim=1).detach().cpu().float().numpy()
            probs.append(p)
            y_true.append(yb.detach().cpu().numpy())
    y_true = np.concatenate(y_true, axis=0) if y_true else np.array([])
    probs = np.concatenate(probs, axis=0) if probs else np.array([])
    avg_loss = total_loss / max(1, total_n)
    return y_true, probs, avg_loss


def multiclass_auc(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    if n_classes == 2:
        return roc_auc_score(y_true, probs[:, 1])
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    return roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")


def confusion_normalized(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return cm.astype(np.float32)


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute 1D WD between two empirical samples without SciPy.
    Uses quantile interpolation on a common grid.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size == 0 and y.size == 0:
        return 0.0
    if x.size == 0:
        return float(np.mean(np.abs(np.sort(y))))
    if y.size == 0:
        return float(np.mean(np.abs(np.sort(x))))
    m = int(max(x.size, y.size))
    # mid-quantiles avoid endpoints instabilities
    q = (np.arange(m) + 0.5) / m
    qx = np.quantile(x, q)
    qy = np.quantile(y, q)
    return float(np.mean(np.abs(qx - qy)))


def wd_sweep_and_select(probs_mc: np.ndarray, y_true: np.ndarray, trace_loss: float = 50.0,
                        wasserstein_threshold: float = None):
    """
    Given MC prediction probabilities with shape (M, N, C), compute per-sample WD to the nearest competitor
    and sweep thresholds.

    Args:
        probs_mc: MC dropout predictions (n_mc, N, C)
        y_true: True labels (N,)
        trace_loss: Maximum trace loss % for auto threshold selection (ignored if wasserstein_threshold is set)
        wasserstein_threshold: Optional fixed Wasserstein threshold (overrides trace_loss auto-selection)

    Returns:
        thresholds: (T,)
        bal_accs: (T,) in percent
        kept_fracs: (T,) in [0,1]
        selected_threshold: float
        mask_selected: (N,) boolean mask where samples are kept
        y_pred_mean: (N,) predicted class indices from mean probs
    """
    assert probs_mc.ndim == 3, "probs_mc must be (n_mc, N, C)"
    mean_probs = probs_mc.mean(axis=0)  # (N, C)
    y_pred = mean_probs.argmax(axis=1)
    N, C = mean_probs.shape

    # WD per sample vs closest other class
    w_dists = np.zeros(N, dtype=np.float32)
    for i in range(N):
        pred = y_pred[i]
        dmin = np.inf
        for c in range(C):
            if c == pred:
                continue
            d = wasserstein_1d(probs_mc[:, i, pred], probs_mc[:, i, c])
            if d < dmin:
                dmin = d
        w_dists[i] = 0.0 if not np.isfinite(dmin) else dmin

    max_wd = float(w_dists.max()) if w_dists.size else 1.0
    thresholds = np.linspace(0.0, max_wd, 50, dtype=np.float32)
    bal_accs = []
    kept_fracs = []
    for thr in thresholds:
        mask = w_dists > thr
        kept = int(mask.sum())
        if kept == 0:
            bal_accs.append(np.nan)
            kept_fracs.append(0.0)
            continue
        if len(np.unique(y_true[mask])) < 2:
            bal_accs.append(np.nan)
            kept_fracs.append(kept / len(y_true))
            continue
        acc = balanced_accuracy_score(y_true[mask], y_pred[mask]) * 100.0
        bal_accs.append(acc)
        kept_fracs.append(kept / len(y_true))

    # Select threshold: use fixed value if provided, otherwise auto-select
    if wasserstein_threshold is not None:
        best_thr = float(wasserstein_threshold)
    else:
        # Auto-select: max accuracy subject to <= trace_loss% removals
        best_thr = 0.0
        best_acc = -np.inf
        for thr, acc, frac in zip(thresholds, bal_accs, kept_fracs):
            removed_pct = (1.0 - frac) * 100.0
            if np.isnan(acc) or removed_pct > trace_loss:
                continue
            if acc > best_acc:
                best_acc = acc
                best_thr = float(thr)

    mask_sel = w_dists > best_thr
    return thresholds, np.array(bal_accs), np.array(kept_fracs), best_thr, mask_sel, y_pred


def aggregate_confusions(cms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(cms, axis=0)  # (K, C, C)
    return stack.mean(axis=0), stack.std(axis=0, ddof=1 if stack.shape[0] > 1 else 0)


def long_confusion_df(mean_cm: np.ndarray, std_cm: np.ndarray, class_names: list[str], model_name: str, dataset_tag: str, chosen_wd: float | None):
    rows = []
    C = len(class_names)
    for i in range(C):
        for j in range(C):
            rows.append({
                "dataset": dataset_tag,
                "model": model_name,
                "row_cls": class_names[i],
                "col_cls": class_names[j],
                "mean": float(mean_cm[i, j]),
                "std": float(std_cm[i, j]),
                **({"chosen_wd": chosen_wd} if chosen_wd is not None else {}),
            })
    return pd.DataFrame(rows)


def _plot_cm_with_std(mean_cm: np.ndarray, std_cm: np.ndarray, class_names: list[str], out_path: str, title: str):
    """Plot a confusion matrix showing mean±std in each cell."""
    plot_confusion_matrix_with_std(mean_cm, std_cm, class_names, save_path=out_path, title=title)


# ---------------------------- Multi-GPU Worker ---------------------------- #

def _run_single_task(
    gpu_id: int,
    model_name: str,
    aug_factor: int,
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    n_splits: int,
    shuffle: bool,
    seed: int,
    cfg: dict,
    run_dir: str,
    accel: dict,
    amp_dtype,
    autocast_ctx,
    scaler,
    dl_kwargs: dict,
    device,
) -> dict:
    """
    Run K-fold CV for a single (model, aug_factor) combination.

    Returns the result dict directly (used by worker loop).
    """
    from utils import maybe_compile, batch_size_hint

    aug_label = "noaug" if aug_factor == 0 else f"aug{aug_factor}x"
    print(f"[GPU {gpu_id}] Starting {model_name} + {aug_label}")

    # Extract config sections
    data_cfg = cfg.get("data", {})
    opt_cfg = cfg.get("optimization", {})
    unc_cfg = cfg.get("uncertainty", {})
    aug_sweep_cfg = cfg.get("augmentation_sweep", {})

    # Data parameters (check data_cfg first, fall back to top-level cfg for CV configs)
    balance_train = bool(data_cfg.get("balance_train", cfg.get("balance_train", True)))
    balance_test = bool(data_cfg.get("balance_test", cfg.get("balance_test", False)))
    balance_val = bool(data_cfg.get("balance_val", cfg.get("balance_val", balance_test)))

    # Batch size
    default_bs = int(opt_cfg.get("batch_size", 64))
    bs_eff = batch_size_hint(default_bs, accel) or default_bs

    # Optimizer defaults
    base_lr = float(opt_cfg.get("lr", 4e-4))
    base_wd = float(opt_cfg.get("weight_decay", 4e-4))
    max_epochs = int(opt_cfg.get("max_epochs", 100))
    patience_limit = int(opt_cfg.get("patience_limit", 10))
    label_smoothing = float(opt_cfg.get("label_smoothing", 0.0) or 0.0)
    threshold_metric = str(opt_cfg.get("threshold_metric", "argmax"))

    # Early stopping configuration
    early_stop_cfg = opt_cfg.get("early_stopping", {})
    auc_min_delta = float(early_stop_cfg.get("auc_min_delta", 0.005))
    loss_mode = str(early_stop_cfg.get("loss_mode", "relative"))
    loss_tolerance = float(early_stop_cfg.get("loss_tolerance", 1.10))
    # Alternative checkpoint: save if AUC within tolerance AND loss improves significantly
    auc_tolerance = early_stop_cfg.get("auc_tolerance")
    if auc_tolerance is not None:
        auc_tolerance = float(auc_tolerance)
    loss_min_delta = early_stop_cfg.get("loss_min_delta")
    if loss_min_delta is not None:
        loss_min_delta = float(loss_min_delta)
    alt_checkpoint_enabled = (auc_tolerance is not None) and (loss_min_delta is not None)

    # Uncertainty
    mc_do = unc_cfg.get("mc_dropout", {})
    mc_enabled = bool(mc_do.get("enabled", True))
    n_mc = int(mc_do.get("n_mc", 100))
    trace_loss = float(mc_do.get("trace_loss", 50.0))
    wasserstein_threshold_cfg = mc_do.get("wasserstein_threshold", None)
    wasserstein_threshold = float(wasserstein_threshold_cfg) if wasserstein_threshold_cfg is not None else None

    # Augmentation parameters
    aug_cfg = aug_sweep_cfg.get("augmentation_params", {})
    time_warp_sigma = float(aug_cfg.get("time_warp_sigma", 0.03))
    noise_sigma = float(aug_cfg.get("noise_sigma", 0.02))
    magnitude_jitter = float(aug_cfg.get("magnitude_jitter", 0.02))
    include_mirror = bool(aug_cfg.get("include_mirror", False))

    # Model parameters
    in_channels = X.shape[1]
    dropout_default = float(cfg.get("model", {}).get("dropout", 0.2))

    model_kwargs_extra = {}
    lr = base_lr
    wd = base_wd
    bs_model = bs_eff
    lbl_smooth = label_smoothing

    # Storage for this combination
    fold_metrics = []
    pre_cms = []
    post_cms = []
    wd_sweep_rows = []

    # Select random fold for loss plot
    rng = np.random.default_rng(seed)
    plot_fold = rng.integers(1, n_splits + 1)

    # K-fold CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros_like(y), y), start=1):
        fold_start_time = time.time()

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Apply augmentation to training data only (if aug_factor > 0 or mirroring enabled)
        if aug_factor > 0 or include_mirror:
            X_train, y_train = apply_augmentation(
                X_train, y_train,
                aug_factor=aug_factor if aug_factor > 0 else 1,
                time_warp_sigma=time_warp_sigma if aug_factor > 0 else 0.0,
                noise_sigma=noise_sigma if aug_factor > 0 else 0.0,
                magnitude_jitter=magnitude_jitter if aug_factor > 0 else 0.0,
                include_mirror=include_mirror,
                random_seed=seed + fold,
            )

        # Balance validation fold if requested
        if balance_val:
            X_val, y_val = _balance_dataset(X_val, y_val, random_seed=seed)

        # Create dataloaders
        train_ds = to_tensor_dataset(X_train, y_train)
        val_ds = to_tensor_dataset(X_val, y_val)

        extra = {}
        if dl_kwargs.get("num_workers", 0) > 0 and "persistent_workers" not in dl_kwargs:
            extra["persistent_workers"] = True

        if balance_train:
            counts = np.bincount(y_train)
            counts[counts == 0] = 1
            class_weights = 1.0 / counts
            sample_weights = class_weights[y_train]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=bs_model, sampler=sampler, **dl_kwargs, **extra)
        else:
            train_loader = DataLoader(train_ds, batch_size=bs_model, shuffle=True, **dl_kwargs, **extra)

        val_loader = DataLoader(val_ds, batch_size=bs_model, shuffle=False, **dl_kwargs, **extra)

        # Build model
        mkwargs = dict(model_kwargs_extra)
        mkwargs.setdefault("dropout", dropout_default)

        build_kwargs = {"in_channels": in_channels, "num_classes": len(class_names)}
        if model_name.lower() == "tcn":
            build_kwargs.pop("in_channels", None)
            build_kwargs["num_inputs"] = in_channels
        build_kwargs.update(mkwargs)

        model = zoo_build(model_name, **build_kwargs).to(device)
        model = maybe_compile(model, accel, enabled=cfg.get("system", {}).get("compile", False))

        # Optimizer and criterion
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        criterion = nn.CrossEntropyLoss(label_smoothing=lbl_smooth if lbl_smooth > 0 else 0.0)

        # Training loop
        best_val_auc = -np.inf
        best_val_loss = np.inf
        best_checkpoint_loss = np.inf  # Loss at last checkpoint (for alternative condition)
        best_epoch = -1
        patience = 0
        best_state = None
        train_losses, val_losses = [], []
        epochs_trained = 0
        training_start = time.time()

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_total = 0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx():
                    logits = model(xb)
                    loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss_sum += loss.item() * xb.size(0)
                train_total += xb.size(0)

            train_loss = train_loss_sum / max(1, train_total)

            # Validation
            y_val_true, val_probs, val_loss = eval_on_loader(model, val_loader, device, autocast_ctx)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if len(y_val_true) == 0:
                break

            val_auc = multiclass_auc(y_val_true, val_probs, len(class_names))

            # Track best loss independently
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Combined early stopping: AUC improvement + loss constraint
            auc_improved = np.isfinite(val_auc) and (val_auc >= best_val_auc + auc_min_delta)

            if loss_mode == "relative":
                loss_ok = val_loss <= best_val_loss * loss_tolerance
            elif loss_mode == "absolute":
                loss_ok = val_loss <= best_val_loss + loss_tolerance
            else:  # "none"
                loss_ok = True

            # Primary checkpoint condition
            primary_checkpoint = auc_improved and loss_ok

            # Alternative checkpoint condition: AUC within tolerance AND loss improved significantly
            alt_checkpoint = False
            if alt_checkpoint_enabled and np.isfinite(val_auc) and best_val_auc > -np.inf:
                auc_within_tolerance = val_auc >= best_val_auc - auc_tolerance
                loss_improved_significantly = val_loss <= best_checkpoint_loss - loss_min_delta
                alt_checkpoint = auc_within_tolerance and loss_improved_significantly

            if primary_checkpoint or alt_checkpoint:
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                best_checkpoint_loss = val_loss
                best_epoch = epoch
                patience = 0
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= patience_limit:
                    epochs_trained = epoch
                    break
            epochs_trained = epoch

        training_time = time.time() - training_start
        time_per_epoch = training_time / max(1, epochs_trained)

        # Save loss plot for selected fold
        if fold == plot_fold and train_losses:
            loss_plot_path = os.path.join(run_dir, f"loss_curve_{_slug(model_name)}_{aug_label}_fold{fold}.pdf")
            plot_losses(train_losses, val_losses, save_path=loss_plot_path)

        # Restore best weights and evaluate
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        y_val_true, val_probs, val_loss = eval_on_loader(model, val_loader, device, autocast_ctx)
        n_classes = len(class_names)
        y_pred = val_probs.argmax(axis=1)
        bal_acc = balanced_accuracy_score(y_val_true, y_pred) * 100.0
        val_auc = multiclass_auc(y_val_true, val_probs, n_classes)

        cm_pre = confusion_normalized(y_val_true, y_pred, labels=np.arange(n_classes))
        pre_cms.append(cm_pre)

        fold_metrics.append({
            "model": model_name,
            "aug_factor": aug_factor,
            "aug_label": aug_label,
            "fold": fold,
            "n_train": len(train_idx) * (aug_factor if aug_factor > 0 else 1),
            "n_val": len(y_val),
            "loss": float(val_loss),
            "bal_acc": float(bal_acc),
            "auc_macro_ovr": float(val_auc),
            "epochs_trained": int(epochs_trained),
            "time_per_epoch_sec": float(time_per_epoch),
        })

        # MC dropout
        if mc_enabled:
            Xv, yv = val_ds.tensors
            probs_mc = mc_dropout_predict(model, Xv, n_mc=n_mc, batch_size=bs_model,
                                         device=device, autocast_ctx=autocast_ctx)
            if probs_mc.ndim == 3:
                thr, accs, fracs, sel_thr, mask_sel, y_pred_mc = wd_sweep_and_select(
                    probs_mc, yv.numpy(), trace_loss=trace_loss, wasserstein_threshold=wasserstein_threshold)
                if mask_sel.sum() > 0:
                    cm_post = confusion_normalized(yv.numpy()[mask_sel], y_pred_mc[mask_sel], labels=np.arange(n_classes))
                else:
                    cm_post = np.zeros((n_classes, n_classes), dtype=np.float32)
                post_cms.append(cm_post)
                for t, a, f in zip(thr, accs, fracs):
                    wd_sweep_rows.append({
                        "model": model_name,
                        "aug_factor": aug_factor,
                        "aug_label": aug_label,
                        "fold": fold,
                        "wd_thresh": float(t),
                        "kept_frac": float(f),
                        "bal_acc": float(a),
                    })

        print(f"[GPU {gpu_id}] {model_name}+{aug_label} fold {fold}/{n_splits}: AUC={val_auc:.3f}, time/epoch={time_per_epoch:.2f}s")

    # Aggregate results
    result = {
        "model": model_name,
        "aug_factor": aug_factor,
        "aug_label": aug_label,
        "fold_metrics": fold_metrics,
        "pre_cms": pre_cms,
        "post_cms": post_cms,
        "wd_sweep_rows": wd_sweep_rows,
    }

    print(f"[GPU {gpu_id}] Completed {model_name} + {aug_label}")
    return result


def _gpu_worker_loop(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    n_splits: int,
    shuffle: bool,
    seed: int,
    cfg: dict,
    run_dir: str,
):
    """
    Worker function that runs on a single GPU.

    Continuously pulls tasks from task_queue until it receives a sentinel (None).
    Results are placed in result_queue.
    """
    import warnings
    warnings.filterwarnings('ignore')

    # Set GPU for this worker process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Re-import torch after setting CUDA_VISIBLE_DEVICES
    import torch
    import torch.nn as nn

    # Detect accelerator for this process (will see only our assigned GPU)
    from utils import detect_accelerator, setup_precision_and_flags, dataloader_kwargs_for

    accel = detect_accelerator()
    amp_dtype, autocast_ctx, scaler = setup_precision_and_flags(accel)
    dl_kwargs = dataloader_kwargs_for(accel)
    device = accel["device"]

    print(f"[GPU {gpu_id}] Worker started on device: {accel['name']}")

    while True:
        # Get next task from queue (blocks until available)
        task = task_queue.get()

        # Sentinel value signals worker to exit
        if task is None:
            print(f"[GPU {gpu_id}] Worker received shutdown signal, exiting")
            break

        model_name, aug_factor = task

        try:
            result = _run_single_task(
                gpu_id=gpu_id,
                model_name=model_name,
                aug_factor=aug_factor,
                X=X,
                y=y,
                class_names=class_names,
                n_splits=n_splits,
                shuffle=shuffle,
                seed=seed,
                cfg=cfg,
                run_dir=run_dir,
                accel=accel,
                amp_dtype=amp_dtype,
                autocast_ctx=autocast_ctx,
                scaler=scaler,
                dl_kwargs=dl_kwargs,
                device=device,
            )
            result_queue.put(result)
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {model_name} + aug{aug_factor}x: {e}")
            # Put an error result so the main process knows this task failed
            result_queue.put({
                "model": model_name,
                "aug_factor": aug_factor,
                "aug_label": "noaug" if aug_factor == 0 else f"aug{aug_factor}x",
                "fold_metrics": [],
                "pre_cms": [],
                "post_cms": [],
                "error": str(e),
            })


def _distribute_tasks_to_gpus(
    tasks: List[Tuple[str, int]],  # List of (model_name, aug_factor)
    n_gpus: int,
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    n_splits: int,
    shuffle: bool,
    seed: int,
    cfg: dict,
    run_dir: str,
) -> List[dict]:
    """
    Distribute (model, aug_factor) tasks across multiple GPUs using dynamic work stealing.

    Each GPU worker continuously pulls tasks from a shared queue until all tasks are done.
    This ensures GPUs don't sit idle while work remains.

    Returns list of results from all tasks.
    """
    from utils import detect_accelerator, setup_precision_and_flags, dataloader_kwargs_for

    if n_gpus <= 1:
        # Sequential execution on single GPU
        accel = detect_accelerator()
        amp_dtype, autocast_ctx, scaler = setup_precision_and_flags(accel)
        dl_kwargs = dataloader_kwargs_for(accel)
        device = accel["device"]

        results = []
        for model_name, aug_factor in tasks:
            result = _run_single_task(
                gpu_id=0,
                model_name=model_name,
                aug_factor=aug_factor,
                X=X, y=y,
                class_names=class_names,
                n_splits=n_splits,
                shuffle=shuffle,
                seed=seed,
                cfg=cfg,
                run_dir=run_dir,
                accel=accel,
                amp_dtype=amp_dtype,
                autocast_ctx=autocast_ctx,
                scaler=scaler,
                dl_kwargs=dl_kwargs,
                device=device,
            )
            results.append(result)
        return results

    # Multi-GPU parallel execution with dynamic work stealing
    mp.set_start_method('spawn', force=True)

    # Create queues for task distribution and result collection
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Populate task queue with all tasks
    for task in tasks:
        task_queue.put(task)

    # Add sentinel values (None) to signal workers to exit
    # One sentinel per worker
    for _ in range(n_gpus):
        task_queue.put(None)

    print(f"Distributing {len(tasks)} tasks across {n_gpus} GPUs with dynamic work stealing")

    # Start one worker process per GPU
    workers = []
    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=_gpu_worker_loop,
            kwargs={
                "gpu_id": gpu_id,
                "task_queue": task_queue,
                "result_queue": result_queue,
                "X": X,
                "y": y,
                "class_names": class_names,
                "n_splits": n_splits,
                "shuffle": shuffle,
                "seed": seed,
                "cfg": cfg,
                "run_dir": run_dir,
            }
        )
        p.start()
        workers.append(p)

    # Collect results as they complete
    results = []
    for _ in range(len(tasks)):
        result = result_queue.get()  # Blocks until a result is available
        results.append(result)
        # Log progress
        completed = len(results)
        remaining = len(tasks) - completed
        print(f"Progress: {completed}/{len(tasks)} tasks complete, {remaining} remaining")

    # Wait for all workers to finish (they should exit after getting sentinel)
    for p in workers:
        p.join()

    print(f"All {len(tasks)} tasks completed")
    return results


# ---------------------------- Visualization Functions ---------------------------- #

def _plot_augmentation_heatmap(
    results_df: pd.DataFrame,
    output_dir: str,
    metric: str = "auc_mean",
    title: str = "AUC vs augmentation factor",
):
    """
    Create a heatmap showing model AUC (mean±std) across augmentation factors.

    Args:
        results_df: DataFrame with columns: model, aug_factor, auc_mean, auc_std, etc.
        output_dir: Directory to save the plot
        metric: Which metric to plot ('auc_mean', 'bal_acc_mean', 'time_per_epoch_mean')
        title: Plot title
    """
    import matplotlib.pyplot as plt

    # Display names for models
    MODEL_DISPLAY_NAMES = {
        "orig_conv_gru": "CNN-GRU",
        "resnet1d": "ResNet",
        "tcn": "TCN",
    }

    # Pivot the data for heatmap (mean values for coloring)
    pivot_mean = results_df.pivot(index="model", columns="aug_factor", values=metric)
    # Get corresponding std column
    std_metric = metric.replace("_mean", "_std")
    pivot_std = results_df.pivot(index="model", columns="aug_factor", values=std_metric)

    # Sort columns (aug_factor) numerically
    pivot_mean = pivot_mean[sorted(pivot_mean.columns)]
    pivot_std = pivot_std[sorted(pivot_std.columns)]

    # Apply display names to row index
    pivot_mean.index = [MODEL_DISPLAY_NAMES.get(m, m) for m in pivot_mean.index]
    pivot_std.index = pivot_mean.index

    n_models = len(pivot_mean.index)
    n_augs = len(pivot_mean.columns)

    fig, ax = plt.subplots(figsize=(max(3.5, n_augs * 1.2), max(2.0, n_models * 0.7)))

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    vmin, vmax = pivot_mean.values.min(), pivot_mean.values.max()
    midpoint = (vmin + vmax) / 2

    im = ax.imshow(pivot_mean.values, cmap=_lipari_cmap.reversed(), aspect='auto')

    # Add colorbar flush with matrix height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    metric_label = {
        "auc_mean": "AUC",
        "bal_acc_mean": "Balanced accuracy (%)",
        "time_per_epoch_mean": "Time per epoch (s)",
    }.get(metric, metric)
    cb.set_label(metric_label, fontsize=FONTSIZE_LEGEND)
    cb.ax.tick_params(labelsize=FONTSIZE_TICK, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)

    # Set ticks and labels
    ax.set_xticks(range(n_augs))
    ax.set_xticklabels([f"{int(c)}x" if c > 0 else "None" for c in pivot_mean.columns], fontsize=FONTSIZE_TICK)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(pivot_mean.index, fontsize=FONTSIZE_TICK, rotation=90, va='center')

    # Add value annotations with mean±std; white text above data midpoint
    for i in range(n_models):
        for j in range(n_augs):
            mean_val = pivot_mean.values[i, j]
            std_val = pivot_std.values[i, j]
            if np.isfinite(mean_val):
                text_color = 'white' if mean_val > midpoint else 'black'
                ax.text(j, i, f"{mean_val:.3f}\n±{std_val:.3f}",
                        ha="center", va="center",
                        color=text_color, fontsize=FONTSIZE_TICK, linespacing=1.3)

    ax.set_xlabel("Augmentation factor", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Model", fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=5)
    ax.tick_params(axis='both', which='major', length=2, width=0.4, direction='out')
    ax.tick_params(axis='both', which='minor', length=1, width=0.3, direction='out')

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap.pdf"), dpi=450, bbox_inches='tight')
    plt.close()


def _aggregate_augmentation_results(all_results: List[dict]) -> pd.DataFrame:
    """
    Aggregate results from multiple (model, aug_factor) combinations.

    Args:
        all_results: List of result dicts from _run_single_combination()

    Returns:
        DataFrame with aggregated statistics per (model, aug_factor)
    """
    aggregated = []

    for result in all_results:
        model_name = result['model']
        aug_factor = result['aug_factor']
        aug_label = result['aug_label']
        fold_metrics = result['fold_metrics']

        if not fold_metrics:
            continue

        aucs = [f['auc_macro_ovr'] for f in fold_metrics]
        accs = [f['bal_acc'] for f in fold_metrics]
        times = [f.get('time_per_epoch_sec', 0) for f in fold_metrics]
        epochs = [f.get('epochs_trained', 0) for f in fold_metrics]

        aggregated.append({
            'model': model_name,
            'aug_factor': aug_factor,
            'aug_label': aug_label,
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'bal_acc_mean': np.mean(accs),
            'bal_acc_std': np.std(accs),
            'time_per_epoch_mean': np.mean(times),
            'time_per_epoch_std': np.std(times),
            'epochs_mean': np.mean(epochs),
            'epochs_std': np.std(epochs),
            'n_folds': len(fold_metrics),
        })

    return pd.DataFrame(aggregated)


# ---------------------------- main ---------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config_cv.yaml")
    parser.add_argument("--models", type=str, default="", help="Comma-separated model names to compare")
    parser.add_argument("--k", type=int, default=0, help="Override n_splits in config")
    parser.add_argument("--aug-factors", type=str, default="",
                        help="Comma-separated augmentation factors to sweep (e.g., '0,2,3,5')")
    parser.add_argument("--n-gpus", type=int, default=0, help="Number of GPUs for parallel execution")
    args = parser.parse_args()

    cfg = load_config(args.config)

    io_cfg     = cfg.get("io", {})
    data_cfg   = cfg.get("data", {})
    opt_cfg    = cfg.get("optimization", {})
    unc_cfg    = cfg.get("uncertainty", {})
    cv_cfg     = cfg.get("cv", {})
    compare    = cfg.get("compare", {})
    aug_sweep_cfg = cfg.get("augmentation_sweep", {})
    system_cfg = cfg.get("system", {})

    output_root = io_cfg.get("output_root", "../Results/CrossVal")
    run_name    = io_cfg.get("run_name", "cv_benchmark")

    dataset_dict = get_dataset(cfg)
    dataset_keys = list(dataset_dict.keys())
    dataset_tag  = "_".join(dataset_keys)

    # CV setup (needed for print statements)
    n_splits = int(args.k or cv_cfg.get("n_splits", 5))
    shuffle  = bool(cv_cfg.get("shuffle", True))
    seed     = int(cv_cfg.get("seed", 840410))

    # Models to compare
    if args.models:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_list = compare.get("models") or [cfg.get("model", {}).get("name", "tcn")]

    # Accelerator and precision
    accel = detect_accelerator()
    amp_dtype, autocast_ctx, scaler = setup_precision_and_flags(accel)
    dl_kwargs = dataloader_kwargs_for(accel)
    device = accel["device"]

    print(f"Using device: {accel['name']} ({accel['type']})")
    print(f"Cross-validation setup: {n_splits} folds, shuffle={shuffle}, seed={seed}")
    print(f"Models to compare: {model_list}")

    # Extract data parameters from correct locations (could be in data section or root)
    trim_end_val = int(data_cfg.get("trim_end") or cfg.get("trim_end", 0) or 0)
    max_traces_val = data_cfg.get("max_traces_per_class") or cfg.get("max_traces_per_class")
    print(f"Data config - trim_end: {trim_end_val}, max_traces_per_class: {max_traces_val}")

    # Build dataset fully in memory
    # Check if dataset contains file paths (old format) or protein keys (new format)
    first_key = next(iter(dataset_dict))
    first_value = dataset_dict[first_key]

    if isinstance(first_value, (list, str)) and (
        (isinstance(first_value, str) and first_value.endswith('.pkl')) or
        (isinstance(first_value, list) and len(first_value) > 0 and first_value[0].endswith('.pkl'))
    ):
        # Old format: file paths specified directly
        print("Using legacy dataset format with explicit file paths")
        X, y, class_map, in_channels = build_dataset(
            dataset_dict,
            trim_end=trim_end_val,
            max_traces_per_class=max_traces_val,
            random_seed=cfg.get("system", {}).get("seed", 840410),
        )
    else:
        # New format: protein keys with file discovery
        traces_path = data_cfg.get("traces_path", "../Data/traces")
        print(f"Using new dataset format with protein keys, discovering files in: {traces_path}")
        X, y, class_map, in_channels = build_dataset_from_keys(
            dataset_dict,
            traces_path,
            trim_end=trim_end_val,
            max_traces_per_class=max_traces_val,
            random_seed=cfg.get("system", {}).get("seed", 840410),
        )
    class_names = [class_map[i] for i in sorted(class_map)]

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} channels, {X.shape[2]} timesteps")
    print(f"Classes ({len(class_names)}): {class_names}")
    class_counts = {class_names[i]: int((y == i).sum()) for i in range(len(class_names))}
    print(f"Class distribution: {class_counts}")

    # Additional CV setup - check both data section and root level
    balance_train = bool(data_cfg.get("balance_train") or cfg.get("balance_train", True))
    balance_test = bool(data_cfg.get("balance_test") or cfg.get("balance_test", False))

    # Batch size
    default_bs = int(opt_cfg.get("batch_size", 64))
    bs_eff = batch_size_hint(default_bs, accel) or default_bs

    # Optimizer defaults
    base_lr = float(opt_cfg.get("lr", 4e-4))
    base_wd = float(opt_cfg.get("weight_decay", 4e-4))
    max_epochs = int(opt_cfg.get("max_epochs", 100))
    patience_limit = int(opt_cfg.get("patience_limit", 10))
    label_smoothing = float(opt_cfg.get("label_smoothing", 0.0) or 0.0)
    threshold_metric = str(opt_cfg.get("threshold_metric", "argmax"))

    # Early stopping configuration
    early_stop_cfg = opt_cfg.get("early_stopping", {})
    auc_min_delta = float(early_stop_cfg.get("auc_min_delta", 0.005))
    loss_mode = str(early_stop_cfg.get("loss_mode", "relative"))
    loss_tolerance = float(early_stop_cfg.get("loss_tolerance", 1.10))
    # Alternative checkpoint: save if AUC within tolerance AND loss improves significantly
    auc_tolerance = early_stop_cfg.get("auc_tolerance")
    if auc_tolerance is not None:
        auc_tolerance = float(auc_tolerance)
    loss_min_delta = early_stop_cfg.get("loss_min_delta")
    if loss_min_delta is not None:
        loss_min_delta = float(loss_min_delta)
    alt_checkpoint_enabled = (auc_tolerance is not None) and (loss_min_delta is not None)

    # Uncertainty
    mc_do = unc_cfg.get("mc_dropout", {})
    mc_enabled = bool(mc_do.get("enabled", True))
    n_mc = int(mc_do.get("n_mc", 100))
    trace_loss = float(mc_do.get("trace_loss", 50.0))
    wasserstein_threshold_cfg = mc_do.get("wasserstein_threshold", None)
    wasserstein_threshold = float(wasserstein_threshold_cfg) if wasserstein_threshold_cfg is not None else None

    # Output directories
    run_dir = make_run_dir(output_root, run_name, dataset_keys, tag=f"{n_splits}fold")

    # logging setup
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(run_dir, f"{ts}_{_slug(run_name)}_cv_{_slug('_'.join(dataset_keys))}.log")
    log_file = open(log_path, "w")
    sys.stdout = sys.stderr = Tee(sys.__stdout__, log_file)
    print(f"Logging to: {log_path}")

    # Save snapshot of config
    with open(os.path.join(run_dir, "config_snapshot.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Check if augmentation sweep is enabled
    aug_sweep_enabled = aug_sweep_cfg.get("enabled", False)

    # Parse augmentation factors from CLI or config
    if args.aug_factors:
        aug_factors = [int(f.strip()) for f in args.aug_factors.split(",") if f.strip()]
        aug_sweep_enabled = True
    else:
        aug_factors = aug_sweep_cfg.get("factors", [])

    # Parse number of GPUs from CLI or config
    n_gpus = args.n_gpus or int(system_cfg.get("n_gpus", 1))

    # ================== AUGMENTATION SWEEP MODE ==================
    if aug_sweep_enabled and aug_factors:
        print("\n" + "=" * 60)
        print("AUGMENTATION SWEEP MODE")
        print("=" * 60)
        print(f"Models: {model_list}")
        print(f"Augmentation factors: {aug_factors}")
        print(f"Number of GPUs: {n_gpus}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"K-folds: {n_splits}")
        print("=" * 60 + "\n")

        # Create task list: all (model, aug_factor) combinations
        tasks = [(model_name, aug_factor) for model_name in model_list for aug_factor in aug_factors]
        print(f"Total tasks to run: {len(tasks)} (models × aug_factors)")

        # Run all tasks distributed across GPUs
        all_results = _distribute_tasks_to_gpus(
            tasks=tasks,
            n_gpus=n_gpus,
            X=X,
            y=y,
            class_names=class_names,
            n_splits=n_splits,
            shuffle=shuffle,
            seed=seed,
            cfg=cfg,
            run_dir=run_dir,
        )

        # Aggregate results
        aggregated_df = _aggregate_augmentation_results(all_results)

        # Create output directory for augmentation sweep results
        aug_sweep_dir = os.path.join(run_dir, "augmentation_sweep")
        os.makedirs(aug_sweep_dir, exist_ok=True)

        # Save aggregated results
        aggregated_df.to_csv(os.path.join(aug_sweep_dir, "data.csv"), index=False)

        # Collect all fold-level metrics
        all_fold_metrics = []
        for result in all_results:
            all_fold_metrics.extend(result['fold_metrics'])
        folds_df = pd.DataFrame(all_fold_metrics)
        folds_df.to_csv(os.path.join(run_dir, "cv_folds_metrics.csv"), index=False)

        # Generate confusion matrices per (model, aug_factor) combination
        confmat_rows = []
        filtered_confmat_rows = []
        all_wd_sweep_rows = []
        for result in all_results:
            model_name = result['model']
            aug_label = result['aug_label']
            aug_factor_val = result['aug_factor']
            pre_cms = result.get('pre_cms', [])
            post_cms = result.get('post_cms', [])

            # Pre-MC confusion matrix
            if pre_cms:
                mean_cm, std_cm = aggregate_confusions(pre_cms)
                _plot_cm_with_std(mean_cm, std_cm, class_names,
                                  os.path.join(aug_sweep_dir, f"confmat_{_slug(model_name)}_{aug_label}.pdf"),
                                  f"Confusion matrix - {model_name} ({aug_label}, {n_splits}-fold CV)")
                df = long_confusion_df(mean_cm, std_cm, class_names, model_name, dataset_tag, chosen_wd=None)
                df['aug_factor'] = aug_factor_val
                df['aug_label'] = aug_label
                confmat_rows.append(df)

            # Post-MC confusion matrix (if MC dropout was enabled)
            if post_cms:
                mean_cm, std_cm = aggregate_confusions(post_cms)
                _plot_cm_with_std(mean_cm, std_cm, class_names,
                                  os.path.join(aug_sweep_dir, f"confmat_{_slug(model_name)}_{aug_label}_filtered.pdf"),
                                  f"Confusion matrix - {model_name} ({aug_label}, {n_splits}-fold CV, filtered)")
                df = long_confusion_df(mean_cm, std_cm, class_names, model_name, dataset_tag, chosen_wd=None)
                df['aug_factor'] = aug_factor_val
                df['aug_label'] = aug_label
                filtered_confmat_rows.append(df)

            all_wd_sweep_rows.extend(result.get('wd_sweep_rows', []))

        if confmat_rows:
            pd.concat(confmat_rows, ignore_index=True).to_csv(os.path.join(aug_sweep_dir, "cv_confmat.csv"), index=False)

        if filtered_confmat_rows:
            pd.concat(filtered_confmat_rows, ignore_index=True).to_csv(
                os.path.join(aug_sweep_dir, "cv_confmat_filtered.csv"), index=False)

        if all_wd_sweep_rows:
            mcd_dir = os.path.join(run_dir, "MCD_results")
            wd_df = pd.DataFrame(all_wd_sweep_rows)
            wd_df.to_csv(os.path.join(mcd_dir, "cv_wd_sweep.csv"), index=False)

            # WD sweep plot: one panel per (model, aug_label)
            # x = retained traces %, y = balanced accuracy %, color = WD threshold
            wd_df['retained_pct'] = wd_df['kept_frac'] * 100.0
            combos_wd = (
                wd_df[['model', 'aug_label', 'aug_factor']]
                .drop_duplicates()
                .sort_values(['model', 'aug_factor'])
                .reset_index(drop=True)
            )
            n_panels_wd = len(combos_wd)
            n_cols_wd = min(n_panels_wd, 4)
            n_rows_wd = (n_panels_wd + n_cols_wd - 1) // n_cols_wd
            fig, axes = plt.subplots(n_rows_wd, n_cols_wd,
                                     figsize=(3.2 * n_cols_wd, 2.6 * n_rows_wd),
                                     squeeze=False)
            for p_idx, combo_row in combos_wd.iterrows():
                mdl  = combo_row['model']
                albl = combo_row['aug_label']
                ax   = axes[p_idx // n_cols_wd][p_idx % n_cols_wd]
                sub  = wd_df[(wd_df['model'] == mdl) & (wd_df['aug_label'] == albl)]
                n_folds_wd = sub['fold'].nunique()
                grp = (
                    sub.groupby('wd_thresh', sort=True)
                    .agg(
                        ret_mean=('retained_pct', 'mean'),
                        acc_mean=('bal_acc',       'mean'),
                        acc_std= ('bal_acc',       'std'),
                    )
                    .reset_index()
                    .dropna(subset=['acc_mean'])
                )
                norm_wd = mcolors.Normalize(vmin=grp['wd_thresh'].min(),
                                            vmax=grp['wd_thresh'].max())
                sc = ax.scatter(
                    grp['ret_mean'], grp['acc_mean'],
                    c=grp['wd_thresh'], cmap=_lapaz_cmap, norm=norm_wd,
                    s=12, zorder=3, linewidths=0,
                )
                if n_folds_wd >= 4:
                    grp_s = grp.sort_values('ret_mean')
                    ax.fill_between(
                        grp_s['ret_mean'],
                        grp_s['acc_mean'] - grp_s['acc_std'].fillna(0),
                        grp_s['acc_mean'] + grp_s['acc_std'].fillna(0),
                        alpha=0.20, color='#808080', zorder=1,
                    )
                ax.set_xlim(105, -5)
                ax.set_xlabel('Retained traces (%)', fontsize=FONTSIZE_LABEL)
                ax.set_ylabel('Balanced accuracy (%)', fontsize=FONTSIZE_LABEL)
                ax.set_title(f'{mdl} — {albl}', fontsize=FONTSIZE_TITLE, pad=4)
                ax.tick_params(axis='both', which='major', length=2, width=0.4, direction='out')
                ax.tick_params(axis='both', which='minor', length=1, width=0.3, direction='out')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(0.55)
                ax.spines['bottom'].set_linewidth(0.55)
                cb = fig.colorbar(sc, ax=ax, pad=0.04, shrink=0.85)
                cb.set_label('WD threshold', fontsize=FONTSIZE_LEGEND)
                cb.ax.tick_params(labelsize=FONTSIZE_TICK, length=2, width=0.4)
                cb.outline.set_linewidth(0.4)
            for p_idx in range(n_panels_wd, n_rows_wd * n_cols_wd):
                axes[p_idx // n_cols_wd][p_idx % n_cols_wd].set_visible(False)
            fig.tight_layout(w_pad=3, h_pad=3)
            fig.savefig(os.path.join(mcd_dir, "wd_sweep.pdf"), dpi=450, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved MCD results to: {mcd_dir}/")

        # Generate visualizations
        if not aggregated_df.empty:
            print("\nGenerating augmentation sweep visualizations...")

            # Heatmap
            _plot_augmentation_heatmap(
                aggregated_df, aug_sweep_dir,
                metric="auc_mean",
                title="AUC vs augmentation factor"
            )
            print(f"Saved heatmap to: {aug_sweep_dir}/heatmap.pdf")

        # Print summary
        print("\n" + "=" * 60)
        print("AUGMENTATION SWEEP SUMMARY")
        print("=" * 60)

        for model_name in model_list:
            model_results = aggregated_df[aggregated_df['model'] == model_name]
            if model_results.empty:
                continue
            print(f"\n{model_name}:")
            for _, row in model_results.iterrows():
                aug_label = row['aug_label']
                print(f"  {aug_label}: AUC = {row['auc_mean']:.3f} ± {row['auc_std']:.3f}, "
                      f"time/epoch = {row['time_per_epoch_mean']:.2f}s ± {row['time_per_epoch_std']:.2f}s")

        # Find best combination
        if not aggregated_df.empty:
            best_row = aggregated_df.loc[aggregated_df['auc_mean'].idxmax()]
            print(f"\nBest combination: {best_row['model']} + {best_row['aug_label']} "
                  f"(AUC = {best_row['auc_mean']:.3f} ± {best_row['auc_std']:.3f})")

        print(f"\nResults saved to: {run_dir}")
        print(f"- Aggregated metrics: augmentation_sweep/data.csv")
        print(f"- Fold-level metrics: cv_folds_metrics.csv")
        print(f"- Confusion matrices (pre-MC): augmentation_sweep/confmat_<model>_<aug>.pdf")
        print(f"- Confusion matrix data (pre-MC): augmentation_sweep/cv_confmat.csv")
        print(f"- Confusion matrices (filtered): augmentation_sweep/confmat_<model>_<aug>_filtered.pdf")
        print(f"- Confusion matrix data (filtered): augmentation_sweep/cv_confmat_filtered.csv")
        print(f"- Heatmap: augmentation_sweep/heatmap.pdf")
        print(f"- WD sweep data: MCD_results/cv_wd_sweep.csv")
        print(f"- WD sweep plot: MCD_results/wd_sweep.pdf")
        print("=" * 60)

        log_file.close()
        return

    # ================== STANDARD MODE (no augmentation sweep) ==================
    print("\n" + "=" * 60)
    print("STANDARD CV MODE")
    print("=" * 60)
    print(f"Models: {model_list}")
    print(f"K-folds: {n_splits}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        from utils import detect_accelerator
        print(f"  Device: {detect_accelerator()['name']}")
    print("=" * 60 + "\n")

    # Storage
    folds_metrics = []
    pre_cm_per_model: dict[str, list[np.ndarray]] = defaultdict(list)
    post_cm_per_model: dict[str, list[np.ndarray]] = defaultdict(list)
    wd_sweep_rows = []

    for model_name in model_list:
        print(f"=== Model: {model_name} ===")

        model_kwargs_extra = {}
        lr = base_lr
        wd = base_wd
        bs_model = bs_eff
        lbl_smooth = label_smoothing

        # Model-level defaults
        dropout_default = float(cfg.get("model", {}).get("dropout", 0.2))

        # Select a random fold to save loss plot (seeded for reproducibility)
        rng = np.random.default_rng(seed)
        plot_fold = rng.integers(1, n_splits + 1)

        for fold, train_idx, val_idx in stratified_kfold_indices(y, n_splits, shuffle, seed):
            print(f"--- Fold {fold}/{n_splits} ---")
            print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

            # DataLoaders
            train_loader, val_loader, train_ds, val_ds = build_dataloaders_from_indices(
                X, y, train_idx, val_idx, bs_model, balance_train, balance_test, dl_kwargs,
                random_seed=seed
            )

            # Build model from zoo
            mkwargs = dict(model_kwargs_extra)
            mkwargs.setdefault("dropout", dropout_default)

            # Base parameters that all models need
            build_kwargs = {"in_channels": in_channels, "num_classes": len(class_names)}

            # Handle TCN special case (expects num_inputs instead of in_channels)
            if model_name.lower() == "tcn":
                build_kwargs.pop("in_channels", None)
                build_kwargs["num_inputs"] = in_channels

            # Apply user overrides (from config)
            build_kwargs.update(mkwargs)


            model = zoo_build(model_name, **build_kwargs).to(device)
            model = maybe_compile(model, accel, enabled=cfg.get("system", {}).get("compile", False))

            # Print model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model: {model_name} | Total params: {total_params:,} | Trainable: {trainable_params:,}")
            print(f"Hyperparameters: lr={lr:.2e}, wd={wd:.2e}, batch_size={bs_model}, label_smoothing={lbl_smooth}")
            if mkwargs:
                print(f"Model kwargs: {mkwargs}")

            # Optimizer and criterion
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

            criterion = nn.CrossEntropyLoss(label_smoothing=lbl_smooth if lbl_smooth > 0 else 0.0)
            print(f"Using Cross-Entropy Loss (label_smoothing={lbl_smooth})")

            # Train with early stopping on validation AUC
            best_val_auc = -np.inf
            best_val_loss = np.inf
            best_checkpoint_loss = np.inf  # Loss at last checkpoint (for alternative condition)
            best_epoch = -1
            patience = 0
            best_state = None

            print(f"Training for max {max_epochs} epochs with patience {patience_limit}...")

            training_start_time = time.time()
            epochs_trained = 0
            train_losses, val_losses = [], []
            for epoch in range(1, max_epochs + 1):
                # Training phase
                model.train()
                train_loss_sum = 0.0
                train_correct = 0
                train_total = 0

                for xb, yb in train_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_ctx():
                        logits = model(xb)
                        loss = criterion(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    # Track training metrics
                    train_loss_sum += loss.item() * xb.size(0)
                    train_total += xb.size(0)
                    train_correct += (logits.argmax(dim=1) == yb).sum().item()

                train_loss = train_loss_sum / max(1, train_total)
                train_acc = train_correct / max(1, train_total)

                # Validation phase
                y_val_true, val_probs, val_loss = eval_on_loader(model, val_loader, device, autocast_ctx)
                if len(y_val_true) == 0:
                    print("Warning: empty validation split.")
                    break

                # Track losses for plotting
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                n_classes = len(class_names)
                val_pred = val_probs.argmax(axis=1)
                val_acc = (val_pred == y_val_true).mean()
                val_auc = multiclass_auc(y_val_true, val_probs, n_classes)

                # Print progress like train.py
                print(f"Epoch {epoch:03d} — train loss: {train_loss:.4f}, acc: {train_acc:.3f} | val loss: {val_loss:.4f}, acc: {val_acc:.3f}, auc: {val_auc:.3f}")

                # Track best loss independently
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                # Combined early stopping: AUC improvement + loss constraint
                auc_improved = np.isfinite(val_auc) and (val_auc >= best_val_auc + auc_min_delta)

                if loss_mode == "relative":
                    loss_ok = val_loss <= best_val_loss * loss_tolerance
                elif loss_mode == "absolute":
                    loss_ok = val_loss <= best_val_loss + loss_tolerance
                else:  # "none"
                    loss_ok = True

                # Primary checkpoint condition
                primary_checkpoint = auc_improved and loss_ok

                # Alternative checkpoint condition: AUC within tolerance AND loss improved significantly
                alt_checkpoint = False
                if alt_checkpoint_enabled and np.isfinite(val_auc) and best_val_auc > -np.inf:
                    auc_within_tolerance = val_auc >= best_val_auc - auc_tolerance
                    loss_improved_significantly = val_loss <= best_checkpoint_loss - loss_min_delta
                    alt_checkpoint = auc_within_tolerance and loss_improved_significantly

                if primary_checkpoint or alt_checkpoint:
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                    best_checkpoint_loss = val_loss
                    best_epoch = epoch
                    patience = 0
                    # keep CPU copy
                    best_state = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= patience_limit:
                        print(f"Early stopping at epoch {epoch} (best AUC {best_val_auc:.3f} at epoch {best_epoch})")
                        epochs_trained = epoch
                        break
                epochs_trained = epoch

            # Calculate training time per epoch
            training_time = time.time() - training_start_time
            time_per_epoch = training_time / max(1, epochs_trained)

            # Save loss plot for the randomly selected fold
            if fold == plot_fold and train_losses:
                loss_plot_path = os.path.join(run_dir, f"loss_curve_{_slug(model_name)}_fold{fold}.pdf")
                plot_losses(train_losses, val_losses, save_path=loss_plot_path)
                print(f"Saved loss plot to: {os.path.basename(loss_plot_path)}")

            # Restore best weights
            if best_state is not None:
                model.load_state_dict(best_state)
                print(f"Restored best model from epoch {best_epoch}")
            model.eval()

            print("Evaluating fold performance...")
            # Deterministic evaluation (pre-MC)
            y_val_true, val_probs, val_loss = eval_on_loader(model, val_loader, device, autocast_ctx)
            n_classes = len(class_names)

            # Generate argmax predictions (baseline)
            y_pred_argmax = val_probs.argmax(axis=1)
            bal_acc_argmax = balanced_accuracy_score(y_val_true, y_pred_argmax) * 100.0

            # Threshold optimization (if not argmax)
            optimal_threshold = None
            if threshold_metric.lower() != "argmax":
                if n_classes == 2:
                    # Binary classification: optimize threshold
                    optimal_threshold, metric_score = optimize_binary_threshold(
                        y_val_true, val_probs[:, 1], metric=threshold_metric
                    )
                    print(f"Fold {fold}: Optimal threshold={optimal_threshold:.3f} ({threshold_metric.upper()}: {metric_score:.3f})")
                elif n_classes > 2:
                    # Multiclass: global confidence threshold
                    optimal_threshold = optimize_multiclass_thresholds(
                        y_val_true, val_probs, metric='macro_f1', mode='global'
                    )
                    print(f"Fold {fold}: Optimal global threshold={optimal_threshold:.3f}")

            # Apply threshold (or use argmax)
            if optimal_threshold is not None:
                y_pred = apply_threshold(val_probs, optimal_threshold, n_classes)
                bal_acc = balanced_accuracy_score(y_val_true, y_pred) * 100.0
            else:
                y_pred = y_pred_argmax
                bal_acc = bal_acc_argmax

            val_auc = multiclass_auc(y_val_true, val_probs, n_classes)
            cm_pre = confusion_normalized(y_val_true, y_pred, labels=np.arange(n_classes))
            pre_cm_per_model[model_name].append(cm_pre)

            print(f"Fold {fold} results: balanced_acc_argmax={bal_acc_argmax:.1f}%, " +
                  (f"balanced_acc_optimal={bal_acc:.1f}%, " if optimal_threshold else "") +
                  f"auc={val_auc:.3f}, loss={val_loss:.4f}")

            fold_metrics_dict = {
                "dataset": dataset_tag,
                "model": model_name,
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "loss": float(val_loss),
                "bal_acc_argmax": float(bal_acc_argmax),
                "bal_acc": float(bal_acc),
                "auc_macro_ovr": float(val_auc),
                "threshold_metric": threshold_metric,
                "seed": seed,
            }
            if optimal_threshold is not None:
                fold_metrics_dict["optimal_threshold"] = float(optimal_threshold)
            folds_metrics.append(fold_metrics_dict)

            # MC dropout and WD sweep
            if mc_enabled:
                print(f"Running Monte Carlo dropout with {n_mc} forward passes...")
                Xv, yv = val_ds.tensors  # (N, C, T) and (N,)
                probs_mc = mc_dropout_predict(
                    model,
                    Xv,
                    n_mc=n_mc,
                    batch_size=bs_model,
                    device=device,
                    autocast_ctx=autocast_ctx,
                )  # expected shape: (n_mc, N, C)
                if probs_mc.ndim != 3:
                    raise RuntimeError(f"mc_dropout_predict must return (n_mc, N, C), got {probs_mc.shape}")

                if wasserstein_threshold is not None:
                    print(f"Performing Wasserstein distance sweep (fixed threshold: {wasserstein_threshold})...")
                else:
                    print(f"Performing Wasserstein distance sweep (max trace_loss: {trace_loss}%)...")
                thr, accs, fracs, sel_thr, mask_sel, y_pred_mean = wd_sweep_and_select(
                    probs_mc, yv.numpy(), trace_loss=trace_loss, wasserstein_threshold=wasserstein_threshold
                )

                kept_samples = int(mask_sel.sum())
                removed_pct = (1.0 - kept_samples / len(yv)) * 100.0
                if kept_samples > 0 and len(np.unique(yv.numpy()[mask_sel])) >= 2:
                    post_bal_acc = balanced_accuracy_score(yv.numpy()[mask_sel], y_pred_mean[mask_sel]) * 100.0
                    print(f"MC Dropout results: kept {kept_samples}/{len(yv)} samples ({removed_pct:.1f}% removed), post-filter acc: {post_bal_acc:.1f}%")
                elif kept_samples > 0:
                    print(f"MC Dropout results: kept {kept_samples}/{len(yv)} samples ({removed_pct:.1f}% removed), post-filter acc: N/A (single class retained)")
                else:
                    print(f"MC Dropout results: no samples kept (all {len(yv)} samples removed)")

                # WD sweep rows
                for t, a, f in zip(thr, accs, fracs):
                    wd_sweep_rows.append({
                        "dataset": dataset_tag,
                        "model": model_name,
                        "fold": fold,
                        "wd_thresh": float(t),
                        "kept_frac": float(f),
                        "bal_acc": float(a),
                    })

                # Post-MC confusion at selected threshold
                if mask_sel.sum() > 0:
                    cm_post = confusion_normalized(yv.numpy()[mask_sel], y_pred_mean[mask_sel], labels=np.arange(len(class_names)))
                else:
                    cm_post = np.zeros((len(class_names), len(class_names)), dtype=np.float32)
                post_cm_per_model[model_name].append(cm_post)

            # Record timing metrics
            fold_metrics_dict["epochs_trained"] = int(epochs_trained)
            fold_metrics_dict["time_per_epoch_sec"] = float(time_per_epoch)
            print(f"Fold {fold} completed: {epochs_trained} epochs, {time_per_epoch:.2f}s/epoch")

    # Aggregate and write CSVs
    folds_df = pd.DataFrame(folds_metrics)
    folds_df.to_csv(os.path.join(run_dir, "cv_folds_metrics.csv"), index=False)

    # Pre-MC consolidated confusion
    pre_rows = []
    for model_name, cms in pre_cm_per_model.items():
        if not cms:
            continue
        mean_cm, std_cm = aggregate_confusions(cms)
        # Plot combined mean±std confusion matrix
        _plot_cm_with_std(mean_cm, std_cm, class_names,
                          os.path.join(run_dir, f"confmat_{_slug(model_name)}.pdf"),
                          f"Confusion matrix - {model_name} ({n_splits}-fold CV)")
        pre_rows.append(long_confusion_df(mean_cm, std_cm, class_names, model_name, dataset_tag, chosen_wd=None))
    if pre_rows:
        pd.concat(pre_rows, ignore_index=True).to_csv(os.path.join(run_dir, "cv_confmat_pre_mc.csv"), index=False)

    # Post-MC consolidated confusion at best threshold under 50% loss constraint
    post_rows = []
    for model_name, cms in post_cm_per_model.items():
        if not cms:
            continue
        mean_cm, std_cm = aggregate_confusions(cms)
        # Plot combined mean±std confusion matrix (post MC dropout filtering)
        _plot_cm_with_std(mean_cm, std_cm, class_names,
                          os.path.join(run_dir, f"confmat_{_slug(model_name)}_filtered.pdf"),
                          f"Confusion matrix - {model_name} ({n_splits}-fold CV, filtered)")
        post_rows.append(long_confusion_df(mean_cm, std_cm, class_names, model_name, dataset_tag, chosen_wd=np.nan))
    if post_rows:
        pd.concat(post_rows, ignore_index=True).to_csv(os.path.join(run_dir, "cv_confmat_post_mc_best.csv"), index=False)

    # WD sweep CSV
    if wd_sweep_rows:
        pd.DataFrame(wd_sweep_rows).to_csv(os.path.join(run_dir, "cv_wd_sweep.csv"), index=False)

    # Model comparison figure (also useful for single models to show fold variability)
    if folds_metrics:
        # Gather summary statistics per model
        comparison_data = []
        for model_name in model_list:
            model_results = [r for r in folds_metrics if r["model"] == model_name]
            if model_results:
                aucs = [r["auc_macro_ovr"] for r in model_results]
                accs = [r["bal_acc"] for r in model_results]
                times = [r.get("time_per_epoch_sec", 0) for r in model_results]
                comparison_data.append({
                    "model": model_name,
                    "auc_mean": np.mean(aucs),
                    "auc_std": np.std(aucs),
                    "bal_acc_mean": np.mean(accs),
                    "bal_acc_std": np.std(accs),
                    "time_per_epoch_mean": np.mean(times),
                    "time_per_epoch_std": np.std(times),
                })

        if comparison_data:
            # Create output folder for comparison figure
            comparison_dir = os.path.join(run_dir, "model_comparison")
            os.makedirs(comparison_dir, exist_ok=True)

            # Save comparison data as CSV
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(os.path.join(comparison_dir, "data.csv"), index=False)

            # Create comparison figure: AUC vs time per epoch
            fig, ax = plt.subplots(figsize=(8, 6))

            color_list = list(COLORS.values())
            markers = ['o', '^', 's', 'D', 'v', 'p', 'h', '*']

            for i, row in enumerate(comparison_data):
                color = color_list[i % len(color_list)]
                marker = markers[i % len(markers)]
                ax.errorbar(
                    row["time_per_epoch_mean"],
                    row["auc_mean"],
                    xerr=row["time_per_epoch_std"],
                    yerr=row["auc_std"],
                    fmt=marker,
                    markersize=10,
                    color=color,
                    capsize=4,
                    capthick=1.5,
                    elinewidth=1.5,
                    label=row["model"],
                )

            ax.set_xlabel("Time per epoch (s)")
            ax.set_ylabel("AUC (macro OvR)")
            ax.set_title("Model comparison: AUC vs training speed")
            ax.legend(loc='best', framealpha=0.95)
            apply_axis_standards(ax)

            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, "plot.pdf"), dpi=450, bbox_inches='tight')
            plt.close()

            print(f"Saved model comparison figure to: {comparison_dir}")

    # Print summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)

    if folds_metrics:
        # Summary per model
        for model_name in model_list:
            model_results = [r for r in folds_metrics if r["model"] == model_name]
            if model_results:
                aucs = [r["auc_macro_ovr"] for r in model_results]
                accs = [r["bal_acc"] for r in model_results]
                times_per_epoch = [r.get("time_per_epoch_sec", 0) for r in model_results]
                epochs = [r.get("epochs_trained", 0) for r in model_results]
                print(f"{model_name}:")
                print(f"  AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
                print(f"  Balanced Accuracy: {np.mean(accs):.1f}% ± {np.std(accs):.1f}%")
                print(f"  Time per epoch: {np.mean(times_per_epoch):.2f}s ± {np.std(times_per_epoch):.2f}s")
                print(f"  Epochs trained: {np.mean(epochs):.1f} ± {np.std(epochs):.1f}")

        print(f"\nResults saved to: {run_dir}")
        print(f"- Fold metrics: cv_folds_metrics.csv")
        print(f"- Loss curves: loss_curve_<model>_fold<N>.pdf (one random fold per model)")
        print(f"- Confusion matrices: confmat_<model>.pdf, confmat_<model>_filtered.pdf")
        print(f"- Model comparison: model_comparison/plot.pdf + data.csv")
        if wd_sweep_rows:
            print(f"- WD sweep results: cv_wd_sweep.csv")
    else:
        print("No results generated.")

    print("="*60)


if __name__ == "__main__":
    main()
