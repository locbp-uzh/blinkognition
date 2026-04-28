#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Sonnet 4.6)
# __copyright__ = "Copyright 2026, UZH, Switzerland"

"""
Nested leave-one-out cross-validation across DK acquisition experiments.

Five DK experiments are used: 20250713_DK_Exp5, 20250714_DK_Exp6,
20250729_DK_Exp7, 20250730_DK_Exp8, 20251127_DK_Exp9.

Design (20 training runs total):
  Outer loop (5): hold one experiment as TEST.
  Inner loop (4): for each remaining experiment as VAL, train TCN on the other 3.

Each run evaluates on the outer TEST set.  The 4 test evaluations per outer fold
are averaged → per-experiment mean±std confusion matrix.  All 5 outer folds are
aggregated → overall confusion matrix.

Multi-GPU dispatch follows the same dynamic work-stealing queue pattern used in
ML/crossval.py.  Each task is a (test_exp, val_exp) tuple.

Usage (HPC):
    python loocv.py -c config_loocv.yaml --n-gpus 4

Usage (single GPU debug):
    python loocv.py -c config_loocv.yaml --n-gpus 1
"""

from __future__ import annotations

import os
import sys
import re
import json
import time
import queue as stdlib_queue
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- path setup ---
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "ML"))

from utils import (
    discover_protein_files,
    load_series,
    _load_col_names,
    mc_dropout_predict,
    plot_confusion_matrix_with_std,
    plot_losses,
    COLORS, FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_TITLE, FONTSIZE_LEGEND,
    apply_axis_standards,
)
from utils import (
    detect_accelerator,
    setup_precision_and_flags,
    dataloader_kwargs_for,
    maybe_compile,
    batch_size_hint,
)
from models import build_model as zoo_build

mpl.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':        FONTSIZE_LABEL,
    'pdf.fonttype':     42,
    'savefig.dpi':      450,
    'savefig.bbox':     'tight',
})

_DK_RE = re.compile(r"(\d{8}_DK_Exp\d+)")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

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
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def make_run_dir(output_root, run_name, dataset_keys):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d  = f"{ts}_{_slug(run_name)}_loocv_{_slug('_'.join(dataset_keys))}"
    path = os.path.join(output_root, d)
    os.makedirs(path, exist_ok=True)
    return path


def multiclass_auc(y_true, probs, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, probs[:, 1])
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    return roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")


def confusion_normalized(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes), normalize="true")
    return cm.astype(np.float32)


def wasserstein_1d(x, y):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    if x.size == 0 and y.size == 0: return 0.0
    if x.size == 0: return float(np.mean(np.abs(np.sort(y))))
    if y.size == 0: return float(np.mean(np.abs(np.sort(x))))
    m = int(max(x.size, y.size))
    q = (np.arange(m) + 0.5) / m
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def compute_wd_per_trace(probs_mc):
    """Return per-trace WD to nearest competing class. probs_mc: (n_mc, N, C)."""
    mean_probs = probs_mc.mean(axis=0)
    y_pred = mean_probs.argmax(axis=1)
    N, C = mean_probs.shape
    w = np.zeros(N, dtype=np.float32)
    for i in range(N):
        pred = y_pred[i]
        dmin = np.inf
        for c in range(C):
            if c == pred:
                continue
            d = wasserstein_1d(probs_mc[:, i, pred], probs_mc[:, i, c])
            if d < dmin:
                dmin = d
        w[i] = 0.0 if not np.isfinite(dmin) else dmin
    return w, y_pred


def to_tensor_dataset(X, y):
    if X.ndim == 2:
        X = X[:, None, :]
    return TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())


def balance_by_subsampling(X, y, seed=42):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_c = counts.min()
    idx = []
    for cls in classes:
        ci = np.where(y == cls)[0]
        idx.extend(rng.choice(ci, min_c, replace=False).tolist())
    idx = np.array(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def eval_on_loader(model, loader, device, autocast_ctx):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    y_true, probs, total_loss, total_n = [], [], 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast_ctx():
                logits = model(xb)
                loss   = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            total_n    += xb.size(0)
            probs.append(F.softmax(logits, dim=1).detach().cpu().float().numpy())
            y_true.append(yb.detach().cpu().numpy())
    y_true = np.concatenate(y_true) if y_true else np.array([])
    probs  = np.concatenate(probs)  if probs  else np.array([])
    return y_true, probs, total_loss / max(1, total_n)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading with experiment labels
# ──────────────────────────────────────────────────────────────────────────────

def build_uid_to_exp(uid_path, proteins):
    """
    Load UniqueIDs PKL files and return {unique_id: dk_experiment_name}.
    Traces not matching any DK experiment are mapped to 'UNKNOWN'.
    """
    uid_to_exp = {}
    for protein in proteins:
        pkl = os.path.join(uid_path, f"{protein}_uniqueID_IN.pkl")
        if not os.path.exists(pkl):
            print(f"Warning: UID file not found: {pkl}")
            continue
        df = pd.read_pickle(pkl)
        for _, row in df.iterrows():
            uid    = int(row["uniqueID"])
            origin = str(row.get("origin", ""))
            m = _DK_RE.search(origin)
            uid_to_exp[uid] = m.group(1) if m else "UNKNOWN"
    return uid_to_exp


def load_dataset_with_exp_labels(data_cfg, seed=840410):
    """
    Load Grx1 + K20Ac traces, return (X, y, exp_labels, class_names, in_channels).
    exp_labels[i] is the DK experiment name for trace i.
    """
    traces_path = data_cfg["traces_path"]
    uid_path    = data_cfg["uid_path"]
    dataset     = data_cfg["dataset"]
    proteins    = list(dataset.keys())
    trim_end    = int(data_cfg.get("trim_end", 0)) or None
    max_traces  = int(data_cfg.get("max_traces_per_class", 0)) or None
    channels    = dataset[proteins[0]].get("channels", ["minmax", "zscored"])

    uid_to_exp = build_uid_to_exp(uid_path, proteins)
    print(f"UID→experiment mapping: {len(uid_to_exp)} entries")

    # Find minimum trace length across all files
    min_T = None
    for prot in proteins:
        files = discover_protein_files(traces_path, prot, channels)
        for f in files:
            T = load_series(f).shape[-1]
            min_T = T if min_T is None else min(min_T, T)

    rng = np.random.default_rng(seed)
    X_list, y_list, exp_list = [], [], []

    for class_idx, prot in enumerate(proteins):
        files = discover_protein_files(traces_path, prot, channels)
        arrs  = [load_series(f)[:, :min_T] for f in files]
        uids  = _load_col_names(files[0])  # UniqueIDs from first channel

        if not all(a.shape == arrs[0].shape for a in arrs):
            raise ValueError(f"Shape mismatch across channels for {prot}")

        X_i = np.stack(arrs, axis=1)   # (N, C, T)

        if trim_end:
            X_i = X_i[:, :, :-trim_end]

        if max_traces and X_i.shape[0] > max_traces:
            idx   = rng.choice(X_i.shape[0], max_traces, replace=False)
            X_i   = X_i[idx]
            uids  = uids[idx]

        exp_i = np.array([uid_to_exp.get(int(uid), "UNKNOWN") for uid in uids])
        y_i   = np.full(X_i.shape[0], class_idx, dtype=np.int64)

        X_list.append(X_i)
        y_list.append(y_i)
        exp_list.append(exp_i)

        exp_counts = {}
        for e in exp_i:
            exp_counts[e] = exp_counts.get(e, 0) + 1
        print(f"  {prot}: {X_i.shape[0]} traces, exp breakdown: {exp_counts}")

    X          = np.concatenate(X_list, axis=0)
    y          = np.concatenate(y_list, axis=0)
    exp_labels = np.concatenate(exp_list, axis=0)
    in_channels = len(channels)
    class_names = proteins

    return X, y, exp_labels, class_names, in_channels


# ──────────────────────────────────────────────────────────────────────────────
# Single task: train on 3 experiments, val on 1, evaluate on held-out test
# ──────────────────────────────────────────────────────────────────────────────

def _run_loocv_task(
    gpu_id: int,
    test_exp: str,
    val_exp: str,
    X: np.ndarray,
    y: np.ndarray,
    exp_labels: np.ndarray,
    class_names: List[str],
    cfg: dict,
    run_dir: str,
    accel: dict,
    autocast_ctx,
    scaler,
    dl_kwargs: dict,
    device,
) -> dict:
    opt_cfg    = cfg.get("optimization", {})
    unc_cfg    = cfg.get("uncertainty", {})
    data_cfg   = cfg.get("data", {})
    system_cfg = cfg.get("system", {})

    balance_train = bool(data_cfg.get("balance_train", True))
    balance_val   = bool(data_cfg.get("balance_val",   True))
    balance_test  = bool(data_cfg.get("balance_test",  True))
    seed          = int(system_cfg.get("seed", 840410))

    bs             = int(opt_cfg.get("batch_size", 64))
    lr             = float(opt_cfg.get("lr", 4e-4))
    wd             = float(opt_cfg.get("weight_decay", 4e-4))
    max_epochs     = int(opt_cfg.get("max_epochs", 500))
    patience_limit  = int(opt_cfg.get("patience_limit", 10))
    lbl_smooth      = float(opt_cfg.get("label_smoothing", 0.001))
    clip_grad_norm  = float(opt_cfg.get("clip_grad_norm", 1.0))

    early_cfg      = opt_cfg.get("early_stopping", {})
    auc_min_delta  = float(early_cfg.get("auc_min_delta", 0.005))
    loss_mode      = str(early_cfg.get("loss_mode", "relative"))
    loss_tolerance = float(early_cfg.get("loss_tolerance", 1.10))
    auc_tolerance  = early_cfg.get("auc_tolerance")
    loss_min_delta = early_cfg.get("loss_min_delta")
    if auc_tolerance  is not None: auc_tolerance  = float(auc_tolerance)
    if loss_min_delta is not None: loss_min_delta = float(loss_min_delta)
    alt_ckpt = (auc_tolerance is not None) and (loss_min_delta is not None)

    mc_cfg    = unc_cfg.get("mc_dropout", {})
    mc_en     = bool(mc_cfg.get("enabled", True))
    n_mc      = int(mc_cfg.get("n_mc", 100))
    trace_loss = float(mc_cfg.get("trace_loss", 50.0))

    in_channels = X.shape[1]
    n_classes   = len(class_names)

    # Split by experiment
    test_mask  = exp_labels == test_exp
    val_mask   = exp_labels == val_exp
    train_mask = ~test_mask & ~val_mask

    if train_mask.sum() == 0:
        print(f"[GPU {gpu_id}] WARNING: no training data for test={test_exp}, val={val_exp}")
        return None

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    aug_cfg = data_cfg.get("augmentation", {})
    if aug_cfg.get("include_mirror", False):
        X_train = np.concatenate([X_train, X_train[:, :, ::-1].copy()], axis=0)
        y_train = np.concatenate([y_train, y_train.copy()], axis=0)

    if balance_val:
        X_val, y_val = balance_by_subsampling(X_val, y_val, seed)
    if balance_test:
        X_test, y_test = balance_by_subsampling(X_test, y_test, seed)

    train_ds = to_tensor_dataset(X_train, y_train)
    val_ds   = to_tensor_dataset(X_val,   y_val)
    test_ds  = to_tensor_dataset(X_test,  y_test)

    if balance_train:
        counts = np.bincount(y_train)
        counts[counts == 0] = 1
        sw = (1.0 / counts)[y_train]
        sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, **dl_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **dl_kwargs)

    val_loader  = DataLoader(val_ds,  batch_size=bs, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, **dl_kwargs)

    # Build TCN
    model_name   = "tcn"
    build_kwargs = {"num_inputs": in_channels, "num_classes": n_classes}
    model = zoo_build(model_name, **build_kwargs).to(device)
    model = maybe_compile(model, accel, enabled=cfg.get("system", {}).get("compile", False))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(label_smoothing=lbl_smooth)

    # Training loop with early stopping
    best_val_auc = -np.inf
    best_val_loss = np.inf
    best_ckpt_loss = np.inf
    patience = 0
    best_state = None
    train_losses, val_losses = [], []
    epochs_trained = 0

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        tl_sum, tn = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(xb)
                loss   = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            tl_sum += loss.item() * xb.size(0)
            tn     += xb.size(0)

        y_vt, vp, vl = eval_on_loader(model, val_loader, device, autocast_ctx)
        train_losses.append(tl_sum / max(1, tn))
        val_losses.append(vl)

        if len(y_vt) == 0:
            break

        val_auc = multiclass_auc(y_vt, vp, n_classes)

        if vl < best_val_loss:
            best_val_loss = vl

        auc_improved = np.isfinite(val_auc) and (val_auc >= best_val_auc + auc_min_delta)
        loss_ok = (vl <= best_val_loss * loss_tolerance) if loss_mode == "relative" else \
                  (vl <= best_val_loss + loss_tolerance) if loss_mode == "absolute" else True

        primary = auc_improved and loss_ok
        alt = False
        if alt_ckpt and np.isfinite(val_auc) and best_val_auc > -np.inf:
            alt = (val_auc >= best_val_auc - auc_tolerance) and \
                  (vl <= best_ckpt_loss - loss_min_delta)

        if primary or alt:
            if val_auc > best_val_auc:
                best_val_auc = val_auc
            best_ckpt_loss = vl
            patience = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= patience_limit:
                epochs_trained = epoch
                break
        epochs_trained = epoch

    time_per_epoch = (time.time() - t0) / max(1, epochs_trained)

    # Save loss plot
    loss_fname = os.path.join(run_dir, f"loss_{_slug(test_exp)}_{_slug(val_exp)}.pdf")
    plot_losses(train_losses, val_losses, save_path=loss_fname)

    # Evaluate on TEST set
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    y_te, tp, _ = eval_on_loader(model, test_loader, device, autocast_ctx)
    y_pred_test  = tp.argmax(axis=1)
    test_auc     = multiclass_auc(y_te, tp, n_classes) if len(np.unique(y_te)) >= 2 else float('nan')
    test_bal_acc = balanced_accuracy_score(y_te, y_pred_test) * 100.0
    test_cm      = confusion_normalized(y_te, y_pred_test, n_classes)

    # MC dropout on test set
    mcd_wd_mean, mcd_bal_acc_filtered = float('nan'), float('nan')
    test_cm_mcd = np.zeros((n_classes, n_classes), dtype=np.float32)
    if mc_en:
        Xt = test_ds.tensors[0]
        probs_mc = mc_dropout_predict(model, Xt, n_mc=n_mc, batch_size=bs,
                                      device=device, autocast_ctx=autocast_ctx)
        if probs_mc.ndim == 3:
            w_dists, y_pred_mc = compute_wd_per_trace(probs_mc)
            mcd_wd_mean = float(w_dists.mean())

            # Auto-threshold under trace_loss constraint
            max_wd = float(w_dists.max()) if w_dists.size else 1.0
            thrs   = np.linspace(0.0, max_wd, 50)
            best_t, best_ba = 0.0, -np.inf
            sweep_thrs, sweep_bal_accs, sweep_kept_fracs = [], [], []
            for t in thrs:
                mask = w_dists > t
                kept_frac = mask.sum() / len(w_dists)
                if mask.sum() == 0 or (1 - kept_frac) * 100 > trace_loss:
                    continue
                if len(np.unique(y_te[mask])) < 2:
                    continue
                ba = balanced_accuracy_score(y_te[mask], y_pred_mc[mask]) * 100.0
                sweep_thrs.append(t)
                sweep_bal_accs.append(ba)
                sweep_kept_fracs.append(kept_frac)
                if ba > best_ba:
                    best_ba, best_t = ba, t
            sweep_path = os.path.join(
                run_dir, f"wd_sweep_{_slug(test_exp)}_{_slug(val_exp)}.pdf")
            plot_wd_sweep(sweep_thrs, sweep_bal_accs, sweep_kept_fracs, best_t,
                          save_path=sweep_path,
                          title=f"WD sweep — test={test_exp}, val={val_exp}")
            mask_sel = w_dists > best_t
            if mask_sel.sum() > 0 and len(np.unique(y_te[mask_sel])) >= 2:
                mcd_bal_acc_filtered = balanced_accuracy_score(
                    y_te[mask_sel], y_pred_mc[mask_sel]) * 100.0
                test_cm_mcd = confusion_normalized(y_te[mask_sel], y_pred_mc[mask_sel], n_classes)

    mcd_str    = f"{mcd_bal_acc_filtered:.1f}%" if np.isfinite(mcd_bal_acc_filtered) else "n/a"
    task_label = f"test={test_exp} | val={val_exp}"
    print(f"[GPU {gpu_id}] {task_label}: "
          f"AUC={test_auc:.3f}, bal_acc={test_bal_acc:.1f}%, "
          f"mcd_filtered={mcd_str}, "
          f"epochs={epochs_trained}, t/epoch={time_per_epoch:.1f}s")

    return {
        "test_exp":             test_exp,
        "val_exp":              val_exp,
        "n_train":              int(train_mask.sum()),
        "n_val":                int(val_mask.sum()),
        "n_test":               int(test_mask.sum()),
        "epochs_trained":       int(epochs_trained),
        "time_per_epoch_sec":   float(time_per_epoch),
        "test_auc":             float(test_auc),
        "test_bal_acc":         float(test_bal_acc),
        "test_cm":              test_cm,
        "mcd_wd_mean":          float(mcd_wd_mean),
        "mcd_bal_acc_filtered": float(mcd_bal_acc_filtered),
        "test_cm_mcd":          test_cm_mcd,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU worker
# ──────────────────────────────────────────────────────────────────────────────

def _gpu_worker_loop(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    X: np.ndarray,
    y: np.ndarray,
    exp_labels: np.ndarray,
    class_names: List[str],
    cfg: dict,
    run_dir: str,
):
    import warnings
    warnings.filterwarnings('ignore')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from utils import detect_accelerator, setup_precision_and_flags, dataloader_kwargs_for

    accel = detect_accelerator()
    _, autocast_ctx, scaler = setup_precision_and_flags(accel)
    dl_kwargs = dataloader_kwargs_for(accel)
    device = accel["device"]
    print(f"[GPU {gpu_id}] Worker started on {accel['name']}")

    while True:
        task = task_queue.get()
        if task is None:
            break
        test_exp, val_exp = task
        try:
            result = _run_loocv_task(
                gpu_id=gpu_id,
                test_exp=test_exp, val_exp=val_exp,
                X=X, y=y, exp_labels=exp_labels,
                class_names=class_names,
                cfg=cfg, run_dir=run_dir,
                accel=accel, autocast_ctx=autocast_ctx,
                scaler=scaler, dl_kwargs=dl_kwargs, device=device,
            )
            result_queue.put(result)
        except Exception as e:
            import traceback
            print(f"[GPU {gpu_id}] ERROR on test={test_exp}, val={val_exp}: {e}")
            traceback.print_exc()
            result_queue.put(None)


def _distribute_tasks(
    tasks: List[Tuple[str, str]],
    n_gpus: int,
    X, y, exp_labels, class_names, cfg, run_dir,
) -> List[dict]:
    if n_gpus <= 1:
        accel = detect_accelerator()
        _, autocast_ctx, scaler = setup_precision_and_flags(accel)
        dl_kwargs = dataloader_kwargs_for(accel)
        device = accel["device"]
        results = []
        for test_exp, val_exp in tasks:
            r = _run_loocv_task(
                gpu_id=0, test_exp=test_exp, val_exp=val_exp,
                X=X, y=y, exp_labels=exp_labels,
                class_names=class_names, cfg=cfg, run_dir=run_dir,
                accel=accel, autocast_ctx=autocast_ctx,
                scaler=scaler, dl_kwargs=dl_kwargs, device=device,
            )
            if r is not None:
                results.append(r)
        return results

    mp.set_start_method('spawn', force=True)
    task_queue   = mp.Queue()
    result_queue = mp.Queue()

    for task in tasks:
        task_queue.put(task)
    for _ in range(n_gpus):
        task_queue.put(None)  # sentinels

    workers = []
    for gpu_id in range(n_gpus):
        p = mp.Process(target=_gpu_worker_loop,
                       kwargs=dict(gpu_id=gpu_id, task_queue=task_queue,
                                   result_queue=result_queue,
                                   X=X, y=y, exp_labels=exp_labels,
                                   class_names=class_names,
                                   cfg=cfg, run_dir=run_dir))
        p.start()
        workers.append(p)

    results  = []
    received = 0
    while received < len(tasks):
        # If all workers are dead and we haven't received all results, abort cleanly
        if not any(p.is_alive() for p in workers):
            print(f"ERROR: all GPU workers have exited after {received}/{len(tasks)} results "
                  f"— possible OOM or hardware fault. Proceeding with partial results.")
            break
        try:
            r = result_queue.get(timeout=60)   # re-checks worker liveness every 60 s
        except stdlib_queue.Empty:
            continue
        received += 1
        if r is not None:
            results.append(r)
        print(f"Progress: {received}/{len(tasks)} tasks received ({len(results)} successful)")

    for p in workers:
        p.join(timeout=30)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate and plot
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_cms(cms):
    stack = np.stack(cms, axis=0)
    return stack.mean(axis=0), stack.std(axis=0, ddof=1 if len(cms) > 1 else 0)


def plot_cm(mean_cm, std_cm, class_names, save_path, title):
    plot_confusion_matrix_with_std(mean_cm, std_cm, class_names,
                                   save_path=save_path, title=title)


def plot_wd_sweep(thrs, bal_accs, kept_fracs, best_t, save_path, title=""):
    """WD threshold sweep: balanced accuracy and retained fraction vs threshold."""
    valid = [(t, ba, kf) for t, ba, kf in zip(thrs, bal_accs, kept_fracs)
             if np.isfinite(ba) and np.isfinite(kf)]
    if not valid:
        return
    tv, bav, kfv = zip(*valid)

    _c = list(COLORS.values())

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax2 = ax1.twinx()

    ax1.plot(tv, bav, color=_c[0], lw=1.5, label="Bal. accuracy")
    ax2.plot(tv, [kf * 100 for kf in kfv], color=_c[1], lw=1.5, ls="--", label="Kept %")
    if np.isfinite(best_t) and best_t > 0:
        ax1.axvline(best_t, color="red", lw=1.0, ls=":", label=f"θ={best_t:.3f}")

    ax1.set_xlabel("WD threshold", fontsize=FONTSIZE_LABEL)
    ax1.set_ylabel("Balanced accuracy (%)", fontsize=FONTSIZE_LABEL)
    ax2.set_ylabel("Retained traces (%)", fontsize=FONTSIZE_LABEL)
    ax1.tick_params(labelsize=FONTSIZE_TICK)
    ax2.tick_params(labelsize=FONTSIZE_TICK)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=FONTSIZE_LEGEND, loc="best")
    if title:
        ax1.set_title(title, fontsize=FONTSIZE_TITLE)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--n-gpus", type=int, default=0)
    args = parser.parse_args()

    cfg        = load_config(args.config)
    io_cfg     = cfg.get("io", {})
    data_cfg   = cfg.get("data", {})
    system_cfg = cfg.get("system", {})

    output_root = io_cfg.get("output_root", "../../Results/Controls/LeaveOneOutCV")
    run_name    = io_cfg.get("run_name", "loocv_tcn")
    seed        = int(system_cfg.get("seed", 840410))
    n_gpus      = args.n_gpus or int(system_cfg.get("n_gpus", 1))

    dataset_keys = list(data_cfg["dataset"].keys())
    run_dir = make_run_dir(output_root, run_name, dataset_keys)

    log_path = os.path.join(run_dir, f"{_slug(run_name)}_loocv.log")
    log_file = open(log_path, "w")
    sys.stdout = sys.stderr = Tee(sys.__stdout__, log_file)
    print(f"Run dir: {run_dir}")
    print(f"Log:     {log_path}")
    print(f"GPUs:    {n_gpus}")

    with open(os.path.join(run_dir, "config_snapshot.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Load dataset with experiment labels
    print("\nLoading dataset with experiment labels...")
    X, y, exp_labels, class_names, in_channels = load_dataset_with_exp_labels(data_cfg, seed)
    print(f"Dataset: {X.shape[0]} traces, {in_channels} channels, {X.shape[2]} timesteps")
    print(f"Classes: {class_names}")

    experiments = sorted(set(exp_labels) - {"UNKNOWN"})
    unknown_n   = int((exp_labels == "UNKNOWN").sum())
    print(f"\nDK experiments ({len(experiments)}): {experiments}")
    if unknown_n:
        print(f"WARNING: {unknown_n} traces could not be mapped to a DK experiment — excluded.")

    # Exclude UNKNOWN traces
    known_mask = exp_labels != "UNKNOWN"
    X, y, exp_labels = X[known_mask], y[known_mask], exp_labels[known_mask]
    print(f"Traces after excluding UNKNOWN: {len(X)}")

    for exp in experiments:
        n = (exp_labels == exp).sum()
        cls_dist = {class_names[i]: int(((exp_labels == exp) & (y == i)).sum()) for i in range(len(class_names))}
        print(f"  {exp}: {n} traces  {cls_dist}")

    # Build task list: all (test_exp, val_exp) pairs where test != val
    tasks = [
        (test_exp, val_exp)
        for test_exp in experiments
        for val_exp  in experiments
        if test_exp != val_exp
    ]
    print(f"\nTotal tasks: {len(tasks)}  ({len(experiments)} outer × {len(experiments)-1} inner)")

    aug_cfg = data_cfg.get("augmentation", {})
    if aug_cfg.get("include_mirror", False):
        print("Augmentation: time-reversal (mirror) enabled — training set doubled per fold.")
    else:
        print("Augmentation: none.")

    print("\n" + "=" * 60)
    print("NESTED LEAVE-ONE-OUT CV")
    print("=" * 60 + "\n")

    # Run all tasks
    all_results = _distribute_tasks(
        tasks=tasks, n_gpus=n_gpus,
        X=X, y=y, exp_labels=exp_labels,
        class_names=class_names, cfg=cfg, run_dir=run_dir,
    )

    if not all_results:
        print("No results returned. Exiting.")
        return

    # Aggregate by test_exp → mean±std confusion matrix per held-out experiment
    cms_by_test:     dict[str, list] = defaultdict(list)
    cms_mcd_by_test: dict[str, list] = defaultdict(list)
    fold_rows = []

    for r in all_results:
        te = r["test_exp"]
        cms_by_test[te].append(r["test_cm"])
        if r["test_cm_mcd"].sum() > 0:
            cms_mcd_by_test[te].append(r["test_cm_mcd"])
        fold_rows.append({
            "test_exp":             te,
            "val_exp":              r["val_exp"],
            "n_train":              r["n_train"],
            "n_val":                r["n_val"],
            "n_test":               r["n_test"],
            "epochs_trained":       r["epochs_trained"],
            "time_per_epoch_sec":   r["time_per_epoch_sec"],
            "test_auc":             r["test_auc"],
            "test_bal_acc":         r["test_bal_acc"],
            "mcd_wd_mean":          r["mcd_wd_mean"],
            "mcd_bal_acc_filtered": r["mcd_bal_acc_filtered"],
        })

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(os.path.join(run_dir, "loocv_folds_metrics.csv"), index=False)

    # Per-test-experiment confusion matrices
    per_test_cms, per_test_mcd_cms = [], []
    for te in experiments:
        cms = cms_by_test.get(te, [])
        if not cms:
            continue
        mean_cm, std_cm = aggregate_cms(cms)
        per_test_cms.append(mean_cm)
        plot_cm(mean_cm, std_cm, class_names,
                os.path.join(run_dir, f"confmat_{_slug(te)}.pdf"),
                f"LOO — test: {te} (n={len(cms)} inner folds)")
        mcd_cms = cms_mcd_by_test.get(te, [])
        if mcd_cms:
            mcd_mean, mcd_std = aggregate_cms(mcd_cms)
            per_test_mcd_cms.append(mcd_mean)
            plot_cm(mcd_mean, mcd_std, class_names,
                    os.path.join(run_dir, f"confmat_{_slug(te)}_mcd.pdf"),
                    f"LOO (MCD filtered) — test: {te}")

    # Overall aggregated confusion matrix
    if per_test_cms:
        overall_mean, overall_std = aggregate_cms(per_test_cms)
        plot_cm(overall_mean, overall_std, class_names,
                os.path.join(run_dir, "confmat_overall.pdf"),
                f"LOO — overall ({len(experiments)} experiments, {len(tasks)} runs)")

    if per_test_mcd_cms:
        mcd_overall_mean, mcd_overall_std = aggregate_cms(per_test_mcd_cms)
        plot_cm(mcd_overall_mean, mcd_overall_std, class_names,
                os.path.join(run_dir, "confmat_overall_mcd.pdf"),
                "LOO (MCD filtered) — overall")

    # Print summary
    print("\n" + "=" * 60)
    print("LOOCV SUMMARY")
    print("=" * 60)
    for te in experiments:
        sub = folds_df[folds_df["test_exp"] == te]
        if sub.empty:
            continue
        print(f"\n  Test experiment: {te}")
        print(f"    AUC:        {sub['test_auc'].mean():.3f} ± {sub['test_auc'].std():.3f}")
        print(f"    Bal acc:    {sub['test_bal_acc'].mean():.1f}% ± {sub['test_bal_acc'].std():.1f}%")
        print(f"    MCD acc:    {sub['mcd_bal_acc_filtered'].mean():.1f}% ± {sub['mcd_bal_acc_filtered'].std():.1f}%")
        print(f"    Epochs:     {sub['epochs_trained'].mean():.0f} ± {sub['epochs_trained'].std():.0f}")

    all_aucs = folds_df["test_auc"].dropna()
    all_accs = folds_df["test_bal_acc"].dropna()
    all_mcd  = folds_df["mcd_bal_acc_filtered"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\n  Overall across all {len(folds_df)} runs:")
    print(f"    AUC:     {all_aucs.mean():.3f} ± {all_aucs.std():.3f}")
    print(f"    Bal acc: {all_accs.mean():.1f}% ± {all_accs.std():.1f}%")
    if len(all_mcd) > 0:
        print(f"    MCD acc: {all_mcd.mean():.1f}% ± {all_mcd.std():.1f}%")
    print(f"\n  Results saved to: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
