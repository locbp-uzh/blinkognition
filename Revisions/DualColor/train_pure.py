#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Train an HT vs SNAP classifier on the single-protein slides of the dual-color datasets and
apply it, once, to the vesicle-labeled traces of the mixed slides. Nothing from the paper's
data is used.

Data. For every dataset in `train_datasets`: the filtered traces (Extraction filters
passed, minmax channel) of the single-protein slides that lie within the slide's match
radius of a vesicle (trace_assignment run, d1 <= radius_px). The protein is the slide's;
with train_label_filter: own_dye (default) the nearest vesicle must also carry the slide's
dye label, since on some single-protein slides (DFK789 slides 2 and 4) many detected objects
are not vesicles of that dye (515/488 = 0.2-0.4 instead of 3 for ATTO520; dim, no 405).
'any' keeps traces at any detected object. train_radius_px, if set, replaces the
assignment radius of the single-protein slides (e.g. 2 px, as on the mixed slides).
Classes: HT (ATTO390 slides), SNAP (ATTO520).
Test truth on the mixed slides: only traces classed IN_ATTO390 / IN_ATTO520 (single-dye
vesicle within the radius, no second one) carry a class.

Validation by FOV. In every single-protein slide a fraction `val_fov_fraction` of the FOVs
(at least one) is held out as validation; traces of one FOV share focus, laser and sample
conditions, so a random trace split would leak them into both sets. Early stopping, the
checkpoint and the Wasserstein threshold use only this validation set.

Augmentation (training set only):
  mirror           the paper's: one extra copy with the GMM active window reversed in place
  time_invariance  per sample and epoch: a random circular shift (0 to T-1 frames) and a
                   time reversal with probability 0.5, so that absolute timing (where in the
                   movie the activity lies), which differed between slides for non-protein
                   reasons, cannot be learned

Model and optimization: the paper's (orig_conv_gru, AdamW, label smoothing, combined AUC +
loss early stopping, class-balanced sampling), trained from scratch. Precision: fp32 by
default (bf16 autocast makes the GRU about 30x slower on GH200).

Test, applied once with everything fixed: for the datasets in `test_datasets`, every filtered
trace of the mixed slides gets MC dropout probabilities (n_mc passes), a Wasserstein
uncertainty and kept = wd > threshold (the threshold auto-selected on the validation set,
trace loss <= trace_loss %). Holdout: `holdout_datasets` are left out of training and
validation entirely and their single-protein slides are predicted as a cross-session test.

Outputs (Results/Revisions/DualColor/models/<timestamp>_<run_name>/): best_model.pth,
config_full.json (this config + resolved inputs), data_split.csv (every trace used, with its
set), predictions.csv (validation, mixed test, holdout), threshold.json, loss_curve/, log.

Usage (Daint, blink2-cuda; from the repo root):
    python Revisions/DualColor/train_pure.py -c Revisions/DualColor/ml/<config>.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "ML"))
from dc import REPO_ROOT, load_config, run_dir  # noqa: E402
from train import build_model_from_config, set_seed  # noqa: E402
from utils import (apply_augmentation, dataloader_kwargs_for, detect_accelerator, evaluate_uncertainty_filtered,  # noqa: E402
                   mc_dropout_predict, plot_losses, train_model)

CLASSES = ["HT", "SNAP"]
DYE_CLASS = {"ATTO390": "HT", "ATTO520": "SNAP"}


class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()


class TimeInvariant(Dataset):
    """Random circular shift and time reversal (p = 0.5) of every sample, drawn anew each time.

    The draws use torch's generator, which DataLoader seeds per worker (base seed + worker id),
    so workers do not repeat each other's draws; the base seed follows torch.manual_seed.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = torch.from_numpy(X).float(), torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.roll(self.X[i], int(torch.randint(0, self.X.shape[-1], (1,))), dims=-1)
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])
        return x, self.y[i]


def traces_of(cfg: dict, keys_ids: pd.DataFrame, ext_run: Path) -> np.ndarray:
    """(N, 1, T) minmax traces for the rows of keys_ids (extraction_key, uniqueID), in order."""
    parts = []
    for key, g in keys_ids.groupby("extraction_key", sort=False):
        df = pd.read_pickle(ext_run / "ProteinTraces" / "Filtered" / f"{key}_filtered_minmax.pkl")
        missing = set(g["uniqueID"]) - set(df.columns)
        if missing:
            raise ValueError(f"{cfg['dataset']} {key}: {len(missing)} uniqueIDs not in the filtered traces")
        parts.append((g.index, df[g["uniqueID"].tolist()].to_numpy(dtype=np.float32).T))
    T = parts[0][1].shape[1]
    out = np.empty((len(keys_ids), 1, T), dtype=np.float32)
    pos = {ix: k for k, ix in enumerate(keys_ids.index)}
    for idx, arr in parts:
        if arr.shape[1] != T:
            raise ValueError(f"{cfg['dataset']}: traces of different lengths ({arr.shape[1]} vs {T})")
        out[[pos[i] for i in idx], 0, :] = arr
    return out


def load_dataset(ds: str, mc: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Filtered traces of one dataset with slide type, class (pure) or vesicle label (mixed)."""
    cfg = load_config(ds)
    ta = run_dir(cfg, "trace_assignment", (mc.get("trace_assignment_runs") or {}).get(ds))
    with open(ta / "manifest.yaml") as f:
        ext_run = REPO_ROOT / yaml.safe_load(f)["extraction_run"]
    tr = pd.read_csv(ta / "traces.csv", dtype={"slide": str, "fov": str, "vesicle_label": str, "class": str})
    tr = tr[tr["filtered"].astype(str) == "True"].copy()
    dye = tr["slide"].map(lambda s: cfg["slides"][s]["dye"])
    tr["dataset"] = ds
    tr["slide_type"] = np.where(dye == "mix", "mixed", "pure")
    tr["in_vesicle"] = tr["d1"] <= tr["radius_px"]
    tr["own_dye"] = tr["vesicle_label"] == dye                    # pure slides: the nearest vesicle carries the slide's dye
    single_dye = tr["class"].isin(["IN_ATTO390", "IN_ATTO520"])   # mixed slides: the label is the truth only here
    tr["true_class"] = np.where(dye == "mix", np.where(single_dye, tr["vesicle_label"].map(DYE_CLASS), None),
                                dye.map(DYE_CLASS))
    tr = tr.reset_index(drop=True)
    X = traces_of(cfg, tr[["extraction_key", "uniqueID"]], ext_run)
    return tr, X, {"dataset": ds, "trace_assignment_run": str(ta.relative_to(REPO_ROOT)),
                   "extraction_run": str(ext_run.relative_to(REPO_ROOT))}


def split_fovs(meta: pd.DataFrame, frac: float, seed: int) -> pd.Series:
    """'train' or 'val' per row: per (dataset, slide), ceil(frac x FOVs) FOVs (at least one) go to val."""
    rng = np.random.default_rng(seed)
    out = pd.Series("train", index=meta.index)
    for (ds, slide), g in meta.groupby(["dataset", "slide"]):
        fovs = np.array(sorted(g["fov"].unique()))
        n_val = max(1, math.ceil(frac * len(fovs)))
        val = set(rng.choice(fovs, size=n_val, replace=False))
        out[g.index[g["fov"].isin(val)]] = "val"
    return out


def predict(model, X: np.ndarray, n_mc: int, device, autocast_ctx, thr: float | None) -> pd.DataFrame:
    from scipy.stats import wasserstein_distance
    probs = mc_dropout_predict(model, torch.from_numpy(X).float(), n_mc=n_mc, batch_size=256, device=device,
                               autocast_ctx=autocast_ctx)
    mean = probs.mean(axis=0)
    pred = mean.argmax(axis=1)
    wd = np.array([min(wasserstein_distance(probs[:, i, pred[i]], probs[:, i, c]) for c in range(probs.shape[2])
                       if c != pred[i]) for i in range(len(pred))])
    df = pd.DataFrame({f"p_{c}": mean[:, k] for k, c in enumerate(CLASSES)})
    df["p_sd_SNAP"] = probs[:, :, CLASSES.index("SNAP")].std(axis=0)
    df["prediction"] = [CLASSES[k] for k in pred]
    df["wasserstein"] = wd
    if thr is not None:
        df["kept"] = wd > thr
    return df, probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        mc = yaml.safe_load(f)
    seed = int(mc.get("seed", 840410))
    set_seed(seed)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = REPO_ROOT / mc.get("output_root", "Results/Revisions/DualColor/models") / f"{ts}_{mc['run_name']}"
    out.mkdir(parents=True)
    log = open(out / f"{mc['run_name']}.log", "w")
    sys.stdout = sys.stderr = Tee(sys.__stdout__, log)
    print(f"Output: {out}\nConfig: {args.config}")

    # Data
    holdout = set(mc.get("holdout_datasets") or [])
    datasets = list(dict.fromkeys(mc["train_datasets"] + mc["test_datasets"] + sorted(holdout)))
    loaded = {ds: load_dataset(ds, mc) for ds in datasets}
    meta = pd.concat([loaded[ds][0] for ds in datasets], ignore_index=True)
    X_all = np.concatenate([loaded[ds][1] for ds in datasets], axis=0)
    T = X_all.shape[-1]
    print(f"{len(meta)} filtered traces of {datasets}, {T} frames")

    in_ves = meta["d1"] <= float(mc["train_radius_px"]) if mc.get("train_radius_px") else meta["in_vesicle"]
    pure = (meta["slide_type"] == "pure") & in_ves
    if mc.get("train_label_filter", "own_dye") == "own_dye":
        pure &= meta["own_dye"]                 # on a single-protein slide, a trace at an object of the slide's own dye
    train_pool = pure & meta["dataset"].isin(mc["train_datasets"]) & ~meta["dataset"].isin(holdout)
    meta["set"] = ""
    meta.loc[train_pool, "set"] = split_fovs(meta[train_pool], float(mc["val_fov_fraction"]), seed)
    test = (meta["slide_type"] == "mixed") & meta["dataset"].isin(mc["test_datasets"]) & ~meta["dataset"].isin(holdout)
    meta.loc[test, "set"] = "test_mixed"
    meta.loc[pure & meta["dataset"].isin(holdout), "set"] = "holdout_pure"
    meta["y"] = meta["true_class"].map({c: k for k, c in enumerate(CLASSES)})
    print(pd.crosstab([meta["set"], meta["dataset"]], meta["true_class"].fillna("unlabeled")).to_string())
    meta.drop(columns=["y"]).to_csv(out / "data_split.csv", index=False)

    tr_idx = meta.index[meta["set"] == "train"].to_numpy()
    va_idx = meta.index[meta["set"] == "val"].to_numpy()
    X_tr, y_tr = X_all[tr_idx], meta.loc[tr_idx, "y"].to_numpy(int)
    X_va, y_va = X_all[va_idx], meta.loc[va_idx, "y"].to_numpy(int)

    aug = mc.get("augmentation", {})
    if aug.get("mirror", True):
        X_tr, y_tr = apply_augmentation(X_tr, y_tr, aug_factor=0, include_mirror=True, random_seed=seed)
    accel = detect_accelerator()
    device = accel["device"]
    precision = mc.get("precision", "fp32")
    if precision == "fp32":
        autocast_ctx = lambda: nullcontext()  # noqa: E731
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    else:
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        autocast_ctx = lambda: torch.autocast(device_type=accel["type"], dtype=dt)  # noqa: E731
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and accel["type"] == "cuda"))
    print(f"Device {accel['name']} ({accel['type']}), precision {precision}")

    opt = mc["optimization"]
    bs = int(opt.get("batch_size", 64))
    dl_kw = dataloader_kwargs_for(accel)
    train_ds = TimeInvariant(X_tr, y_tr) if aug.get("time_invariance") else \
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
    w = 1.0 / np.bincount(y_tr, minlength=len(CLASSES))[y_tr]
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(y_tr), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, **dl_kw)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long()),
                            batch_size=bs, shuffle=False, **dl_kw)
    print(f"Train {len(y_tr)} (after augmentation) {dict(zip(CLASSES, np.bincount(y_tr, minlength=2)))}; "
          f"val {len(y_va)} {dict(zip(CLASSES, np.bincount(y_va, minlength=2)))}")

    mcfg = mc.get("model", {"name": "orig_conv_gru"})
    model = build_model_from_config(model_name=mcfg["name"], in_channels=1, num_classes=len(CLASSES),
                                    dropout=mcfg.get("dropout"), model_kwargs=mcfg.get("kwargs", {}), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt["lr"]), weight_decay=float(opt["weight_decay"]))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(opt.get("label_smoothing", 0.0)))
    es = opt.get("early_stopping", {})
    model_path, ckpt_path = out / "best_model.pth", out / "checkpoint.pth"
    tl, vl, va, _ = train_model(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion,
                                optimizer=optimizer, device=device, num_classes=len(CLASSES),
                                max_epochs=int(opt["max_epochs"]), patience_limit=int(opt["patience_limit"]),
                                model_save_path=str(model_path), checkpoint_path=str(ckpt_path),
                                autocast_ctx=autocast_ctx, scaler=scaler, clip_grad_norm=float(opt.get("clip_grad_norm", 1.0)),
                                threshold_metric="argmax", auc_min_delta=float(es.get("auc_min_delta", 0.005)),
                                loss_mode=str(es.get("loss_mode", "relative")),
                                loss_tolerance=float(es.get("loss_tolerance", 1.10)),
                                auc_tolerance=es.get("auc_tolerance"), loss_min_delta=es.get("loss_min_delta"))
    plot_losses(tl, vl, save_path=str(out / "loss_curve"))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Threshold on validation, then the test sets with everything fixed
    mcd = mc.get("mc_dropout", {})
    n_mc, trace_loss = int(mcd.get("n_mc", 100)), float(mcd.get("trace_loss", 50.0))
    val_pred, val_probs = predict(model, X_va, n_mc, device, autocast_ctx, None)
    sel = evaluate_uncertainty_filtered(val_probs, y_true=y_va, threshold=None, class_names=CLASSES, show_plots=False,
                                        save_dir=str(out / "validation_MCD"), trace_loss=trace_loss)
    thr = float(sel["selected_threshold"])
    val_pred["kept"] = val_pred["wasserstein"] > thr
    with open(out / "threshold.json", "w") as f:
        json.dump({"wasserstein_threshold": thr, "selected_on": "validation (held-out FOVs of the single-protein slides)",
                   "trace_loss_max_pct": trace_loss, "val_initial_accuracy": sel["initial_accuracy"],
                   "val_filtered_accuracy": sel["filtered_accuracy"], "val_removed_pct": sel["removed_percent"]},
                  f, indent=2)
    preds = [meta.loc[va_idx].reset_index(drop=True).join(val_pred)]
    for s in ("test_mixed", "holdout_pure"):
        idx = meta.index[meta["set"] == s].to_numpy()
        if len(idx):
            p, _ = predict(model, X_all[idx], n_mc, device, autocast_ctx, thr)
            preds.append(meta.loc[idx].reset_index(drop=True).join(p))
    pd.concat(preds, ignore_index=True).drop(columns=["y"]).to_csv(out / "predictions.csv", index=False)

    with open(out / "config_full.json", "w") as f:
        json.dump({"config_file": args.config, "config": mc, "inputs": [loaded[d][2] for d in datasets],
                   "classes": CLASSES, "n_frames": int(T), "device": str(accel["name"]), "precision": precision,
                   "torch": torch.__version__, "wasserstein_threshold": thr, "epochs_run": len(tl),
                   "best_val_auc": float(max(va)) if va else None}, f, indent=2)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
