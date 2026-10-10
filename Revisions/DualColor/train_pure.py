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
train_label_filter: paper_in uses the paper's IN rule instead of the vesicle table
(TRAINING_NOTES.md point 7; paper_in.py): a trace of a single-protein slide is in the pool if
a vesicle-channel ground-truth cluster lies within 4 px (the dataset's latest paper_in run,
or paper_in_runs: {<dataset>: run_NNN}).
Classes: HT (ATTO390 slides), SNAP (ATTO520).
Test truth on the mixed slides: only traces classed IN_ATTO390 / IN_ATTO520 (single-dye
vesicle within the radius, no second one) carry a class.

Validation by FOV (balance_val: true subsamples its larger class, as the paper does). In every single-protein slide a fraction `val_fov_fraction` of the FOVs
(at least one) is held out as validation; traces of one FOV share focus, laser and sample
conditions, so a random trace split would leak them into both sets. Early stopping, the
checkpoint and the Wasserstein threshold use only this validation set.

FOV-grouped k-fold (kfold: {n_folds, fold, partition_seed}; replaces val_fov_fraction;
TRAINING_NOTES.md point 9): in every single-protein slide the FOVs with training-pool traces
are shuffled with partition_seed and dealt in turn into n_folds folds from a random starting
fold (kfold_folds). Fold `fold` is the test fold ('test_oof': all its pool traces, not
balanced, predicted like the mixed slides), fold (fold + 1) mod n_folds the validation fold,
the rest the training set. The fold of every pool trace is written to data_split.csv
(kfold_fold).

Augmentation (training set only):
  aug_factor       the paper's sweep (ML/utils.apply_augmentation): N - 1 extra copies with time
                   warping (clipped to +-5 %), Gaussian noise and magnitude jitter (clipped to
                   +-5 %), parameters time_warp_sigma, noise_sigma, magnitude_jitter
  mirror           the paper's: one extra copy with the GMM active window reversed in place
  time_invariance  per sample and epoch: a random circular shift (0 to T-1 frames) and a
                   time reversal with probability 0.5, so that absolute timing (where in the
                   movie the activity lies), which differed between slides for non-protein
                   reasons, cannot be learned

Model and optimization: the paper's (orig_conv_gru, AdamW, label smoothing, combined AUC +
loss early stopping, class-balanced sampling), trained from scratch. optimization.early_stopping.
warmup_epochs W (default 0, the paper's): the first W epochs only train and early stopping
starts at epoch W + 1 (TRAINING_NOTES.md point 10). Precision: fp32 by
default (bf16 autocast makes the GRU about 30x slower on GH200).

Test, applied once with everything fixed: for the datasets in `test_datasets`, every filtered
trace of the mixed slides gets MC dropout probabilities (n_mc passes), a Wasserstein
uncertainty and kept = wd > threshold (the threshold auto-selected on the validation set,
trace loss <= trace_loss %). Holdout: `holdout_datasets` are left out of training and
validation entirely and their single-protein slides are predicted as a cross-session test.

Model: model.name orig_conv_gru (the paper's final model), resnet1d or tcn (the paper's other
two), with the model zoo's defaults (ML/models.py) unless model.kwargs overrides them.

Controls (the paper's, SI "Further control experiments"):
  label_scramble: true   the classes of the training pool (training and validation traces of
                         the single-protein slides) are permuted before the FOV split, with
                         np.random.default_rng(scramble_seed) (default: seed), so that several
                         permutations can share one FOV split and training seed; class counts
                         are kept, everything else is the same. The mixed-slide truth is not
                         scrambled. Read across permutations: one scrambled model is one draw
                         of an arbitrary function of the traces, whose mixed-slide AUC can lie
                         far from 0.5 (the classes differ in trace properties).
  trace_source: background
                         the same model and recipe trained on background traces (background.py)
                         instead of protein traces, labeled with the slide's protein. The
                         protein sets are built first (same FOV split, same validation
                         balancing); then, per dataset, slide and set (train / val), as many
                         background traces as protein traces are drawn (seeded) from that
                         set's FOVs, so the background model has the protein model's data
                         budget, slides and FOVs. Predicted: its validation set (which chose
                         the checkpoint) and 'val_unused', the other background traces of the
                         validation FOVs; holdout datasets' single-protein background as
                         'holdout_pure'. Slides differ in their background (even slides of one
                         protein), so the readout is per dataset and two-sided, against the
                         same-protein slide pairs (evaluate.py slide_pairs.csv).
  background_inference: true
                         after the protein predictions, classify_background.py's control on
                         the trained model (background_inference/).

Outputs (Results/Revisions/DualColor/models/<timestamp>_<run_name>/): best_model.pth,
config_full.json (this config + resolved inputs), data_split.csv (every trace used, with its
set), predictions.csv (validation, mixed test, k-fold test fold, holdout), threshold.json,
loss_curve/, log.

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
from background import load_background  # noqa: E402
from dc import REPO_ROOT, load_config, run_dir  # noqa: E402
from train import build_model_from_config, set_seed  # noqa: E402
from utils import (_balance_dataset, apply_augmentation, dataloader_kwargs_for, detect_accelerator, evaluate_uncertainty_filtered,  # noqa: E402
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
    info = {"dataset": ds, "trace_assignment_run": str(ta.relative_to(REPO_ROOT)),
            "extraction_run": str(ext_run.relative_to(REPO_ROOT))}
    if mc.get("train_label_filter") == "paper_in":
        pr = run_dir(cfg, "paper_in", (mc.get("paper_in_runs") or {}).get(ds))
        pin = pd.read_csv(pr / "paper_in.csv")[["extraction_key", "uniqueID", "paper_in", "gt_dist_px"]]
        tr = tr.merge(pin, on=["extraction_key", "uniqueID"], how="left", validate="one_to_one")
        if tr.loc[tr["slide_type"] == "pure", "paper_in"].isna().any():
            raise ValueError(f"{ds}: single-protein-slide traces missing from {pr / 'paper_in.csv'}")
        tr["paper_in"] = tr["paper_in"].fillna(False).astype(bool)
        info["paper_in_run"] = str(pr.relative_to(REPO_ROOT))
    tr = tr.reset_index(drop=True)
    X = traces_of(cfg, tr[["extraction_key", "uniqueID"]], ext_run)
    return tr, X, info


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


def kfold_folds(meta: pd.DataFrame, n_folds: int, partition_seed: int) -> pd.Series:
    """Fold (0 .. n_folds - 1) per row: per (dataset, slide), the FOVs in a seeded random order are
    dealt in turn into the folds from a random starting fold, so every fold gets floor(n / n_folds)
    or ceil(n / n_folds) of the slide's FOVs."""
    rng = np.random.default_rng(partition_seed)
    out = pd.Series(-1, index=meta.index, dtype=int)
    for (ds, slide), g in meta.groupby(["dataset", "slide"]):
        fovs = rng.permutation(np.array(sorted(g["fov"].unique())))
        start = int(rng.integers(n_folds))
        fold_of = {f: (start + i) % n_folds for i, f in enumerate(fovs)}
        out[g.index] = g["fov"].map(fold_of).to_numpy()
    return out


def kfold_sets(folds: pd.Series, n_folds: int, fold: int) -> pd.Series:
    """'test_oof' (fold `fold`), 'val' (the next fold) or 'train' (the others) per row."""
    if not 0 <= fold < n_folds or n_folds < 3:
        raise ValueError(f"kfold: fold {fold} of {n_folds} (needs n_folds >= 3, 0 <= fold < n_folds)")
    return pd.Series(np.select([folds == fold, folds == (fold + 1) % n_folds], ["test_oof", "val"], "train"),
                     index=folds.index)


def background_sets(meta: pd.DataFrame, mc: dict, seed: int, datasets: list[str], holdout: set
                    ) -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
    """Background traces in place of the protein sets of `meta` (trace_source: background).

    Each FOV of the protein training pool keeps its side of the split (train, or val for
    val and val_unused). Per (dataset, slide, set) of the protein train and val sets, as many
    background traces as protein traces are drawn without replacement from that slide's
    FOVs on that side (background.py, background_min_gap_px). Background of the other validation
    FOVs is 'val_unused'; val_fov marks
    every background trace of a validation FOV. Single-protein background of holdout datasets
    is 'holdout_pure'. Background of FOVs without protein training-pool traces and of the
    mixed slides is not used.
    """
    loaded = [load_background(ds, (mc.get("trace_assignment_runs") or {}).get(ds),
                              int(mc.get("background_min_gap_px", 3))) for ds in datasets]
    bg = pd.concat([x[0] for x in loaded], ignore_index=True)
    X = np.concatenate([x[1] for x in loaded], axis=0)
    pool = meta[meta["set"].isin(["train", "val", "val_unused"])]
    sides = (pool.assign(side=pool["set"].replace({"val_unused": "val"}))
             .groupby(["dataset", "slide", "fov"])["side"].agg(lambda s: sorted(set(s))))
    if (sides.map(len) > 1).any():
        raise ValueError("a FOV of the protein training pool lies on both sides of the split")
    side = sides.map(lambda s: s[0])
    bg["fov_side"] = side.reindex(pd.MultiIndex.from_frame(bg[["dataset", "slide", "fov"]])).to_numpy()
    bg["set"] = ""
    rng = np.random.default_rng(seed)
    need = meta[meta["set"].isin(["train", "val"])].groupby(["dataset", "slide", "set"]).size()
    for (ds, slide, s), n in need.items():
        cand = bg.index[(bg["dataset"] == ds) & (bg["slide"] == slide) & (bg["fov_side"] == s)].to_numpy()
        if len(cand) < n:
            raise ValueError(f"{ds} {slide} {s}: {len(cand)} background traces for {n} protein traces")
        bg.loc[rng.choice(cand, size=n, replace=False), "set"] = s
    bg["val_fov"] = bg["fov_side"] == "val"
    bg.loc[bg["val_fov"] & (bg["set"] == ""), "set"] = "val_unused"
    bg.loc[(bg["slide_type"] == "pure") & bg["dataset"].isin(holdout), "set"] = "holdout_pure"
    bg["y_class"] = bg["true_class"].map({c: k for k, c in enumerate(CLASSES)})
    print(f"Background in place of protein traces: {len(bg)} filtered background traces, "
          f"{int(bg['val_fov'].sum())} in validation FOVs")
    return bg, X, [x[2] for x in loaded]


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
    ap.add_argument("--dry-run", action="store_true", help="build the sets and the augmented training set, then stop")
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

    label_filter = mc.get("train_label_filter", "own_dye")
    if label_filter == "paper_in":              # the paper's IN rule (vesicle-channel clusters within 4 px)
        pure = (meta["slide_type"] == "pure") & meta["paper_in"]
    else:
        in_ves = meta["d1"] <= float(mc["train_radius_px"]) if mc.get("train_radius_px") else meta["in_vesicle"]
        pure = (meta["slide_type"] == "pure") & in_ves
        if label_filter == "own_dye":
            pure &= meta["own_dye"]             # on a single-protein slide, a trace at an object of the slide's own dye
    train_pool = pure & meta["dataset"].isin(mc["train_datasets"]) & ~meta["dataset"].isin(holdout)
    if mc.get("label_scramble", False):
        # the paper's control: permute the classes of the whole training pool before the split
        meta["true_class_unscrambled"] = meta["true_class"]
        tp = meta.index[train_pool]
        rng_s = np.random.default_rng(int(mc.get("scramble_seed", seed)))
        meta.loc[tp, "true_class"] = rng_s.permutation(meta.loc[tp, "true_class"].to_numpy())
        same = (meta.loc[tp, "true_class"] == meta.loc[tp, "true_class_unscrambled"]).mean()
        print(f"Labels scrambled: {len(tp)} training-pool traces, {same:.3f} keep their class by chance")
    meta["set"] = ""
    kf = mc.get("kfold")
    if kf:
        if "val_fov_fraction" in mc:
            raise ValueError("kfold replaces val_fov_fraction; set one of them")
        if mc.get("trace_source", "protein") == "background":
            raise ValueError("kfold is not implemented for trace_source: background")
        n_folds, fold = int(kf["n_folds"]), int(kf["fold"])
        meta["kfold_fold"] = -1
        meta.loc[train_pool, "kfold_fold"] = kfold_folds(meta[train_pool], n_folds, int(kf["partition_seed"]))
        meta.loc[train_pool, "set"] = kfold_sets(meta.loc[train_pool, "kfold_fold"], n_folds, fold)
        print(f"k-fold: fold {fold} of {n_folds} (partition seed {kf['partition_seed']}) is the test fold, "
              f"fold {(fold + 1) % n_folds} the validation fold")
    else:
        meta.loc[train_pool, "set"] = split_fovs(meta[train_pool], float(mc["val_fov_fraction"]), seed)
    test = (meta["slide_type"] == "mixed") & meta["dataset"].isin(mc["test_datasets"]) & ~meta["dataset"].isin(holdout)
    meta.loc[test, "set"] = "test_mixed"
    meta.loc[pure & meta["dataset"].isin(holdout), "set"] = "holdout_pure"
    meta["y_class"] = meta["true_class"].map({c: k for k, c in enumerate(CLASSES)})
    if mc.get("balance_val", False):
        # as the paper (balance_val: true): subsample the larger class of the validation set;
        # the traces left out are neither trained on nor validated ('val_unused')
        vi = meta.index[meta["set"] == "val"].to_numpy()
        _, _, keep = _balance_dataset(vi, meta.loc[vi, "y_class"].to_numpy(int), random_seed=seed)
        meta.loc[np.setdiff1d(vi, vi[keep]), "set"] = "val_unused"
    background = mc.get("trace_source", "protein") == "background"
    if background:
        if mc.get("label_scramble", False):
            raise ValueError("label_scramble and trace_source: background are separate controls")
        meta.drop(columns=["y_class"]).to_csv(out / "data_split_protein.csv", index=False)
        meta, X_all, bg_inputs = background_sets(meta, mc, seed, datasets, holdout)
    print(pd.crosstab([meta["set"], meta["dataset"]], meta["true_class"].fillna("unlabeled")).to_string())
    meta.drop(columns=["y_class"]).to_csv(out / "data_split.csv", index=False)

    tr_idx = meta.index[meta["set"] == "train"].to_numpy()
    va_idx = meta.index[meta["set"] == "val"].to_numpy()
    X_tr, y_tr = X_all[tr_idx], meta.loc[tr_idx, "y_class"].to_numpy(int)
    X_va, y_va = X_all[va_idx], meta.loc[va_idx, "y_class"].to_numpy(int)

    # The paper's augmentation (ML/utils.apply_augmentation): aug_factor N = the originals plus N - 1
    # warped / noisy / jittered copies, plus one mirrored copy with include_mirror; 0 or 1 = no copies.
    aug = mc.get("augmentation", {})
    aug_factor = int(aug.get("aug_factor", 0))
    if aug.get("mirror", True) or aug_factor > 1:
        X_tr, y_tr = apply_augmentation(X_tr, y_tr, aug_factor=aug_factor,
                                        time_warp_sigma=float(aug.get("time_warp_sigma", 0.03)),
                                        noise_sigma=float(aug.get("noise_sigma", 0.02)),
                                        magnitude_jitter=float(aug.get("magnitude_jitter", 0.02)),
                                        include_mirror=bool(aug.get("mirror", True)), random_seed=seed)
    if args.dry_run:
        print(f"Dry run: train {len(y_tr)} after augmentation {dict(zip(CLASSES, np.bincount(y_tr, minlength=2)))}, "
              f"val {len(y_va)} {dict(zip(CLASSES, np.bincount(y_va, minlength=2)))}; finite {np.isfinite(X_tr).all()}, "
              f"range {X_tr.min():.3f}..{X_tr.max():.3f}")
        return
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
                                auc_tolerance=es.get("auc_tolerance"), loss_min_delta=es.get("loss_min_delta"),
                                warmup_epochs=int(es.get("warmup_epochs", 0)))
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
    extra = {s: meta.index[meta["set"] == s].to_numpy()
             for s in ("test_mixed", "test_oof", "holdout_pure") + (("val_unused",) if background else ())}
    for s, idx in extra.items():
        if len(idx):
            p, _ = predict(model, X_all[idx], n_mc, device, autocast_ctx, thr)
            preds.append(meta.loc[idx].reset_index(drop=True).assign(set=s).join(p))
    pd.concat(preds, ignore_index=True).drop(columns=["y_class"]).to_csv(out / "predictions.csv", index=False)

    with open(out / "config_full.json", "w") as f:
        json.dump({"config_file": args.config, "config": mc, "inputs": [loaded[d][2] for d in datasets],
                   "classes": CLASSES, "n_frames": int(T), "device": str(accel["name"]), "precision": precision,
                   "torch": torch.__version__, "wasserstein_threshold": thr, "epochs_run": len(tl),
                   "best_val_auc": float(max(va)) if va else None,
                   **({"background_inputs": bg_inputs} if background else {})}, f, indent=2)

    if mc.get("background_inference", False) and not background:
        from classify_background import classify_run
        classify_run(out)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
