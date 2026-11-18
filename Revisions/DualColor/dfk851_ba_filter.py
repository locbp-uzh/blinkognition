#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Balanced accuracy of DFK851 design B (in session; TRAINING_NOTES.md point 13) before and after the paper's
MC-dropout uncertainty filter, with the round-6 C models applied to DFK851 as the reference (user request
2026-11-18). Same form as round 4's (README "Round 4", round4_summary.py).

Prediction: the paper's, argmax of the mean over the 100 MC-dropout passes (threshold_metric argmax, so
p_SNAP >= 0.5 -> SNAP; the 'prediction' column). Balanced accuracy (BA) = mean of the HT and SNAP recalls.
Uncertainty: the paper's Wasserstein certainty per trace (ML/utils.evaluate_uncertainty_filtered: distance
between the MC distributions of the predicted and the other class); a trace is kept when it is above the
threshold. The threshold is the paper's rule: the value on a 50-point grid from 0 to the maximum that gives
the highest BA while removing at most 50 % of the traces. Two places to pick it:
  own_val   picked on the run's own validation set (held-out FOVs of the single-protein slides) and applied
            unchanged to the mixed slide (threshold.json, the 'kept' column; train_pure.py; round 4's form)
  on_test   picked on the scored mixed-slide traces themselves, as the paper's ML/train.py does on its test
            set; optimistic (the threshold is chosen on the traces it is scored on), shown for comparison
Units: the single-dye traces of DFK851's mixed slide (slide 3; true_class HT / SNAP); and each run's own
validation set (balanced, own_val only, where the threshold was picked: optimistic too).

Per run (ba_filter_runs.csv) and mean (SD) over the five seeds per cell (ba_filter_summary.csv): BA and the
two recalls on all traces, kept fraction, BA and recalls of the kept traces. Ensemble (ba_filter_ensemble.csv;
the user's rule, 2026-11-18): mean p_SNAP over the five runs of a cell, argmax; certainty = the mean of the five
runs' Wasserstein certainties (two classes: a run's certainty does not depend on its predicted class); kept =
mean certainty above a threshold, own_val = the mean of the five runs' own-validation thresholds (each the
paper's rule), on_test = the paper's rule applied to the mean certainty on the scored mixed-slide traces. BA on
all and on kept traces with 95 % FOV-cluster bootstrap intervals (2000 draws, FOVs with replacement; thresholds
fixed), B minus C on the same draws. (Until the user's rule: kept by at least 3 of the 5 runs at own_val.)
Reading (balanced accuracy is a result, the user's decision of 2026-11-18, after the results): a B - C BA
difference is read as real when its interval excludes 0 and it exceeds twice the larger seed SD of the matching
per-run metric (BA all; BA kept at the run's own-validation threshold for own_val; BA kept at the paper's rule on
the test traces for on_test); kept fractions are coverage, not read.

    python Revisions/DualColor/dfk851_ba_filter.py [--c-transfer <dir>] [--out <dir>] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dfk851_summary as s  # noqa: E402  (DS, COLS, DT, MODELS, run_log)
from dc import REPO_ROOT  # noqa: E402
from round6_summary import PLATEAU_LOSS, latest  # noqa: E402

TRACE_LOSS = 50.0          # the paper's mc_dropout.trace_loss (ML/config_train.yaml)
CLS = {"HT": 0, "SNAP": 1}


def recalls(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """(BA, recall HT, recall SNAP); a class absent from the kept traces counts 0, as the paper's sweep."""
    r = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    return float(r.mean()), float(r[0]), float(r[1])


def paper_threshold(w: np.ndarray, y: np.ndarray, pred: np.ndarray) -> float:
    """ML/utils.evaluate_uncertainty_filtered's auto-selection (threshold=None): grid linspace(0, max, 50),
    keep w > thr, the highest BA with at most TRACE_LOSS % removed; the first maximum on ties."""
    best = None
    for thr in np.linspace(0, w.max() if w.size else 1.0, 50):
        m = w > thr
        if m.sum() == 0:
            continue
        if (1 - m.sum() / len(y)) * 100 > TRACE_LOSS:
            continue
        b = recalls(y[m], pred[m])[0] * 100
        if best is None or b > best[1]:
            best = (thr, b)
    return float(best[0])


def frame(pr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = pr["true_class"].map(CLS).to_numpy(int)
    pred = pr["prediction"].map(CLS).to_numpy(int)
    return y, pred, pr["wasserstein"].to_numpy(float), pr["kept"].astype(str).eq("True").to_numpy()


def row(cell: str, name: str, unit: str, y, pred, kept, thr_rule: str, thr: float) -> dict:
    a, k = recalls(y, pred), recalls(y[kept], pred[kept]) if kept.any() else (np.nan,) * 3
    return {"cell": cell, "run_name": name, "unit": unit, "threshold_rule": thr_rule, "threshold": thr,
            "n": len(y), "n_HT": int((y == 0).sum()), "n_SNAP": int((y == 1).sum()),
            "ba_all": a[0], "recall_HT_all": a[1], "recall_SNAP_all": a[2], "kept_frac": float(kept.mean()),
            "n_kept": int(kept.sum()), "ba_kept": k[0], "recall_HT_kept": k[1], "recall_SNAP_kept": k[2]}


def run_rows(cell: str, name: str, run: Path, mixed: pd.DataFrame, val: pd.DataFrame | None) -> list[dict]:
    import json
    thr_val = json.load(open(run / "threshold.json"))["wasserstein_threshold"]
    y, pred, w, kept = frame(mixed)
    if not np.array_equal(kept, w > thr_val):
        raise ValueError(f"{name}: kept column is not wasserstein > threshold.json")
    t_test = paper_threshold(w, y, pred)
    out = [row(cell, name, "DFK851 slide3", y, pred, kept, "own_val", thr_val),
           row(cell, name, "DFK851 slide3", y, pred, w > t_test, "on_test", t_test)]
    if val is not None:
        yv, pv, wv, kv = frame(val)
        out.append(row(cell, name, "own validation", yv, pv, kv, "own_val", thr_val))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c-transfer", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/dfk851_b/c_transfer")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/dfk851_b")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    keep_cols = s.COLS + ["p_SNAP", "prediction", "wasserstein", "kept"]
    rows, preds = [], []
    for cfg in sorted((HERE / "ml/dfk851").glob(f"r6_within_{s.DS}_s*.yaml")):
        name = cfg.stem
        run = latest(s.MODELS, name)
        losses, _ = s.run_log(run, name)
        if not losses or min(losses) >= PLATEAU_LOSS:
            raise SystemExit(f"{name}: not trained; the ensemble rule of point 13 would drop it")
        pr = pd.read_csv(run / "predictions.csv", dtype=s.DT, low_memory=False)
        pr = pr[pr["true_class"].isin(["HT", "SNAP"])]
        mixed = pr[(pr["set"] == "test_mixed") & (pr["dataset"] == s.DS) & (pr["slide"] == "slide3")]
        rows += run_rows("B", name, run, mixed, pr[pr["set"] == "val"])
        preds.append(mixed[keep_cols].assign(cell="B", run_name=name))
    for f in sorted(a.c_transfer.glob("r6_pooled_s*.csv")):
        name = f.stem
        pr = pd.read_csv(f, dtype=s.DT, low_memory=False)
        run = s.MODELS / pr["run"].iloc[0]
        mixed = pr[(pr["dataset"] == s.DS) & (pr["slide"] == "slide3") & pr["true_class"].isin(["HT", "SNAP"])]
        own = pd.read_csv(run / "predictions.csv", dtype=s.DT, low_memory=False)
        own = own[(own["set"] == "val") & own["true_class"].isin(["HT", "SNAP"])]
        rows += run_rows("C transfer", name, run, mixed, own)
        preds.append(mixed[keep_cols].assign(cell="C transfer", run_name=name))
    runs = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(a.out / "ba_filter_runs.csv", index=False)
    met = ["ba_all", "recall_HT_all", "recall_SNAP_all", "kept_frac", "ba_kept", "recall_HT_kept", "recall_SNAP_kept"]
    summ = runs.groupby(["cell", "unit", "threshold_rule"], sort=False)[met].agg(["mean", "std"])
    summ.columns = [f"{m}_{st}" for m, st in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(a.out / "ba_filter_summary.csv", index=False)

    # ensemble (the user's rule): mean p_SNAP, argmax; mean certainty against the mean of the runs' thresholds
    p = pd.concat(preds, ignore_index=True)
    e = (p.groupby(["cell"] + s.COLS, as_index=False)
          .agg(p_SNAP=("p_SNAP", "mean"), wd=("wasserstein", "mean"), n_runs=("run_name", "nunique")))
    if (e["n_runs"] != 5).any():
        raise ValueError("a trace is not predicted by all five runs of its cell")
    e["pred"] = (e["p_SNAP"] >= 0.5).astype(int)
    e["y"] = e["true_class"].map(CLS)
    own = runs[(runs["unit"] == "DFK851 slide3") & (runs["threshold_rule"] == "own_val")]
    thr = {}
    for c, g in e.groupby("cell"):
        thr[(c, "own_val")] = float(own.loc[own["cell"] == c, "threshold"].mean())
        thr[(c, "on_test")] = paper_threshold(g["wd"].to_numpy(), g["y"].to_numpy(int), g["pred"].to_numpy(int))
        e.loc[g.index, "kept"] = g["wd"] > thr[(c, "own_val")]
        e.loc[g.index, "kept_test"] = g["wd"] > thr[(c, "on_test")]
    w = e.pivot_table(index=s.COLS, columns="cell", values=["pred", "kept", "kept_test"], aggfunc="first")
    w.columns = [f"{v}_{c}" for v, c in w.columns]
    w = w.reset_index()
    w["y"] = w["true_class"].map(CLS)
    if len(w) != len(e) // 2:
        raise ValueError("B and C cover different traces")
    fov = (w["slide"] + "/" + w["fov"]).to_numpy()
    ufov = np.unique(fov)
    idx = {f: np.flatnonzero(fov == f) for f in ufov}
    y = w["y"].to_numpy(int)

    def stats(ix: np.ndarray) -> np.ndarray:
        out = []
        for c in ("B", "C transfer"):
            pr_ = w[f"pred_{c}"].to_numpy(int)[ix]
            out.append(recalls(y[ix], pr_)[0])
            for kc in ("kept", "kept_test"):
                k = w[f"{kc}_{c}"].to_numpy(bool)[ix]
                out += [recalls(y[ix][k], pr_[k])[0] if k.any() else np.nan, k.mean()]
        return np.array(out)

    pt = stats(np.arange(len(w)))
    rng = np.random.default_rng(20261118)
    bs = np.array([stats(np.concatenate([idx[f] for f in rng.choice(ufov, len(ufov))])) for _ in range(a.n_boot)])
    names = ["ba_all", "ba_kept_own_val", "kept_frac_own_val", "ba_kept_on_test", "kept_frac_on_test"]
    nn = len(names)
    erows = []
    for j, c in enumerate(("B", "C transfer")):
        for i, nm in enumerate(names):
            col = nn * j + i
            lo, hi = np.nanpercentile(bs[:, col], [2.5, 97.5])
            t = next((thr[(c, r)] for r in ("own_val", "on_test") if nm.endswith(r)), None)
            erows.append({"cell": c, "metric": nm, "threshold": t, "value": pt[col], "lo": lo, "hi": hi})
    sl3 = runs[runs["unit"] == "DFK851 slide3"]
    sd_of = {"ba_all": ("own_val", "ba_all"), "ba_kept_own_val": ("own_val", "ba_kept"), "ba_kept_on_test": ("on_test", "ba_kept")}
    for i, nm in enumerate(names):
        d = bs[:, i] - bs[:, nn + i]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        r = {"cell": "B - C transfer", "metric": nm, "threshold": None, "value": pt[i] - pt[nn + i], "lo": lo, "hi": hi}
        if nm in sd_of:
            rule, col = sd_of[nm]
            sds = sl3[sl3["threshold_rule"] == rule].groupby("cell")[col].std()
            big = float(max(sds["B"], sds["C transfer"]))
            r.update({"seed_sd_B": float(sds["B"]), "seed_sd_C": float(sds["C transfer"]),
                      "read_as_real": bool((lo > 0 or hi < 0) and abs(r["value"]) > 2 * big)})
        erows.append(r)
    ens = pd.DataFrame(erows)
    ens.insert(0, "n_fovs", len(ufov))
    ens.to_csv(a.out / "ba_filter_ensemble.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(runs.drop(columns=["n"]).to_string(index=False))
        print("\nMean (SD) over the five seeds:\n" + summ.to_string(index=False))
        print(f"\nEnsemble (mean p_SNAP of 5 runs; mean certainty > mean of the runs' thresholds), {len(ufov)} FOVs, "
              f"{int((y == 0).sum())} HT, {int((y == 1).sum())} SNAP:\n" + ens.to_string(index=False))
    print(f"\nOutputs: {a.out}")


if __name__ == "__main__":
    main()
