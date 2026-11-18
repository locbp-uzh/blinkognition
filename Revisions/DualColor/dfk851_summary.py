#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Readouts of DFK851 in design B (TRAINING_NOTES.md point 13), on the single-dye traces of DFK851's
mixed slide (slide 3).

B: the runs of ml/dfk851/r6_within_DFK851_s<seed>.yaml (latest run folder of each name with a
predictions.csv). Reference: the five round-6 C models (r6_pooled_s<seed>, trained on the six other
sessions), applied unchanged to DFK851 by apply_models.py (train_pure's MC-dropout prediction), their
equal-recall cuts from their own validation sets (predictions.csv set val).

Per run (runs.csv): epochs, trained (training loss below 0.65 at some epoch, point 10), training samples
after augmentation and optimizer steps per epoch (B only), the equal-recall cut t_eer, AUC, within-FOV
AUC, balanced accuracy at 0.5 and at t_eer, MC-dropout kept fraction and balanced accuracy of the kept.
Ensembles (mean p_SNAP over the trained runs of a cell; cut = median of the members' t_eer) on the units
slide 3, its early half and its late half (split at the median FOV in acquisition order): AUC,
within-FOV AUC, balanced accuracy at 0.5 and at the cut, each with a 95 % FOV-cluster bootstrap interval
(FOVs with replacement, 2000 draws), and B minus reference on the same draws (paired.csv; read_as_real
for the AUC: interval excludes 0 and |difference| > 2 x the larger trained-seed SD of the two cells on
that unit, point 11; a unit with fewer than 50 traces of a class is reported, not read, n_HT / n_SNAP beside it).
seed_sd_B / seed_sd_C are each metric's own trained-seed SD and rule_met what the rule gives on that metric
(before the 2026-11-18 verification the AUC's SD stood on every row). read_as_real: the AUC, and since the
user's decision of 2026-11-18 (after the results) the balanced accuracy at 0.5 and at the cut. Transfer
row (transfer_B.csv): the B ensemble's AUC on each other session's mixed slides, with intervals; for a session
with two mixed slides (DFK785) also per slide (point 11); the share of HT and of SNAP traces called SNAP at 0.5;
and point 11's floor (reading).

    python Revisions/DualColor/dfk851_summary.py [--c-transfer <dir>] [--n-boot 2000] [--out <dir>]
    # test on a round-6 session: B = round 6 B, reference = the C runs' own predictions
    python Revisions/DualColor/dfk851_summary.py --ds DFK846 --b-configs Revisions/DualColor/ml/round6 --c-transfer <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc import REPO_ROOT  # noqa: E402
from round6_summary import (PLATEAU_LOSS, SESSIONS, boot_groups, draw, eer_threshold, fov_pair_stats, latest,  # noqa: E402
                            rows_of, within_fov_auc, within_of_draw)

DS = "DFK851"            # --ds: another session for testing against round 6 (ml/round6 B configs)
COLS = ["dataset", "extraction_key", "uniqueID", "slide", "fov", "true_class"]
DT = {"slide": str, "fov": str, "true_class": str, "uniqueID": str, "extraction_key": str}
MODELS = REPO_ROOT / "Results/Revisions/DualColor/models"
READ = ("auc", "ba_05", "ba_teer")   # metrics read with point 11's rule (BA added by the user 2026-11-18, after the results)


def ba(y: np.ndarray, p: np.ndarray, t) -> float:
    t = np.broadcast_to(np.asarray(t, float), p.shape)
    return (np.mean(p[y == 0] < t[y == 0]) + np.mean(p[y == 1] >= t[y == 1])) / 2


def run_log(run: Path, name: str) -> tuple[list[float], int | None]:
    f = run / f"{name}.log"
    log = f.read_text() if f.exists() else ""
    losses = [float(x) for x in re.findall(r"^Epoch \d+ \S+ train loss: ([0-9.]+)", log, re.M)]
    m = re.search(r"^Train (\d+) \(after augmentation\)", log, re.M)
    return losses, (int(m.group(1)) if m else None)


def per_run(name: str, cell: str, run: Path, preds: pd.DataFrame, t: float, losses: list[float], n_train) -> dict:
    y, p = (preds["true_class"] == "SNAP").to_numpy(int), preds["p_SNAP"].to_numpy()
    k = preds["kept"].astype(bool).to_numpy()
    return {"cell": cell, "run_name": name, "run": run.name, "epochs": len(losses),
            "trained": (not losses) or min(losses) < PLATEAU_LOSS, "min_train_loss": min(losses) if losses else np.nan,
            "n_train_aug": n_train, "steps_per_epoch": math.ceil(n_train / 64) if n_train else np.nan, "t_eer": t,
            "n_HT": int((y == 0).sum()), "n_SNAP": int((y == 1).sum()), "auc": roc_auc_score(y, p),
            "auc_within_fov": within_fov_auc(y, p, (preds["slide"] + "/" + preds["fov"]).to_numpy()),
            "ba_05": ba(y, p, 0.5), "ba_teer": ba(y, p, t), "kept_frac": k.mean(),
            "ba_kept": ba(y[k], p[k], 0.5) if k.any() and len(np.unique(y[k])) == 2 else np.nan}


def val_cut(run: Path) -> float:
    pr = pd.read_csv(run / "predictions.csv", dtype=DT, low_memory=False)
    v = pr[(pr["set"] == "val") & pr["true_class"].isin(["HT", "SNAP"])]
    return eer_threshold((v["true_class"] == "SNAP").to_numpy(int), v["p_SNAP"].to_numpy())


def load_b(cfg_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, preds = [], []
    for cfg in sorted(cfg_dir.glob(f"r6_within_{DS}_s*.yaml")):
        name = cfg.stem
        run = latest(MODELS, name)
        if run is None:
            print(f"missing: {name}")
            continue
        pr = pd.read_csv(run / "predictions.csv", dtype=DT, low_memory=False)
        pr = pr[pr["true_class"].isin(["HT", "SNAP"])]
        v = pr[pr["set"] == "val"]
        t = eer_threshold((v["true_class"] == "SNAP").to_numpy(int), v["p_SNAP"].to_numpy())
        losses, n_train = run_log(run, name)
        mx = pr[pr["set"] == "test_mixed"]
        runs.append(per_run(name, "B", run, mx[mx["dataset"] == DS], t, losses, n_train))
        preds.append(mx[COLS + ["p_SNAP"]].assign(run_name=name))
    return pd.DataFrame(runs), pd.concat(preds, ignore_index=True)


def load_c(cdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, preds = [], []
    for f in sorted(cdir.glob("r6_pooled_s*.csv")):
        name = f.stem
        pr = pd.read_csv(f, dtype=DT, low_memory=False)
        run = MODELS / pr["run"].iloc[0]
        pr = pr[(pr["dataset"] == DS) & pr["true_class"].isin(["HT", "SNAP"])]
        losses, _ = run_log(run, name)
        runs.append(per_run(name, "C transfer", run, pr, val_cut(run), losses, None))
        preds.append(pr[COLS + ["p_SNAP"]].assign(run_name=name))
    return pd.DataFrame(runs), pd.concat(preds, ignore_index=True)


def ensemble(runs: pd.DataFrame, preds: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    tr = runs[runs["trained"]]
    e = preds[preds["run_name"].isin(tr["run_name"])].groupby(COLS, as_index=False)["p_SNAP"].mean()
    return e, float(tr["t_eer"].median())


def stats_of(y, p, t, fov) -> tuple:
    return roc_auc_score(y, p), within_fov_auc(y, p, fov), ba(y, p, 0.5), ba(y, p, t)


def main() -> None:
    global DS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c-transfer", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/dfk851_b/c_transfer")
    ap.add_argument("--ds", default=DS)
    ap.add_argument("--b-configs", type=Path, default=HERE / "ml/dfk851")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/dfk851_b")  # not dfk851: the same folder as DFK851 on a case-insensitive disk
    a = ap.parse_args()
    DS = a.ds
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261118)
    rb, pb = load_b(a.b_configs)
    rc, pc = load_c(a.c_transfer)
    runs = pd.concat([rb, rc], ignore_index=True)
    runs.to_csv(a.out / "runs.csv", index=False)
    if not rb["trained"].any():
        # as round 6 (B DFK846): no trained seed, no ensemble; the per-run table is the result
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
            print(runs.drop(columns=["run"]).to_string(index=False))
        print(f"\nNo trained B run: no ensemble, no paired difference. Outputs: {a.out}")
        return
    eb, tb = ensemble(rb, pb[pb["dataset"] == DS])
    ec, tc = ensemble(rc, pc)
    m = eb.merge(ec, on=COLS, suffixes=("_B", "_C"), validate="one_to_one")
    if len(m) != len(eb) or len(m) != len(ec):
        raise ValueError("B and the C transfer cover different DFK851 traces")
    fovs = pd.read_csv(REPO_ROOT / f"Results/Revisions/DualColor/{DS}/vesicles/run_001/fovs.csv", dtype={"fov": str})
    order = fovs.set_index(["slide", "fov"])["acq_order"].astype(int)
    m["acq_order"] = [order.loc[(s, f)] for s, f in zip(m["slide"], m["fov"])]
    m["y"] = (m["true_class"] == "SNAP").astype(int)
    members = {c: set(runs.loc[(runs["cell"] == c) & runs["trained"], "run_name"]) for c in ("B", "C transfer")}
    per_run_preds = {"B": pb[(pb["dataset"] == DS) & pb["run_name"].isin(members["B"])],
                     "C transfer": pc[pc["run_name"].isin(members["C transfer"])]}
    units = {}
    for sl, g in m.groupby("slide"):
        med = np.median(sorted(g.drop_duplicates("fov")["acq_order"]))
        units[f"{sl}"] = (m["slide"] == sl).to_numpy()
        units[f"{sl} early half"] = ((m["slide"] == sl) & (m["acq_order"] <= med)).to_numpy()
        units[f"{sl} late half"] = ((m["slide"] == sl) & (m["acq_order"] > med)).to_numpy()
    erows, prows = [], []
    names = ("auc", "auc_within_fov", "ba_05", "ba_teer")
    for u, mask in units.items():
        h = m[mask].reset_index(drop=True)
        y, fov = h["y"].to_numpy(), (h["slide"] + "/" + h["fov"]).to_numpy()
        pv = {"B": (h["p_SNAP_B"].to_numpy(), tb), "C transfer": (h["p_SNAP_C"].to_numpy(), tc)}
        pt = {c: stats_of(y, p, t, fov) for c, (p, t) in pv.items()}
        # trained-seed SD of each metric over the cell's runs on this unit (point 11: "the trained-seed SD of both
        # cells on that unit"); each run's balanced accuracy at its own t_eer. Until 2026-11-18 (verification of
        # that day) the AUC's SD was printed on every metric row.
        tcut = {c: runs.loc[runs["cell"] == c].set_index("run_name")["t_eer"] for c in pv}
        sd = {c: pd.DataFrame([stats_of((g["true_class"] == "SNAP").to_numpy(int), g["p_SNAP"].to_numpy(), tcut[c][rn],
                                         (g["slide"] + "/" + g["fov"]).to_numpy())
                               for rn, g in per_run_preds[c].merge(h[COLS], on=COLS).groupby("run_name")]).std().to_numpy()
              for c in pv}
        groups = boot_groups(h)
        fs = {c: fov_pair_stats(groups, y, p) for c, (p, _) in pv.items()}
        bs = {c: [] for c in pv}
        for _ in range(a.n_boot):
            pk = draw(groups, rng)
            ix, _ = rows_of(groups, pk)
            if len(np.unique(y[ix])) < 2:
                continue
            for c, (p, t) in pv.items():
                bs[c].append((roc_auc_score(y[ix], p[ix]), within_of_draw(fs[c], pk), ba(y[ix], p[ix], 0.5), ba(y[ix], p[ix], t)))
        bs = {c: np.array(v) for c, v in bs.items()}
        for c in pv:
            r = {"cell": c, "unit": u, "n_fovs": len(np.unique(fov)), "n_HT": int((y == 0).sum()), "n_SNAP": int((y == 1).sum()),
                 "cut": pv[c][1], "members": int(runs[(runs["cell"] == c) & runs["trained"]].shape[0])}
            for j, k in enumerate(names):
                r[k] = pt[c][j]
                r[f"{k}_lo"], r[f"{k}_hi"] = np.nanpercentile(bs[c][:, j], [2.5, 97.5])
            erows.append(r)
        diff = bs["B"] - bs["C transfer"]
        for j, k in enumerate(names):
            d0 = pt["B"][j] - pt["C transfer"][j]
            lo, hi = np.nanpercentile(diff[:, j], [2.5, 97.5])
            big = max(sd["B"][j], sd["C transfer"][j])
            met = bool((lo > 0 or hi < 0) and abs(d0) > 2 * big) if np.isfinite(big) else None
            # read_as_real: the rule on the AUC (declared before extraction) and, by the user's decision of
            # 2026-11-18 (made after the results), on the balanced accuracy at 0.5 and at the cut, each with its own
            # seed SD; rule_met: the same rule on every metric (within-FOV AUC: reported, not read)
            prows.append({"unit": u, "metric": k, "n_HT": int((y == 0).sum()), "n_SNAP": int((y == 1).sum()),
                          "B": pt["B"][j], "C_transfer": pt["C transfer"][j], "diff": d0,
                          "lo": lo, "hi": hi, "seed_sd_B": sd["B"][j], "seed_sd_C": sd["C transfer"][j],
                          "rule_met": met, "read_as_real": met if k in READ else None})
    ens, paired = pd.DataFrame(erows), pd.DataFrame(prows)
    ens.to_csv(a.out / "ensembles.csv", index=False)
    paired.to_csv(a.out / "paired.csv", index=False)
    # transfer row: the B ensemble on the other sessions' mixed slides
    tro = []
    eall, _ = ensemble(rb, pb)
    for ds in [d for d in SESSIONS if d != DS]:
        h = eall[eall["dataset"] == ds].reset_index(drop=True)
        if h.empty:
            continue
        y, p = (h["true_class"] == "SNAP").to_numpy(int), h["p_SNAP"].to_numpy()
        groups = boot_groups(h)
        bo = []
        for _ in range(a.n_boot):
            ix, _ = rows_of(groups, draw(groups, rng))
            if len(np.unique(y[ix])) == 2:
                bo.append(roc_auc_score(y[ix], p[ix]))
        tro.append({"session": ds, "unit": "mixed slides", "n_HT": int((y == 0).sum()), "n_SNAP": int((y == 1).sum()),
                    "auc": roc_auc_score(y, p), "auc_lo": np.percentile(bo, 2.5), "auc_hi": np.percentile(bo, 97.5),
                    "ba_05": ba(y, p, 0.5)})
    # added 2026-11-18 after the verification: point 11's per-slide rows for a session with two mixed slides (DFK785:
    # slide 4 at chance in every model so far), on a separate RNG so that the rows above keep their draws; the share
    # of each class called SNAP at 0.5; and point 11's floor (fewer than 50 traces of a class: reported, not read)
    rng2 = np.random.default_rng(20261119)
    for ds in [d for d in SESSIONS if d != DS]:
        h0 = eall[eall["dataset"] == ds]
        if h0["slide"].nunique() < 2:
            continue
        for sl, h in h0.groupby("slide"):
            h = h.reset_index(drop=True)
            y, p = (h["true_class"] == "SNAP").to_numpy(int), h["p_SNAP"].to_numpy()
            groups = boot_groups(h)
            bo = []
            for _ in range(a.n_boot):
                ix, _ = rows_of(groups, draw(groups, rng2))
                if len(np.unique(y[ix])) == 2:
                    bo.append(roc_auc_score(y[ix], p[ix]))
            tro.append({"session": ds, "unit": sl, "n_HT": int((y == 0).sum()), "n_SNAP": int((y == 1).sum()),
                        "auc": roc_auc_score(y, p), "auc_lo": np.percentile(bo, 2.5), "auc_hi": np.percentile(bo, 97.5),
                        "ba_05": ba(y, p, 0.5)})
    tr_df = pd.DataFrame(tro)
    called = {}
    for r in tr_df.itertuples():
        h = eall[(eall["dataset"] == r.session) & ((eall["slide"] == r.unit) if r.unit != "mixed slides" else True)]
        called[r.Index] = ((h.loc[h["true_class"] == "HT", "p_SNAP"] >= 0.5).mean(),
                           (h.loc[h["true_class"] == "SNAP", "p_SNAP"] >= 0.5).mean())
    tr_df["HT_called_SNAP"] = [called[i][0] for i in tr_df.index]
    tr_df["SNAP_called_SNAP"] = [called[i][1] for i in tr_df.index]
    tr_df["reading"] = np.where((tr_df["n_HT"] >= 50) & (tr_df["n_SNAP"] >= 50), "read", "reported, not read (< 50 of a class)")
    tr_df.to_csv(a.out / "transfer_B.csv", index=False)
    tro = tr_df
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200,
                           "display.float_format", "{:.3f}".format):
        print(runs.drop(columns=["run"]).to_string(index=False))
        print("\n" + ens.to_string(index=False))
        print("\n" + paired.to_string(index=False))
        print("\n" + tro.to_string(index=False))
    print(f"\nOutputs: {a.out}")


if __name__ == "__main__":
    main()
