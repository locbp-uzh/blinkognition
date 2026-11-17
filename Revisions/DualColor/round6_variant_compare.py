#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Design A (leave one experiment out) of round 6 against a variant of it (TRAINING_NOTES.md point 12):
by default the minmax runs (ml/round6/r6_loeo_*) against the clipmax runs (ml/round6_clipmax/r6c_loeo_*),
same sessions, seeds, splits and mixed test traces.

Per variant and session, each session's labeled mixed traces scored by the ensemble (mean p_SNAP of the
trained seeds) of the A runs that left it out: AUC, within-FOV AUC, balanced accuracy at 0.5 and at the
session's own cut (median of the members' equal-recall thresholds on their validation set = the
left-out session's single-protein slides); also all six sessions together and without DFK785 slide 4
(each session at its own ensemble and cut). Per run: AUC, balanced accuracy at 0.5, of the MC-dropout
kept traces, kept fraction, trained flag. Differences (variant - reference) on the same resampled FOVs
(95 % FOV-cluster bootstrap, FOVs with replacement within slide, 2000 draws), with the seed SD of the
per-run AUCs of both variants; read as real only if the interval excludes 0 and the AUC difference
exceeds twice the larger seed SD (point 11's rule).

    python Revisions/DualColor/round6_variant_compare.py [--ref r6_loeo --ref-dir ml/round6]
        [--var r6c_loeo --var-dir ml/round6_clipmax] [--n-boot 2000] [--out <dir>]
Outputs: <out>/compare_ensembles.csv, compare_runs.csv, compare_paired.csv.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc import REPO_ROOT  # noqa: E402
from round6_summary import (KEY, PLATEAU_LOSS, SESSIONS, boot_groups, draw, eer_threshold, fov_pair_stats,  # noqa: E402
                            latest, rows_of, within_fov_auc, within_of_draw)

COLS = KEY + ["slide", "fov", "true_class"]


def ba(y, p, t) -> float:
    t = np.broadcast_to(np.asarray(t, float), p.shape)
    return (np.mean(p[y == 0] < t[y == 0]) + np.mean(p[y == 1] >= t[y == 1])) / 2


def load_variant(models: Path, cfg_dir: Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-run table and the session-own mixed predictions of every run of one variant."""
    runs, preds = [], []
    for cfg in sorted(cfg_dir.glob(f"{prefix}_DFK*_s*.yaml")):
        name = cfg.stem
        m = re.match(rf"^{prefix}_(DFK\d+)_s(\d+)$", name)
        run = latest(models, name)
        if run is None:
            print(f"missing: {name}")
            continue
        ds, seed = m.group(1), int(m.group(2))
        logf = run / f"{name}.log"
        log = logf.read_text() if logf.exists() else None
        losses = [float(x) for x in re.findall(r"^Epoch \d+ \S+ train loss: ([0-9.]+)", log or "", re.M)]
        pr = pd.read_csv(run / "predictions.csv", dtype={"slide": str, "fov": str, "true_class": str, "uniqueID": str,
                                                        "extraction_key": str}, low_memory=False)
        pr = pr[pr["true_class"].isin(["HT", "SNAP"])]
        val = pr[pr["set"] == "val"]
        t = eer_threshold((val["true_class"] == "SNAP").to_numpy(int), val["p_SNAP"].to_numpy())
        mx = pr[(pr["set"] == "test_mixed") & (pr["dataset"] == ds)]
        y, p, k = (mx["true_class"] == "SNAP").to_numpy(int), mx["p_SNAP"].to_numpy(), mx["kept"].astype(bool).to_numpy()
        runs.append({"run_name": name, "run": run.name, "session": ds, "seed": seed, "epochs": len(losses),
                     "trained": log is None or (bool(losses) and min(losses) < PLATEAU_LOSS), "t_eer": t,
                     "auc": roc_auc_score(y, p), "ba_05": ba(y, p, 0.5), "ba_teer": ba(y, p, t),
                     "kept_frac": k.mean(), "ba_kept": ba(y[k], p[k], 0.5) if k.any() else np.nan})
        preds.append(mx[COLS + ["p_SNAP"]].assign(run_name=name, session=ds))
    return pd.DataFrame(runs), pd.concat(preds, ignore_index=True)


def ensemble(runs: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    tr = runs[runs["trained"]]
    p = preds[preds["run_name"].isin(tr["run_name"])]
    e = p.groupby(COLS + ["session"], as_index=False)["p_SNAP"].mean()
    return e.merge(tr.groupby("session")["t_eer"].median().rename("t_eer").reset_index(), on="session")


def units(e: pd.DataFrame) -> dict:
    u = {ds: (e["session"] == ds).to_numpy() for ds in SESSIONS}
    u["all six sessions"] = np.ones(len(e), bool)
    u["all without DFK785 slide4"] = ~((e["session"] == "DFK785") & (e["slide"] == "slide4")).to_numpy()
    return u


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="r6_loeo")
    ap.add_argument("--ref-dir", default="ml/round6")
    ap.add_argument("--var", default="r6c_loeo")
    ap.add_argument("--var-dir", default="ml/round6_clipmax")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--models", default=str(REPO_ROOT / "Results/Revisions/DualColor/models"))
    ap.add_argument("--out", default=str(REPO_ROOT / "Results/Revisions/DualColor/round6_clipmax"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261117)
    loaded = {lab: load_variant(Path(a.models), HERE / d, pre) for lab, pre, d in
              (("ref", a.ref, a.ref_dir), ("var", a.var, a.var_dir))}
    runs = pd.concat([r.assign(variant=lab) for lab, (r, _) in loaded.items()], ignore_index=True)
    runs.to_csv(out / "compare_runs.csv", index=False)
    ens = {lab: ensemble(r, p) for lab, (r, p) in loaded.items()}
    m = ens["ref"].merge(ens["var"], on=COLS + ["session"], suffixes=("_ref", "_var"), validate="one_to_one")
    if len(m) != len(ens["ref"]) or len(m) != len(ens["var"]):
        raise ValueError("the two variants' ensembles cover different mixed traces")
    m["y"] = (m["true_class"] == "SNAP").astype(int)
    fov_id = (m["dataset"] + "/" + m["slide"] + "/" + m["fov"]).to_numpy()
    sd = runs[runs["trained"]].groupby(["variant", "session"])["auc"].std()
    erows, prows = [], []
    for u, mask in units(m).items():
        h = m[mask].reset_index(drop=True)
        y = h["y"].to_numpy()
        stats = {}
        for lab in ("ref", "var"):
            p, t = h[f"p_SNAP_{lab}"].to_numpy(), h[f"t_eer_{lab}"].to_numpy()
            stats[lab] = (p, t)
            erows.append({"variant": lab, "unit": u, "n_HT": int((y == 0).sum()), "n_SNAP": int((y == 1).sum()),
                          "auc": roc_auc_score(y, p), "auc_within_fov": within_fov_auc(y, p, fov_id[mask]),
                          "ba_05": ba(y, p, 0.5), "ba_teer": ba(y, p, t)})
        groups = boot_groups(h)
        fs = {lab: fov_pair_stats(groups, y, stats[lab][0]) for lab in stats}
        bs = []
        for _ in range(a.n_boot):
            pk = draw(groups, rng)
            ix, _ = rows_of(groups, pk)
            if len(np.unique(y[ix])) < 2:
                continue
            row = []
            for lab in ("ref", "var"):
                p, t = stats[lab]
                row.append((roc_auc_score(y[ix], p[ix]), within_of_draw(fs[lab], pk), ba(y[ix], p[ix], 0.5),
                            ba(y[ix], p[ix], t[ix])))
            bs.append(np.subtract(row[1], row[0]))
        bs = np.array(bs)
        ref_r, var_r = (next(r for r in erows[::-1] if r["variant"] == lab and r["unit"] == u) for lab in ("ref", "var"))
        sds = [sd.get((lab, u), np.nan) for lab in ("ref", "var")] if u in SESSIONS else [np.nan, np.nan]
        for j, k in enumerate(("auc", "auc_within_fov", "ba_05", "ba_teer")):
            d0 = var_r[k] - ref_r[k]
            lo, hi = np.nanpercentile(bs[:, j], [2.5, 97.5])
            big = max(sds) if all(np.isfinite(sds)) else np.nan
            prows.append({"unit": u, "metric": k, "ref": ref_r[k], "var": var_r[k], "diff": d0, "lo": lo, "hi": hi,
                          "seed_sd_ref": sds[0], "seed_sd_var": sds[1],
                          "read_as_real": (bool((lo > 0 or hi < 0) and abs(d0) > 2 * big) if k == "auc" and np.isfinite(big)
                                           else None)})
    pd.DataFrame(erows).to_csv(out / "compare_ensembles.csv", index=False)
    pt = pd.DataFrame(prows)
    pt.to_csv(out / "compare_paired.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        print(runs.groupby(["variant", "session"])[["trained", "auc", "ba_05", "ba_teer", "kept_frac", "ba_kept"]]
              .agg({"trained": "sum", "auc": ["mean", "std"], "ba_05": "mean", "ba_teer": "mean", "kept_frac": "mean",
                    "ba_kept": "mean"}).to_string())
        print("\n" + pt.to_string(index=False))
    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
