#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Readouts of TRAINING_NOTES.md point 16 (C fine-tuned on DFK851's single-protein slides) on the single-dye
traces of DFK851's mixed slide (slide 3).

Cells, five runs each (seeds 840410, 1, 2, 3, 4), every run's own MC-dropout predictions:
  C        r6_pooled_s<seed> (never saw DFK851), applied unchanged (apply_models.py, dfk851_b/c_transfer)
  B        r6_within_DFK851_s<seed> (DFK851's single-protein slides, from scratch; point 13)
  FT head  ft_head_DFK851_s<seed> (C's weights, only fc trained on B's data)
  FT full  ft_full_DFK851_s<seed> (C's weights, every weight trained at a tenth of the learning rate)

Per run (runs.csv): epochs, best epoch, trained (point 10), the equal-recall cut t_eer from its own validation
set, AUC, within-FOV AUC, balanced accuracy at 0.5 and at t_eer, kept fraction and balanced accuracy of the
kept (Wasserstein certainty > the run's threshold.json), the fine-tune source checksum.
Ensembles (ensembles.csv; mean p_SNAP of the trained runs, cut = median t_eer, filter = mean certainty > mean
of the runs' thresholds, the user's rule of point 13): the same metrics with 95 % FOV-cluster bootstrap
intervals (slide 3's FOVs with replacement, 2000 draws, the same draws for every cell).
Paired (paired.csv): FT head - C, FT full - C, FT head - B, FT full - B, B - C on those draws; read_as_real for
the AUC and the balanced accuracies (point 11's rule: the interval excludes 0 and |difference| > 2 x the larger
seed SD of the two cells, each metric's own per-run SD; balanced accuracy is a result, point 13).
FOV structure and limits (lod.csv, lod_paired.csv): point 15's two-component model (dfk851_lod.components,
dfk851_lod.cell) on each ensemble's p over point 15's molecules (112 FOVs with at least 10 single-dye
vesicles, 1828 traces): ICC by class, limits of detection both ways and minimum detectable differences on the
diagonal (16/1 to 500/24), quantifiable ranges (pure-slide constants = the mean of each cell's runs'
validation class means); FOV-cluster bootstrap over the 112 FOVs, paired across cells. These p are the runs'
own predictions, not point 15's regenerated MC passes, so B's and C's limits differ from point 15's by the MC
draw (mean |p difference| about 0.005).
Retention (retention.csv, secondary, reported only): each cell's ensemble AUC on the older sessions' mixed
slides (C trained on those sessions; the fine-tunes start from it).
Figures (locbplots): ensemble metrics with intervals; limits of detection on the diagonal.

    python Revisions/DualColor/dfk851_ft_summary.py [--n-boot 2000] [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dfk851_lod as L  # noqa: E402
from dc import REPO_ROOT  # noqa: E402
from dfk851_fov_ratio import MIN_VESICLES, truth_table  # noqa: E402
from dfk851_summary import COLS, DS, DT, MODELS, ba, per_run, run_log  # noqa: E402
from round6_summary import (PLATEAU_LOSS, boot_groups, draw, eer_threshold, fov_pair_stats, latest,  # noqa: E402
                            rows_of, within_fov_auc, within_of_draw)

CELLS = ("C", "B", "FT head", "FT full")
LABELS = {"C": "C, six other sessions", "B": "B, DFK851 alone", "FT head": "C fine-tuned, last layer",
          "FT full": "C fine-tuned, all weights", "P": "7 sessions, single-protein validation",
          "M": "7 sessions, mixed-slide validation"}
COLOR_KEYS = ("orange", "blue", "green", "vermillion", "pink", "sky_blue")   # Okabe-Ito, in the order of CELLS
FIG_NAME = "finetune"
CAPTION_LEAD = ("C fine-tuned on DFK851's single-protein slides, compared with C (trained on six other sessions) "
                "and B (DFK851's single-protein slides alone, from scratch) on DFK851's mixed slide; five models per "
                "ensemble.")
PATTERN = {"B": "r6_within_DFK851_s*.yaml", "FT head": "ft_head_DFK851_s*.yaml", "FT full": "ft_full_DFK851_s*.yaml"}
PAIRS = (("FT head", "C"), ("FT full", "C"), ("FT head", "B"), ("FT full", "B"), ("B", "C"))
METRICS = ("auc", "auc_within_fov", "ba_05", "ba_teer", "kept_frac", "ba_kept")
READ = ("auc", "ba_05", "ba_teer", "ba_kept")
KEY = ["extraction_key", "uniqueID"]
SEED = 20261118
LOCBPLOTS = Path(os.environ.get("LOCBPLOTS_MATERIALS", "/Volumes/MediumBerth/Agent/repos/Lab/locbplots/Materials"))


def best_epoch(run: Path, name: str) -> float:
    f = run / f"{name}.log"
    m = re.search(r"Early stopping at epoch \d+ \(best AUC [0-9.]+ at epoch (\d+)\)", f.read_text() if f.exists() else "")
    return float(m.group(1)) if m else np.nan


def val_means(pr: pd.DataFrame) -> tuple[float, float]:
    v = pr[(pr["set"] == "val") & pr["true_class"].isin(["HT", "SNAP"])]
    return float(v.loc[v["true_class"] == "HT", "p_SNAP"].mean()), float(v.loc[v["true_class"] == "SNAP", "p_SNAP"].mean())


def load_cell(cell: str, c_transfer: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """Per-run rows, DFK851 slide-3 predictions, other sessions' mixed predictions, (a, b) per run."""
    keep = COLS + ["p_SNAP", "wasserstein", "kept"]
    runs, mixed, other, ab = [], [], [], []
    if cell == "C":
        sources = [(f.stem, MODELS / pd.read_csv(f, nrows=1)["run"].iloc[0], f) for f in sorted(c_transfer.glob("r6_pooled_s*.csv"))]
    else:
        sources = []
        for cfg in sorted((HERE / "ml/dfk851").glob(PATTERN[cell])):
            run = latest(MODELS, cfg.stem)
            if run is None:
                raise SystemExit(f"{cell}: no finished run {cfg.stem}")
            sources.append((cfg.stem, run, None))
    for name, run, csv in sources:
        own = pd.read_csv(run / "predictions.csv", dtype=DT, low_memory=False)
        own = own[own["true_class"].isin(["HT", "SNAP"])]
        v = own[own["set"] == "val"]
        t = eer_threshold((v["true_class"] == "SNAP").to_numpy(int), v["p_SNAP"].to_numpy())
        thr = json.load(open(run / "threshold.json"))["wasserstein_threshold"]
        if csv is not None:   # C: apply_models' prediction of DFK851
            mx = pd.read_csv(csv, dtype=DT, low_memory=False)
            mx = mx[(mx["dataset"] == DS) & mx["true_class"].isin(["HT", "SNAP"])]
        else:
            mx = own[(own["set"] == "test_mixed") & (own["dataset"] == DS)]
        mx = mx[mx["slide"] == "slide3"]
        if not np.array_equal(mx["kept"].astype(bool).to_numpy(), (mx["wasserstein"] > thr).to_numpy()):
            raise ValueError(f"{name}: kept is not wasserstein > threshold.json")
        losses, n_train = run_log(run, name)
        r = per_run(name, cell, run, mx, t, losses, n_train)
        cfg = json.load(open(run / "config_full.json"))
        r.update({"best_epoch": best_epoch(run, name), "threshold": thr,
                  "ft_source_md5": (cfg.get("finetune_source") or {}).get("best_model_md5")})
        runs.append(r)
        mixed.append(mx[keep].assign(run_name=name))
        om = own[(own["set"] == "test_mixed") & (own["dataset"] != DS)]
        other.append(om[COLS + ["p_SNAP"]].assign(run_name=name))
        ab.append(val_means(own))
    return pd.DataFrame(runs), pd.concat(mixed, ignore_index=True), pd.concat(other, ignore_index=True), ab


def ens_metrics(y, p, t, w, thr, fov) -> np.ndarray:
    k = w > thr
    return np.array([roc_auc_score(y, p), within_fov_auc(y, p, fov), ba(y, p, 0.5), ba(y, p, t), k.mean(),
                     ba(y[k], p[k], 0.5) if k.any() and len(np.unique(y[k])) == 2 else np.nan])


def molecules(e: pd.DataFrame) -> tuple[pd.DataFrame, list, np.ndarray, np.ndarray]:
    """Point 15's molecules (labeled traces of the FOVs with >= MIN_VESICLES single-dye vesicles) with each cell's p."""
    truth = truth_table()
    lab = truth[truth["y"].notna()]
    ves = lab.groupby("fov")["vesicle_id"].nunique()
    inc = sorted(ves.index[ves >= MIN_VESICLES])
    mol = lab[lab["fov"].isin(inc)].merge(e, on=KEY, how="left", validate="one_to_one", suffixes=("", "_e"))
    if mol[[f"p_{c}" for c in CELLS]].isna().any().any() or len(mol) != 1828:
        raise ValueError("point 15's molecules are not all predicted by every cell")
    return mol, inc, pd.Categorical(mol["fov"], categories=inc).codes.astype(int), mol["y"].to_numpy().astype(int)


def lod_tables(e: pd.DataFrame, ab: dict, n_boot: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point 15's model on each cell's ensemble p over point 15's molecules; paired FOV bootstrap. Also the
    counting line of point 17: the uncalibrated share reads offset + slope x s on DFK851's mixed slide."""
    mol, inc, fov, cls = molecules(e)
    J = len(inc)
    cnt, S, Q = np.zeros((J, 2)), np.zeros((J, 2, len(CELLS))), np.zeros((J, 2, len(CELLS)))
    for c in (0, 1):
        sel = cls == c
        cnt[:, c] = np.bincount(fov[sel], minlength=J)
        for mi, cell in enumerate(CELLS):
            p = mol[f"p_{cell}"].to_numpy()[sel]
            S[:, c, mi] = np.bincount(fov[sel], weights=p, minlength=J)
            Q[:, c, mi] = np.bincount(fov[sel], weights=p ** 2, minlength=J)
    keys = ("lod_snap", "lod_ht", "mdd_0.10", "mdd_0.50", "loq_snap_uncal", "loq_ht_uncal", "loq_snap_cal", "loq_ht_cal")

    def one(idx) -> dict:
        out = {}
        for mi, cell in enumerate(CELLS):
            cp = L.components(cnt[idx], S[idx], Q[idx], mi)
            r = L.comp_row(cp)
            out[(cell, "icc_HT")], out[(cell, "icc_SNAP")], out[(cell, "separation")] = r["icc_HT"], r["icc_SNAP"], r["separation"]
            a0, b0 = ab[cell]
            out[(cell, "count_offset")] = (r["a_mixed"] - a0) / (b0 - a0)   # what an all-HT sample reads
            out[(cell, "count_slope")] = r["separation"] / (b0 - a0)        # 1 = no compression
            for n, k in L.DIAG:
                c = L.cell(cp, ab[cell], n, k)
                for key in keys:
                    out[(cell, f"{key}@{n}/{k}")] = c[key]
        return out

    pt = one(np.arange(J))
    boot = [one(rng.integers(0, J, J)) for _ in range(n_boot)]
    rows = []
    for (cell, m), v in pt.items():
        x = np.array([b[(cell, m)] for b in boot], float)
        lo, hi = L.ci(x)
        rows.append({"cell": cell, "metric": m, "value": v, "lo": lo, "hi": hi, "boot_not_reached": int(np.isnan(x).sum())})
    prow = []
    for a_, b_ in PAIRS:
        for m in sorted({m for _, m in pt}):
            d = np.array([b[(a_, m)] - b[(b_, m)] for b in boot], float)
            fin = np.isfinite(d)
            lo, hi = np.quantile(d[fin], [0.025, 0.975]) if fin.mean() >= 0.975 else (np.nan, np.nan)
            prow.append({"pair": f"{a_} - {b_}", "metric": m, "diff": pt[(a_, m)] - pt[(b_, m)], "lo": lo, "hi": hi,
                         "boot_not_finite": int((~fin).sum())})
    return pd.DataFrame(rows), pd.DataFrame(prow)


def figures(ens: pd.DataFrame, lod: pd.DataFrame, out: Path) -> None:
    sys.path.insert(0, str(LOCBPLOTS))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.plot_standards import COLORS, apply_axis_standards, apply_rcparams, save_figure, save_source_data
    apply_rcparams()
    col = {c: COLORS[k] for c, k in zip(CELLS, COLOR_KEYS)}
    lab = LABELS
    fdir = out / "figures"
    shown = (("auc", "AUC"), ("ba_05", "BA at 0.5"), ("ba_teer", "BA at the cut"), ("ba_kept", "BA of kept traces"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    for i, (m, name) in enumerate(shown):
        for j, cell in enumerate(CELLS):
            r = ens[(ens["cell"] == cell) & (ens["metric"] == m)].iloc[0]
            x = i + (j - 1.5) * 0.17
            ax.errorbar([x], [r["value"]], yerr=[[r["value"] - r["lo"]], [r["hi"] - r["value"]]], fmt="o", ms=3.5,
                        color=col[cell], elinewidth=0.8, capsize=0, label=lab[cell] if i == 0 else None)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([n for _, n in shown])
    ax.set_ylabel("Ensemble value on DFK851's mixed slide")
    ax.text(-0.16, 1.04, "a", transform=ax.transAxes, fontsize=8, fontweight="bold")
    apply_axis_standards(ax)
    save_source_data(ens[ens["metric"].isin([m for m, _ in shown])], fdir, f"{FIG_NAME}_panel_a.csv")
    ax = axes[1]
    rows = []
    for j, cell in enumerate(CELLS):
        g = lod[(lod["cell"] == cell) & lod["metric"].str.startswith("lod_snap@")].copy()
        g["n"] = g["metric"].str.extract(r"@(\d+)/").astype(int)[0]
        g = g.sort_values("n")
        x = g["n"].to_numpy() * (1 + 0.04 * (j - 1.5))
        y, lo, hi = g["value"].to_numpy(), g["lo"].to_numpy(), np.minimum(g["hi"].to_numpy(), 1.0)
        ok = np.isfinite(y)
        ax.errorbar(x[ok], y[ok], yerr=[y[ok] - lo[ok], hi[ok] - y[ok]], fmt="o-", ms=3, lw=1.0, color=col[cell],
                    elinewidth=0.6, capsize=0)
        rows.append(g)
    ax.set_xscale("log")
    ax.set_xticks([n for n, _ in L.DIAG])
    ax.set_xticklabels([f"{n}/{k}" for n, k in L.DIAG])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("Molecules / FOVs")
    ax.set_ylabel("Limit of detection, SNAP in HT (share)")
    ax.text(-0.2, 1.04, "b", transform=ax.transAxes, fontsize=8, fontweight="bold")
    apply_axis_standards(ax)
    save_source_data(pd.concat(rows), fdir, f"{FIG_NAME}_panel_b.csv")
    fig.legend(frameon=False, loc="lower center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, -0.08))
    save_figure(fig, fdir, f"{FIG_NAME}.png")
    plt.close(fig)
    (fdir / "caption.txt").write_text(
        CAPTION_LEAD + " (a) Ensemble AUC and balanced accuracy (BA) at a p_SNAP cut of 0.5, at the equal-recall cut of the members' "
        "validation sets, and of the traces kept by the uncertainty filter; points, values on all single-dye "
        "traces of the slide (n = 1023 HT, 854 SNAP); bars, 95 % FOV-cluster bootstrap intervals (2000 draws). "
        "(b) Limit of detection of SNAP in HT (5 % false positives and negatives; CLSI EP17) for designs with about "
        "16 molecules per FOV, from a two-component variance model fitted to each ensemble's output on n = 1828 "
        "traces in n = 112 FOVs; bars, 95 % FOV-cluster bootstrap intervals, upper ends above 1 drawn at 1; "
        "missing points, not detectable. Paired differences and the reading rule: paired.csv, lod_paired.csv.\n")
    shutil.copy(__file__, fdir)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c-transfer", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/dfk851_b/c_transfer")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/dfk851_b/finetune")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    loaded = {c: load_cell(c, a.c_transfer) for c in CELLS}
    runs = pd.concat([v[0] for v in loaded.values()], ignore_index=True)
    runs.to_csv(a.out / "runs.csv", index=False)
    if not runs["trained"].all():
        print("untrained runs:", runs.loc[~runs["trained"], "run_name"].tolist())

    # ensembles on the same traces
    e, tcut, thr, ab = None, {}, {}, {}
    for c, (r, mx, _, abl) in loaded.items():
        tr = r[r["trained"]]
        g = (mx[mx["run_name"].isin(tr["run_name"])].groupby(COLS, as_index=False)
             .agg(**{f"p_{c}": ("p_SNAP", "mean"), f"w_{c}": ("wasserstein", "mean"), f"n_{c}": ("run_name", "nunique")}))
        if (g[f"n_{c}"] != len(tr)).any():
            raise ValueError(f"{c}: a trace is not predicted by every run")
        tcut[c], thr[c] = float(tr["t_eer"].median()), float(tr["threshold"].mean())
        ab[c] = tuple(np.mean(abl, axis=0))
        e = g if e is None else e.merge(g, on=COLS, validate="one_to_one")
    if any(len(v[1].drop_duplicates(COLS)) != len(e) for v in loaded.values()):
        raise ValueError("the cells cover different traces")
    e["y"] = (e["true_class"] == "SNAP").astype(int)
    y, fov = e["y"].to_numpy(), (e["slide"] + "/" + e["fov"]).to_numpy()
    pv = {c: (e[f"p_{c}"].to_numpy(), e[f"w_{c}"].to_numpy()) for c in CELLS}
    pt = {c: ens_metrics(y, p, tcut[c], w, thr[c], fov) for c, (p, w) in pv.items()}
    groups = boot_groups(e)
    fs = {c: fov_pair_stats(groups, y, p) for c, (p, _) in pv.items()}
    bs = {c: [] for c in CELLS}
    for _ in range(a.n_boot):
        pk = draw(groups, rng)
        ix, _ = rows_of(groups, pk)
        for c, (p, w) in pv.items():
            k = w[ix] > thr[c]
            yk, pkk = y[ix][k], p[ix][k]
            bs[c].append([roc_auc_score(y[ix], p[ix]), within_of_draw(fs[c], pk), ba(y[ix], p[ix], 0.5),
                          ba(y[ix], p[ix], tcut[c]), k.mean(),
                          ba(yk, pkk, 0.5) if len(np.unique(yk)) == 2 else np.nan])
    bs = {c: np.array(v) for c, v in bs.items()}
    sd = runs.groupby("cell")[list(METRICS)].std()
    erows = []
    for c in CELLS:
        for i, m in enumerate(METRICS):
            lo, hi = np.nanpercentile(bs[c][:, i], [2.5, 97.5])
            erows.append({"cell": c, "metric": m, "value": pt[c][i], "lo": lo, "hi": hi, "seed_sd": sd.loc[c, m],
                          "per_run_mean": runs.loc[runs["cell"] == c, m].mean(),
                          "cut": tcut[c] if m == "ba_teer" else (thr[c] if m in ("kept_frac", "ba_kept") else np.nan)})
    ens = pd.DataFrame(erows)
    ens.to_csv(a.out / "ensembles.csv", index=False)
    prows = []
    for a_, b_ in PAIRS:
        for i, m in enumerate(METRICS):
            d = bs[a_][:, i] - bs[b_][:, i]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            diff = pt[a_][i] - pt[b_][i]
            margin = 2 * max(sd.loc[a_, m], sd.loc[b_, m])
            rule = bool((lo > 0 or hi < 0) and abs(diff) > margin)
            prows.append({"pair": f"{a_} - {b_}", "metric": m, "diff": diff, "lo": lo, "hi": hi,
                          f"seed_sd_{a_}": sd.loc[a_, m], f"seed_sd_{b_}": sd.loc[b_, m], "margin": margin,
                          "rule_met": rule, "read_as_real": rule if m in READ else None})
    paired = pd.DataFrame(prows)
    paired.to_csv(a.out / "paired.csv", index=False)

    # point 15's limits for every cell
    lod, lodp = lod_tables(e[KEY + [f"p_{c}" for c in CELLS]], ab, a.n_boot, rng)
    lod.to_csv(a.out / "lod.csv", index=False)
    lodp.to_csv(a.out / "lod_paired.csv", index=False)

    # retention on the older sessions' mixed slides (reported only)
    ret = []
    for c, (r, _, om, _) in loaded.items():
        tr = r[r["trained"]]["run_name"]
        g = om[om["run_name"].isin(tr)].groupby(COLS, as_index=False)["p_SNAP"].mean()
        for ds, h in g.groupby("dataset"):
            yy = (h["true_class"] == "SNAP").to_numpy(int)
            ret.append({"cell": c, "dataset": ds, "n_HT": int((yy == 0).sum()), "n_SNAP": int((yy == 1).sum()),
                        "auc": roc_auc_score(yy, h["p_SNAP"]) if len(np.unique(yy)) == 2 else np.nan})
    pd.DataFrame(ret).pivot(index="dataset", columns="cell", values="auc").to_csv(a.out / "retention.csv")

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(runs[["cell", "run_name", "epochs", "best_epoch", "trained", "t_eer", "auc", "ba_05", "ba_teer", "kept_frac",
                    "ba_kept", "ft_source_md5"]].to_string(index=False))
        print(ens.to_string(index=False))
        print(paired[paired["metric"].isin(READ)].to_string(index=False))
        print(lod[lod["metric"].str.match(r"(lod_snap|lod_ht|mdd_0.50)@|icc|count")].to_string(index=False))
        print(lodp[lodp["metric"].str.match(r"(lod_snap|lod_ht)@|icc|count")].to_string(index=False))
        print(pd.read_csv(a.out / "retention.csv").to_string(index=False))
    if not a.no_figures:
        figures(ens, lod, a.out)
    return {"e": e, "ab": ab, "runs": runs, "out": a.out, "rng": rng}


if __name__ == "__main__":
    main()
