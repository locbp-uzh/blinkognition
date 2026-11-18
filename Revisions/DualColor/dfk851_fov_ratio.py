#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Per-FOV HT:SNAP ratio on DFK851's mixed slide (slide 3), with the per-trace MC-dropout uncertainty
(TRAINING_NOTES.md point 14, design fixed before any per-FOV estimate).

Truth: the SNAP share among a FOV's single-dye traces (filtered traces of class IN_ATTO390 -> HT, IN_ATTO520 ->
SNAP; trace_assignment run_001). FOVs with at least 10 single-dye vesicles carrying a filtered trace.

Models: B = the five r6_within_DFK851 runs (primary), C = the five round-6 r6_pooled runs (reference). Per run,
the 100 MC passes of p_SNAP on slide 3 (apply_models.py --save-passes, in <passes>/{B,C}/) and, from the run
folder, its validation predictions (a = mean p_SNAP of true-HT, b = of true-SNAP validation traces; the paper
filter's per-class retention and the confusion of the kept calls) and threshold.json. Ensemble = averages over
the five runs (the user's rule, point 13): trace p = mean of the runs' MC means, certainty = mean of the runs'
Wasserstein certainties, threshold = mean of theirs; a, b and the filter's rates = means of the runs'.

Estimators of a sample's SNAP share (a FOV, or a designed mixture):
  pacc  primary, adjusted probabilistic count: clip((mean p - a) / (b - a), 0, 1)
        interval: draws of (run, MC pass) shared by all traces of the sample; each trace's class drawn from that
        pass's p; share -> clip((share - a_run) / (b_run - a_run)); 2.5 / 97.5 percentiles
  hf    the paper's filter: kept = certainty > threshold, call = p >= 0.5; kept SNAP / HT call counts corrected
        for class-dependent retention and confusion by inverting the 2 x 2 matrix of validation rates; clipped
  cc    naive classify and count (share of p >= 0.5)
Baselines: the slide-level pacc (all included FOVs pooled) for every FOV; the true slide share (oracle).

Outputs (<out>): fov_estimates.csv, fov_summary.csv, fov_paired.csv, mixtures.csv, mixtures_summary.csv,
mixtures_curve.csv, passes_check.csv; figures/ (locbplots standards: PNG + PDF, one source-data CSV per panel,
caption.txt, a copy of this script).

    python Revisions/DualColor/dfk851_fov_ratio.py [--passes <dir>] [--out <dir>] [--n-boot 2000] [--n-sim 4000]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc import REPO_ROOT  # noqa: E402
from round6_summary import latest  # noqa: E402

DS, SLIDE = "DFK851", "slide3"
SEEDS = (840410, 1, 2, 3, 4)
RUNS = {"B": [f"r6_within_{DS}_s{s}" for s in SEEDS], "C": [f"r6_pooled_s{s}" for s in SEEDS]}
MODELS = REPO_ROOT / "Results/Revisions/DualColor/models"
DFK = REPO_ROOT / "Results/Revisions/DualColor/dfk851_b"
MIN_VESICLES = 10
SHARES = tuple(round(x, 1) for x in np.arange(0.1, 0.95, 0.1))
SIZES = (16, 50, 200)
N_MIX, N_SIM_MIX = 500, 400
SEED = 20261118
KEY = ["extraction_key", "uniqueID"]
LOCBPLOTS = Path(os.environ.get("LOCBPLOTS_MATERIALS", "/Volumes/MediumBerth/Agent/repos/Lab/locbplots/Materials"))


# --- loading -----------------------------------------------------------------------------------------------

def truth_table() -> pd.DataFrame:
    tr = pd.read_csv(REPO_ROOT / f"Results/Revisions/DualColor/{DS}/trace_assignment/run_001/traces.csv",
                     dtype=str, low_memory=False)
    t = tr[(tr["slide"] == SLIDE) & (tr["filtered"] == "True")].copy()
    t["y"] = t["class"].map({"IN_ATTO390": 0.0, "IN_ATTO520": 1.0})
    return t[KEY + ["fov", "class", "vesicle_id", "y"]].reset_index(drop=True)


def load_run(cell: str, name: str, passes: Path) -> dict:
    # our own apply_models.py output; the first one (S3IT job 8522378) stored the keys as object arrays
    z = np.load(passes / cell / f"{name}_passes.npz", allow_pickle=True)
    keys = pd.DataFrame({"extraction_key": z["extraction_key"].astype(str), "uniqueID": z["uniqueID"].astype(str)})
    csv = pd.read_csv(passes / cell / f"{name}.csv", dtype={"extraction_key": str, "uniqueID": str, "slide": str,
                                                              "fov": str, "true_class": str}, low_memory=False)
    if not (csv[KEY].reset_index(drop=True) == keys).all().all():
        raise ValueError(f"{name}: passes and CSV in different trace order")
    run = latest(MODELS, name)
    thr = json.load(open(run / "threshold.json"))["wasserstein_threshold"]
    v = pd.read_csv(run / "predictions.csv", dtype={"true_class": str}, low_memory=False)
    v = v[(v["set"] == "val") & v["true_class"].isin(["HT", "SNAP"])]
    y, p = (v["true_class"] == "SNAP").to_numpy(), v["p_SNAP"].to_numpy(float)
    kept, call = v["kept"].astype(str).eq("True").to_numpy(), (v["prediction"] == "SNAP").to_numpy()
    rates = {"k_HT": kept[~y].mean(), "k_SNAP": kept[y].mean(),
             "tpr": call[y & kept].mean(), "fpr": call[~y & kept].mean()}
    return {"name": name, "run": run.name, "P": z["p_snap"].astype(np.float32), "keys": keys,
            "wd": csv["wasserstein"].to_numpy(float), "csv": csv, "thr": thr,
            "a": p[~y].mean(), "b": p[y].mean(), **rates}


def ensemble(runs: list[dict]) -> dict:
    k0 = runs[0]["keys"]
    for r in runs[1:]:
        if not (r["keys"] == k0).all().all():
            raise ValueError("runs predicted different traces or orders")
    e = {"keys": k0, "P": np.stack([r["P"] for r in runs]),                       # runs x passes x traces
         "p": np.mean([r["P"].mean(axis=0) for r in runs], axis=0),
         "wd": np.mean([r["wd"] for r in runs], axis=0), "thr": np.mean([r["thr"] for r in runs]),
         "a_run": np.array([r["a"] for r in runs]), "b_run": np.array([r["b"] for r in runs])}
    e["a"], e["b"] = e["a_run"].mean(), e["b_run"].mean()
    for k in ("k_HT", "k_SNAP", "tpr", "fpr"):
        e[k] = float(np.mean([r[k] for r in runs]))
    return e


# --- estimators --------------------------------------------------------------------------------------------

def clip01(x):
    return np.clip(x, 0.0, 1.0)


def pacc(e: dict, ix: np.ndarray) -> float:
    return float(clip01((e["p"][ix].mean() - e["a"]) / (e["b"] - e["a"])))


def pacc_interval(e: dict, ix: np.ndarray, n_sim: int, rng: np.random.Generator) -> tuple[float, float]:
    r = rng.integers(0, e["P"].shape[0], n_sim)
    t = rng.integers(0, e["P"].shape[1], n_sim)
    pp = e["P"][r[:, None], t[:, None], ix[None, :]]                               # n_sim x traces
    share = (rng.random(pp.shape) < pp).mean(axis=1)
    est = clip01((share - e["a_run"][r]) / (e["b_run"][r] - e["a_run"][r]))
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(lo), float(hi)


def cc(e: dict, ix: np.ndarray) -> float:
    return float((e["p"][ix] >= 0.5).mean())


def hf(e: dict, ix: np.ndarray) -> tuple[float, int]:
    kept = e["wd"][ix] > e["thr"]
    call = e["p"][ix] >= 0.5
    c = np.array([(kept & call).sum(), (kept & ~call).sum()], float)
    if c.sum() == 0:
        return np.nan, 0
    m = np.array([[e["k_SNAP"] * e["tpr"], e["k_HT"] * e["fpr"]],
                  [e["k_SNAP"] * (1 - e["tpr"]), e["k_HT"] * (1 - e["fpr"])]])
    n_snap, n_ht = np.clip(np.linalg.solve(m, c), 0, None)
    return (float(n_snap / (n_snap + n_ht)) if n_snap + n_ht > 0 else np.nan), int(kept.sum())


# --- summaries ---------------------------------------------------------------------------------------------

def metrics(est: np.ndarray, truth: np.ndarray, lo=None, hi=None) -> dict:
    err = est - truth
    out = {"n": len(est), "bias": err.mean(), "mae": np.abs(err).mean(), "rmse": np.sqrt((err ** 2).mean())}
    if np.ptp(est) > 1e-9:   # not a constant predictor
        r, p = stats.pearsonr(est, truth)
        z, se = np.arctanh(r), 1 / np.sqrt(len(est) - 3)
        slope, intercept = np.polyfit(truth, est, 1)
        out.update({"pearson_r": r, "r_lo": np.tanh(z - 1.96 * se), "r_hi": np.tanh(z + 1.96 * se), "p": p,
                    "slope": slope, "intercept": intercept})
    if lo is not None:
        out.update({"coverage": float(((lo <= truth) & (truth <= hi)).mean()), "mean_width": float((hi - lo).mean())})
    return out


def boot_mae_diff(x: np.ndarray, y: np.ndarray, truth: np.ndarray, n_boot: int, rng) -> tuple[float, float, float]:
    """MAE(x) - MAE(y) over FOVs, 95 % interval from FOVs resampled with replacement."""
    ax, ay = np.abs(x - truth), np.abs(y - truth)
    d = [ax[i].mean() - ay[i].mean() for i in (rng.integers(0, len(truth), len(truth)) for _ in range(n_boot))]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(ax.mean() - ay.mean()), float(lo), float(hi)


# --- figures (locbplots standards) -------------------------------------------------------------------------

def figures(fov: pd.DataFrame, curve: pd.DataFrame, summ: pd.DataFrame, msumm: pd.DataFrame, out: Path) -> None:
    sys.path.insert(0, str(LOCBPLOTS))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.plot_standards import COLORS, apply_axis_standards, apply_rcparams, save_figure, save_source_data
    apply_rcparams()
    col = {"B": COLORS["blue"], "C": COLORS["orange"]}
    lab = {"B": "B, DFK851 alone", "C": "C, six other sessions"}
    fdir = out / "figures"

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharex=True, sharey=True)
    for ax, cell, letter in zip(axes, ("B", "C"), "ab"):
        x, yv = fov["truth"].to_numpy(), fov[f"pacc_{cell}"].to_numpy()
        lo, hi = fov[f"pacc_lo_{cell}"].to_numpy(), fov[f"pacc_hi_{cell}"].to_numpy()
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=COLORS["black"])
        ax.errorbar(x, yv, yerr=[yv - lo, hi - yv], fmt="none", ecolor=col[cell], elinewidth=0.4, alpha=0.25)
        ax.scatter(x, yv, s=9, color=col[cell], edgecolor="none", zorder=3)
        sl, ic = np.polyfit(x, yv, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        boots = []
        brng = np.random.default_rng(SEED)
        for _ in range(1000):
            i = brng.integers(0, len(x), len(x))
            b1, b0 = np.polyfit(x[i], yv[i], 1)
            boots.append(b0 + b1 * xx)
        blo, bhi = np.percentile(boots, [2.5, 97.5], axis=0)
        ax.fill_between(xx, blo, bhi, color=col[cell], alpha=0.2, lw=0)
        ax.plot(xx, ic + sl * xx, color=col[cell], lw=1.2)
        s = summ[(summ["cell"] == cell) & (summ["estimator"] == "pacc")].iloc[0]
        ax.set_title(f"{lab[cell]}\nr = {s['pearson_r']:.2f} [{s['r_lo']:.2f}, {s['r_hi']:.2f}], p = {s['p']:.1g}; "
                     f"slope {s['slope']:.2f}; MAE {s['mae']:.3f}; interval coverage {s['coverage']:.0%}",
                     color="black", fontsize=6)
        ax.set_xlabel("True SNAP share of single-dye traces (fraction)")
        if cell == "B":
            ax.set_ylabel("Estimated SNAP share, adjusted probabilistic count (fraction)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(-0.18 if cell == "B" else -0.08, 1.1, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
        apply_axis_standards(ax)
        save_source_data(fov[["fov", "n_labeled", "truth", f"pacc_{cell}", f"pacc_lo_{cell}", f"pacc_hi_{cell}"]],
                         fdir, f"fov_ratio_panel_{letter}.csv")
    handles = [Line2D([], [], ls="none", marker="o", ms=3, color=COLORS["black"], label="field of view (ensemble estimate)"),
               Line2D([], [], ls="-", lw=0.6, color=COLORS["black"], alpha=0.4, label="95 % interval (MC-dropout simulation)"),
               Line2D([], [], color=COLORS["black"], lw=1.2, label="least-squares fit (95 % CI band)"),
               Line2D([], [], ls="--", color=COLORS["black"], lw=0.8, label="identity")]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, -0.06))
    save_figure(fig, fdir, "fov_ratio.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(SIZES), figsize=(7.0, 2.9), sharey=True)
    for ax, n, letter in zip(axes, SIZES, "abc"):
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=COLORS["black"])
        rows = []
        for cell in ("B", "C"):
            g = curve[(curve["cell"] == cell) & (curve["estimator"] == "pacc") & (curve["size"] == n)]
            ax.fill_between(g["share"], g["mean"] - g["sd"], g["mean"] + g["sd"], color=col[cell], alpha=0.2, lw=0)
            ax.plot(g["share"], g["mean"], color=col[cell], lw=1.2, marker="o", ms=2.5)
            rows.append(g)
        ms = msumm[(msumm["estimator"] == "pacc") & (msumm["size"] == n)].set_index("cell")
        ax.set_title(f"{n} traces per mixture\n" + "\n".join(
            f"{c}: slope {ms.loc[c, 'slope']:.2f}, RMSE {ms.loc[c, 'rmse']:.3f}, coverage {ms.loc[c, 'coverage']:.0%}"
            for c in ("B", "C")), color="black", fontsize=6)
        ax.set_xlabel("Designed SNAP share (fraction)")
        if n == SIZES[0]:
            ax.set_ylabel("Estimated SNAP share (fraction)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(-0.2 if n == SIZES[0] else -0.08, 1.2, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
        apply_axis_standards(ax)
        save_source_data(pd.concat(rows), fdir, f"mixtures_panel_{letter}.csv")
    handles = [Line2D([], [], color=col[c], lw=1.2, marker="o", ms=2.5, label=f"{lab[c]} (mean ± SD)") for c in ("B", "C")]
    handles.append(Line2D([], [], ls="--", color=COLORS["black"], lw=0.8, label="identity"))
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, -0.07))
    save_figure(fig, fdir, "mixtures_calibration.png")
    plt.close(fig)

    n_f = len(fov)
    (fdir / "caption.txt").write_text(
        "Per-field-of-view HT:SNAP composition estimated from single-trace classifications on the DFK851 mixed "
        "slide. (a, b) Estimated versus true SNAP share of single-dye traces per field of view for models trained "
        f"on DFK851 alone (a) or on six other sessions (b); n = {n_f} distinct fields of view with at least "
        f"{MIN_VESICLES} single-dye vesicles carrying a trace; points, ensemble estimates (adjusted probabilistic "
        "count of five models); bars, 95 % intervals simulated from the Monte Carlo dropout passes; line and band, "
        "least-squares fit with 95 % bootstrap confidence band; dashed, identity. Pearson correlation with Fisher "
        "95 % confidence interval.\n\n"
        "Composition estimates for designed mixtures of single-dye traces drawn from the same slide. (a-c) Mean "
        f"(line) and SD (band) of the estimate over {N_MIX} mixtures per designed share at {SIZES[0]}, {SIZES[1]} "
        f"and {SIZES[2]} traces per mixture; dashed, identity. No statistical comparisons were performed.\n")
    shutil.copy(__file__, fdir)


# --- main --------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passes", type=Path, default=DFK / "mc_passes")
    ap.add_argument("--out", type=Path, default=DFK / "fov_ratio")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-sim", type=int, default=4000)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    truth = truth_table()
    runs = {c: [load_run(c, n, a.passes) for n in names] for c, names in RUNS.items()}
    ens = {c: ensemble(rs) for c, rs in runs.items()}
    if not (ens["B"]["keys"] == ens["C"]["keys"]).all().all():
        raise ValueError("B and C passes cover different traces")
    keys = ens["B"]["keys"]
    t = keys.merge(truth, on=KEY, how="left", validate="one_to_one")
    if t["fov"].isna().any() or len(truth) != len(keys):
        raise ValueError(f"passes ({len(keys)} traces) and the slide-3 filtered traces ({len(truth)}) differ")

    # the regenerated passes against the recorded predictions (B: the runs' predictions.csv; C: c_transfer)
    chk = []
    for c, rs in runs.items():
        for r in rs:
            rec = (pd.read_csv(MODELS / r["run"] / "predictions.csv", dtype={"extraction_key": str, "uniqueID": str},
                               low_memory=False).query("set == 'test_mixed' and dataset == @DS") if c == "B" else
                   pd.read_csv(DFK / "c_transfer" / f"{r['name']}.csv", dtype={"extraction_key": str, "uniqueID": str},
                               low_memory=False))
            m = r["keys"].assign(p_new=r["P"].mean(axis=0)).merge(rec[KEY + ["p_SNAP", "kept"]], on=KEY, validate="one_to_one")
            kept_new = r["csv"]["kept"].astype(str).eq("True").to_numpy()
            chk.append({"cell": c, "run": r["name"], "n": len(m), "corr_p": np.corrcoef(m["p_new"], m["p_SNAP"])[0, 1],
                        "max_abs_diff_p": float((m["p_new"] - m["p_SNAP"]).abs().max()),
                        "mean_abs_diff_p": float((m["p_new"] - m["p_SNAP"]).abs().mean()),
                        "kept_agree": float((kept_new == m["kept"].astype(str).eq("True").to_numpy()).mean()),
                        "val_a": r["a"], "val_b": r["b"], "thr": r["thr"]})
    pd.DataFrame(chk).to_csv(a.out / "passes_check.csv", index=False)

    lab = t["y"].notna().to_numpy()
    ves = t[lab].groupby("fov")["vesicle_id"].nunique()
    inc = sorted(ves.index[ves >= MIN_VESICLES])
    print(f"{len(keys)} slide-3 traces, {lab.sum()} single-dye; FOVs {t['fov'].nunique()}, included {len(inc)} "
          f"(>= {MIN_VESICLES} single-dye vesicles with a trace)")

    # per FOV
    rows = []
    for f in inc:
        ix = np.flatnonzero(lab & (t["fov"] == f).to_numpy())
        ia = np.flatnonzero((t["fov"] == f).to_numpy())
        r = {"fov": f, "n_labeled": len(ix), "n_vesicles": int(ves[f]), "n_all": len(ia), "truth": t["y"].to_numpy()[ix].mean()}
        for c, e in ens.items():
            r[f"pacc_{c}"] = pacc(e, ix)
            r[f"pacc_lo_{c}"], r[f"pacc_hi_{c}"] = pacc_interval(e, ix, a.n_sim, rng)
            r[f"hf_{c}"], r[f"n_kept_{c}"] = hf(e, ix)
            r[f"cc_{c}"] = cc(e, ix)
            r[f"pacc_all_traces_{c}"] = pacc(e, ia)
        rows.append(r)
    fov = pd.DataFrame(rows)
    pooled = np.flatnonzero(lab & t["fov"].isin(inc).to_numpy())
    truth_f = fov["truth"].to_numpy()
    base = {"oracle_slide_share": np.full(len(fov), t["y"].to_numpy()[pooled].mean())}
    for c, e in ens.items():
        fov[f"slide_pacc_{c}"] = pacc(e, pooled)
    fov.to_csv(a.out / "fov_estimates.csv", index=False)

    srows, prows = [], []
    for c in ens:
        base_c = {"slide_level_pacc": fov[f"slide_pacc_{c}"].to_numpy(), **base}
        for est in ("pacc", "hf", "cc"):
            x = fov[f"{est}_{c}"].to_numpy()
            ok = np.isfinite(x)
            extra = (fov[f"pacc_lo_{c}"].to_numpy(), fov[f"pacc_hi_{c}"].to_numpy()) if est == "pacc" else (None, None)
            srows.append({"cell": c, "estimator": est, **metrics(x[ok], truth_f[ok], *(v[ok] if v is not None else None for v in extra))})
            for bn, bv in base_c.items():
                d, lo, hi = boot_mae_diff(x[ok], bv[ok], truth_f[ok], a.n_boot, rng)
                prows.append({"cell": c, "estimator": est, "minus": bn, "mae_diff": d, "lo": lo, "hi": hi})
        for bn, bv in base_c.items():
            srows.append({"cell": c, "estimator": f"baseline: {bn}", **metrics(bv, truth_f)})
    d, lo, hi = boot_mae_diff(fov["pacc_B"].to_numpy(), fov["pacc_C"].to_numpy(), truth_f, a.n_boot, rng)
    prows.append({"cell": "B - C", "estimator": "pacc", "minus": "pacc C", "mae_diff": d, "lo": lo, "hi": hi})
    summ, paired = pd.DataFrame(srows), pd.DataFrame(prows)
    summ.to_csv(a.out / "fov_summary.csv", index=False)
    paired.to_csv(a.out / "fov_paired.csv", index=False)

    # designed mixtures: the same trace draws for B and C
    pool = {1: np.flatnonzero((t["y"] == 1).to_numpy() & t["fov"].isin(inc).to_numpy()),
            0: np.flatnonzero((t["y"] == 0).to_numpy() & t["fov"].isin(inc).to_numpy())}
    mrows = []
    for n in SIZES:
        for s in SHARES:
            k = int(round(s * n))
            for m in range(N_MIX):
                ix = np.concatenate([rng.choice(pool[1], k, replace=False), rng.choice(pool[0], n - k, replace=False)])
                for c, e in ens.items():
                    lo, hi = pacc_interval(e, ix, N_SIM_MIX, rng)
                    mrows.append({"size": n, "share": s, "mixture": m, "truth": k / n, "cell": c, "pacc": pacc(e, ix),
                                  "pacc_lo": lo, "pacc_hi": hi, "hf": hf(e, ix)[0], "cc": cc(e, ix)})
    mix = pd.DataFrame(mrows)
    mix.to_csv(a.out / "mixtures.csv", index=False)
    ms, cv = [], []
    for (c, n), g in mix.groupby(["cell", "size"]):
        for est in ("pacc", "hf", "cc"):
            x, ok = g[est].to_numpy(), g[est].notna().to_numpy()
            lohi = (g["pacc_lo"].to_numpy()[ok], g["pacc_hi"].to_numpy()[ok]) if est == "pacc" else (None, None)
            mm = metrics(x[ok], g["truth"].to_numpy()[ok], *lohi)
            ms.append({"cell": c, "size": n, "estimator": est, **{k_: mm.get(k_) for k_ in
                       ("n", "bias", "mae", "rmse", "slope", "intercept", "coverage", "mean_width")}})
            for s, h in g.groupby("share"):
                cv.append({"cell": c, "size": n, "estimator": est, "share": s, "truth": h["truth"].iloc[0],
                           "mean": h[est].mean(), "sd": h[est].std(), "n_mixtures": int(h[est].notna().sum())})
    msumm, curve = pd.DataFrame(ms), pd.DataFrame(cv)
    msumm.to_csv(a.out / "mixtures_summary.csv", index=False)
    curve.to_csv(a.out / "mixtures_curve.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200,
                           "display.float_format", "{:.3f}".format):
        print("Regenerated passes vs recorded predictions:\n" + pd.DataFrame(chk).to_string(index=False))
        print("\nEnsemble constants:", {c: {k: round(float(e[k]), 4) for k in ("a", "b", "thr", "k_HT", "k_SNAP", "tpr", "fpr")}
                                        for c, e in ens.items()})
        print(f"\nPer FOV ({len(fov)} FOVs; truth: mean {truth_f.mean():.3f}, SD {truth_f.std(ddof=1):.3f}):\n"
              + summ.to_string(index=False))
        print("\nMAE differences (95 % FOV bootstrap):\n" + paired.to_string(index=False))
        print("\nDesigned mixtures:\n" + msumm.to_string(index=False))
    if not a.no_figures:
        figures(fov, curve, summ, msumm, a.out)
    print(f"\nOutputs: {a.out}")


if __name__ == "__main__":
    main()
