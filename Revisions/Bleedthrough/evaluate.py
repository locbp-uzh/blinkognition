#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Analysis step 4: evaluation of the classification on the in-silico mix.

Every object's true dye is its slide's dye. Per true dye, each object falls in one
of: own dye, other dye (the critical error), dual, no label, unassigned. These are
reported separately, not as one accuracy, because 'no label' on a slide can be a
correctly flagged impurity. Rates are computed per FOV and summarized as mean +- SD
across FOVs (the FOV is the level at which measurements repeat).

1. Base: the labels of the step 3 classification run (model A and unmixing B).
2. Held-out FOVs: for each FOV, the step 1 signatures are recomputed and the
   mixture model refitted (BIC, labeling) without that FOV; the held-out FOV is
   then classified with the refitted model. Unmixing B is rerun with the fold's
   signatures.
3. Mixing ratios: objects are subsampled at random to other ratios of the two
   dyes (n_draws draws each), the mixture model is refitted per draw, and the
   pooled rates of the draw are summarized as mean +- SD across draws.
4. Threshold sensitivity: the base mixture model relabeled for every combination
   of presence_snr and min_probability; unmixing B for every presence_snr.
5. Chance overlap: for vesicles placed at random, the probability that a vesicle
   has one of the other dye within d px is 1 - exp(-density * pi * d^2). Reported
   at the densities observed here and as the density giving 1 % overlap.

Outputs (Results/Revisions/Bleedthrough/evaluation/run_NNN/):
    base_per_fov.csv, base_summary.csv
    heldout_per_fov.csv, heldout_summary.csv, heldout_models.csv
    mixing_per_draw.csv, mixing_summary.csv
    threshold_per_fov.csv, threshold_summary.csv
    overlap.csv
    evaluation.pdf       figure; panel_<x>.csv are its source data
    caption.txt
    manifest.yaml, config.yaml, code/

Usage:
    python Revisions/Bleedthrough/evaluate.py [--config path] [--set key=value ...]
"""

from __future__ import annotations

import argparse
import logging
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from matplotlib.lines import Line2D

import classify as cl
import coefficients as co
from common import (COLORS, STR_COLUMNS, apply_axis_standards, dye_colors, load_config, make_run_dir,
                    rel_to_repo, resolve_run, save_pdf, setup_logging, write_manifest)

# --- Constants ---
OWN, OTHER = "own dye", "other dye"
CATEGORIES = [OWN, OTHER, cl.DUAL, cl.NONE, cl.UNASSIGNED]
METHODS = {"A": ("model A", COLORS["black"], "o"),
           "A_heldout": ("model A, held-out FOV", COLORS["sky_blue"], "s"),
           "B": ("unmixing B", COLORS["vermillion"], "D")}
PROB_STYLES = {0.90: ":", 0.95: "-", 0.99: "--"}
FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 5.4
PANEL_LABEL_SIZE = 8
FOV_POINT_SIZE = 9.0
METHOD_OFFSET = 0.22


# =============================================================================
# Scoring
# =============================================================================


def categorize(true_dye: pd.Series, label: pd.Series) -> pd.Series:
    """Map (true dye, assigned label) to own dye / other dye / dual / no label / unassigned."""
    out = label.astype(object).copy()
    is_dye = ~label.isin([cl.DUAL, cl.NONE, cl.UNASSIGNED])
    out[is_dye & (label == true_dye)] = OWN
    out[is_dye & (label != true_dye)] = OTHER
    return out


def score_per_fov(df: pd.DataFrame, label_col: str, **tags) -> pd.DataFrame:
    """Fraction of each category per (true dye, slide, FOV)."""
    cat = categorize(df["dye"], df[label_col])
    rows = []
    for (dye, slide, fov), idx in df.groupby(["dye", "slide", "fov"]).groups.items():
        c = cat.loc[idx]
        rows.append({**tags, "dye": dye, "slide": slide, "fov": fov, "n": len(c),
                     **{k: float((c == k).mean()) for k in CATEGORIES}})
    return pd.DataFrame(rows)


def score_pooled(df: pd.DataFrame, label_col: str, **tags) -> pd.DataFrame:
    cat = categorize(df["dye"], df[label_col])
    return pd.DataFrame([{**tags, "dye": dye, "n": len(idx), **{k: float((cat.loc[idx] == k).mean()) for k in CATEGORIES}}
                         for dye, idx in df.groupby("dye").groups.items()])


def summarize(table: pd.DataFrame, by: list[str], unit: str) -> pd.DataFrame:
    """Mean and SD (ddof=1) of every category over the rows of each group (rows = FOVs or draws)."""
    rows = []
    for key, g in table.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key)) | {f"n_{unit}": len(g), "n_objects": int(g["n"].sum())}
        for k in CATEGORIES:
            row[f"{k}_mean"] = float(g[k].mean())
            row[f"{k}_sd"] = float(g[k].std(ddof=1)) if len(g) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Model runs
# =============================================================================


def fit_and_classify(x_train, sd_train, x_test, sig, inp, cc, presence_snr=None, min_probability=None):
    """Fit model A on the training objects and label the test objects. Returns (labels, n_components)."""
    gmm, _ = cl.fit_gmm(x_train, cc)
    comp_label, _, _ = cl.label_components(gmm, gmm.predict_proba(x_train), sd_train, sig, inp["dyes"],
                                           inp["home_idx"], inp["cofactor"],
                                           cc["presence_snr"] if presence_snr is None else presence_snr)
    _, labels = cl.assign_labels(gmm.predict_proba(x_test), comp_label, inp["dyes"],
                                 cc["min_probability"] if min_probability is None else min_probability)
    return labels, gmm.n_components


def fold_signatures(cfg: dict, inp: dict, held: tuple[str, str]) -> pd.DataFrame:
    """Step 1 signatures recomputed without one FOV."""
    ves = inp["calibrated_vesicles"]
    ves = ves[~((ves["slide"] == held[0]) & (ves["fov"] == held[1]))]
    bt = cfg["bleedthrough"]
    sel = {d: co.select(ves, d, v["home_channel"], bt) for d, v in cfg["dyes"].items()}
    pf = pd.concat([co.estimate_per_fov(sel[d], d, cfg["dyes"][d]["home_channel"], co.target_channels(cfg, d))
                    for d in sel], ignore_index=True)
    return cl.signatures(cfg, co.summarize(pf, sel), inp["channels"])


def run_heldout(cfg: dict, inp: dict, held: tuple[str, str]) -> tuple[pd.DataFrame, dict]:
    cc = cfg["classification"]
    feat, sd = inp["feat"], inp["sd"]
    test = ((feat["slide"] == held[0]) & (feat["fov"] == held[1])).to_numpy()
    ucols = [f"u_{c}" for c in inp["channels"]]
    x = feat[ucols].to_numpy()
    sig = fold_signatures(cfg, inp, held)
    labels, n_comp = fit_and_classify(x[~test], sd[~test], x[test], sig, inp, cc)
    _, labels_b = cl.label_objects_unmixing(inp["flux"][test], sd[test], sig, inp["dyes"], inp["home_idx"],
                                            cc["presence_snr"])
    df = feat.loc[test, ["slide", "dye", "fov"]].copy()
    df["label_A"], df["label_B"] = labels, labels_b
    return df, {"slide": held[0], "fov": held[1], "n_components": n_comp,
                **{f"k_{d}_{c}": sig.loc[d, c] for d in sig.index for c in sig.columns}}


def draw_counts(n_avail: dict[str, int], ratio: float, dyes: list[str]) -> dict[str, int]:
    """Largest subsample with first:second = ratio that the available objects allow."""
    a, b = dyes
    if n_avail[a] / n_avail[b] >= ratio:
        return {a: int(round(n_avail[b] * ratio)), b: n_avail[b]}
    return {a: n_avail[a], b: int(round(n_avail[a] / ratio))}


def run_mixing_draw(cfg: dict, inp: dict, ratio: float, draw: int) -> pd.DataFrame:
    ev, cc = cfg["evaluation"], cfg["classification"]
    feat = inp["feat"]
    rng = np.random.default_rng([ev["seed"], draw, int(round(1000 * ratio))])
    n = draw_counts(feat["dye"].value_counts().to_dict(), ratio, inp["dyes"])
    idx = np.concatenate([rng.choice(np.flatnonzero(feat["dye"].to_numpy() == d), size=n[d], replace=False)
                          for d in inp["dyes"]])
    x = feat[[f"u_{c}" for c in inp["channels"]]].to_numpy()[idx]
    labels, n_comp = fit_and_classify(x, inp["sd"][idx], x, inp["sig"], inp, cc)
    df = feat.iloc[idx][["slide", "dye", "fov"]].copy()
    df["label_A"] = labels
    out = score_pooled(df, "label_A", ratio=ratio, draw=draw, n_components=n_comp)
    return out


def overlap_table(inp: dict, cfg: dict, pixel_um: float) -> pd.DataFrame:
    """Chance-overlap probabilities at the observed densities, and the density giving 1 % overlap."""
    ev = cfg["evaluation"]
    border = cfg["detection"]["border_px"]
    area_px = (512 - 2 * border) ** 2       # detection area per FOV
    feat = inp["feat"]
    dens = {d: feat[feat["dye"] == d].groupby(["slide", "fov"]).size().mean() / area_px for d in inp["dyes"]}
    rows = []
    for d in inp["dyes"]:
        other = [o for o in inp["dyes"] if o != d][0]
        for r in ev["overlap_radii_px"]:
            p = 1 - np.exp(-dens[other] * np.pi * r ** 2)
            rho1 = -np.log(0.99) / (np.pi * r ** 2)
            rows.append({"dye": d, "other_dye": other, "radius_px": r, "radius_um": r * pixel_um,
                         "other_density_per_fov": dens[other] * area_px,
                         "p_other_within_radius": p,
                         "other_density_for_1pct_per_fov": rho1 * area_px,
                         "other_density_for_1pct_per_um2": rho1 / pixel_um ** 2})
    return pd.DataFrame(rows)


# =============================================================================
# Figure
# =============================================================================


def _category_panel(ax, per_fov: dict[str, pd.DataFrame], dye: str, rng) -> None:
    for m, (name, color, marker) in METHODS.items():
        t = per_fov[m]
        t = t[t["dye"] == dye]
        off = (list(METHODS).index(m) - 1) * METHOD_OFFSET
        for i, k in enumerate(CATEGORIES):
            v = 100 * t[k].to_numpy()
            xs = i + off + rng.uniform(-0.04, 0.04, len(v))
            ax.scatter(xs, v, s=FOV_POINT_SIZE, color=color, marker=marker, edgecolors="none", alpha=0.8, zorder=3)
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            ax.errorbar(i + off, v.mean(), yerr=sd, fmt="_", color="black", ms=8, capsize=2, lw=0.8, zorder=4)
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=30, ha="right")
    ax.set_ylabel("Objects of the slide (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{dye} slide", color="black")
    apply_axis_standards(ax)


def make_figure(base_fov, held_fov, mix_sum, thr_sum, cfg, out: Path) -> None:
    colors = dye_colors(cfg)
    dyes = list(cfg["dyes"])
    rng = np.random.default_rng(cfg["evaluation"]["seed"])
    fig, axs = plt.subplots(2, 3, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    per_fov = {"A": base_fov[base_fov.method == "A"], "A_heldout": held_fov[held_fov.method == "A"],
               "B": base_fov[base_fov.method == "B"]}
    for ax, dye, label in zip(axs[0, :2], dyes, "ab"):
        _category_panel(ax, per_fov, dye, rng)
        pd.concat([t.assign(method=m) for m, t in per_fov.items()]).query("dye == @dye").to_csv(
            out / f"panel_{label}.csv", index=False)

    # mixing ratio
    ax_own, ax_other = axs[0, 2], axs[1, 0]
    order = mix_sum.sort_values("ratio")
    xs = np.arange(len(order["ratio"].unique()))
    ratios = sorted(order["ratio"].unique())
    tick = [f"1:{1 / r:g}" if r < 1 else f"{r:.3g}:1" for r in ratios]
    for dye in dyes:
        g = order[order["dye"] == dye].set_index("ratio").loc[ratios]
        for ax, k in ((ax_own, OWN), (ax_other, OTHER)):
            ax.errorbar(xs, 100 * g[f"{k}_mean"], yerr=100 * g[f"{k}_sd"].fillna(0), fmt="o-", color=colors[dye],
                        ms=3, lw=0.9, capsize=2)
    ax_own.set_ylim(top=100)        # fractions cannot exceed 100 %; SD bars are cut there
    for ax, k in ((ax_own, OWN), (ax_other, OTHER)):
        ax.set_xticks(xs, tick)
        ax.set_xlabel(f"Mixing ratio {dyes[0]}:{dyes[1]}")
        ax.set_ylabel(f"Called {k} (%)")
        apply_axis_standards(ax)
    order.to_csv(out / "panel_c.csv", index=False)
    order.to_csv(out / "panel_d.csv", index=False)

    # threshold sensitivity
    ax_t_own, ax_t_other = axs[1, 1], axs[1, 2]
    for dye in dyes:
        for mp, ls in PROB_STYLES.items():
            g = thr_sum[(thr_sum.dye == dye) & (thr_sum.method == "A") & np.isclose(thr_sum.min_probability, mp)]
            g = g.sort_values("presence_snr")
            for ax, k in ((ax_t_own, OWN), (ax_t_other, OTHER)):
                ax.plot(g["presence_snr"], 100 * g[f"{k}_mean"], ls=ls, marker="o", ms=2.5, lw=0.9, color=colors[dye])
        g = thr_sum[(thr_sum.dye == dye) & (thr_sum.method == "B")].sort_values("presence_snr")
        for ax, k in ((ax_t_own, OWN), (ax_t_other, OTHER)):
            ax.plot(g["presence_snr"], 100 * g[f"{k}_mean"], ls="-.", marker="D", ms=2.5, lw=0.9, color=colors[dye],
                    mfc="white")
    for ax, k in ((ax_t_own, OWN), (ax_t_other, OTHER)):
        ax.set_xlabel("Presence threshold (noise SDs)")
        ax.set_ylabel(f"Called {k} (%)")
        ax.set_xticks(cfg["evaluation"]["presence_snr_grid"])
        apply_axis_standards(ax)
    thr_sum.to_csv(out / "panel_e.csv", index=False)
    thr_sum.to_csv(out / "panel_f.csv", index=False)

    for ax, label in zip([axs[0, 0], axs[0, 1], ax_own, ax_other, ax_t_own, ax_t_other], string.ascii_lowercase):
        ax.text(-0.32, 1.06, label, transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="bold",
                va="bottom", color="black")

    h1 = [Line2D([], [], ls="", marker=mk, color=c, ms=np.sqrt(FOV_POINT_SIZE), alpha=0.8, mew=0) for _, c, mk in
          METHODS.values()]
    h1 += [Line2D([], [], ls="", marker="_", color="black", ms=8)]
    n1 = [n for n, _, _ in METHODS.values()] + ["mean ± SD"]
    h2 = [Line2D([], [], ls="-", color=colors[d], lw=0.9) for d in dyes]
    h2 += [Line2D([], [], ls=ls, color="gray", lw=0.9) for ls in PROB_STYLES.values()]
    h2 += [Line2D([], [], ls="-.", marker="D", color="gray", mfc="white", ms=2.5, lw=0.9)]
    n2 = [f"{d} slide" for d in dyes] + [f"A, p ≥ {p:.2f}" for p in PROB_STYLES] + ["unmixing B"]
    fig.legend(h1 + h2, n1 + n2, loc="lower center", ncol=5, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.17, wspace=0.55, hspace=0.75)
    save_pdf(fig, out / "evaluation.pdf")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    ev, cc = cfg["evaluation"], cfg["classification"]
    if len(cfg["dyes"]) != 2:
        raise ValueError("The mixing-ratio and overlap evaluations assume exactly two dyes")

    cls_run = resolve_run(cfg, "classification", ev["classification_run"])
    inp = cl.load_inputs(cfg)
    base = pd.read_csv(cls_run / "classification.csv", dtype=STR_COLUMNS)
    if len(base) != len(inp["feat"]) or not (base["vesicle_id"].to_numpy() == inp["feat"]["vesicle_id"].to_numpy()).all():
        raise ValueError("classification run and features run do not describe the same objects")
    run = make_run_dir(cfg, "evaluation")

    # 1. base
    base_fov = pd.concat([score_per_fov(base, "label_A", method="A"), score_per_fov(base, "label_B", method="B")])
    base_sum = summarize(base_fov, ["method", "dye"], "fov")
    base_fov.to_csv(run / "base_per_fov.csv", index=False)
    base_sum.to_csv(run / "base_summary.csv", index=False)

    # 2. held-out FOVs
    folds = list(inp["feat"][["slide", "fov"]].drop_duplicates().itertuples(index=False, name=None))
    logging.info(f"Held-out FOVs: {len(folds)} folds")
    out = Parallel(n_jobs=ev["n_jobs"])(delayed(run_heldout)(cfg, inp, f) for f in folds)
    held = pd.concat([o[0] for o in out], ignore_index=True)
    held_models = pd.DataFrame([o[1] for o in out])
    held_fov = pd.concat([score_per_fov(held, "label_A", method="A"), score_per_fov(held, "label_B", method="B")])
    held_sum = summarize(held_fov, ["method", "dye"], "fov")
    held_fov.to_csv(run / "heldout_per_fov.csv", index=False)
    held_sum.to_csv(run / "heldout_summary.csv", index=False)
    held_models.to_csv(run / "heldout_models.csv", index=False)

    # 3. mixing ratios (the natural ratio is also refitted by subsampling nothing)
    n_avail = inp["feat"]["dye"].value_counts()
    natural = n_avail[inp["dyes"][0]] / n_avail[inp["dyes"][1]]
    ratios = sorted(set(ev["mixing_ratios"]))
    jobs = [(r, d) for r in ratios for d in range(ev["n_draws"])]
    logging.info(f"Mixing ratios {ratios} x {ev['n_draws']} draws, plus the natural ratio {natural:.2f}")
    mix = pd.concat(Parallel(n_jobs=ev["n_jobs"])(delayed(run_mixing_draw)(cfg, inp, r, d) for r, d in jobs),
                    ignore_index=True)
    nat = score_pooled(base, "label_A", ratio=natural, draw=0, n_components=np.nan)
    mix = pd.concat([mix, nat], ignore_index=True)
    mix_sum = summarize(mix, ["ratio", "dye"], "draws")
    mix.to_csv(run / "mixing_per_draw.csv", index=False)
    mix_sum.to_csv(run / "mixing_summary.csv", index=False)

    # 4. thresholds (base model refitted once, deterministic, then relabeled)
    x = inp["feat"][[f"u_{c}" for c in inp["channels"]]].to_numpy()
    gmm, _ = cl.fit_gmm(x, cc)
    resp = gmm.predict_proba(x)
    rows = []
    for ps in ev["presence_snr_grid"]:
        comp_label, _, _ = cl.label_components(gmm, resp, inp["sd"], inp["sig"], inp["dyes"], inp["home_idx"],
                                               inp["cofactor"], ps)
        for mp in ev["min_probability_grid"]:
            _, lab = cl.assign_labels(resp, comp_label, inp["dyes"], mp)
            rows.append(score_per_fov(base.assign(lab=lab), "lab", method="A", presence_snr=ps, min_probability=mp))
        _, lab_b = cl.label_objects_unmixing(inp["flux"], inp["sd"], inp["sig"], inp["dyes"], inp["home_idx"], ps)
        rows.append(score_per_fov(base.assign(lab=lab_b), "lab", method="B", presence_snr=ps, min_probability=np.nan))
    thr_fov = pd.concat(rows, ignore_index=True)
    thr_sum = summarize(thr_fov.fillna({"min_probability": -1}), ["method", "presence_snr", "min_probability", "dye"],
                        "fov").replace({"min_probability": {-1: np.nan}})
    thr_fov.to_csv(run / "threshold_per_fov.csv", index=False)
    thr_sum.to_csv(run / "threshold_summary.csv", index=False)

    # 5. chance overlap
    with open(inp["seg_run"] / "manifest.yaml") as f:
        pixel_um = yaml.safe_load(f)["pixel_um"]
    ovl = overlap_table(inp, cfg, pixel_um)
    ovl.to_csv(run / "overlap.csv", index=False)

    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        logging.info("Base:\n" + base_sum.to_string(index=False))
        logging.info("Held-out:\n" + held_sum.to_string(index=False))
        logging.info("Held-out models:\n" + held_models.to_string(index=False))
        logging.info("Mixing:\n" + mix_sum.to_string(index=False))
        logging.info("Overlap:\n" + ovl.to_string(index=False))

    make_figure(base_fov, held_fov, mix_sum, thr_sum, cfg, run)
    dyes = inp["dyes"]
    (run / "caption.txt").write_text(
        "Evaluation of the vesicle-label classification on an in-silico mix of two single-label slides. "
        f"(a, b) Fraction of the objects of each slide called their own dye, the other dye, dual, no label or "
        "unassigned, for the mixture model (A), the mixture model refitted without the scored field of view "
        "(FOV) (A, held-out FOV) and per-object unmixing (B); points, FOVs; bars, mean ± SD across FOVs. "
        f"(c, d) Fraction called own dye and other dye after subsampling to other {dyes[0]}:{dyes[1]} ratios and "
        f"refitting; mean ± SD across {ev['n_draws']} random draws, bars cut at 100 % (natural ratio: the full "
        "data, single fit). "
        "(e, f) The same fractions against the threshold for calling a label present, for three probability "
        "cutoffs of the mixture model and for unmixing B; mean across FOVs. "
        "[Vesicle composition, dye mol%, buffer and temperature to be added.] "
        + "; ".join(f"{d}: n = {int((base['dye'] == d).sum())} objects from "
                    f"{base.loc[base['dye'] == d, ['slide', 'fov']].drop_duplicates().shape[0]} FOVs" for d in dyes)
        + ". FOVs are repeated measurements of one slide per label. No statistical comparisons were performed.\n")
    write_manifest(run, cfg, Path(__file__), [cls_run / "classification.csv", inp["feat_run"] / "features.csv"],
                   extra={"classification_run": rel_to_repo(cls_run), "features_run": rel_to_repo(inp["feat_run"]),
                          "coefficients_run": rel_to_repo(inp["coef_run"])})


if __name__ == "__main__":
    main()
