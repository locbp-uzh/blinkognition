#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Analysis step 3: classification model.

Option A (the model): a Gaussian mixture (full covariance) is fitted to the pooled
objects of all slides in the step 2 feature space (u = asinh(z / 5) of 405, 488,
515) without using the slide labels. The number of components is chosen by BIC.
Each component is then labeled by linear unmixing of its center with the step 1
signatures:

    F = a_390 * S_390 + a_525 * S_525,   S_D[home] = 1, S_D[c] = k_{D->c}

solved by non-negative least squares with each channel weighted by 1 / noise SD.
The component center (u) is converted to flux with the responsibility-weighted
noise SD of its objects. With t_D = a_D / (noise SD of D's home channel):

    ATTO390 only (t_390 >= T > t_525), ATTO525 only, dual (both >= T), no label (neither)

with T = classification.presence_snr. An object's probability of a label is the
summed membership probability of the components with that label; it gets the most
probable label, or 'unassigned' if that probability is below min_probability.

Option B (cross-check): the same unmixing and thresholds applied to every object
directly, without a mixture model.

The cross-tabs of true dye (slide) against assigned label are descriptive only;
the evaluation (uncertainty, cross-validation, other mixing ratios) is step 4.

Outputs (Results/Revisions/Bleedthrough/classification/run_NNN/):
    classification.csv   one row per object: ids, true dye, flags, component, p_<label>,
                         label_A, t_<dye>, label_B
    components.csv       weight, center (z), unmixed amounts and label per component
    bic.csv              BIC per number of components
    crosstab_A.csv, crosstab_B.csv, agreement.csv
    classification.pdf   figure; panel_<x>.csv are its source data
    caption.txt
    manifest.yaml, config.yaml, code/

Usage:
    python Revisions/Bleedthrough/classify.py [--config path] [--set key=value ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from scipy.optimize import nnls
from sklearn.mixture import GaussianMixture

from common import (COLORS, STR_COLUMNS, apply_axis_standards, calibrate, load_config, load_segmentation,
                    make_run_dir, rel_to_repo, resolve_run, save_pdf, setup_logging, write_manifest)

# --- Constants ---
DUAL, NONE, UNASSIGNED = "dual", "no label", "unassigned"
LABEL_COLORS_EXTRA = {DUAL: COLORS["green"], NONE: COLORS["black"], UNASSIGNED: COLORS["pink"]}
FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 2.8
POINT_SIZE = 2.0
POINT_ALPHA = 0.5
TICK_Z = [0, 10, 100, 1000]
PANEL_LABEL_SIZE = 8


# =============================================================================
# Model
# =============================================================================


def signatures(cfg: dict, coef: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Rows = dyes, columns = feature channels: flux per unit of home-channel flux (step 1 k)."""
    rows = {}
    for dye, d in cfg["dyes"].items():
        k = coef[coef["dye"] == dye].set_index("channel")["k_mean"]
        rows[dye] = {c: 1.0 if c == d["home_channel"] else float(k[c]) for c in channels}
    return pd.DataFrame(rows).T[channels]


def unmix(flux: np.ndarray, sd: np.ndarray, sig: pd.DataFrame) -> np.ndarray:
    """Noise-weighted NNLS amounts (n x dyes, in home-channel flux units) for flux rows with noise sd rows."""
    a = sig.to_numpy().T                       # channels x dyes
    out = np.empty((len(flux), a.shape[1]))
    for i, (f, s) in enumerate(zip(flux, sd)):
        out[i], _ = nnls(a / s[:, None], f / s)
    return out


def label_from_t(t: np.ndarray, dyes: list[str], threshold: float) -> np.ndarray:
    """Label per row of t (n x dyes, amounts in home-channel noise SDs)."""
    present = t >= threshold
    labels = np.full(len(t), NONE, dtype=object)
    labels[present.sum(axis=1) > 1] = DUAL
    for j, d in enumerate(dyes):
        labels[present[:, j] & (present.sum(axis=1) == 1)] = d
    return labels


def fit_gmm(x: np.ndarray, cc: dict) -> tuple[GaussianMixture, pd.DataFrame]:
    lo, hi = cc["n_components"]
    fits, rows = {}, []
    for k in range(lo, hi + 1):
        g = GaussianMixture(n_components=k, covariance_type=cc["covariance_type"], n_init=cc["n_init"],
                            random_state=cc["seed"]).fit(x)
        fits[k] = g
        rows.append({"n_components": k, "bic": g.bic(x), "converged": g.converged_})
    bic = pd.DataFrame(rows)
    best = int(bic.loc[bic["bic"].idxmin(), "n_components"])
    if best == hi:
        logging.warning(f"BIC is lowest at the upper end of the search range ({hi}); consider widening it")
    return fits[best], bic


def label_components(gmm: GaussianMixture, resp: np.ndarray, sd: np.ndarray, sig: pd.DataFrame, dyes: list[str],
                     home_idx: list[int], cofactor: float, threshold: float):
    """Label each component by unmixing its center (u -> z -> flux with the responsibility-weighted noise SD).

    Returns (labels, t per component x dye, centers in z).
    """
    centers_z = cofactor * np.sinh(gmm.means_)
    comp_sd = (resp.T @ sd) / resp.sum(axis=0)[:, None]
    comp_t = unmix(centers_z * comp_sd, comp_sd, sig) / comp_sd[:, home_idx]
    return label_from_t(comp_t, dyes, threshold), comp_t, centers_z


def assign_labels(resp: np.ndarray, comp_label: np.ndarray, dyes: list[str], min_probability: float):
    """Per-object label probabilities (summed over components) and the label, or 'unassigned'."""
    label_set = dyes + [DUAL, NONE]
    p = pd.DataFrame({f"p_{l}": resp[:, comp_label == l].sum(axis=1) for l in label_set})
    best = np.array(label_set, dtype=object)[p.to_numpy().argmax(axis=1)]
    return p, np.where(p.max(axis=1).to_numpy() >= min_probability, best, UNASSIGNED)


def label_objects_unmixing(flux: np.ndarray, sd: np.ndarray, sig: pd.DataFrame, dyes: list[str],
                           home_idx: list[int], threshold: float):
    """Option B: per-object unmixed amounts t (in home-channel noise SDs) and labels."""
    t = unmix(flux, sd, sig) / sd[:, home_idx]
    return t, label_from_t(t, dyes, threshold)


def load_inputs(cfg: dict) -> dict:
    """Features, per-object noise SD and flux, signatures and the runs they come from."""
    cc = cfg["classification"]
    feat_run = resolve_run(cfg, "features", cc["features_run"])
    coef_run = resolve_run(cfg, "coefficients", cc["coefficients_run"])
    with open(feat_run / "config.yaml") as f:
        fcfg = yaml.safe_load(f)["features"]
    with open(feat_run / "manifest.yaml") as f:
        seg_run = resolve_run(cfg, "segmentation", yaml.safe_load(f)["segmentation_run"])
    channels, cofactor = fcfg["channels"], fcfg["asinh_cofactor"]
    dyes = list(cfg["dyes"])

    feat = pd.read_csv(feat_run / "features.csv", dtype=STR_COLUMNS)
    ves, blanks, _ = load_segmentation(seg_run, cfg)
    ves, cal = calibrate(ves, blanks, list(cfg["channels"]))
    sd_wide = cal.pivot_table(index=["slide", "fov"], columns="channel", values="noise_sd")[channels]
    sd = feat[["slide", "fov"]].merge(sd_wide, left_on=["slide", "fov"], right_index=True, how="left")[channels]
    if sd.isna().any().any():
        raise ValueError("Missing noise SD for some objects")
    sd = sd.to_numpy()
    flux = feat[[f"z_{c}" for c in channels]].to_numpy() * sd
    coef = pd.read_csv(coef_run / "coefficients.csv", dtype={"channel": str, "home": str})
    return {"feat": feat, "sd": sd, "flux": flux, "coef": coef, "sig": signatures(cfg, coef, channels),
            "channels": channels, "cofactor": cofactor, "dyes": dyes,
            "home_idx": [channels.index(cfg["dyes"][d]["home_channel"]) for d in dyes],
            "calibrated_vesicles": ves, "feat_run": feat_run, "coef_run": coef_run, "seg_run": seg_run}


# =============================================================================
# Figure
# =============================================================================


def _z_ticks(ax, cofactor: float, lim: tuple[float, float]) -> None:
    z = np.array(TICK_Z, dtype=float)
    t = np.arcsinh(z / cofactor)
    keep = (t >= lim[0]) & (t <= lim[1])
    ax.set_xticks(t[keep], [f"{v:g}" for v in z[keep]])
    ax.set_yticks(t[keep], [f"{v:g}" for v in z[keep]])
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)


def make_figure(res: pd.DataFrame, bic: pd.DataFrame, best: int, colors: dict, cofactor: float, out: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax = axs[0]
    ax.plot(bic["n_components"], bic["bic"] / 1000, "o-", color="black", ms=3, lw=0.9)
    ax.plot(best, bic.loc[bic.n_components == best, "bic"].iloc[0] / 1000, "o", mfc="white", mec="black", ms=6,
            mew=0.9)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("BIC (×1000)")
    ax.set_xticks(bic["n_components"])
    apply_axis_standards(ax)
    bic.to_csv(out / "panel_a.csv", index=False)

    lim = (min(res["u_515"].min(), res["u_405"].min()) - 0.2, max(res["u_515"].max(), res["u_405"].max()) + 0.2)
    order = [l for l in colors if l != UNASSIGNED] + [UNASSIGNED]
    for ax, col, title, panel in ((axs[1], "label_A", "Mixture model (A)", "b"),
                                  (axs[2], "label_B", "Per-object unmixing (B)", "c")):
        for lab in order:
            g = res[res[col] == lab]
            ax.scatter(g["u_515"], g["u_405"], s=POINT_SIZE, color=colors[lab], alpha=POINT_ALPHA, linewidths=0,
                       rasterized=True)
        _z_ticks(ax, cofactor, lim)
        ax.set_xlabel("Flux, 515 channel (noise SDs)")
        ax.set_ylabel("Flux, 405 channel (noise SDs)")
        ax.set_title(title, color="black")
        apply_axis_standards(ax)
        res[["slide", "dye", "fov", "vesicle_id", "u_515", "u_405", col]].to_csv(out / f"panel_{panel}.csv",
                                                                                index=False)
    for ax, label in zip(axs, "abc"):
        ax.text(-0.3, 1.04, label, transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="bold",
                va="bottom", color="black")

    labs = [l for l in order if l in set(res["label_A"]) | set(res["label_B"])]
    handles = [Line2D([], [], ls="", marker="o", color=colors[l], ms=np.sqrt(POINT_SIZE) * 1.6, mew=0) for l in labs]
    fig.legend(handles, labs, loc="lower center", ncol=len(labs), frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.27, wspace=0.5)
    save_pdf(fig, out / "classification.pdf")
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
    cc = cfg["classification"]

    inp = load_inputs(cfg)
    feat, sd, flux, sig = inp["feat"], inp["sd"], inp["flux"], inp["sig"]
    channels, cofactor, dyes, home_idx = inp["channels"], inp["cofactor"], inp["dyes"], inp["home_idx"]
    ucols = [f"u_{c}" for c in channels]

    # --- Option A: mixture model ---
    x = feat[ucols].to_numpy()
    gmm, bic = fit_gmm(x, cc)
    resp = gmm.predict_proba(x)
    comp = resp.argmax(axis=1)
    comp_label, comp_t, centers_z = label_components(gmm, resp, sd, sig, dyes, home_idx, cofactor,
                                                     cc["presence_snr"])
    components = pd.DataFrame({"component": np.arange(gmm.n_components), "weight": gmm.weights_,
                               "n_objects": np.bincount(comp, minlength=gmm.n_components),
                               **{f"center_z_{c}": centers_z[:, i] for i, c in enumerate(channels)},
                               **{f"t_{d}": comp_t[:, j] for j, d in enumerate(dyes)},
                               "label": comp_label})
    p, label_a = assign_labels(resp, comp_label, dyes, cc["min_probability"])

    # --- Option B: per-object unmixing ---
    t, label_b = label_objects_unmixing(flux, sd, sig, dyes, home_idx, cc["presence_snr"])

    res = feat[["slide", "dye", "fov", "vesicle_id", "detected_by", "crowded", "nonlinear"] + ucols].copy()
    res["component"] = comp
    res = pd.concat([res, p], axis=1)
    res["label_A"] = label_a
    for j, d in enumerate(dyes):
        res[f"t_{d}"] = t[:, j]
    res["label_B"] = label_b

    run = make_run_dir(cfg, "classification")
    res.to_csv(run / "classification.csv", index=False)
    components.to_csv(run / "components.csv", index=False)
    bic.to_csv(run / "bic.csv", index=False)
    cols = dyes + [DUAL, NONE, UNASSIGNED]
    xa = pd.crosstab(res["dye"], res["label_A"]).reindex(columns=cols, fill_value=0)
    xb = pd.crosstab(res["dye"], res["label_B"]).reindex(columns=cols, fill_value=0)
    agree = pd.crosstab(res["label_A"], res["label_B"])
    xa.to_csv(run / "crosstab_A.csv")
    xb.to_csv(run / "crosstab_B.csv")
    agree.to_csv(run / "agreement.csv")
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        logging.info(f"BIC chose {gmm.n_components} components")
        logging.info("Components:\n" + components.to_string(index=False))
        logging.info("True dye x label, model A:\n" + xa.to_string())
        logging.info("True dye x label, unmixing B:\n" + xb.to_string())
        logging.info("A (rows) x B (columns):\n" + agree.to_string())

    colors = {d: COLORS[cfg["dyes"][d]["color"]] for d in dyes} | LABEL_COLORS_EXTRA
    make_figure(res, bic, gmm.n_components, colors, cofactor, run)
    (run / "caption.txt").write_text(
        "Classification of the in-silico mix. (a) Bayesian information criterion (BIC) of Gaussian mixture "
        f"models with {cc['n_components'][0]}-{cc['n_components'][1]} components (open circle, chosen). "
        "(b) Objects colored by the label of the mixture model: components labeled by linear unmixing of their "
        f"centers with the bleed-through signatures (label present at >= {cc['presence_snr']:g} noise SDs), "
        f"objects with label probability below {cc['min_probability']:g} unassigned. (c) Objects colored by "
        "per-object unmixing with the same threshold. Axes as in the feature-space figure (405 vs 515 only; "
        "the model also uses 488). Channels are named by excitation wavelength (nm). "
        "[Vesicle composition, dye mol%, buffer and temperature to be added.] "
        + "; ".join(f"{d}: n = {int((res['dye'] == d).sum())} objects" for d in dyes)
        + ". No statistical comparisons were performed.\n")
    write_manifest(run, cfg, Path(__file__), [inp["feat_run"] / "features.csv", inp["coef_run"] / "coefficients.csv"],
                   extra={"features_run": rel_to_repo(inp["feat_run"]), "coefficients_run": rel_to_repo(inp["coef_run"]),
                          "segmentation_run": rel_to_repo(inp["seg_run"]), "n_components": int(gmm.n_components)})


if __name__ == "__main__":
    main()
