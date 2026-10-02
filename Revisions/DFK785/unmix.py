#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Vesicle labels for DFK785 by per-object unmixing with 405 + 488 (goal 1).

1. Calibration: per slide, FOV and channel, the blank median is the zero point
   and the blank robust SD the noise (common.calibrate).
2. Signatures from the single-label slides (slide 1 ATTO520, slide 2 ATTO390),
   exactly as in Revisions/Bleedthrough step 1: per FOV the median of
   per-vesicle ratios F_c / F_home over vesicles with home signal >= min_home_snr
   noise SDs, not crowded, not nonlinear; mean over FOVs. Home channels: 405 for
   ATTO390, 488 for ATTO520 (slides 1-2 have no 515, so only 405 and 488 can be
   used for both dyes).
3. Every vesicle on every slide is unmixed with those signatures (noise-weighted
   NNLS, Revisions/Bleedthrough/classify.py) into ATTO390 and ATTO520 amounts t,
   in home-channel noise SDs, and labeled ATTO390, ATTO520, dual (both >= T) or no
   label (neither), with T = unmixing.presence_snr.
4. Check with 515 on the mixed slides (not used for the labels): per label, the
   515 signal relative to the home channel; ATTO520 vesicles should show one tight
   515/488 ratio and ATTO390 vesicles almost no 515.

Outputs (Results/Revisions/DFK785/unmixing/run_NNN/):
    labels.csv              one row per vesicle: ids, slide dye, flags, F_<ch>, z_<ch>, t_<dye>, label
    signatures.csv          k per dye and channel (mean +- SD across FOVs); signatures_per_fov.csv
    label_counts.csv        slide x label
    check_515.csv           per mixed slide and label: median 515 ratios and the fraction with 515 signal
    unmixing.pdf            figure; panel_<x>.csv are its source data
    manifest.yaml, config.yaml, code/

Usage:
    python Revisions/DFK785/unmix.py [--config path]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Bleedthrough"))
import classify as cl  # noqa: E402
import coefficients as co  # noqa: E402
from common import (COLORS, STR_COLUMNS, apply_axis_standards, calibrate, dye_colors, load_config,  # noqa: E402
                    make_run_dir, rel_to_repo, resolve_run, robust_sigma, save_pdf, setup_logging, write_manifest)

# --- Constants ---
SNAPSHOT = ["405", "488", "515"]
COFACTOR = 5.0                      # asinh display scale, as in Revisions/Bleedthrough step 2
TICK_Z = [0, 10, 100, 1000]
FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 4.8
POINT_SIZE = 1.5
POINT_ALPHA = 0.4


def load_vesicles(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    run = resolve_run(cfg, "vesicles", cfg["unmixing"]["vesicles_run"])
    ves = pd.read_csv(run / "vesicles.csv", dtype=STR_COLUMNS)
    blanks = pd.read_csv(run / "blanks.csv", dtype=STR_COLUMNS)
    return ves, blanks, run


def estimate_signatures(ves: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 1 of Revisions/Bleedthrough on the single-label slides, restricted to the unmixing channels."""
    uc = cfg["unmixing"]
    sub = {"dyes": cfg["dyes"], "channels": {c: c for c in uc["channels"]},
           "bleedthrough": {k: uc[k] for k in ("min_home_snr", "exclude_crowded", "exclude_nonlinear")}}
    sel = {d: co.select(ves, d, v["home_channel"], sub["bleedthrough"]) for d, v in cfg["dyes"].items()}
    for d, s in sel.items():
        if s.empty:
            raise ValueError(f"No single-label {d} vesicles pass the signature selection")
    pf = pd.concat([co.estimate_per_fov(sel[d], d, cfg["dyes"][d]["home_channel"], co.target_channels(sub, d))
                    for d in sel], ignore_index=True)
    return co.summarize(pf, sel), pf


def label_vesicles(ves: pd.DataFrame, cal: pd.DataFrame, sig: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    uc = cfg["unmixing"]
    ch, dyes = uc["channels"], list(cfg["dyes"])
    sd_wide = cal.pivot_table(index=["slide", "fov"], columns="channel", values="noise_sd")[ch]
    sd = ves[["slide", "fov"]].merge(sd_wide, left_on=["slide", "fov"], right_index=True, how="left")[ch].to_numpy()
    flux = ves[[f"F_{c}" for c in ch]].to_numpy()
    home_idx = [ch.index(cfg["dyes"][d]["home_channel"]) for d in dyes]
    t, labels = cl.label_objects_unmixing(flux, sd, sig, dyes, home_idx, uc["presence_snr"])
    out = ves.copy()
    for j, d in enumerate(dyes):
        out[f"t_{d}"] = t[:, j]
    out["label"] = labels
    return out


def check_515(lab: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """515 against the home channel per label on the slides that have 515."""
    rows = []
    for (slide, label), g in lab[lab["F_515"].notna()].groupby(["slide", "label"]):
        row = {"slide": slide, "label": label, "n": len(g), "frac_z515_ge_5": float((g["z_515"] >= 5).mean())}
        for d, v in cfg["dyes"].items():
            h = v["home_channel"]
            bright = g[g[f"z_{h}"] >= cfg["unmixing"]["min_home_snr"]]
            row[f"median_515_over_{h}"] = float(np.median(bright["F_515"] / bright[f"F_{h}"])) if len(bright) else np.nan
            row[f"n_for_ratio_{h}"] = len(bright)
        rows.append(row)
    return pd.DataFrame(rows)


def _ticks(ax, lim):
    t = np.arcsinh(np.array(TICK_Z, float) / COFACTOR)
    keep = (t >= lim[0]) & (t <= lim[1])
    ax.set_xticks(t[keep], [f"{z:g}" for z in np.array(TICK_Z)[keep]])
    ax.set_yticks(t[keep], [f"{z:g}" for z in np.array(TICK_Z)[keep]])
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)


def make_figure(lab: pd.DataFrame, cfg: dict, out: Path) -> None:
    colors = dye_colors(cfg) | {cl.DUAL: COLORS["green"], cl.NONE: COLORS["black"]}
    order = list(cfg["dyes"]) + [cl.DUAL, cl.NONE]
    slides = list(cfg["slides"])
    u = {c: np.arcsinh(lab[f"z_{c}"] / COFACTOR) for c in SNAPSHOT}
    lim = (min(np.nanmin(u[c]) for c in SNAPSHOT) - 0.2, max(np.nanmax(u[c]) for c in SNAPSHOT) + 0.2)
    fig, axs = plt.subplots(2, 3, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    panels = [(s, "488", "405") for s in slides] + [(s, "488", "515") for s in slides if cfg["slides"][s]["dye"] == "mix"]
    for ax, (slide, cx, cy), label in zip(axs.ravel(), panels, "abcdef"):
        g = lab["slide"] == slide
        for lb in order:
            m = g & (lab["label"] == lb)
            ax.scatter(u[cx][m], u[cy][m], s=POINT_SIZE, color=colors[lb], alpha=POINT_ALPHA, linewidths=0,
                       rasterized=True)
        _ticks(ax, lim)
        ax.set_xlabel(f"Flux, {cx} channel (noise SDs)")
        ax.set_ylabel(f"Flux, {cy} channel (noise SDs)")
        ax.set_title(f"{slide} ({cfg['slides'][slide]['dye']})", color="black", fontsize=7)
        ax.text(-0.3, 1.04, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
        apply_axis_standards(ax)
        lab.loc[g, ["slide", "fov", "vesicle_id", f"z_{cx}", f"z_{cy}", "label"]].to_csv(out / f"panel_{label}.csv",
                                                                                       index=False)
    for ax in axs.ravel()[len(panels):]:
        ax.set_axis_off()
    handles = [Line2D([], [], ls="", marker="o", color=colors[l], ms=3, mew=0) for l in order]
    fig.legend(handles, order, loc="lower center", ncol=len(order), frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.14, hspace=0.6, wspace=0.5)
    save_pdf(fig, out / "unmixing.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    ves, blanks, ves_run = load_vesicles(cfg)
    ves, cal = calibrate(ves, blanks, SNAPSHOT)

    sig_summary, sig_fov = estimate_signatures(ves, cfg)
    sig = cl.signatures(cfg, sig_summary, cfg["unmixing"]["channels"])
    logging.info("Signatures (flux per unit of home-channel flux):\n" + sig.to_string())
    lab = label_vesicles(ves, cal, sig, cfg)
    counts = pd.crosstab(lab["slide"], lab["label"]).reindex(columns=list(cfg["dyes"]) + [cl.DUAL, cl.NONE],
                                                              fill_value=0)
    chk = check_515(lab, cfg)

    run = make_run_dir(cfg, "unmixing")
    keep = ["slide", "dye", "fov", "acq_order", "vesicle_id", "y_px", "x_px", "detected_by", "detected_in",
            "nn_dist_px", "crowded", "nonlinear", "file_640"]
    keep += [f"{p}_{c}" for c in SNAPSHOT for p in ("F", "z")] + [f"t_{d}" for d in cfg["dyes"]] + ["label"]
    lab[keep].to_csv(run / "labels.csv", index=False)
    sig_summary.to_csv(run / "signatures.csv", index=False)
    sig_fov.to_csv(run / "signatures_per_fov.csv", index=False)
    counts.to_csv(run / "label_counts.csv")
    chk.to_csv(run / "check_515.csv", index=False)
    cal.to_csv(run / "calibration.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        logging.info("Signature estimates:\n" + sig_summary.to_string(index=False))
        logging.info("Labels per slide:\n" + counts.to_string())
        logging.info("515 check:\n" + chk.to_string(index=False))
    make_figure(lab, cfg, run)
    write_manifest(run, cfg, Path(__file__), [ves_run / "vesicles.csv", ves_run / "blanks.csv"],
                   extra={"vesicles_run": rel_to_repo(ves_run)})


if __name__ == "__main__":
    main()
