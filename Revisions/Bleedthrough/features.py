#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Analysis step 2: the features the classification model sees.

Per object and feature channel c (405, 488, 515):
    z_c = (flux_c - blank median) / blank robust SD        per slide, FOV and channel
    u_c = asinh(z_c / cofactor)                             cofactor in noise SDs (5)

asinh is linear near zero and logarithmic for bright signals, so objects whose
signal in a channel is within the noise (often negative after background
subtraction) keep their measurement, while the size variation of bright vesicles
becomes a shift along a diagonal. Every detected object is kept, with its flags
(crowded, nonlinear), since a real mix has to classify everything. 640 (protein)
is not a feature.

The figure shows the in-silico mix (both slides pooled, colored by true dye), the
empty apertures, and the curves expected for a pure dye from the step 1
coefficients, as a check that the feature space looks as the physics predicts.

Outputs (Results/Revisions/Bleedthrough/features/run_NNN/):
    features.csv            one row per object: ids, true dye, flags, z_<ch>, u_<ch>
    features_blanks.csv     the same for the empty apertures
    feature_space.pdf       figure; panel_<x>.csv and expected_curves.csv are its source data
    caption.txt
    manifest.yaml, config.yaml, code/

Usage:
    python Revisions/Bleedthrough/features.py [--config path] [--set key=value ...]
"""

from __future__ import annotations

import argparse
import itertools
import logging
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import (COLORS, apply_axis_standards, calibrate, dye_colors, load_config, load_segmentation,
                    make_run_dir, rel_to_repo, resolve_run, save_pdf, setup_logging, write_manifest)

# --- Constants ---
ID_COLUMNS = ["slide", "dye", "fov", "vesicle_id", "detected_by", "crowded", "nonlinear"]
FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 2.9
POINT_SIZE = 2.0
POINT_ALPHA = 0.4
BLANK_COLOR = COLORS["black"]
BLANK_ALPHA = 0.15
TICK_Z = [-10, 0, 10, 100, 1000, 10000]     # tick positions in noise SDs
CURVE_Z = np.logspace(0, 4, 200)            # home-channel z along the expected curves
PANEL_LABEL_SIZE = 8


def to_features(df: pd.DataFrame, channels: list[str], cofactor: float) -> pd.DataFrame:
    """u_c = asinh(z_c / cofactor) for each feature channel (z_c from common.calibrate)."""
    out = df.copy()
    for ch in channels:
        out[f"u_{ch}"] = np.arcsinh(out[f"z_{ch}"] / cofactor)
    return out


def expected_curves(cfg: dict, coef: pd.DataFrame, cal: pd.DataFrame, ves: pd.DataFrame) -> pd.DataFrame:
    """Pure-dye curves in u space: z_c = k_c * z_home * sd_home / sd_c (median noise SD of that dye's FOVs)."""
    fc = cfg["features"]
    rows = []
    for dye, d in cfg["dyes"].items():
        home = d["home_channel"]
        fovs = ves.loc[ves["dye"] == dye, ["slide", "fov"]].drop_duplicates()
        sd = cal.merge(fovs, on=["slide", "fov"]).groupby("channel")["noise_sd"].median()
        k = coef[coef["dye"] == dye].set_index("channel")["k_mean"]
        z_max = ves.loc[ves["dye"] == dye, f"z_{home}"].max()
        for z_home in CURVE_Z[CURVE_Z <= z_max]:
            row = {"dye": dye, "z_home": z_home}
            for ch in fc["channels"]:
                z = z_home if ch == home else k[ch] * z_home * sd[home] / sd[ch]
                row[f"u_{ch}"] = np.arcsinh(z / fc["asinh_cofactor"])
            rows.append(row)
    return pd.DataFrame(rows)


def _z_axis(ax, axis: str, cofactor: float, lim: tuple[float, float]) -> None:
    """Ticks labeled in noise SDs at their asinh positions; only those inside lim (else the axis grows)."""
    z = np.array(TICK_Z, dtype=float)
    ticks = np.arcsinh(z / cofactor)
    keep = (ticks >= lim[0]) & (ticks <= lim[1])
    labels = [f"{v:g}" for v in z[keep]]
    if axis == "x":
        ax.set_xticks(ticks[keep], labels)
        ax.set_xlim(*lim)
    else:
        ax.set_yticks(ticks[keep], labels)
        ax.set_ylim(*lim)


def make_figure(feat: pd.DataFrame, blanks: pd.DataFrame, curves: pd.DataFrame, cfg: dict, out: Path) -> None:
    fc, colors = cfg["features"], dye_colors(cfg)
    pairs = list(itertools.combinations(fc["channels"], 2))     # (405, 488), (405, 515), (488, 515)
    pairs = [(b, a) for a, b in pairs]                           # longer wavelength on x
    fig, axs = plt.subplots(1, len(pairs), figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ucols = [f"u_{c}" for c in fc["channels"]]
    lim = (min(feat[ucols].min().min(), blanks[ucols].min().min()) - 0.2, feat[ucols].max().max() + 0.2)
    for ax, (cx, cy), label in zip(axs, pairs, string.ascii_lowercase):
        ax.scatter(blanks[f"u_{cx}"], blanks[f"u_{cy}"], s=POINT_SIZE, color=BLANK_COLOR, alpha=BLANK_ALPHA,
                   linewidths=0, rasterized=True)
        for dye in cfg["dyes"]:
            g = feat[feat["dye"] == dye]
            ax.scatter(g[f"u_{cx}"], g[f"u_{cy}"], s=POINT_SIZE, color=colors[dye], alpha=POINT_ALPHA,
                       linewidths=0, rasterized=True)
            cv = curves[curves["dye"] == dye]
            ax.plot(cv[f"u_{cx}"], cv[f"u_{cy}"], ls="--", lw=0.9, color=colors[dye])
        # same scale on every axis, so band positions are comparable
        _z_axis(ax, "x", fc["asinh_cofactor"], lim)
        _z_axis(ax, "y", fc["asinh_cofactor"], lim)
        ax.set_xlabel(f"Flux, {cx} channel (noise SDs)")
        ax.set_ylabel(f"Flux, {cy} channel (noise SDs)")
        ax.text(-0.3, 1.04, label, transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="bold",
                va="bottom", color="black")
        apply_axis_standards(ax)
        cols = ["slide", "dye", "fov", "vesicle_id", f"u_{cx}", f"u_{cy}"]
        feat[cols].to_csv(out / f"panel_{label}.csv", index=False)

    handles = [Line2D([], [], ls="", marker="o", color=colors[d], ms=np.sqrt(POINT_SIZE), alpha=POINT_ALPHA, mew=0)
               for d in cfg["dyes"]]
    handles += [Line2D([], [], ls="", marker="o", color=BLANK_COLOR, ms=np.sqrt(POINT_SIZE), alpha=BLANK_ALPHA, mew=0)]
    handles += [Line2D([], [], ls="--", lw=0.9, color=colors[d]) for d in cfg["dyes"]]
    names = [f"{d} slide object" for d in cfg["dyes"]] + ["empty aperture"] + [f"expected pure {d}" for d in cfg["dyes"]]
    fig.legend(handles, names, loc="lower center", ncol=len(handles), frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.27, wspace=0.45)
    save_pdf(fig, out / "feature_space.pdf")
    plt.close(fig)
    curves.to_csv(out / "expected_curves.csv", index=False)


def write_caption(out: Path, cfg: dict, feat: pd.DataFrame, blanks: pd.DataFrame) -> None:
    fc = cfg["features"]
    counts = "; ".join(f"{d}: n = {int((feat['dye'] == d).sum())} objects from "
                       f"{feat.loc[feat['dye'] == d, ['slide', 'fov']].drop_duplicates().shape[0]} FOVs"
                       for d in cfg["dyes"])
    text = (
        "Feature space of the classification model. Objects from both single-label slides pooled (in silico "
        "mix) and colored by the label of their slide; gray, empty apertures; dashed lines, flux expected for "
        "a vesicle carrying only one label, from the bleed-through coefficients. Axes: background-corrected "
        "flux in units of the empty-aperture noise SD of that channel and field of view (FOV), on an asinh "
        f"scale with cofactor {fc['asinh_cofactor']:g} (linear within a few noise SDs, logarithmic above). "
        "Channels are named by excitation wavelength (nm). "
        "[Vesicle composition, dye mol%, buffer and temperature to be added.] "
        f"{counts}; {len(blanks)} empty apertures. All detected objects are shown. "
        "FOVs are repeated measurements of one slide per label. No statistical comparisons were performed.\n"
    )
    (out / "caption.txt").write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    fc = cfg["features"]

    seg_run = resolve_run(cfg, "segmentation", fc["segmentation_run"])
    coef_run = resolve_run(cfg, "coefficients", fc["coefficients_run"])
    ves, blanks, _ = load_segmentation(seg_run, cfg)
    ves, cal = calibrate(ves, blanks, fc["channels"])
    blanks, _ = calibrate(blanks, blanks, fc["channels"])
    feat = to_features(ves, fc["channels"], fc["asinh_cofactor"])
    bfeat = to_features(blanks, fc["channels"], fc["asinh_cofactor"])
    coef = pd.read_csv(coef_run / "coefficients.csv", dtype={"channel": str, "home": str})

    zcols = [f"z_{c}" for c in fc["channels"]]
    ucols = [f"u_{c}" for c in fc["channels"]]
    run = make_run_dir(cfg, "features")
    feat[ID_COLUMNS + zcols + ucols].to_csv(run / "features.csv", index=False)
    bfeat[["slide", "dye", "fov", "blank_id"] + zcols + ucols].to_csv(run / "features_blanks.csv", index=False)
    for dye in cfg["dyes"]:
        g = feat[feat["dye"] == dye]
        logging.info(f"{dye}: {len(g)} objects ({int(g['crowded'].sum())} crowded, "
                     f"{int(g['nonlinear'].sum())} nonlinear)")

    curves = expected_curves(cfg, coef, cal, ves)
    make_figure(feat, bfeat, curves, cfg, run)
    write_caption(run, cfg, feat, bfeat)
    write_manifest(run, cfg, Path(__file__), [seg_run / "vesicles.csv", seg_run / "blanks.csv",
                                             coef_run / "coefficients.csv"],
                   extra={"segmentation_run": rel_to_repo(seg_run), "coefficients_run": rel_to_repo(coef_run)})


if __name__ == "__main__":
    main()
