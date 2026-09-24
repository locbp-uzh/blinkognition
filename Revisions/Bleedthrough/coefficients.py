#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Analysis step 1: bleed-through coefficients.

For a vesicle carrying only dye D, the signal in channel c is proportional to its
signal in D's home channel h: F_c = k * F_h, with the same k for every vesicle of
that dye whatever its size. Per FOV, k is the median of the per-vesicle ratios
F_c / F_h (estimator A); the FOVs are summarized as mean +- SD (uncertainty A,
the FOV being the level at which measurements are repeated). The least-squares
slope through the origin, sum(F_c F_h) / sum(F_h^2), is computed alongside as a
cross-check (estimator B): if A and B disagree by more than the FOV-to-FOV SD,
outliers are influencing B.

Calibration (from the empty-aperture control in segmentation/blanks.csv):
    F_c = flux_c - median(blank flux_c)          per FOV and channel (zero point)
    z_c = F_c / robust_sd(blank flux_c)          per FOV and channel (noise)

Selection per slide: z_home >= bleedthrough.min_home_snr, and optionally not
crowded and not saturated. This drops objects without the slide's dye (they have
no home signal), so the odd objects do not enter the estimate.

Input: the segmentation run named by bleedthrough.segmentation_run (or --seg-run).

Outputs (Results/Revisions/Bleedthrough/coefficients/run_NNN/):
    coefficients.csv           one row per dye x channel
    coefficients_per_fov.csv   one row per dye x channel x FOV
    calibration.csv            zero point and noise per FOV x channel
    bleedthrough.pdf           figure; panel_<x>.csv and panel_<x>_fov.csv are its source data
    caption.txt                draft caption
    manifest.yaml, config.yaml, coefficients.py

Usage:
    python Revisions/Bleedthrough/coefficients.py [--seg-run DIR] [--config path] [--set key=value ...]
"""

from __future__ import annotations

import argparse
import logging
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from common import (REPO_ROOT, SLIDE_COLORS, apply_axis_standards, load_config, make_run_dir,
                    robust_sigma, save_pdf, setup_logging, write_manifest)

# --- Constants ---
FIG_WIDTH_IN = 7.1                  # Nature double column (180 mm)
FIG_HEIGHT_IN = 5.6
RATIO_YLIM_PERCENTILES = (2, 98)    # display range of the ratio axis (points outside are counted)
RATIO_YLIM_PAD = 0.15               # fractional padding of that range
ANNOTATION_HEADROOM = 0.35          # extra empty range above the data, as a fraction of the data range
POINT_SIZE = 2.0
POINT_ALPHA = 0.35
STRIP_JITTER = 0.25
PANEL_LABEL_SIZE = 8


# =============================================================================
# Data
# =============================================================================


def segmentation_run(cfg: dict, override: Path | None) -> Path:
    """The segmentation run to analyze: --seg-run if given, else bleedthrough.segmentation_run."""
    run = Path(override) if override else REPO_ROOT / cfg["output_root"] / "segmentation" / cfg["bleedthrough"]["segmentation_run"]
    if not (run / "vesicles.csv").exists():
        raise FileNotFoundError(f"No vesicles.csv in {run}")
    return run.resolve()


def load_segmentation(seg_run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ves = pd.read_csv(seg_run / "vesicles.csv", dtype={"detected_by": str})
    blanks = pd.read_csv(seg_run / "blanks.csv")
    return ves, blanks


def calibrate(ves: pd.DataFrame, blanks: pd.DataFrame, channels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add zero-point-corrected flux F_<ch> and z_<ch> = F / blank robust SD, per FOV and channel."""
    rows = []
    for (slide, fov), g in blanks.groupby(["slide", "fov"]):
        for ch in channels:
            b = g[f"flux_{ch}"].to_numpy()
            rows.append({"slide": slide, "fov": fov, "channel": ch, "n_blanks": len(b),
                         "zero_point": float(np.median(b)), "noise_sd": robust_sigma(b)})
    cal = pd.DataFrame(rows)
    missing = set(ves["fov"]) - set(cal["fov"])
    if missing:
        raise ValueError(f"No blanks for FOVs {sorted(missing)}")

    out = ves.copy()
    for ch in channels:
        c = cal[cal["channel"] == ch].set_index("fov")
        zp = out["fov"].map(c["zero_point"])
        sd = out["fov"].map(c["noise_sd"])
        out[f"F_{ch}"] = out[f"flux_{ch}"] - zp
        out[f"z_{ch}"] = out[f"F_{ch}"] / sd
    return out, cal


def select(ves: pd.DataFrame, slide: str, home: str, bt: dict) -> pd.DataFrame:
    m = (ves["slide"] == slide) & (ves[f"z_{home}"] >= bt["min_home_snr"])
    if bt["exclude_crowded"]:
        m &= ~ves["crowded"]
    if bt["exclude_saturated"]:
        m &= ~ves["saturated"]
    return ves[m]


# =============================================================================
# Estimation
# =============================================================================


def estimate_per_fov(sel: pd.DataFrame, dye: str, home: str, targets: list[str]) -> pd.DataFrame:
    """Estimator A (median ratio) and B (LS slope through the origin) per FOV and target channel."""
    rows = []
    for fov, g in sel.groupby("fov"):
        x = g[f"F_{home}"].to_numpy()
        for ch in targets:
            y = g[f"F_{ch}"].to_numpy()
            rows.append({"dye": dye, "home": home, "channel": ch, "fov": fov, "n_vesicles": len(g),
                         "k_median_ratio": float(np.median(y / x)),
                         "k_ls_origin": float(np.sum(x * y) / np.sum(x * x))})
    return pd.DataFrame(rows)


def summarize(per_fov: pd.DataFrame, sel_by_dye: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (dye, home, ch), g in per_fov.groupby(["dye", "home", "channel"], sort=False):
        a, b = g["k_median_ratio"], g["k_ls_origin"]
        sel = sel_by_dye[dye]
        x, y = sel[f"F_{home}"].to_numpy(), sel[f"F_{ch}"].to_numpy()
        sd = float(a.std(ddof=1)) if len(a) > 1 else np.nan
        rows.append({"dye": dye, "home": home, "channel": ch, "n_fov": len(g),
                     "n_vesicles": int(g["n_vesicles"].sum()),
                     "k_mean": float(a.mean()), "k_sd": sd,
                     "k_ls_mean": float(b.mean()), "k_ls_sd": float(b.std(ddof=1)) if len(b) > 1 else np.nan,
                     "k_pooled_median": float(np.median(y / x)),
                     "a_vs_b_disagree": bool(abs(a.mean() - b.mean()) > sd)})
    return pd.DataFrame(rows)


# =============================================================================
# Figure
# =============================================================================


def _decimals(sd: float) -> int:
    """Decimal places that give the SD two significant digits."""
    if not np.isfinite(sd) or sd == 0:
        return 3
    return max(0, 1 - int(np.floor(np.log10(abs(sd)))))


def _fmt(v: float, decimals: int) -> str:
    return f"{v:.{decimals}f}"


def target_channels(cfg: dict, dye: str) -> list[str]:
    """All channels except the dye's home channel, in wavelength order with 640 (protein) last."""
    home = cfg["bleedthrough"]["home_channel"][dye]
    return sorted((c for c in cfg["channels"] if c != home), key=lambda c: (c == "640", int(c)))


def make_figure(sel_by_dye: dict, per_fov: pd.DataFrame, summary: pd.DataFrame, cfg: dict,
                out: Path, rng: np.random.Generator) -> list[str]:
    """Ratio vs home brightness per dye x target channel, with per-FOV k strips. Returns caption notes."""
    bt = cfg["bleedthrough"]
    dyes = list(bt["home_channel"])
    targets = {d: target_channels(cfg, d) for d in dyes}
    ncol = max(len(t) for t in targets.values())

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    gs = fig.add_gridspec(2 * len(dyes), ncol, height_ratios=[1, 0.32] * len(dyes), hspace=0.9, wspace=0.45)
    labels = iter(string.ascii_lowercase)
    notes = []

    for r, dye in enumerate(dyes):
        home, sel, color = bt["home_channel"][dye], sel_by_dye[dye], SLIDE_COLORS[dye]
        for c, ch in enumerate(targets[dye]):
            label = next(labels)
            ax = fig.add_subplot(gs[2 * r, c])
            axs = fig.add_subplot(gs[2 * r + 1, c])
            s = summary[(summary.dye == dye) & (summary.channel == ch)].iloc[0]
            pf = per_fov[(per_fov.dye == dye) & (per_fov.channel == ch)]

            x = sel[f"F_{home}"].to_numpy()
            ratio = sel[f"F_{ch}"].to_numpy() / x
            lo, hi = np.percentile(ratio, RATIO_YLIM_PERCENTILES)
            pad = RATIO_YLIM_PAD * (hi - lo)
            lo, hi = lo - pad, hi + pad
            n_out = int(((ratio < lo) | (ratio > hi)).sum())
            hi_axis = hi + ANNOTATION_HEADROOM * (hi - lo)     # empty band above the data for the text

            ax.scatter(x, 100 * ratio, s=POINT_SIZE, color=color, alpha=POINT_ALPHA, linewidths=0,
                       rasterized=True)
            edges = np.logspace(np.log10(x.min()), np.log10(x.max()), bt["n_brightness_bins"] + 1)
            idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
            centers = np.sqrt(edges[:-1] * edges[1:])
            med = [np.median(ratio[idx == i]) if (idx == i).sum() >= 5 else np.nan for i in range(len(centers))]
            ax.plot(centers, 100 * np.asarray(med), "o", mfc="white", mec="black", ms=3.5, mew=0.8)
            ax.axhline(100 * s.k_mean, color="black", ls="--", lw=0.9)
            ax.axhline(0, color="black", lw=0.5, alpha=0.4)
            ax.set_xscale("log")
            ax.set_ylim(100 * lo, 100 * hi_axis)
            ax.set_xlabel(f"Flux, {home} channel (counts)")
            ax.set_ylabel(f"Flux {ch} / flux {home} (%)")
            ax.set_title(f"{dye} into {ch}", color="black")
            dec = _decimals(100 * s.k_sd)
            ax.text(0.03, 0.97, f"$F_{{{ch}}} = k \\cdot F_{{{home}}}$,  "
                                f"k = {_fmt(100 * s.k_mean, dec)} ± {_fmt(100 * s.k_sd, dec)} %\n"
                                f"LS cross-check: {_fmt(100 * s.k_ls_mean, dec)} %",
                    transform=ax.transAxes, va="top", ha="left", fontsize=6, color="black")
            ax.text(-0.28, 1.12, label, transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
                    fontweight="bold", va="bottom", color="black")
            apply_axis_standards(ax)

            jitter = rng.uniform(-STRIP_JITTER, STRIP_JITTER, len(pf))
            axs.scatter(100 * pf["k_median_ratio"], jitter, s=10, color=color, edgecolors="black",
                        linewidths=0.4, zorder=3)
            axs.errorbar(100 * s.k_mean, 0, xerr=100 * s.k_sd, fmt="|", color="black", ms=8,
                         capsize=2.5, lw=0.9, zorder=4)
            axs.plot(100 * s.k_ls_mean, 0, "D", mfc="white", mec="black", ms=3.5, mew=0.8, zorder=5)
            axs.set_ylim(-1, 1)
            axs.set_yticks([])
            axs.spines["left"].set_visible(False)
            axs.set_xlabel("k per FOV (%)")
            apply_axis_standards(axs)
            axs.spines["left"].set_visible(False)
            xl = np.array([pf["k_median_ratio"].min(), pf["k_median_ratio"].max(), s.k_ls_mean,
                           s.k_mean - s.k_sd, s.k_mean + s.k_sd]) * 100
            span = max(xl.max() - xl.min(), 1e-3)
            axs.set_xlim(xl.min() - 0.25 * span, xl.max() + 0.25 * span)

            pd.DataFrame({"fov": sel["fov"], "vesicle_id": sel["vesicle_id"], f"F_{home}": x,
                          f"F_{ch}": sel[f"F_{ch}"], "ratio": ratio}).to_csv(out / f"panel_{label}.csv", index=False)
            pf[["fov", "n_vesicles", "k_median_ratio", "k_ls_origin"]].to_csv(out / f"panel_{label}_fov.csv",
                                                                                index=False)
            notes.append(f"{label}: {dye} into {ch}, {len(x)} vesicles from {len(pf)} FOVs, "
                         f"{n_out} ratios outside the displayed range")

    handles = [Line2D([], [], ls="", marker="o", color=SLIDE_COLORS[d], ms=3) for d in dyes]
    fov_handles = tuple(Line2D([], [], ls="", marker="o", color=SLIDE_COLORS[d], mec="black", mew=0.4, ms=3.5)
                        for d in dyes)
    handles = [*handles,
               Line2D([], [], ls="", marker="o", mfc="white", mec="black", ms=3.5),
               Line2D([], [], ls="--", color="black", lw=0.9),
               fov_handles,
               Line2D([], [], ls="", marker="|", color="black", ms=8),
               Line2D([], [], ls="", marker="D", mfc="white", mec="black", ms=3.5)]
    names = [f"{d} vesicle" for d in dyes] + ["binned median", "k (mean of FOVs)", "k per FOV",
                                               "mean ± SD", "LS cross-check"]
    fig.legend(handles, names, loc="lower center", ncol=len(handles), frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, -0.01), handler_map={tuple: HandlerTuple(ndivide=None)})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12)
    save_pdf(fig, out / "bleedthrough.pdf")
    plt.close(fig)
    return notes


def write_caption(out: Path, summary: pd.DataFrame, notes: list[str], bt: dict) -> None:
    n = summary.groupby("dye")[["n_fov", "n_vesicles"]].first()
    parts = ", ".join(f"{d}: n = {int(r.n_vesicles)} vesicles from {int(r.n_fov)} FOVs" for d, r in n.iterrows())
    text = (
        "Spectral bleed-through of the vesicle labels. "
        "(a-c) ATTO390 vesicles, signal in the 488, 515 and 640 channels relative to the 405 channel; "
        "(d-f) ATTO525 vesicles, signal in the 405, 488 and 640 channels relative to the 515 channel. "
        "Top: per-vesicle flux ratio against home-channel flux, with medians in log-spaced brightness bins "
        "and the coefficient k (dashed). Bottom: k per FOV (median of per-vesicle ratios), mean ± SD across "
        "FOVs, and the least-squares slope through the origin as a cross-check. "
        "[Vesicle composition, dye mol%, buffer and temperature to be added.] "
        f"Vesicles with home-channel signal at least {bt['min_home_snr']} times the empty-aperture noise, "
        "not crowded and not saturated; "
        f"{parts}; FOVs are repeated measurements of one slide per label. "
        "Center: mean of per-FOV medians; spread: SD across FOVs; individual FOVs shown. "
        "No statistical comparisons were performed.\n\n"
        "Display notes (not for the caption): " + "; ".join(notes) + "\n"
    )
    (out / "caption.txt").write_text(text)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-run", type=Path, default=None,
                    help="segmentation run directory (default: bleedthrough.segmentation_run)")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    bt = cfg["bleedthrough"]
    rng = np.random.default_rng(42)

    seg_run = segmentation_run(cfg, args.seg_run)
    ves, blanks = load_segmentation(seg_run)
    channels = list(cfg["channels"])
    ves, cal = calibrate(ves, blanks, channels)

    per_fov, sel_by_dye = [], {}
    for dye, home in bt["home_channel"].items():
        sel = select(ves, dye, home, bt)
        sel_by_dye[dye] = sel
        n_all = int((ves["slide"] == dye).sum())
        logging.info(f"{dye}: {len(sel)} of {n_all} vesicles selected (z_{home} >= {bt['min_home_snr']}, "
                     f"crowded excluded: {bt['exclude_crowded']}, saturated excluded: {bt['exclude_saturated']})")
        per_fov.append(estimate_per_fov(sel, dye, home, target_channels(cfg, dye)))
    per_fov = pd.concat(per_fov, ignore_index=True)
    summary = summarize(per_fov, sel_by_dye)

    run = make_run_dir(cfg, "coefficients")
    cal.to_csv(run / "calibration.csv", index=False)
    per_fov.to_csv(run / "coefficients_per_fov.csv", index=False)
    summary.to_csv(run / "coefficients.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.5f}".format):
        logging.info("Coefficients (fractions, not %):\n" + summary.to_string(index=False))
    if summary["a_vs_b_disagree"].any():
        logging.warning("Estimators A and B disagree by more than the FOV SD for: "
                        + ", ".join(f"{r.dye}->{r.channel}" for r in summary[summary.a_vs_b_disagree].itertuples()))

    notes = make_figure(sel_by_dye, per_fov, summary, cfg, run, rng)
    write_caption(run, summary, notes, bt)
    write_manifest(run, cfg, Path(__file__), [seg_run / "vesicles.csv", seg_run / "blanks.csv"],
                   extra={"segmentation_run": str(seg_run.relative_to(REPO_ROOT)),
                          "segmentation_config_overrides": _seg_overrides(seg_run)})


def _seg_overrides(seg_run: Path) -> list:
    import yaml

    with open(seg_run / "manifest.yaml") as f:
        return yaml.safe_load(f).get("config_overrides", [])


if __name__ == "__main__":
    main()
