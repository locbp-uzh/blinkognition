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
cross-check (estimator B). B weights vesicles by F_h^2, so it follows a few bright
or unusual objects; the verification of 2026-09-24 (README) found that A and B
disagree on these data because of real sample heterogeneity, not the pipeline.

Calibration (from the empty-aperture control in segmentation/blanks.csv):
    F_c = flux_c - median(blank flux_c)          per slide, FOV and channel (zero point)
    z_c = F_c / robust_sd(blank flux_c)          per slide, FOV and channel (noise)

Selection per dye: vesicles on slides of that dye with z_home >=
bleedthrough.min_home_snr, and optionally not crowded and not nonlinear. This
drops objects without the slide's dye (they have no home signal), so the free odd
objects do not enter the estimate. FOVs in exclude_fovs are dropped even if the
segmentation run still contains them.

Input: the segmentation run named by bleedthrough.segmentation_run (or --seg-run).

Outputs (Results/Revisions/Bleedthrough/coefficients/run_NNN/):
    coefficients.csv           one row per dye x channel
    coefficients_per_fov.csv   one row per dye x channel x FOV
    calibration.csv            zero point and noise per FOV x channel
    bleedthrough.pdf           figure
    panel_<x>.csv, panel_<x>_fov.csv, panel_<x>_bins.csv, panel_<x>_summary.csv   its source data
    caption.txt                caption draft; display_notes.txt for what the figure omits
    manifest.yaml, config.yaml, code/

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
import yaml
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from common import (REPO_ROOT, STR_COLUMNS, apply_axis_standards, dye_colors, excluded_fovs, fov_key,
                    load_config, make_run_dir, rel_to_repo, robust_sigma, save_pdf, setup_logging,
                    write_manifest)

# --- Constants ---
FIG_WIDTH_IN = 7.0                  # within Nature double column (180 mm) including the save pad
FIG_HEIGHT_IN = 5.6
RATIO_YLIM_PERCENTILES = (2, 98)    # display range of the ratio axis (points outside are counted)
RATIO_YLIM_PAD = 0.15               # fractional padding of that range
ANNOTATION_HEADROOM = 0.35          # empty range above the data for the text, as a fraction of the data range
POINT_SIZE = 2.0                    # scatter marker area (pt^2)
POINT_ALPHA = 0.35
FOV_POINT_SIZE = 10.0
MARKER_EDGE = 0.8
STRIP_Y_FOV, STRIP_Y_LS, STRIP_JITTER = 0.4, -0.45, 0.2
PANEL_LABEL_SIZE = 8
ZERO_LINE = {"color": "black", "lw": 0.5, "alpha": 0.4}


# =============================================================================
# Data
# =============================================================================


def segmentation_run(cfg: dict, override: Path | None) -> Path:
    """The segmentation run to analyze: --seg-run if given, else bleedthrough.segmentation_run."""
    run = Path(override) if override else REPO_ROOT / cfg["output_root"] / "segmentation" / cfg["bleedthrough"]["segmentation_run"]
    if not (run / "vesicles.csv").exists():
        raise FileNotFoundError(f"No vesicles.csv in {run}")
    return run.resolve()


def load_segmentation(seg_run: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """vesicles, blanks and the run's manifest, minus FOVs excluded in the current config."""
    ves = pd.read_csv(seg_run / "vesicles.csv", dtype=STR_COLUMNS)
    blanks = pd.read_csv(seg_run / "blanks.csv", dtype=STR_COLUMNS)
    with open(seg_run / "manifest.yaml") as f:
        manifest = yaml.safe_load(f)
    excluded = excluded_fovs(cfg)
    for name, df in (("vesicles", ves), ("blanks", blanks)):
        key = df["slide"] + "/" + df["fov"]
        drop = key.isin(excluded)
        if drop.any():
            logging.info(f"Dropping {int(drop.sum())} {name} from excluded FOVs {sorted(set(key[drop]))}")
        df.drop(index=df.index[drop], inplace=True)
    return ves.reset_index(drop=True), blanks.reset_index(drop=True), manifest


def calibrate(ves: pd.DataFrame, blanks: pd.DataFrame, channels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add zero-point-corrected flux F_<ch> and z_<ch> = F / blank robust SD, per slide, FOV and channel."""
    rows = []
    for (slide, fov), g in blanks.groupby(["slide", "fov"]):
        for ch in channels:
            b = g[f"flux_{ch}"].to_numpy()
            rows.append({"slide": slide, "fov": fov, "channel": ch, "n_blanks": len(b),
                         "zero_point": float(np.median(b)), "noise_sd": robust_sigma(b)})
    cal = pd.DataFrame(rows)
    have = set(zip(cal["slide"], cal["fov"])) if len(cal) else set()
    missing = set(zip(ves["slide"], ves["fov"])) - have
    if missing:
        raise ValueError(f"No blanks for FOVs {sorted(fov_key(*m) for m in missing)}")

    out = ves.copy()
    for ch in channels:
        c = cal[cal["channel"] == ch][["slide", "fov", "zero_point", "noise_sd"]]
        merged = out[["slide", "fov"]].merge(c, on=["slide", "fov"], how="left", validate="many_to_one")
        out[f"F_{ch}"] = out[f"flux_{ch}"].to_numpy() - merged["zero_point"].to_numpy()
        out[f"z_{ch}"] = out[f"F_{ch}"].to_numpy() / merged["noise_sd"].to_numpy()
    return out, cal


def select(ves: pd.DataFrame, dye: str, home: str, bt: dict) -> pd.DataFrame:
    m = (ves["dye"] == dye) & (ves[f"z_{home}"] >= bt["min_home_snr"])
    if bt["exclude_crowded"]:
        m &= ~ves["crowded"]
    if bt["exclude_nonlinear"]:
        m &= ~ves["nonlinear"]
    return ves[m]


def target_channels(cfg: dict, dye: str) -> list[str]:
    """All channels except the dye's home channel, in wavelength order with 640 (protein) last."""
    home = cfg["dyes"][dye]["home_channel"]
    return sorted((c for c in cfg["channels"] if c != home), key=lambda c: (c == "640", int(c)))


# =============================================================================
# Estimation
# =============================================================================


def estimate_per_fov(sel: pd.DataFrame, dye: str, home: str, targets: list[str]) -> pd.DataFrame:
    """Estimator A (median ratio) and B (LS slope through the origin) per FOV and target channel."""
    rows = []
    for (slide, fov), g in sel.groupby(["slide", "fov"]):
        x = g[f"F_{home}"].to_numpy()
        for ch in targets:
            y = g[f"F_{ch}"].to_numpy()
            rows.append({"dye": dye, "home": home, "channel": ch, "slide": slide, "fov": fov,
                         "n_vesicles": len(g),
                         "k_median_ratio": float(np.median(y / x)),
                         "k_ls_origin": float(np.sum(x * y) / np.sum(x * x))})
    return pd.DataFrame(rows)


def _sd(s: pd.Series) -> float:
    return float(s.std(ddof=1)) if len(s) > 1 else np.nan


def summarize(per_fov: pd.DataFrame, sel_by_dye: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (dye, home, ch), g in per_fov.groupby(["dye", "home", "channel"], sort=False):
        a, b = g["k_median_ratio"], g["k_ls_origin"]
        sel = sel_by_dye[dye]
        x, y = sel[f"F_{home}"].to_numpy(), sel[f"F_{ch}"].to_numpy()
        sd = _sd(a)
        rows.append({"dye": dye, "home": home, "channel": ch, "n_fov": len(g),
                     "n_vesicles": int(g["n_vesicles"].sum()),
                     "k_mean": float(a.mean()), "k_sd": sd,
                     "k_ls_mean": float(b.mean()), "k_ls_sd": _sd(b),
                     "k_pooled_median": float(np.median(y / x)),
                     # undefined (NA) with a single FOV, since there is no FOV-to-FOV SD
                     "a_vs_b_disagree": bool(abs(a.mean() - b.mean()) > sd) if np.isfinite(sd) else pd.NA})
    return pd.DataFrame(rows)


# =============================================================================
# Figure
# =============================================================================


def _decimals(sd: float) -> int:
    """Decimal places that give the SD two significant digits."""
    if not np.isfinite(sd) or sd == 0:
        return 3
    return max(0, 1 - int(np.floor(np.log10(abs(sd)))))


def _pm(mean: float, sd: float) -> str:
    dec = _decimals(sd)
    return f"{mean:.{dec}f} ± {sd:.{dec}f}" if np.isfinite(sd) else f"{mean:.{dec}f} (1 FOV)"


def _binned_medians(x: np.ndarray, ratio: np.ndarray, n_bins: int, min_n: int) -> pd.DataFrame:
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        m = idx == i
        rows.append({"edge_lo": edges[i], "edge_hi": edges[i + 1], "center": np.sqrt(edges[i] * edges[i + 1]),
                     "n": int(m.sum()), "median_ratio_fraction": float(np.median(ratio[m])) if m.sum() >= min_n else np.nan})
    return pd.DataFrame(rows)


def make_figure(sel_by_dye: dict, per_fov: pd.DataFrame, summary: pd.DataFrame, cfg: dict,
                out: Path, rng: np.random.Generator) -> tuple[list[str], dict[str, int]]:
    """Ratio vs home brightness per dye x target channel, with per-FOV k strips.

    Returns (display notes, number of ratios outside the drawn range per panel).
    """
    bt, colors = cfg["bleedthrough"], dye_colors(cfg)
    dyes = list(cfg["dyes"])
    targets = {d: target_channels(cfg, d) for d in dyes}
    ncol = max(len(t) for t in targets.values())

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    gs = fig.add_gridspec(2 * len(dyes), ncol, height_ratios=[1, 0.32] * len(dyes), hspace=0.9, wspace=0.45)
    labels = iter(string.ascii_lowercase)
    notes, outside = [], {}
    err_handle = None
    kilo = FuncFormatter(lambda v, _: f"{v / 1000:g}")

    for r, dye in enumerate(dyes):
        home, sel, color = cfg["dyes"][dye]["home_channel"], sel_by_dye[dye], colors[dye]
        for c, ch in enumerate(targets[dye]):
            label = next(labels)
            ax = fig.add_subplot(gs[2 * r, c])
            axs = fig.add_subplot(gs[2 * r + 1, c])
            s = summary[(summary.dye == dye) & (summary.channel == ch)].iloc[0]
            pf = per_fov[(per_fov.dye == dye) & (per_fov.channel == ch)]

            x = sel[f"F_{home}"].to_numpy()
            ratio = sel[f"F_{ch}"].to_numpy() / x
            bins = _binned_medians(x, ratio, bt["n_brightness_bins"], bt["min_per_bin"])
            med = bins["median_ratio_fraction"].to_numpy()

            lo, hi = np.percentile(ratio, RATIO_YLIM_PERCENTILES)
            lo, hi = min(lo, np.nanmin(med, initial=lo)), max(hi, np.nanmax(med, initial=hi))
            pad = RATIO_YLIM_PAD * (hi - lo)
            lo, hi = lo - pad, hi + pad
            shown = (ratio >= lo) & (ratio <= hi)      # the band above hi stays empty for the text
            outside[label] = int((~shown).sum())

            ax.scatter(x[shown], 100 * ratio[shown], s=POINT_SIZE, color=color, alpha=POINT_ALPHA,
                       linewidths=0, rasterized=True)
            ax.plot(bins["center"], 100 * med, "o", mfc="white", mec="black", ms=3.5, mew=MARKER_EDGE)
            ax.axhline(100 * s.k_mean, color="black", ls="--", lw=0.9)
            ax.axhline(0, **ZERO_LINE)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(kilo)
            ax.set_ylim(100 * lo, 100 * (hi + ANNOTATION_HEADROOM * (hi - lo)))
            ax.set_xlabel(f"Flux, {home} channel (×1000 counts)")
            ax.set_ylabel(f"Flux {ch} / flux {home} (%)")
            ax.set_title(f"{dye} into {ch}", color="black")
            ax.text(0.03, 0.97, f"F{ch} = k · F{home},  k = {_pm(100 * s.k_mean, 100 * s.k_sd)} %\n"
                                f"LS cross-check: {_pm(100 * s.k_ls_mean, 100 * s.k_ls_sd)} %",
                    transform=ax.transAxes, va="top", ha="left", fontsize=6, color="black")
            ax.text(-0.28, 1.12, label, transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
                    fontweight="bold", va="bottom", color="black")
            apply_axis_standards(ax)

            jitter = rng.uniform(-STRIP_JITTER, STRIP_JITTER, len(pf))
            axs.scatter(100 * pf["k_median_ratio"], STRIP_Y_FOV + jitter, s=FOV_POINT_SIZE, color=color,
                        edgecolors="black", linewidths=0.4, zorder=3)
            sd_a = 100 * s.k_sd if np.isfinite(s.k_sd) else 0.0
            err_handle = axs.errorbar(100 * s.k_mean, STRIP_Y_FOV, xerr=sd_a, fmt="none", ecolor="black",
                                      capsize=2.5, lw=0.9, zorder=4)
            sd_b = 100 * s.k_ls_sd if np.isfinite(s.k_ls_sd) else 0.0
            axs.errorbar(100 * s.k_ls_mean, STRIP_Y_LS, xerr=sd_b, fmt="D", mfc="white", mec="black",
                         ms=3.5, mew=MARKER_EDGE, ecolor="black", capsize=2.5, lw=0.9, zorder=5)
            axs.set_ylim(-1, 1)
            axs.set_yticks([])
            axs.set_xlabel("k per FOV (%)")
            apply_axis_standards(axs)
            axs.spines["left"].set_visible(False)
            xl = 100 * np.array([pf["k_median_ratio"].min(), pf["k_median_ratio"].max(),
                                 s.k_mean - np.nan_to_num(s.k_sd), s.k_mean + np.nan_to_num(s.k_sd),
                                 s.k_ls_mean - np.nan_to_num(s.k_ls_sd), s.k_ls_mean + np.nan_to_num(s.k_ls_sd)])
            span = max(xl.max() - xl.min(), 1e-3)
            axs.set_xlim(xl.min() - 0.15 * span, xl.max() + 0.15 * span)

            pd.DataFrame({"slide": sel["slide"], "fov": sel["fov"], "vesicle_id": sel["vesicle_id"],
                          f"F_{home}": x, f"F_{ch}": sel[f"F_{ch}"], "ratio_fraction": ratio,
                          "drawn": shown}).to_csv(out / f"panel_{label}.csv", index=False)
            pf[["slide", "fov", "n_vesicles", "k_median_ratio", "k_ls_origin"]].to_csv(
                out / f"panel_{label}_fov.csv", index=False)
            bins.to_csv(out / f"panel_{label}_bins.csv", index=False)
            pd.DataFrame([{"dye": dye, "home": home, "channel": ch, "n_fov": s.n_fov, "n_vesicles": s.n_vesicles,
                           "k_mean_pct": 100 * s.k_mean, "k_sd_pct": 100 * s.k_sd,
                           "k_ls_mean_pct": 100 * s.k_ls_mean, "k_ls_sd_pct": 100 * s.k_ls_sd}]).to_csv(
                out / f"panel_{label}_summary.csv", index=False)
            notes.append(f"{label}: {dye} into {ch}, {len(x)} vesicles from {len(pf)} FOVs, "
                         f"{outside[label]} ratios outside the drawn range, bins with n < {bt['min_per_bin']} "
                         f"not drawn (bin n = {bins['n'].tolist()})")

    vesicle_handles = [Line2D([], [], ls="", marker="o", color=colors[d], ms=np.sqrt(POINT_SIZE),
                              alpha=POINT_ALPHA, mew=0) for d in dyes]
    fov_handles = tuple(Line2D([], [], ls="", marker="o", color=colors[d], mec="black", mew=0.4,
                               ms=np.sqrt(FOV_POINT_SIZE)) for d in dyes)
    handles = [*vesicle_handles,
               Line2D([], [], ls="", marker="o", mfc="white", mec="black", ms=3.5, mew=MARKER_EDGE),
               Line2D([], [], ls="--", color="black", lw=0.9),
               Line2D([], [], **ZERO_LINE),
               fov_handles,
               err_handle,
               Line2D([], [], ls="", marker="D", mfc="white", mec="black", ms=3.5, mew=MARKER_EDGE)]
    names = [f"{d} vesicle" for d in dyes] + ["binned median", "k (mean of FOVs)", "zero", "k per FOV",
                                               "mean ± SD", "LS mean ± SD"]
    fig.legend(handles, names, loc="lower center", ncol=int(np.ceil(len(handles) / 2)), frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, -0.02), handler_map={tuple: HandlerTuple(ndivide=None)})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.14)
    save_pdf(fig, out / "bleedthrough.pdf")
    plt.close(fig)
    return notes, outside


def write_caption(out: Path, cfg: dict, ves: pd.DataFrame, summary: pd.DataFrame, outside: dict,
                  notes: list[str], pixel_um: float) -> None:
    """Caption built from the config and the data actually drawn."""
    bt = cfg["bleedthrough"]
    dyes = list(cfg["dyes"])
    labels = iter(string.ascii_lowercase)
    panel_text, counts = [], []
    for d in dyes:
        tg = target_channels(cfg, d)
        ls = [next(labels) for _ in tg]
        chans = ", ".join(tg[:-1]) + f" and {tg[-1]}" if len(tg) > 1 else tg[0]
        panel_text.append(f"({ls[0]}–{ls[-1]}) {d} vesicles: flux in the {chans} channels relative to "
                          f"the {cfg['dyes'][d]['home_channel']} channel.")
        s = summary[summary.dye == d].iloc[0]
        n_det = int((ves["dye"] == d).sum())
        counts.append(f"{d}: n = {int(s.n_vesicles)} of {n_det} detected vesicles from {int(s.n_fov)} FOVs")
    excluded = excluded_fovs(cfg)
    ex_by_dye = {}
    for key in excluded:
        slide = key.split("/")[0]
        dye = cfg["slides"].get(slide, {}).get("dye", slide)
        ex_by_dye[dye] = ex_by_dye.get(dye, 0) + 1
    ex_text = "; ".join(f"{n} {d} FOV{'s' if n > 1 else ''} excluded as a preliminary measurement"
                        for d, n in ex_by_dye.items())
    proteins = ", ".join(sorted({s["protein"] for s in cfg["slides"].values() if s.get("protein")}))

    criteria = [f"home-channel flux at least {bt['min_home_snr']} times the robust SD of flux in empty apertures"]
    if bt["exclude_crowded"]:
        criteria.append(f"no other vesicle within {cfg['photometry']['crowding_radius_px'] * pixel_um:.2f} µm")
    if bt["exclude_nonlinear"]:
        criteria.append("no pixel in the nonlinear range of the camera")
    n_out = ", ".join(f"{v} ({k})" for k, v in outside.items())

    text = (
        "Spectral bleed-through of the vesicle labels. " + " ".join(panel_text) + " "
        "Channels are named by excitation wavelength (nm). "
        "Top: per-vesicle flux ratio against home-channel flux; open circles, medians of log-spaced "
        f"brightness bins with at least {bt['min_per_bin']} vesicles (no error bars); dashed line, the model "
        "F_c = k·F_home, with k the mean over fields of view (FOVs) of the per-FOV median of per-vesicle ratios; "
        "gray line, zero. Bottom: k per FOV (circles) with mean ± SD across FOVs, and the mean ± SD of per-FOV "
        "least-squares (LS) slopes through the origin (diamond) as a cross-check. "
        f"Both slides also carry an HMSiR-labeled protein ({proteins}) that is detected in the 640 channel, so "
        "the 640 panels include protein signal. "
        "[Vesicle composition, dye mol%, buffer and temperature to be added.] "
        "Vesicles with " + ", ".join(criteria) + "; " + "; ".join(counts)
        + (f"; {ex_text}" if ex_text else "") + ". FOVs are repeated measurements of one slide per label. "
        f"Ratios outside the axis range: {n_out}. "
        "Center: mean of per-FOV medians; spread: SD across FOVs; individual FOVs shown. "
        "No statistical comparisons were performed.\n"
    )
    (out / "caption.txt").write_text(text)
    (out / "display_notes.txt").write_text("\n".join(notes) + "\n")


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
    bt["segmentation_run"] = rel_to_repo(seg_run)       # recorded in the effective config
    ves, blanks, seg_manifest = load_segmentation(seg_run, cfg)
    channels = list(cfg["channels"])
    ves, cal = calibrate(ves, blanks, channels)

    per_fov, sel_by_dye = [], {}
    for dye, dcfg in cfg["dyes"].items():
        home = dcfg["home_channel"]
        sel = select(ves, dye, home, bt)
        if sel.empty:
            raise ValueError(f"No {dye} vesicles pass the selection")
        sel_by_dye[dye] = sel
        n_all = int((ves["dye"] == dye).sum())
        logging.info(f"{dye}: {len(sel)} of {n_all} vesicles selected (z_{home} >= {bt['min_home_snr']}, "
                     f"crowded excluded: {bt['exclude_crowded']}, nonlinear excluded: {bt['exclude_nonlinear']})")
        per_fov.append(estimate_per_fov(sel, dye, home, target_channels(cfg, dye)))
    per_fov = pd.concat(per_fov, ignore_index=True)
    summary = summarize(per_fov, sel_by_dye)

    run = make_run_dir(cfg, "coefficients")
    cal.to_csv(run / "calibration.csv", index=False)
    per_fov.to_csv(run / "coefficients_per_fov.csv", index=False)
    summary.to_csv(run / "coefficients.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.5f}".format):
        logging.info("Coefficients (fractions, not %):\n" + summary.to_string(index=False))
    flagged = summary[summary["a_vs_b_disagree"].fillna(False).astype(bool)]
    if len(flagged):
        logging.info("A and B differ by more than the FOV SD for: "
                     + ", ".join(f"{r.dye}->{r.channel}" for r in flagged.itertuples())
                     + " (expected on these data, see README)")

    notes, outside = make_figure(sel_by_dye, per_fov, summary, cfg, run, rng)
    write_caption(run, cfg, ves, summary, outside, notes, seg_manifest["pixel_um"])
    write_manifest(run, cfg, Path(__file__), [seg_run / "vesicles.csv", seg_run / "blanks.csv"],
                   extra={"segmentation_run": rel_to_repo(seg_run),
                          "segmentation_config_overrides": seg_manifest.get("config_overrides", [])})


if __name__ == "__main__":
    main()
