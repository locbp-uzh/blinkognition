#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Protein occupancy of the vesicles of DFK785 from the 640 (HMSiR) movies (goal 2).

1. Trace per vesicle and per blank: in every frame of the 6000-frame 640 movie,
   the sum over the aperture (r = 3 px) minus n_aperture x the median of the
   annulus (5-8 px), at the vesicle position plus the slide's registration offset.
   The per-frame local background removes the FOV-wide drift of the 640 baseline.
2. Registration: on a few FOVs per slide, clearly positive vesicles (score A >=
   registration.min_score) give the offset of the 640 signal from the vesicle
   position as the intensity-weighted centroid of their ON frames (> on_sigma
   robust SDs). A vesicle's position comes from its detection channel (405 for
   most ATTO390 vesicles, 488 or 515 for ATTO520 ones) and the chromatic shift to
   640 differs between them, so the offset is the median per detection channel,
   pooled over slides, applied if larger than apply_above_px. Blanks are random
   positions and need none.
3. Scores. A: (max - median) / MAD of the trace (the repo's robust outlier score,
   Extraction/utils.robust_background_filter; pure noise gives ~5.8 for 6000
   frames). B: number of ON frames from Extraction/utils.gmm_classify_frames with
   the repo's posterior threshold and minimum separation; run where A >=
   b_prescreen_score plus a random audit sample of the rest (B is 1.9 s per trace).
4. Thresholds: for each score, the smallest value that at most blank_pass_fraction
   of all blank traces (pooled over slides) reach. A vesicle is positive if it
   reaches it.
5. Chance correction: blanks also catch blinks that do not belong to a vesicle
   (protein on the glass, noise). With the slide's blank positive rate p_b and a
   vesicle positive rate p_v, the occupancy is (p_v - p_b) / (1 - p_b), assuming
   such blinks land independently of the vesicles.
6. Occupancy per FOV and label (ATTO390 = HT7, ATTO520 = SNAP; dual and no label
   reported too), crowded vesicles excluded, summarized as mean +- SD across FOVs.

Outputs (Results/Revisions/DFK785/occupancy/run_NNN/):
    registration.csv          per-vesicle offsets used for registration; registration_summary.csv
    scores_vesicles.csv       one row per vesicle: ids, label, flags, brightness, score A, n_on (B), positive A/B
    scores_blanks.csv         the same for blanks
    thresholds.csv
    occupancy_per_fov.csv     per slide x FOV x label: n, positives, raw and corrected occupancy (A and B)
    occupancy_summary.csv     per slide x label: mean +- SD across FOVs, pooled values, blank rates
    occupancy_by_brightness.csv, sensitivity.csv, b_audit.csv
    occupancy.pdf, maxproj_qc.pdf   figures; panel_<x>.csv are their source data
    traces/<slide>_<fov>.npz  corrected traces (float32) of vesicles and blanks
    manifest.yaml, config.yaml, code/

Usage:
    python Revisions/DFK785/occupancy.py [--config path]
"""

from __future__ import annotations

import argparse
import logging
import string
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nd2
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Bleedthrough"))
from common import (COLORS, STR_COLUMNS, apply_axis_standards, dye_colors, load_config, load_scm,  # noqa: E402
                    make_run_dir, rel_to_repo, resolve_run, save_pdf, setup_logging, write_manifest)
from utils import gmm_classify_frames  # noqa: E402  (Extraction/utils.py, on sys.path via common)

# --- Constants ---
DUAL, NONE = "dual", "no label"
FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 5.2
MAD_FLOOR = 1e-6


# =============================================================================
# Traces
# =============================================================================


def extract_traces(movie: np.ndarray, pos: np.ndarray, offset: np.ndarray, r_ap: float, r_in: float,
                   r_out: float) -> np.ndarray:
    """(n, T) background-corrected aperture traces at pos + offset (offset (2,) or one row per position)."""
    h = int(np.ceil(r_out)) + 1
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    out = np.empty((len(pos), movie.shape[0]), dtype=np.float32)
    for i, (y, x) in enumerate(pos + offset):
        iy, ix = int(round(y)), int(round(x))
        win = movie[:, iy - h:iy + h + 1, ix - h:ix + h + 1].astype(np.float32)
        d = np.hypot(yy + iy - y, xx + ix - x)
        ap, ann = d <= r_ap, (d >= r_in) & (d <= r_out)
        out[i] = win[:, ap].sum(axis=1) - ap.sum() * np.median(win[:, ann], axis=1)
    return out


def score_a(traces: np.ndarray) -> np.ndarray:
    med = np.median(traces, axis=1, keepdims=True)
    mad = np.median(np.abs(traces - med), axis=1)
    return (traces.max(axis=1) - med[:, 0]) / np.maximum(mad, MAD_FLOOR)


def on_frame_offset(movie: np.ndarray, pos: np.ndarray, trace: np.ndarray, rc: dict) -> np.ndarray:
    """Intensity-weighted centroid (dy, dx) of the 640 signal in ON frames, relative to pos."""
    med = np.median(trace)
    sig = 1.4826 * np.median(np.abs(trace - med))
    on = trace > med + rc["on_sigma"] * sig
    w = rc["window_half_px"]
    y, x = pos
    iy, ix = int(round(y)), int(round(x))
    win = movie[on][:, iy - w:iy + w + 1, ix - w:ix + w + 1].astype(np.float64)
    off = movie[~on][::10, iy - w:iy + w + 1, ix - w:ix + w + 1].astype(np.float64).mean(axis=0)
    img = np.clip(win.mean(axis=0) - off, 0, None)
    yy, xx = np.mgrid[-w:w + 1, -w:w + 1]
    s = img.sum()
    return np.array([(img * (yy + iy - y)).sum() / s, (img * (xx + ix - x)).sum() / s]) if s > 0 else np.full(2, np.nan)


def load_movie(cfg: dict, rel: str) -> np.ndarray:
    return nd2.imread(str(Path(cfg["input_root"]) / rel))


def registration_fov(cfg: dict, slide: str, fov: str, rel: str, pos: np.ndarray, ids: np.ndarray,
                     det_by: np.ndarray) -> list[dict]:
    oc = cfg["occupancy"]
    movie = load_movie(cfg, rel)
    tr = extract_traces(movie, pos, np.zeros(2), oc["aperture_radius_px"], *oc["annulus_px"])
    a = score_a(tr)
    rows = []
    for i in np.flatnonzero(a >= oc["registration"]["min_score"]):
        dy, dx = on_frame_offset(movie, pos[i], tr[i], oc["registration"])
        rows.append({"slide": slide, "fov": fov, "vesicle_id": int(ids[i]), "detected_by": det_by[i],
                     "score_a": float(a[i]), "dy": dy, "dx": dx})
    return rows


def traces_fov(cfg: dict, slide: str, fov: str, rel: str, vpos: np.ndarray, bpos: np.ndarray, voff: np.ndarray,
               out: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vesicle traces at their position plus the offset of their detection channel; blanks at their position."""
    oc = cfg["occupancy"]
    movie = load_movie(cfg, rel)
    tv = extract_traces(movie, vpos, voff, oc["aperture_radius_px"], *oc["annulus_px"])
    tb = extract_traces(movie, bpos, np.zeros(2), oc["aperture_radius_px"], *oc["annulus_px"])
    np.savez_compressed(out / f"{slide}_{fov}.npz", vesicles=tv, blanks=tb)
    return score_a(tv), score_a(tb)


def n_on_frames(trace: np.ndarray, oc: dict) -> int:
    _, _, signal, _ = gmm_classify_frames(trace.astype(np.float64), proba_threshold=oc["gmm_proba_threshold"],
                                          min_separation=oc["gmm_min_separation"], return_separation=True)
    return int(signal.sum())


def b_for_fov(cfg: dict, path: Path, run_v: np.ndarray, run_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oc = cfg["occupancy"]
    z = np.load(path)
    nv = np.full(len(run_v), np.nan)
    nb = np.full(len(run_b), np.nan)
    for i in np.flatnonzero(run_v):
        nv[i] = n_on_frames(z["vesicles"][i], oc)
    for i in np.flatnonzero(run_b):
        nb[i] = n_on_frames(z["blanks"][i], oc)
    return nv, nb


# =============================================================================
# Occupancy
# =============================================================================


def threshold_for(blank_values: np.ndarray, pass_fraction: float, integer: bool = False) -> float:
    """Smallest threshold t with P(blank >= t) <= pass_fraction."""
    v = np.sort(blank_values[np.isfinite(blank_values)])
    cand = np.unique(v) if not integer else np.arange(1, int(v.max()) + 2)
    for t in cand:
        if (v >= t).mean() <= pass_fraction:
            return float(t)
    return float(v.max() + 1)


def corrected(p_v, p_b):
    return (p_v - p_b) / (1 - p_b)


def report_groups(cfg: dict) -> list[tuple[str, str]]:
    """(slide, label) pairs that answer the question: a single-label slide's own dye, both dyes on a mixed slide."""
    out = []
    for slide, s in cfg["slides"].items():
        out += [(slide, d) for d in cfg["dyes"]] if s["dye"] == "mix" else [(slide, s["dye"])]
    return out


def occupancy_tables(sv: pd.DataFrame, sb: pd.DataFrame, labels: list[str], methods: dict,
                     exclude_crowded: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-FOV and per-slide occupancy for each method (column of positives)."""
    use = sv[~sv["crowded"]] if exclude_crowded else sv
    rows = []
    for (slide, fov), g in use.groupby(["slide", "fov"]):
        for lab in labels:
            gl = g[g["label"] == lab]
            if gl.empty:
                continue
            row = {"slide": slide, "fov": fov, "label": lab, "n": len(gl)}
            for m, col in methods.items():
                pb = sb.loc[sb["slide"] == slide, col].mean()
                pv = gl[col].mean()
                row |= {f"positive_{m}": int(gl[col].sum()), f"p_v_{m}": pv, f"p_b_{m}": pb,
                        f"occupancy_{m}": corrected(pv, pb)}
            rows.append(row)
    per_fov = pd.DataFrame(rows)
    rows = []
    for (slide, lab), g in per_fov.groupby(["slide", "label"]):
        row = {"slide": slide, "label": lab, "n_fov": len(g), "n_vesicles": int(g["n"].sum())}
        for m, col in methods.items():
            pb = sb.loc[sb["slide"] == slide, col].mean()
            pooled = g[f"positive_{m}"].sum() / g["n"].sum()
            row |= {f"occupancy_{m}_mean": g[f"occupancy_{m}"].mean(),
                    f"occupancy_{m}_sd": g[f"occupancy_{m}"].std(ddof=1) if len(g) > 1 else np.nan,
                    f"p_v_{m}_pooled": pooled, f"p_b_{m}": pb, f"occupancy_{m}_pooled": corrected(pooled, pb),
                    f"n_blanks": int((sb["slide"] == slide).sum())}
        rows.append(row)
    return per_fov, pd.DataFrame(rows)


# =============================================================================
# Figures
# =============================================================================


SLIDE_MARKERS = ["o", "s", "^", "D"]


def make_figure(sv, sb, per_fov, summary, reg, sens, bright, thr, cfg, out: Path) -> None:
    colors = dye_colors(cfg)
    dyes = list(cfg["dyes"])
    slides = list(cfg["slides"])
    groups = report_groups(cfg)
    marker = dict(zip(slides, SLIDE_MARKERS))
    style = {s: ("-" if cfg["slides"][s]["dye"] != "mix" else "--") for s in slides}
    fig, axs = plt.subplots(2, 3, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    rng = np.random.default_rng(cfg["occupancy"]["seed"])

    # a: score A distributions
    ax = axs[0, 0]
    bins = np.logspace(np.log10(3), np.log10(max(sv["score_a"].max(), 100)), 50)
    ax.hist(sb["score_a"], bins=bins, histtype="step", color="black", density=True, lw=0.9)
    own = sv[[(s, l) in set(groups) for s, l in zip(sv["slide"], sv["label"])] & ~sv["crowded"]]
    for d in dyes:
        ax.hist(own.loc[own["label"] == d, "score_a"], bins=bins, histtype="step", color=colors[d], density=True, lw=0.9)
    ax.axvline(thr["A"], color="black", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Score A, (max - median) / MAD")
    ax.set_ylabel("Density")
    pd.concat([sb[["slide", "score_a"]].assign(group="blank"),
               sv.loc[~sv["crowded"], ["slide", "label", "score_a"]].rename(columns={"label": "group"})]).to_csv(
        out / "panel_a.csv", index=False)

    # b: occupancy per slide and label, A filled / B open, FOV points
    ax = axs[0, 1]
    xt, xl = [], []
    k = 0
    for slide, d in groups:
            g = per_fov[(per_fov.slide == slide) & (per_fov.label == d)]
            if g.empty:
                continue
            for j, (m, mk, fill) in enumerate((("A", "o", True), ("B", "o", False))):
                v = 100 * g[f"occupancy_{m}"].to_numpy()
                xs = k + (j - 0.5) * 0.35 + rng.uniform(-0.05, 0.05, len(v))
                ax.scatter(xs, v, s=7, marker=mk, color=colors[d] if fill else "white", edgecolors=colors[d],
                           linewidths=0.6, zorder=3)
                ax.errorbar(k + (j - 0.5) * 0.35, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, fmt="_",
                            color="black", ms=7, capsize=2, lw=0.8, zorder=4)
            xt.append(k); xl.append(f"{slide.replace('slide', 'slide ')}\n{d}")
            k += 1
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_xticks(xt, xl, fontsize=5, rotation=45, ha="right")
    ax.set_ylabel("Vesicles with protein (%)")
    per_fov.to_csv(out / "panel_b.csv", index=False)

    # c: occupancy vs brightness tertile (A, pooled per slide)
    ax = axs[0, 2]
    for (slide, d), g in bright.groupby(["slide", "label"]):
        if (slide, d) not in groups:
            continue
        ax.plot(g["tertile_index"], 100 * g["occupancy_A"], ls=style[slide], marker=marker[slide], ms=2.5, lw=0.9,
                color=colors[d])
    ax.set_xticks([0, 1, 2], ["dim", "middle", "bright"])
    ax.set_xlabel("Vesicle brightness (label, tertile)")
    ax.set_ylabel("Vesicles with protein (%)")
    bright.to_csv(out / "panel_c.csv", index=False)

    # d: registration offsets
    ax = axs[1, 0]
    for ch, c in zip(cfg["detection_channels"], [COLORS["blue"], COLORS["green"], COLORS["orange"]]):
        g = reg[reg.detected_by == ch]
        ax.scatter(g["dx"], g["dy"], s=4, color=c, alpha=0.6, linewidths=0, label=f"detected in {ch}")
    ax.axhline(0, color="black", lw=0.4, alpha=0.4); ax.axvline(0, color="black", lw=0.4, alpha=0.4)
    ax.set_xlabel("640 offset x (px)"); ax.set_ylabel("640 offset y (px)")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.legend(frameon=False, fontsize=5, markerscale=1.5)
    reg.to_csv(out / "panel_d.csv", index=False)

    # e: sensitivity to the A threshold (corrected occupancy, pooled)
    ax = axs[1, 1]
    for (slide, d), g in sens.groupby(["slide", "label"]):
        if (slide, d) not in groups:
            continue
        ax.plot(g["threshold"], 100 * g["occupancy_pooled"], ls=style[slide], marker=marker[slide], ms=2.5, lw=0.9,
                color=colors[d])
    ax.axvline(thr["A"], color="black", ls=":", lw=0.8)
    ax.set_xlabel("Score A threshold")
    ax.set_ylabel("Vesicles with protein (%)")
    sens.to_csv(out / "panel_e.csv", index=False)

    # f: blank positive rate vs A threshold
    ax = axs[1, 2]
    for slide in slides:
        g = sens[(sens.slide == slide)].drop_duplicates("threshold")
        ax.plot(g["threshold"], 100 * g["p_b"], ls=style[slide], marker=marker[slide], ms=2.5, lw=0.9, color="black")
    ax.axvline(thr["A"], color="black", ls=":", lw=0.8)
    ax.set_xlabel("Score A threshold"); ax.set_ylabel("Blanks positive (%)")
    for a_, lab in zip(axs.ravel(), string.ascii_lowercase):
        apply_axis_standards(a_)
        a_.text(-0.3, 1.05, lab, transform=a_.transAxes, fontsize=8, fontweight="bold", va="bottom")
    h = [Line2D([], [], color="black", lw=0.9)] + [Line2D([], [], color=colors[d], lw=0.9) for d in dyes]
    h += [Line2D([], [], ls="", marker="o", color="gray", ms=3),
          Line2D([], [], ls="", marker="o", mfc="white", mec="gray", ms=3),
          Line2D([], [], ls="", marker="_", color="black", ms=7)]
    h += [Line2D([], [], ls=style[s], marker=marker[s], color="gray", ms=3, lw=0.9) for s in slides]
    n = ["blanks", *[f"{d} vesicles" for d in dyes], "score A (per FOV)", "score B (per FOV)", "mean ± SD"]
    n += [f"{s.replace('slide', 'slide ')} ({cfg['slides'][s]['dye']})" for s in slides]
    fig.legend(h, n, loc="lower center", ncol=5, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.2, hspace=0.9, wspace=0.55)
    save_pdf(fig, out / "occupancy.pdf")
    plt.close(fig)


def maxproj_qc(cfg: dict, sv: pd.DataFrame, offsets: dict, out: Path) -> None:
    """Option C as a visual check: 640 max projection with vesicles (filled = positive by A)."""
    colors = dye_colors(cfg) | {DUAL: COLORS["green"], NONE: COLORS["black"]}
    slides = list(cfg["slides"])
    cmap = load_scm("grayC")
    fig, axs = plt.subplots(1, len(slides), figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN / len(slides) + 0.5))
    for ax, slide in zip(axs, slides):
        g = sv[(sv.slide == slide) & (sv.acq_order == 1)]
        mx = load_movie(cfg, g["file_640"].iloc[0]).max(axis=0)
        lo, hi = np.percentile(mx, (1, 99.8))
        ax.imshow(mx, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        for lab, gl in g.groupby("label"):
            off = np.stack([offsets[c] for c in gl["detected_by"]])
            x, y = gl["x_px"].to_numpy() + off[:, 1], gl["y_px"].to_numpy() + off[:, 0]
            pos = gl["positive_A"].to_numpy()
            ax.scatter(x, y, s=9, facecolors="none", edgecolors=colors[lab], linewidths=0.4)
            ax.scatter(x[pos], y[pos], s=9, color=colors[lab], linewidths=0)
        ax.set_title(f"{slide} {g['fov'].iloc[0]}", fontsize=6, color="black")
        ax.set_axis_off()
    fig.tight_layout()
    save_pdf(fig, out / "maxproj_qc.pdf")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def compute_scores(cfg: dict, run: Path, lab: pd.DataFrame, blanks: pd.DataFrame):
    """Registration, traces, score A for every trace and score B where needed."""
    oc, rc = cfg["occupancy"], cfg["occupancy"]["registration"]
    rng = np.random.default_rng(oc["seed"])
    # FOVs from vesicles AND blanks: a FOV without vesicles still contributes blanks to the thresholds and p_b.
    fovs = pd.concat([lab[["slide", "fov", "file_640"]], blanks[["slide", "fov", "file_640"]]]).drop_duplicates(
        ["slide", "fov"]).reset_index(drop=True)
    (run / "traces").mkdir()
    # --- 1. registration on a subset of FOVs per slide ---
    sub = fovs[fovs.set_index(["slide", "fov"]).index.isin(lab.set_index(["slide", "fov"]).index)]
    sub = sub.groupby("slide", group_keys=False).apply(
        lambda g: g.sample(min(rc["fovs_per_slide"], len(g)), random_state=oc["seed"]))
    jobs = []
    for r in sub.itertuples():
        g = lab[(lab.slide == r.slide) & (lab.fov == r.fov) & ~lab["crowded"]]
        jobs.append((r.slide, r.fov, r.file_640, g[["y_px", "x_px"]].to_numpy(), g["vesicle_id"].to_numpy(),
                     g["detected_by"].to_numpy()))
    reg = pd.DataFrame([row for rows in Parallel(n_jobs=cfg["n_jobs"])(delayed(registration_fov)(cfg, *j) for j in jobs)
                        for row in rows]).dropna(subset=["dy", "dx"])
    agg = dict(n=("dy", "size"), dy_median=("dy", "median"), dx_median=("dx", "median"),
               dy_iqr=("dy", lambda s: s.quantile(.75) - s.quantile(.25)),
               dx_iqr=("dx", lambda s: s.quantile(.75) - s.quantile(.25)))
    reg_sum = reg.groupby("detected_by").agg(**agg).reset_index().assign(slide="all")
    reg_check = reg.groupby(["slide", "detected_by"]).agg(**agg).reset_index()
    offsets = {}
    for r in reg_sum.itertuples():
        o = np.array([r.dy_median, r.dx_median])
        offsets[r.detected_by] = o if np.hypot(*o) > rc["apply_above_px"] else np.zeros(2)
    for ch in cfg["detection_channels"]:
        if ch not in offsets:
            logging.warning(f"No registration vesicles detected by {ch}; using no offset for them")
            offsets[ch] = np.zeros(2)
    reg_sum["applied_dy"] = reg_sum["detected_by"].map(lambda c: offsets[c][0])
    reg_sum["applied_dx"] = reg_sum["detected_by"].map(lambda c: offsets[c][1])
    reg_sum = pd.concat([reg_sum, reg_check], ignore_index=True)
    reg.to_csv(run / "registration.csv", index=False)
    reg_sum.to_csv(run / "registration_summary.csv", index=False)
    logging.info("Registration:\n" + reg_sum.to_string(index=False))

    # --- 2. traces and score A for all FOVs ---
    jobs = []
    for r in fovs.itertuples():
        gv = lab[(lab.slide == r.slide) & (lab.fov == r.fov)]
        gb = blanks[(blanks.slide == r.slide) & (blanks.fov == r.fov)]
        voff = np.stack([offsets[c] for c in gv["detected_by"]]) if len(gv) else np.zeros((0, 2))
        jobs.append((r.slide, r.fov, r.file_640, gv[["y_px", "x_px"]].to_numpy(), gb[["y_px", "x_px"]].to_numpy(),
                     voff, run / "traces"))
    h = int(np.ceil(oc["annulus_px"][1])) + 1
    for slide, fov, rel, vpos, bpos, voff, _ in jobs:
        allpos = np.vstack([vpos + voff, bpos]) if len(vpos) else bpos
        if len(allpos) and (np.round(allpos).min() < h or np.round(allpos).max() > 229 - h):
            raise ValueError(f"{slide} {fov}: a trace window (half-size {h} px) would cross the image edge after "
                             "the registration offset; raise detection.border_px")
    logging.info(f"Extracting 640 traces in {len(jobs)} FOVs")
    res = Parallel(n_jobs=cfg["n_jobs"])(delayed(traces_fov)(cfg, *j) for j in jobs)
    sv_parts, sb_parts = [], []
    for (slide, fov, *_), (av, ab) in zip(jobs, res):
        gv = lab[(lab.slide == slide) & (lab.fov == fov)].copy()
        gb = blanks[(blanks.slide == slide) & (blanks.fov == fov)][["slide", "fov", "blank_id", "y_px", "x_px"]].copy()
        gv["score_a"], gb["score_a"] = av, ab
        sv_parts.append(gv); sb_parts.append(gb)
    sv, sb = pd.concat(sv_parts, ignore_index=True), pd.concat(sb_parts, ignore_index=True)

    # --- 3. score B where A >= prescreen, plus an audit sample ---
    sv["b_run"] = (sv["score_a"] >= oc["b_prescreen_score"]) | (rng.random(len(sv)) < oc["b_audit_fraction"])
    sb["b_run"] = (sb["score_a"] >= oc["b_prescreen_score"]) | (rng.random(len(sb)) < oc["b_audit_fraction"])
    sv["b_audit"] = sv["b_run"] & (sv["score_a"] < oc["b_prescreen_score"])
    sb["b_audit"] = sb["b_run"] & (sb["score_a"] < oc["b_prescreen_score"])
    logging.info(f"Score B on {int(sv.b_run.sum())} vesicle and {int(sb.b_run.sum())} blank traces")
    bjobs = []
    for slide, fov, *_ in jobs:
        mv = ((sv.slide == slide) & (sv.fov == fov)).to_numpy()
        mb = ((sb.slide == slide) & (sb.fov == fov)).to_numpy()
        bjobs.append((mv, mb, run / "traces" / f"{slide}_{fov}.npz"))
    bres = Parallel(n_jobs=cfg["n_jobs"] * 2)(delayed(b_for_fov)(cfg, p, sv.loc[mv, "b_run"].to_numpy(),
                                                                 sb.loc[mb, "b_run"].to_numpy()) for mv, mb, p in bjobs)
    sv["n_on"], sb["n_on"] = np.nan, np.nan
    for (mv, mb, _), (nv, nb) in zip(bjobs, bres):
        sv.loc[mv, "n_on"], sb.loc[mb, "n_on"] = nv, nb
    audit = pd.DataFrame([{"set": name, "n_audited": int(d["b_audit"].sum()),
                           "n_audited_with_on_frames": int((d.loc[d["b_audit"], "n_on"] > 0).sum())}
                          for name, d in (("vesicles", sv), ("blanks", sb))])
    audit.to_csv(run / "b_audit.csv", index=False)
    logging.info("B audit below the prescreen:\n" + audit.to_string(index=False))

    return sv, sb, reg, offsets


def load_scores(run: Path) -> tuple:
    """Scores, registration and offsets of an earlier run (for --rescore)."""
    sv = pd.read_csv(run / "scores_vesicles.csv", dtype=STR_COLUMNS)
    sb = pd.read_csv(run / "scores_blanks.csv", dtype=STR_COLUMNS)
    reg = pd.read_csv(run / "registration.csv", dtype=STR_COLUMNS)
    rs = pd.read_csv(run / "registration_summary.csv", dtype=STR_COLUMNS)
    rs = rs[rs["slide"] == "all"]
    offsets = {r.detected_by: np.array([r.applied_dy, r.applied_dx]) for r in rs.itertuples()}
    for ch in sv["detected_by"].unique():
        offsets.setdefault(ch, np.zeros(2))          # channels without registration vesicles got no offset
    return sv, sb, reg, offsets


SCORING_KEYS = ["unmixing_run", "aperture_radius_px", "annulus_px", "registration", "gmm_proba_threshold",
                "gmm_min_separation", "b_prescreen_score", "b_audit_fraction", "seed"]


def check_rescore_config(cfg: dict, src: Path) -> None:
    """--rescore reuses the source run's scores, so its scoring parameters must equal the current ones."""
    with open(src / "config.yaml") as f:
        old = yaml.safe_load(f)["occupancy"]
    diff = [k for k in SCORING_KEYS if old.get(k) != cfg["occupancy"].get(k)]
    if diff:
        raise ValueError(f"--rescore: scoring parameters differ from {src.name}: {diff}; rerun without --rescore")


def summarize(cfg: dict, run: Path, sv: pd.DataFrame, sb: pd.DataFrame, reg: pd.DataFrame, offsets: dict) -> dict:
    """Thresholds, occupancy tables, sensitivity analyses and figures."""
    oc = cfg["occupancy"]
    # --- 4. thresholds from the blanks ---
    thr = {"A": threshold_for(sb["score_a"].to_numpy(), oc["blank_pass_fraction"]),
           "B": threshold_for(sb["n_on"].fillna(0).to_numpy(), oc["blank_pass_fraction"], integer=True)}
    for d in (sv, sb):
        d["positive_A"] = d["score_a"] >= thr["A"]
        d["positive_B"] = d["n_on"].fillna(0) >= thr["B"]
    pd.DataFrame([{"score": k, "threshold": v, "blank_pass_fraction": oc["blank_pass_fraction"],
                   "blanks_passing": float((sb[f"positive_{k}"]).mean())} for k, v in thr.items()]).to_csv(
        run / "thresholds.csv", index=False)
    logging.info(f"Thresholds: {thr}")

    # --- 5-6. occupancy ---
    labels = list(cfg["dyes"]) + [DUAL, NONE]
    per_fov, summary = occupancy_tables(sv, sb, labels, {"A": "positive_A", "B": "positive_B"})
    per_fov.to_csv(run / "occupancy_per_fov.csv", index=False)
    summary.to_csv(run / "occupancy_summary.csv", index=False)

    rows = []                                        # brightness dependence (A)
    for (slide, d), g in sv[~sv["crowded"] & sv["label"].isin(list(cfg["dyes"]))].groupby(["slide", "label"]):
        h = cfg["dyes"][d]["home_channel"]
        q = pd.qcut(g[f"z_{h}"], 3, labels=False)
        pb = sb.loc[sb.slide == slide, "positive_A"].mean()
        for t in range(3):
            gt = g[q == t]
            rows.append({"slide": slide, "label": d, "tertile_index": t, "n": len(gt),
                         "median_home_z": float(gt[f"z_{h}"].median()), "p_v_A": gt["positive_A"].mean(),
                         "occupancy_A": corrected(gt["positive_A"].mean(), pb)})
    bright = pd.DataFrame(rows)
    bright.to_csv(run / "occupancy_by_brightness.csv", index=False)

    rows = []                                        # sensitivity to the A threshold
    for t in sorted(set(oc["sensitivity_scores"]) | {thr["A"]}):
        for slide in cfg["slides"]:
            pb = (sb.loc[sb.slide == slide, "score_a"] >= t).mean()
            for d in labels:
                g = sv[(sv.slide == slide) & (sv.label == d) & ~sv["crowded"]]
                if g.empty:
                    continue
                pv = (g["score_a"] >= t).mean()
                rows.append({"threshold": t, "slide": slide, "label": d, "n": len(g), "p_v": pv, "p_b": pb,
                             "occupancy_pooled": corrected(pv, pb)})
    sens = pd.DataFrame(rows)
    sens.to_csv(run / "sensitivity.csv", index=False)

    rows = []                                        # sensitivity: crowded vesicles included
    _, with_crowded = occupancy_tables(sv, sb, labels, {"A": "positive_A", "B": "positive_B"}, exclude_crowded=False)
    with_crowded.to_csv(run / "occupancy_summary_including_crowded.csv", index=False)

    keep_v = ["slide", "dye", "fov", "acq_order", "vesicle_id", "label", "detected_by", "crowded", "nonlinear", "y_px", "x_px",
              "z_405", "z_488", "z_515", "score_a", "b_run", "b_audit", "n_on", "positive_A", "positive_B", "file_640"]
    sv[keep_v].to_csv(run / "scores_vesicles.csv", index=False)
    sb.to_csv(run / "scores_blanks.csv", index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        logging.info("Occupancy:\n" + summary.to_string(index=False))

    make_figure(sv, sb, per_fov, summary, reg, sens, bright, thr, cfg, run)
    maxproj_qc(cfg, sv, offsets, run)
    return thr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--rescore", type=Path, default=None,
                    help="reuse the scores and traces of this occupancy run; redo thresholds, tables and figures")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    oc = cfg["occupancy"]
    um_run = resolve_run(cfg, "unmixing", oc["unmixing_run"])
    with open(um_run / "manifest.yaml") as f:
        ves_run = resolve_run(cfg, "vesicles", yaml.safe_load(f)["vesicles_run"])
    src = resolve_run(cfg, "occupancy", str(args.rescore)) if args.rescore else None
    if src is not None:
        check_rescore_config(cfg, src)               # before creating the run folder
    run = make_run_dir(cfg, "occupancy")
    if src is not None:
        sv, sb, reg, offsets = load_scores(src)
        for f in ("registration.csv", "registration_summary.csv", "b_audit.csv"):
            (run / f).write_bytes((src / f).read_bytes())
        traces_run = rel_to_repo(src)
        inputs = [src / "scores_vesicles.csv", src / "scores_blanks.csv"]
    else:
        lab = pd.read_csv(um_run / "labels.csv", dtype=STR_COLUMNS)
        blanks = pd.read_csv(ves_run / "blanks.csv", dtype=STR_COLUMNS)
        sv, sb, reg, offsets = compute_scores(cfg, run, lab, blanks)
        traces_run = rel_to_repo(run)
        inputs = [um_run / "labels.csv", ves_run / "blanks.csv"]
    thr = summarize(cfg, run, sv, sb, reg, offsets)
    write_manifest(run, cfg, Path(__file__), inputs,
                   extra={"unmixing_run": rel_to_repo(um_run), "vesicles_run": rel_to_repo(ves_run),
                          "traces_run": traces_run, "thresholds": {k: float(v) for k, v in thr.items()},
                          "offsets_applied_by_detection_channel": {k: [float(x) for x in v] for k, v in offsets.items()}})


if __name__ == "__main__":
    main()
