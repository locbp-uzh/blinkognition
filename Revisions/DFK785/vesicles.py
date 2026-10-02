#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Detect vesicles in every FOV of DFK785 and measure them in the snapshot channels.

FOVs are the time-paired channel groups of fov_qc.fov_groups (file numbers are
shifted between channels on slide 3). Per FOV, the 10-frame movies of the snapshot
channels present on that slide (405, 488 and, on the mixed slides, 515) are
averaged; vesicles are detected in each detection channel separately and the peak
lists merged (merge_peaks), then measured in every snapshot channel at the same
position with the photometry of
Revisions/Bleedthrough/segment.py (aperture sum minus annulus median). Blanks
(empty apertures) are measured the same way. An object is flagged 'nonlinear' if
any aperture pixel reaches camera.nonlinear_above_counts in any single frame.

The ND2 metadata of this dataset does not record per-channel settings (every file
of a slide carries the same microscope state), so acquisition settings come from
the file names and the data readme, not from here.

Outputs (Results/Revisions/DFK785/vesicles/run_NNN/):
    vesicles.csv   one row per vesicle: slide, dye, fov, acq_order, time, ids, position, flags,
                   flux/flux_err/bg/peak_raw per snapshot channel (NaN where the channel is absent),
                   file_640 (the protein movie of that FOV)
    blanks.csv     the same photometry at 150 random empty positions per FOV
    fovs.csv       one row per FOV with its files and counts
    qc/<slide>_<fov>.pdf for the first FOV of each slide
    manifest.yaml, config.yaml, code/

Usage:
    python Revisions/DFK785/vesicles.py [--config path]
"""

from __future__ import annotations

import argparse
import logging
import sys
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import nd2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Bleedthrough"))
from common import (COLORS, find_peaks, flatten, fov_key, load_config, load_scm, make_run_dir,  # noqa: E402
                    refine_centroid, sample_blank_positions, save_pdf, setup_logging, snr_map, write_manifest)
from fov_qc import fov_groups  # noqa: E402
from segment import PHOT_COLUMNS, photometry, window_half  # noqa: E402

# --- Constants ---
QC_PANEL_IN = 2.4


def merge_peaks(flats: dict, snrs: dict, d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Peaks found in each channel separately, merged across channels.

    A per-pixel maximum of the channels' SNR maps (as in Revisions/Bleedthrough)
    loses vesicles in a mix: a dim 405-only vesicle on the wing of a bright
    515 vesicle is no longer a local maximum of the combined map (46 % of the
    405 vesicles of slide 3 were missed that way). Here every channel is searched
    on its own; peaks closer than detection.merge_radius_px are one vesicle (the
    radius also absorbs the ~1 px chromatic shift of ATTO390 between 405 and 488),
    placed at the centroid of its highest-SNR detection.

    Returns positions (n, 2), detected_by (highest-SNR channel), that SNR, and
    detected_in (comma-separated channels with a peak within the radius).
    """
    cands = []
    for c in flats:
        pk = find_peaks(snrs[c], d["snr_threshold"], d["min_separation_px"], d["border_px"])
        if len(pk):
            xy = refine_centroid(flats[c], pk)
            cands += [(snrs[c][y, x], c, p) for (y, x), p in zip(pk, xy)]
    cands.sort(key=lambda t: -t[0])
    pos, by, best, seen = [], [], [], []
    for snr, c, p in cands:
        if pos:
            dist = np.hypot(*(np.asarray(pos) - p).T)
            j = int(dist.argmin())
            if dist[j] <= d["merge_radius_px"]:
                if c not in seen[j]:
                    seen[j].append(c)
                continue
        pos.append(p), by.append(c), best.append(snr), seen.append([c])
    order = [c for c in flats]
    return (np.asarray(pos).reshape(-1, 2), np.asarray(by, dtype=object), np.asarray(best),
            np.asarray([",".join(sorted(s, key=order.index)) for s in seen], dtype=object))


def measure_fov(cfg: dict, slide: str, order: int, group: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    s, d, p, b = cfg["slides"][slide], cfg["detection"], cfg["photometry"], cfg["blanks"]
    fov = f"FOV{order:02d}"
    chans = [c for c in cfg["snapshot_channels"] if c in group]
    mean, peak = {}, {}
    for ch in chans:
        mov = nd2.imread(str(group[ch][0])).astype(np.float64)
        mean[ch], peak[ch] = mov.mean(axis=0), mov.max(axis=0)
    det = [c for c in cfg["detection_channels"] if c in chans]
    flats = {c: flatten(mean[c], cfg["background"]["median_size_px"])[0] for c in det}
    snrs = {c: snr_map(flats[c], d["psf_sigma_px"]) for c in det}
    pos, det_by, snr_best, seen_in = merge_peaks(flats, snrs, d)
    shape = next(iter(flats.values())).shape

    time = min(t for _, t in group.values())
    base = {"slide": slide, "dye": s["dye"], "fov": fov, "acq_order": order, "time": time}
    ves = pd.DataFrame({**base, "vesicle_id": np.arange(len(pos)), "y_px": pos[:, 0], "x_px": pos[:, 1],
                        "detected_by": det_by, "detected_in": seen_in, "snr_detect": snr_best})
    nn = cKDTree(pos).query(pos, k=2)[0][:, 1] if len(pos) > 1 else np.full(len(pos), np.inf)
    ves["nn_dist_px"], ves["crowded"] = nn, nn < p["crowding_radius_px"]

    rng = np.random.default_rng([b["seed"], zlib.crc32(fov_key(slide, fov).encode())])
    bpos = sample_blank_positions(shape, pos, b["n_per_fov"], b["min_dist_px"], d["border_px"], rng)
    blanks = pd.DataFrame({**base, "blank_id": np.arange(len(bpos)), "y_px": bpos[:, 0], "x_px": bpos[:, 1]})

    nonlinear = np.zeros(len(ves), dtype=bool)
    for ch in cfg["snapshot_channels"]:
        if ch not in chans:
            for df in (ves, blanks):
                for col in PHOT_COLUMNS:
                    df[f"{col}_{ch}"] = np.nan
            continue
        for df, xy in ((ves, pos), (blanks, bpos)):
            ph = photometry(mean[ch], xy, p["aperture_radius_px"], *p["annulus_px"])
            for col in PHOT_COLUMNS:
                df[f"{col}_{ch}"] = ph[col].to_numpy()
        # nonlinearity is a property of single frames, so use the per-pixel maximum over frames
        pk_raw = photometry(peak[ch], pos, p["aperture_radius_px"], *p["annulus_px"])["peak_raw"].to_numpy()
        ves[f"peak_raw_{ch}"] = pk_raw
        nonlinear |= pk_raw >= cfg["camera"]["nonlinear_above_counts"]
    ves["nonlinear"] = nonlinear
    rel = {c: str(g[0].relative_to(cfg["input_root"])) for c, g in group.items()}
    ves["file_640"] = blanks["file_640"] = rel["640"]
    info = {**base, "n_vesicles": len(ves), "n_blanks": len(blanks), **{f"file_{c}": f for c, f in rel.items()}}
    return ves, blanks, {"info": info, "mean": mean, "pos": pos}


def qc_figure(slide: str, fov: str, mean: dict, pos: np.ndarray, cfg: dict, out: Path) -> None:
    cmap = load_scm("grayC")
    fig, axs = plt.subplots(1, len(mean), figsize=(QC_PANEL_IN * len(mean), QC_PANEL_IN + 0.3))
    for ax, (ch, im) in zip(np.atleast_1d(axs), mean.items()):
        lo, hi = np.percentile(im, (1, 99.8))
        ax.imshow(im, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.add_collection(PatchCollection([Circle((x, y), cfg["photometry"]["aperture_radius_px"]) for y, x in pos],
                                          facecolor="none", edgecolor=COLORS["vermillion"], linewidth=0.3))
        ax.set_title(f"{ch} channel", color="black")
        ax.set_axis_off()
    fig.suptitle(f"{slide} {fov}: n = {len(pos)} vesicles", fontsize=7)
    fig.tight_layout()
    save_pdf(fig, out / f"{slide}_{fov}.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    h = window_half(cfg["photometry"]["annulus_px"][1])
    if cfg["detection"]["border_px"] < h + 1:
        raise ValueError(f"detection.border_px must be >= {h + 1} for the photometry window")

    jobs = [(slide, order, g) for slide in cfg["slides"] for order, g in enumerate(fov_groups(cfg, slide), start=1)]
    logging.info(f"{len(jobs)} FOVs")
    res = Parallel(n_jobs=cfg["n_jobs"])(delayed(measure_fov)(cfg, *j) for j in jobs)
    run = make_run_dir(cfg, "vesicles")
    (run / "qc").mkdir()
    ves = pd.concat([r[0] for r in res], ignore_index=True)
    blanks = pd.concat([r[1] for r in res], ignore_index=True)
    fovs = pd.DataFrame([r[2]["info"] for r in res])
    ves.to_csv(run / "vesicles.csv", index=False)
    blanks.to_csv(run / "blanks.csv", index=False)
    fovs.to_csv(run / "fovs.csv", index=False)
    for (slide, order, _), r in zip(jobs, res):
        if order == 1:
            qc_figure(slide, r[2]["info"]["fov"], r[2]["mean"], r[2]["pos"], cfg, run / "qc")
    summary = ves.groupby("slide").agg(fovs=("fov", "nunique"), vesicles=("vesicle_id", "size"),
                                       crowded=("crowded", "sum"), nonlinear=("nonlinear", "sum"))
    logging.info("Per slide:\n" + summary.to_string())
    inputs = [Path(cfg["input_root"]) / f for c in cfg["snapshot_channels"] if f"file_{c}" in fovs
              for f in fovs[f"file_{c}"].dropna()]
    write_manifest(run, cfg, Path(__file__), inputs, extra={"n_vesicles": int(len(ves)), "n_fovs": int(len(fovs))})


if __name__ == "__main__":
    main()
