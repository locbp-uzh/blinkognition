#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Detect vesicles once per FOV and measure them in all four channels.

Detection is channel-agnostic: each detection channel (405, 488, 515) is
background-flattened, matched-filtered and scaled to its own noise, and the
per-pixel maximum of those SNR maps is searched for peaks. A vesicle is thus
found no matter which dye it carries, and the same position is then measured in
every channel. Segmenting each channel separately would only find the ATTO525
vesicles that happen to bleed strongly into 405, biasing the bleed-through
estimate toward the brightest vesicles.

Photometry per vesicle and channel, on the raw image:
    flux     = sum(aperture) - n_aperture * median(annulus)
    flux_err = robust_sigma(annulus) * sqrt(n_aperture)     (background noise only)

Outputs (Results/Revisions/Bleedthrough/segmentation/run_NNN/):
    vesicles.csv      one row per vesicle (all FOVs), columns documented in README.md
    qc/<fov>.pdf      the four channels of each FOV with the detections circled
    manifest.yaml, config.yaml, segment.py

Usage:
    python Revisions/Bleedthrough/segment.py [--config path] [--no-qc]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from scipy.spatial import cKDTree

from common import (COLORS, discover_fovs, find_peaks, flatten, load_config, load_fov, load_scm,
                    make_run_dir, refine_centroid, robust_sigma, save_pdf, setup_logging, snr_map,
                    write_manifest)

# --- Constants ---
QC_CONTRAST_PERCENTILES = (1.0, 99.8)
QC_PANEL_IN = 3.5           # inches per QC panel


def photometry(img: np.ndarray, pos: np.ndarray, r_ap: float, r_in: float, r_out: float,
               sat_value: float) -> pd.DataFrame:
    """Aperture flux with local annulus background for each (y, x) in pos."""
    h = int(np.ceil(r_out)) + 1
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    rows = []
    for y, x in pos:
        iy, ix = int(round(y)), int(round(x))
        win = img[iy - h:iy + h + 1, ix - h:ix + h + 1]
        d = np.hypot(yy + iy - y, xx + ix - x)
        ap, ann = win[d <= r_ap], win[(d >= r_in) & (d <= r_out)]
        bg, sig = np.median(ann), robust_sigma(ann)
        n = ap.size
        rows.append({"flux": ap.sum() - n * bg, "flux_err": sig * np.sqrt(n), "bg": bg,
                     "peak_raw": ap.max(), "saturated": bool(ap.max() >= sat_value)})
    return pd.DataFrame(rows)


def process_fov(f: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    images, meta = load_fov(f["path"], cfg)
    d, p = cfg["detection"], cfg["photometry"]
    det_ch = cfg["detection_channels"]

    flats, snrs = {}, {}
    for ch in det_ch:
        flats[ch], _ = flatten(images[ch], cfg["background"]["median_size_px"])
        snrs[ch] = snr_map(flats[ch], d["psf_sigma_px"])
    stack = np.stack([snrs[c] for c in det_ch])
    combined = stack.max(axis=0)
    best = stack.argmax(axis=0)

    pk = find_peaks(combined, d["snr_threshold"], d["min_separation_px"], d["border_px"])
    pos = np.empty((len(pk), 2))
    det_by = np.array([det_ch[best[y, x]] for y, x in pk], dtype=object)
    for ch in det_ch:
        m = det_by == ch
        if m.any():
            pos[m] = refine_centroid(flats[ch], pk[m])

    df = pd.DataFrame({"slide": f["slide"], "fov": f["fov"], "vesicle_id": np.arange(len(pk)),
                       "y_px": pos[:, 0], "x_px": pos[:, 1], "detected_by": det_by,
                       "snr_detect": combined[pk[:, 0], pk[:, 1]]})
    for ch in det_ch:
        df[f"snr_{ch}"] = snrs[ch][pk[:, 0], pk[:, 1]]

    nn = cKDTree(pos).query(pos, k=2)[0][:, 1] if len(pos) > 1 else np.full(len(pos), np.inf)
    df["nn_dist_px"] = nn
    df["crowded"] = nn < p["crowding_radius_px"]

    sat_any = np.zeros(len(df), dtype=bool)
    for ch, img in images.items():
        ph = photometry(img, pos, p["aperture_radius_px"], *p["annulus_px"], meta["saturation_value"])
        for col in ("flux", "flux_err", "bg"):
            df[f"{col}_{ch}"] = ph[col].to_numpy()
        sat_any |= ph["saturated"].to_numpy()
    df["saturated"] = sat_any
    return df, {"images": images, "pos": pos, "meta": meta}


def qc_figure(f: dict, images: dict, pos: np.ndarray, cfg: dict, out: Path) -> None:
    cmap = load_scm("grayC")
    chans = list(cfg["channels"])
    fig, axs = plt.subplots(1, len(chans), figsize=(QC_PANEL_IN * len(chans), QC_PANEL_IN + 0.4))
    r = cfg["photometry"]["aperture_radius_px"]
    for ax, ch in zip(axs, chans):
        im = images[ch]
        lo, hi = np.percentile(im, QC_CONTRAST_PERCENTILES)
        ax.imshow(im, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        circles = [Circle((x, y), r) for y, x in pos]
        ax.add_collection(PatchCollection(circles, facecolor="none", edgecolor=COLORS["vermillion"],
                                          linewidth=0.3))
        ax.set_title(f"{ch} channel", color="black")
        ax.set_axis_off()
    fig.suptitle(f"{f['fov']} ({f['slide']} slide), n = {len(pos)} vesicles", fontsize=7)
    fig.tight_layout()
    save_pdf(fig, out / f"{f['fov']}.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--no-qc", action="store_true", help="skip the per-FOV QC figures")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    fovs = discover_fovs(cfg)
    run = make_run_dir(cfg, "segmentation")
    (run / "qc").mkdir()

    tables, pixel_um = [], None
    for f in fovs:
        df, extra = process_fov(f, cfg)
        pixel_um = extra["meta"]["pixel_um"]
        logging.info(f"{f['fov']}: {len(df)} vesicles "
                     f"(detected by {df['detected_by'].value_counts().to_dict()}, "
                     f"{int(df['crowded'].sum())} crowded, {int(df['saturated'].sum())} saturated)")
        tables.append(df)
        if not args.no_qc:
            qc_figure(f, extra["images"], extra["pos"], cfg, run / "qc")

    ves = pd.concat(tables, ignore_index=True)
    ves.to_csv(run / "vesicles.csv", index=False)
    logging.info(f"Wrote {len(ves)} vesicles to {run / 'vesicles.csv'}")
    write_manifest(run, cfg, Path(__file__), [f["path"] for f in fovs],
                   extra={"pixel_um": pixel_um, "n_vesicles": int(len(ves)),
                          "n_per_slide": ves["slide"].value_counts().to_dict()})


if __name__ == "__main__":
    main()
