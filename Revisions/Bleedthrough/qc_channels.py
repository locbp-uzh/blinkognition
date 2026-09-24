#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Channel QC before segmentation.

1. Registration: detect spots independently in every channel, match them between
   channel pairs by nearest neighbor, and report the median (dy, dx) offset. A
   pair is only reported where one dye is visible in both channels (e.g. 405 vs
   515 on the ATTO390 slide, 488 vs 515 on the ATTO525 slide).
2. PSF width: second-moment sigma of bright, isolated spots per channel, to set
   detection.psf_sigma_px.

Outputs (Results/Revisions/Bleedthrough/qc_channels/run_NNN/):
    offsets.csv       one row per slide x channel pair (pooled over FOVs)
    offsets_per_fov.csv  same, per FOV (pairs with >= qc.min_matches_fov matches)
    psf_sigma.csv     one row per slide x channel
    manifest.yaml, config.yaml, qc_channels.py

Usage:
    python Revisions/Bleedthrough/qc_channels.py [--config path]
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import (discover_fovs, find_peaks, flatten, load_config, load_fov, make_run_dir,
                    refine_centroid, second_moment_sigma, setup_logging, snr_map, write_manifest)

# --- Constants ---
BRIGHT_SNR = 20.0       # spots used for the PSF-width estimate
ISOLATION_PX = 10       # ... and no other spot within this distance


def detect(images: dict, cfg: dict) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per channel: (subpixel positions, SNR at peak, flattened image)."""
    d = cfg["detection"]
    out = {}
    for ch, img in images.items():
        flat, _ = flatten(img, cfg["background"]["median_size_px"])
        snr = snr_map(flat, d["psf_sigma_px"])
        pk = find_peaks(snr, d["snr_threshold"], d["min_separation_px"], d["border_px"])
        out[ch] = (refine_centroid(flat, pk), snr[pk[:, 0], pk[:, 1]], flat, pk)
    return out


def match_offsets(ref: np.ndarray, mov: np.ndarray, radius: float) -> np.ndarray:
    """(dy, dx) = mov - ref for mutual nearest neighbors within radius."""
    if len(ref) == 0 or len(mov) == 0:
        return np.empty((0, 2))
    d1, j = cKDTree(mov).query(ref, distance_upper_bound=radius)
    _, i_back = cKDTree(ref).query(mov, distance_upper_bound=radius)
    ok = np.isfinite(d1)
    ok[ok] &= i_back[j[ok]] == np.arange(len(ref))[ok]
    return mov[j[ok]] - ref[ok]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    fovs = discover_fovs(cfg)
    run = make_run_dir(cfg, "qc_channels")
    q = cfg["qc"]

    offsets, sigmas, per_fov = {}, [], []
    for f in fovs:
        images, meta = load_fov(f["path"], cfg)
        det = detect(images, cfg)
        logging.info(f"{f['fov']}: spots per channel " + ", ".join(f"{c}={len(v[0])}" for c, v in det.items()))
        for a, b in itertools.combinations(cfg["channels"], 2):
            d = match_offsets(det[a][0], det[b][0], q["match_radius_px"])
            offsets.setdefault((f["slide"], a, b), []).append(d)
            if len(d) >= q["min_matches_fov"]:
                med = np.median(d, axis=0)
                per_fov.append({"slide": f["slide"], "fov": f["fov"], "ref": a, "mov": b, "n_matched": len(d),
                                "dy_median_px": med[0], "dx_median_px": med[1]})
        for ch, (pos, snr, flat, pk) in det.items():
            if len(pos) < 2:
                continue
            nn, _ = cKDTree(pos).query(pos, k=2)
            keep = (snr > BRIGHT_SNR) & (nn[:, 1] > ISOLATION_PX)
            for p in pk[keep]:
                sigmas.append({"slide": f["slide"], "fov": f["fov"], "channel": ch,
                               "sigma_px": second_moment_sigma(flat, tuple(p))})

    rows = []
    for (slide, a, b), parts in offsets.items():
        d = np.concatenate(parts)
        row = {"slide": slide, "ref": a, "mov": b, "n_matched": len(d)}
        if len(d) >= q["min_matches"]:
            med = np.median(d, axis=0)
            iqr = np.subtract(*np.percentile(d, [75, 25], axis=0))
            row.update(dy_median_px=med[0], dx_median_px=med[1], dy_iqr_px=iqr[0], dx_iqr_px=iqr[1])
        rows.append(row)
    off = pd.DataFrame(rows)
    off.to_csv(run / "offsets.csv", index=False)
    off_fov = pd.DataFrame(per_fov)
    off_fov.to_csv(run / "offsets_per_fov.csv", index=False)

    sig = pd.DataFrame(sigmas)
    sig_sum = (sig.groupby(["slide", "channel"])["sigma_px"]
               .agg(n="count", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
               .reset_index())
    sig_sum.to_csv(run / "psf_sigma.csv", index=False)

    with pd.option_context("display.width", 200, "display.precision", 3):
        logging.info("Channel offsets (mov - ref, px):\n" + off.to_string(index=False))
        logging.info("Per-FOV offsets (px):\n" + off_fov.to_string(index=False))
        logging.info("Spot sigma (px):\n" + sig_sum.to_string(index=False))
    write_manifest(run, cfg, Path(__file__), [f["path"] for f in fovs])


if __name__ == "__main__":
    main()
