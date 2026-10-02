#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Per-FOV vesicle counts in acquisition order, to find where a slide dried out.

For every FOV of every slide: the acquisition time (from the ND2 metadata, since
the file numbers do not follow acquisition order on slide 3), and the number of
vesicles detected in each counting channel (10-frame mean, background flattened,
matched filter, clipped-MAD noise, peaks above snr_threshold; the detection code of
Revisions/Bleedthrough). A dried slide shows almost no vesicles.

Outputs (Results/Revisions/DFK785/fov_qc/run_NNN/):
    fov_counts.csv     one row per slide x FOV: id, files, time, acquisition order, counts
    fov_counts.pdf     counts against acquisition order per slide; source data in fov_counts.csv
    manifest.yaml, config.yaml, code/

Usage:
    python Revisions/DFK785/fov_qc.py [--config path]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nd2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Bleedthrough"))
from common import (COLORS, apply_axis_standards, find_peaks, flatten, load_config, make_run_dir,  # noqa: E402
                    save_pdf, setup_logging, snr_map, write_manifest)

# --- Constants ---
FOV_RE = re.compile(r"_(\d{4})\.nd2$")
CHANNEL_COLORS = {"405": COLORS["blue"], "488": COLORS["green"], "515": COLORS["orange"]}
FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 4.6


def acquisition_time(path: Path) -> datetime:
    with nd2.ND2File(str(path)) as h:
        return datetime.strptime(h.text_info["date"].replace("  ", " "), "%d/%m/%Y %H:%M:%S")


def fov_groups(cfg: dict, slide: str) -> list[dict[str, tuple[Path, datetime]]]:
    """FOVs of a slide in acquisition order, each {channel: (path, time)}.

    A FOV is the set of channel files recorded together: files are sorted by their
    ND2 acquisition time and split wherever consecutive files are more than
    fov_max_gap_s apart. The ND2 'date' stamp is not the frame time (by frame
    times the 405/488 snapshots fall 2 s after their own 640 movie and 2 s before
    the next FOV's), but the stamps of one acquisition iteration cluster within
    seconds; the pairing was confirmed by colocalization (vesicles blink in their
    own 640 movie at 11 %, in the neighboring FOVs' movies at 3.5-4 %). File numbers are NOT used to pair channels, because on
    slide 3 the numbering is shifted between channels (file _0031 of one channel
    was recorded two hours apart from _0031 of another).
    """
    s = cfg["slides"][slide]
    folder = Path(cfg["input_root"]) / s["folder"]
    files = []
    for ch in s["channels"]:
        stem = f"{s['prefix']}_{cfg['channel_tags'][ch]}"
        for p in folder.glob(f"{stem}*.nd2"):
            if p.name.startswith("._") or not (FOV_RE.search(p.name) or p.name == f"{stem}.nd2"):
                continue
            files.append((acquisition_time(p), ch, p))
    files.sort()
    groups, current, last = [], {}, None
    for t, ch, p in files:
        if last is not None and (t - last).total_seconds() > cfg["fov_max_gap_s"]:
            groups.append(current)
            current = {}
        if ch in current:
            raise ValueError(f"{slide}: two {ch} files in one FOV group: {current[ch][0].name}, {p.name}")
        current[ch] = (p, t)
        last = t
    groups.append(current)
    bad = [i for i, g in enumerate(groups) if set(g) != set(s["channels"])]
    if bad:
        raise ValueError(f"{slide}: FOV groups {bad} lack channels: "
                         + "; ".join(f"{i}: {sorted(set(s['channels']) - set(groups[i]))}" for i in bad))
    return groups


def count_vesicles(path: Path, cfg: dict) -> int:
    d = cfg["detection"]
    img = nd2.imread(str(path)).astype(np.float64)
    img = img.mean(axis=0) if img.ndim == 3 else img
    flat, _ = flatten(img, cfg["background"]["median_size_px"])
    snr = snr_map(flat, d["psf_sigma_px"])
    return len(find_peaks(snr, d["snr_threshold"], d["min_separation_px"], d["border_px"]))


def make_figure(df: pd.DataFrame, cfg: dict, out: Path) -> None:
    slides = list(cfg["slides"])
    fig, axs = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharey=True)
    for ax, slide, label in zip(axs.ravel(), slides, "abcd"):
        g = df[df["slide"] == slide]
        for ch in cfg["count_channels"]:
            if ch in cfg["slides"][slide]["channels"]:
                ax.plot(g["acq_order"], g[f"n_{ch}"], "o-", ms=2.5, lw=0.8, color=CHANNEL_COLORS[ch], label=ch)
        ax.set_title(f"{slide}: {cfg['slides'][slide]['sample']}", color="black", fontsize=6)
        ax.set_xlabel("FOV, in acquisition order")
        ax.set_ylabel("Vesicles detected")
        ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
        apply_axis_standards(ax)
    handles = [plt.Line2D([], [], marker="o", ms=2.5, lw=0.8, color=c) for c in CHANNEL_COLORS.values()]
    fig.legend(handles, [f"{c} channel" for c in CHANNEL_COLORS], loc="lower center", ncol=3, frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.17, hspace=0.6, wspace=0.25)
    save_pdf(fig, out / "fov_counts.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)

    rows, inputs = [], []
    for slide, s in cfg["slides"].items():
        groups = fov_groups(cfg, slide)
        for order, g in enumerate(groups, start=1):
            times = [t for _, t in g.values()]
            ids = {ch: (FOV_RE.search(p.name).group(1) if FOV_RE.search(p.name) else "none") for ch, (p, _) in g.items()}
            row = {"slide": slide, "acq_order": order, "time": min(times),
                   "time_span_s": (max(times) - min(times)).total_seconds(),
                   "file_numbers": ",".join(f"{ch}:{ids[ch]}" for ch in s["channels"]),
                   **{f"file_{ch}": str(p.relative_to(cfg["input_root"])) for ch, (p, _) in g.items()}}
            for ch in cfg["count_channels"]:
                if ch in g:
                    row[f"n_{ch}"] = count_vesicles(g[ch][0], cfg)
            rows.append(row)
            inputs += [p for ch, (p, _) in g.items() if ch in cfg["count_channels"]]
        logging.info(f"{slide}: {len(groups)} FOVs")
    df = pd.DataFrame(rows)
    df["minutes_from_start"] = df.groupby("slide")["time"].transform(lambda t: (t - t.min()).dt.total_seconds() / 60)

    run = make_run_dir(cfg, "fov_qc")
    df.to_csv(run / "fov_counts.csv", index=False)
    make_figure(df, cfg, run)
    with pd.option_context("display.width", 200, "display.max_rows", 300):
        cols = ["slide", "acq_order", "file_numbers", "time", "time_span_s"] + [c for c in df if c.startswith("n_")]
        logging.info("Counts:\n" + df[cols].to_string(index=False))
    # Raw-data checksums would read 75 GB of 640 movies; only the counted files are hashed.
    write_manifest(run, cfg, Path(__file__), inputs)


if __name__ == "__main__":
    main()
