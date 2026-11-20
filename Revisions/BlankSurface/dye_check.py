#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Which vesicle dye is on which BlankSurface slide (R3.1)? DFK786's readme gives ATTO390 vesicles for both
slides 2 and 3, but its folders are ATT390 (files ATTO390_*) and ATTO520 (files blank_*).

Per slide of config.yaml and per field of view, from the 10-frame snapshots at 405, 488 and 515 nm: the
mean over frames of (99.9th percentile - median) of the pixel values, a robust measure of bright objects
above background. ATTO390 is excited at 405 nm, ATTO520 at 488 and 515 nm; slides of known dye (DFK787,
DFK788) are the reference. Descriptive.

Outputs (Results/Revisions/BlankSurface/dye_check/run_NNN/): snapshots.csv (one row per slide, field and
channel), summary.csv (median over fields), manifest.yaml, config.yaml, code/.

Usage (clone root, on an S3IT compute node; reads ND2 headers and 10 frames per snapshot):
    python Revisions/BlankSurface/dye_check.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "PaperData"))
from paperdata import load_config, make_run_dir, write_manifest  # noqa: E402

CHANNELS = ("405", "488", "515")
FOV_RE = re.compile(r"_(\d{4})\.nd2$")


def main() -> None:
    import nd2
    with open(HERE / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(HERE / "config.yaml")
    run = make_run_dir(load_config(), "dye_check", topic_root=cfg["output_root"])
    rows, inputs = [], []
    for s in cfg["slides"]:
        src = Path(cfg["source_root"]) / s["folder"]
        for ch in CHANNELS:
            for p in sorted(src.glob(f"{s['prefix']}_{ch}nm_*.nd2")):
                if p.name.startswith("._"):
                    continue
                with nd2.ND2File(str(p)) as h:
                    x = np.asarray(h.asarray(), dtype=np.float64)
                x = x.reshape(-1, *x.shape[-2:])
                score = float(np.mean([np.percentile(f, 99.9) - np.median(f) for f in x]))
                m = FOV_RE.search(p.name)
                rows.append({"session": s["session"], "slide": s["name"], "folder": s["folder"], "channel": ch,
                             "fov": int(m.group(1)) if m else 0, "n_frames": int(x.shape[0]),
                             "bright_minus_median_adu": score, "file": str(p)})
                inputs.append(p)
    df = pd.DataFrame(rows)
    df.to_csv(run / "snapshots.csv", index=False)
    summ = df.groupby(["session", "slide", "folder", "channel"])["bright_minus_median_adu"].agg(
        ["median", "min", "max", "size"]).reset_index()
    summ.to_csv(run / "summary.csv", index=False, float_format="%.1f")
    write_manifest(run, cfg, Path(__file__), [], hash_inputs=False,
                   extra={"inputs_nd2": [str(p) for p in inputs]})
    print(summ.to_string(index=False))
    print(run)


if __name__ == "__main__":
    main()
