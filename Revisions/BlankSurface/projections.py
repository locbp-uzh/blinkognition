#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Images for the R3.1 figure: per field of view of every slide in config.yaml, read from the raw movies in place,
  - 640 nm: maximum projection and mean over all frames (the mean is the field's background level);
  - vesicle channel (the slide's gt channel): mean of the snapshot frames.
The figure script picks the fields to show (from count.py's numbers) and sets one common intensity scale.

Outputs (Results/Revisions/BlankSurface/projections/run_NNN/): images.npz (keys
<session>|<slide>|<fov>|max640, |mean640, |vesicle; float32, 230 x 230), fields.csv (per field: files, frames,
percentiles of each image, ND2 camera settings and laser power), manifest.yaml, config.yaml, code/.

Usage (root of the S3IT review clone, on a compute node, picasso-env):
    python Revisions/BlankSurface/projections.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "PaperData"))
from paperdata import load_config as load_paper_config, make_run_dir, nd2_camera, write_manifest  # noqa: E402

FOV_RE = re.compile(r"_(\d{4})\.nd2$")
CHUNK = 500
N_JOBS = 16


def fov_of(name: str) -> str:
    m = FOV_RE.search(name)
    return m.group(1) if m else "0000"


def field_images(s: dict, p640: Path, pgt: Path) -> tuple[dict, dict]:
    import nd2
    with nd2.ND2File(str(p640)) as h:
        d = h.to_dask()
        n = d.shape[0]
        mx = np.full(d.shape[-2:], -np.inf)
        sm = np.zeros(d.shape[-2:])
        for i in range(0, n, CHUNK):
            a = np.asarray(d[i:i + CHUNK], dtype=np.float64)
            mx = np.maximum(mx, a.max(axis=0))
            sm += a.sum(axis=0)
    with nd2.ND2File(str(pgt)) as h:
        g = np.asarray(h.asarray(), dtype=np.float64)
    g = g.reshape(-1, *g.shape[-2:]).mean(axis=0)
    imgs = {"max640": mx.astype(np.float32), "mean640": (sm / n).astype(np.float32), "vesicle": g.astype(np.float32)}
    row = {"session": s["session"], "slide": s["name"], "role": s["role"], "gt": s["gt"], "fov": fov_of(p640.name),
           "file_640": str(p640), "file_vesicle": str(pgt), "n_frames_640": int(n)}
    for k, im in imgs.items():
        for q in (1, 50, 99, 99.9):
            row[f"{k}_p{q:g}"] = float(np.percentile(im, q))
    cam = nd2_camera(p640)
    row.update({k: cam.get(k) for k in ("GainMultiplier", "ExposureTime", "laser_640nm_pct_median", "dt_median_ms",
                                        "date")})
    return row, imgs


def main() -> None:
    with open(HERE / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(HERE / "config.yaml")
    pcfg = load_paper_config()
    jobs = []
    for s in cfg["slides"]:
        src = Path(cfg["source_root"]) / s["folder"]
        files = [p for p in src.iterdir() if p.suffix.lower() == ".nd2" and not p.name.startswith("._")
                 and p.name.startswith(s["prefix"] + "_")]
        gts = {fov_of(p.name): p for p in files if f"_{s['gt']}nm_" in p.name}
        for p in sorted(p for p in files if "_640nm_" in p.name):
            jobs.append((s, p, gts[fov_of(p.name)]))
    print(f"{len(jobs)} fields", flush=True)
    out = make_run_dir(pcfg, "projections", topic_root=cfg["output_root"])
    res = Parallel(n_jobs=N_JOBS)(delayed(field_images)(*j) for j in jobs)
    rows, arrays = [], {}
    for row, imgs in res:
        rows.append(row)
        for k, im in imgs.items():
            arrays[f"{row['session']}|{row['slide']}|{row['fov']}|{k}"] = im
    np.savez_compressed(out / "images.npz", **arrays)
    fields = pd.DataFrame(rows)
    fields.to_csv(out / "fields.csv", index=False, float_format="%.6g")
    write_manifest(out, cfg, Path(__file__), [], hash_inputs=False,
                   extra={"n_fields": len(rows), "chunk_frames": CHUNK})
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(fields.groupby(["session", "slide"])[["n_frames_640", "mean640_p50", "max640_p50", "max640_p99.9",
                                                    "vesicle_p99.9", "laser_640nm_pct_median"]].median())
    print(out)


if __name__ == "__main__":
    main()
