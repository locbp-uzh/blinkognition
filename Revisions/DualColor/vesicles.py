#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Vesicle detection and photometry for one dual-color dataset.

The DFK785 implementation (Revisions/DFK785/vesicles.py: FOV pairing by acquisition time,
per-channel detection merged across channels, aperture photometry, blanks) run on a dataset
config of Revisions/DualColor/datasets/. Outputs as there, under
<output_root>/vesicles/run_NNN/: vesicles.csv, blanks.csv, fovs.csv, qc/, manifest.

Usage:
    python Revisions/DualColor/vesicles.py DFK788 [--data-root DIR]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc import load_config, load_dfk785, write_manifest  # noqa: E402

v785 = load_dfk785("vesicles")
from common import make_run_dir, setup_logging  # noqa: E402  (Bleedthrough/, on the path through dc)
from fov_qc import fov_groups  # noqa: E402  (DFK785/)
from segment import window_half  # noqa: E402  (Bleedthrough/)


def _measure(cfg: dict, slide: str, order: int, group: dict):
    """measure_fov in a joblib worker, which cannot unpickle a module loaded by path."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dc import load_dfk785 as _load
    return _load("vesicles").measure_fov(cfg, slide, order, group)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.dataset, a.data_root)
    h = window_half(cfg["photometry"]["annulus_px"][1])
    if cfg["detection"]["border_px"] < h + 1:
        raise ValueError(f"detection.border_px must be >= {h + 1} for the photometry window")

    jobs = [(slide, order, g) for slide in cfg["slides"] for order, g in enumerate(fov_groups(cfg, slide), start=1)]
    logging.info(f"{cfg['dataset']}: {len(jobs)} FOVs")
    res = Parallel(n_jobs=cfg["n_jobs"])(delayed(_measure)(cfg, *j) for j in jobs)
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
            v785.qc_figure(slide, r[2]["info"]["fov"], r[2]["mean"], r[2]["pos"], cfg, run / "qc")
    summary = ves.groupby("slide").agg(fovs=("fov", "nunique"), vesicles=("vesicle_id", "size"),
                                       crowded=("crowded", "sum"), nonlinear=("nonlinear", "sum"))
    logging.info("Per slide:\n" + summary.to_string())
    inputs = [Path(cfg["input_root"]) / f for c in cfg["snapshot_channels"] if f"file_{c}" in fovs
              for f in fovs[f"file_{c}"].dropna()]
    write_manifest(run, cfg, Path(__file__), inputs, extra={"n_vesicles": int(len(ves)), "n_fovs": int(len(fovs))},
                   modules=[HERE.parent / "DFK785" / "vesicles.py", HERE.parent / "DFK785" / "fov_qc.py"])


if __name__ == "__main__":
    main()
