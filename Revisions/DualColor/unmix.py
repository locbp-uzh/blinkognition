#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Vesicle labels for one dual-color dataset by per-object unmixing.

The DFK785 implementation (Revisions/DFK785/unmix.py): bleed-through signatures from the
dataset's own single-label slides (all slides of a dye pooled), per-vesicle NNLS unmixing
into ATTO390 / ATTO520 amounts, labels ATTO390, ATTO520, dual or no label, and the 515
check. Input: the latest vesicles run (or runs.vesicles). Outputs under
<output_root>/unmixing/run_NNN/ as there.

Usage:
    python Revisions/DualColor/unmix.py DFK788 [--data-root DIR]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc import load_config, load_dfk785, run_dir, write_manifest  # noqa: E402

u785 = load_dfk785("unmix")
from common import STR_COLUMNS, calibrate, make_run_dir, rel_to_repo, setup_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.dataset, a.data_root)
    ves_run = run_dir(cfg, "vesicles")
    ves = pd.read_csv(ves_run / "vesicles.csv", dtype=STR_COLUMNS)
    blanks = pd.read_csv(ves_run / "blanks.csv", dtype=STR_COLUMNS)
    ves, cal = calibrate(ves, blanks, u785.SNAPSHOT)

    sig_summary, sig_fov = u785.estimate_signatures(ves, cfg)
    sig = u785.cl.signatures(cfg, sig_summary, cfg["unmixing"]["channels"])
    logging.info("Signatures (flux per unit of home-channel flux):\n" + sig.to_string())
    lab = u785.label_vesicles(ves, cal, sig, cfg)
    counts = pd.crosstab(lab["slide"], lab["label"]).reindex(
        columns=list(cfg["dyes"]) + [u785.cl.DUAL, u785.cl.NONE], fill_value=0)
    chk = u785.check_515(lab, cfg)

    run = make_run_dir(cfg, "unmixing")
    keep = ["slide", "dye", "fov", "acq_order", "vesicle_id", "y_px", "x_px", "detected_by", "detected_in",
            "nn_dist_px", "crowded", "nonlinear", "file_640"]
    keep += [f"{p}_{c}" for c in u785.SNAPSHOT for p in ("F", "z")] + [f"t_{d}" for d in cfg["dyes"]] + ["label"]
    lab[keep].to_csv(run / "labels.csv", index=False)
    sig_summary.to_csv(run / "signatures.csv", index=False)
    sig_fov.to_csv(run / "signatures_per_fov.csv", index=False)
    counts.to_csv(run / "label_counts.csv")
    chk.to_csv(run / "check_515.csv", index=False)
    cal.to_csv(run / "calibration.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        logging.info("Signature estimates:\n" + sig_summary.to_string(index=False))
        logging.info("Labels per slide:\n" + counts.to_string())
        logging.info("515 check:\n" + chk.to_string(index=False))
    u785.make_figure(lab, cfg, run)
    write_manifest(run, cfg, Path(__file__), [ves_run / "vesicles.csv", ves_run / "blanks.csv"],
                   extra={"vesicles_run": rel_to_repo(ves_run)}, modules=[HERE.parent / "DFK785" / "unmix.py"])


if __name__ == "__main__":
    main()
