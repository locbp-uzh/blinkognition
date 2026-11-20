#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Bright-frame counts of every unfiltered trace, inside vesicles (IN) and at background positions (R3.3b).

Per background-removed trace: median, sigma = mad_to_sigma x MAD (both robust to the few bright frames of
a mostly-OFF blinking trace), and the number of frames above median + k x sigma for each k in
config.yaml `bright_frames.k_values`. Independent of the paper's GMM; molecule_like.py uses the
background counts of each session as the noise reference.

Outputs (Results/Revisions/PhotonsFrames/bright_frames/run_NNN/): counts_IN.csv, counts_background.csv
(construct, uniqueID, session, movie, median_adu, sigma_adu, n_bright_k<k>), manifest.yaml, config.yaml,
code/.

Usage (repo root; on S3IT with --remote through s3it/bright_frames.sbatch):
    python Revisions/PhotonsFrames/bright_frames.py [--remote] [--constructs ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "PaperData"))
from events import load_topic_config  # noqa: E402
from paperdata import load_config, load_index, make_run_dir, trace_path, write_manifest  # noqa: E402


def bright_counts(x: np.ndarray, ks: list[float], mad_to_sigma: float) -> dict:
    """x: frames x traces (ADU). Robust median and sigma per trace, and counts above median + k sigma."""
    med = np.median(x, axis=0)
    sigma = mad_to_sigma * np.median(np.abs(x - med), axis=0)
    out = {"median_adu": med, "sigma_adu": sigma}
    for k in ks:
        out[f"n_bright_k{k}"] = (x > med + k * sigma).sum(axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--constructs", nargs="+")
    ap.add_argument("--remote", action="store_true", help="read the published run on S3IT")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tcfg, pcfg = load_topic_config(), load_config(local=not args.remote)
    b = tcfg["bright_frames"]
    constructs = args.constructs or tcfg["constructs"]
    run = make_run_dir(pcfg, "bright_frames", topic_root=tcfg["output_root"])
    t0, inputs = time.time(), []
    for where in ("IN", "background"):
        frames = []
        for c in constructs:
            p = trace_path(pcfg, c, "bg_rm", where=where, filtered=False)
            inputs.append(p)
            bg = pd.read_pickle(p)
            res = bright_counts(bg.to_numpy(np.float64), b["k_values"], b["mad_to_sigma"])
            idx = load_index(pcfg, c, where=where).set_index("uniqueID")
            df = pd.DataFrame({"construct": c, "uniqueID": bg.columns, **res})
            df["session"] = idx.loc[df["uniqueID"], "session"].to_numpy()
            df["movie"] = idx.loc[df["uniqueID"], "movie"].to_numpy()
            if df["session"].isna().any():
                raise ValueError(f"{c} {where}: traces without a session")
            frames.append(df)
            logging.info("%s %s: %d traces (%.0f s)", c, where, len(df), time.time() - t0)
        pd.concat(frames).to_csv(run / f"counts_{where}.csv", index=False, float_format="%.4f")
    write_manifest(run, {**tcfg, "paperdata_config": pcfg["_path"], "published_run": pcfg["published_run"],
                         "step0_run": pcfg["step0_run"]}, Path(__file__), inputs,
                   extra={"constructs": constructs, "elapsed_s": round(time.time() - t0)})
    logging.info("done: %s", run)


if __name__ == "__main__":
    main()
