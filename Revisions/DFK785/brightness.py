#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Compare HMSiR brightness per localization between the paper acquisitions and DFK785.

The ND2 metadata records the laser setting (90 %) but not the power, and the data
readme says the 640 laser ran at 12 mW, weaker than usual. This puts a number on the
difference with the paper's own localization settings (Extraction/config.yaml camera
parameters, box 5, MLE, one fixed gradient for every movie):

    paper   one 640 movie per protein and experiment from AllMovies, localized here
    DFK785  the 640 locs of the paper-pipeline extraction runs of slides 1 and 2

Per movie and pooled per set: localizations, photons (median, 90th percentile),
background per pixel, PSF width sx and localization precision lpx, frames <
max_frame only (the paper movies have 8000 frames, DFK785 and the traces 6000).

A fixed gradient drops the dimmest events, so in dimmer data the surviving median
is biased upward: the ratios are upper bounds on the relative brightness.

Outputs (Results/Revisions/DFK785/brightness/run_NNN/):
    paper/                 links to the paper movies and their Picasso locs
    per_movie.csv          one row per movie
    summary.csv            one row per set, with ratios to the paper set of the same protein
    manifest.yaml, config.yaml, code/

Usage:
    python Revisions/DFK785/brightness.py [--config path] [--reuse-locs DIR]

--reuse-locs copies existing <link name>_locs.hdf5/.yaml files from DIR instead of
localizing again; their Picasso parameters are checked against the config.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Bleedthrough"))
from common import REPO_ROOT, load_config, make_run_dir, rel_to_repo, setup_logging, write_manifest  # noqa: E402
from utils import get_picasso_python  # noqa: E402  (Extraction/, on the path through common)
from localize import run_picasso_localize  # noqa: E402

PICASSO_KEYS = {"Box Size": "boxsize", "Min. Net Gradient": "gradient", "Baseline": "baseline",
                "Sensitivity": "sensitivity", "Gain": "gain", "Fit method": "method"}


def read_locs(path: Path, max_frame: int) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        locs = pd.DataFrame(f["locs"][...])
    return locs[locs["frame"] < max_frame]


def check_picasso_params(locs_yaml: Path, bc: dict) -> None:
    """The localization yaml must match the configured parameters (first document = localize)."""
    with open(locs_yaml) as f:
        docs = [d for d in yaml.load_all(f, Loader=yaml.FullLoader) if d]   # Picasso writes python/tuple tags
    loc = next(d for d in docs if "Min. Net Gradient" in d)
    want = {**bc["camera"], "gradient": bc["gradient"]}
    for key, name in PICASSO_KEYS.items():
        got, exp = loc[key], want[name]
        if str(got).lower() != str(exp).lower() and not (isinstance(exp, (int, float)) and float(got) == float(exp)):
            raise ValueError(f"{locs_yaml.name}: {key} = {got}, config has {exp}")


def localize_paper(run: Path, bc: dict, reuse: Path | None) -> list[dict]:
    """Link every paper movie into run/paper/ and localize it (or copy reused locs)."""
    out = run / "paper"
    out.mkdir()
    cam = bc["camera"]
    cfg_loc = {"boxsize": cam["boxsize"], "drift": 0, "baseline": cam["baseline"],
               "sensitivity": cam["sensitivity"], "gain": cam["gain"], "quantum_efficiency": cam["quantum_efficiency"]}
    python = None if reuse else get_picasso_python()
    movies = []
    for protein, files in bc["paper_movies"].items():
        for rel in files:
            raw = Path(bc["paper_root"]) / rel
            experiment, folder = Path(rel).parts[:2]
            link = out / f"{experiment}_{folder}_{Path(rel).stem.rsplit('_', 1)[-1]}.nd2"
            link.symlink_to(raw)
            if reuse:
                for ext in ("_locs.hdf5", "_locs.yaml"):
                    shutil.copy2(reuse / f"{link.stem}{ext}", out / f"{link.stem}{ext}")
            else:
                logging.info(f"Localizing {rel}")
                run_picasso_localize(link, bc["gradient"], cam["method"], cfg_loc, python)
            check_picasso_params(out / f"{link.stem}_locs.yaml", bc)
            movies.append({"set": f"paper {protein}", "protein": protein, "movie": rel,
                           "locs": out / f"{link.stem}_locs.hdf5"})
    return movies


def dfk785_movies(bc: dict) -> list[dict]:
    movies = []
    for name, d in bc["dfk785_sets"].items():
        files = sorted(p for p in (REPO_ROOT / d["run"]).rglob("*640nm*_locs.hdf5") if not p.name.startswith("._"))
        if not files:
            raise FileNotFoundError(f"No 640 locs under {d['run']}")
        for p in files:
            check_picasso_params(p.with_suffix(".yaml"), bc)
            movies.append({"set": name, "protein": d["protein"], "movie": p.name.replace("_locs.hdf5", ".nd2"),
                           "locs": p})
    return movies


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--reuse-locs", type=Path, default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    bc = cfg["brightness"]
    mf = int(bc["max_frame"])

    run = make_run_dir(cfg, "brightness")
    movies = localize_paper(run, bc, args.reuse_locs) + dfk785_movies(bc)

    per, pooled = [], {}
    for m in movies:
        locs = read_locs(m["locs"], mf)
        per.append({"set": m["set"], "protein": m["protein"], "movie": m["movie"], "n_locs": len(locs),
                    "photons_median": locs["photons"].median(), "bg_median": locs["bg"].median(),
                    "sx_median": locs["sx"].median(), "lpx_median": locs["lpx"].median()})
        pooled.setdefault(m["set"], []).append(locs[["photons", "bg"]])
    per = pd.DataFrame(per)
    per.to_csv(run / "per_movie.csv", index=False)

    rows = []
    for name, parts in pooled.items():
        p, a = per[per["set"] == name], pd.concat(parts)
        rows.append({"set": name, "protein": p["protein"].iloc[0], "movies": len(p),
                     "locs_per_movie_median": p["n_locs"].median(),
                     "photons_median": a["photons"].median(), "photons_p90": a["photons"].quantile(0.9),
                     "bg_median": a["bg"].median(), "sx_median": p["sx_median"].median(),
                     "lpx_median": p["lpx_median"].median(),
                     "movie_photons_min": p["photons_median"].min(), "movie_photons_max": p["photons_median"].max()})
    summ = pd.DataFrame(rows)
    ref = summ[summ["set"].str.startswith("paper")].set_index("protein")
    summ["photons_ratio_to_paper"] = summ["photons_median"] / summ["protein"].map(ref["photons_median"])
    summ["bg_ratio_to_paper"] = summ["bg_median"] / summ["protein"].map(ref["bg_median"])
    summ.to_csv(run / "summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3g}".format):
        logging.info("Summary:\n" + summ.to_string(index=False))

    write_manifest(run, cfg, Path(__file__), [m["locs"] for m in movies],
                   extra={"reuse_locs": str(args.reuse_locs) if args.reuse_locs else None,
                          "dfk785_runs": {k: v["run"] for k, v in bc["dfk785_sets"].items()},
                          "results": rel_to_repo(run)})


if __name__ == "__main__":
    main()
