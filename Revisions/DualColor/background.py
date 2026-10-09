#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Filtered background traces of one dual-color dataset, with slide, FOV and class.

The extraction samples n_background_traces (50) background boxes per movie at random pixels
whose box does not overlap a protein ROI box, drops those whose top-left corner lies within
background_buffer (5 px, Euclidean) of a protein ROI's top-left corner, processes them as the
protein traces (5 x 5 box sum, GMM background removal, per-trace min-max) and keeps those
without spikes (robust outlier score (max - median) / MAD on the raw trace <= 6.0,
background_rob_threshold of extraction/config_640only_<ID>.yaml): BackgroundTraces/Filtered/
<key>_background_filtered_minmax.pkl, one column per uniqueID of UniqueIDs/
<key>_uniqueID_background.pkl.

With 5 x 5 boxes the 5 px buffer still keeps boxes that touch a protein box, and those show
the neighbor's blinking (review 2026-10-09). Here a background box is kept only if a gap of
at least min_gap_px (3) pixels separates it from every protein ROI box of the same movie
(UniqueIDs/<key>_uniqueID_all.pkl, single and overlapping ROIs): Chebyshev distance between
top-left corners >= boxsize + min_gap_px.

Each trace is mapped to its movie through the staged link map and to its (slide, FOV)
through the vesicles run recorded in the dataset's trace_assignment manifest, the same
table that gave the protein traces their FOVs, using the same extraction run.

Class: the slide's protein on the single-protein slides (HT on ATTO390 slides, SNAP on
ATTO520 slides); none on the mixed slides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dc import REPO_ROOT, load_config, run_dir

# Nothing from Revisions/DFK785 or Revisions/Bleedthrough is imported here: this module is used
# by train_pure.py next to ML/, and Bleedthrough/common.py imports Extraction/utils.py under the
# name `utils` (and puts Bleedthrough, with its own classify.py and evaluate.py, first on
# sys.path), which would shadow ML/utils.py. The two pieces needed are restated:
STR_COLUMNS = {"slide": str, "dye": str, "fov": str, "detected_by": str}   # Bleedthrough/common.py
DYE_CLASS = {"ATTO390": "HT", "ATTO520": "SNAP"}
MIN_GAP_PX = 3


def load_link_map(path: Path) -> dict[str, str]:
    """'<Slide>/<link name>' -> raw ND2 path (as Revisions/DFK785/assign_traces.load_link_map)."""
    m = pd.read_csv(path, sep="\t", header=None, names=["link", "raw"])
    return dict(zip(m["link"], m["raw"]))


def movie_link(key: str, origin: str, suffix: str) -> str:
    """'<key>/<movie>.nd2' from a per-movie trace file path ending in suffix."""
    return f"{key}/" + Path(origin).name.replace(suffix, ".nd2")


def gap_to_protein(bg: pd.DataFrame, rois: pd.DataFrame) -> np.ndarray:
    """Per background box: Chebyshev distance (px) between its top-left corner and the nearest
    protein ROI top-left corner of the same movie (inf if the movie has no ROI)."""
    out = np.full(len(bg), np.inf)
    by_movie = {m: g[["x", "y"]].to_numpy(float) for m, g in rois.groupby("movie")}
    for m, idx in bg.groupby("movie").indices.items():
        r = by_movie.get(m)
        if r is None or not len(r):
            continue
        p = bg[["x", "y"]].to_numpy(float)[idx]
        out[idx] = np.abs(p[:, None, :] - r[None, :, :]).max(axis=2).min(axis=1)
    return out


def load_background(ds: str, trace_assignment_run: str | None = None, min_gap_px: int = MIN_GAP_PX
                    ) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """(meta, X (N, 1, T) float32, info) for every kept filtered background trace of dataset ds."""
    cfg = load_config(ds)
    ta = run_dir(cfg, "trace_assignment", trace_assignment_run)
    with open(ta / "manifest.yaml") as f:
        ta_man = yaml.safe_load(f)
    ext_run, ves_run = REPO_ROOT / ta_man["extraction_run"], REPO_ROOT / ta_man["vesicles_run"]
    tc = cfg["trace_assignment"]
    box = int(tc["boxsize"])
    link_map = REPO_ROOT / tc.get("link_map", f"Inputs/DualColor_640only/{ds}/link_map.tsv")
    links = load_link_map(link_map)
    fovs = pd.read_csv(ves_run / "fovs.csv", dtype=STR_COLUMNS)
    fov_of = {r.file_640: (r.slide, r.fov) for r in fovs.itertuples()}

    metas, Xs, counts = [], [], {}
    for slide, sc in cfg["slides"].items():
        key = sc["key"]
        u = pd.read_pickle(ext_run / "UniqueIDs" / f"{key}_uniqueID_background.pkl").drop(columns=["trace"])
        filt = pd.read_pickle(ext_run / "BackgroundTraces" / "Filtered" / f"{key}_background_filtered_minmax.pkl")
        if not set(filt.columns) <= set(u["uniqueID"]):
            raise ValueError(f"{ds} {key}: filtered background columns that are not background uniqueIDs")
        u = u[u["uniqueID"].isin(set(filt.columns))].reset_index(drop=True)
        u["movie"] = u["file_origin"].map(lambda o: movie_link(key, o, "_background_traces.pkl"))
        rois = pd.read_pickle(ext_run / "UniqueIDs" / f"{key}_uniqueID_all.pkl").drop(columns=["trace"])
        rois["movie"] = rois["origin"].map(lambda o: movie_link(key, o, "_traces.pkl"))
        if not set(rois["movie"]) <= set(links):
            raise ValueError(f"{ds} {key}: protein ROIs from movies not in the link map")
        u["gap_px"] = gap_to_protein(u, rois) - box          # free pixels between the two boxes
        n_filtered = len(u)
        u = u[u["gap_px"] >= min_gap_px].reset_index(drop=True)
        counts[key] = {"filtered": n_filtered, "kept": len(u)}
        u["movie_raw"] = u["movie"].map(links)
        if u["movie_raw"].isna().any():
            raise ValueError(f"{ds} {key}: {int(u['movie_raw'].isna().sum())} background traces from movies not "
                             f"in the link map")
        sf = u["movie_raw"].map(fov_of)
        if sf.isna().any():
            raise ValueError(f"{ds} {key}: {int(sf.isna().sum())} background traces from movies without a vesicle FOV")
        u["slide"], u["fov"] = sf.map(lambda t: t[0]), sf.map(lambda t: t[1])
        if (u["slide"] != slide).any():
            raise ValueError(f"{ds} {key}: movies paired with FOVs of another slide")
        u["extraction_key"] = key
        metas.append(u[["extraction_key", "slide", "fov", "uniqueID", "x", "y", "gap_px", "movie_raw"]])
        Xs.append(filt[u["uniqueID"].tolist()].to_numpy(dtype=np.float32).T[:, None, :])

    meta = pd.concat(metas, ignore_index=True)
    X = np.concatenate(Xs, axis=0)
    dye = meta["slide"].map(lambda s: cfg["slides"][s]["dye"])
    meta["dataset"] = ds
    meta["slide_type"] = np.where(dye == "mix", "mixed", "pure")
    meta["true_class"] = dye.map(DYE_CLASS)               # NaN on the mixed slides
    meta["trace_source"] = "background"
    rel = lambda p: str(Path(p).relative_to(REPO_ROOT))   # noqa: E731
    return meta, X, {"dataset": ds, "trace_assignment_run": rel(ta), "extraction_run": rel(ext_run),
                     "vesicles_run": rel(ves_run), "link_map": rel(link_map), "min_gap_px": min_gap_px,
                     "traces": counts}
