#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Assign the protein traces of the paper's extraction pipeline to labeled vesicles.

The 640 movies are processed by Extraction/run_pipeline.py in single-channel mode
(config extraction/config_640only.yaml), so the traces are exactly those the
classifier was trained on (Picasso localization, linking, 5x5 ROI, GMM background,
minmax, the five filtering criteria). Instead of the paper's single ground-truth
channel, each trace is assigned to a vesicle of the DFK785 vesicle table (unmix.py:
per-channel detection, bleed-through unmixing):

    protein position = ROI top-left corner + boxsize // 2          (640 frame)
    vesicle position = vesicle centroid + 640 offset of the channel it was
                       detected in (occupancy registration_summary.csv)
    IN <label>   if the nearest vesicle is within match_radius_px and no other one is
    ambiguous    if two or more vesicles are within match_radius_px
    OUT          otherwise

The vesicle label is the unmixing label: ATTO390 (HT7 protein), ATTO520 (SNAP
protein), dual or no label. Movies are matched to vesicle FOVs through the link map
of the staged inputs and the time-paired FOV table of the vesicle run, so the
shifted file numbering of slide 3 does not matter.

Outputs (Results/Revisions/DFK785/trace_assignment/run_NNN/):
    traces.csv            one row per single (non-overlapping) ROI: ids, position, filtered,
                          d1/d2 to the nearest vesicles, vesicle id and label, class
    counts.csv            slide x class, all single ROIs and filtered ones
    residual_offset.csv   median protein - vesicle offset of IN pairs per detection channel
    manifest.yaml, config.yaml, code/

Usage:
    python Revisions/DFK785/assign_traces.py [--config path]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Bleedthrough"))
from common import (REPO_ROOT, STR_COLUMNS, load_config, make_run_dir, rel_to_repo, resolve_run,  # noqa: E402
                    setup_logging, write_manifest)

CLASSES = ["IN_ATTO390", "IN_ATTO520", "IN_dual", "IN_no label", "ambiguous", "OUT"]


def load_link_map(path: Path) -> dict[str, str]:
    """'<Slide>/<link name>' -> raw ND2 path."""
    m = pd.read_csv(path, sep="\t", header=None, names=["link", "raw"])
    return dict(zip(m["link"], m["raw"]))


def offsets_by_channel(reg_summary: Path) -> dict[str, np.ndarray]:
    rs = pd.read_csv(reg_summary, dtype=STR_COLUMNS)
    rs = rs[rs["slide"] == "all"]
    return {r.detected_by: np.array([r.applied_dy, r.applied_dx]) for r in rs.itertuples()}


def assign_slide(key: str, ext_run: Path, links: dict, raw_root: Path, fov_of: dict, ves: pd.DataFrame,
                 offs: dict, radius: float, half_box: int) -> pd.DataFrame:
    u = pd.read_pickle(ext_run / "UniqueIDs" / f"{key}_uniqueID_all.pkl")
    u = u[u["overlap"] == "single"].drop(columns=["trace"])
    filt = pd.read_pickle(ext_run / "ProteinTraces" / "Filtered" / f"{key}_filtered_minmax.pkl")
    u["filtered"] = u["uniqueID"].isin(set(filt.columns))
    movie = u["origin"].map(lambda o: Path(o).name.replace("_traces.pkl", ".nd2"))
    raw = movie.map(lambda m: links[f"{key}/{m}"])
    rel = raw.map(lambda r: str(Path(r).relative_to(raw_root)))
    sf = rel.map(fov_of)
    if sf.isna().any():
        raise ValueError(f"{key}: {int(sf.isna().sum())} traces from movies without a vesicle FOV")
    u["slide"], u["fov"] = sf.map(lambda t: t[0]), sf.map(lambda t: t[1])
    u["movie_raw"] = raw
    u["row"], u["col"] = u["y"] + half_box, u["x"] + half_box

    out = []
    for (slide, fov), g in u.groupby(["slide", "fov"]):
        v = ves[(ves["slide"] == slide) & (ves["fov"] == fov)]
        g = g.copy()
        if v.empty:
            g["d1"] = g["d2"] = np.inf
            g["dy"] = g["dx"] = np.nan
            g["vesicle_id"], g["vesicle_label"], g["vesicle_detected_by"] = -1, "", ""
        else:
            off = np.stack([offs.get(c, np.zeros(2)) for c in v["detected_by"]])
            vpos = v[["y_px", "x_px"]].to_numpy() + off
            k = min(2, len(v))
            d, j = cKDTree(vpos).query(g[["row", "col"]].to_numpy(), k=k)
            d, j = (d[:, None], j[:, None]) if k == 1 else (d, j)
            g["d1"] = d[:, 0]
            g["d2"] = d[:, 1] if k == 2 else np.inf
            g["vesicle_id"] = v["vesicle_id"].to_numpy()[j[:, 0]]
            g["vesicle_label"] = v["label"].to_numpy()[j[:, 0]]
            g["vesicle_detected_by"] = v["detected_by"].to_numpy()[j[:, 0]]
            g["dy"] = g["row"] - vpos[j[:, 0], 0]
            g["dx"] = g["col"] - vpos[j[:, 0], 1]
        out.append(g)
    g = pd.concat(out, ignore_index=True)
    inside = g["d1"] <= radius
    amb = inside & (g["d2"] <= radius)
    g["class"] = "OUT"
    g.loc[inside, "class"] = "IN_" + g.loc[inside, "vesicle_label"]
    g.loc[amb, "class"] = "ambiguous"
    g["extraction_key"] = key
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    tc = cfg["trace_assignment"]

    ext_run = REPO_ROOT / tc["extraction_run"]
    links = load_link_map(REPO_ROOT / tc["link_map"])
    um_run = resolve_run(cfg, "unmixing", tc["unmixing_run"])
    occ_run = resolve_run(cfg, "occupancy", tc["registration_from_occupancy_run"])
    with open(um_run / "manifest.yaml") as f:
        ves_run = resolve_run(cfg, "vesicles", yaml.safe_load(f)["vesicles_run"])
    ves = pd.read_csv(um_run / "labels.csv", dtype=STR_COLUMNS)
    fovs = pd.read_csv(ves_run / "fovs.csv", dtype=STR_COLUMNS)
    fov_of = {r.file_640: (r.slide, r.fov) for r in fovs.itertuples()}
    offs = offsets_by_channel(occ_run / "registration_summary.csv")
    half_box = int(tc["boxsize"]) // 2

    parts = [assign_slide(k, ext_run, links, Path(cfg["input_root"]), fov_of, ves, offs, tc["match_radius_px"],
                          half_box) for k in tc["extraction_keys"]]
    tr = pd.concat(parts, ignore_index=True)
    run = make_run_dir(cfg, "trace_assignment")
    cols = ["extraction_key", "slide", "fov", "uniqueID", "x", "y", "row", "col", "filtered", "d1", "d2", "dy", "dx",
            "vesicle_id", "vesicle_label", "vesicle_detected_by", "class", "movie_raw"]
    tr[cols].to_csv(run / "traces.csv", index=False)

    counts = pd.concat([pd.crosstab(tr["slide"], tr["class"]).assign(set="single ROIs"),
                        pd.crosstab(tr.loc[tr.filtered, "slide"], tr.loc[tr.filtered, "class"]).assign(set="filtered")])
    counts = counts.reindex(columns=CLASSES + ["set"], fill_value=0).reset_index()
    counts.to_csv(run / "counts.csv", index=False)
    res = (tr[tr["class"].str.startswith("IN_")].groupby("vesicle_detected_by")[["dy", "dx"]]
           .agg(["median", "count"]).reset_index())
    res.columns = ["detected_by", "dy_median", "n", "dx_median", "n2"]
    res.drop(columns="n2").to_csv(run / "residual_offset.csv", index=False)
    with pd.option_context("display.width", 200):
        logging.info("Counts:\n" + counts.to_string(index=False))
        logging.info("Residual offset protein - vesicle (px):\n" + res.to_string(index=False))
    write_manifest(run, cfg, Path(__file__), [um_run / "labels.csv", REPO_ROOT / tc["link_map"]],
                   extra={"extraction_run": tc["extraction_run"], "unmixing_run": rel_to_repo(um_run),
                          "registration_from": rel_to_repo(occ_run)})


if __name__ == "__main__":
    main()
