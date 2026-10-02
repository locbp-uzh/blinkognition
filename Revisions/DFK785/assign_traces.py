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

    protein position = ROI top-left corner + boxsize / 2           (640 frame)
    vesicle position = vesicle centroid + 640 offset of the channel it was
                       detected in (occupancy registration_summary.csv)
    IN <label>   if the nearest vesicle is within match_radius_px and no other one is
    ambiguous    if two or more vesicles are within match_radius_px
    OUT          otherwise

The extraction sets top-left = round(x - boxsize / 2) on Picasso coordinates, which,
like the vesicle centroids, put pixel centers on integers; + boxsize / 2 is therefore
the unbiased protein position (+ boxsize // 2 sits 0.5 px up-left on both axes).

The vesicle label is the unmixing label: ATTO390 (HT7 protein), ATTO520 (SNAP
protein), dual or no label. Movies are matched to vesicle FOVs through the link map
of the staged inputs and the time-paired FOV table of the vesicle run, so the
shifted file numbering of slide 3 does not matter.

Dropped before assignment (dropped.csv): ROIs at (0, 0) or with a negative corner
(extraction artifacts with empty traces) and repeats of an identical box in the same
movie (the extraction labels such pairs 'single').

Chance coincidences (null.csv): the same ROIs are assigned to the vesicles of the
previous and next FOV of the slide (acquisition order), at every radius in
null_radii_px. Model: a fraction f of ROIs is in a vesicle and always near one; the
others lie independently of the vesicle pattern and are near one with the null
probability q. Observed near fraction = f + (1 - f) q, so f = (obs - q) / (1 - q), and
the expected chance hits of a class are (1 - f) x its null count. 'Near' is IN or
ambiguous. Filtered traces only.

Outputs (Results/Revisions/DFK785/trace_assignment/run_NNN/):
    traces.csv            one row per single ROI: ids, position, filtered, d1/d2 to the
                          nearest vesicles, vesicle id and label, class
    counts.csv            slide x class, all single ROIs and filtered ones
    null.csv              slide x radius x class: observed, null, chance estimate (filtered)
    residual_offset.csv   median protein - vesicle offset of IN pairs per detection channel
    dropped.csv           artifact and duplicate ROIs left out
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


def vesicle_positions(v: pd.DataFrame, offs: dict) -> np.ndarray:
    """Vesicle centroids in the 640 frame."""
    off = np.stack([offs.get(c, np.zeros(2)) for c in v["detected_by"]]) if len(v) else np.zeros((0, 2))
    return v[["y_px", "x_px"]].to_numpy() + off


def nearest_two(pts: np.ndarray, vpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance to the nearest and second-nearest vesicle (inf if none), and the nearest one's index."""
    n = len(pts)
    if len(vpos) == 1:
        d, j = cKDTree(vpos).query(pts, k=1)
        return d, np.full(n, np.inf), j
    d, j = cKDTree(vpos).query(pts, k=2)
    return d[:, 0], d[:, 1], j[:, 0]


def classify(d1: np.ndarray, d2: np.ndarray, label: np.ndarray, radius: float) -> np.ndarray:
    inside = d1 <= radius
    cls = np.where(inside, np.char.add("IN_", label.astype(str)), "OUT").astype(object)
    cls[inside & (d2 <= radius)] = "ambiguous"
    return cls


def drop_artifacts(u: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove ROIs at (0, 0) or with a negative corner, and repeats of a box within a movie."""
    art = (u["x"] < 0) | (u["y"] < 0) | ((u["x"] == 0) & (u["y"] == 0))
    dup = u.duplicated(subset=["origin", "x", "y"], keep="first") & ~art
    dropped = u[art | dup].assign(reason=np.where(art[art | dup], "artifact corner", "duplicate box"))
    return u[~(art | dup)], dropped


def load_slide(key: str, ext_run: Path, links: dict, raw_root: Path, fov_of: dict, half_box: float
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single ROIs of one extraction key with slide, FOV, filtered flag and position in the 640 frame."""
    u = pd.read_pickle(ext_run / "UniqueIDs" / f"{key}_uniqueID_all.pkl")
    u = u[u["overlap"] == "single"].drop(columns=["trace"])
    u, dropped = drop_artifacts(u)
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
    u["extraction_key"] = key
    dropped = dropped.assign(extraction_key=key, filtered=dropped["uniqueID"].isin(set(filt.columns)))
    return u, dropped


def assign(u: pd.DataFrame, ves_by_fov: dict, offs: dict, radius: float) -> pd.DataFrame:
    out = []
    for (slide, fov), g in u.groupby(["slide", "fov"]):
        v = ves_by_fov.get((slide, fov))
        g = g.copy()
        if v is None:
            g["d1"] = g["d2"] = np.inf
            g["dy"] = g["dx"] = np.nan
            g["vesicle_id"], g["vesicle_label"], g["vesicle_detected_by"] = -1, "", ""
        else:
            vpos = vesicle_positions(v, offs)
            g["d1"], g["d2"], j = nearest_two(g[["row", "col"]].to_numpy(), vpos)
            g["vesicle_id"] = v["vesicle_id"].to_numpy()[j]
            g["vesicle_label"] = v["label"].to_numpy()[j]
            g["vesicle_detected_by"] = v["detected_by"].to_numpy()[j]
            g["dy"] = g["row"] - vpos[j, 0]
            g["dx"] = g["col"] - vpos[j, 1]
        g["class"] = classify(g["d1"].to_numpy(), g["d2"].to_numpy(), g["vesicle_label"].to_numpy(), radius)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def null_table(tr: pd.DataFrame, ves_by_fov: dict, fovs: pd.DataFrame, offs: dict, radii: list[float]) -> pd.DataFrame:
    """Observed vs neighbor-FOV class counts of the filtered traces, per slide and radius."""
    rows = []
    f = tr[tr["filtered"]]
    for slide, fs in f.groupby("slide"):
        order = fovs[fovs["slide"] == slide].sort_values("acq_order")["fov"].tolist()
        obs = {r: pd.Series(classify(fs["d1"].to_numpy(), fs["d2"].to_numpy(), fs["vesicle_label"].to_numpy(), r))
               .value_counts() for r in radii}
        null = {r: pd.Series(0.0, index=CLASSES) for r in radii}
        for fov, g in fs.groupby("fov"):
            i = order.index(fov)
            nbrs = [order[k] for k in (i - 1, i + 1) if 0 <= k < len(order) and (slide, order[k]) in ves_by_fov]
            for nb in nbrs:
                v = ves_by_fov[(slide, nb)]
                d1, d2, j = nearest_two(g[["row", "col"]].to_numpy(), vesicle_positions(v, offs))
                lab = v["label"].to_numpy()[j]
                for r in radii:
                    null[r] = null[r].add(pd.Series(classify(d1, d2, lab, r)).value_counts() / len(nbrs), fill_value=0)
        n = len(fs)
        for r in radii:
            o = obs[r].reindex(CLASSES, fill_value=0)
            q_near = (n - null[r].get("OUT", 0)) / n
            o_near = (n - o["OUT"]) / n
            fr = (o_near - q_near) / (1 - q_near)
            for c in CLASSES[:-1]:
                chance = (1 - fr) * null[r].get(c, 0)
                rows.append({"slide": slide, "radius_px": r, "class": c, "n_filtered": n, "observed": int(o[c]),
                             "null": round(float(null[r].get(c, 0)), 2), "near_obs": round(o_near, 4),
                             "near_null": round(q_near, 4), "f": round(fr, 4), "chance_est": round(chance, 2),
                             "chance_share": round(chance / o[c], 3) if o[c] else np.nan})
    return pd.DataFrame(rows)


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
    half_box = float(tc["boxsize"]) / 2
    ves_by_fov = {k: g for k, g in ves.groupby(["slide", "fov"])}

    loaded = [load_slide(k, ext_run, links, Path(cfg["input_root"]), fov_of, half_box) for k in tc["extraction_keys"]]
    u = pd.concat([x[0] for x in loaded], ignore_index=True)
    dropped = pd.concat([x[1] for x in loaded], ignore_index=True)
    tr = assign(u, ves_by_fov, offs, tc["match_radius_px"])

    run = make_run_dir(cfg, "trace_assignment")
    cols = ["extraction_key", "slide", "fov", "uniqueID", "x", "y", "row", "col", "filtered", "d1", "d2", "dy", "dx",
            "vesicle_id", "vesicle_label", "vesicle_detected_by", "class", "movie_raw"]
    tr[cols].to_csv(run / "traces.csv", index=False)
    dropped[["extraction_key", "uniqueID", "x", "y", "origin", "filtered", "reason"]].to_csv(run / "dropped.csv",
                                                                                             index=False)

    counts = pd.concat([pd.crosstab(tr["slide"], tr["class"]).assign(set="single ROIs"),
                        pd.crosstab(tr.loc[tr.filtered, "slide"], tr.loc[tr.filtered, "class"]).assign(set="filtered")])
    counts = counts.reindex(columns=CLASSES + ["set"], fill_value=0).reset_index()
    counts.to_csv(run / "counts.csv", index=False)
    nt = null_table(tr, ves_by_fov, fovs, offs, [float(r) for r in tc["null_radii_px"]])
    nt.to_csv(run / "null.csv", index=False)
    res = (tr[tr["class"].str.startswith("IN_")].groupby("vesicle_detected_by")[["dy", "dx"]]
           .agg(["median", "count"]).reset_index())
    res.columns = ["detected_by", "dy_median", "n", "dx_median", "n2"]
    res.drop(columns="n2").to_csv(run / "residual_offset.csv", index=False)
    with pd.option_context("display.width", 200):
        logging.info(f"Dropped: {dropped['reason'].value_counts().to_dict()} "
                     f"({int(dropped['filtered'].sum())} of them filtered)")
        logging.info("Counts:\n" + counts.to_string(index=False))
        main_cls = nt[nt["class"].isin(["IN_ATTO390", "IN_ATTO520"])]
        logging.info("Chance coincidences (filtered):\n" + main_cls.to_string(index=False))
        logging.info("Residual offset protein - vesicle (px):\n" + res.to_string(index=False))
    write_manifest(run, cfg, Path(__file__),
                   [um_run / "labels.csv", ves_run / "fovs.csv", occ_run / "registration_summary.csv",
                    REPO_ROOT / tc["link_map"]],
                   extra={"extraction_run": tc["extraction_run"], "unmixing_run": rel_to_repo(um_run),
                          "registration_from": rel_to_repo(occ_run)})


if __name__ == "__main__":
    main()
