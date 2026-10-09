#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Assign the extracted protein traces of one dual-color dataset to labeled vesicles.

As Revisions/DFK785/assign_traces.py (whose functions are reused: artifact and duplicate
drop, nearest-two search, classes, neighbor-FOV chance estimate), with three differences:

- The match radius depends on the slide: trace_assignment.match_radius_px.pure on the
  single-protein slides (training data; no vesicle of the other protein exists there) and
  .mix on the mixed slides (test data).
- The 640 offset of each vesicle detection channel is estimated from the data instead of
  the occupancy analysis: protein positions (ROI top-left + boxsize / 2) are paired with
  their nearest vesicle within registration.search_px, the median (dy, dx) per detection
  channel is applied, and the estimate is repeated with the pairing radius
  registration.refine_px until it changes by less than 0.01 px.
- The link map of the staged inputs is relative to the dataset's input_root.

    IN <label>   nearest vesicle within the slide's radius and no second one
    ambiguous    two or more vesicles within the radius
    OUT          otherwise

Inputs: the extraction run (Results/Revisions/DualColor/<ID>/Extraction/<run>, from Daint),
the staged link map, the latest unmixing run (labels) and its vesicles run (fovs).
Outputs (<output_root>/trace_assignment/run_NNN/): traces.csv, counts.csv, null.csv
(chance coincidences per slide at null_radii_px), registration.csv, residual_offset.csv,
dropped.csv, manifest.

Usage:
    python Revisions/DualColor/assign_traces.py DFK788 [--extraction-run NAME] [--data-root DIR]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc import REPO_ROOT, load_config, load_dfk785, match_radius, run_dir, write_manifest  # noqa: E402

a785 = load_dfk785("assign_traces")
from common import STR_COLUMNS, make_run_dir, rel_to_repo, setup_logging  # noqa: E402


def extraction_run(cfg: dict, name: str | None) -> Path:
    base = REPO_ROOT / cfg["output_root"] / "Extraction"
    name = name or (cfg.get("runs") or {}).get("extraction")
    if name:
        return base / name
    runs = sorted(p for p in base.iterdir() if p.is_dir() and (p / "UniqueIDs").is_dir())
    if len(runs) != 1:
        raise ValueError(f"{len(runs)} extraction runs under {base}; name one with --extraction-run or runs.extraction")
    return runs[0]


def load_slide(cfg: dict, slide: str, ext_run: Path, links: dict, fov_of: dict, half_box: float
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = cfg["slides"][slide]["key"]
    u = pd.read_pickle(ext_run / "UniqueIDs" / f"{key}_uniqueID_all.pkl")
    u = u[u["overlap"] == "single"].drop(columns=["trace"])
    u, dropped = a785.drop_artifacts(u)
    filt = pd.read_pickle(ext_run / "ProteinTraces" / "Filtered" / f"{key}_filtered_minmax.pkl")
    u["filtered"] = u["uniqueID"].isin(set(filt.columns))
    link = u["origin"].map(lambda o: f"{key}/" + Path(o).name.replace("_traces.pkl", ".nd2"))
    u["movie_raw"] = link.map(links)
    if u["movie_raw"].isna().any():
        raise ValueError(f"{key}: {int(u['movie_raw'].isna().sum())} traces from movies not in the link map")
    sf = u["movie_raw"].map(fov_of)
    if sf.isna().any():
        raise ValueError(f"{key}: {int(sf.isna().sum())} traces from movies without a vesicle FOV")
    u["slide"], u["fov"] = sf.map(lambda t: t[0]), sf.map(lambda t: t[1])
    if (u["slide"] != slide).any():
        raise ValueError(f"{key}: movies paired with FOVs of another slide")
    u["row"], u["col"] = u["y"] + half_box, u["x"] + half_box
    u["extraction_key"] = key
    dropped = dropped.assign(extraction_key=key, filtered=dropped["uniqueID"].isin(set(filt.columns)))
    return u, dropped


def estimate_offsets(u: pd.DataFrame, ves_by_fov: dict, rc: dict) -> tuple[dict, pd.DataFrame]:
    """Per detection channel: median protein - vesicle offset of nearest pairs, iterated."""
    channels = sorted({c for v in ves_by_fov.values() for c in v["detected_by"].unique()})
    offs = {c: np.zeros(2) for c in channels}
    hist = []
    for it in range(int(rc["max_iter"])):
        radius = rc["search_px"] if it == 0 else rc["refine_px"]
        pairs = []
        for (slide, fov), g in u.groupby(["slide", "fov"]):
            v = ves_by_fov.get((slide, fov))
            if v is None:
                continue
            vpos = a785.vesicle_positions(v, offs)
            d1, _, j = a785.nearest_two(g[["row", "col"]].to_numpy(), vpos)
            ok = d1 <= radius
            pairs.append(pd.DataFrame({"ch": v["detected_by"].to_numpy()[j[ok]],
                                       "dy": g["row"].to_numpy()[ok] - vpos[j[ok], 0],
                                       "dx": g["col"].to_numpy()[ok] - vpos[j[ok], 1]}))
        p = pd.concat(pairs, ignore_index=True)
        step = 0.0
        for c, gc in p.groupby("ch"):
            delta = np.array([gc["dy"].median(), gc["dx"].median()])
            offs[c] = offs[c] + delta
            step = max(step, float(np.abs(delta).max()))
            hist.append({"iteration": it, "radius_px": radius, "detected_by": c, "n_pairs": len(gc),
                         "applied_dy": offs[c][0], "applied_dx": offs[c][1], "step_px": float(np.abs(delta).max())})
        if it > 0 and step < 0.01:
            break
    return offs, pd.DataFrame(hist)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--extraction-run", default=None)
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.dataset, a.data_root)
    tc = cfg["trace_assignment"]

    ext_run = extraction_run(cfg, a.extraction_run)
    link_map = REPO_ROOT / tc.get("link_map", f"Inputs/DualColor_640only/{cfg['dataset']}/link_map.tsv")
    links = a785.load_link_map(link_map)
    um_run = run_dir(cfg, "unmixing")
    with open(um_run / "manifest.yaml") as f:
        ves_run = REPO_ROOT / yaml.safe_load(f)["vesicles_run"]
    ves = pd.read_csv(um_run / "labels.csv", dtype=STR_COLUMNS)
    fovs = pd.read_csv(ves_run / "fovs.csv", dtype=STR_COLUMNS)
    fov_of = {r.file_640: (r.slide, r.fov) for r in fovs.itertuples()}
    ves_by_fov = {k: g for k, g in ves.groupby(["slide", "fov"])}
    half_box = float(tc["boxsize"]) / 2

    loaded = [load_slide(cfg, s, ext_run, links, fov_of, half_box) for s in cfg["slides"]]
    u = pd.concat([x[0] for x in loaded], ignore_index=True)
    dropped = pd.concat([x[1] for x in loaded], ignore_index=True)

    offs, reg = estimate_offsets(u, ves_by_fov, tc["registration"])
    logging.info("Registration (vesicle -> 640 frame, px):\n" + reg.groupby("detected_by").tail(1).to_string(index=False))
    parts = [a785.assign(g, ves_by_fov, offs, match_radius(cfg, s)).assign(radius_px=match_radius(cfg, s))
             for s, g in u.groupby("slide")]
    tr = pd.concat(parts, ignore_index=True)

    run = make_run_dir(cfg, "trace_assignment")
    cols = ["extraction_key", "slide", "fov", "uniqueID", "x", "y", "row", "col", "filtered", "radius_px", "d1", "d2",
            "dy", "dx", "vesicle_id", "vesicle_label", "vesicle_detected_by", "class", "movie_raw"]
    tr[cols].to_csv(run / "traces.csv", index=False)
    dropped[["extraction_key", "uniqueID", "x", "y", "origin", "filtered", "reason"]].to_csv(run / "dropped.csv",
                                                                                             index=False)
    reg.to_csv(run / "registration.csv", index=False)
    counts = pd.concat([pd.crosstab(tr["slide"], tr["class"]).assign(set="single ROIs"),
                        pd.crosstab(tr.loc[tr.filtered, "slide"], tr.loc[tr.filtered, "class"]).assign(set="filtered")])
    counts = counts.reindex(columns=a785.CLASSES + ["set"], fill_value=0).reset_index()
    counts.to_csv(run / "counts.csv", index=False)
    nt = a785.null_table(tr, ves_by_fov, fovs, offs, [float(r) for r in tc["null_radii_px"]])
    nt.to_csv(run / "null.csv", index=False)
    res = (tr[tr["class"].str.startswith("IN_")].groupby("vesicle_detected_by")[["dy", "dx"]]
           .agg(["median", "count"]).reset_index())
    res.columns = ["detected_by", "dy_median", "n", "dx_median", "n2"]
    res.drop(columns="n2").to_csv(run / "residual_offset.csv", index=False)
    with pd.option_context("display.width", 200):
        logging.info(f"Dropped: {dropped['reason'].value_counts().to_dict()} "
                     f"({int(dropped['filtered'].sum())} of them filtered)")
        logging.info("Counts:\n" + counts.to_string(index=False))
        logging.info("Residual offset protein - vesicle (px):\n" + res.to_string(index=False))
    write_manifest(run, cfg, Path(__file__), [um_run / "labels.csv", ves_run / "fovs.csv", link_map],
                   extra={"extraction_run": rel_to_repo(ext_run), "unmixing_run": rel_to_repo(um_run),
                          "vesicles_run": rel_to_repo(ves_run)},
                   modules=[HERE.parent / "DFK785" / "assign_traces.py"])


if __name__ == "__main__":
    main()
