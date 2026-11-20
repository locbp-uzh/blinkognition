#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
How many molecules did the published extraction drop because they touch the movie's first or last frame?

Found 2026-11-20 in R3.1 (Revisions/BlankSurface): Extraction/utils.py get_and_link_locs links each movie's 640 nm
localizations with Picasso's postprocess.link and its default remove_ambiguous_lengths=True, which removes
every linked group whose first frame is 0 (or whose last frame is not before the movie's frame count). With
max_darktime = 4000 frames a linked group is nearly a whole molecule, so a molecule ON in frame 0 is never
extracted. This script quantifies that on the published run, reading its localization files in place.

Per 640 nm movie of the published run (step 0 camera.csv), with the extraction parameters of
Extraction/config.yaml (max_distance, max_darktime, boxsize, overlap_threshold, max_distance_ground_truth,
min_on_ground_truth, max_dist_closest_ground_truth):
  - n_linked       linked groups as the pipeline made them (= the published ROIs; checked per construct and
                   session against step 0 rois.csv);
  - n_linked_all   the same link without the removal; n_dropped = the difference, split into n_dropped_start
                   (first frame 0) and n_dropped_other;
  - dropped groups inside vesicles (IN: the published rule, box corner within 4 px of a vesicle cluster's corner,
    vesicle clusters recomputed from the published 488 nm localizations as extract.py does) and of those the
    single ones (no overlap, Extraction/utils.py get_overlapping_rois over all groups of the movie):
    n_dropped_in, n_dropped_single_in = upper bound of the single IN traces the paper lost (their quality-filter
    fate is unknown without extracting their traces).
Per construct: dropped single IN against the published single IN ROIs (rois.csv) and filtered IN traces.

Outputs (Results/Revisions/PaperData/edge_groups/run_NNN/): movies.csv, summary.csv (per construct),
summary_session.csv, checks.json, manifest.yaml, config.yaml, code/.

Usage (root of the S3IT review clone, compute node, picasso-env):
    python Revisions/PaperData/edge_groups.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "Extraction"))
from paperdata import load_config, make_run_dir, write_manifest  # noqa: E402

FOV_RE = re.compile(r"_(\d{4})_locs\.hdf5$")
N_JOBS = 16


def fov_key(name: str) -> str:
    m = FOV_RE.search(name)
    return m.group(1) if m else "NONE"


def movie_counts(row: dict, ecfg: dict, run: Path) -> dict:
    from picasso import io as pio, lib, postprocess
    from scipy import spatial
    from utils import get_and_cluster_locs, get_overlapping_rois
    folder = run / row["session"] / row["folder"]
    stem = Path(row["file"]).stem
    out = {k: row[k] for k in ("session", "folder", "construct")}
    out["movie"] = stem
    lp = folder / f"{stem}_locs.hdf5"
    if not lp.exists():
        out["error"] = "no 640 nm locs"
        return out
    locs, info = pio.load_locs(str(lp))
    out["n_locs"] = len(locs)
    out["n_frames_locs"] = int(info[0]["Frames"])
    locs = lib.ensure_sanity(locs, info)
    kw = {"r_max": float(ecfg["max_distance"]), "max_dark_time": int(ecfg["max_darktime"])}
    if len(locs) == 0:
        out.update(n_linked=0, n_linked_all=0, n_dropped=0, n_dropped_start=0, n_dropped_other=0,
                   n_dropped_in=0, n_dropped_single_in=0)
        return out
    linked = postprocess.link(locs, info, **kw)
    la = postprocess.link(locs, info, remove_ambiguous_lengths=False, **kw).reset_index(drop=True)
    first = la["frame"].to_numpy()
    last = first + la["len"].to_numpy() - 1
    kept = (first > 0) & (last < out["n_frames_locs"])
    out.update(n_linked=len(linked), n_linked_all=len(la), n_dropped=int((~kept).sum()),
               n_dropped_start=int((first == 0).sum()), n_dropped_other=int((~kept & (first != 0)).sum()),
               kept_rule_matches_picasso=bool(int(kept.sum()) == len(linked)))
    # box corners as get_and_link_locs makes them; overlap over all groups (pipeline order: by group)
    shift = int(ecfg["boxsize"]) / 2
    xt = np.round(la["x"].to_numpy() - shift).astype(int)
    yt = np.round(la["y"].to_numpy() - shift).astype(int)
    ok = np.isfinite(la["x"].to_numpy()) & np.isfinite(la["y"].to_numpy())
    xt, yt = np.where(ok, xt, 0), np.where(ok, yt, 0)
    shape = (230, 230)

    def single_of(idx: np.ndarray) -> np.ndarray:
        """Single (non-overlapping) flags for the groups idx, overlap computed among those groups only."""
        ov, _, _ = get_overlapping_rois(xt[idx], yt[idx], np.zeros(shape), np.zeros(shape),
                                        box_size=int(ecfg["boxsize"]), overlap_threshold=int(ecfg["overlap_threshold"]))
        flags = np.ones(len(idx), bool)
        flags[list(set(ov))] = False
        return flags

    single = single_of(np.arange(len(la)))            # with the dropped groups put back (estimate)
    single_kept = np.zeros(len(la), bool)              # as the pipeline computed it (kept groups only; check)
    single_kept[np.flatnonzero(kept)] = single_of(np.flatnonzero(kept))
    # vesicles from the published 488 nm localizations of the same field
    gts = {fov_key(p.name): p for p in folder.glob(f"{row['folder']}_488nm_*_locs.hdf5") if not p.name.startswith("._")}
    gp = gts.get(fov_key(lp.name))
    is_in = np.zeros(len(la), bool)
    out["n_vesicles"] = np.nan
    if gp is not None:
        try:
            _, _, xg, yg, _, n = get_and_cluster_locs(str(gp), box_size=int(ecfg["boxsize"]),
                                                      max_distance=float(ecfg["max_distance_ground_truth"]),
                                                      min_locs=int(ecfg["min_on_ground_truth"]))
            out["n_vesicles"] = int(n)
            if n:
                tree = spatial.KDTree(np.stack((xg[1:], yg[1:]), axis=1))
                d, _ = tree.query(np.stack((xt, yt), axis=1))
                is_in = d < float(ecfg["max_dist_closest_ground_truth"])
        except Exception as e:  # extract.py labels every trace OUT in that case
            out["vesicle_error"] = f"{type(e).__name__}: {e}"
    else:
        out["vesicle_error"] = "no 488 nm locs"
    drop = ~kept
    out.update(n_dropped_in=int((drop & is_in).sum()), n_dropped_single_in=int((drop & is_in & single).sum()),
               n_kept_single_in_recomputed=int((kept & is_in & single_kept).sum()))
    return out


def main() -> None:
    cfg = load_config()
    with open(REPO_ROOT / "Extraction" / "config.yaml") as f:
        ecfg = yaml.safe_load(f)
    keys = ("max_distance", "max_darktime", "boxsize", "overlap_threshold", "max_distance_ground_truth",
            "min_on_ground_truth", "max_dist_closest_ground_truth")
    ecfg = {k: ecfg[k] for k in keys}
    run = Path(cfg["published_run"])
    cam = pd.read_csv(REPO_ROOT / cfg["step0_run"] / "camera.csv")
    key2name = {v: k for k, v in cfg["constructs"].items()}
    cam["construct"] = cam["folder"].map(key2name)
    rows = cam[["session", "folder", "construct", "file"]].to_dict("records")
    out = make_run_dir(cfg, "edge_groups")
    print(f"{len(rows)} movies", flush=True)
    res = Parallel(n_jobs=N_JOBS)(delayed(movie_counts)(r, ecfg, run) for r in rows)
    mv = pd.DataFrame(res)
    mv.to_csv(out / "movies.csv", index=False)
    rois = pd.read_csv(REPO_ROOT / cfg["step0_run"] / "rois.csv")
    pub = rois.groupby(["construct", "session"])["n_rois"].sum().rename("published_rois")
    mine = mv.groupby(["construct", "session"])["n_linked"].sum().rename("n_linked")
    cmp_ = pd.concat([pub, mine], axis=1).fillna(0)
    agg = {c: "sum" for c in ("n_linked", "n_linked_all", "n_dropped", "n_dropped_start", "n_dropped_other",
                              "n_dropped_in", "n_dropped_single_in", "n_kept_single_in_recomputed")}
    ses = mv.groupby(["construct", "session"]).agg(agg).reset_index()
    single_in = rois[(rois["overlap"] == "single") & (rois["ground_truth"] == "IN")]
    ses = ses.merge(single_in.groupby(["construct", "session"])["n_rois"].sum().rename("published_single_in").reset_index(),
                    on=["construct", "session"], how="left")
    ses.to_csv(out / "summary_session.csv", index=False)
    con = ses.drop(columns="session").groupby("construct").sum().reset_index()
    con["dropped_frac_of_all_groups"] = con["n_dropped"] / con["n_linked_all"]
    con["dropped_single_in_per_published_single_in"] = con["n_dropped_single_in"] / con["published_single_in"]
    con.to_csv(out / "summary.csv", index=False, float_format="%.5g")
    checks = {
        "every_movie_has_locs": bool(mv.get("error", pd.Series(dtype=str)).isna().all()),
        "kept_rule_matches_picasso": bool(mv["kept_rule_matches_picasso"].fillna(True).all()),
        "n_linked_equals_published_rois_per_construct_session": bool((cmp_["published_rois"] == cmp_["n_linked"]).all()),
        "linked_vs_published_mismatches": cmp_[cmp_["published_rois"] != cmp_["n_linked"]].reset_index().to_dict("records"),
        "recomputed_single_in_equals_published": bool((ses["n_kept_single_in_recomputed"] == ses["published_single_in"]).all()),
        "single_in_mismatches": ses.loc[ses["n_kept_single_in_recomputed"] != ses["published_single_in"],
                                        ["construct", "session", "n_kept_single_in_recomputed", "published_single_in"]].to_dict("records"),
        "vesicle_errors": int(mv.get("vesicle_error", pd.Series(dtype=str)).notna().sum()),
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=2, default=str))
    write_manifest(out, cfg, Path(__file__), [REPO_ROOT / cfg["step0_run"] / "camera.csv"], hash_inputs=False,
                   extra={"extraction_parameters": ecfg})
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(con.to_string(index=False))
        print(json.dumps(checks, indent=1, default=str)[:3000])
    print(out)


if __name__ == "__main__":
    main()
