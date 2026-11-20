#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Step 0 of the first-round review: check the published extraction before anything is computed from it,
and write the per-trace index the other topics use. No statistics.

Per construct (IN traces):
  - the four filtered forms (raw, bg_rm, minmax, zscored) have identical columns and frames;
  - min-max and z-score of bg_rm reproduce the stored normalized traces;
  - raw minus bg_rm is one constant per trace (the GMM background);
  - the filtered minmax and zscored pickles have the published md5 (Inputs/FinalTraces, Zenodo);
  - filtered traces are a subset of the unfiltered ones, with unchanged bg_rm values;
  - the unfiltered traces are exactly the UniqueIDs IN rows; each trace's session, movie and box position.
OUT and background traces (unless --skip-out-bg): unfiltered and filtered counts and index; UniqueIDs_all
gives the ROI counts by overlap and IN/OUT label (the start of the trace funnel).
ND2 headers of every 640 nm movie (unless --no-nd2): EM gain, readout indices, exposure, frame count,
frame interval.

Outputs (Results/Revisions/PaperData/check/run_NNN/): checks.json, funnel.csv, sessions.csv, rois.csv,
index_<where>_<construct>.csv (uniqueID, session, folder, movie, x, y, passed_filter), camera.csv,
manifest.yaml, config.yaml, code/.

Usage (repo root; on S3IT through Revisions/PaperData/s3it/check_data.sbatch):
    python Revisions/PaperData/check_data.py [--constructs HaloK117 ...] [--skip-out-bg] [--no-nd2]
           [--run-folder <other extraction, for tests>]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paperdata import (KINDS, SESSION_RE, ids_path, load_config, load_ids, load_traces, make_run_dir,  # noqa: E402
                       md5, nd2_camera, trace_path, write_manifest)

TOL = 1e-6   # float64 traces; normalizations recomputed in float64


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs(a - b)
    return float(np.nanmax(d)) if d.size else 0.0


def check_filtered_in(cfg: dict, c: str, run: Path, inputs: list) -> tuple[dict, pd.DataFrame]:
    """The checks on one construct's IN traces; returns (results, index of all IN traces)."""
    r: dict = {"construct": c, "key": cfg["constructs"][c]}
    mats = {}
    for k in KINDS:
        p = trace_path(cfg, c, k)
        inputs.append(p)
        mats[k] = load_traces(cfg, c, k)
    ref = mats["minmax"]
    r["filtered_n_traces"], r["n_frames"] = int(ref.shape[1]), int(ref.shape[0])
    r["dtypes"] = sorted({str(m.dtypes.iloc[0]) for m in mats.values()})
    r["forms_same_columns_and_frames"] = all(m.columns.equals(ref.columns) and m.index.equals(ref.index)
                                             for m in mats.values())
    bg = mats["bg_rm"].to_numpy(np.float64)
    mn, mx = bg.min(axis=0), bg.max(axis=0)
    r["minmax_from_bg_rm_max_abs_diff"] = _max_abs((bg - mn) / (mx - mn), mats["minmax"].to_numpy(np.float64))
    r["zscored_from_bg_rm_max_abs_diff"] = _max_abs((bg - bg.mean(axis=0)) / bg.std(axis=0),
                                                    mats["zscored"].to_numpy(np.float64))
    d = mats["raw"].to_numpy(np.float64) - bg
    r["raw_minus_bg_rm_max_spread_within_trace"] = float(np.max(d.max(axis=0) - d.min(axis=0)))
    for k in ("minmax", "zscored"):
        r[f"md5_{k}_matches_published"] = md5(trace_path(cfg, c, k)) == cfg["published_md5"][c][k]
    del mats["raw"], mats["minmax"], mats["zscored"], d

    allbg_p = trace_path(cfg, c, "bg_rm", filtered=False)
    inputs.append(allbg_p)
    allbg = pd.read_pickle(allbg_p)
    r["unfiltered_n_traces"] = int(allbg.shape[1])
    cols = mats["bg_rm"].columns
    r["filtered_subset_of_unfiltered"] = bool(cols.isin(allbg.columns).all())
    if r["filtered_subset_of_unfiltered"]:
        r["filtered_bg_rm_vs_unfiltered_max_abs_diff"] = _max_abs(allbg[cols].to_numpy(np.float64), bg)
    del allbg, bg, mats

    inputs.append(ids_path(cfg, c, "IN"))
    ids = load_ids(cfg, c, "IN")
    r["ids_rows"] = int(len(ids))
    r["ids_uniqueID_unique"] = bool(ids["uniqueID"].is_unique)
    r["ids_overlap_values"] = ids["overlap"].value_counts().to_dict()
    r["ids_ground_truth_values"] = ids["ground_truth"].value_counts().to_dict()
    r["ids_equal_unfiltered_columns"] = r["ids_rows"] == r["unfiltered_n_traces"]
    r["ids_sessions_parsed"] = bool((ids["session"] != "").all())
    idx = ids[["uniqueID", "session", "folder", "movie", "x", "y"]].copy()
    idx["passed_filter"] = idx["uniqueID"].isin(cols)
    r["filtered_found_in_ids"] = int(idx["passed_filter"].sum()) == r["filtered_n_traces"]
    idx.to_csv(run / f"index_IN_{c}.csv", index=False)
    return r, idx


def count_where(cfg: dict, c: str, where: str, run: Path, inputs: list) -> pd.DataFrame:
    """Index of OUT or background traces with passed_filter, from UniqueIDs and the filtered columns."""
    fp = trace_path(cfg, c, "bg_rm", where=where)
    inputs.extend([fp, ids_path(cfg, c, where)])
    cols = pd.read_pickle(fp).columns
    ids = load_ids(cfg, c, where)
    idx = ids[["uniqueID", "session", "folder", "movie", "x", "y"]].copy()
    idx["passed_filter"] = idx["uniqueID"].isin(cols)
    if int(idx["passed_filter"].sum()) != len(cols):
        logging.warning("%s %s: %d filtered columns, %d found in UniqueIDs", c, where, len(cols),
                        int(idx["passed_filter"].sum()))
    idx.to_csv(run / f"index_{where}_{c}.csv", index=False)
    return idx


def roi_counts(cfg: dict, c: str, inputs: list) -> pd.DataFrame:
    """ROIs by session, overlap and IN/OUT label (UniqueIDs_all), before any trace filtering."""
    p = ids_path(cfg, c, "all")
    inputs.append(p)
    u = load_ids(cfg, c, "all")
    out = u.groupby(["session", "overlap", "ground_truth"]).size().rename("n_rois").reset_index()
    out.insert(0, "construct", c)
    return out


def scan_nd2(cfg: dict) -> pd.DataFrame:
    """Headers of every 640 nm movie of the constructs' folders in raw_root."""
    keys = set(cfg["constructs"].values())
    rows = []
    for sess in sorted(p for p in Path(cfg["raw_root"]).iterdir() if p.is_dir() and SESSION_RE.fullmatch(p.name)):
        for folder in sorted(p for p in sess.iterdir() if p.is_dir() and p.name in keys):
            for mov in sorted(folder.glob("*_640nm_*.nd2")):
                if mov.name.startswith("._"):
                    continue
                try:
                    rows.append({"session": sess.name, "folder": folder.name, **nd2_camera(mov)})
                except Exception as e:  # noqa: BLE001  (record and continue)
                    rows.append({"session": sess.name, "folder": folder.name, "file": str(mov), "error": repr(e)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--run-folder", help="another extraction with the same layout (tests only)")
    ap.add_argument("--constructs", nargs="+")
    ap.add_argument("--skip-out-bg", action="store_true")
    ap.add_argument("--no-nd2", action="store_true")
    ap.add_argument("--no-hash", action="store_true", help="skip input sha256 in the manifest (tests)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config, args.run_folder)
    constructs = args.constructs or list(cfg["constructs"])
    run = make_run_dir(cfg, "check")
    logging.info("run %s, published run %s", run, cfg["published_run"])
    t0 = time.time()

    inputs: list[Path] = []
    results, funnel, sessions, rois = [], [], [], []
    for c in constructs:
        logging.info("%s: IN", c)
        r, idx = check_filtered_in(cfg, c, run, inputs)
        results.append(r)
        groups = {"IN": idx}
        if not args.skip_out_bg:
            for where in ("OUT", "background"):
                logging.info("%s: %s", c, where)
                groups[where] = count_where(cfg, c, where, run, inputs)
            rois.append(roi_counts(cfg, c, inputs))
        for where, g in groups.items():
            funnel.append({"construct": c, "where": where, "n_unfiltered": len(g),
                           "n_filtered": int(g["passed_filter"].sum())})
            s = g.groupby("session")["passed_filter"].agg(n_unfiltered="size", n_filtered="sum").reset_index()
            s.insert(0, "where", where)
            s.insert(0, "construct", c)
            sessions.append(s)
        logging.info("%s done (%.0f s)", c, time.time() - t0)

    funnel_df = pd.DataFrame(funnel)
    funnel_df["pct_filtered"] = 100.0 * funnel_df["n_filtered"] / funnel_df["n_unfiltered"]
    funnel_df.to_csv(run / "funnel.csv", index=False)
    pd.concat(sessions).to_csv(run / "sessions.csv", index=False)
    if rois:
        pd.concat(rois).to_csv(run / "rois.csv", index=False)

    if not args.no_nd2:
        logging.info("ND2 headers")
        cam = scan_nd2(cfg)
        cam.to_csv(run / "camera.csv", index=False)
        logging.info("%d movies read", len(cam))

    bool_checks = [k for k in results[0] if isinstance(results[0][k], bool)]
    num_checks = [k for k in results[0] if k.endswith(("_max_abs_diff", "_max_spread_within_trace"))]
    all_ok = all(r[k] for r in results for k in bool_checks) and \
        all(r.get(k, 0.0) < TOL for r in results for k in num_checks)
    with open(run / "checks.json", "w") as f:
        json.dump({"all_ok": all_ok, "tolerance": TOL, "constructs": results}, f, indent=2, default=str)
    logging.info("all checks passed: %s", all_ok)

    write_manifest(run, cfg, Path(__file__), inputs, extra={"constructs": constructs,
                   "skip_out_bg": args.skip_out_bg, "nd2": not args.no_nd2,
                   "elapsed_s": round(time.time() - t0)}, hash_inputs=not args.no_hash)
    logging.info("done: %s (%.0f s)", run, time.time() - t0)


if __name__ == "__main__":
    main()
