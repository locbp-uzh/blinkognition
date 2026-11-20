#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
R3.1 counts: blank coverslips and empty vesicles through the paper's extraction (design confirmed by the user
2026-11-20; descriptive, no tests).

Reads the six BlankSurface extraction runs (vesicle channel 488 / 405 nm x protein gradient 10000 / 20000 /
40000; for each, the newest run folder under Results/Revisions/BlankSurface/Extraction_gt<ch>_g<gradient>/
staged from config.yaml's stage_root) and counts, per field of view (FOV, one 640 nm movie):
  - n_locs        Picasso localizations at 640 nm (the run folder's _locs.hdf5). A movie without any is
                  skipped by Extraction/extract.py ("No localizations", FileLists/failed_file_list.pkl): it
                  counts here as a FOV with zero of everything, not as missing;
  - n_vesicles    vesicle-channel clusters, recomputed exactly as extract.py does (Extraction/utils.py
                  get_and_cluster_locs with the run's max_distance_ground_truth and min_on_ground_truth);
                  in_area_fraction = share of the FOV's pixels within max_dist_closest_ground_truth (4 px) of a
                  vesicle center, the area where a spot is labeled IN;
  - n_rois, n_single, n_overlap, n_single_in, n_single_out  from the movie's _traces.pkl;
  - n_pass_in, n_pass_out  single traces passing the paper's quality filter (the pipeline's Filtered/);
  - camera and laser settings from the ND2 headers (Revisions/PaperData/paperdata.py nd2_camera).
Per trace (every single IN and OUT trace that reaches the filter): the filter replay of R3.3b
(Revisions/PhotonsFrames/filter_audit.py audit_trace, published filter parameters): the criterion that removes
it, the GMM separation, the number of ON events and the median detected photons per event (16.0 / 300 photons
per ADU, Picasso's convention as everywhere in the revision). The replay must reproduce the pipeline's
filtered set trace by trace (checks.json).

References for the per-FOV numbers:
  - the same-power protein slide of the same vesicle channel and gradient (DFK788 SNAP520 for 488 nm, HT390
    for 405 nm);
  - the paper: passing IN traces per 640 nm movie of each construct in the published extraction (step 0 index
    and camera.csv of Revisions/PaperData, check run_003; movies without passing traces count as zero).
summary.csv gives per slide and run the FOV means; reference.csv the ratios of passing IN traces per FOV on the
blank and empty-vesicle slides to those references. On blank slides (no vesicles) every spot is OUT, so their
passing OUT traces per FOV, all counted as IN, are an upper bound for surface emitters inside vesicles;
expected_in uses the same-power protein slide's in_area_fraction instead.

Outputs (Results/Revisions/BlankSurface/count/run_NNN/): fov.csv, traces.csv, summary.csv, reference.csv,
checks.json, manifest.yaml, config.yaml, code/.

Usage (root of the S3IT review clone, on a compute node, picasso-env):
    python Revisions/BlankSurface/count.py
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "PaperData"))
sys.path.insert(0, str(HERE.parent / "PhotonsFrames"))
sys.path.insert(0, str(REPO_ROOT / "Extraction"))
from paperdata import load_config as load_paper_config, make_run_dir, nd2_camera, to_photons, write_manifest  # noqa: E402
from events import load_topic_config  # noqa: E402
from filter_audit import audit_trace  # noqa: E402

FOV_RE = re.compile(r"_(\d{4})\.nd2$")
CH640 = "_640nm_"
N_JOBS = 16


def fov_of(name: str) -> str:
    m = FOV_RE.search(name)
    return m.group(1) if m else "0000"


def find_runs(cfg: dict) -> list[dict]:
    """Newest run folder per (vesicle channel, gradient) staged from cfg['stage_root']."""
    runs = []
    for gt in sorted({s["gt"] for s in cfg["slides"]}):
        for grad in cfg["gradient_protein"]:
            base = REPO_ROOT / cfg["output_root"] / f"Extraction_gt{gt}_g{grad}"
            cands = []
            for d in sorted(base.iterdir()) if base.exists() else []:
                bc = d / "blanksurface_config.yaml"
                if d.name.startswith("._") or not bc.exists():
                    continue
                with open(bc) as f:
                    rc = yaml.safe_load(f)
                if str(rc["input_folder"]).startswith(str(cfg["stage_root"])):
                    cands.append((d, rc))
            if not cands:
                raise SystemExit(f"no run under {base} staged from {cfg['stage_root']}")
            d, rc = cands[-1]
            runs.append({"gt": gt, "gradient": grad, "run": d, "run_cfg": rc, "n_candidates": len(cands)})
    return runs


def vesicles(locs_path: Path, rc: dict, shape: tuple[int, int]) -> tuple[int, float, str]:
    """(clusters, share of pixels within the IN distance of a cluster center, error) as in extract.py."""
    from scipy.ndimage import distance_transform_edt
    from utils import get_and_cluster_locs
    try:
        _, _, xt, yt, _, n = get_and_cluster_locs(str(locs_path), box_size=int(rc["boxsize"]),
                                                  max_distance=float(rc.get("max_distance_ground_truth", 2.5)),
                                                  min_locs=int(rc.get("min_on_ground_truth", 3)))
    except Exception as e:  # extract.py: "Failed to process ground truth" -> every trace OUT
        return 0, 0.0, f"{type(e).__name__}: {e}"
    if n == 0:
        return 0, 0.0, ""
    # extract.py compares the trace's box corner (x, y) with the cluster's box corner (xt, yt); the same
    # translation applies to both, so distances between centers are the same.
    occ = np.zeros(shape, bool)
    yy = np.clip(np.round(np.asarray(yt[1:], float)).astype(int), 0, shape[0] - 1)
    xx = np.clip(np.round(np.asarray(xt[1:], float)).astype(int), 0, shape[1] - 1)
    occ[yy, xx] = True
    d = distance_transform_edt(~occ)
    return int(n), float((d < float(rc.get("max_dist_closest_ground_truth", 4.0))).mean()), ""


def count_fov(row: pd.Series, run: dict, failed: set[str], gt_links: dict, shape: tuple[int, int]) -> dict:
    import h5py
    R, rc = run["run"], run["run_cfg"]
    folder = R / row["session"] / row["slide"]
    stem = Path(row["link"]).stem
    out = {"gt": run["gt"], "gradient": run["gradient"], "session": row["session"], "slide": row["slide"],
           "fov": fov_of(row["link"]), "movie": stem, "skipped_no_locs": row["link"] in failed}
    lp = folder / f"{stem}_locs.hdf5"
    if lp.exists():
        with h5py.File(lp, "r") as h:
            out["n_locs"] = int(len(h["locs"]))
    else:
        out["n_locs"] = np.nan
    gl = gt_links.get((row["session"], row["slide"], out["fov"]))
    glp = folder / f"{Path(gl).stem}_locs.hdf5" if gl else None
    if glp is not None and glp.exists():
        out["n_vesicles"], out["in_area_fraction"], out["vesicle_error"] = vesicles(glp, rc, shape)
    else:
        out["n_vesicles"], out["in_area_fraction"], out["vesicle_error"] = np.nan, np.nan, "no vesicle locs"
    tp = folder / f"{stem}_traces.pkl"
    if tp.exists():
        t = pd.read_pickle(tp)
        single = t["overlap"] == "single"
        out.update(n_rois=len(t), n_single=int(single.sum()), n_overlap=int((~single).sum()),
                   n_single_in=int((single & (t["ground_truth"] == "IN")).sum()),
                   n_single_out=int((single & (t["ground_truth"] == "OUT")).sum()))
    else:
        out.update(n_rois=0, n_single=0, n_overlap=0, n_single_in=0, n_single_out=0)
    cam = nd2_camera(Path(row["target"]))
    out.update({k: cam.get(k) for k in ("GainMultiplier", "ExposureTime", "laser_640nm_pct_median", "n_frames",
                                        "dt_median_ms", "date")})
    return out


def audit_slide(run: dict, slide: str, filt: dict, pcfg: dict) -> pd.DataFrame:
    R = run["run"]
    rows = []
    for where in ("IN", "OUT"):
        uid = pd.read_pickle(R / "UniqueIDs" / f"{slide}_uniqueID_{where}.pkl")
        if uid.empty:
            continue
        uid = uid.drop(columns=[c for c in ("trace",) if c in uid.columns])
        allbg = pd.read_pickle(R / f"ProteinTraces{where}" / "AllBG" / f"{slide}_{where}_bg_rm.pkl")
        fl = pd.read_pickle(R / f"ProteinTraces{where}" / "Filtered" / f"{slide}_{where}_filtered_bg_rm.pkl")
        passed = set(fl.columns) if not fl.empty else set()
        cols = list(allbg.columns) if not allbg.empty else []
        res = Parallel(n_jobs=N_JOBS)(delayed(audit_trace)(allbg[c].to_numpy(float), filt) for c in cols)
        audit = dict(zip(cols, res))
        for _, u in uid.iterrows():
            o = Path(str(u["origin"]))
            a = audit.get(u["uniqueID"])
            r = {"gt": run["gt"], "gradient": run["gradient"], "session": o.parent.parent.name, "slide": slide,
                 "where": where, "uniqueID": int(u["uniqueID"]), "movie": o.name[: -len("_traces.pkl")],
                 "x": int(u["x"]), "y": int(u["y"]), "reached_filter": a is not None,
                 "passed_pipeline": u["uniqueID"] in passed}
            r["fov"] = fov_of(r["movie"] + ".nd2")
            if a is not None:
                r.update(a)
                r["median_photons_per_event_free"] = float(to_photons(a["median_adu_per_event_free"], pcfg))
                r.pop("median_adu_per_event_free")
            rows.append(r)
    return pd.DataFrame(rows)


def paper_reference(pcfg: dict) -> pd.DataFrame:
    """Passing IN traces per 640 nm movie and construct in the published extraction (step 0)."""
    s0 = REPO_ROOT / pcfg["step0_run"]
    cam = pd.read_csv(s0 / "camera.csv")
    cam["movie"] = [Path(f).stem for f in cam["file"]]
    key2name = {v: k for k, v in pcfg["constructs"].items()}
    out = []
    for name in pcfg["constructs"]:
        idx = pd.read_csv(s0 / f"index_IN_{name}.csv")
        npass = idx[idx["passed_filter"]].groupby(["session", "folder", "movie"]).size()
        nin = idx.groupby(["session", "folder", "movie"]).size()
        mv = cam[cam["folder"].map(key2name) == name][["session", "folder", "movie"]]
        k = list(zip(mv["session"], mv["folder"], mv["movie"]))
        p = np.array([npass.get(x, 0) for x in k], float)
        i = np.array([nin.get(x, 0) for x in k], float)
        out.append({"construct": name, "n_movies": len(k), "pass_in_per_fov_mean": p.mean(),
                    "pass_in_per_fov_median": float(np.median(p)), "single_in_per_fov_mean": i.mean()})
    return pd.DataFrame(out)


def summarize(fov: pd.DataFrame) -> pd.DataFrame:
    g = fov.groupby(["gt", "gradient", "session", "slide", "role"], sort=False)
    s = g.agg(n_fov=("fov", "size"), n_fov_no_locs=("skipped_no_locs", "sum"),
              locs_per_fov_mean=("n_locs", "mean"), locs_per_fov_median=("n_locs", "median"),
              locs_per_fov_max=("n_locs", "max"), vesicles_per_fov_mean=("n_vesicles", "mean"),
              in_area_fraction_mean=("in_area_fraction", "mean"), rois_per_fov_mean=("n_rois", "mean"),
              single_per_fov_mean=("n_single", "mean"), single_in_per_fov_mean=("n_single_in", "mean"),
              single_out_per_fov_mean=("n_single_out", "mean"), pass_in_total=("n_pass_in", "sum"),
              pass_out_total=("n_pass_out", "sum"), pass_in_per_fov_mean=("n_pass_in", "mean"),
              pass_out_per_fov_mean=("n_pass_out", "mean"), fov_with_pass_in=("n_pass_in", lambda v: int((v > 0).sum())),
              fov_with_pass_any=("n_pass_any", lambda v: int((v > 0).sum())),
              laser_640nm_pct=("laser_640nm_pct_median", "median"), em_gain=("GainMultiplier", "first"),
              exposure_ms=("ExposureTime", "first"), dt_median_ms=("dt_median_ms", "median"))
    return s.reset_index()


def reference_table(summ: pd.DataFrame, paper: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for _, r in summ[summ["role"] != "protein"].iterrows():
        prot = summ[(summ["gt"] == r["gt"]) & (summ["gradient"] == r["gradient"]) & (summ["role"] == "protein")]
        if len(prot) != 1:
            raise SystemExit(f"expected one same-power protein slide for gt {r['gt']} gradient {r['gradient']}")
        p = prot.iloc[0]
        # blank slides: every spot is OUT; all passing traces counted as IN (upper bound), or scaled by the
        # same-power protein slide's IN area fraction (expected)
        n_as_in = r["pass_in_per_fov_mean"] if r["role"] == "empty_vesicles" else r["pass_out_per_fov_mean"] + r["pass_in_per_fov_mean"]
        expected = (r["pass_in_per_fov_mean"] if r["role"] == "empty_vesicles"
                    else r["pass_in_per_fov_mean"] + r["pass_out_per_fov_mean"] * p["in_area_fraction_mean"])
        row = {"gt": r["gt"], "gradient": r["gradient"], "session": r["session"], "slide": r["slide"],
               "role": r["role"], "impurity_in_per_fov_upper": n_as_in, "impurity_in_per_fov_expected": expected,
               "protein_slide": f"{p['session']} {p['slide']}", "protein_pass_in_per_fov": p["pass_in_per_fov_mean"],
               "ratio_to_protein_slide_upper": n_as_in / p["pass_in_per_fov_mean"],
               "ratio_to_protein_slide_expected": expected / p["pass_in_per_fov_mean"]}
        for _, q in paper.iterrows():
            row[f"ratio_to_paper_{q['construct']}_upper"] = n_as_in / q["pass_in_per_fov_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    with open(HERE / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(HERE / "config.yaml")
    pcfg = load_paper_config()
    filt = load_topic_config()["filter"]
    roles = {(s["session"], s["name"]): s["role"] for s in cfg["slides"]}
    runs = find_runs(cfg)
    out = make_run_dir(pcfg, "count", topic_root="Results/Revisions/BlankSurface")
    fovs, traces, checks, inputs = [], [], {}, []
    for run in runs:
        R, tag = run["run"], f"gt{run['gt']}_g{run['gradient']}"
        print(f"{tag}: {R.name} ({run['n_candidates']} candidate runs)", flush=True)
        lm = pd.read_csv(R / "link_map.tsv", sep="\t")
        with open(R / "FileLists" / "failed_file_list.pkl", "rb") as fh:
            failed = {Path(p).name for p in pickle.load(fh)}
        gtok = f"_{run['gt']}nm_"
        gt_links = {(r["session"], r["slide"], fov_of(r["link"])): r["link"] for _, r in lm.iterrows() if gtok in r["link"]}
        m640 = lm[lm["link"].str.contains(CH640)]
        shape = (230, 230)
        rows = Parallel(n_jobs=N_JOBS)(delayed(count_fov)(r, run, failed, gt_links, shape) for _, r in m640.iterrows())
        fov = pd.DataFrame(rows)
        fov["role"] = [roles[(s, n)] for s, n in zip(fov["session"], fov["slide"])]
        tr = pd.concat([audit_slide(run, s, filt, pcfg) for s in sorted(set(m640["slide"]))], ignore_index=True)
        if not tr.empty:
            key = ["session", "slide", "movie"]
            for where in ("IN", "OUT"):
                n = tr[(tr["where"] == where) & tr["passed_pipeline"]].groupby(key).size().rename(f"n_pass_{where.lower()}")
                fov = fov.merge(n, left_on=key, right_index=True, how="left")
        for c in ("n_pass_in", "n_pass_out"):
            fov[c] = fov[c].fillna(0).astype(int) if c in fov else 0
        fov["n_pass_any"] = fov["n_pass_in"] + fov["n_pass_out"]
        # checks
        ck = {
            "every_640_movie_has_locs": bool(fov["n_locs"].notna().all()),
            "skipped_movies_have_zero_locs": bool((fov.loc[fov["skipped_no_locs"], "n_locs"] == 0).all()),
            "zero_locs_movies_are_skipped": bool((fov.loc[fov["n_locs"] == 0, "skipped_no_locs"]).all()),
            "failed_list_only_640_movies_of_this_run": failed <= set(m640["link"]),
            "n_skipped": int(fov["skipped_no_locs"].sum()),
            "single_in_matches_uniqueids": int(fov["n_single_in"].sum()) == int((tr["where"] == "IN").sum()) if not tr.empty else int(fov["n_single_in"].sum()) == 0,
            "single_out_matches_uniqueids": int(fov["n_single_out"].sum()) == int((tr["where"] == "OUT").sum()) if not tr.empty else int(fov["n_single_out"].sum()) == 0,
            "every_slide_has_traces_combined": {s: int((tr["slide"] == s).sum()) if not tr.empty else 0 for s in sorted(set(m640["slide"]))},
        }
        if not tr.empty:
            reached = tr[tr["reached_filter"]]
            ck["replay_equals_pipeline"] = bool(((reached["criterion"] == "passed") == reached["passed_pipeline"]).all())
            ck["replay_mismatches"] = int(((reached["criterion"] == "passed") != reached["passed_pipeline"]).sum())
            ck["dropped_before_filter_zero_or_nan_frames"] = int((~tr["reached_filter"]).sum())
            ck["passed_but_not_reached"] = int((tr["passed_pipeline"] & ~tr["reached_filter"]).sum())
        rc = run["run_cfg"]
        ck["filter_params_match_run"] = all(
            float(rc.get(k, 3.0 if k == "gmm_min_separation" else np.nan)) == float(v) for k, v in filt.items())
        expected_vesicle_free = fov["role"] == "blank"
        ck["blank_fovs_with_vesicles"] = int((fov.loc[expected_vesicle_free, "n_vesicles"] > 0).sum())
        ck["vesicle_errors"] = int((fov["vesicle_error"].fillna("") != "").sum())
        checks[tag] = ck
        print(json.dumps(ck), flush=True)
        fovs.append(fov)
        traces.append(tr)
        inputs += [R / "link_map.tsv", R / "blanksurface_config.yaml"]
    fov = pd.concat(fovs, ignore_index=True)
    tr = pd.concat(traces, ignore_index=True)
    fov.to_csv(out / "fov.csv", index=False, float_format="%.6g")
    tr.to_csv(out / "traces.csv", index=False, float_format="%.6g")
    summ = summarize(fov)
    summ.to_csv(out / "summary.csv", index=False, float_format="%.5g")
    paper = paper_reference(pcfg)
    paper.to_csv(out / "paper_reference.csv", index=False, float_format="%.5g")
    ref = reference_table(summ, paper, cfg)
    ref.to_csv(out / "reference.csv", index=False, float_format="%.4g")
    ok = all(v for c in checks.values() for k, v in c.items()
             if isinstance(v, bool) and k not in ("passed_but_not_reached",))
    checks["all_boolean_checks_pass"] = ok
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    write_manifest(out, cfg, Path(__file__), inputs, hash_inputs=False,
                   extra={"runs": {f"gt{r['gt']}_g{r['gradient']}": str(r["run"]) for r in runs},
                          "filter": filt, "paper_config": pcfg["_path"], "paper_step0_run": pcfg["step0_run"]})
    import shutil
    for py in (HERE.parent / "PhotonsFrames" / "filter_audit.py", HERE.parent / "PhotonsFrames" / "events.py"):
        shutil.copy(py, out / "code" / py.name)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summ.to_string(index=False))
        print(ref.to_string(index=False))
        print(paper.to_string(index=False))
    print("all boolean checks pass:", ok)
    print(out)


if __name__ == "__main__":
    main()
