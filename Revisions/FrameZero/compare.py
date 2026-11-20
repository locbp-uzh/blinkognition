#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
FrameZero: compare the rerun that keeps molecules ON in the first frame with the published extraction.

Every ROI of both runs is identified by (session, folder, movie, x, y) (box corner; linking forms the same
groups in both runs, only the removal of groups starting in frame 0 differs). Per construct:
  - matched ROIs: in both runs. Checks: same overlap and IN/OUT status unless a recovered neighbor now
    overlaps it; for single IN ROIs in both, raw and background-removed traces identical to the published ones
    and the same quality-filter fate (the rerun must reproduce the paper wherever nothing changed);
  - recovered ROIs: only in the rerun (the molecules the published run dropped); single IN ones and how many
    pass the quality filter;
  - lost ROIs: single IN in the published run but not in the rerun (now overlapping a recovered neighbor);
    how many of them had passed the filter.
For the recovered single IN traces that pass the filter: their filtered pickles (bg_rm, minmax, zscored; columns
= rerun uniqueIDs) under recovered/ProteinTracesIN/Filtered/<construct>_IN_filtered_<kind>.pkl (paper construct
names, for ML/classify.py), and their ON events (Revisions/PhotonsFrames/events.py trace_events, photons at
0.062 per ADU, the same definition as events run_002 of the published traces).

Outputs (Results/Revisions/FrameZero/compare/run_NNN/): rois.csv (every single IN ROI of either run with its
status and filter fate), summary.csv (per construct), summary_session.csv, recovered/ (pickles),
recovered_molecules.csv, checks.json, manifest.yaml, config.yaml, code/.

Usage (root of the S3IT review clone, compute node, picasso-env):
    python Revisions/FrameZero/compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "PaperData"))
sys.path.insert(0, str(HERE.parent / "PhotonsFrames"))
from paperdata import load_config, make_run_dir, parse_origin, to_photons, write_manifest  # noqa: E402
from events import load_topic_config, trace_events  # noqa: E402

KEY = ["session", "folder", "movie", "x", "y"]


def ids(run: Path, key: str, where: str) -> pd.DataFrame:
    u = pd.read_pickle(run / "UniqueIDs" / f"{key}_uniqueID_{where}.pkl")
    u = u.drop(columns=[c for c in ("trace",) if c in u.columns])
    parsed = [parse_origin(o) for o in u["origin"]]
    u["session"] = [s for s, _, _ in parsed]
    u["folder"] = [f for _, f, _ in parsed]
    u["movie"] = [m for _, _, m in parsed]
    return u[["uniqueID", *KEY, "overlap", "ground_truth"]]


def kinds(run: Path, key: str, filtered: bool) -> dict[str, Path]:
    base = run / "ProteinTracesIN"
    if filtered:
        return {k: base / "Filtered" / f"{key}_IN_filtered_{k}.pkl" for k in ("raw", "bg_rm", "minmax", "zscored")}
    return {"raw": base / "AllRaw" / f"{key}_IN_raw.pkl", "bg_rm": base / "AllBG" / f"{key}_IN_bg_rm.pkl"}


def main() -> None:
    with open(HERE / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(HERE / "config.yaml")
    pcfg = load_config()
    tcfg = load_topic_config()
    pub = Path(cfg["published_run"])
    new = Path(cfg["run_root"]) / cfg["run_name"]
    out = make_run_dir(pcfg, "compare", topic_root=cfg["output_root"])
    rec_dir = out / "recovered" / "ProteinTracesIN" / "Filtered"
    rec_dir.mkdir(parents=True)
    checks, summ, summ_ses, rows_all, mols = {}, [], [], [], []
    for name, key in pcfg["constructs"].items():
        print(f"{name} ({key})", flush=True)
        pa, na = ids(pub, key, "all"), ids(new, key, "all")
        dup_p, dup_n = int(pa.duplicated(KEY).sum()), int(na.duplicated(KEY).sum())
        m = na.merge(pa, on=KEY, how="outer", suffixes=("_new", "_pub"), indicator=True)
        pin = [float(u) for u in ids(pub, key, "IN")["uniqueID"]]
        nin = [float(u) for u in ids(new, key, "IN")["uniqueID"]]
        m["single_in_pub"] = m["uniqueID_pub"].isin(pin)
        m["single_in_new"] = m["uniqueID_new"].isin(nin)
        pf = [float(u) for u in pd.read_pickle(kinds(pub, key, True)["bg_rm"]).columns]
        nf = [float(u) for u in pd.read_pickle(kinds(new, key, True)["bg_rm"]).columns]
        m["passed_pub"] = m["uniqueID_pub"].isin(pf)
        m["passed_new"] = m["uniqueID_new"].isin(nf)
        m["status"] = np.select(
            [(m["_merge"] == "left_only"), (m["_merge"] == "right_only"),
             m["single_in_pub"] & m["single_in_new"], m["single_in_pub"] & ~m["single_in_new"],
             ~m["single_in_pub"] & m["single_in_new"]],
            ["recovered", "missing_in_rerun", "matched_single_in", "lost_single_in", "gained_single_in"],
            default="matched_other")
        rel = m[m["single_in_pub"] | m["single_in_new"]].copy()
        rel.insert(0, "construct", name)
        rows_all.append(rel.drop(columns="_merge"))

        # verification on matched single IN: identical traces and filter fate
        both = m[m["status"] == "matched_single_in"]
        ck = {"duplicate_keys_published": dup_p, "duplicate_keys_rerun": dup_n,
              "missing_in_rerun": int((m["status"] == "missing_in_rerun").sum()),
              "matched_status_changed_without_overlap": int(
                  ((m["_merge"] == "both") & (m["ground_truth_new"] != m["ground_truth_pub"])).sum())}
        for kind, p in kinds(pub, key, False).items():
            a = pd.read_pickle(p)
            b = pd.read_pickle(kinds(new, key, False)[kind])
            ca, cb = [float(c) for c in a.columns], [float(c) for c in b.columns]
            have = both[both["uniqueID_pub"].isin(ca) & both["uniqueID_new"].isin(cb)]
            d = np.abs(a[have["uniqueID_pub"].astype(int).to_list()].to_numpy()
                       - b[have["uniqueID_new"].astype(int).to_list()].to_numpy())
            ck[f"{kind}_compared"] = len(have)
            ck[f"{kind}_max_abs_diff"] = float(d.max()) if d.size else 0.0
            ck[f"{kind}_traces_differing"] = int((d.max(axis=0) > 0).sum()) if d.size else 0
            if kind == "bg_rm":
                ck["unfiltered_presence_same"] = bool((both["uniqueID_pub"].isin(ca)
                                                       == both["uniqueID_new"].isin(cb)).all())
            del a, b
        ck["filter_fate_same_for_matched"] = bool((both["passed_pub"] == both["passed_new"]).all())
        ck["filter_fate_differing"] = int((both["passed_pub"] != both["passed_new"]).sum())
        checks[name] = ck

        rec = m[(m["status"] == "recovered") & m["single_in_new"]]
        rec_pass = rec[rec["passed_new"]]
        lost = m[m["status"] == "lost_single_in"]
        summ.append({"construct": name, "single_in_pub": int(m["single_in_pub"].sum()),
                     "single_in_new": int(m["single_in_new"].sum()),
                     "recovered_rois": int((m["status"] == "recovered").sum()),
                     "recovered_single_in": len(rec), "recovered_single_in_passed": len(rec_pass),
                     "lost_single_in": len(lost), "lost_single_in_had_passed": int(lost["passed_pub"].sum()),
                     "gained_single_in": int((m["status"] == "gained_single_in").sum()),
                     "passed_pub": int(m["passed_pub"].sum()), "passed_new": int(m["passed_new"].sum())})
        for s, g in m.groupby("session"):
            summ_s = {"construct": name, "session": s,
                      "single_in_pub": int(g["single_in_pub"].sum()),
                      "recovered_single_in": int(((g["status"] == "recovered") & g["single_in_new"]).sum()),
                      "recovered_single_in_passed": int(((g["status"] == "recovered") & g["single_in_new"] & g["passed_new"]).sum()),
                      "lost_single_in_had_passed": int(((g["status"] == "lost_single_in") & g["passed_pub"]).sum()),
                      "passed_pub": int(g["passed_pub"].sum()), "passed_new": int(g["passed_new"].sum())}
            summ_ses.append(summ_s)

        # recovered passing traces: pickles for classification, events for features
        cols = rec_pass["uniqueID_new"].astype(int).to_list()
        rec_pass = rec_pass.assign(uniqueID_new=rec_pass["uniqueID_new"].astype(int))
        for kind, p in kinds(new, key, True).items():
            pd.read_pickle(p)[cols].to_pickle(rec_dir / f"{name}_IN_filtered_{kind}.pkl")
        if cols:
            ph = to_photons(pd.read_pickle(kinds(new, key, True)["bg_rm"])[cols], pcfg)
            meta = rec_pass.set_index("uniqueID_new")
            for u in cols:
                evs, mol = trace_events(ph[u].to_numpy(np.float64), tcfg)
                first_on = evs[0]["start_frame"] if evs else -1
                mols.append({"construct": name, "uniqueID": u, "session": meta.loc[u, "session"],
                             "movie": meta.loc[u, "movie"], "first_on_frame": first_on, **mol})
    rows = pd.concat(rows_all, ignore_index=True)
    rows.to_csv(out / "rois.csv", index=False)
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(summ_ses).to_csv(out / "summary_session.csv", index=False)
    pd.DataFrame(mols).to_csv(out / "recovered_molecules.csv", index=False, float_format="%.4f")
    checks["all_matched_identical"] = all(
        c["raw_max_abs_diff"] == 0 and c["bg_rm_max_abs_diff"] == 0 and c["filter_fate_same_for_matched"]
        and c["missing_in_rerun"] == 0 and c["matched_status_changed_without_overlap"] == 0
        for k, c in checks.items() if isinstance(c, dict))
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    write_manifest(out, cfg, Path(__file__), [], hash_inputs=False,
                   extra={"published_run": str(pub), "rerun": str(new), "events_config": tcfg.get("_path")})
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.read_csv(out / "summary.csv").to_string(index=False))
    print(json.dumps(checks, indent=1))
    print(out)


if __name__ == "__main__":
    main()
