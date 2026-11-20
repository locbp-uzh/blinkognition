#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
FrameZero: kinetic features of the recovered molecules (ON in the first frame, dropped by the published
extraction) against the published ones, and whether including them changes the paper's feature differences.

Features (user, 2026-11-20): mean ON time, mean OFF time and duty cycle per molecule, as in R3.2 and Figs. 5D and
6C (the paper's ON classification on the full 6000 frames, Revisions/PhotonsFrames/events.py trace_events;
34.49 ms per frame). Photons are not used.
  - Published molecules: every published filtered trace inside vesicles (events run_002 of R3.3a, the same
    definition). The rerun's dataset = published minus the published traces that a recovered neighbor now
    overlaps (compare.py "lost") plus the recovered passing traces.
  - Recovered molecules: their filtered traces from compare.py, same trace_events. Their first ON event started
    before the recording, so it is cut short; as a sensitivity check the features are also computed without
    the first event (mean ON over the later events; duty cycle over the span from the second event's start to
    the last event's end; mean OFF unchanged, the first gap being complete).
Per construct: recovered against published (kept) molecules, n, median and quartiles, rank-biserial r (recovered
first: r > 0, recovered larger). Per pair of the paper (Halo-D106 vs Halo-K117, Halo-D106 vs SNAP-C148, scGrx1
vs scGrx1-AcK20): rank-biserial r per feature (first protein of the pair: r > 0, first larger), published
dataset against rerun dataset (and rerun dataset with the recovered features without their first event),
pooled and within single sessions with >= 20 molecules of each protein. Descriptive; no p-values.

Outputs (Results/Revisions/FrameZero/features/run_NNN/): molecules_recovered.csv, per_construct.csv, pairs.csv,
checks.json, manifest.yaml, config.yaml, code/.

Usage (repo root, locbplots environment):
    python Revisions/FrameZero/features.py --compare Results/Revisions/FrameZero/s3it/compare/run_NNN
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "PaperData"))
sys.path.insert(0, str(HERE.parent / "PhotonsFrames"))
from paperdata import load_config, make_run_dir, to_photons, write_manifest  # noqa: E402
from events import load_topic_config, trace_events  # noqa: E402

EVENTS_RUN = "Results/Revisions/PhotonsFrames/events/run_002"
FEATURES = ["mean_on_ms", "mean_off_ms", "duty_cycle"]
PAIRS = [("HaloD106", "HaloK117"), ("HaloD106", "SNAPC148"), ("scGrx1", "scGrx1AcK20")]
MIN_PER_SESSION = 20


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(2 * mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)) - 1)


def recovered_features(ph: np.ndarray, tcfg: dict) -> dict:
    evs, mol = trace_events(ph, tcfg)
    out = {"n_events": mol["n_events"], "first_on_frame": evs[0]["start_frame"] if evs else -1,
           "mean_on_frames": mol["mean_on_frames"], "mean_off_frames": mol["mean_off_frames"],
           "duty_cycle": mol["duty_cycle"]}
    if len(evs) > 1:
        later = evs[1:]
        span = later[-1]["start_frame"] + later[-1]["n_frames"] - later[0]["start_frame"]
        out["mean_on_frames_no_first"] = float(np.mean([e["n_frames"] for e in later]))
        out["duty_cycle_no_first"] = float(sum(e["n_frames"] for e in later) / span)
    else:
        out["mean_on_frames_no_first"] = out["duty_cycle_no_first"] = np.nan
    return out


def summarize(x: pd.Series) -> dict:
    x = x.dropna()
    return {"n": len(x), "median": x.median(), "q25": x.quantile(0.25), "q75": x.quantile(0.75)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", type=Path, required=True)
    args = ap.parse_args()
    with open(HERE / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(HERE / "config.yaml")
    pcfg = load_config()
    tcfg = load_topic_config()
    ms = float(pcfg["frame_interval_ms"])
    cmp_ = REPO_ROOT / args.compare
    out = make_run_dir(pcfg, "features", topic_root=cfg["output_root"])

    pub = pd.read_csv(REPO_ROOT / EVENTS_RUN / "molecules.csv")
    pub["mean_on_ms"] = pub["mean_on_frames"] * ms
    pub["mean_off_ms"] = pub["mean_off_frames"] * ms
    rois = pd.read_csv(cmp_ / "rois.csv")
    lost = rois[(rois["status"] == "lost_single_in") & rois["passed_pub"]]
    lost_ids = {(c, int(u)) for c, u in zip(lost["construct"], lost["uniqueID_pub"])}
    pub["lost"] = [(c, int(u)) in lost_ids for c, u in zip(pub["construct"], pub["uniqueID"])]

    rec_rows = []
    fdir = cmp_ / "recovered" / "ProteinTracesIN" / "Filtered"
    meta = rois[rois["status"] == "recovered"].set_index(["construct", "uniqueID_new"])
    for c in pcfg["constructs"]:
        bg = pd.read_pickle(fdir / f"{c}_IN_filtered_bg_rm.pkl")
        ph = to_photons(bg, pcfg)
        for u in ph.columns:
            r = recovered_features(ph[u].to_numpy(np.float64), tcfg)
            m = meta.loc[(c, float(u))] if (c, float(u)) in meta.index else meta.loc[(c, int(u))]
            rec_rows.append({"construct": c, "uniqueID_rerun": int(u), "session": m["session"], "movie": m["movie"], **r})
    rec = pd.DataFrame(rec_rows)
    for col in ("mean_on_frames", "mean_off_frames", "mean_on_frames_no_first"):
        rec[col.replace("frames", "ms")] = rec[col] * ms
    rec.to_csv(out / "molecules_recovered.csv", index=False, float_format="%.5g")

    # check: same events as compare.py computed on the cluster
    cm = pd.read_csv(cmp_ / "recovered_molecules.csv").rename(columns={"uniqueID": "uniqueID_rerun"})
    chk = rec.merge(cm, on=["construct", "uniqueID_rerun"], suffixes=("", "_cmp"))
    checks = {"recovered_molecules": len(rec), "matched_with_compare": len(chk),
              "events_identical_to_compare": bool((chk["n_events"] == chk["n_events_cmp"]).all()
                                                  and np.allclose(chk["duty_cycle"], chk["duty_cycle_cmp"], atol=1e-4,
                                                                  equal_nan=True)),
              "first_on_frame_0_fraction": float((rec["first_on_frame"] == 0).mean()),
              "published_lost_removed": int(pub["lost"].sum())}

    # per construct: recovered vs published (kept)
    rows = []
    for c in pcfg["constructs"]:
        p_ = pub[(pub["construct"] == c) & ~pub["lost"]]
        r_ = rec[rec["construct"] == c]
        for f in FEATURES + ["mean_on_ms_no_first", "duty_cycle_no_first"]:
            pf = f.replace("_no_first", "")
            rows.append({"construct": c, "feature": f,
                         **{f"published_{k}": v for k, v in summarize(p_[pf]).items()},
                         **{f"recovered_{k}": v for k, v in summarize(r_[f]).items()},
                         "r_recovered_vs_published": rank_biserial(r_[f].to_numpy(float), p_[pf].to_numpy(float))})
    pd.DataFrame(rows).to_csv(out / "per_construct.csv", index=False, float_format="%.5g")

    # pairs: published dataset vs rerun dataset
    def dataset(kind: str) -> pd.DataFrame:
        base = pub[["construct", "session"] + FEATURES].copy() if kind == "published" else \
            pub.loc[~pub["lost"], ["construct", "session"] + FEATURES].copy()
        if kind == "published":
            return base
        r = rec[["construct", "session"]].copy()
        if kind == "rerun":
            r[FEATURES] = rec[FEATURES].to_numpy()
        else:  # rerun, recovered without their first event
            r["mean_on_ms"], r["mean_off_ms"], r["duty_cycle"] = (rec["mean_on_ms_no_first"], rec["mean_off_ms"],
                                                                   rec["duty_cycle_no_first"])
        return pd.concat([base, r], ignore_index=True)

    prow = []
    for kind in ("published", "rerun", "rerun_recovered_without_first_event"):
        d = dataset(kind)
        for a, b in PAIRS:
            scopes = [("pooled", "all", d)]
            for s, g in d[d["construct"].isin([a, b])].groupby("session"):
                if (g["construct"] == a).sum() >= MIN_PER_SESSION and (g["construct"] == b).sum() >= MIN_PER_SESSION:
                    scopes.append(("session", s, g))
            for scope, s, g in scopes:
                for f in FEATURES:
                    xa, xb = g.loc[g["construct"] == a, f].to_numpy(float), g.loc[g["construct"] == b, f].to_numpy(float)
                    prow.append({"dataset": kind, "pair": f"{a}_{b}", "scope": scope, "session": s, "feature": f,
                                 f"n_first": int(np.isfinite(xa).sum()), "n_second": int(np.isfinite(xb).sum()),
                                 "median_first": np.nanmedian(xa), "median_second": np.nanmedian(xb),
                                 "rank_biserial_r": rank_biserial(xa, xb)})
    pairs = pd.DataFrame(prow)
    pairs.to_csv(out / "pairs.csv", index=False, float_format="%.5g")
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    write_manifest(out, cfg, Path(__file__), [REPO_ROOT / EVENTS_RUN / "molecules.csv", cmp_ / "rois.csv"],
                   hash_inputs=True, extra={"compare_run": str(args.compare), "events_run": EVENTS_RUN})
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(json.dumps(checks))
        print(pd.read_csv(out / "per_construct.csv").round(3).to_string(index=False))
        piv = pairs[pairs["scope"] == "pooled"].pivot_table(index=["pair", "feature"], columns="dataset",
                                                             values="rank_biserial_r")
        print(piv.round(3))
        ps = pairs[(pairs["scope"] == "session") & (pairs["pair"] != "HaloD106_SNAPC148")].pivot_table(
            index=["pair", "session", "feature"], columns="dataset", values="rank_biserial_r")
        print(ps.round(3))
    print(out)


if __name__ == "__main__":
    main()
