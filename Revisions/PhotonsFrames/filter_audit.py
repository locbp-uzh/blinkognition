#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Replay of the paper's quality filter on every unfiltered trace inside vesicles, recording which criterion
removes each trace (R3.3b).

The cascade is Extraction/filter.py, in its order, with the published parameters (config.yaml `filter`):
  0. zscore_zero     the z-scored trace has a frame exactly 0 (filter_and_augment_compound drops these as
                     invalid before the peak criteria)
then `_filter_particle`: one GMM fit per background-removed trace (Extraction/utils.py gmm_classify_frames,
posterior >= gmm_proba_threshold, gmm_min_separation), and
  1. snr_gate        separation < snr_min_separation (2.0)
  2. duty_cycle      fraction of ON frames > max_duty_cycle
  3. too_few_peaks   number of ON runs <= min_peak_number; split by whether the separation was below
                     gmm_min_separation (3.0), where the classifier returns no ON frames at all
  4. first_peak_late first peak at or after first_peak_time
  5. first_gap_long  fewer than 2 peaks, or the first two peaks at least delta_first_second apart
  6. last_peak_early last peak at or before last_peak_time
Peaks are the argmax frame of each ON run, as in filter.py. The replay must reproduce the published
filtered set trace by trace (checks.json).

Per trace it also records the events of the classifier without the separation gate (min_separation 0, the
classification Figs. 5D and 6C and events.py use), so traces removed by the gate can be described.

Outputs (Results/Revisions/PhotonsFrames/filter_audit/run_NNN/): traces.csv (one row per unfiltered IN
trace), fate.csv (counts per construct and criterion), fate_session.csv, checks.json, manifest.yaml,
config.yaml, code/.

Usage (repo root):
    python Revisions/PhotonsFrames/filter_audit.py [--constructs HaloK117 ...]
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
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "PaperData"))
from events import load_topic_config, runs_of  # noqa: E402
from paperdata import load_config, load_index, load_traces, make_run_dir, to_photons, trace_path, write_manifest  # noqa: E402
from utils import gmm_classify_frames  # noqa: E402  (Extraction/utils.py, on sys.path via events)

CRITERIA = ["zscore_zero", "snr_gate", "too_few_peaks_no_on_frames", "duty_cycle", "too_few_peaks",
            "first_peak_late", "first_gap_long", "last_peak_early", "passed"]


def audit_trace(bg: np.ndarray, f: dict) -> dict:
    """The filter cascade on one bg_rm trace (ADU), with the first failing criterion. Before the cascade,
    filter.py drops traces whose z-scored version (AllNorm) has an exact 0 (it treats 0 as invalid data);
    the stored z-score equals (bg_rm - mean) / SD exactly (PaperData step 0), so it is recomputed here."""
    zero = bool(((bg - bg.mean()) / bg.std() == 0).any())
    _, _, sig_free, sep = gmm_classify_frames(bg, f["gmm_proba_threshold"], 0.0, return_separation=True)
    no_on = sep < f["gmm_min_separation"]
    signal = np.zeros_like(sig_free) if no_on else sig_free   # what filter.py's classifier returns
    runs = runs_of(signal, f["min_peak_width"])
    peaks = np.array([s + int(np.argmax(bg[s:e + 1])) for s, e in runs])
    n = len(peaks)
    if zero:
        crit = "zscore_zero"
    elif f["snr_min_separation"] > 0 and sep < f["snr_min_separation"]:
        crit = "snr_gate"
    elif f["max_duty_cycle"] > 0 and signal.mean() > f["max_duty_cycle"]:
        crit = "duty_cycle"
    elif n <= f["min_peak_number"]:
        crit = "too_few_peaks_no_on_frames" if no_on else "too_few_peaks"
    elif peaks[0] >= f["first_peak_time"]:
        crit = "first_peak_late"
    elif n < 2 or (peaks[1] - peaks[0]) >= f["delta_first_second"]:
        crit = "first_gap_long"
    elif peaks[-1] <= f["last_peak_time"]:
        crit = "last_peak_early"
    else:
        crit = "passed"
    free = runs_of(sig_free, f["min_peak_width"])
    dur = np.array([e - s + 1 for s, e in free])
    return {"separation": float(sep), "criterion": crit, "n_peaks_filter": n,
            "duty_filter": float(signal.mean()),
            "first_peak": int(peaks[0]) if n else -1, "last_peak": int(peaks[-1]) if n else -1,
            "n_events_free": len(free), "frac_events_1frame_free": float((dur == 1).mean()) if len(free) else np.nan,
            "median_adu_per_event_free": float(np.median([bg[s:e + 1].sum() for s, e in free])) if len(free) else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--constructs", nargs="+")
    ap.add_argument("--where", choices=["IN", "background"], default="IN",
                    help="background: the unfiltered background traces (random positions >= 5 px from any "
                         "protein ROI, same movies) through the same cascade, as a noise reference; their "
                         "published 'Filtered' set is a different filter (combine.py robust outlier score), "
                         "so published_passed means 'kept by that filter' there and no reproduction is checked")
    ap.add_argument("--remote", action="store_true", help="read the published run on S3IT, not the local mirror")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tcfg, pcfg = load_topic_config(), load_config(local=not args.remote)
    f = tcfg["filter"]
    constructs = args.constructs or tcfg["constructs"]
    stage = "filter_audit" if args.where == "IN" else "filter_audit_background"
    run = make_run_dir(pcfg, stage, topic_root=tcfg["output_root"])
    t0, inputs, rows, checks = time.time(), [], [], {}
    for c in constructs:
        p = trace_path(pcfg, c, "bg_rm", where=args.where, filtered=False)
        inputs.append(p)
        bg = pd.read_pickle(p)
        idx = load_index(pcfg, c, where=args.where).set_index("uniqueID")
        ids = list(bg.columns)
        res = Parallel(n_jobs=tcfg["n_jobs"])(delayed(audit_trace)(bg[u].to_numpy(np.float64), f) for u in ids)
        published = set(idx.index[idx["passed_filter"]])
        mine = {u for u, r in zip(ids, res) if r["criterion"] == "passed"}
        checks[c] = {"n_unfiltered": len(ids), "n_passed_replay": len(mine), "n_passed_published": len(published),
                     "identical_sets": mine == published, "passed_not_published": len(mine - published),
                     "published_not_passed": len(published - mine)}
        for u, r in zip(ids, res):
            r["median_photons_per_event_free"] = float(to_photons(r.pop("median_adu_per_event_free"), pcfg))
            rows.append({"construct": c, "uniqueID": u, "session": idx.loc[u, "session"], "movie": idx.loc[u, "movie"],
                         **r, "published_passed": u in published})
        logging.info("%s: %s (%.0f s)", c, checks[c], time.time() - t0)
    tr = pd.DataFrame(rows)
    tr.to_csv(run / "traces.csv", index=False, float_format="%.5f")
    for by, name in ((["construct"], "fate.csv"), (["construct", "session"], "fate_session.csv")):
        fate = tr.groupby(by + ["criterion"]).size().unstack(fill_value=0).reindex(columns=CRITERIA, fill_value=0)
        fate.insert(0, "n_unfiltered", fate.sum(axis=1))
        fate.reset_index().to_csv(run / name, index=False)
    all_ok = all(v["identical_sets"] for v in checks.values()) if args.where == "IN" else None
    with open(run / "checks.json", "w") as fh:
        json.dump({"where": args.where, "replay_reproduces_published": all_ok, "constructs": checks}, fh, indent=2)
    write_manifest(run, {**tcfg, "paperdata_config": pcfg["_path"], "published_run": pcfg["published_run"]},
                   Path(__file__), inputs, extra={"constructs": constructs, "where": args.where,
                                                  "elapsed_s": round(time.time() - t0)})
    logging.info("replay reproduces the published filtered set: %s; done: %s", all_ok, run)


if __name__ == "__main__":
    main()
