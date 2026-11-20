#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Do the molecule-like traces that the SNR gates rejected blink differently from the traces the paper kept
(R3.3b bias check)?

For every molecule-like rejected trace (molecule_like.py, k_main), the per-molecule event summary of
events.py (the paper's frame classifier without the separation gate, photons from the local mirror of the
unfiltered bg_rm traces); the passed traces' summaries come from the events run of R3.3a (same
classifier on the same values). Compared per construct and session as medians and quartiles over
molecules: number of ON events, fraction of single-frame events, mean ON (frames), median photons per
event, mean OFF (frames), duty cycle. Descriptive; no tests.

Outputs (Results/Revisions/PhotonsFrames/flagged_features/run_NNN/): molecules_flagged.csv,
comparison.csv (construct, session, group, n_molecules, <feature>_median/_q25/_q75), manifest.yaml,
config.yaml, code/.

Usage (repo root):
    python Revisions/PhotonsFrames/flagged_features.py --molecule-like <run> --events <run>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "PaperData"))
from events import load_topic_config, trace_events  # noqa: E402
from paperdata import load_config, make_run_dir, to_photons, trace_path, write_manifest  # noqa: E402

FEATURES = ["n_events", "frac_events_1frame", "mean_on_frames", "median_photons_per_event", "mean_off_frames",
            "duty_cycle"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--molecule-like", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    args = ap.parse_args()
    tcfg, pcfg = load_topic_config(), load_config(local=True)
    k = tcfg["bright_frames"]["k_main"]
    run = make_run_dir(pcfg, "flagged_features", topic_root=tcfg["output_root"])
    ml_p, ev_p = args.molecule_like / "traces.csv", args.events / "molecules.csv"
    inputs = [ml_p, ev_p]
    ml = pd.read_csv(ml_p)
    flagged = ml[ml[f"molecule_like_k{k}"]]
    rows = []
    for c, g in flagged.groupby("construct"):
        p = trace_path(pcfg, c, "bg_rm", filtered=False)
        inputs.append(p)
        ph = to_photons(pd.read_pickle(p)[g["uniqueID"].to_list()], pcfg)
        res = Parallel(n_jobs=tcfg["n_jobs"])(delayed(trace_events)(ph[u].to_numpy(np.float64), tcfg)
                                              for u in g["uniqueID"])
        for (_, r), (_, mol) in zip(g.iterrows(), res):
            rows.append({"construct": c, "uniqueID": r["uniqueID"], "session": r["session"], "movie": r["movie"],
                         "separation": r["separation"], "criterion": r["criterion"], **mol})
    fl = pd.DataFrame(rows)
    fl.to_csv(run / "molecules_flagged.csv", index=False, float_format="%.4f")

    passed = pd.read_csv(ev_p)
    both = pd.concat([passed.assign(group="passed"), fl.assign(group="molecule-like rejected")], ignore_index=True)
    out = []
    for (c, s, grp), g in both.groupby(["construct", "session", "group"]):
        r = {"construct": c, "session": s, "group": grp, "n_molecules": len(g)}
        for f in FEATURES:
            v = g[f].dropna().to_numpy(float)
            r[f"{f}_median"] = float(np.median(v)) if v.size else np.nan
            r[f"{f}_q25"] = float(np.percentile(v, 25)) if v.size else np.nan
            r[f"{f}_q75"] = float(np.percentile(v, 75)) if v.size else np.nan
        out.append(r)
    pd.DataFrame(out).to_csv(run / "comparison.csv", index=False, float_format="%.4f")
    write_manifest(run, {**tcfg, "molecule_like_run": str(args.molecule_like), "events_run": str(args.events)},
                   Path(__file__), inputs)
    print(run)


if __name__ == "__main__":
    main()
