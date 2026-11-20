#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Which traces rejected by the SNR gates look like molecules, judged against background traces of the same
session (R3.3b; rule confirmed by the user 2026-11-20, config.yaml `bright_frames`).

For each construct and session and each k: threshold = the reference_quantile (0.99) of the bright-frame
counts (frames above median + k sigma) of that session's unfiltered background traces. A trace inside
vesicles that the paper's filter rejected with GMM separation < separation_cut (3.0; the explicit gate
at 2.0 and the classifier's 3.0 minimum) is molecule-like if its count exceeds the threshold. About 1 %
of background traces exceed it by construction (reported per session as the false-flag rate).
The traces rejected for an exact-zero z-score are left out (invalid data, 4 traces).

The "gain" (molecule-like rejected / passed) is an upper bound on what a looser SNR gate could add: the
flagged traces were not tested against the later criteria (peak number and timing).

Inputs: a bright_frames run (counts_IN.csv, counts_background.csv) and the reference filter audit
(traces.csv: criterion, separation per IN trace).
Outputs (Results/Revisions/PhotonsFrames/molecule_like/run_NNN/): thresholds.csv, traces.csv (IN traces
with counts and flags), summary_session.csv, summary_construct.csv, manifest.yaml, config.yaml, code/.

Usage (repo root):
    python Revisions/PhotonsFrames/molecule_like.py --counts <bright_frames run> --audit <filter_audit run>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "PaperData"))
from events import load_topic_config  # noqa: E402
from paperdata import load_config, make_run_dir, write_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    args = ap.parse_args()
    tcfg, pcfg = load_topic_config(), load_config(local=True)
    b = tcfg["bright_frames"]
    q, cut, ks = b["reference_quantile"], b["separation_cut"], b["k_values"]
    run = make_run_dir(pcfg, "molecule_like", topic_root=tcfg["output_root"])
    inputs = [args.counts / "counts_IN.csv", args.counts / "counts_background.csv", args.audit / "traces.csv"]
    cin, cbg = pd.read_csv(inputs[0]), pd.read_csv(inputs[1])
    audit = pd.read_csv(inputs[2])[["construct", "uniqueID", "criterion", "separation"]]
    tr = cin.merge(audit, on=["construct", "uniqueID"], how="inner", validate="one_to_one")
    if len(tr) != len(cin) or len(tr) != len(audit):
        raise ValueError(f"counts ({len(cin)}) and audit ({len(audit)}) traces do not match ({len(tr)} joined)")

    thr = []
    for (c, s), g in cbg.groupby(["construct", "session"]):
        for k in ks:
            v = g[f"n_bright_k{k}"].to_numpy()
            t = float(np.quantile(v, q))
            thr.append({"construct": c, "session": s, "k": k, "n_background": len(v), "threshold": t,
                        "background_pct_above": 100 * float((v > t).mean())})
    thr = pd.DataFrame(thr)
    thr.to_csv(run / "thresholds.csv", index=False)

    tr["rejected_snr"] = (tr["separation"] < cut) & (tr["criterion"] != "zscore_zero")
    tr["passed"] = tr["criterion"] == "passed"
    for k in ks:
        t = tr.merge(thr[thr["k"] == k][["construct", "session", "threshold"]], on=["construct", "session"],
                     how="left", validate="many_to_one")["threshold"].to_numpy()
        if np.isnan(t).any():
            raise ValueError(f"k={k}: IN traces in sessions without background traces")
        tr[f"threshold_k{k}"] = t
        tr[f"molecule_like_k{k}"] = tr["rejected_snr"] & (tr[f"n_bright_k{k}"] > t)
    tr.to_csv(run / "traces.csv", index=False, float_format="%.4f")

    def summarize(g: pd.DataFrame) -> dict:
        r = {"n_in": len(g), "n_passed": int(g["passed"].sum()), "n_rejected_snr": int(g["rejected_snr"].sum())}
        for k in ks:
            n = int(g[f"molecule_like_k{k}"].sum())
            r[f"n_molecule_like_k{k}"] = n
            r[f"pct_of_rejected_k{k}"] = 100 * n / r["n_rejected_snr"] if r["n_rejected_snr"] else np.nan
            r[f"gain_pct_of_passed_k{k}"] = 100 * n / r["n_passed"] if r["n_passed"] else np.nan
        return r

    rows = []
    for (c, s), g in tr.groupby(["construct", "session"]):
        r = {"construct": c, "session": s, **summarize(g)}
        for k in ks:
            bg = thr[(thr["construct"] == c) & (thr["session"] == s) & (thr["k"] == k)].iloc[0]
            r[f"threshold_k{k}"] = bg["threshold"]
            r[f"background_pct_above_k{k}"] = bg["background_pct_above"]
            r[f"expected_false_flags_k{k}"] = r["n_rejected_snr"] * bg["background_pct_above"] / 100
        rows.append(r)
    pd.DataFrame(rows).to_csv(run / "summary_session.csv", index=False, float_format="%.3f")
    pd.DataFrame([{"construct": c, **summarize(g)} for c, g in tr.groupby("construct")]).to_csv(
        run / "summary_construct.csv", index=False, float_format="%.3f")
    write_manifest(run, {**tcfg, "counts_run": str(args.counts), "audit_run": str(args.audit)}, Path(__file__), inputs)
    print(run)


if __name__ == "__main__":
    main()
