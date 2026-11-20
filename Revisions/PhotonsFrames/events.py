#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
ON events of every published filtered trace inside vesicles, with their photons (R3.3; reused by R3.2).

Per trace (the published filtered IN set, from the local mirror of the published extraction,
../PaperData): ON frames from the paper's classifier on the min-max trace (config gmm_proba), runs of
ON frames as events (config min_peak_width), exactly as Features/features.py does for Figs. 5D and 6C,
here on the full 6000 frames. Photons are the bg_rm traces x camera.photons_per_adu (../PaperData; 0.062, measured; run_001 used 16.0 / 300).

Outputs (Results/Revisions/PhotonsFrames/events/run_NNN/):
  events.csv.gz   one row per event: construct, uniqueID, event, start_frame, n_frames, photons (sum
                  over the event's frames), photons_padded (plus pad_frames on each side), peak_photons,
                  interior_frames and interior_photons_mean (inner frames of events >= interior_min_frames)
  molecules.csv   one row per trace: construct, uniqueID, session, movie, n_events, n_on_frames,
                  frac_events_1frame, mean_on_frames, mean_off_frames, duty_cycle (as Features: ON frames
                  over the active window), median_photons_per_event, median_photons_per_event_padded,
                  photons_per_full_frame (median over all inner frames of the molecule's long events),
                  n_interior_frames, total_photons_on (sum over all ON frames)
  manifest.yaml, config.yaml, code/

Usage (repo root):
    python Revisions/PhotonsFrames/events.py [--constructs HaloK117 ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "PaperData"))
sys.path.insert(0, str(HERE.parents[1] / "Extraction"))
from paperdata import (load_config, load_index, load_traces, make_run_dir, to_photons,  # noqa: E402
                       trace_path, write_manifest)
from utils import gmm_classify_frames  # noqa: E402  (the paper's frame classifier)


def load_topic_config(path: Path | None = None) -> dict:
    path = Path(path) if path else HERE / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg


def runs_of(mask: np.ndarray, min_width: int = 1) -> list[tuple[int, int]]:
    """(start, end) inclusive runs of True of at least min_width frames."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return [(s, e) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1) if e - s + 1 >= min_width]


def on_frames(photons: np.ndarray, gmm_proba: float) -> np.ndarray:
    """The paper's ON classification of one trace (min-max, then gmm_classify_frames, no separation gate)."""
    mm = (photons - photons.min()) / (photons.max() - photons.min())
    _, _, on = gmm_classify_frames(mm, gmm_proba, 0.0)
    return on


def trace_events(photons: np.ndarray, tcfg: dict) -> tuple[list[dict], dict]:
    """Events and per-molecule summary of one trace (photons per frame, background removed)."""
    on = on_frames(photons, tcfg["gmm_proba"])
    runs = runs_of(on, tcfg["min_peak_width"])
    pad, nmin, T = tcfg["pad_frames"], tcfg["interior_min_frames"], photons.size
    events, interior = [], []
    for k, (s, e) in enumerate(runs):
        n = e - s + 1
        inner = photons[s + 1:e] if n >= nmin else np.array([])
        interior.append(inner)
        events.append({"event": k, "start_frame": s, "n_frames": n, "photons": float(photons[s:e + 1].sum()),
                       "photons_padded": float(photons[max(s - pad, 0):min(e + pad, T - 1) + 1].sum()),
                       "peak_photons": float(photons[s:e + 1].max()), "interior_frames": int(inner.size),
                       "interior_photons_mean": float(inner.mean()) if inner.size else np.nan})
    inner_all = np.concatenate(interior) if interior else np.array([])
    durations = np.array([e - s + 1 for s, e in runs])
    offs = np.array([runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)])
    active = runs[-1][1] - runs[0][0] + 1 if runs else 0
    ph = np.array([ev["photons"] for ev in events])
    php = np.array([ev["photons_padded"] for ev in events])
    mol = {"n_events": len(runs), "n_on_frames": int(on.sum()),
           "frac_events_1frame": float((durations == 1).mean()) if runs else np.nan,
           "mean_on_frames": float(durations.mean()) if runs else np.nan,
           "mean_off_frames": float(offs.mean()) if offs.size else np.nan,
           "duty_cycle": float(on.sum() / active) if active else np.nan,
           "median_photons_per_event": float(np.median(ph)) if runs else np.nan,
           "median_photons_per_event_padded": float(np.median(php)) if runs else np.nan,
           "photons_per_full_frame": float(np.median(inner_all)) if inner_all.size else np.nan,
           "n_interior_frames": int(inner_all.size),
           "total_photons_on": float(photons[on].sum())}
    return events, mol


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--constructs", nargs="+")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tcfg = load_topic_config(args.config)
    pcfg = load_config(local=True)
    constructs = args.constructs or tcfg["constructs"]
    run = make_run_dir(pcfg, "events", topic_root=tcfg["output_root"])
    t0, inputs, ev_all, mol_all = time.time(), [], [], []
    for c in constructs:
        inputs.append(trace_path(pcfg, c, "bg_rm"))
        ph = to_photons(load_traces(pcfg, c, "bg_rm"), pcfg)
        idx = load_index(pcfg, c).set_index("uniqueID")
        ids = list(ph.columns)
        res = Parallel(n_jobs=tcfg["n_jobs"])(delayed(trace_events)(ph[u].to_numpy(np.float64), tcfg) for u in ids)
        for u, (evs, mol) in zip(ids, res):
            for ev in evs:
                ev_all.append({"construct": c, "uniqueID": u, **ev})
            mol_all.append({"construct": c, "uniqueID": u, "session": idx.loc[u, "session"],
                            "movie": idx.loc[u, "movie"], **mol})
        logging.info("%s: %d traces, %d events (%.0f s)", c, len(ids), sum(len(e) for e, _ in res), time.time() - t0)
    pd.DataFrame(ev_all).to_csv(run / "events.csv.gz", index=False, float_format="%.3f")
    pd.DataFrame(mol_all).to_csv(run / "molecules.csv", index=False, float_format="%.4f")
    cfg = {**tcfg, "paperdata_config": pcfg["_path"], "published_run": pcfg["published_run"],
           "frame_interval_ms": pcfg["frame_interval_ms"], "camera": pcfg["camera"]}
    write_manifest(run, cfg, Path(__file__), inputs, extra={"constructs": constructs,
                   "elapsed_s": round(time.time() - t0)})
    logging.info("done: %s", run)


if __name__ == "__main__":
    main()
