#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
The paper's IN rule on the mixed slides of one dual-color dataset, in both vesicle channels, for the
representative-FOV figures (user request 2026-11-19: circles only on localizations that are paper-IN,
two-color vesicles included; everything fully outside vesicles is noise or impurities).

paper_in.py scores the single-protein slides against their own dye's channel; a mixed slide holds both dyes,
so here every ROI of the 640-only extraction (traces.csv of the trace_assignment run, filtered or not) is scored
against the ground-truth clusters of the 405 nm (ATTO390) and of the 488 nm (ATTO520) snapshot of its FOV:
in_405 / in_488 = nearest cluster box corner closer than max_dist_closest_ground_truth (4 px) to the ROI's box
corner, paper_in = in_405 or in_488. No channel registration, as in the paper and paper_in.py.

The snapshots are localized as the dataset's ground-truth runs localized the single-protein slides: Picasso's
localize CLI (the pipeline's Extraction/localize.py command) with config_gt<ch>_<dataset>.yaml's box size,
drift, baseline, sensitivity, gain and quantum efficiency, MLE, and the net gradient those runs used (read from
their _locs.yaml; the paramfinder is not rerun), with the Picasso 0.8.8 MLE-race fix of run_local.sh. The ND2s
are copied into the run folder first (Picasso writes its _locs next to the movie; the raw data are not touched),
the copies removed afterwards. Clusters: paper_in.clusters_of (the pipeline's get_and_cluster_locs). Chance as
paper_in.py: every FOV's ROIs also scored against the clusters of the next FOV in acquisition order.

Outputs (Results/Revisions/DualColor/<ID>/paper_in_mixed/run_NNN/): paper_in_mixed.csv (one row per ROI:
extraction_key, uniqueID, slide, fov, filtered, d_405, d_488, in_405, in_488, paper_in, in_channels, null_*),
clusters.csv (per FOV and channel: cluster box corners), summary.csv, locs/ (the _locs.hdf5 / .yaml), manifest.

Usage (picasso-env, Picasso 0.8.8, from the repo root):
    python Revisions/DualColor/paper_in_mixed.py DFK851 [--workers 12]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_in as pin  # noqa: E402  (its clusters_of / nearest / gt_run; it sets up the Extraction utils)
from dc import REPO_ROOT, load_config, run_dir, write_manifest  # noqa: E402
from common import STR_COLUMNS, make_run_dir, rel_to_repo, setup_logging  # noqa: E402

CHANNELS = ("405", "488")


def run_gradient(gt: Path) -> int:
    """The net gradient the dataset's ground-truth run localized with (all its _locs.yaml agree)."""
    g = set()
    for f in gt.rglob("*_locs.yaml"):
        if f.name.startswith("._"):            # AppleDouble files of the exFAT volume
            continue
        # Picasso's _locs.yaml holds python/tuple tags a safe loader refuses: read the one key as text
        g.update(int(m) for m in re.findall(r"^Min\. Net Gradient: (\d+)\s*$", f.read_text(encoding="utf-8"), re.M))
    if len(g) != 1:
        raise SystemExit(f"{gt}: gradients {g}")
    return g.pop()


def localize(nd2: Path, gcfg: dict, gradient: int) -> Path:
    cmd = [sys.executable, "-m", "picasso", "localize", nd2.name, "-b", str(int(gcfg["boxsize"])),
           "-g", str(int(gradient)), "-d", str(int(gcfg["drift"])), "-bl", str(int(gcfg["baseline"])),
           "-s", str(float(gcfg["sensitivity"])), "-ga", str(int(gcfg["gain"])),
           "-qe", str(float(gcfg["quantum_efficiency"])), "-a", "mle"]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", DUALCOLOR_PICASSO_SYNC_MLE="1",
               PYTHONPATH=str(HERE / "daint" / "picasso_sync"))
    subprocess.run(cmd, cwd=str(nd2.parent), env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    locs = nd2.with_name(nd2.stem + "_locs.hdf5")
    if not locs.exists():
        raise RuntimeError(f"no locs for {nd2}")
    return locs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--from-locs", type=Path, help="an existing paper_in_mixed run folder: reuse its locs/, no localization")
    a = ap.parse_args()
    setup_logging()
    import picasso
    if picasso.__version__ != "0.8.8":
        raise SystemExit(f"Picasso {picasso.__version__}: the ground-truth runs used 0.8.8")
    cfg = load_config(a.dataset)
    ds = cfg["dataset"]
    ta = run_dir(cfg, "trace_assignment")
    ta_man = yaml.safe_load(open(ta / "manifest.yaml"))
    fovs = pd.read_csv(REPO_ROOT / ta_man["vesicles_run"] / "fovs.csv", dtype=STR_COLUMNS)
    tr = pd.read_csv(ta / "traces.csv", dtype={"slide": str, "fov": str, "class": str})
    tr["filtered"] = tr["filtered"].astype(str) == "True"
    gcfg = {ch: yaml.safe_load(open(HERE / "extraction" / f"config_gt{ch}_{ds}.yaml")) for ch in CHANNELS}
    gt = {ch: pin.gt_run(cfg, ch, None) for ch in CHANNELS}
    grad = {ch: run_gradient(gt[ch]) for ch in CHANNELS}
    if a.from_locs:                       # reuse a run's localizations (e.g. after a failure past localization)
        out = a.from_locs if a.from_locs.is_absolute() else REPO_ROOT / a.from_locs
    else:
        out = make_run_dir(cfg, "paper_in_mixed")
        (out / "locs").mkdir()
    stage = out / "stage"
    logging.info(f"{ds}: gradients {grad} (from {[rel_to_repo(g) for g in gt.values()]}), output {rel_to_repo(out)}")

    mixed = [s for s, v in cfg["slides"].items() if v["dye"] == "mix"]
    jobs = []
    for slide in mixed:
        f = fovs[fovs["slide"] == slide]
        for r in f.itertuples():
            for ch in CHANNELS:
                src = Path(cfg["input_root"]) / getattr(r, f"file_{ch}")
                dst = stage / slide / r.fov / ch / src.name
                if not a.from_locs:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                jobs.append((slide, r.fov, ch, src, dst))
    if a.from_locs:
        locs = [out / "locs" / s / fv / ch / (src.stem + "_locs.hdf5") for s, fv, ch, src, _ in jobs]
        missing = [p for p in locs if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} localizations missing in {out / 'locs'}, e.g. {missing[0]}")
        logging.info(f"reusing {len(locs)} localizations from {rel_to_repo(out / 'locs')}")
    else:
        logging.info(f"localizing {len(jobs)} snapshots ({a.workers} at a time)")
        with ThreadPoolExecutor(a.workers) as ex:
            locs = list(ex.map(lambda j: localize(j[4], gcfg[j[2]], grad[j[2]]), jobs))
    cl, crow = {}, []
    for (slide, fov, ch, src, dst), lp in zip(jobs, locs):
        keep = out / "locs" / slide / fov / ch
        if not a.from_locs:
            keep.mkdir(parents=True, exist_ok=True)
            for p in (lp, lp.with_suffix(".yaml")):
                shutil.move(str(p), keep / p.name)
        c = pin.clusters_of(keep / lp.name, gcfg[ch])
        cl[(slide, fov, ch)] = c
        crow += [{"slide": slide, "fov": fov, "channel": ch, "x": int(x), "y": int(y)} for x, y in c]
        crow += [] if len(c) else [{"slide": slide, "fov": fov, "channel": ch, "x": np.nan, "y": np.nan}]
    if stage.exists():   # the copies; macOS AppleDouble (._) siblings vanish with their files on exFAT mid-walk
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
        if stage.exists() and any(p for p in stage.rglob("*.nd2") if not p.name.startswith("._")):
            logging.warning(f"some staged copies remain in {rel_to_repo(stage)}")

    rows = []
    maxd = {ch: float(gcfg[ch]["max_dist_closest_ground_truth"]) for ch in CHANNELS}
    for slide in mixed:
        order = list(fovs[fovs["slide"] == slide].sort_values("acq_order")["fov"])
        nxt = {fv: order[(i + 1) % len(order)] for i, fv in enumerate(order)}
        for fv, g in tr[tr["slide"] == slide].groupby("fov"):
            pos = g[["x", "y"]].to_numpy(float)
            r = g[["extraction_key", "uniqueID", "slide", "fov", "filtered"]].copy()
            for ch in CHANNELS:
                r[f"d_{ch}"] = pin.nearest(pos, cl[(slide, fv, ch)])
                r[f"in_{ch}"] = r[f"d_{ch}"] < maxd[ch]
                r[f"null_d_{ch}"] = pin.nearest(pos, cl[(slide, nxt[fv], ch)])
                r[f"null_in_{ch}"] = r[f"null_d_{ch}"] < maxd[ch]
            r["paper_in"] = r["in_405"] | r["in_488"]
            r["null_in"] = r["null_in_405"] | r["null_in_488"]
            r["in_channels"] = np.select([r["in_405"] & r["in_488"], r["in_405"], r["in_488"]], ["both", "405", "488"], "none")
            r["null_fov"] = nxt[fv]
            rows.append(r)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "paper_in_mixed.csv", index=False)
    pd.DataFrame(crow).to_csv(out / "clusters.csv", index=False)
    summ = []
    for (slide, subset), g in [((s, "all ROIs"), g) for s, g in res.groupby("slide")] + \
                              [((s, "filtered"), g[g["filtered"]]) for s, g in res.groupby("slide")]:
        summ.append({"slide": slide, "subset": subset, "n": len(g), "paper_in": int(g["paper_in"].sum()),
                     "in_405_only": int((g["in_channels"] == "405").sum()), "in_488_only": int((g["in_channels"] == "488").sum()),
                     "in_both": int((g["in_channels"] == "both").sum()), "null_in": int(g["null_in"].sum())})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)
    write_manifest(out, cfg, Path(__file__), [ta / "traces.csv"],
                   extra={"trace_assignment_run": rel_to_repo(ta), "vesicles_run": ta_man["vesicles_run"],
                          **{f"gt{ch}_run": rel_to_repo(gt[ch]) for ch in CHANNELS},
                          **{f"gt{ch}_gradient": grad[ch] for ch in CHANNELS},
                          "picasso": picasso.__version__, "snapshots_localized": len(jobs)})
    logging.info("\n" + summ.to_string(index=False))


if __name__ == "__main__":
    main()
