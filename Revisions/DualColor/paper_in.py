#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
The paper's IN rule on the single-protein slides of one dual-color dataset (TRAINING_NOTES.md
point 7), applied to the ROIs of the dataset's trace_assignment run.

As Extraction/extract.py in ground-truth mode: the vesicle-channel snapshot of each FOV (405 nm
on ATTO390 / HT slides, 488 nm on ATTO520 / SNAP slides), localized by the ground-truth-mode
extraction run (config_gt<ch>_<ID>.yaml, Extraction_gt<ch>/<run>/), is clustered with the
pipeline's own get_and_cluster_locs (max_distance_ground_truth 2.5 px, min_on_ground_truth 3
localizations; cluster centers as rounded box corners, the dummy index 0 dropped), and a ROI is
IN if the nearest cluster lies closer than max_dist_closest_ground_truth (4 px) to its box corner
(x, y). No channel registration, as in the paper. The 640 ROIs are those of the 640-only run
(the ground-truth runs localize 640 at the same gradient; their own ROIs are not used).

Chance: every FOV's ROIs are also scored against the clusters of the next FOV of the same slide
in acquisition order (the last against the first), i.e. the same vesicle pattern statistics
without the true positions; null_in is the IN rate this gives. The excess over chance,
f = (in - null_in) / (1 - null_in), is the share of IN calls that are real colocalizations.

Outputs (Results/Revisions/DualColor/<ID>/paper_in/run_NNN/): paper_in.csv (one row per ROI of
the single-protein slides: extraction_key, uniqueID, slide, fov, filtered, gt_channel,
gt_dist_px, paper_in, null_dist_px, null_in), clusters.csv (clusters per FOV), summary.csv (per
slide, all ROIs and filtered: n, IN, null IN, f), manifest.

Usage (picasso-env, from the repo root):
    python Revisions/DualColor/paper_in.py DFK788 [--gt405-run NAME] [--gt488-run NAME]
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import spatial

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc import REPO_ROOT, load_config, load_dfk785, run_dir, write_manifest  # noqa: E402

load_dfk785("assign_traces")            # puts Revisions/Bleedthrough on sys.path for common
from common import STR_COLUMNS, make_run_dir, rel_to_repo, setup_logging  # noqa: E402

_spec = importlib.util.spec_from_file_location("extraction_utils", REPO_ROOT / "Extraction" / "utils.py")
xu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xu)


def gt_run(cfg: dict, ch: str, name: str | None) -> Path:
    base = REPO_ROOT / cfg["output_root"] / f"Extraction_gt{ch}"
    runs = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    if name:
        return base / name
    if len(runs) != 1:
        raise ValueError(f"{len(runs)} runs under {base}; name one")
    return runs[0]


def clusters_of(locs: Path, gcfg: dict) -> np.ndarray:
    """(n, 2) cluster box corners (x, y) of one vesicle-channel movie, as extract.py."""
    if not locs.exists():
        return np.empty((0, 2))
    _, _, x, y, _, _ = xu.get_and_cluster_locs(str(locs), box_size=int(gcfg["boxsize"]),
                                               max_distance=float(gcfg.get("max_distance_ground_truth", 2.5)),
                                               min_locs=int(gcfg.get("min_on_ground_truth", 3)))
    return np.stack((x[1:], y[1:]), axis=1)


def nearest(pos: np.ndarray, cl: np.ndarray) -> np.ndarray:
    if not len(cl):
        return np.full(len(pos), np.inf)
    return spatial.KDTree(cl).query(pos)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--gt405-run", default=None)
    ap.add_argument("--gt488-run", default=None)
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.dataset)
    ds, tags = cfg["dataset"], cfg["channel_tags"]
    ta = run_dir(cfg, "trace_assignment")
    with open(ta / "manifest.yaml") as f:
        ta_man = yaml.safe_load(f)
    fovs = pd.read_csv(REPO_ROOT / ta_man["vesicles_run"] / "fovs.csv", dtype=STR_COLUMNS)
    tr = pd.read_csv(ta / "traces.csv", dtype={"slide": str, "fov": str, "class": str})
    tr["filtered"] = tr["filtered"].astype(str) == "True"

    rows, crow, inputs, extra = [], [], [ta / "traces.csv"], {"trace_assignment_run": rel_to_repo(ta)}
    for slide, s in cfg["slides"].items():
        if s["dye"] == "mix":
            continue
        ch = str(cfg["dyes"][s["dye"]]["home_channel"])
        run = gt_run(cfg, ch, getattr(a, f"gt{ch}_run"))
        gcfg = yaml.safe_load(open(HERE / "extraction" / f"config_gt{ch}_{ds}.yaml"))
        extra[f"gt{ch}_run"] = rel_to_repo(run)
        f = fovs[fovs["slide"] == slide].sort_values("acq_order").reset_index(drop=True)
        key, link640 = s["key"], {}
        cl = {}
        for r in f.itertuples():
            name640 = key + Path(r.file_640).name[len(s["prefix"]):]
            suffix = name640[len(key + "_" + tags["640"]):]                          # '_NNNN.nd2' or '.nd2'
            locs = run / ds / key / f"{key}_{tags[ch]}{suffix[:-4]}_locs.hdf5"
            cl[r.fov] = clusters_of(locs, gcfg)
            link640[r.fov] = name640
            crow.append({"slide": slide, "fov": r.fov, "gt_channel": ch, "locs": rel_to_repo(locs),
                         "locs_found": locs.exists(), "n_clusters": len(cl[r.fov])})
        order = list(f["fov"])
        nxt = {fv: order[(i + 1) % len(order)] for i, fv in enumerate(order)}
        g = tr[tr["slide"] == slide]
        for fv, gf in g.groupby("fov"):
            pos = gf[["x", "y"]].to_numpy(float)
            d, dn = nearest(pos, cl[fv]), nearest(pos, cl[nxt[fv]])
            rows.append(gf[["extraction_key", "uniqueID", "slide", "fov", "filtered"]]
                        .assign(gt_channel=ch, gt_dist_px=d, paper_in=d < float(gcfg["max_dist_closest_ground_truth"]),
                                null_fov=nxt[fv], null_dist_px=dn,
                                null_in=dn < float(gcfg["max_dist_closest_ground_truth"])))
        logging.info(f"{ds} {slide} ({key}, {ch} nm): {len(f)} FOVs, "
                     f"{sum(len(cl[fv]) for fv in order)} clusters, {len(g)} ROIs")
    pin = pd.concat(rows, ignore_index=True)
    out = make_run_dir(cfg, "paper_in")
    pin.to_csv(out / "paper_in.csv", index=False)
    cdf = pd.DataFrame(crow)
    cdf.to_csv(out / "clusters.csv", index=False)
    if not cdf["locs_found"].all():
        logging.warning(f"{int((~cdf['locs_found']).sum())} FOVs without vesicle-channel locs (all their ROIs OUT)")
    summ = []
    for (slide, subset), g in [((s, "all ROIs"), g) for s, g in pin.groupby("slide")] + \
                              [((s, "filtered"), g[g["filtered"]]) for s, g in pin.groupby("slide")]:
        i, n0 = g["paper_in"].mean(), g["null_in"].mean()
        summ.append({"slide": slide, "subset": subset, "n": len(g), "paper_in": int(g["paper_in"].sum()),
                     "in_rate": i, "null_in_rate": n0, "excess_f": (i - n0) / (1 - n0) if n0 < 1 else np.nan})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        logging.info("Paper IN rule per slide:\n" + summ.to_string(index=False))
    write_manifest(out, cfg, Path(__file__), inputs, extra=extra)
    logging.info(f"Done: {out}")


if __name__ == "__main__":
    main()
