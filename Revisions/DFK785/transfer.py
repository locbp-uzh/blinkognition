#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Join the ML/classify.py predictions of the DFK785 traces with their vesicle provenance.

Every filtered trace of the 640-only run carries a prediction (HaloD106 or SNAPC148, MC
dropout mean, Wasserstein uncertainty, kept at the training run's fixed threshold) and a
provenance category from overview.py (clean, crowded, wrong label, dual, no label,
ambiguous, OUT border, OUT interior; trace_assignment at 2 px).

    pure slides    the slide's protein is the truth for every trace: accuracy per category
    mixed slides   the vesicle label is the truth only for the single-dye categories: agreement
                   of the prediction with the label's protein (HT -> HaloD106, SNAP -> SNAPC148);
                   other categories get their predicted composition

Intervals: Wilson 95 % for a binomial proportion, and a cluster bootstrap over FOVs (FOVs
resampled with replacement, 2000 times; percentile 95 %), since traces of one FOV share
focus, laser and sample conditions and are not independent.

Outputs (Results/Revisions/DFK785/transfer/run_NNN/):
    traces.csv      trace_assignment traces (filtered) + category + prediction columns
    accuracy.csv    slide x category x subset (all, kept): n, correct, accuracy, Wilson and
                    FOV-bootstrap intervals, predicted SNAP fraction
    paper_in.csv    the paper-IN sets (ground-truth mode, 4 px) from the classify summary
    manifest.yaml, config.yaml, code/

Usage:
    python Revisions/DFK785/transfer.py [--config path]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Bleedthrough"))
sys.path.insert(0, str(HERE))
from common import REPO_ROOT, STR_COLUMNS, load_config, make_run_dir, rel_to_repo, resolve_run, setup_logging, \
    write_manifest  # noqa: E402
from overview import categorize  # noqa: E402

MODEL_CLASS = {"HT": "HaloD106", "SNAP": "SNAPC148"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fov_bootstrap(correct: pd.Series, fov: pd.Series, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile 95 % interval of the pooled accuracy with FOVs resampled with replacement."""
    g = pd.DataFrame({"c": correct.astype(float), "f": fov}).groupby("f")["c"].agg(["sum", "size"])
    if len(g) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    s, n = g["sum"].to_numpy()[idx].sum(axis=1), g["size"].to_numpy()[idx].sum(axis=1)
    acc = s / np.where(n > 0, n, np.nan)
    return tuple(np.nanpercentile(acc, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    tc = cfg["transfer"]
    slides = cfg["slides"]
    rng = np.random.default_rng(int(tc["seed"]))

    cl_run = REPO_ROOT / tc["classify_run"]
    pred = pd.read_csv(cl_run / "predictions.csv", dtype={"set": str, "label": str})
    ta_run = resolve_run(cfg, "trace_assignment", tc["trace_assignment_run"])
    with open(ta_run / "config.yaml") as f:
        ta_cfg = yaml.safe_load(f)
    um_run = resolve_run(cfg, "unmixing", ta_cfg["trace_assignment"]["unmixing_run"])
    ves = pd.read_csv(um_run / "labels.csv", dtype=STR_COLUMNS)
    tr = pd.read_csv(ta_run / "traces.csv", dtype=STR_COLUMNS)
    tr = tr[tr["filtered"].astype(str) == "True"]
    t = categorize(tr, ves, slides, float(cfg["occupancy"]["crowding_radius_px"]),
                   float(cfg["detection"]["border_px"]), int(cfg["overview"]["image_width_px"]))

    keys = ta_cfg["trace_assignment"]["extraction_keys"]
    p = pred[pred["set"].isin(keys)].rename(columns={"set": "extraction_key"})
    m = t.merge(p, on=["extraction_key", "uniqueID"], how="left", indicator=True)
    unmatched_pred = len(p) - int((m["_merge"] == "both").sum())
    if (m["_merge"] != "both").any():
        raise ValueError(f"{int((m['_merge'] != 'both').sum())} filtered traces have no prediction")
    logging.info(f"{len(m)} filtered traces joined; {unmatched_pred} predictions without a trace "
                 f"(dropped duplicate boxes)")
    m["kept"] = m["kept"].astype(str) == "True"
    m["truth"] = m["protein"].map(MODEL_CLASS)            # NaN for 'unknown' (mixed, no single-dye label)
    m["correct"] = m["prediction"] == m["truth"]

    run = make_run_dir(cfg, "transfer")
    m.drop(columns=["_merge"]).to_csv(run / "traces.csv", index=False)

    rows = []
    levels = {"by category": (m, ["slide", "experiment", "protein", "category"]),
              "single-dye IN": (m[m["category"].isin(["clean", "crowded", "wrong label"])],
                                ["slide", "experiment", "protein"]),
              "all filtered": (m, ["slide", "experiment"])}
    for level, (sub, by) in levels.items():
        for key, g in sub.groupby(by):
            info = dict(zip(by, key if isinstance(key, tuple) else (key,)))
            for subset, gg in (("all", g), ("kept", g[g["kept"]])):
                r = {"level": level, **info, "subset": subset, "n": len(gg),
                     "frac_pred_SNAPC148": (gg["prediction"] == "SNAPC148").mean() if len(gg) else np.nan}
                lab = gg[gg["truth"].notna()]
                # on a mixed slide only the single-dye traces have a truth; 'all filtered' there is composition only
                if len(lab) and not (level == "all filtered" and info["experiment"] == "mixed"):
                    k, n = int(lab["correct"].sum()), len(lab)
                    r.update({"n_with_truth": n, "correct": k, "accuracy": k / n})
                    r["wilson_lo"], r["wilson_hi"] = wilson(k, n)
                    r["fov_boot_lo"], r["fov_boot_hi"] = fov_bootstrap(lab["correct"], lab["fov"],
                                                                       int(tc["n_boot"]), rng)
                rows.append(r)
    acc = pd.DataFrame(rows)
    acc.to_csv(run / "accuracy.csv", index=False)

    cs = pd.read_csv(cl_run / "summary.csv")
    paper = cs[cs["set"].isin(tc["paper_in_sets"])]
    paper.to_csv(run / "paper_in.csv", index=False)

    with pd.option_context("display.width", 240, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        cols = ["level", "slide", "protein", "category", "subset", "n", "n_with_truth", "accuracy", "wilson_lo",
                "wilson_hi", "fov_boot_lo", "fov_boot_hi", "frac_pred_SNAPC148"]
        logging.info("Accuracy:\n" + acc.reindex(columns=cols).to_string(index=False))
        logging.info("Paper-IN sets:\n" + paper[["set", "n", "n_kept", "kept_frac", "accuracy", "ci95_lo", "ci95_hi",
                                                 "accuracy_kept", "ci95_lo_kept", "ci95_hi_kept"]].to_string(index=False))
    write_manifest(run, cfg, Path(__file__), [cl_run / "predictions.csv", ta_run / "traces.csv", um_run / "labels.csv"],
                   extra={"classify_run": rel_to_repo(cl_run), "trace_assignment_run": rel_to_repo(ta_run)})


if __name__ == "__main__":
    main()
