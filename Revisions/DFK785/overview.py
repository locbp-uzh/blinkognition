#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Overview of the DFK785 data: usable FOVs, vesicle crowding and the provenance of the
fully filtered protein traces of a trace_assignment run.

Every filtered trace (Extraction filters passed) gets one category:

    clean          IN a single-dye vesicle (ATTO390 or ATTO520) whose nearest other vesicle is
                   at least occupancy.crowding_radius_px away (clean aperture, 5 px)
    crowded        IN a single-dye vesicle with a neighbor closer than that (its label and
                   photometry can carry the neighbor's light)
    wrong label    single-label slides only: IN a vesicle labeled with the other dye
    dual           IN a vesicle labeled with both dyes
    no label       IN a vesicle with neither dye above threshold
    ambiguous      two vesicles within the match radius
    OUT border     no vesicle within the radius, center within detection.border_px of the
                   edge (the vesicle table has no vesicles there)
    OUT interior   no vesicle within the radius elsewhere

Protein: on the single-label slides the slide's protein for every trace; on the mixed
slides HT for ATTO390-labeled and SNAP for ATTO520-labeled traces, unknown otherwise.

Expected wrong labels on the mixed slides (mixed_expected_errors.csv), per labeled class:
    chance      chance coincidences from the trace_assignment null.csv at the match radius
                (upper bound: a chance hit is only wrong if the protein is the other one)
    unmixing    the other protein's traces in vesicles mislabeled as this dye, at the rate
                seen on the single-label slides (wrong label / correct label there) times
                the other class's real count (observed - chance)

Outputs (Results/Revisions/DFK785/overview/run_NNN/):
    fovs.csv, crowding.csv, trace_categories.csv (long), trace_summary.csv (wide),
    mixed_expected_errors.csv, manifest.yaml, config.yaml, code/

Usage:
    python Revisions/DFK785/overview.py [--config path]
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
from common import REPO_ROOT, STR_COLUMNS, load_config, make_run_dir, rel_to_repo, resolve_run, setup_logging, \
    write_manifest  # noqa: E402

PROTEIN = {"ATTO390": "HT", "ATTO520": "SNAP"}
CATEGORIES = ["clean", "crowded", "wrong label", "dual", "no label", "ambiguous", "OUT border", "OUT interior"]


def categorize(tr: pd.DataFrame, ves: pd.DataFrame, slides: dict, crowd_px: float, border_px: float,
               width: int) -> pd.DataFrame:
    nn = ves.set_index(["slide", "fov", "vesicle_id"])["nn_dist_px"]
    t = tr.copy()
    key = pd.MultiIndex.from_frame(t[["slide", "fov", "vesicle_id"]])
    t["vesicle_nn_px"] = nn.reindex(key).to_numpy()
    edge = np.minimum.reduce([t["row"], t["col"], width - 1 - t["row"], width - 1 - t["col"]])
    dye = t["slide"].map(lambda s: slides[s]["dye"])
    single = t["class"].isin(["IN_ATTO390", "IN_ATTO520"])
    lab = t["class"].str.replace("IN_", "", regex=False)
    cat = pd.Series("OUT interior", index=t.index)
    cat[(t["class"] == "OUT") & (edge < border_px)] = "OUT border"
    cat[t["class"] == "ambiguous"] = "ambiguous"
    cat[t["class"] == "IN_dual"] = "dual"
    cat[t["class"] == "IN_no label"] = "no label"
    cat[single] = np.where(t.loc[single, "vesicle_nn_px"] >= crowd_px, "clean", "crowded")
    cat[single & (dye != "mix") & (lab != dye)] = "wrong label"
    t["category"] = cat
    t["experiment"] = np.where(dye == "mix", "mixed", "pure")
    t["protein"] = np.where(dye == "mix", lab.map(PROTEIN).fillna("unknown"), dye.map(PROTEIN))
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    oc = cfg["overview"]
    slides = cfg["slides"]

    ta_run = resolve_run(cfg, "trace_assignment", oc["trace_assignment_run"])
    with open(ta_run / "config.yaml") as f:
        ta_cfg = yaml.safe_load(f)
    radius = float(ta_cfg["trace_assignment"]["match_radius_px"])
    um_run = resolve_run(cfg, "unmixing", ta_cfg["trace_assignment"]["unmixing_run"])
    with open(um_run / "manifest.yaml") as f:
        ves_run = resolve_run(cfg, "vesicles", yaml.safe_load(f)["vesicles_run"])
    ves = pd.read_csv(um_run / "labels.csv", dtype=STR_COLUMNS)
    fov_table = pd.read_csv(ves_run / "fovs.csv", dtype=STR_COLUMNS)
    moved = pd.read_csv(Path(cfg["input_root"]) / "bad_data" / "moved_files.csv", dtype=STR_COLUMNS)
    tr = pd.read_csv(ta_run / "traces.csv", dtype=STR_COLUMNS)
    null = pd.read_csv(ta_run / "null.csv", dtype=STR_COLUMNS)
    crowd_px = float(cfg["occupancy"]["crowding_radius_px"])

    run = make_run_dir(cfg, "overview")

    # FOVs
    dried = moved.groupby("slide")["acq_order"].nunique()
    fovs = pd.DataFrame([{"slide": s, "experiment": "mixed" if d["dye"] == "mix" else "pure", "sample": d["sample"],
                          "usable": int((fov_table["slide"] == s).sum()), "dried_moved": int(dried.get(s, 0))}
                         for s, d in slides.items()])
    fovs["acquired"] = fovs["usable"] + fovs["dried_moved"]
    fovs.to_csv(run / "fovs.csv", index=False)

    # Crowding
    rows = []
    for s, v in ves.groupby("slide"):
        per_fov = v.groupby("fov").size()
        shares = v["label"].value_counts(normalize=True)
        rows.append({"slide": s, "vesicles": len(v), "per_fov_median": per_fov.median(),
                     "per_fov_min": per_fov.min(), "per_fov_max": per_fov.max(),
                     "nn_median_px": v["nn_dist_px"].median(),
                     f"frac_nn_lt_{crowd_px:g}px": (v["nn_dist_px"] < crowd_px).mean(),
                     "frac_nn_lt_8px": (v["nn_dist_px"] < 8).mean(),
                     **{f"share_{k}": shares.get(k, 0.0) for k in ["ATTO390", "ATTO520", "dual", "no label"]}})
    crowding = pd.DataFrame(rows)
    crowding.to_csv(run / "crowding.csv", index=False)

    # Filtered traces by category
    t = categorize(tr[tr["filtered"].astype(str) == "True"], ves, slides, crowd_px,
                   float(cfg["detection"]["border_px"]), int(oc["image_width_px"]))
    long = t.groupby(["slide", "experiment", "protein", "category"]).size().rename("n").reset_index()
    long.to_csv(run / "trace_categories.csv", index=False)
    wide = (long.pivot_table(index=["experiment", "slide", "protein"], columns="category", values="n", fill_value=0)
            .reindex(columns=CATEGORIES, fill_value=0))
    wide["total"] = wide.sum(axis=1)
    wide = wide.reset_index()
    wide.to_csv(run / "trace_summary.csv", index=False)

    # Expected wrong labels on the mixed slides
    pure = long[long["experiment"] == "pure"]
    def rate(slide: str) -> float:
        p = pure[pure["slide"] == slide].set_index("category")["n"]
        correct = p.get("clean", 0) + p.get("crowded", 0)
        return p.get("wrong label", 0) / correct if correct else np.nan
    rate_into = {"SNAP": rate(next(s for s, d in slides.items() if d["dye"] == "ATTO390")),   # HT mislabeled ATTO520
                 "HT": rate(next(s for s, d in slides.items() if d["dye"] == "ATTO520"))}     # SNAP mislabeled ATTO390
    nr = null[null["radius_px"].astype(float) == radius]
    rows = []
    for s in [s for s, d in slides.items() if d["dye"] == "mix"]:
        obs = {p: int(((t["slide"] == s) & (t["protein"] == p) & t["category"].isin(["clean", "crowded"])).sum())
               for p in ["HT", "SNAP"]}
        chance = {PROTEIN[c.replace("IN_", "")]: float(x) for c, x in
                  nr[(nr["slide"] == s) & nr["class"].isin(["IN_ATTO390", "IN_ATTO520"])][["class", "chance_est"]]
                  .itertuples(index=False)}
        real = {p: obs[p] - chance[p] for p in obs}
        for p, other in [("HT", "SNAP"), ("SNAP", "HT")]:
            unmix = rate_into[p] * real[other]
            rows.append({"slide": s, "label_protein": p, "observed": obs[p], "chance_upper": round(chance[p], 1),
                         "unmixing_from_other": round(unmix, 1),
                         "expected_correct_min": round(obs[p] - chance[p] - unmix, 1),
                         "purity_min": round((obs[p] - chance[p] - unmix) / obs[p], 3) if obs[p] else np.nan,
                         "unmixing_rate_used": round(rate_into[p], 4)})
    errs = pd.DataFrame(rows)
    errs.to_csv(run / "mixed_expected_errors.csv", index=False)

    with pd.option_context("display.width", 220, "display.float_format", "{:.3g}".format):
        logging.info(f"Match radius {radius:g} px, crowding cutoff {crowd_px:g} px")
        logging.info("FOVs:\n" + fovs.to_string(index=False))
        logging.info("Crowding:\n" + crowding.to_string(index=False))
        logging.info("Filtered traces:\n" + wide.to_string(index=False))
        logging.info("Mixed slides, expected wrong labels:\n" + errs.to_string(index=False))
    write_manifest(run, cfg, Path(__file__),
                   [ta_run / "traces.csv", ta_run / "null.csv", um_run / "labels.csv", ves_run / "fovs.csv",
                    Path(cfg["input_root"]) / "bad_data" / "moved_files.csv"],
                   extra={"trace_assignment_run": rel_to_repo(ta_run), "match_radius_px": radius})


if __name__ == "__main__":
    main()
