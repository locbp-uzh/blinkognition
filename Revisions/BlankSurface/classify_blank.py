#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
R3.1: the paper's final models applied to every trace that passed the quality filter on the blank, empty-vesicle
and same-power protein slides (design confirmed by the user 2026-11-20: "passing blank traces, if any,
classified by the final models").

For each final model in config.yaml `classify` (Halo-D106 vs Halo-K117, CNN-GRU; scGrx1 vs scGrx1-AcK20, TCN),
ML/classify.py (unchanged) runs on every non-empty set of filtered traces (vesicle channel x gradient x slide x
IN/OUT), with the model's channels (min-max and z-scored), trimmed to its training length (4000 frames), MC
dropout with n_mc passes, and the paper's Wasserstein threshold (0.7, fixed; never tuned on these traces). No set
has a true class of either model, so there is no accuracy: the question is how many impurity traces a model
would keep with confidence, against the protein slides' traces.

Outputs (Results/Revisions/BlankSurface/classify/run_NNN/): classify_<model>/ (ML/classify.py's own run folder:
predictions.csv, summary.csv, log, config), predictions.csv (all models, joined with count.py's per-trace
table: session, field, position, filter replay values), summary.csv (per model, run, slide, IN/OUT: traces,
kept, kept fraction, predicted classes among kept), manifest.yaml, config.yaml, code/.

Usage (repo root, an environment with torch, e.g. blink2env):
    python Revisions/BlankSurface/classify_blank.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "PaperData"))
from paperdata import load_config as load_paper_config, make_run_dir, write_manifest  # noqa: E402


def trace_sets(root: Path) -> list[dict]:
    sets = []
    for tag_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("._")):
        for where in ("IN", "OUT"):
            fdir = tag_dir / f"ProteinTraces{where}" / "Filtered"
            for mm in sorted(fdir.glob(f"*_{where}_filtered_minmax.pkl")):
                if mm.name.startswith("._"):
                    continue
                slide = mm.name[: -len(f"_{where}_filtered_minmax.pkl")]
                if pd.read_pickle(mm).shape[1] == 0:
                    continue
                sets.append({"name": f"{tag_dir.name}|{slide}|{where}", "path": str(fdir.resolve()),
                             "key": f"{slide}_{where}_filtered"})
    return sets


def main() -> None:
    with open(HERE / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(HERE / "config.yaml")
    cc = cfg["classify"]
    pcfg = load_paper_config()
    out = make_run_dir(pcfg, "classify", topic_root=cfg["output_root"])
    sets = trace_sets(REPO_ROOT / cc["filtered_root"])
    print(f"{len(sets)} non-empty trace sets", flush=True)
    preds, summs = [], []
    for model, mdir in cc["final_models"].items():
        mcfg = {"model_dir": mdir, "run_name": f"blanksurface_{model}", "output_root": str(out / f"classify_{model}"),
                "trace_sets": sets, "mc_dropout": {"n_mc": int(cc["n_mc"]),
                                                   "wasserstein_threshold": float(cc["wasserstein_threshold"])},
                "system": {"seed": 840410}}
        cpath = out / f"config_classify_{model}.yaml"
        with open(cpath, "w") as f:
            yaml.safe_dump(mcfg, f, sort_keys=False, allow_unicode=True)
        subprocess.run([sys.executable, "classify.py", "-c", str(cpath)], cwd=REPO_ROOT / "ML", check=True)
        rdir = sorted((out / f"classify_{model}").iterdir())[-1]
        p = pd.read_csv(rdir / "predictions.csv")
        p.insert(0, "model", model)
        preds.append(p)
        s = pd.read_csv(rdir / "summary.csv")
        s.insert(0, "model", model)
        summs.append(s)
    pred = pd.concat(preds, ignore_index=True)
    parts = pred["set"].str.split("|", expand=True)
    pred["tag"], pred["slide"], pred["where"] = parts[0], parts[1], parts[2]
    pred["gt"] = pred["tag"].str.extract(r"gt(\d+)_")[0].astype(int)
    pred["gradient"] = pred["tag"].str.extract(r"_g(\d+)$")[0].astype(int)
    tr = pd.read_csv(REPO_ROOT / cc["count_run"] / "traces.csv")
    key = ["gt", "gradient", "slide", "where", "uniqueID"]
    pred = pred.merge(tr, on=key, how="left", validate="many_to_one")
    if pred["session"].isna().any():
        raise SystemExit("predictions without a count.py trace row")
    if not pred["passed_pipeline"].all():
        raise SystemExit("classified a trace that did not pass the filter")
    pred.to_csv(out / "predictions.csv", index=False, float_format="%.5g")
    roles = {s["name"]: s["role"] for s in cfg["slides"]}
    rows = []
    for (model, gt, grad, session, slide, where), g in pred.groupby(["model", "gt", "gradient", "session", "slide", "where"]):
        k = g[g["kept"]]
        r = {"model": model, "gt": gt, "gradient": grad, "session": session, "slide": slide, "role": roles[slide],
             "where": where, "n": len(g), "n_kept": len(k), "kept_frac": len(k) / len(g)}
        for c in sorted(pred["prediction"].unique()):
            r[f"n_kept_pred_{c}"] = int((k["prediction"] == c).sum())
        rows.append(r)
    summ = pd.DataFrame(rows).fillna(0)
    summ.to_csv(out / "summary.csv", index=False, float_format="%.4g")
    write_manifest(out, cfg, Path(__file__), [REPO_ROOT / cc["count_run"] / "traces.csv"], hash_inputs=True,
                   extra={"n_sets": len(sets), "models": cc["final_models"]})
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
