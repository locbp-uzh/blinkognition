#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
FrameZero: the paper's final models on the recovered molecules (ON in the first frame, dropped by the published
extraction) that pass the quality filter (compare.py's recovered pickles).

ML/classify.py (unchanged) with the final model of Fig. 5 (Halo-D106 vs Halo-K117, CNN-GRU) on the recovered
Halo-D106 and Halo-K117 traces, and that of Fig. 6 (scGrx1 vs scGrx1-AcK20, TCN) on the recovered scGrx1 and
scGrx1-AcK20 traces, each set with its true label: first 4000 frames, both channels, n_mc 100, the paper's
Wasserstein threshold 0.7 (fixed). These molecules were never seen in training (they were not in the published
dataset). SNAP-C148 has no final model. classify.py reports per set the kept fraction and the accuracy with a
Wilson 95 % interval, for all traces and for the kept ones.

Outputs (Results/Revisions/FrameZero/classify/run_NNN/): classify_<model>/ (ML/classify.py's run folders),
summary.csv, predictions.csv, manifest.yaml, config.yaml, code/.

Usage (repo root, an environment with torch, e.g. blink2env):
    python Revisions/FrameZero/classify_recovered.py --compare Results/Revisions/FrameZero/s3it/compare/run_NNN
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "PaperData"))
from paperdata import load_config as load_paper_config, make_run_dir, write_manifest  # noqa: E402

MODELS = {"HaloD106_HaloK117": ("2026-05-19_08-21-58_orig_conv_gru_mirrored_IN_training_HaloD106_HaloK117",
                                ["HaloD106", "HaloK117"]),
          "scGrx1_scGrx1AcK20": ("2026-05-19_10-14-25_tcn_mirrored_IN_training_scGrx1_scGrx1AcK20",
                                 ["scGrx1", "scGrx1AcK20"])}
THRESHOLD = 0.7
N_MC = 100


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", type=Path, required=True, help="compare.py run folder (repo-relative)")
    args = ap.parse_args()
    with open(HERE / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(HERE / "config.yaml")
    pcfg = load_paper_config()
    fdir = (REPO_ROOT / args.compare / "recovered" / "ProteinTracesIN" / "Filtered").resolve()
    final_root = Path(pcfg["general_data_local"]) / "MLresults" / "FinalModels"
    out = make_run_dir(pcfg, "classify", topic_root=cfg["output_root"])
    preds, summs = [], []
    for model, (mdir, classes) in MODELS.items():
        sets = []
        for c in classes:
            if pd.read_pickle(fdir / f"{c}_IN_filtered_minmax.pkl").shape[1]:
                sets.append({"name": f"recovered|{c}", "path": str(fdir), "key": f"{c}_IN_filtered", "label": c})
        if not sets:
            continue
        mcfg = {"model_dir": str(final_root / mdir), "run_name": f"framezero_{model}",
                "output_root": str(out / f"classify_{model}"), "trace_sets": sets,
                "mc_dropout": {"n_mc": N_MC, "wasserstein_threshold": THRESHOLD}, "system": {"seed": 840410}}
        cpath = out / f"config_classify_{model}.yaml"
        with open(cpath, "w") as f:
            yaml.safe_dump(mcfg, f, sort_keys=False, allow_unicode=True)
        subprocess.run([sys.executable, "classify.py", "-c", str(cpath)], cwd=REPO_ROOT / "ML", check=True)
        rdir = sorted((out / f"classify_{model}").iterdir())[-1]
        for kind, store in (("predictions", preds), ("summary", summs)):
            d = pd.read_csv(rdir / f"{kind}.csv")
            d.insert(0, "model", model)
            store.append(d)
    pd.concat(preds).to_csv(out / "predictions.csv", index=False, float_format="%.5g")
    summ = pd.concat(summs)
    summ.to_csv(out / "summary.csv", index=False, float_format="%.4g")
    write_manifest(out, cfg, Path(__file__), [fdir / p.name for p in sorted(fdir.glob("*.pkl"))], hash_inputs=True,
                   extra={"compare_run": str(args.compare), "threshold": THRESHOLD, "n_mc": N_MC})
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
