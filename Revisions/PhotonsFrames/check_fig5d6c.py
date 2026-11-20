#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Check that this topic's event definition reproduces the published features of Figs. 5D and 6C.

For the final models' test traces (GeneralData MLresults/FinalModels/*/MCD_results/
traces_with_wasserstein.npz), rebuilt from the published intensities (../PaperData local mirror):
  1. the min-max channel the final models saw (min-max over 6000 frames, first 4000 frames) equals the
     npz's channel 0;
  2. mean OFF time, mean ON time and duty cycle per trace (events.py's classifier and runs, 30 ms per
     frame as published) of the traces with WD >= 0.7, grouped by true label as Features/features.py does,
     equal the published panel values (GeneralData MLresults/Features/*/features/data_panel_{A,B,C}.csv,
     compared as sorted lists because the panels carry no trace IDs).

Outputs: Results/Revisions/PhotonsFrames/check_fig5d6c/run_NNN/ (check.json, manifest, config, code).

Usage (repo root):
    python Revisions/PhotonsFrames/check_fig5d6c.py --general-data <GeneralData>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "PaperData"))
from events import load_topic_config, runs_of  # noqa: E402
from paperdata import load_config, load_traces, make_run_dir, write_manifest  # noqa: E402
from utils import gmm_classify_frames  # noqa: E402  (Extraction/utils.py, on sys.path via events)

PUBLISHED_FRAME_MS = 30.0            # the published features' conversion (Features/config.yaml)
FINAL = {"HaloD106_HaloK117": ("2026-05-19_08-21-58_orig_conv_gru_mirrored_IN_training_HaloD106_HaloK117",
                               "HaloD106_HaloK117_mcd_20260519_102602", {"HTHTL": "HaloD106", "HTIA": "HaloK117"}),
         "scGrx1_scGrx1AcK20": ("2026-05-19_10-14-25_tcn_mirrored_IN_training_scGrx1_scGrx1AcK20",
                                "scGrx1_scGrx1AcK20_mcd_20260519_103909", {"Grx1": "scGrx1", "K20Ac": "scGrx1AcK20"})}
PANELS = {"A": "mean_off_ms", "B": "mean_on_ms", "C": "duty_cycle"}
WD_MIN = 0.7
N_FRAMES_MODEL = 4000


def features(minmax: np.ndarray, gmm_proba: float, min_width: int) -> dict:
    _, _, on = gmm_classify_frames(minmax, gmm_proba, 0.0)
    runs = runs_of(on, min_width)
    dur = [e - s + 1 for s, e in runs]
    offs = [runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)]
    active = runs[-1][1] - runs[0][0] + 1 if runs else np.nan
    return {"mean_on_ms": np.mean(dur) * PUBLISHED_FRAME_MS if dur else np.nan,
            "mean_off_ms": np.mean(offs) * PUBLISHED_FRAME_MS if offs else np.nan,
            "duty_cycle": on.sum() / active if runs else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--general-data", type=Path, required=True)
    args = ap.parse_args()
    tcfg, pcfg = load_topic_config(), load_config(local=True)
    run = make_run_dir(pcfg, "check_fig5d6c", topic_root=tcfg["output_root"])
    out, inputs = {}, []
    for pair, (model, feat_run, keymap) in FINAL.items():
        npz_p = args.general_data / "MLresults" / "FinalModels" / model / "MCD_results" / "traces_with_wasserstein.npz"
        inputs.append(npz_p)
        z = np.load(npz_p)
        names = [keymap[str(c)] for c in z["class_names"]]
        bg = {c: load_traces(pcfg, c, "bg_rm") for c in names}
        max_ch0, rows = 0.0, []
        for i, uid in enumerate(z["unique_ids"]):
            c = names[int(z["labels"][i])]
            x = bg[c][int(uid)].to_numpy(np.float64)
            mm = ((x - x.min()) / (x.max() - x.min()))[:N_FRAMES_MODEL]
            max_ch0 = max(max_ch0, float(np.abs(mm - z["traces"][i, 0]).max()))
            if z["wasserstein_distances"][i] >= WD_MIN:
                rows.append({"protein": c, **features(mm, tcfg["gmm_proba"], tcfg["min_peak_width"])})
        mine = pd.DataFrame(rows)
        res = {"npz_channel0_max_abs_diff": max_ch0}
        for letter, feat in PANELS.items():
            p = args.general_data / "MLresults" / "Features" / feat_run / "features" / f"data_panel_{letter}.csv"
            inputs.append(p)
            pub = pd.read_csv(p)
            for c in names:
                a = np.sort(pub[c].dropna().to_numpy())
                b = np.sort(mine.loc[mine["protein"] == c, feat].dropna().to_numpy())
                res[f"{feat}_{c}"] = {"n_published": int(a.size), "n_recomputed": int(b.size),
                                      "max_abs_diff_sorted": float(np.abs(a - b).max()) if a.size == b.size else None}
        out[pair] = res
    with open(run / "check.json", "w") as f:
        json.dump(out, f, indent=2)
    write_manifest(run, {**tcfg, "general_data": str(args.general_data)}, Path(__file__), inputs)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
