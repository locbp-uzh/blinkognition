#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
The `clipmax` trace normalization (TRAINING_NOTES.md point 12) for the filtered traces of one dual-color
extraction run: <key>_filtered_clipmax.pkl next to the run's <key>_filtered_minmax.pkl, same frames and
uniqueIDs.

Ported from the blinkognition2 pipeline (gitlab locbp/blinkognition2, branch clarissa-local, commit
64f1a48, Extraction/combine.py compute_extra_norms, the 'clipmax' entry), per trace of the
background-removed matrix (<key>_filtered_bg_rm.pkl):
1. signal mask: Extraction/utils.gmm_classify_frames (2-component GMM; identical source in both repos)
   with proba_threshold = the run config's gmm_proba_threshold and min_separation = its
   snr_min_separation, i.e. the same on/off decision as the trace filter (0.9 and 2.0 in every
   DualColor config; blinkognition2's own config used 0.6 and 2). A trace whose GMM components are
   separated by less than min_separation noise SDs gets an empty mask.
2. every non-signal frame and every negative value -> 0;
3. divide by the trace's maximum, so signal frames lie in (0, 1] and the rest is exactly 0; an empty
   mask gives an all-zero trace (counted and reported), never NaN.

    picasso-env or blink2env python Revisions/DualColor/clipmax.py <dataset> [--trace-assignment-run run_NNN] [--n-jobs 14]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc import REPO_ROOT, load_config, run_dir  # noqa: E402

_spec = importlib.util.spec_from_file_location("extraction_utils", REPO_ROOT / "Extraction" / "utils.py")
xu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xu)


def signal_masks(block: np.ndarray, proba: float, sep: float) -> np.ndarray:
    """(T, n) boolean signal masks of the columns of block (gmm_classify_frames' signal mask)."""
    return np.stack([xu.gmm_classify_frames(block[:, j], proba, sep)[2] for j in range(block.shape[1])], axis=1)


def clipmax(bg: pd.DataFrame, proba: float, sep: float, n_jobs: int) -> tuple[pd.DataFrame, int]:
    """blinkognition2 compute_extra_norms 'clipmax' on a (frames x traces) bg_rm matrix."""
    arr = bg.to_numpy(dtype=np.float64)
    chunks = np.array_split(np.arange(arr.shape[1]), max(1, min(arr.shape[1], 8 * n_jobs)))
    parts = Parallel(n_jobs=n_jobs)(delayed(signal_masks)(arr[:, c], proba, sep) for c in chunks if len(c))
    mask = pd.DataFrame(np.concatenate(parts, axis=1), index=bg.index, columns=bg.columns)
    clipped = bg.where(mask, other=0.0).clip(lower=0.0)
    on_max = clipped.max(axis=0)
    n_zeroed = int((on_max == 0).sum())
    on_max = on_max.where(on_max != 0, other=1.0)
    return (clipped / on_max).fillna(0.0), n_zeroed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--trace-assignment-run", default=None, help="the run whose manifest names the extraction run")
    ap.add_argument("--n-jobs", type=int, default=14)
    a = ap.parse_args()
    cfg = load_config(a.dataset)
    ta = run_dir(cfg, "trace_assignment", a.trace_assignment_run)
    ext = REPO_ROOT / yaml.safe_load(open(ta / "manifest.yaml"))["extraction_run"]
    xcfg = yaml.safe_load(open(HERE / "extraction" / f"config_640only_{a.dataset}.yaml"))
    proba, sep = float(xcfg["gmm_proba_threshold"]), float(xcfg["snr_min_separation"])
    print(f"{a.dataset}: {ext.relative_to(REPO_ROOT)}; gmm_proba_threshold {proba}, min_separation {sep}")
    for f in sorted((ext / "ProteinTraces" / "Filtered").glob("*_filtered_bg_rm.pkl")):
        if f.name.startswith("._"):
            continue
        bg = pd.read_pickle(f)
        mm = pd.read_pickle(f.with_name(f.name.replace("_bg_rm", "_minmax")))
        if list(bg.columns) != list(mm.columns) or bg.shape != mm.shape:
            raise ValueError(f"{f.name}: bg_rm and minmax differ in traces or frames")
        cm, n_zeroed = clipmax(bg, proba, sep, a.n_jobs)
        out = f.with_name(f.name.replace("_bg_rm", "_clipmax"))
        cm.to_pickle(out)
        v = cm.to_numpy()
        print(f"  {out.name}: {v.shape[1]} traces, zero frames {np.mean(v == 0):.4f}, "
              f"empty masks {n_zeroed}, max per trace {np.unique(v.max(axis=0).round(6))[:3]}")


if __name__ == "__main__":
    main()
