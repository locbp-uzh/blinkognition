#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Revisions/DualColor/clipmax.py reproduces blinkognition2's clipmax exactly (TRAINING_NOTES.md point 12).

The reference is the original code, read from the local blinkognition2 clone at commit 64f1a48
(Extraction/combine.py compute_extra_norms and its helpers, Extraction/utils.py gmm_signal_mask /
gmm_classify_frames), run with clipmax on. Compared on real filtered bg_rm traces of DFK785 (both
single-protein slides) plus an all-noise trace (empty mask -> all zero) and a trace shifted down so that
signal frames are negative (they must clip to 0; a port without the clip fails here): identical frames, uniqueIDs and values (np.array_equal), for 1 and 4 processes.

    python Revisions/DualColor/tests/test_clipmax_port.py [n_traces]
"""

import ast
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import clipmax as cm  # noqa: E402

B2 = Path("/Volumes/MediumBerth/Agent/repos/Lab/blinkognition2")
COMMIT = "64f1a48"


def reference():
    """compute_extra_norms of blinkognition2 at COMMIT, with its own utils module."""
    utils_src = subprocess.run(["git", "-C", str(B2), "show", f"{COMMIT}:Extraction/utils.py"],
                               capture_output=True, text=True, check=True).stdout
    u = types.ModuleType("b2_utils")
    exec(compile(utils_src, "b2_utils.py", "exec"), u.__dict__)
    comb = subprocess.run(["git", "-C", str(B2), "show", f"{COMMIT}:Extraction/combine.py"],
                          capture_output=True, text=True, check=True).stdout
    tree = ast.parse(comb)
    want = {"_snr_norm_col", "_signal_mask_col", "compute_extra_norms"}
    src = "\n\n".join(ast.get_source_segment(comb, n) for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name in want)
    import logging
    from typing import Dict
    from joblib import Parallel, delayed
    ns = {"np": np, "pd": pd, "logging": logging, "Dict": Dict, "Parallel": Parallel, "delayed": delayed,
          "gmm_signal_mask": u.gmm_signal_mask, "snr_normalize_trace": u.snr_normalize_trace}
    exec(compile(src, "b2_combine_extract.py", "exec"), ns)
    return ns["compute_extra_norms"]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    ext = sorted((cm.REPO_ROOT / "Results/Revisions/DualColor/DFK785/Extraction").glob("*_optparam_003"))[-1]
    parts = [pd.read_pickle(ext / "ProteinTraces" / "Filtered" / f"{k}_filtered_bg_rm.pkl").iloc[:, : n // 2]
             .add_prefix(f"{k}_") for k in ("Slide1SNAP", "Slide2HT")]   # uniqueIDs repeat across keys
    bg = pd.concat(parts, axis=1)
    rng = np.random.default_rng(0)
    bg["noise_only"] = rng.normal(0, 1, len(bg))                    # unimodal: empty mask expected
    # shifted down by its 99.5th percentile: the GMM mask is shift-invariant, so many SIGNAL frames are now
    # negative and must be clipped to 0 (on real bg_rm traces the mask alone removes the negatives)
    x0 = bg.iloc[:, 0].to_numpy()
    bg["with_negatives"] = x0 - np.percentile(x0, 99.5)
    ref_fn = reference()
    for proba, sep in ((0.9, 2.0), (0.6, 2.0)):
        ref = ref_fn(bg, {"clipmax": True}, n_jobs=1, proba_threshold=proba, min_separation=sep)["clipmax"]
        for nj in (1, 4):
            got, n_zeroed = cm.clipmax(bg, proba, sep, nj)
            assert list(got.columns) == list(ref.columns) and list(got.index) == list(ref.index)
            assert np.array_equal(got.to_numpy(), ref.to_numpy()), f"values differ (proba {proba}, {nj} jobs)"
        assert (got["noise_only"] == 0).all(), "a unimodal noise trace must clip to all zero"
        assert (got["with_negatives"] > 0).sum() < (got.iloc[:, 0] > 0).sum(), "negative signal frames must clip to 0"
        print(f"proba {proba}, sep {sep}: {bg.shape[1]} traces identical to blinkognition2 {COMMIT} "
              f"(1 and 4 processes); zero frames {np.mean(got.to_numpy() == 0):.4f}; empty masks {n_zeroed}")
    print("OK")


if __name__ == "__main__":
    main()
