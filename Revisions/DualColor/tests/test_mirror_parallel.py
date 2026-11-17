#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
train_pure.mirror_augment (augmentation.mirror_n_jobs, TRAINING_NOTES.md point 11) gives exactly
the output of ML/utils.apply_augmentation(aug_factor=0, include_mirror=True): the same traces in
the same order with the same labels, for 1 and several processes, on real minmax traces (the first
filtered traces of DFK785 slide 1 and slide 2) plus traces with no ON frame (flat, so the whole
trace is reversed). A mirror_augment that drops the shuffle, shuffles with another draw, or
mirrors a different window fails here.

    python Revisions/DualColor/tests/test_mirror_parallel.py [n_traces]
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import train_pure as tp  # noqa: E402
from utils import apply_augmentation  # noqa: E402  (ML/, put on sys.path by train_pure)


def real_traces(n: int) -> np.ndarray:
    ext = sorted((tp.REPO_ROOT / "Results/Revisions/DualColor/DFK785/Extraction").glob("*_optparam_*"))[-1]
    parts = []
    for key in ("Slide1SNAP", "Slide2HT"):
        df = pd.read_pickle(ext / "ProteinTraces" / "Filtered" / f"{key}_filtered_minmax.pkl")
        parts.append(df.iloc[:, : n // 2].to_numpy(np.float32).T)
    return np.concatenate(parts)[:, None, :]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    X = real_traces(n)
    X = np.concatenate([X, np.full((2, 1, X.shape[-1]), 0.3, np.float32)])   # flat: no ON frame
    y = np.arange(len(X)) % 2
    for seed in (840410, 7):
        with contextlib.redirect_stdout(io.StringIO()):
            Xr, yr = apply_augmentation(X, y, aug_factor=0, include_mirror=True, random_seed=seed)
        for n_jobs in (1, 4):
            with contextlib.redirect_stdout(io.StringIO()):
                Xp, yp = tp.mirror_augment(X, y, seed, n_jobs)
            assert Xp.shape == Xr.shape and np.array_equal(yp, yr), f"seed {seed}, {n_jobs} jobs: labels/order differ"
            assert np.array_equal(Xp, Xr), f"seed {seed}, {n_jobs} jobs: traces differ"
        print(f"seed {seed}: {len(X)} traces -> {len(Xr)}, identical for 1 and 4 processes")
    print("OK")


if __name__ == "__main__":
    main()
