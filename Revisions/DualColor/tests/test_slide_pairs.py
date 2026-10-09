#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
evaluate.slide_pairs: every two single-protein slides of a dataset exactly once, cross-protein
pairs oriented HT (a) -> SNAP (b), AUC matching scores that differ by slide. A first version
reassigned the slide names inside the inner loop and returned duplicates and missing pairs.

    python Revisions/DualColor/tests/test_slide_pairs.py
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import slide_pairs  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    # DFK788-like layout: SNAP slide1, HT slide2, SNAP slide4, HT slide5; slide k scores ~ k
    slides = {"slide1": "SNAP", "slide2": "HT", "slide4": "SNAP", "slide5": "HT"}
    rows = []
    for s, c in slides.items():
        for f in range(4):
            for _ in range(20):
                rows.append({"dataset": "D", "slide": s, "fov": f"{s}_F{f}", "true_class": c,
                             "p_SNAP": int(s[-1]) + rng.normal(0, 0.1)})
    rows.append({"dataset": "D", "slide": "slide3", "fov": "m", "true_class": None, "p_SNAP": 0.0})  # mixed: ignored
    sp = slide_pairs(pd.DataFrame(rows), "p_SNAP", n_boot=20, rng=rng)

    got = sorted(tuple(sorted(p)) for p in zip(sp["slide_a"], sp["slide_b"]))
    want = sorted(combinations(sorted(slides), 2))
    assert got == want, f"pairs {got} != {want}"
    cross = sp[sp["pair"] == "HT vs SNAP"]
    assert (cross["class_a"] == "HT").all() and (cross["class_b"] == "SNAP").all(), "cross pairs not HT -> SNAP"
    same = sp[sp["pair"] == "same protein"]
    assert set(map(tuple, same[["slide_a", "slide_b"]].values)) == {("slide1", "slide4"), ("slide2", "slide5")}
    # scores rise with the slide number: AUC 1 when b's number is higher, 0 when lower
    for r in sp.itertuples():
        expect = 1.0 if int(r.slide_b[-1]) > int(r.slide_a[-1]) else 0.0
        assert abs(r.auc - expect) < 1e-9, (r.slide_a, r.slide_b, r.auc)
    print(f"ok: {len(sp)} pairs\n{sp[['slide_a', 'class_a', 'slide_b', 'class_b', 'pair', 'auc']].to_string(index=False)}")


if __name__ == "__main__":
    main()
