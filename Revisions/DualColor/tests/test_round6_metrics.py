#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
round6_summary.eer_threshold and within_fov_auc (TRAINING_NOTES.md point 11):
1. eer_threshold: with HT scores uniform on [0, 1] and SNAP scores uniform on [0.5, 1.5], recall
   HT = t and recall SNAP = 1.5 - t, equal at t = 0.75; on a large sample the cut lands there and
   the two recalls at the returned cut differ by at most one trace's share. A threshold of 0.5
   (the argmax) gives recall HT 0.5 / SNAP 1.0 on the same data, so this case would fail an
   implementation that returned the argmax or swapped the classes.
2. within_fov_auc: two FOVs, each perfectly separated inside (AUC 1) but with SNAP of FOV a scoring
   below HT of FOV b, so the pooled AUC is below 1 while the within-FOV AUC is exactly 1; FOVs
   with one class only contribute no pairs; pair-count weighting checked on a FOV at AUC 0.
3. boot_groups / resample (FOV-cluster bootstrap): on a table of two slides, every group holds the
   rows of exactly one FOV (a first version took positions inside each slide as row numbers, so the
   second slide's groups pointed at the first slide's rows); every bootstrap copy of a FOV gets its
   own id, so a FOV drawn twice counts as two FOVs, not as one with four times the pairs.

    python Revisions/DualColor/tests/test_round6_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
from round6_summary import (boot_groups, draw, eer_threshold, fov_pair_stats, resample, rows_of,  # noqa: E402
                            within_fov_auc, within_of_draw)


def main() -> None:
    rng = np.random.default_rng(0)
    n = 20000
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    p = np.r_[rng.uniform(0, 1, n), rng.uniform(0.5, 1.5, n)]
    t = eer_threshold(y, p)
    rh, rs = np.mean(p[y == 0] < t), np.mean(p[y == 1] >= t)
    assert abs(t - 0.75) < 0.01, f"EER cut {t:.4f}, expected 0.75"
    assert abs(rh - rs) <= 1.0 / n + 1e-12, f"recalls at the cut {rh:.4f} / {rs:.4f}"
    print(f"eer_threshold: t = {t:.4f}, recall HT {rh:.4f}, SNAP {rs:.4f}")

    # within-FOV: a = HT 0.1, 0.2 | SNAP 0.3, 0.4 ; b = HT 0.5, 0.6 | SNAP 0.7, 0.8 ; c = one class only
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0, 0.05])
    fov = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 2)
    assert roc_auc_score(y, p) < 1
    assert within_fov_auc(y, p, fov) == 1.0, within_fov_auc(y, p, fov)
    # add FOV d (1 HT x 2 SNAP = 2 pairs) at AUC 0: within = (4 + 4 + 0) / (4 + 4 + 2) = 0.8
    y2, p2, f2 = np.r_[y, [0, 1, 1]], np.r_[p, [0.9, 0.1, 0.2]], np.r_[fov, ["d"] * 3]
    assert abs(within_fov_auc(y2, p2, f2) - 0.8) < 1e-12, within_fov_auc(y2, p2, f2)
    print("within_fov_auc: 1.0 with pooled AUC", round(roc_auc_score(y, p), 3), "; weighted 0.8 as expected")

    df = pd.DataFrame({"dataset": "D", "slide": ["s1"] * 5 + ["s2"] * 4,
                       "fov": ["a", "a", "b", "b", "b", "c", "c", "d", "d"]}, index=range(100, 109))
    groups = boot_groups(df)
    d = df.reset_index(drop=True)
    for fovs in groups:
        for ix in fovs:
            assert d.loc[ix, "fov"].nunique() == 1 and len(ix) == int((d["fov"] == d.loc[ix[0], "fov"]).sum()), \
                f"group rows {list(ix)} are not one whole FOV"
    assert sorted(np.concatenate([np.concatenate(f) for f in groups]).tolist()) == list(range(9))
    rng = np.random.default_rng(1)
    for _ in range(20):
        rows, ids = resample(groups, rng)
        for k in np.unique(ids):
            assert d.loc[rows[ids == k], "fov"].nunique() == 1, "a copy id spans two FOVs"
        assert len(np.unique(ids)) == 4, "one id per drawn FOV (2 per slide here)"
    # weighting: FOV a (1 HT, 1 SNAP) at AUC 1 drawn twice, FOV b at AUC 0 once -> (1 + 1 + 0) / 3
    y3, p3 = np.array([0, 1, 0, 1, 0, 1]), np.array([0.1, 0.9, 0.1, 0.9, 0.9, 0.1])
    assert abs(within_fov_auc(y3, p3, np.array([0, 0, 1, 1, 2, 2])) - 2 / 3) < 1e-12
    print("boot_groups / resample: whole FOVs per group, one id per drawn copy")

    # 4. within_of_draw (per-FOV stats, the fast path) equals within_fov_auc on the drawn rows with copy ids
    rng = np.random.default_rng(3)
    big = pd.DataFrame({"dataset": "D", "slide": rng.choice(["s1", "s2", "s3"], 600),
                        "fov": rng.choice([f"F{i}" for i in range(12)], 600)})
    big = big.sort_values(["slide", "fov"]).reset_index(drop=True)
    yb = rng.integers(0, 2, len(big))
    pb = rng.normal(yb * 0.3, 1.0)
    g = boot_groups(big)
    st = fov_pair_stats(g, yb, pb)
    for _ in range(50):
        pk = draw(g, rng)
        ix, ids = rows_of(g, pk)
        slow, fast = within_fov_auc(yb[ix], pb[ix], ids), within_of_draw(st, pk)
        assert abs(slow - fast) < 1e-12, (slow, fast)
    print("within_of_draw: equals within_fov_auc on 50 draws")
    print("OK")


if __name__ == "__main__":
    main()
