#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
The two controls of TRAINING_NOTES.md point 18.

    ccont_s<seed>  (CC)  round 6's C config r6_pooled_s<seed> continued as point 16's FT full: finetune from
                         r6_pooled_s<seed>, lr 0.00004, warm-up 0; DFK851 added to test_datasets only (pins as
                         B's), so its mixed slide is predicted. Seeds 840410, 1-4.
    size_s<seed>   (S)   point 17's P config (pool7_valfov_s<seed>; seed 6 from seed 840410's with seed and run_name
                         changed) with DFK851's pool cut to the median older pool (841 traces): whole FOVs, stratified
                         by slide in proportion to the slide's traces, each FOV in a random order (seeded by the run's
                         seed) kept if it brings the slide's count closer to its target; the rest excluded through
                         pool_exclude. Seeds 840410, 1-6.

The pool is the one train_pure.py builds (load_dataset; paper-IN traces of the single-protein slides). Writes
daint/queues/controls.txt (three Daint debug jobs of four runs) and ml/dfk851/size_subsample.csv (the FOVs kept per
seed and slide). Existing configs are read, never rewritten.

    python Revisions/DualColor/ml/dfk851/make_controls_configs.py
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
DC = HERE.parents[1]
sys.path.insert(0, str(DC))
from train_pure import load_dataset  # noqa: E402

DS = "DFK851"
OLD = ["DFK785", "DFK788", "DFK789", "DFK842", "DFK843", "DFK846"]
CC_SEEDS = (840410, 1, 2, 3, 4)
S_SEEDS = (840410, 1, 2, 3, 4, 5, 6)
TARGET = 841          # the median of the six older sessions' pools (point 18)
FT_LR = 0.00004


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def dfk851_pool(cfg: dict) -> pd.DataFrame:
    meta, _, _ = load_dataset(DS, cfg)
    pool = meta[(meta["slide_type"] == "pure") & meta["paper_in"]]
    return pool.groupby(["slide", "fov"]).size().rename("n").reset_index()


def subsample(per_fov: pd.DataFrame, seed: int) -> tuple[dict, list[dict]]:
    """FOVs kept per slide and the per-slide record."""
    total = per_fov["n"].sum()
    rng = np.random.default_rng([seed, 18])
    kept, rec = {}, []
    for slide, g in per_fov.groupby("slide"):
        target = round(TARGET * g["n"].sum() / total)
        n = dict(zip(g["fov"], g["n"]))
        keep, cum = [], 0
        for f in rng.permutation(sorted(n)):
            if abs(cum + n[f] - target) < abs(cum - target):
                keep.append(f)
                cum += n[f]
        kept[slide] = sorted(keep)
        rec.append({"seed": seed, "slide": slide, "target": target, "kept_traces": cum, "kept_fovs": len(keep),
                    "fovs": len(n), "pool_traces": int(g["n"].sum())})
    return kept, rec


def main() -> None:
    paths, recs = {"cc": [], "s": []}, []
    for s in CC_SEEDS:
        c6 = yaml.safe_load(open(DC / "ml/round6" / f"r6_pooled_s{s}.yaml"))
        b = yaml.safe_load(open(HERE / f"r6_within_{DS}_s{s}.yaml"))
        c = copy.deepcopy(c6)
        c["run_name"] = f"ccont_s{s}"
        c["test_datasets"] = OLD + [DS]
        c["trace_assignment_runs"] = {**c6["trace_assignment_runs"], DS: b["trace_assignment_runs"][DS]}
        c["paper_in_runs"] = {**c6["paper_in_runs"], DS: b["paper_in_runs"][DS]}
        c["finetune"] = {"init_from": f"r6_pooled_s{s}"}
        c["optimization"]["lr"] = FT_LR
        c["optimization"]["early_stopping"]["warmup_epochs"] = 0
        changed = {k for k in set(c6) | set(c) if c6.get(k) != c.get(k)}
        assert changed == {"run_name", "test_datasets", "trace_assignment_runs", "paper_in_runs", "finetune",
                           "optimization"}, changed
        assert c["train_datasets"] == OLD
        ft = yaml.safe_load(open(HERE / f"ft_full_{DS}_s{s}.yaml"))   # the recipe is FT full's
        assert ft["optimization"] == c["optimization"], "CC's optimization differs from FT full's"
        p = HERE / f"{c['run_name']}.yaml"
        p.write_text("# Point 18 control CC (TRAINING_NOTES.md), written by ml/dfk851/make_controls_configs.py from C's\n"
                     "# config of the same seed; fixed before any run.\n" + yaml.dump(c, Dumper=NoAliasDumper, sort_keys=False))
        paths["cc"].append(str(p.relative_to(DC.parents[1])))
    per_fov = None
    for s in S_SEEDS:
        src = s if (HERE / f"pool7_valfov_s{s}.yaml").exists() else 840410
        p7 = yaml.safe_load(open(HERE / f"pool7_valfov_s{src}.yaml"))
        if per_fov is None:
            per_fov = dfk851_pool(p7)
            assert per_fov["n"].sum() == 5911, per_fov["n"].sum()
        c = copy.deepcopy(p7)
        c["run_name"], c["seed"] = f"size_s{s}", s
        kept, rec = subsample(per_fov, s)
        recs += rec
        for slide, keep in kept.items():
            drop = sorted(set(per_fov.loc[per_fov["slide"] == slide, "fov"]) - set(keep))
            c["pool_exclude"].append({"dataset": DS, "slide": slide, "fovs": drop,
                                      "reason": f"point 18 size control: DFK851's pool cut to {TARGET} traces "
                                                f"(seed {s}; {len(keep)} FOVs kept on this slide)"})
        changed = {k for k in set(p7) | set(c) if p7.get(k) != c.get(k)}
        assert changed == {"run_name", "seed", "pool_exclude"} or (src == s and changed == {"run_name", "pool_exclude"}), changed
        p = HERE / f"{c['run_name']}.yaml"
        p.write_text("# Point 18 control S (TRAINING_NOTES.md), written by ml/dfk851/make_controls_configs.py from point\n"
                     "# 17's P config of the same seed; fixed before any run.\n" + yaml.dump(c, Dumper=NoAliasDumper, sort_keys=False))
        paths["s"].append(str(p.relative_to(DC.parents[1])))
    rec = pd.DataFrame(recs)
    rec.to_csv(HERE / "size_subsample.csv", index=False)
    print(rec.groupby("seed")[["target", "kept_traces", "kept_fovs"]].sum().to_string())
    order = paths["s"] + paths["cc"]
    lines = [" ".join(order[i:i + 4]) for i in range(0, len(order), 4)]
    assert all(len(ln.split()) == 4 for ln in lines), "a job would leave GPUs idle"
    q = DC / "daint/queues/controls.txt"
    q.write_text("# Point 18 (TRAINING_NOTES.md): controls S (7) and CC (5), four runs per Daint debug job\n"
                 "#   nohup Revisions/DualColor/daint/run_train_queue.sh Revisions/DualColor/daint/queues/controls.txt > <log> 2>&1 &\n"
                 + "\n".join(lines) + "\n")
    print(f"{len(order)} configs; {len(lines)} jobs in {q}")


if __name__ == "__main__":
    main()
