#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Seven-session pool (TRAINING_NOTES.md point 17): each config is round 6's C config of the same seed
(ml/round6/r6_pooled_s<seed>.yaml) with DFK851 added to train_datasets and test_datasets and its pins
(trace_assignment and paper_in run_001, as B's configs), and:

    pool7_valfov_s<seed>  (P)  validation as C: 20 % of each single-protein slide's FOVs
    pool7_valmix_s<seed>  (M)  val_mixed: the labeled traces of the older sessions' mixed slides, DFK785 slide 4
                               excluded; val_fov_fraction removed (every pool trace is trained on)

Also writes daint/queues/pool7.txt (four runs per Daint debug job; daint/run_train_queue.sh). The round-6 and
DFK851 configs are read, never rewritten.

    python Revisions/DualColor/ml/dfk851/make_pool7_configs.py
"""

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DC = HERE.parents[1]
SEEDS = (840410, 1, 2, 3, 4, 5)   # 5 added by the user's amendment (point 17): 12 runs fill three 4-GPU nodes
BASE_SEED = 840410                 # round 6 has no seed 5: its configs are seed 840410's with seed and run_name changed
DS = "DFK851"
OLD = ["DFK785", "DFK788", "DFK789", "DFK842", "DFK843", "DFK846"]
VAL_MIXED = {"datasets": OLD,
             "exclude_slides": [{"dataset": "DFK785", "slide": "slide4",
                                 "reason": "at chance in every model so far; adds only noise to the stopping signal "
                                           "(user decision 2026-11-18, point 17)"}]}


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def main() -> None:
    head = ("# Seven-session pool (TRAINING_NOTES.md point 17), written by ml/dfk851/make_pool7_configs.py from round 6's\n"
            "# C config of the same seed; fixed before any run.\n")
    paths = {"valmix": [], "valfov": []}
    for s in SEEDS:
        src = s if (DC / "ml/round6" / f"r6_pooled_s{s}.yaml").exists() else BASE_SEED
        c6 = yaml.safe_load(open(DC / "ml/round6" / f"r6_pooled_s{src}.yaml"))
        b = yaml.safe_load(open(HERE / f"r6_within_{DS}_s{src}.yaml"))
        if src != s:
            c6["seed"], c6["run_name"] = s, f"r6_pooled_s{s}"
        assert c6["train_datasets"] == OLD and c6["test_datasets"] == OLD and c6["val_fov_fraction"] == 0.2
        for arm in ("valfov", "valmix"):
            c = copy.deepcopy(c6)
            c["run_name"] = f"pool7_{arm}_s{s}"
            c["train_datasets"] = OLD + [DS]
            c["test_datasets"] = OLD + [DS]
            c["trace_assignment_runs"] = {**c6["trace_assignment_runs"], DS: b["trace_assignment_runs"][DS]}
            c["paper_in_runs"] = {**c6["paper_in_runs"], DS: b["paper_in_runs"][DS]}
            if arm == "valmix":
                del c["val_fov_fraction"]
                c["val_mixed"] = copy.deepcopy(VAL_MIXED)
            # everything else is C's
            changed = {k for k in set(c6) | set(c) if c6.get(k) != c.get(k)}
            allowed = {"run_name", "train_datasets", "test_datasets", "trace_assignment_runs", "paper_in_runs"}
            assert changed == (allowed | {"val_fov_fraction", "val_mixed"} if arm == "valmix" else allowed), changed
            assert c["pool_exclude"] == b["pool_exclude"], "DFK851 exclusions differ from B's (point 13: none)"
            p = HERE / f"{c['run_name']}.yaml"
            p.write_text(head + yaml.dump(c, Dumper=NoAliasDumper, sort_keys=False))
            paths[arm].append(str(p.relative_to(DC.parents[1])))
    order = [p for pair in zip(paths["valmix"], paths["valfov"]) for p in pair]   # M and P of a seed in one job
    lines = [" ".join(order[i:i + 4]) for i in range(0, len(order), 4)]
    q = DC / "daint/queues/pool7.txt"
    q.write_text("# Point 17 (TRAINING_NOTES.md): seven-session pool, M (valmix) and P (valfov), four per Daint debug job\n"
                 "#   nohup Revisions/DualColor/daint/run_train_queue.sh Revisions/DualColor/daint/queues/pool7.txt > <log> 2>&1 &\n"
                 + "\n".join(lines) + "\n")
    print(f"{sum(map(len, paths.values()))} configs in {HERE}; {len(lines)} jobs in {q}")


if __name__ == "__main__":
    main()
