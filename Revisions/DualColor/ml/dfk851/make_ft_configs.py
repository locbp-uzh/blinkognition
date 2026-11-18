#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
C fine-tuned on DFK851's single-protein slides (TRAINING_NOTES.md point 16): each config is B's
r6_within_DFK851_s<seed>.yaml (point 13; same data, FOV split, augmentation and test sets) with only
run_name, the finetune block, the learning rate and the warm-up changed:

    ft_head_DFK851_s<seed>   start from r6_pooled_s<seed>; only fc trains; lr 0.0004 (C's and B's)
    ft_full_DFK851_s<seed>   start from r6_pooled_s<seed>; every weight trains; lr 0.00004 (a tenth)
    both                     warmup_epochs 0 (the model starts past the plateau)

Also writes daint/queues/dfk851_ft.txt (four runs per Daint debug job; daint/run_train_queue.sh).
B's configs are read, never rewritten.

    python Revisions/DualColor/ml/dfk851/make_ft_configs.py
"""

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DC = HERE.parents[1]
SEEDS = (840410, 1, 2, 3, 4)
VARIANTS = {"head": {"train_layers": ["fc"], "lr": 0.0004},
            "full": {"train_layers": None, "lr": 0.00004}}


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def main() -> None:
    head = ("# C fine-tuned on DFK851's single-protein slides (TRAINING_NOTES.md point 16), written by\n"
            "# ml/dfk851/make_ft_configs.py from B's config of the same seed; fixed before any run.\n")
    paths = []
    for v, spec in VARIANTS.items():
        for s in SEEDS:
            b = yaml.safe_load(open(HERE / f"r6_within_DFK851_s{s}.yaml"))
            c = copy.deepcopy(b)
            c["run_name"] = f"ft_{v}_DFK851_s{s}"
            ft = {"init_from": f"r6_pooled_s{s}"}
            if spec["train_layers"]:
                ft["train_layers"] = spec["train_layers"]
            c["finetune"] = ft
            c["optimization"]["lr"] = spec["lr"]
            c["optimization"]["early_stopping"]["warmup_epochs"] = 0
            # everything else is B's: check that only the intended keys differ
            changed = {k for k in set(b) | set(c) if b.get(k) != c.get(k)}
            assert changed == {"run_name", "finetune", "optimization"}, changed
            ob, oc = b["optimization"], c["optimization"]
            assert {k for k in ob if ob[k] != oc[k]} == ({"lr", "early_stopping"} if spec["lr"] != ob["lr"] else {"early_stopping"})
            assert {k for k in ob["early_stopping"] if ob["early_stopping"][k] != oc["early_stopping"][k]} == {"warmup_epochs"}
            p = HERE / f"{c['run_name']}.yaml"
            p.write_text(head + yaml.dump(c, Dumper=NoAliasDumper, sort_keys=False))
            paths.append(str(p.relative_to(DC.parents[1])))
    # full fine-tunes first (they may run longest), four per debug job
    order =[p for p in paths if "ft_full" in p] + [p for p in paths if "ft_head" in p]
    lines = [" ".join(order[i:i + 4]) for i in range(0, len(order), 4)]
    q = DC / "daint/queues/dfk851_ft.txt"
    q.write_text("# Point 16 (TRAINING_NOTES.md): C fine-tuned on DFK851, 10 runs, four per Daint debug job\n"
                 "#   nohup Revisions/DualColor/daint/run_train_queue.sh Revisions/DualColor/daint/queues/dfk851_ft.txt > <log> 2>&1 &\n"
                 + "\n".join(lines) + "\n")
    print(f"{len(paths)} configs in {HERE}; {len(lines)} jobs in {q}")


if __name__ == "__main__":
    main()
