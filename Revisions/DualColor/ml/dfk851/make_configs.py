#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
DFK851 in design B (TRAINING_NOTES.md point 13): round 6's within-experiment design (point 11) on the
seventh session, nothing else changed.

    B  r6_within_DFK851_s<seed>   train and validate (FOV groups, 0.2 per slide) on DFK851's paper-IN
                                  pool; predict the mixed slides of all seven sessions

The recipe, seeds, pins of the six round-6 sessions and their pool exclusions are read from
ml/round6/make_configs.py, so a config here differs from a round-6 B config only in run_name,
train_datasets, test_datasets (+ DFK851), the DFK851 pins and the DFK851 exclusions (point 13's rules).
Also writes s3it/queues/dfk851.txt: the five runs and the C-model reference as the tasks of one S3IT GPU job
(s3it/train.sbatch; Daint was in maintenance on 2026-11-18).

    python Revisions/DualColor/ml/dfk851/make_configs.py
"""

import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DC = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("round6_make_configs", HERE.parent / "round6" / "make_configs.py")
r6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r6)

DS = "DFK851"
TA_RUN = "run_001"
PIN_RUN = "run_001"      # the paper_in run that satisfies point 7's gradient rule (checks/20261118_dfk851)
# Point 13's mechanical exclusions, filled from the first look (checks/20261118_dfk851) before any training
EXCLUDE_DFK851: list[dict] = []


def check_exclusions(res: Path) -> None:
    """EXCLUDE_DFK851 must equal checks/20261118_dfk851/exclusions.yaml (written by qc_rules.py, read by the
    gradient rule), so that the paper_in gradient was chosen on the FOVs these configs train on."""
    ex = res / "checks/20261118_dfk851/exclusions.yaml"
    if not ex.exists():
        raise SystemExit(f"{ex} missing: run qc_rules.py (point 13) first")
    key = lambda es: sorted((e["dataset"], e["slide"], e["fovs"] if e["fovs"] == "all" else tuple(map(str, e["fovs"])))
                            for e in es)
    if key(yaml.safe_load(open(ex)) or []) != key(EXCLUDE_DFK851):
        raise SystemExit(f"EXCLUDE_DFK851 differs from {ex}: copy the qc_rules.py entries here")


def main() -> None:
    res = DC.parents[1] / "Results/Revisions/DualColor"
    check_exclusions(res)
    for kind, run, f in (("trace_assignment", TA_RUN, "traces.csv"), ("paper_in", PIN_RUN, "paper_in.csv")):
        if not (res / DS / kind / run / f).exists():
            raise SystemExit(f"{DS}: pinned {kind} {run} missing ({res / DS / kind / run / f})")
    r6.check_pins()
    man = yaml.safe_load(open(res / DS / "paper_in" / PIN_RUN / "manifest.yaml"))
    print(f"{DS}: trace_assignment {TA_RUN}, paper_in {PIN_RUN}",
          {k: v for k, v in man.items() if k.startswith("gt") and k.endswith("_gradient")})
    head = ("# DFK851 design B (TRAINING_NOTES.md point 13), written by ml/dfk851/make_configs.py; fixed before any run.\n"
            "# Round 6's design B (point 11, recipe round 5b) on the seventh session.\n")
    paths = []
    for s in r6.SEEDS_BC:
        c = {"run_name": f"r6_within_{DS}_s{s}", "train_datasets": [DS], "test_datasets": r6.SESSIONS + [DS],
             "val_fov_fraction": 0.2, "seed": s,
             "trace_assignment_runs": {**r6.TA_RUNS, DS: TA_RUN}, "paper_in_runs": {**r6.PIN_RUNS, DS: PIN_RUN},
             "pool_exclude": r6.EXCLUDE + EXCLUDE_DFK851, **r6.recipe()}
        p = HERE / f"{c['run_name']}.yaml"
        p.write_text(head + yaml.dump(c, Dumper=r6.NoAliasDumper, sort_keys=False))
        paths.append(str(p.relative_to(DC.parents[1])))
    # the reference of point 13: the five round-6 C models applied unchanged, as a task of the same job
    apply = (f"apply:Results/Revisions/DualColor/dfk851/c_transfer:{DS}={TA_RUN}:"
             + ",".join(f"r6_pooled_s{s}" for s in r6.SEEDS_BC))
    q = DC / "s3it/queues/dfk851.txt"
    q.parent.mkdir(exist_ok=True)
    q.write_text("# DFK851 design B (TRAINING_NOTES.md point 13): 5 runs and the C-model reference, one S3IT GPU job\n"
                 "#   sbatch Revisions/DualColor/s3it/train.sbatch $(grep -v '^#' Revisions/DualColor/s3it/queues/dfk851.txt)\n"
                 + " ".join(paths + [apply]) + "\n")
    print(f"{len(paths)} configs in {HERE}, one job in {q}")


if __name__ == "__main__":
    main()
