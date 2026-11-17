#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Round 6 configs (TRAINING_NOTES.md point 11): the paper-IN pool of all six sessions, three designs.

    A  r6_loeo_<DS>_s<seed>     train on the other five sessions, validate on <DS>'s pool (val_datasets)
    B  r6_within_<DS>_s<seed>   train and validate (FOV groups, 0.2) on <DS>'s pool alone
    C  r6_pooled_s<seed>        all six pools, FOV-grouped validation 0.2

Every run predicts the mixed slides of all six sessions. Recipe: round 5b's (ml/kfold_warmup/
r5b_kfold_p840410_f0.yaml) without the k-fold. Also writes daint/queues/round6.txt (four runs per
debug job: C first, then A, then B).

    python Revisions/DualColor/ml/round6/make_configs.py
"""

from pathlib import Path

import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


HERE = Path(__file__).resolve().parent
DC = HERE.parents[1]
SESSIONS = ["DFK785", "DFK788", "DFK789", "DFK842", "DFK843", "DFK846"]
TA_RUNS = {"DFK785": "run_002", "DFK788": "run_001", "DFK789": "run_001",
           "DFK842": "run_001", "DFK843": "run_001", "DFK846": "run_001"}
PIN_RUNS = {"DFK785": "run_001", "DFK788": "run_001", "DFK789": "run_002",
            "DFK842": "run_001", "DFK843": "run_001", "DFK846": "run_001"}
EXCLUDE = [
    {"dataset": "DFK846", "slide": "slide1", "fovs": "all",
     "reason": "Slide1HT: background below the session's level, ramping over its first 9 FOVs (first look 2026-11-16)"},
    {"dataset": "DFK846", "slide": "slide3", "fovs": [f"FOV{i:02d}" for i in range(13, 41)],
     "reason": "Slide3SNAP FOVs 13-40: ATTO520 fades to 1-3 % of its start, 640 yield falls 70 % (first look 2026-11-16)"},
]
SEEDS_A = [840410, 1, 2]
SEEDS_BC = [840410, 1, 2, 3, 4]


def recipe() -> dict:
    with open(DC / "ml/kfold_warmup/r5b_kfold_p840410_f0.yaml") as f:
        r = yaml.safe_load(f)
    for k in ("run_name", "kfold", "seed", "train_datasets", "test_datasets", "trace_assignment_runs", "paper_in_runs"):
        r.pop(k)
    r["augmentation"]["mirror_n_jobs"] = 32   # four runs per node; -1 (288 each) hit the open-file limit (job 5377506)
    return r


def config(name: str, seed: int, train: list, split: dict) -> dict:
    return {"run_name": name, "train_datasets": train, "test_datasets": SESSIONS, **split, "seed": seed,
            "trace_assignment_runs": TA_RUNS, "paper_in_runs": PIN_RUNS, "pool_exclude": EXCLUDE, **recipe()}


def check_pins() -> None:
    """Every pinned trace_assignment / paper_in run must exist (the new sessions' paper_in runs are
    made after the gradient rule, TRAINING_NOTES.md point 11); prints each paper_in run's gradients."""
    res = DC.parents[1] / "Results/Revisions/DualColor"
    for ds in SESSIONS:
        ta, pi = res / ds / "trace_assignment" / TA_RUNS[ds], res / ds / "paper_in" / PIN_RUNS[ds]
        missing = [str(x) for x in (ta / "traces.csv", pi / "paper_in.csv") if not x.exists()]
        if missing:
            raise SystemExit(f"{ds}: pinned run files missing: {missing}")
        man = yaml.safe_load(open(pi / "manifest.yaml"))
        grads = {k: v for k, v in man.items() if k.startswith("gt") and k.endswith("_gradient")}
        print(f"{ds}: trace_assignment {TA_RUNS[ds]}, paper_in {PIN_RUNS[ds]} {grads}")


def main() -> None:
    check_pins()
    runs = {"C": [], "A": [], "B": []}
    for s in SEEDS_BC:
        runs["C"].append(config(f"r6_pooled_s{s}", s, SESSIONS, {"val_fov_fraction": 0.2}))
    for s in SEEDS_A:
        for ds in SESSIONS:
            runs["A"].append(config(f"r6_loeo_{ds}_s{s}", s, SESSIONS, {"val_datasets": [ds]}))
    for s in SEEDS_BC:
        for ds in SESSIONS:
            runs["B"].append(config(f"r6_within_{ds}_s{s}", s, [ds], {"val_fov_fraction": 0.2}))
    head = ("# Round 6 (TRAINING_NOTES.md point 11), written by ml/round6/make_configs.py; fixed before any round-6 run.\n"
            "# Design {d}. Recipe: round 5b (ml/kfold_warmup/r5b_kfold_p840410_f0.yaml) without the k-fold.\n")
    paths = []
    for d, cfgs in runs.items():
        for c in cfgs:
            p = HERE / f"{c['run_name']}.yaml"
            p.write_text(head.format(d=d) + yaml.dump(c, Dumper=NoAliasDumper, sort_keys=False))
            paths.append(str(p.relative_to(DC.parents[1])))
    # Point 12: design A again with input_channel clipmax, everything else identical (ml/round6_clipmax/)
    cdir = HERE.parent / "round6_clipmax"
    cdir.mkdir(exist_ok=True)
    cpaths = []
    for c in runs["A"]:
        cc = {**c, "run_name": c["run_name"].replace("r6_loeo_", "r6c_loeo_"), "input_channel": "clipmax"}
        q = cdir / f"{cc['run_name']}.yaml"
        q.write_text("# Round 6 clipmax (TRAINING_NOTES.md point 12), written by ml/round6/make_configs.py; fixed before any run.\n"
                     f"# Identical to ml/round6/{c['run_name']}.yaml except run_name and input_channel: clipmax.\n"
                     + yaml.dump(cc, Dumper=NoAliasDumper, sort_keys=False))
        cpaths.append(str(q.relative_to(DC.parents[1])))
    (DC / "daint/queues/round6_clipmax.txt").write_text(
        "# Round 6 clipmax (TRAINING_NOTES.md point 12): design A with input_channel clipmax, 18 runs.\n"
        + "\n".join(" ".join(cpaths[i:i + 4]) for i in range(0, len(cpaths), 4)) + "\n")
    print(f"{len(cpaths)} clipmax configs in {cdir}")
    lines = [" ".join(paths[i:i + 4]) for i in range(0, len(paths), 4)]
    q = DC / "daint/queues/round6.txt"
    q.write_text("# Round 6 (TRAINING_NOTES.md point 11): C (5 runs), A (18), B (30), four runs per debug job.\n"
                 + "\n".join(lines) + "\n")
    print(f"{len(paths)} configs ({', '.join(f'{d} {len(c)}' for d, c in runs.items())}), {len(lines)} jobs in {q}")


if __name__ == "__main__":
    main()
