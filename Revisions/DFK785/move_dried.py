#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Move the acquisitions of dried-out slides into <input_root>/bad_data/.

Rule (per slide listed under dried.slides): the slide is dried from the first FOV,
in acquisition order, whose vesicle count (sum of dried.count_channels) is below
dried.fraction x the median count of the FOVs before it (at least dried.min_before
of them), and every later FOV is dried too, since a dried slide does not recover
(isolated FOVs with a few vesicles after that point are remnants, not usable data).

Files keep their path relative to input_root under bad_data/, so a move can be
undone with the same relative paths. bad_data/moved_files.csv lists every move with
the FOV's acquisition order, time and counts; bad_data/README.md says why. The
macOS AppleDouble companion ('._<name>') of each file is moved with it if present.
Locked files (macOS 'uchg' flag) are detected before anything is moved, and the
script stops without moving any file.

Without --apply only the plan is printed.

Usage:
    python Revisions/DFK785/move_dried.py [--fov-qc-run DIR] [--apply]
"""

from __future__ import annotations

import argparse
import logging
import os
import stat
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Bleedthrough"))
from common import REPO_ROOT, load_config, rel_to_repo, setup_logging  # noqa: E402


def dried_from(g: pd.DataFrame, dc: dict) -> int | None:
    """Acquisition order of the first dried FOV, or None."""
    total = g[[f"n_{c}" for c in dc["count_channels"]]].sum(axis=1).to_numpy()
    for i in range(dc["min_before"], len(g)):
        if total[i] < dc["fraction"] * pd.Series(total[:i]).median():
            return int(g["acq_order"].iloc[i])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--fov-qc-run", type=Path, default=None, help="fov_qc run (default: dried.fov_qc_run)")
    ap.add_argument("--apply", action="store_true", help="actually move the files")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    dc = cfg["dried"]
    root = Path(cfg["input_root"])
    qc_run = args.fov_qc_run or REPO_ROOT / cfg["output_root"] / "fov_qc" / dc["fov_qc_run"]
    df = pd.read_csv(qc_run / "fov_counts.csv")

    plan = []
    for slide in dc["slides"]:
        g = df[df["slide"] == slide].sort_values("acq_order")
        start = dried_from(g, dc)
        if start is None:
            logging.info(f"{slide}: no drying detected")
            continue
        bad = g[g["acq_order"] >= start]
        logging.info(f"{slide}: dried from FOV {start} of {len(g)} ({bad['time'].iloc[0]}); "
                     f"{len(bad)} FOVs to move, {len(g) - len(bad)} kept")
        for _, r in bad.iterrows():
            for ch in cfg["slides"][slide]["channels"]:
                plan.append({"slide": slide, "acq_order": r["acq_order"], "time": r["time"], "channel": ch,
                             "file": r[f"file_{ch}"],
                             **{f"n_{c}": r.get(f"n_{c}") for c in cfg["count_channels"]}})
    plan = pd.DataFrame(plan)
    if plan.empty:
        logging.info("Nothing to move")
        return
    missing = [f for f in plan["file"] if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} planned files are not in place (already moved?): {missing[:3]}")
    # Locked files (macOS 'uchg', Finder 'Locked') cannot be renamed; check all before moving any.
    locked = [f for f in plan["file"] if os.stat(root / f).st_flags & (stat.UF_IMMUTABLE | stat.SF_IMMUTABLE)]
    if locked:
        raise PermissionError(f"{len(locked)} of {len(plan)} planned files are locked (uchg), e.g. {locked[0]}; "
                              "nothing was moved")
    logging.info(f"{len(plan)} files planned under {root / 'bad_data'}")
    if not args.apply:
        logging.info("Dry run; rerun with --apply to move")
        return

    bad_root = root / "bad_data"
    for f in plan["file"]:
        src, dst = root / f, bad_root / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise FileExistsError(dst)
        os.rename(src, dst)         # same volume: a rename, never a copy that could be left behind
        side = src.with_name("._" + src.name)
        if side.exists():
            os.rename(side, dst.with_name("._" + dst.name))
    log = bad_root / "moved_files.csv"
    plan.assign(moved_at=datetime.now().isoformat(timespec="seconds"), fov_qc_run=rel_to_repo(qc_run)).to_csv(
        log, mode="a", header=not log.exists(), index=False)
    summary = plan.groupby("slide").agg(fovs=("acq_order", "nunique"), first=("acq_order", "min"),
                                        since=("time", "min"))
    lines = [f"- {s}: FOVs {int(r['first'])}-end in acquisition order ({int(r['fovs'])} FOVs, from {r['since']})"
             for s, r in summary.iterrows()]
    (bad_root / "README.md").write_text(
        "# Acquisitions from dried-out slides\n\n"
        "Moved here by repos/Hub/blinkognition/Revisions/DFK785/move_dried.py. The slides dried out during "
        "acquisition (see 20260926_readme.txt); from the first FOV whose 405 + 488 vesicle count fell below "
        f"{dc['fraction']:g} x the median of the earlier FOVs, every later FOV of that slide is here. Paths below "
        "bad_data/ mirror the original ones, and moved_files.csv lists every file with its FOV's acquisition "
        f"order, time and vesicle counts (from fov_qc {rel_to_repo(qc_run)}).\n\n" + "\n".join(lines) + "\n\n"
        "File numbers do not follow acquisition order on slide 3 and are shifted between channels there (515 "
        "file N belongs with 405/488/640 file N-1 for most FOVs); FOVs were matched by acquisition time.\n")
    logging.info(f"Moved {len(plan)} files; log in {log}")


if __name__ == "__main__":
    main()
