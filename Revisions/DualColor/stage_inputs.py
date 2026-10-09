#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Stage the 640 movies of a dataset for Extraction/run_pipeline.py (single-channel mode).

The extraction expects <input_folder>/<experiment>/<protein>/<files> and finds a protein's
files by name prefix, so every slide becomes a protein folder named by its key (Slide1SNAP,
Slide3Mix, ...) holding one symlink per 640 movie, renamed <key>_<rest of the raw name>:

    <stage_root>/<dataset>/<dataset>/<key>/<key>_640nm_..._NNNN.nd2 -> <input_root>/<folder>/<raw>
    <stage_root>/<dataset>/link_map.tsv    '<key>/<link name>' TAB raw path relative to input_root

The link map is relative to input_root, so a staging made on Daint maps onto the local copy
of the data. Picasso writes its _locs files next to the links, i.e. into the staging folder.
Existing links are left alone; a link pointing elsewhere is an error.

Ground-truth mode (--gt 405 or --gt 488), for the paper's IN rule on the single-protein
slides (TRAINING_NOTES.md point 7): only the slides whose dye's home channel is the given one
(405: ATTO390 / HT slides, 488: ATTO520 / SNAP slides), each FOV's 640 movie under the same
link name as in the 640-only staging and its vesicle-channel movie under the same file number,
paired through the vesicles run's fovs.csv (the run named in the trace_assignment manifest):

    <stage_root>/<dataset>/<dataset>/<key>/<key>_640nm_..._NNNN.nd2 and <key>_405nm_..._NNNN.nd2

Usage:
    python Revisions/DualColor/stage_inputs.py DFK788 --stage-root Inputs/DualColor_640only [--data-root DIR]
    python Revisions/DualColor/stage_inputs.py DFK788 --stage-root Inputs/DualColor_gt405 --gt 405
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from dc import REPO_ROOT, load_config, run_dir, slide_movies  # noqa: E402


def link(path: Path, target: Path) -> None:
    if path.is_symlink():
        if Path(os.readlink(path)) != target:
            sys.exit(f"{path} points to {os.readlink(path)}, expected {target}")
    else:
        path.symlink_to(target)


def stage_gt(cfg: dict, root: Path, base: Path, gt: str) -> list[str]:
    """640 + vesicle-channel links of the single-protein slides whose dye's home channel is gt."""
    ds, tags = cfg["dataset"], cfg["channel_tags"]
    with open(run_dir(cfg, "trace_assignment") / "manifest.yaml") as f:
        fovs = pd.read_csv(REPO_ROOT / yaml.safe_load(f)["vesicles_run"] / "fovs.csv", dtype=str)
    rows = []
    for slide, s in cfg["slides"].items():
        dye = s["dye"]
        if dye == "mix" or str(cfg["dyes"][dye]["home_channel"]) != gt:
            continue
        d = base / ds / s["key"]
        d.mkdir(parents=True, exist_ok=True)
        f = fovs[fovs["slide"] == slide]
        if f[f"file_{gt}"].isna().any():
            sys.exit(f"{ds} {slide}: FOVs without a {gt} movie")
        for r in f.itertuples():
            m640, mgt = root / getattr(r, "file_640"), root / getattr(r, f"file_{gt}")
            name640 = s["key"] + m640.name[len(s["prefix"]):]
            suffix = name640[len(s["key"] + "_" + tags["640"]):]          # '_NNNN.nd2' or '.nd2'
            namegt = f"{s['key']}_{tags[gt]}{suffix}"
            for nm, m in ((name640, m640), (namegt, mgt)):
                if not m.exists():
                    sys.exit(f"missing {m}")
                link(d / nm, m)
                rows.append(f"{s['key']}/{nm}\t{m.relative_to(root)}")
        print(f"{ds} {slide} {s['key']}: {len(f)} FOVs, 640 + {gt}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--stage-root", type=Path, required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--gt", choices=["405", "488"], default=None, help="ground-truth mode: 640 + this vesicle channel")
    a = ap.parse_args()
    cfg = load_config(a.dataset, a.data_root)
    ds, root = cfg["dataset"], Path(cfg["input_root"])
    if not root.is_dir():
        sys.exit(f"input_root {root} does not exist")
    base = a.stage_root / ds
    if a.gt:
        rows = stage_gt(cfg, root, base, a.gt)
        (base / "link_map.tsv").write_text("\n".join(rows) + "\n")
        print(f"{len(rows)} links, map {base / 'link_map.tsv'}")
        return
    rows = []
    for slide, s in cfg["slides"].items():
        movies = slide_movies(cfg, slide, "640")
        if not movies:
            sys.exit(f"{ds} {slide}: no 640 movies in {root / s['folder']}")
        d = base / ds / s["key"]
        d.mkdir(parents=True, exist_ok=True)
        for m in movies:
            link = d / (s["key"] + m.name[len(s["prefix"]):])
            if link.is_symlink():
                if Path(os.readlink(link)) != m:
                    sys.exit(f"{link} points to {os.readlink(link)}, expected {m}")
            else:
                link.symlink_to(m)
            rows.append(f"{s['key']}/{link.name}\t{m.relative_to(root)}")
        print(f"{ds} {slide} {s['key']}: {len(movies)} movies")
    (base / "link_map.tsv").write_text("\n".join(rows) + "\n")
    print(f"{len(rows)} links, map {base / 'link_map.tsv'}")


if __name__ == "__main__":
    main()
