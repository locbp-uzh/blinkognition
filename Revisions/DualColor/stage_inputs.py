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

Usage:
    python Revisions/DualColor/stage_inputs.py DFK788 --stage-root Inputs/DualColor_640only [--data-root DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dc import load_config, slide_movies  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--stage-root", type=Path, required=True)
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    cfg = load_config(a.dataset, a.data_root)
    ds, root = cfg["dataset"], Path(cfg["input_root"])
    if not root.is_dir():
        sys.exit(f"input_root {root} does not exist")
    base = a.stage_root / ds
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
