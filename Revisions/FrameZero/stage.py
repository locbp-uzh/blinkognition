#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Stage the FrameZero rerun: a new run folder <run_root>/<run_name>/ with the published run's run_info.yaml
(copied, unchanged: input folder, proteins, channels) and symbolic links to every *_locs.hdf5 and *_locs.yaml of
the published run in the same session/folder layout, so that Extraction/extract.py finds the published
localizations and writes its outputs into the new folder. The run config (framezero_config.yaml) is the
paper's Extraction/config.yaml with config.yaml `overrides`. Refuses an existing run folder. Writes nothing in
the published run or the raw data. Prints the run folder.

Usage (root of the S3IT review clone):
    python Revisions/FrameZero/stage.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def main() -> None:
    with open(HERE / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    pub = Path(cfg["published_run"])
    run = Path(cfg["run_root"]) / cfg["run_name"]
    if run.exists():
        sys.exit(f"{run} exists; refusing to mix runs")
    links = sorted(p for p in pub.glob("*/*/*_locs.*") if p.suffix in (".hdf5", ".yaml") and not p.name.startswith("._"))
    if not links:
        sys.exit(f"no localization files under {pub}")
    run.mkdir(parents=True)
    shutil.copy(pub / "run_info.yaml", run / "run_info.yaml")
    for p in links:
        dst = run / p.relative_to(pub)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(p)
    with open(REPO_ROOT / cfg["base_config"]) as f:
        rc = yaml.safe_load(f)
    rc.update(cfg["overrides"])
    with open(run / "framezero_config.yaml", "w") as f:
        f.write(f"# FrameZero run config: {cfg['base_config']} with Revisions/FrameZero/config.yaml overrides\n")
        yaml.safe_dump(rc, f, sort_keys=False)
    (run / "framezero_links.txt").write_text(f"{len(links)} links to {pub}\n")
    print(run)


if __name__ == "__main__":
    main()
