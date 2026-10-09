#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Shared helpers of Revisions/DualColor: dataset configs and file discovery.

A dataset config (datasets/<ID>.yaml) names a base config (`base: ../base.yaml`); the two are
deep-merged, dataset keys winning. input_root = data_root / data_folder, where data_root
comes from --data-root, the environment variable DUALCOLOR_DATA_ROOT, or the base config, so
the same config serves the local copy and the copy on Daint scratch.

The vesicle analysis reuses Revisions/Bleedthrough/common.py and the DFK785 code; its
functions take the merged config dict.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATASETS = HERE / "datasets"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def config_path(dataset: str | Path) -> Path:
    """A dataset ID (e.g. DFK788) or a path to a dataset config."""
    p = Path(dataset)
    return p if p.suffix in (".yaml", ".yml") else DATASETS / f"{dataset}.yaml"


def load_config(dataset: str | Path, data_root: str | Path | None = None) -> dict:
    path = config_path(dataset).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("base"):
        with open(path.parent / cfg["base"]) as f:
            cfg = _merge(yaml.safe_load(f), {k: v for k, v in cfg.items() if k != "base"})
    root = data_root or os.environ.get("DUALCOLOR_DATA_ROOT") or cfg["data_root"]
    cfg["data_root"] = str(root)
    cfg["input_root"] = str(Path(root) / cfg["data_folder"])
    cfg["_config_path"] = str(path)
    return cfg


def slide_movies(cfg: dict, slide: str, channel: str) -> list[Path]:
    """The ND2 files of one channel of a slide, in file-number order (unnumbered first)."""
    s = cfg["slides"][slide]
    folder = Path(cfg["input_root"]) / s["folder"]
    stem = f"{s['prefix']}_{cfg['channel_tags'][channel]}"
    pat = re.compile(rf"^{re.escape(stem)}(?:_(\d{{4}}))?\.nd2$")
    found = [(int(m.group(1) or -1), p) for p in folder.iterdir() if (m := pat.match(p.name))]
    return [p for _, p in sorted(found)]


def match_radius(cfg: dict, slide: str) -> float:
    """Protein-vesicle match radius: trace_assignment.match_radius_px.pure or .mix by slide type."""
    r = cfg["trace_assignment"]["match_radius_px"]
    return float(r["mix"] if cfg["slides"][slide]["dye"] == "mix" else r["pure"]) if isinstance(r, dict) else float(r)
