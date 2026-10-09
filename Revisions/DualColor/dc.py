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


def load_dfk785(name: str):
    """Import Revisions/DFK785/<name>.py under the name dfk785_<name>.

    The DualColor entry points share file names with the DFK785 modules they reuse, so a plain
    import would find the entry point itself. Bleedthrough/ and DFK785/ go to the END of
    sys.path, for the DFK785 modules' own imports (fov_qc, segment, common, classify, ...).
    """
    import importlib.util
    import sys
    for d in (HERE.parent / "Bleedthrough", HERE.parent / "DFK785"):
        if str(d) not in sys.path:
            sys.path.append(str(d))
    key = f"dfk785_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE.parent / "DFK785" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


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
    cfg["_path"] = str(path)          # common.write_manifest records it as config_source
    cfg["_overrides"] = []
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


# --- Runs and provenance ---
def run_dir(cfg: dict, stage: str, name: str | None = None) -> Path:
    """<output_root>/<stage>/<name>: name from the argument, cfg['runs'][stage], or the latest run_NNN."""
    base = REPO_ROOT / cfg["output_root"] / stage
    name = name or (cfg.get("runs") or {}).get(stage)
    if not name:
        runs = sorted(p for p in base.glob("run_[0-9][0-9][0-9]") if (p / "manifest.yaml").exists())
        if not runs:
            raise FileNotFoundError(f"No finished {stage} run under {base}")
        return runs[-1]
    p = base / name
    if not (p / "manifest.yaml").exists():
        raise FileNotFoundError(f"No manifest.yaml in {p}")
    return p


def write_manifest(run: Path, cfg: dict, script: Path, inputs: list, extra: dict | None = None,
                   modules: list[Path] | None = None) -> None:
    """common.write_manifest, plus a copy and sha256 of dc.py and the DFK785 modules the entry script imports."""
    import hashlib
    import shutil
    import sys as _sys
    _sys.path.insert(0, str(HERE.parent / "Bleedthrough"))
    from common import rel_to_repo, write_manifest as _wm
    _wm(run, {k: v for k, v in cfg.items()}, script, inputs, extra)
    extra_code = {}
    for py in [HERE / "dc.py", *(modules or [])]:
        py = Path(py).resolve()
        dest = run / "code" / (py.name if py.parent == HERE else f"{py.parent.name}__{py.name}")
        shutil.copy(py, dest)
        extra_code[rel_to_repo(py)] = hashlib.sha256(py.read_bytes()).hexdigest()
    with open(run / "manifest.yaml") as f:
        man = yaml.safe_load(f)
    man["dataset"] = cfg.get("dataset")
    man["dataset_config"] = rel_to_repo(Path(cfg["_config_path"]))
    man["extra_code_sha256"] = extra_code
    with open(run / "manifest.yaml", "w") as f:
        yaml.safe_dump(man, f, sort_keys=False)
