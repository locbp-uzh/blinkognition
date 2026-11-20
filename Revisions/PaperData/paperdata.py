#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Shared access to the paper's published extraction (Puentener 2026) for the first-round review topics.

The published run (config.yaml: published_run, on S3IT) holds, per construct, the traces in four
forms (raw 5 x 5 box sums, bg_rm = raw minus the per-trace GMM background, and the min-max and z-score
normalizations of bg_rm), before filtering (AllRaw/, AllBG/) and after (Filtered/), for traces inside
vesicles (IN), outside (OUT) and background pixels; and UniqueIDs/ with each trace's origin movie and
box position. Column labels of every trace matrix are uniqueIDs.

Nothing here changes the published data or parameters; it reads them.

    from paperdata import load_config, load_traces, load_ids, to_photons
    cfg = load_config()
    bg = load_traces(cfg, "HaloK117", "bg_rm")            # filtered IN traces, frames x traces, ADU
    ids = load_ids(cfg, "HaloK117")                       # uniqueID, x, y, session, folder, movie
    ph = to_photons(bg, cfg)                              # detected photons per frame
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
KINDS = ("raw", "bg_rm", "minmax", "zscored")
WHERE_DIR = {"IN": "ProteinTracesIN", "OUT": "ProteinTracesOUT", "background": "BackgroundTraces"}
SESSION_RE = re.compile(r"(\d{8}[_-][A-Z]{2}_Exp\d+)")


# --- Config ---

def load_config(path: Path | None = None, run_folder: str | None = None, local: bool = False) -> dict:
    """Read config.yaml. `run_folder` replaces published_run (tests on another extraction);
    `local=True` points published_run at the local partial mirror (config local_mirror)."""
    path = Path(path) if path else HERE / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    if local and not run_folder:
        run_folder = str(REPO_ROOT / cfg["local_mirror"])
    if run_folder:
        cfg["published_run"] = str(run_folder)
        cfg["_run_folder_override"] = str(run_folder)
    return cfg


def key_of(cfg: dict, construct: str) -> str:
    """Extraction key (file prefix) of a paper construct name, e.g. HaloD106 -> HTHTL."""
    return cfg["constructs"][construct]


# --- Paths ---

def trace_path(cfg: dict, construct: str, kind: str, where: str = "IN", filtered: bool = True) -> Path:
    """Pickle of one trace form. filtered=False gives the traces before the quality filter
    (only raw and bg_rm exist there)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    key, run = key_of(cfg, construct), Path(cfg["published_run"])
    if filtered:
        return run / WHERE_DIR[where] / "Filtered" / f"{key}_{where}_filtered_{kind}.pkl"
    if kind not in ("raw", "bg_rm"):
        raise ValueError("unfiltered traces exist as raw and bg_rm only (AllNorm is not used here)")
    sub = "AllRaw" if kind == "raw" else "AllBG"
    return run / WHERE_DIR[where] / sub / f"{key}_{where}_{kind}.pkl"


def ids_path(cfg: dict, construct: str, where: str = "IN") -> Path:
    """UniqueIDs pickle; where in IN, OUT, background, all."""
    return Path(cfg["published_run"]) / "UniqueIDs" / f"{key_of(cfg, construct)}_uniqueID_{where}.pkl"


# --- Loading ---

def load_traces(cfg: dict, construct: str, kind: str, where: str = "IN", filtered: bool = True) -> pd.DataFrame:
    """Trace matrix, frames x traces, columns = uniqueIDs."""
    return pd.read_pickle(trace_path(cfg, construct, kind, where, filtered))


def parse_origin(origin: str) -> tuple[str, str, str]:
    """(session, protein folder, movie stem) from a UniqueIDs origin path
    .../<session>/<folder>/<movie>_traces.pkl or .../<movie>_background_traces.pkl."""
    p = Path(str(origin))
    m = SESSION_RE.search(str(origin))
    movie = p.stem
    for suffix in ("_background_traces.pkl", "_traces.pkl"):
        if p.name.endswith(suffix):
            movie = p.name[: -len(suffix)]
            break
    return (m.group(1) if m else ""), p.parent.name, movie


def load_ids(cfg: dict, construct: str, where: str = "IN") -> pd.DataFrame:
    """UniqueIDs table without the stored trace arrays, plus session, folder and movie.
    x and y are as stored by the extraction (integer box coordinates). Background tables store the
    source path in `file_origin` (their `origin` is just "background", combine.py)."""
    u = pd.read_pickle(ids_path(cfg, construct, where))
    u = u.drop(columns=[c for c in ("trace",) if c in u.columns])
    src = "file_origin" if "file_origin" in u.columns else "origin"
    parsed = [parse_origin(o) for o in u[src]]
    u["session"] = [s for s, _, _ in parsed]
    u["folder"] = [f for _, f, _ in parsed]
    u["movie"] = [m for _, _, m in parsed]
    return u.reset_index(drop=True)


def load_index(cfg: dict, construct: str, where: str = "IN") -> pd.DataFrame:
    """Step 0's per-trace index (uniqueID, session, folder, movie, x, y, in_unfiltered,
    passed_filter); no UniqueIDs pickle needed."""
    return pd.read_csv(REPO_ROOT / cfg["step0_run"] / f"index_{where}_{construct}.csv")


# --- Units ---

def to_photons(adu, cfg: dict):
    """Background-removed ADU to detected photons (photoelectrons): ADU * sensitivity / EM gain.
    Picasso's convention since v0.6.0 (no QE division). Use on bg_rm traces, never on raw ones
    (raw still carries baseline and background)."""
    cam = cfg["camera"]
    return adu * (cam["sensitivity_e_per_adu"] / cam["em_gain"])


def frame_s(cfg: dict) -> float:
    """Frame interval in seconds (ND2 timestamps), not the 30 ms exposure."""
    return cfg["frame_interval_ms"] / 1000.0


# --- ND2 metadata ---

_ND2_FIELDS = ("GainMultiplier", "EnableGainMultiplier", "ConversionGain", "ReadoutSpeed",
               "VerticalShiftSpeed", "SensorMode", "ModeFQ", "ExposureTime", "DesiredTemperature",
               "BinningX", "BinningY")


def nd2_camera(path: Path) -> dict:
    """Camera settings and frame timing of one ND2 movie, from its headers (no pixel data read).
    Readout speed and conversion gain are NIS-Elements indices, not physical values."""
    import nd2
    out = {"file": str(path)}
    with nd2.ND2File(str(path)) as h:
        s = json.dumps(h.unstructured_metadata(), default=str)
        for k in _ND2_FIELDS:
            m = re.search(r'"' + k + r'"\s*:\s*([^,}\]]+)', s)
            out[k] = m.group(1).strip().strip('"') if m else None
        out["date"] = h.text_info.get("date")
        ev = h.events(orient="list")
        t = np.asarray(ev.get("Time [s]", []), dtype=float)
        # laser power per frame (the text header's laser line can belong to another channel's setup)
        for k, v in ev.items():
            m = re.fullmatch(r"(\d+) nm \[%\]", k)
            if m:
                out[f"laser_{m.group(1)}nm_pct_median"] = float(np.nanmedian(np.asarray(v, dtype=float)))
    out["n_frames"] = int(t.size)
    if t.size > 1:
        dt = np.diff(t) * 1000.0
        out.update(dt_median_ms=float(np.median(dt)), dt_min_ms=float(dt.min()), dt_max_ms=float(dt.max()))
    return out


# --- Runs and manifests ---

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def make_run_dir(cfg: dict, stage: str, topic_root: str | None = None) -> Path:
    """Next free <output_root>/<stage>/run_NNN (topic_root replaces output_root for other topics)."""
    base = REPO_ROOT / (topic_root or cfg["output_root"]) / stage
    base.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name[4:]) for p in base.glob("run_[0-9][0-9][0-9]")] or [0])
    run = base / f"run_{n:03d}"
    run.mkdir()
    return run


def rel_to_repo(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_manifest(run: Path, cfg: dict, script: Path, inputs: list[Path], extra: dict | None = None,
                   hash_inputs: bool = True) -> None:
    """config.yaml (effective), code/ (this module and the entry script), manifest.yaml (git state,
    environment, inputs with size and sha256)."""
    with open(run / "config.yaml", "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, sort_keys=False,
                       allow_unicode=True)
    (run / "code").mkdir(exist_ok=True)
    code = {}
    for py in {Path(__file__).resolve(), Path(script).resolve()}:
        shutil.copy(py, run / "code" / py.name)
        code[rel_to_repo(py)] = sha256(py)
    pkgs = {"numpy": np.__version__, "pandas": pd.__version__}
    for name in ("scipy", "sklearn", "nd2"):
        try:
            pkgs[name] = __import__(name).__version__
        except Exception:
            pass
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "script": rel_to_repo(script),
        "code_sha256": code,
        "config_source": rel_to_repo(Path(cfg["_path"])),
        "run_folder_override": cfg.get("_run_folder_override"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "Revisions/PaperData",
                               rel_to_repo(Path(script).resolve().parent))),
        "python": platform.python_version(),
        "packages": pkgs,
        "inputs": {str(p): {"bytes": Path(p).stat().st_size, "sha256": sha256(Path(p)) if hash_inputs else None}
                   for p in inputs},
    }
    if extra:
        manifest.update(extra)
    with open(run / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
