#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Shared helpers for the ATTO390 / ATTO525 bleed-through analysis.

- Config, ND2 loading (with channel-order check) and FOV discovery
- Background flattening, matched-filter SNR maps and peak finding
- Run folders with a provenance manifest (config, git commit, input checksums)
- Plot helpers following the lab standards (rcParams come from Extraction/utils.py)
"""

from __future__ import annotations

import hashlib
import logging
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# Extraction/utils.py sets the lab rcParams on import and provides setup_logging.
sys.path.insert(0, str(REPO_ROOT / "Extraction"))
from utils import _COLORS as COLORS, setup_logging  # noqa: E402

__all__ = ["COLORS", "setup_logging"]

# PDF font subsetting logs every glyph at INFO level.
logging.getLogger("fontTools").setLevel(logging.WARNING)

# Okabe-Ito assignment per slide, used in every figure.
SLIDE_COLORS = {"ATTO390": COLORS["blue"], "ATTO525": COLORS["orange"]}


# =============================================================================
# Config and data
# =============================================================================


def load_config(path: Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load config.yaml (default: the one next to this file) and apply 'a.b=value' overrides.

    Values are parsed as YAML, so '--set photometry.aperture_radius_px=4' gives an int.
    The applied overrides are kept in cfg['_overrides'] and end up in the manifest.
    """
    path = Path(path) if path else HERE / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override '{item}' is not of the form key.path=value")
        node, *parts = key.split(".")
        target = cfg
        for part in [node, *parts][:-1]:
            if part not in target:
                raise KeyError(f"Override '{item}': '{part}' not in config")
            target = target[part]
        leaf = [node, *parts][-1]
        if leaf not in target:
            raise KeyError(f"Override '{item}': '{leaf}' not in config")
        target[leaf] = yaml.safe_load(value)
    cfg["_path"] = str(path.resolve())
    cfg["_overrides"] = list(overrides or [])
    return cfg


def discover_fovs(cfg: dict) -> list[dict]:
    """List every ND2 file per slide as {slide, fov, path}, minus cfg['exclude_fovs']."""
    root = Path(cfg["input_root"])
    excluded = cfg.get("exclude_fovs") or {}
    out = []
    for slide, s in cfg["slides"].items():
        files = sorted(p for p in (root / s["folder"]).glob("*.nd2") if not p.name.startswith("._"))
        if not files:
            raise FileNotFoundError(f"No ND2 files for slide {slide} in {root / s['folder']}")
        for p in files:
            if p.stem in excluded:
                logging.info(f"Excluding {p.stem}: {excluded[p.stem]}")
                continue
            out.append({"slide": slide, "fov": p.stem, "path": p})
    unknown = set(excluded) - {p.stem for s in cfg["slides"].values()
                               for p in (root / s["folder"]).glob("*.nd2")}
    if unknown:
        raise ValueError(f"exclude_fovs lists FOVs that do not exist: {sorted(unknown)}")
    return out


def load_fov(path: Path, cfg: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Load one ND2 as {channel_key: float64 image}; verify channel names and order."""
    import nd2

    with nd2.ND2File(str(path)) as h:
        if set(h.sizes) != {"C", "Y", "X"}:
            raise ValueError(f"{path.name}: expected a single CYX frame, got sizes {h.sizes}")
        names = [c.channel.name for c in h.metadata.channels]
        arr = h.asarray()
        bits = h.metadata.channels[0].volume.bitsPerComponentSignificant
        meta = {"pixel_um": h.voxel_size().x, "channel_names": names, "bit_depth": bits}
        planes = _plane_settings(h.text_info.get("description", ""))

    images, acq = {}, {}
    for key, prefix in cfg["channels"].items():
        idx = [i for i, n in enumerate(names) if n.startswith(prefix)]
        if len(idx) != 1:
            raise ValueError(f"{path.name}: channel prefix '{prefix}' matches {idx} in {names}")
        images[key] = arr[idx[0]].astype(np.float64)
        acq[key] = next((p for p in planes if p["name"].startswith(prefix)), {})
    meta["saturation_value"] = 2**bits - 1
    meta["acquisition"] = acq
    return images, meta


def _plane_settings(description: str) -> list[dict]:
    """Per-plane channel name, exposure (ms) and EM gain from the ND2 description text."""
    import re

    out = []
    for block in re.split(r"Plane #\d+:", description)[1:]:
        name = re.search(r"Name:\s*([^\r\n]+)", block)
        exp = re.search(r"Exposure:\s*([\d.]+)\s*ms", block)
        gain = re.search(r"Multiplier:\s*(\d+)", block)
        out.append({"name": name.group(1).strip() if name else "",
                    "exposure_ms": float(exp.group(1)) if exp else np.nan,
                    "em_gain": int(gain.group(1)) if gain else -1})
    return out


# =============================================================================
# Image processing
# =============================================================================


def flatten(img: np.ndarray, median_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the slowly varying background (median filter). Returns (flat, background)."""
    bg = ndimage.median_filter(img, size=median_size, mode="reflect")
    return img - bg, bg


def robust_sigma(x: np.ndarray) -> float:
    """Noise estimate from the median absolute deviation (Gaussian-consistent)."""
    x = np.asarray(x).ravel()
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def snr_map(flat: np.ndarray, psf_sigma: float) -> np.ndarray:
    """Matched-filter (Gaussian) response divided by its own robust noise."""
    filt = ndimage.gaussian_filter(flat, psf_sigma)
    return filt / robust_sigma(filt)


def find_peaks(snr: np.ndarray, threshold: float, min_sep: int, border: int) -> np.ndarray:
    """Local maxima of an SNR map above threshold, away from the border. Returns (N, 2) as (y, x)."""
    size = 2 * min_sep + 1
    is_max = ndimage.maximum_filter(snr, size=size, mode="nearest") == snr
    mask = is_max & (snr > threshold)
    mask[:border, :] = mask[-border:, :] = False
    mask[:, :border] = mask[:, -border:] = False
    return np.argwhere(mask)


def refine_centroid(flat: np.ndarray, peaks: np.ndarray, half: int = 2) -> np.ndarray:
    """Intensity-weighted centroid in a (2*half+1)^2 window (negative pixels clipped). (N, 2) as (y, x)."""
    out = np.empty((len(peaks), 2), dtype=float)
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    for i, (y, x) in enumerate(peaks):
        w = np.clip(flat[y - half:y + half + 1, x - half:x + half + 1], 0, None)
        s = w.sum()
        out[i] = (y + (w * yy).sum() / s, x + (w * xx).sum() / s) if s > 0 else (y, x)
    return out


def sample_blank_positions(shape: tuple[int, int], occupied: np.ndarray, n: int, min_dist: float,
                           border: int, rng: np.random.Generator) -> np.ndarray:
    """Random subpixel (y, x) positions at least min_dist from every occupied position.

    Disk-based counterpart of Extraction/utils.sample_background_positions (which
    works on top-left box corners and seeds the global RNG).
    """
    free = np.ones(shape, dtype=bool)
    for y, x in np.round(occupied).astype(int):
        free[y, x] = False
    dist = ndimage.distance_transform_edt(free)
    valid = dist >= min_dist
    valid[:border, :] = valid[-border:, :] = False
    valid[:, :border] = valid[:, -border:] = False
    cand = np.argwhere(valid)
    if len(cand) < n:
        logging.warning(f"Only {len(cand)} valid blank positions (requested {n})")
        n = len(cand)
    pick = cand[rng.choice(len(cand), size=n, replace=False)]
    return pick + rng.uniform(-0.5, 0.5, size=pick.shape)


def second_moment_sigma(flat: np.ndarray, peak: tuple[int, int], half: int = 4) -> float:
    """Spot width (sigma, px) from the second moment in a window around a peak."""
    y, x = peak
    w = np.clip(flat[y - half:y + half + 1, x - half:x + half + 1], 0, None)
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    s = w.sum()
    if s <= 0:
        return np.nan
    cy, cx = (w * yy).sum() / s, (w * xx).sum() / s
    var = (w * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / s / 2.0
    return float(np.sqrt(var))


# =============================================================================
# Provenance
# =============================================================================


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def make_run_dir(cfg: dict, stage: str) -> Path:
    """Create Results/.../<stage>/run_NNN/ (next free number)."""
    base = REPO_ROOT / cfg["output_root"] / stage
    base.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name[4:]) for p in base.glob("run_[0-9][0-9][0-9]")] or [0])
    run = base / f"run_{n:03d}"
    run.mkdir()
    return run


def write_manifest(run: Path, cfg: dict, script: Path, inputs: list[Path], extra: dict | None = None) -> None:
    """Record what produced this run: config, script, git state, input checksums, environment."""
    import nd2, scipy, sklearn  # noqa: E401  (versions only)

    # The effective config (after --set overrides), not the file on disk.
    with open(run / "config.yaml", "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, sort_keys=False)
    shutil.copy(script, run / Path(script).name)
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(script).relative_to(REPO_ROOT)),
        "config_source": cfg["_path"],
        "config_overrides": cfg.get("_overrides", []),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "Revisions")),
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "nd2": nd2.__version__},
        "inputs": {str(p): sha256(p) for p in inputs},
    }
    if extra:
        manifest.update(extra)
    with open(run / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    logging.info(f"Manifest written to {run / 'manifest.yaml'}")


# =============================================================================
# Plotting
# =============================================================================


def apply_axis_standards(ax) -> None:
    """No top/right spines, 1.1 pt left/bottom spines, outward ticks (lab standard)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", which="major", labelsize=6, length=4, width=0.8, direction="out")
    ax.tick_params(axis="both", which="minor", length=2, width=0.6, direction="out")


def load_scm(name: str):
    """Crameri colormap: prefer the locbplots copy, fall back to a matplotlib gray map."""
    from matplotlib.colors import ListedColormap

    txt = REPO_ROOT.parents[1] / "Lab" / "locbplots" / "Materials" / "ScientificColourMaps8" / name / f"{name}.txt"
    if txt.exists():
        return ListedColormap(np.loadtxt(txt), name=name)
    logging.warning(f"Crameri map {name} not found at {txt}; using 'gray'")
    return "gray"


def save_pdf(fig, path: Path) -> None:
    fig.savefig(path)
    logging.info(f"Saved {path}")
