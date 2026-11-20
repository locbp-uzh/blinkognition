#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Empirical check of the photon conversion (detected photons per ADU = sensitivity / EM gain; config 16.0 / 300)
from the noise of the paper's raw movies (mean-variance analysis; choices confirmed by the user 2026-11-20).

For an EMCCD, variance (ADU^2) = F^2 x (G / S) x mean signal (ADU) + read-noise term, with F^2 = 2 (excess
noise factor at high EM gain). The slope k of variance against mean gives S / G = 2 / k photons per ADU,
whatever the camera's actual EM gain.

Per paper session, the first two 640 nm movies (by file name) of its first protein folder that have the
published Picasso localizations and a matching 488 nm vesicle snapshot:
  - dim pixels: the 640 nm movie, frames FRAMES (most molecules bleached), pixels farther than MASK_PX from
    every localization of that movie in the published run (background only, no blinking);
  - bright pixels: the 488 nm snapshot (10 frames, same camera settings), pixels with mean below MAX_MEAN_ADU
    (the iXon 888 at EM gain 300 turns nonlinear above about 30,000 ADU);
  - per pixel: mean over frames, and noise variance as half the variance of consecutive-frame differences
    (removes slow drift and bleaching);
  - pixels binned by mean (N_BINS log-spaced bins, bins with >= MIN_PIXELS pixels); weighted least squares of
    the bins' mean variance on their mean (weights: pixels per bin): slope k, intercept.
Reported per movie (all pixels; and 640 nm and 488 nm pixels separately as a diagnostic), with photons per
ADU = 2 / k and the implied sensitivity at the nominal gain, S = 2 x 300 / k.

Outputs (Results/Revisions/PaperData/conversion/run_NNN/): movies.csv, bins.csv, summary.csv,
conversion_fits.pdf (diagnostic), manifest.yaml, config.yaml, code/.

Usage (clone root, on an S3IT compute node; reads the raw movies in place):
    python Revisions/PaperData/conversion_check.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paperdata import REPO_ROOT, load_config, make_run_dir, write_manifest  # noqa: E402

FRAMES = (4000, 6000)
MASK_PX = 5
MAX_MEAN_ADU = 20000.0
N_BINS = 50
MIN_PIXELS = 30
MOVIES_PER_SESSION = 2
F2 = 2.0                               # EM excess noise factor squared, high-gain limit
FOV_RE = re.compile(r"_(\d{4})\.nd2$")


def pixel_stats(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """frames x H x W -> per-pixel mean and half the variance of consecutive-frame differences."""
    x = stack.astype(np.float64)
    return x.mean(axis=0), np.diff(x, axis=0).var(axis=0) / 2.0


def binned_fit(mean: np.ndarray, var: np.ndarray) -> tuple[pd.DataFrame, float, float]:
    edges = np.logspace(np.log10(mean.min()), np.log10(mean.max()), N_BINS + 1)
    idx = np.clip(np.digitize(mean, edges) - 1, 0, N_BINS - 1)
    rows = [{"bin": b, "n_pixels": int((idx == b).sum()), "mean_adu": float(mean[idx == b].mean()),
             "variance_adu2": float(var[idx == b].mean())} for b in range(N_BINS) if (idx == b).sum() >= MIN_PIXELS]
    bins = pd.DataFrame(rows)
    w = bins["n_pixels"].to_numpy(float)
    X = np.vstack([np.ones(len(bins)), bins["mean_adu"].to_numpy()]).T
    W = np.diag(w)
    a, k = np.linalg.solve(X.T @ W @ X, X.T @ W @ bins["variance_adu2"].to_numpy())
    return bins, float(k), float(a)


def pick_movies(cfg: dict) -> pd.DataFrame:
    cam = pd.read_csv(REPO_ROOT / cfg["step0_run"] / "camera.csv")
    run = Path(cfg["published_run"])
    out = []
    for ses, g in cam.groupby("session"):
        folder = sorted(g["folder"].unique())[0]
        n = 0
        for f in sorted(g.loc[g["folder"] == folder, "file"]):
            p = Path(f)
            locs = run / ses / folder / (p.stem + "_locs.hdf5")
            snap = p.with_name(p.name.replace("_640nm_TIRF2xLP_90pr_", "_488nm_TIRF2x_6pr_"))
            if locs.exists() and snap.exists():
                out.append({"session": ses, "folder": folder, "movie_640": str(p), "locs_640": str(locs),
                            "snapshot_488": str(snap)})
                n += 1
            if n == MOVIES_PER_SESSION:
                break
    return pd.DataFrame(out)


def main() -> None:
    import h5py
    import nd2
    from scipy.ndimage import distance_transform_edt
    cfg = load_config()
    run = make_run_dir(cfg, "conversion")
    picks = pick_movies(cfg)
    movies, allbins, inputs = [], [], []
    for _, m in picks.iterrows():
        with nd2.ND2File(m["movie_640"]) as h:
            stack = np.asarray(h.to_dask()[FRAMES[0]:FRAMES[1]])
        with h5py.File(m["locs_640"], "r") as f:
            locs = f["locs"][:]
        occupied = np.zeros(stack.shape[1:], bool)
        yy = np.clip(np.round(locs["y"]).astype(int), 0, occupied.shape[0] - 1)
        xx = np.clip(np.round(locs["x"]).astype(int), 0, occupied.shape[1] - 1)
        occupied[yy, xx] = True
        background = distance_transform_edt(~occupied) > MASK_PX
        m640, v640 = pixel_stats(stack)
        with nd2.ND2File(m["snapshot_488"]) as h:
            snap = np.asarray(h.asarray())
        m488, v488 = pixel_stats(snap.reshape(-1, *snap.shape[-2:]))
        keep488 = m488 < MAX_MEAN_ADU
        parts = {"640": (m640[background], v640[background]), "488": (m488[keep488], v488[keep488])}
        parts["all"] = (np.concatenate([parts["640"][0], parts["488"][0]]),
                        np.concatenate([parts["640"][1], parts["488"][1]]))
        row = {k: m[k] for k in ("session", "folder")}
        row.update(movie=Path(m["movie_640"]).name, n_localizations=int(len(locs)),
                   n_background_pixels_640=int(background.sum()), n_pixels_488=int(keep488.sum()))
        for name, (mu, var) in parts.items():
            bins, k, a = binned_fit(mu, var)
            bins.insert(0, "pixels", name)
            bins.insert(0, "movie", row["movie"])
            bins.insert(0, "session", row["session"])
            allbins.append(bins)
            row.update({f"slope_{name}": k, f"intercept_{name}": a,
                        f"photons_per_adu_{name}": F2 / k, f"sensitivity_at_gain300_{name}": F2 * 300 / k,
                        f"mean_range_{name}": f"{mu.min():.0f}-{mu.max():.0f}"})
        movies.append(row)
        inputs += [Path(m["movie_640"]), Path(m["locs_640"]), Path(m["snapshot_488"])]
        print(f"{row['session']} {row['movie']}: photons/ADU {row['photons_per_adu_all']:.4f} "
              f"(640 {row['photons_per_adu_640']:.4f}, 488 {row['photons_per_adu_488']:.4f})", flush=True)
    mv = pd.DataFrame(movies)
    mv.to_csv(run / "movies.csv", index=False, float_format="%.6g")
    pd.concat(allbins).to_csv(run / "bins.csv", index=False, float_format="%.6g")
    summ = []
    for scope, g in [("all movies", mv)] + [(s, g) for s, g in mv.groupby("session")]:
        for name in ("all", "640", "488"):
            v = g[f"photons_per_adu_{name}"]
            summ.append({"scope": scope, "pixels": name, "n_movies": len(g), "photons_per_adu_median": v.median(),
                         "photons_per_adu_min": v.min(), "photons_per_adu_max": v.max(),
                         "sensitivity_at_gain300_median": (F2 * 300 / g[f"slope_{name}"]).median()})
    pd.DataFrame(summ).to_csv(run / "summary.csv", index=False, float_format="%.5g")
    plot(pd.concat(allbins), mv, run / "conversion_fits.pdf", cfg)
    write_manifest(run, cfg, Path(__file__), inputs, hash_inputs=False,
                   extra={"frames": list(FRAMES), "mask_px": MASK_PX, "max_mean_adu": MAX_MEAN_ADU, "n_bins": N_BINS,
                          "min_pixels": MIN_PIXELS, "F2": F2, "config_photons_per_adu":
                          cfg["camera"]["sensitivity_e_per_adu"] / cfg["camera"]["em_gain"]})
    print(pd.DataFrame(summ).to_string(index=False))
    print(run)


def plot(bins: pd.DataFrame, mv: pd.DataFrame, path: Path, cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "pdf.fonttype": 42})
    sessions = sorted(mv["session"].unique())
    fig, axes = plt.subplots(2, 5, figsize=(180 / 25.4, 80 / 25.4), sharex=True, sharey=True)
    c_cfg = cfg["camera"]["sensitivity_e_per_adu"] / cfg["camera"]["em_gain"]
    for ax, s in zip(axes.ravel(), sessions):
        for (mov, pix), g in bins[(bins["session"] == s) & (bins["pixels"] != "all")].groupby(["movie", "pixels"]):
            ax.plot(g["mean_adu"], g["variance_adu2"], ls="none", marker="o", ms=1.5,
                    color="#0072B2" if pix == "640" else "#009E73")
        for _, r in mv[mv["session"] == s].iterrows():
            x = np.logspace(2, np.log10(MAX_MEAN_ADU), 50)
            ax.plot(x, r["intercept_all"] + r["slope_all"] * x, color="#000000", lw=0.7)
        x = np.logspace(2, np.log10(MAX_MEAN_ADU), 50)
        ax.plot(x, F2 / c_cfg * x, color="#D55E00", lw=0.7, ls="--")
        ax.set(xscale="log", yscale="log", title=s.split("_")[-1])
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Pixel mean (ADU)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Noise variance (ADU²)")
    fig.suptitle("Points: binned pixels (blue 640 nm background, green 488 nm snapshot); black: fit per movie; "
                 "dashed: 16.0/300 photons per ADU", fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
