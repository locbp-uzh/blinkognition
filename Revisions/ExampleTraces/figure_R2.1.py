#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Figure R2.1 (first-round review, Reviewer 2, point 1): example single-molecule fluorescence traces of
every protein construct in the paper.

For each construct (Halo-D106, SNAP-C148, Halo-K117, scGrx1, scGrx1-AcK20), N_PER_CONSTRUCT traces
drawn at random (fixed seed) from all traces inside vesicles that passed the paper's quality filter,
i.e. the traces the classifiers were trained and tested on. Not selected by appearance or by the
model's output.

Left: the full trace (6000 frames, 34.49 ms per frame). Right: the 10 s window with the most frames
classified ON, with the ON frames marked. Intensity: the 5 x 5 pixel sum with the per-trace GMM
background removed (the extraction's bg_rm traces), in detected photons per frame
(ADU x 16.0 e-/ADU / EM gain 300, Picasso's convention). ON frames: the paper's frame classifier
(Extraction/utils.py gmm_classify_frames on the min-max trace, posterior >= 0.9), as in Figs. 5D, 6C.

Usage:
    python figure_R2.1.py
        Replot from the CSVs next to this script (works in the figure folder as distributed).
    python figure_R2.1.py --build [--out <dir>]
        From the blinkognition repo: select and extract the traces from the published extraction
        (local mirror, Revisions/PaperData), write the CSVs, replot, and copy this script into --out.

Outputs (in --out, default: this script's folder): R2.1.pdf, R2.1_traces.csv (time and, per trace,
photons per frame and ON state), R2.1_selection.csv (which traces, zoom windows), R2.1_caption.txt,
R2.1_provenance.yaml and figure_R2.1.py (build only).
"""

from __future__ import annotations

# Standard library
import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# --- Constants ---
FIG = "R2.1"
SEED = 42
N_PER_CONSTRUCT = 3
CONSTRUCTS = ["HaloD106", "SNAPC148", "HaloK117", "scGrx1", "scGrx1AcK20"]   # paper order (Figs. 4-6)
LABELS = {"HaloD106": "Halo-D106", "HaloK117": "Halo-K117", "SNAPC148": "SNAP-C148",
          "scGrx1": "scGrx1", "scGrx1AcK20": "scGrx1-AcK20"}
COLORS = {"black": "#000000", "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
          "pink": "#CC79A7", "vermillion": "#D55E00"}
PROTEIN_COLORS = {"HaloD106": COLORS["blue"], "HaloK117": COLORS["orange"], "SNAPC148": COLORS["pink"],
                  "scGrx1": COLORS["green"], "scGrx1AcK20": COLORS["vermillion"]}
FRAME_S = 0.03449                    # s per frame (ND2 timestamps; 30 ms exposure plus readout)
ZOOM_S = 10.0                        # zoom window length
GMM_PROBA = 0.9                      # the paper's ON threshold (Features/config.yaml)
Y_STEP = 500                         # photons; common y-axis limits are rounded to this

FIG_WIDTH_IN = 180 / 25.4
FIG_HEIGHT_IN = 225 / 25.4
LW_TRACE = 0.5


# --- Style ---

def apply_style() -> None:
    """The paper's figure style (locbplots): Helvetica, else Arial; 7 pt text, 6 pt ticks and legend."""
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Microsoft/Arial.ttf"):
        if Path(path).exists():
            fm.fontManager.addfont(path)
            break
    available = {f.name for f in fm.fontManager.ttflist}
    sans = ["Helvetica", "Arial", "DejaVu Sans"] if "Helvetica" in available else ["Arial", "DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": sans, "font.size": 7,
        "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "lines.linewidth": 1.2, "axes.grid": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": "tight",
    })


def axis_standards(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", which="major", labelsize=6, length=4, width=0.8, direction="out")


# --- Build ---

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) inclusive frame runs of True."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def _movie_number(movie: str) -> int:
    """Acquisition file suffix (_0012 -> 12); 0 for the unsuffixed first file."""
    tail = movie.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() and len(tail) == 4 else 0


def build(out: Path) -> list[Path]:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "Revisions" / "PaperData"))
    sys.path.insert(0, str(repo / "Extraction"))
    from paperdata import frame_s, load_config, load_index, load_traces, to_photons, trace_path  # noqa: E402
    from utils import gmm_classify_frames  # noqa: E402  (Extraction/utils.py, the paper's classifier)

    cfg = load_config(local=True)
    if abs(frame_s(cfg) - FRAME_S) > 1e-9:
        raise ValueError("frame interval differs from PaperData config")
    rng = np.random.default_rng(SEED)
    zoom_n = int(round(ZOOM_S / FRAME_S))
    cols, sel, inputs = {}, [], []
    n_frames = None
    for c in CONSTRUCTS:
        idx = load_index(cfg, c)
        pool = idx[idx["passed_filter"]].sort_values("uniqueID").reset_index(drop=True)
        pick = pool.iloc[np.sort(rng.choice(len(pool), size=N_PER_CONSTRUCT, replace=False))]
        inputs.append(trace_path(cfg, c, "bg_rm"))
        bg = load_traces(cfg, c, "bg_rm")
        for k, (_, row) in enumerate(pick.iterrows(), start=1):
            adu = bg[row["uniqueID"]].to_numpy(np.float64)
            n_frames = n_frames or adu.size
            minmax = (adu - adu.min()) / (adu.max() - adu.min())
            _, _, on = gmm_classify_frames(minmax, GMM_PROBA, 0.0)
            counts = np.convolve(on.astype(int), np.ones(zoom_n, dtype=int), mode="valid")
            z0 = int(np.argmax(counts))                  # earliest window with the most ON frames
            name = f"{c}_{k}"
            cols[f"{name}_photons"] = to_photons(adu, cfg)
            cols[f"{name}_on"] = on.astype(int)
            sel.append({"construct": c, "label": LABELS[c], "trace": k, "column": name,
                        "uniqueID": int(row["uniqueID"]), "session": row["session"], "movie": row["movie"],
                        "movie_number": _movie_number(row["movie"]), "x": int(row["x"]), "y": int(row["y"]),
                        "n_on_frames": int(on.sum()), "n_on_events": len(_runs(on)),
                        "zoom_start_frame": z0, "zoom_end_frame": z0 + zoom_n - 1,
                        "pool_size": len(pool)})
    frames = np.arange(n_frames)
    traces = pd.DataFrame({"frame": frames, "time_s": frames * FRAME_S, **cols})
    traces.to_csv(out / f"{FIG}_traces.csv", index=False, float_format="%.4f")
    pd.DataFrame(sel).to_csv(out / f"{FIG}_selection.csv", index=False)
    return inputs + [repo / cfg["step0_run"] / f"index_IN_{c}.csv" for c in CONSTRUCTS]


# --- Plot ---

def plot(out: Path) -> Path:
    apply_style()                        # here, not only in main: importing Extraction/utils.py resets the fonts
    traces = pd.read_csv(out / f"{FIG}_traces.csv")
    sel = pd.read_csv(out / f"{FIG}_selection.csv")
    t = traces["time_s"].to_numpy()
    ph_cols = [f"{n}_photons" for n in sel["column"]]
    lo = Y_STEP * np.floor(traces[ph_cols].min().min() / Y_STEP)
    hi = Y_STEP * np.ceil(traces[ph_cols].max().max() / Y_STEP)

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    outer = fig.add_gridspec(len(CONSTRUCTS) + 1, 1, height_ratios=[1] * len(CONSTRUCTS) + [0.12],
                             hspace=0.32, left=0.11, right=0.99, top=0.985, bottom=0.035)
    blocks = {c: outer[j].subgridspec(N_PER_CONSTRUCT, 2, width_ratios=[3.2, 1], hspace=0.18, wspace=0.12)
              for j, c in enumerate(CONSTRUCTS)}
    for _, r in sel.iterrows():
        color = PROTEIN_COLORS[r["construct"]]
        ph = traces[f"{r['column']}_photons"].to_numpy()
        on = traces[f"{r['column']}_on"].to_numpy().astype(bool)
        z0, z1 = int(r["zoom_start_frame"]), int(r["zoom_end_frame"])
        g = blocks[r["construct"]]
        a, b = fig.add_subplot(g[r["trace"] - 1, 0]), fig.add_subplot(g[r["trace"] - 1, 1])

        a.axvspan(t[z0], t[z1], color=color, alpha=0.15, lw=0)
        a.plot(t, ph, color=color, lw=LW_TRACE)
        a.set(xlim=(0, t[-1]), ylim=(lo, hi))
        a.text(0.995, 0.97, f"{r['session'].split('_')[-1]}, movie {r['movie_number']}, trace {r['uniqueID']}",
               transform=a.transAxes, fontsize=6, color="black", va="top", ha="right")

        tz = t[z0:z1 + 1] - t[z0]                        # time from the window start
        b.plot(tz, ph[z0:z1 + 1], color=color, lw=LW_TRACE)
        for s, e in _runs(on[z0:z1 + 1]):
            b.plot([tz[s] - FRAME_S / 2, tz[e] + FRAME_S / 2], [hi, hi], color=COLORS["black"],
                   lw=2.0, solid_capstyle="butt", clip_on=False)
        b.set(xlim=(-FRAME_S / 2, tz[-1] + FRAME_S / 2), ylim=(lo, hi))
        b.set_yticklabels([])

        last_of_group = r["trace"] == N_PER_CONSTRUCT
        for ax in (a, b):
            axis_standards(ax)
            if not last_of_group:
                ax.set_xticklabels([])
        if r["trace"] == (N_PER_CONSTRUCT + 1) // 2:
            a.set_ylabel("Photons per frame")
        if r["trace"] == 1:
            a.text(-0.105, 1.0, "abcde"[CONSTRUCTS.index(r["construct"])], transform=a.transAxes, fontsize=8,
                   fontweight="bold", va="bottom", ha="left")
            a.set_title(r["label"], loc="left", pad=3, fontsize=7)
        if r["construct"] == CONSTRUCTS[-1] and last_of_group:
            a.set_xlabel("Time (s)")
            b.set_xlabel("Time from window start (s)")

    leg = fig.add_subplot(outer[len(CONSTRUCTS)])
    leg.axis("off")
    leg.legend(handles=[
        Line2D([], [], color=COLORS["black"], lw=2.0, solid_capstyle="butt",
               label="ON frames (paper's frame classifier, posterior ≥ 0.9)"),
        Line2D([], [], color=COLORS["black"], lw=6, alpha=0.15, label="Window shown on the right (10 s)"),
    ], loc="center", ncol=2, frameon=False, handlelength=2.0)

    pdf = out / f"{FIG}.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


# --- Caption and provenance ---

def caption(out: Path) -> str:
    sel = pd.read_csv(out / f"{FIG}_selection.csv")
    pools = ", ".join(f"{LABELS[c]} {int(sel.loc[sel['construct'] == c, 'pool_size'].iloc[0])}" for c in CONSTRUCTS)
    return (
        "Figure R2.1. Example single-molecule fluorescence traces of HMSiR on every protein construct. "
        "a) Halo-D106, b) SNAP-C148, c) Halo-K117, d) scGrx1, e) scGrx1-AcK20. For each construct, "
        f"{N_PER_CONSTRUCT} traces drawn at random from all traces inside vesicles that passed the quality "
        f"filter ({pools} traces). Left, full trace (6000 frames, 34.49 ms per frame, 640 nm excitation); "
        "right, the 10 s window with the most ON frames (shaded on the left; time from the window start), "
        "with ON frames (black bars) as classified for Figures 5D and 6C (two-component Gaussian mixture, "
        "posterior probability >= 0.9). Intensity is the 5 x 5 pixel sum with the per-trace background "
        "removed, in detected photons per frame; all panels share one intensity scale. Each trace is a "
        "different molecule; labels give the session, the movie and the trace identifier of the published "
        "dataset. No statistical comparisons were performed."
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_provenance(out: Path, inputs: list[Path]) -> None:
    import yaml
    here = Path(__file__).resolve()
    git = lambda *a: subprocess.run(["git", "-C", str(here.parent), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {
        "figure": FIG, "created": datetime.now().isoformat(timespec="seconds"), "script": str(here),
        "script_sha256": _sha256(here), "git_commit": git("rev-parse", "HEAD") or None,
        "git_dirty": bool(git("status", "--porcelain", "--", str(here), str(here.parents[1] / "PaperData"))),
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "seed": SEED, "n_per_construct": N_PER_CONSTRUCT,
        "inputs": {str(p): _sha256(p) for p in inputs},
    }
    with open(out / f"{FIG}_provenance.yaml", "w") as f:
        yaml.safe_dump(prov, f, sort_keys=False, allow_unicode=True)


# --- Main ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="select and extract the traces (blinkognition repo)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    apply_style()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.build:
        inputs = build(out)
        (out / f"{FIG}_caption.txt").write_text(caption(out) + "\n")
        if Path(__file__).resolve() != (out / Path(__file__).name).resolve():
            shutil.copy(Path(__file__).resolve(), out / Path(__file__).name)
        write_provenance(out, inputs)
    print(f"Saved: {plot(out)}")


if __name__ == "__main__":
    main()
