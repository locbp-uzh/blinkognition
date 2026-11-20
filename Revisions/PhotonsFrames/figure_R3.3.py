#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Figure R3.3 (first-round review, Reviewer 3, point 3): detected photons per ON event against ON time, and
photon numbers per molecule, for every construct. Trimmed to the request (user, 2026-11-20): the same-day
photons per fully-ON frame panel of the first version (R3.3a h) and the quality-filter analysis (R3.3b) are not
part of the revision (revisions_figures/not_in_revision/; git history, commit 5bf026c).

Data: every ON event of every published filtered trace inside vesicles (events.py run): ON frames from the
paper's classifier (as for Figs. 5D and 6C), photons = background-removed 5 x 5 pixel sum x 0.062 photons
per ADU (measured; Revisions/PaperData camera.photons_per_adu), 34.49 ms per frame.

a-e) Per construct, 2D histogram of photons per event against ON duration (percentage of the construct's
     events per bin, log color scale, Crameri batlow). Solid line: median over molecules of each
     molecule's median photons per event of that duration (durations with >= MIN_MOLECULES molecules).
     Dashed line: the construct's median photons per fully-ON frame x duration (the photons a molecule
     emitting through whole frames would give).
f)   Median photons per event, one point per molecule.
g)   Total detected photons per molecule over the ON frames of the 207 s trace, one point per molecule.
Table (R3.3_table_construct.csv): photon budget per construct (photons per event, per fully-ON frame, per
molecule; median and quartiles over molecules).
Boxes: median and quartiles, whiskers 1.5 x IQR (Tukey). Summaries are per molecule first, then over
molecules; events are never treated as independent. No statistical tests.

Usage:
    python figure_R3.3.py
        Replot from the CSVs next to this script (works in the figure folder as distributed).
    python figure_R3.3.py --build --events <events run> [--out <dir>]
        From the blinkognition repo: build the CSVs from the events run, replot, copy this script and the
        color map into --out.
"""

from __future__ import annotations

# Standard library
import argparse
import hashlib
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# --- Constants ---
FIG = "R3.3"
FRAME_MS = 34.49
CONSTRUCTS = ["HaloD106", "SNAPC148", "HaloK117", "scGrx1", "scGrx1AcK20"]
LABELS = {"HaloD106": "Halo-D106", "HaloK117": "Halo-K117", "SNAPC148": "SNAP-C148",
          "scGrx1": "scGrx1", "scGrx1AcK20": "scGrx1-AcK20"}
PROTEIN_COLORS = {"HaloD106": "#0072B2", "HaloK117": "#E69F00", "SNAPC148": "#CC79A7",
                  "scGrx1": "#009E73", "scGrx1AcK20": "#D55E00"}
BLACK = "#000000"
MAX_FRAMES = 300                     # events up to 300 frames (10.3 s) shown; longer ones counted
PHOTON_EDGES = np.logspace(2, 6, 41)  # 100 to 10^6 photons per event
MIN_MOLECULES = 20                   # per-duration median shown when this many molecules contribute
COLORMAP_FILE = "R3.3_colormap_batlow.txt"  # Crameri batlow (assets/colormaps/batlow.txt in the repo)
JITTER = 0.32
RNG_SEED = 42

FIG_WIDTH_IN = 180 / 25.4
FIG_HEIGHT_IN = 115 / 25.4


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
    ax.tick_params(axis="both", which="minor", length=2, width=0.6, direction="out")


# --- Build ---

def _q(s: pd.Series) -> tuple[float, float, float, int]:
    v = s.dropna().to_numpy(float)
    if not v.size:
        return np.nan, np.nan, np.nan, 0
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), int(v.size)


def summarize(mol: pd.DataFrame, ev: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, m in mol.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        e = ev[ev["uniqueID"].isin(m["uniqueID"]) & (ev["construct"] == m["construct"].iloc[0])]
        r = dict(zip(by, key))
        r.update(n_molecules=len(m), n_events=len(e), pct_events_1frame=100 * (e["n_frames"] == 1).mean(),
                 n_events_nonpositive_photons=int((e["photons"] <= 0).sum()),
                 n_events_longer_than_shown=int((e["n_frames"] > MAX_FRAMES).sum()))
        for col, name in (("mean_on_frames", "mean_on_frames"), ("median_photons_per_event", "photons_per_event"),
                          ("median_photons_per_event_padded", "photons_per_event_padded"),
                          ("photons_per_full_frame", "photons_per_full_frame"),
                          ("total_photons_on", "total_photons_per_molecule")):
            med, q25, q75, n = _q(m[col])
            r.update({f"{name}_median": med, f"{name}_q25": q25, f"{name}_q75": q75, f"{name}_n_molecules": n})
        r["mean_on_ms_median"] = r["mean_on_frames_median"] * FRAME_MS
        rows.append(r)
    return pd.DataFrame(rows)


def build(events_run: Path, out: Path) -> list[Path]:
    ev_p, mol_p = events_run / "events.csv.gz", events_run / "molecules.csv"
    ev, mol = pd.read_csv(ev_p), pd.read_csv(mol_p)
    hist, med = [], []
    for c in CONSTRUCTS:
        e = ev[ev["construct"] == c]
        shown = e[(e["photons"] > 0) & (e["n_frames"] <= MAX_FRAMES)]
        counts, _, _ = np.histogram2d(shown["n_frames"], shown["photons"],
                                      bins=[np.arange(0.5, MAX_FRAMES + 1.5), PHOTON_EDGES])
        pct = 100 * counts / len(e)
        for i in range(MAX_FRAMES):
            for j in range(len(PHOTON_EDGES) - 1):
                if counts[i, j]:
                    hist.append({"construct": c, "n_frames": i + 1, "duration_ms": (i + 1) * FRAME_MS,
                                 "photon_bin": j, "photons_lo": PHOTON_EDGES[j], "photons_hi": PHOTON_EDGES[j + 1],
                                 "n_events": int(counts[i, j]), "pct_of_events": pct[i, j]})
        per_mol = e.groupby(["n_frames", "uniqueID"])["photons"].median().reset_index()
        rate = float(mol.loc[mol["construct"] == c, "photons_per_full_frame"].median())
        for n, g in per_mol.groupby("n_frames"):
            if n > MAX_FRAMES:
                break
            m50, q25, q75, k = _q(g["photons"])
            med.append({"construct": c, "n_frames": n, "duration_ms": n * FRAME_MS, "n_molecules": k,
                        "median_photons_per_event": m50, "q25": q25, "q75": q75, "shown": k >= MIN_MOLECULES,
                        "full_frame_photons_x_duration": rate * n})
    pd.DataFrame(hist).to_csv(out / f"{FIG}_panels_a-e_histogram.csv", index=False)
    pd.DataFrame(med).to_csv(out / f"{FIG}_panels_a-e_medians.csv", index=False)
    cols = ["construct", "session", "uniqueID", "movie", "n_events", "mean_on_frames", "median_photons_per_event",
            "median_photons_per_event_padded", "photons_per_full_frame", "n_interior_frames", "total_photons_on"]
    mol[cols].to_csv(out / f"{FIG}_panels_f-g_molecules.csv", index=False)
    summarize(mol, ev, ["construct"]).to_csv(out / f"{FIG}_table_construct.csv", index=False)
    repo = Path(__file__).resolve().parents[2]
    shutil.copy(repo / "assets" / "colormaps" / "batlow.txt", out / COLORMAP_FILE)
    return [ev_p, mol_p, events_run / "manifest.yaml"]


# --- Plot ---

def _letter(ax, letter: str, x: float = -0.32) -> None:
    ax.text(x, 1.10, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def _box_strip(ax, groups: list[np.ndarray], colors: list[str], rng: np.random.Generator) -> None:
    for i, (v, col) in enumerate(zip(groups, colors)):
        x = i + rng.uniform(-JITTER, JITTER, v.size)
        ax.scatter(x, v, s=1.0, color=col, alpha=0.35, lw=0, rasterized=True, zorder=1)
    ax.boxplot(groups, positions=range(len(groups)), widths=0.55, whis=1.5, showfliers=False, zorder=2,
               medianprops={"color": BLACK, "lw": 1.0}, boxprops={"facecolor": "none", "lw": 0.8},
               whiskerprops={"lw": 0.8}, capprops={"lw": 0.8}, patch_artist=True)


def plot(out: Path) -> Path:
    apply_style()
    hist = pd.read_csv(out / f"{FIG}_panels_a-e_histogram.csv")
    med = pd.read_csv(out / f"{FIG}_panels_a-e_medians.csv")
    mol = pd.read_csv(out / f"{FIG}_panels_f-g_molecules.csv")
    cmap = ListedColormap(np.loadtxt(out / COLORMAP_FILE), name="batlow")
    rng = np.random.default_rng(RNG_SEED)

    x_edges = (np.arange(0.5, MAX_FRAMES + 1.5)) * FRAME_MS
    vmax = hist["pct_of_events"].max()
    norm = LogNorm(vmin=1e-3, vmax=vmax)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    gs = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.55, left=0.07, right=0.97, top=0.93, bottom=0.12)
    axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(7)]
    mesh = None
    for k, c in enumerate(CONSTRUCTS):
        ax = axes[k]
        grid = np.full((MAX_FRAMES, len(PHOTON_EDGES) - 1), np.nan)
        h = hist[hist["construct"] == c]
        grid[h["n_frames"].to_numpy() - 1, h["photon_bin"].to_numpy()] = h["pct_of_events"]
        mesh = ax.pcolormesh(x_edges, PHOTON_EDGES, grid.T, cmap=cmap, norm=norm, rasterized=True)
        m = med[(med["construct"] == c) & med["shown"]]
        ax.plot(m["duration_ms"], m["full_frame_photons_x_duration"], color=BLACK, ls="--", lw=0.8)
        ax.plot(m["duration_ms"], m["median_photons_per_event"], color="white", lw=2.2)
        ax.plot(m["duration_ms"], m["median_photons_per_event"], color=BLACK, lw=0.9, marker="o", ms=1.6)
        ax.set(xscale="log", yscale="log", xlim=(x_edges[0], x_edges[-1]), ylim=(PHOTON_EDGES[0], PHOTON_EDGES[-1]),
               xlabel="ON duration (ms)", ylabel="Photons per event")
        ax.set_title(LABELS[c], pad=3)
        axis_standards(ax)
        _letter(ax, "abcde"[k])
    cax = axes[4].inset_axes([0.52, 0.10, 0.44, 0.045])     # empty corner: long, dim events do not occur
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label("Events (%)", fontsize=6, labelpad=1)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5, length=2, width=0.6, pad=1)
    cb.outline.set_linewidth(0.6)
    axes[0].legend(handles=[
        Line2D([], [], color=BLACK, lw=0.9, marker="o", ms=1.6, label="Median"),
        Line2D([], [], color=BLACK, lw=0.8, ls="--", label="Full-frame photons × duration")],
        loc="upper left", frameon=False, handlelength=1.8, borderaxespad=0.2)

    colors = [PROTEIN_COLORS[c] for c in CONSTRUCTS]
    for ax, col, ylabel, letter in ((axes[5], "median_photons_per_event", "Photons per event", "f"),
                                    (axes[6], "total_photons_on", "Photons per molecule", "g")):
        groups = [mol.loc[mol["construct"] == c, col].dropna().to_numpy() for c in CONSTRUCTS]
        groups = [g[g > 0] for g in groups]
        _box_strip(ax, groups, colors, rng)
        ax.set(yscale="log", ylabel=ylabel, xlim=(-0.6, len(CONSTRUCTS) - 0.4))
        ax.set_xticks(range(len(CONSTRUCTS)), [LABELS[c] for c in CONSTRUCTS], rotation=40, ha="right")
        axis_standards(ax)
        _letter(ax, letter)


    pdf = out / f"{FIG}.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


# --- Caption and provenance ---

def caption(out: Path) -> str:
    t = pd.read_csv(out / f"{FIG}_table_construct.csv").set_index("construct")
    n = ", ".join(f"{LABELS[c]} {int(t.loc[c, 'n_molecules'])} molecules / {int(t.loc[c, 'n_events'])} events"
                  for c in CONSTRUCTS)
    return (
        "Figure R3.3. Detected photons per ON event and per molecule. a-e) Photons per ON event against "
        "ON duration for every event of every filtered single-molecule trace inside vesicles (Halo-D106, "
        "SNAP-C148, Halo-K117, scGrx1, scGrx1-AcK20); color, percentage of the construct's events per bin "
        "(log scale); solid line, median over molecules of each molecule's median photons per event of that "
        f"duration (durations with at least {MIN_MOLECULES} molecules); dashed line, the construct's median "
        "photons per fully-ON frame (inner frames of events of three or more frames) times the duration. "
        "ON frames as classified for Figures 5D and 6C; an event lasting n frames lasted between n - 1 and "
        "n + 1 frames (34.49 ms per frame; 30 ms exposure). Events with zero or negative photon sums or "
        f"longer than {MAX_FRAMES} frames are not shown (table). f) Median photons per event and g) total "
        "detected photons over the ON frames of the 207 s trace, one point per molecule. Photons are detected photons (background-removed 5 x 5 pixel sum x 0.062 photons per ADU, measured "
        "from the camera noise by mean-variance analysis of the raw movies; no quantum-efficiency correction). Boxes, median and quartiles; whiskers, 1.5 x IQR. "
        "Molecules are distinct; sessions differ in excitation power, so f and g mix construct and session "
        f"effects. n: {n}. Photon budget per construct: Table SX. No statistical comparisons were performed."
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
        "git_dirty": bool(git("status", "--porcelain", "--", str(here.parent))),
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "inputs": {str(p): _sha256(p) for p in inputs},
    }
    with open(out / f"{FIG}_provenance.yaml", "w") as f:
        yaml.safe_dump(prov, f, sort_keys=False, allow_unicode=True)


# --- Main ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--events", type=Path, help="events.py run folder (events.csv.gz, molecules.csv)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.build:
        if not args.events:
            ap.error("--build needs --events")
        inputs = build(args.events, out)
        (out / f"{FIG}_caption.txt").write_text(caption(out) + "\n")
        if Path(__file__).resolve() != (out / Path(__file__).name).resolve():
            shutil.copy(Path(__file__).resolve(), out / Path(__file__).name)
        write_provenance(out, inputs)
    print(f"Saved: {plot(out)}")


if __name__ == "__main__":
    main()
