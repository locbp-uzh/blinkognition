#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Figure R3.3b (first-round review, Reviewer 3, point 3): does the quality filter's signal-to-noise gate
discard fast-blinking molecules?

a) Fate of every trace inside vesicles, per construct: passed; rejected with GMM separation < 3 (the
   explicit gate at 2 plus the frame classifier's minimum of 3) and molecule-like; rejected with
   separation < 3, not molecule-like; rejected by the other criteria (peak number, timing, invalid data).
   From the replay of the paper's filter, which reproduces the published selection exactly.
b) GMM separation of the traces inside vesicles (per construct) and of background traces (random
   positions >= 5 px from any protein ROI, same movies, all constructs pooled), cumulative fractions;
   dotted lines at 2 (stated in the SI) and 3 (effective).
c) Halo-D106 and Halo-K117 on the days both were measured: molecule-like rejected traces as a percentage
   of the passed traces (bars) and the number expected from noise alone (open markers: rejected traces x
   the session's background false-flag rate).
d, e) Halo-K117, same days: per molecule, median photons per ON event (d) and fraction of single-frame
   events (e), passed traces against molecule-like rejected ones.
f) Median photons per ON event per molecule, passed against molecule-like rejected traces, every
   construct (sessions pooled); dotted line, median per-frame noise (1.4826 x MAD) of the passed traces.
Molecule-like: more frames above the trace's median + 5 sigma than 99 % of the background traces of the
same session. Boxes: median and quartiles, whiskers 1.5 x IQR. No statistical tests.

Usage:
    python figure_R3.3b.py
        Replot from the CSVs next to this script (works in the figure folder as distributed).
    python figure_R3.3b.py --build [--out <dir>] [--audit-in ... --audit-bg ... --molecule-like ...
                                    --flagged ... --events ... --counts ...]
        From the blinkognition repo: build the CSVs from the analysis runs (defaults: the R3.3b reference
        runs under Results/Revisions/PhotonsFrames), replot, copy this script into --out.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# --- Constants ---
FIG = "R3.3b"
CONSTRUCTS = ["HaloD106", "SNAPC148", "HaloK117", "scGrx1", "scGrx1AcK20"]
LABELS = {"HaloD106": "Halo-D106", "HaloK117": "Halo-K117", "SNAPC148": "SNAP-C148",
          "scGrx1": "scGrx1", "scGrx1AcK20": "scGrx1-AcK20"}
PROTEIN_COLORS = {"HaloD106": "#0072B2", "HaloK117": "#E69F00", "SNAPC148": "#CC79A7",
                  "scGrx1": "#009E73", "scGrx1AcK20": "#D55E00"}
BLACK, SKY_BLUE, YELLOW, WHITE = "#000000", "#56B4E9", "#F0E442", "#FFFFFF"
FATES = [("passed", "Passed", BLACK), ("snr_molecule_like", "SNR < 3, molecule-like", SKY_BLUE),
         ("snr_other", "SNR < 3, other", WHITE), ("other", "Other criteria", YELLOW)]
SNR_STATED, SNR_EFFECTIVE = 2.0, 3.0
SHARED_SESSIONS = ["20231130_SP_Exp1", "20231201_SP_Exp2"]
K_MAIN = 5
PHOTONS_PER_ADU = 16.0 / 300.0         # sensitivity / EM gain (Revisions/PaperData/config.yaml)
JITTER = 0.32
RNG_SEED = 42
FIG_WIDTH_IN = 180 / 25.4
FIG_HEIGHT_IN = 120 / 25.4

R = "Results/Revisions/PhotonsFrames"
DEFAULTS = {"audit_in": f"{R}/s3it/filter_audit/run_001", "audit_bg": f"{R}/s3it/filter_audit_background/run_001",
            "molecule_like": f"{R}/molecule_like/run_001", "flagged": f"{R}/flagged_features/run_001",
            "events": f"{R}/events/run_001", "counts": f"{R}/s3it/bright_frames/run_001"}


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

def build(repo: Path, a: argparse.Namespace, out: Path) -> list[Path]:
    p = {k: repo / getattr(a, k) for k in DEFAULTS}
    files = {"audit_in": p["audit_in"] / "traces.csv", "audit_bg": p["audit_bg"] / "traces.csv",
             "ml": p["molecule_like"] / "traces.csv", "ml_session": p["molecule_like"] / "summary_session.csv",
             "ml_thr": p["molecule_like"] / "thresholds.csv", "flagged": p["flagged"] / "molecules_flagged.csv",
             "comparison": p["flagged"] / "comparison.csv", "events": p["events"] / "molecules.csv",
             "counts": p["counts"] / "counts_IN.csv"}
    ml = pd.read_csv(files["ml"])
    ml["fate"] = np.select([ml["passed"], ml[f"molecule_like_k{K_MAIN}"], ml["rejected_snr"]],
                           ["passed", "snr_molecule_like", "snr_other"], "other")
    fates = ml.groupby(["construct", "fate"]).size().rename("n_traces").reset_index()
    fates["pct"] = 100 * fates["n_traces"] / fates.groupby("construct")["n_traces"].transform("sum")
    fates.to_csv(out / f"{FIG}_panel_a_fates.csv", index=False)

    ain = pd.read_csv(files["audit_in"])[["construct", "uniqueID", "session", "separation", "criterion"]]
    abg = pd.read_csv(files["audit_bg"])[["construct", "uniqueID", "separation", "criterion"]]
    pd.concat([ain.assign(traces="IN"), abg.assign(traces="background")], ignore_index=True).to_csv(
        out / f"{FIG}_panel_b_separation.csv", index=False, float_format="%.5f")

    s = pd.read_csv(files["ml_session"])
    s = s[s["construct"].isin(["HaloD106", "HaloK117"]) & s["session"].isin(SHARED_SESSIONS)].copy()
    s = s[["construct", "session", "n_in", "n_passed", "n_rejected_snr", f"n_molecule_like_k{K_MAIN}",
           f"expected_false_flags_k{K_MAIN}", f"threshold_k{K_MAIN}", f"background_pct_above_k{K_MAIN}"]]
    s.columns = ["construct", "session", "n_in", "n_passed", "n_rejected_snr", "n_molecule_like",
                 "expected_false_flags", "background_threshold_frames", "background_pct_above"]
    s["molecule_like_pct_of_passed"] = 100 * s["n_molecule_like"] / s["n_passed"]
    s["expected_pct_of_passed"] = 100 * s["expected_false_flags"] / s["n_passed"]
    s.to_csv(out / f"{FIG}_panel_c_same_day.csv", index=False, float_format="%.4f")

    cols = ["construct", "session", "uniqueID", "n_events", "frac_events_1frame", "mean_on_frames",
            "median_photons_per_event", "mean_off_frames", "duty_cycle"]
    mol = pd.concat([pd.read_csv(files["events"])[cols].assign(group="passed"),
                     pd.read_csv(files["flagged"])[cols].assign(group="molecule-like rejected")], ignore_index=True)
    mol.to_csv(out / f"{FIG}_panels_d-f_molecules.csv", index=False, float_format="%.4f")

    cnt = pd.read_csv(files["counts"])[["construct", "uniqueID", "sigma_adu"]]
    passed = ml.loc[ml["passed"], ["construct", "uniqueID"]].merge(cnt, on=["construct", "uniqueID"])
    noise = passed.groupby("construct")["sigma_adu"].median().mul(PHOTONS_PER_ADU).rename("noise_sigma_photons")
    noise.reset_index().to_csv(out / f"{FIG}_panel_f_noise.csv", index=False, float_format="%.2f")

    ain.groupby(["construct", "criterion"]).size().unstack(fill_value=0).reset_index().to_csv(
        out / f"{FIG}_table_criteria.csv", index=False)
    pd.read_csv(files["ml_session"]).to_csv(out / f"{FIG}_table_molecule_like_session.csv", index=False,
                                            float_format="%.3f")
    pd.read_csv(files["ml_thr"]).to_csv(out / f"{FIG}_table_background_thresholds.csv", index=False,
                                        float_format="%.3f")
    pd.read_csv(files["comparison"]).to_csv(out / f"{FIG}_table_flagged_vs_passed.csv", index=False,
                                            float_format="%.4f")
    return list(files.values()) + [p[k] / "manifest.yaml" for k in DEFAULTS]


# --- Plot ---

def _letter(ax, letter: str, x: float = -0.30) -> None:
    ax.text(x, 1.08, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def _box_strip(ax, groups, colors, rng, s=1.2) -> None:
    for i, (v, col) in enumerate(zip(groups, colors)):
        ax.scatter(i + rng.uniform(-JITTER, JITTER, v.size), v, s=s, color=col, alpha=0.4, lw=0,
                   rasterized=True, zorder=1)
    ax.boxplot(groups, positions=range(len(groups)), widths=0.55, whis=1.5, showfliers=False, zorder=2,
               medianprops={"color": BLACK, "lw": 1.0}, boxprops={"facecolor": "none", "lw": 0.8},
               whiskerprops={"lw": 0.8}, capprops={"lw": 0.8}, patch_artist=True)


def plot(out: Path) -> Path:
    apply_style()
    rng = np.random.default_rng(RNG_SEED)
    fates = pd.read_csv(out / f"{FIG}_panel_a_fates.csv")
    sep = pd.read_csv(out / f"{FIG}_panel_b_separation.csv")
    same = pd.read_csv(out / f"{FIG}_panel_c_same_day.csv")
    mol = pd.read_csv(out / f"{FIG}_panels_d-f_molecules.csv")
    noise = pd.read_csv(out / f"{FIG}_panel_f_noise.csv").set_index("construct")["noise_sigma_photons"]

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    gs = fig.add_gridspec(2, 3, hspace=0.80, wspace=0.55, left=0.10, right=0.98, top=0.88, bottom=0.21,
                          width_ratios=[1.15, 1, 1])
    ax = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

    # a) fates
    a = ax[0]
    y = np.arange(len(CONSTRUCTS))[::-1]
    left = np.zeros(len(CONSTRUCTS))
    for key, label, col in FATES:
        v = np.array([fates.loc[(fates["construct"] == c) & (fates["fate"] == key), "pct"].sum() for c in CONSTRUCTS])
        a.barh(y, v, left=left, color=col, edgecolor=BLACK, lw=0.6, height=0.7, label=label)
        left += v
    a.set_yticks(y, [LABELS[c] for c in CONSTRUCTS])
    a.set(xlim=(0, 100), xlabel="Traces inside vesicles (%)")
    a.legend(handles=[Patch(facecolor=col, edgecolor=BLACK, lw=0.6, label=label) for _, label, col in FATES],
             loc="lower left", bbox_to_anchor=(-0.02, 1.01), frameon=False, ncol=2, handlelength=1.2,
             fontsize=5.5, columnspacing=0.8)
    axis_standards(a)
    a.text(-0.45, 1.30, "a", transform=a.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")

    # b) separation ECDFs
    b = ax[1]
    for c in CONSTRUCTS:
        v = np.sort(sep.loc[(sep["traces"] == "IN") & (sep["construct"] == c), "separation"].to_numpy())
        b.plot(v, np.arange(1, v.size + 1) / v.size, color=PROTEIN_COLORS[c], lw=1.0)
    v = np.sort(sep.loc[sep["traces"] == "background", "separation"].to_numpy())
    b.plot(v, np.arange(1, v.size + 1) / v.size, color=BLACK, lw=1.0, ls="--")
    for x in (SNR_STATED, SNR_EFFECTIVE):
        b.axvline(x, color=BLACK, ls=":", lw=0.8)
    b.set(xscale="log", xlim=(0.3, 300), ylim=(0, 1.01), xlabel="GMM separation (signal-to-noise)",
          ylabel="Cumulative fraction of traces")
    axis_standards(b)
    _letter(b, "b")

    # c) same-day Halo-D106 vs Halo-K117
    c_ax = ax[2]
    xs, labels = [], []
    for i, ses in enumerate(SHARED_SESSIONS):
        for j, c in enumerate(("HaloD106", "HaloK117")):
            r = same[(same["construct"] == c) & (same["session"] == ses)].iloc[0]
            x = i * 2.6 + j
            c_ax.bar(x, r["molecule_like_pct_of_passed"], width=0.8, color=PROTEIN_COLORS[c], edgecolor=BLACK, lw=0.6)
            c_ax.plot(x, r["expected_pct_of_passed"], ls="none", marker="o", ms=3.5, mfc=WHITE, mec=BLACK, mew=0.8)
            c_ax.text(x, max(r["molecule_like_pct_of_passed"], r["expected_pct_of_passed"]) + 0.25,
                      f"{int(r['n_molecule_like'])}/{int(r['n_passed'])}", ha="center", va="bottom", fontsize=5,
                      color=BLACK)
            xs.append(x)
            labels.append(f"{LABELS[c]}\n{ses.split('_')[-1]}")
    c_ax.set_xticks(xs, labels, rotation=40, ha="right")
    c_ax.set(ylabel="Molecule-like rejected\n(% of passed traces)", xlim=(-0.7, xs[-1] + 0.7))
    axis_standards(c_ax)
    _letter(c_ax, "c")

    # d, e) Halo-K117 passed vs molecule-like rejected, same days
    k = mol[(mol["construct"] == "HaloK117") & mol["session"].isin(SHARED_SESSIONS)]
    order = [(s_, g_) for s_ in SHARED_SESSIONS for g_ in ("passed", "molecule-like rejected")]
    ticks = [f"{s_.split('_')[-1]}, {'passed' if g_ == 'passed' else 'rejected'}" for s_, g_ in order]
    colors = [PROTEIN_COLORS["HaloK117"] if g_ == "passed" else BLACK for _, g_ in order]
    for axx, col, ylabel, logy, letter in ((ax[3], "median_photons_per_event", "Photons per event", True, "d"),
                                           (ax[4], "frac_events_1frame", "Single-frame events (fraction)", False, "e")):
        groups = [k.loc[(k["session"] == s_) & (k["group"] == g_), col].dropna().to_numpy() for s_, g_ in order]
        _box_strip(axx, groups, colors, rng)
        axx.set(ylabel=ylabel, xlim=(-0.6, len(order) - 0.4))
        if logy:
            axx.set_yscale("log")
        else:
            axx.set_ylim(0, 1.02)
        axx.set_xticks(range(len(order)), ticks, rotation=40, ha="right")
        axx.set_title("Halo-K117", pad=3)
        axis_standards(axx)
        _letter(axx, letter)

    # f) all constructs, passed vs molecule-like rejected
    f = ax[5]
    groups, colors, ticks, pos = [], [], [], []
    for i, c in enumerate(CONSTRUCTS):
        for j, g_ in enumerate(("passed", "molecule-like rejected")):
            groups.append(mol.loc[(mol["construct"] == c) & (mol["group"] == g_), "median_photons_per_event"]
                          .dropna().to_numpy())
            colors.append(PROTEIN_COLORS[c] if g_ == "passed" else BLACK)
    _box_strip(f, [g[g > 0] for g in groups], colors, rng, s=0.8)
    f.axhline(float(noise.median()), color=BLACK, ls=":", lw=0.8)
    f.set_xticks([2 * i + 0.5 for i in range(len(CONSTRUCTS))], [LABELS[c] for c in CONSTRUCTS], rotation=40,
                 ha="right")
    f.set(yscale="log", ylabel="Photons per event", xlim=(-0.6, len(groups) - 0.4))
    axis_standards(f)
    _letter(f, "f")

    fig.legend(handles=[Line2D([], [], color=PROTEIN_COLORS[c], lw=1.2, label=LABELS[c]) for c in CONSTRUCTS]
               + [Line2D([], [], color=BLACK, lw=1.0, ls="--", label="Background traces (b)"),
                  Line2D([], [], color=BLACK, lw=0.8, ls=":", label="Separation 2 and 3 (b); per-frame noise (f)"),
                  Line2D([], [], ls="none", marker="o", ms=3, color=BLACK,
                         label="Rejected, molecule-like (d-f; passed in construct color)"),
                  Line2D([], [], ls="none", marker="o", ms=3.5, mfc=WHITE, mec=BLACK, mew=0.8,
                         label="Expected from noise (c)")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, handlelength=1.6,
               fontsize=5.5, columnspacing=1.0)

    pdf = out / f"{FIG}.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


# --- Caption and provenance ---

def caption(out: Path) -> str:
    fates = pd.read_csv(out / f"{FIG}_panel_a_fates.csv")
    n_in = fates.groupby("construct")["n_traces"].sum()
    sep = pd.read_csv(out / f"{FIG}_panel_b_separation.csv")
    n_bg = int((sep["traces"] == "background").sum())
    same = pd.read_csv(out / f"{FIG}_panel_c_same_day.csv")
    mol = pd.read_csv(out / f"{FIG}_panels_d-f_molecules.csv")
    nk = mol[(mol["construct"] == "HaloK117") & mol["session"].isin(SHARED_SESSIONS)].groupby(["session", "group"]).size()
    return (
        "Figure R3.3b. Traces removed by the signal-to-noise criterion of the quality filter. a) Fate of every "
        "trace detected inside vesicles (" + ", ".join(f"{LABELS[c]} {int(n_in[c])}" for c in CONSTRUCTS) +
        ") in a replay of the quality filter that reproduces the published selection exactly. Traces with "
        "GMM separation (SNR) below 3 are rejected (the explicit gate at 2 and the frame classifier's minimum of 3); "
        "molecule-like: more frames above the trace's median + 5 sigma (sigma = 1.4826 x MAD) than 99 % of "
        "the background traces of the same session. b) Cumulative distributions of the GMM separation of "
        f"traces inside vesicles and of {n_bg} background traces (random positions at least 5 pixels from any "
        "detected protein, same movies). c) Halo-D106 and Halo-K117 measured on the same days: molecule-like "
        "rejected traces as a percentage of the passed traces; open circles, number expected from noise "
        "(rejected traces x the session's background false-flag rate, 0.9-1.0 %); labels, molecule-like / "
        "passed. d, e) Halo-K117 on the same days, one point per molecule: median detected photons per ON "
        "event (d) and fraction of single-frame ON events (e), passed and molecule-like rejected traces (n = "
        + ", ".join(f"{s.split('_')[-1]} {g} {int(nk[(s, g)])}" for s in SHARED_SESSIONS
                    for g in ("passed", "molecule-like rejected")) + "). f) Median photons per ON event per "
        "molecule, passed (construct color) and molecule-like rejected (black) traces, sessions pooled; dotted "
        "line, median per-frame noise of the passed traces. ON events as classified for Figures 5D and 6C "
        "(without the separation minimum). Boxes, median and quartiles; whiskers, 1.5 x IQR. Molecules are "
        "distinct. No statistical comparisons were performed."
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
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, default=v, help=f"repo-relative run folder (default {v})")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.build:
        repo = Path(__file__).resolve().parents[2]
        inputs = build(repo, args, out)
        (out / f"{FIG}_caption.txt").write_text(caption(out) + "\n")
        if Path(__file__).resolve() != (out / Path(__file__).name).resolve():
            shutil.copy(Path(__file__).resolve(), out / Path(__file__).name)
        write_provenance(out, inputs)
    print(f"Saved: {plot(out)}")


if __name__ == "__main__":
    main()
