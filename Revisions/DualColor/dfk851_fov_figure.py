#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Representative fields of view of DFK851's mixed slide (slide 3) for the paper (user request 2026-11-19): the
two vesicle channels as one dual-color image, and a circle on every protein localization that the paper's IN rule
places on a vesicle (paper_in_mixed.py: within 4 px of a 405 nm or a 488 nm cluster, two-color vesicles included;
localizations fully outside vesicles are not drawn). Model: point 17's P (seven-session pool, six models).

Circles (radius 4 px, the IN distance, centred on the localization in the 640 nm frame):
  grey, dotted       the trace failed the quality filters (never classified)
  coloured, solid    classified and kept by the uncertainty filter (ensemble mean Wasserstein certainty above the
                     mean of the models' validation thresholds, the user's rule of point 13)
  coloured, dashed   classified but removed by the uncertainty filter
  colour             Crameri bam on its inner 75 % on each side (positions 0.125-0.875, the user's choice): the
                     ensemble's p_SNAP; magenta end = HT called with certainty, green end = SNAP, white = undecided.
The vesicles are drawn in the same two hues: ATTO390 (HT vesicles) in the magenta end colour, ATTO520 (SNAP
vesicles) in the green one, so a circle whose colour differs from its vesicle is a wrong call; two-colour
vesicles appear white.

Image: the 405 nm (ATTO390) and 488 nm (ATTO520) snapshots, mean of their 10 frames, background (the image
median) subtracted; bleed-through corrected per pixel with the unmixing run's signatures (ATTO390 in 488 nm,
ATTO520 in 405 nm, flux per unit home-channel flux: the 2 x 2 linear inverse, clipped at 0); each channel shifted
into the 640 nm frame by the trace_assignment run's registration offset (vesicle in 640 = vesicle + offset); one
linear display range per channel for all rendered FOVs (0 to the 99.7th percentile of the candidate FOVs pooled).
Scale bar 5 um (0.13 um per pixel, from the ND2 metadata).

FOV choice (the user's rule): among slide-3 FOVs with at least 5 labelled classified paper-IN traces of which at
least one HT and one SNAP, the distance to the median in (a) the number of classified paper-IN traces and (b) P's
per-FOV accuracy on the labelled ones (single-dye vesicle within 2 px; call at p_SNAP 0.5), each scaled by its
interquartile range; the closest --n FOVs are rendered (candidates.csv lists every FOV with its score).

Outputs (<out>): fov_<FOV>_P.png / .pdf (one per candidate), candidates.png (side by side), candidates.csv,
fov_<FOV>_P_circles.csv (source data: one row per circle), caption.txt, a copy of this script.

    python Revisions/DualColor/dfk851_fov_figure.py [--n 3] [--fovs FOV07 FOV19] [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc import REPO_ROOT, load_config, run_dir  # noqa: E402

DS, SLIDE = "DFK851", "slide3"
CELL_GLOB = "*_pool7_valfov_s*"           # P (point 17)
LO, HI = 0.125, 0.875                     # bam positions for p_SNAP 0 and 1 (inner 75 % per side)
RADIUS_PX, PX_UM, BAR_UM = 4.0, 0.13, 5.0
MIN_LABELLED = 5
LOCBPLOTS = Path(os.environ.get("LOCBPLOTS_MATERIALS", "/Volumes/MediumBerth/Agent/repos/Lab/locbplots/Materials"))
MODELS = REPO_ROOT / "Results/Revisions/DualColor/models"
TRUTH = {"IN_ATTO390": "HT", "IN_ATTO520": "SNAP"}


def p_ensemble() -> tuple[pd.DataFrame, float, int]:
    """P's ensemble on every filtered slide-3 trace: mean p_SNAP, mean certainty; mean threshold; n models."""
    runs = {}
    for d in sorted(MODELS.glob(CELL_GLOB)):
        if (d / "predictions.csv").exists():
            runs[d.name.split("_", 2)[2]] = d                     # latest folder per run name
    parts, thr = [], []
    for name, d in sorted(runs.items()):
        pr = pd.read_csv(d / "predictions.csv", dtype={"uniqueID": str, "extraction_key": str, "fov": str}, low_memory=False)
        pr = pr[(pr["set"] == "test_mixed") & (pr["dataset"] == DS) & (pr["slide"] == SLIDE)]
        parts.append(pr[["extraction_key", "uniqueID", "p_SNAP", "wasserstein"]].assign(run=name))
        thr.append(json.load(open(d / "threshold.json"))["wasserstein_threshold"])
    p = pd.concat(parts)
    e = p.groupby(["extraction_key", "uniqueID"], as_index=False).agg(p_SNAP=("p_SNAP", "mean"),
                                                                       certainty=("wasserstein", "mean"), n=("run", "nunique"))
    if (e["n"] != len(runs)).any():
        raise ValueError("a trace is not predicted by every P model")
    return e.drop(columns="n"), float(np.mean(thr)), len(runs)


def snapshot(path: Path) -> np.ndarray:
    import nd2
    with nd2.ND2File(path) as f:
        a = f.asarray().astype(float)
        vs = f.voxel_size()
    if abs(vs.x - PX_UM) > 1e-6:
        raise ValueError(f"{path}: pixel size {vs.x} um, expected {PX_UM}")
    return a.mean(axis=0) if a.ndim == 3 else a


def dye_images(cfg: dict, frow: pd.Series, k390_488: float, k520_405: float, off: dict) -> dict:
    """Bleed-through-corrected ATTO390 / ATTO520 images in the 640 nm frame."""
    from scipy import ndimage
    img = {}
    for ch in ("405", "488"):
        a = snapshot(Path(cfg["input_root"]) / frow[f"file_{ch}"])
        a = a - np.median(a)
        img[ch] = ndimage.shift(a, off[ch], order=1, mode="constant", cval=0.0)
    det = 1 - k390_488 * k520_405
    return {"ATTO390": np.clip((img["405"] - k520_405 * img["488"]) / det, 0, None),
            "ATTO520": np.clip((img["488"] - k390_488 * img["405"]) / det, 0, None)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--fovs", nargs="*", help="render these FOVs instead of the closest --n")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "Results/Revisions/DualColor/dfk851_b/pool7/figures/fov")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(DS)
    ta = run_dir(cfg, "trace_assignment")
    ta_man = yaml.safe_load(open(ta / "manifest.yaml"))
    fovs = pd.read_csv(REPO_ROOT / ta_man["vesicles_run"] / "fovs.csv", dtype={"fov": str}).query("slide == @SLIDE").set_index("fov")
    tr = pd.read_csv(ta / "traces.csv", dtype={"slide": str, "fov": str, "class": str, "uniqueID": str, "extraction_key": str},
                     low_memory=False).query("slide == @SLIDE")
    pim = run_dir(cfg, "paper_in_mixed")
    pin = pd.read_csv(pim / "paper_in_mixed.csv", dtype={"fov": str, "uniqueID": str, "extraction_key": str})
    reg = pd.read_csv(ta / "registration.csv", dtype={"detected_by": str})
    reg = reg[reg["iteration"] == reg["iteration"].max()].set_index("detected_by")
    off = {ch: (float(reg.loc[ch, "applied_dy"]), float(reg.loc[ch, "applied_dx"])) for ch in ("405", "488")}
    sig = pd.read_csv(run_dir(cfg, "unmixing") / "signatures.csv", dtype={"home": str, "channel": str}).set_index("dye")
    k390_488, k520_405 = float(sig.loc["ATTO390", "k_mean"]), float(sig.loc["ATTO520", "k_mean"])
    assert sig.loc["ATTO390", "channel"] == "488" and sig.loc["ATTO520", "channel"] == "405"
    ens, thr, n_models = p_ensemble()

    # one row per localization on slide 3 that the paper's IN rule puts on a vesicle
    c = tr.merge(pin[["extraction_key", "uniqueID", "paper_in", "in_channels", "d_405", "d_488"]],
                 on=["extraction_key", "uniqueID"], how="left", validate="one_to_one")
    if c["paper_in"].isna().any():
        raise ValueError("ROIs missing from paper_in_mixed.csv")
    c = c[c["paper_in"].astype(bool)].copy()
    c["filtered"] = c["filtered"].astype(str) == "True"
    c = c.merge(ens, on=["extraction_key", "uniqueID"], how="left", validate="one_to_one")
    if c.loc[c["filtered"], "p_SNAP"].isna().any() or c.loc[~c["filtered"], "p_SNAP"].notna().any():
        raise ValueError("predictions do not cover exactly the filtered traces")
    c["truth"] = c["class"].map(TRUTH)
    c["call"] = np.where(c["p_SNAP"] >= 0.5, "SNAP", "HT")
    c.loc[~c["filtered"], "call"] = None
    c["kept"] = c["filtered"] & (c["certainty"] > thr)
    c["correct"] = np.where(c["truth"].notna() & c["filtered"], c["call"] == c["truth"], np.nan)

    # FOV choice
    rows = []
    for fv, g in c.groupby("fov"):
        lab = g[g["filtered"] & g["truth"].notna()]
        rows.append({"fov": fv, "acq_order": int(fovs.loc[fv, "acq_order"]), "circles": len(g),
                     "classified": int(g["filtered"].sum()), "kept": int(g["kept"].sum()), "labelled": len(lab),
                     "labelled_HT": int((lab["truth"] == "HT").sum()), "labelled_SNAP": int((lab["truth"] == "SNAP").sum()),
                     "accuracy": float(lab["correct"].astype(float).mean()) if len(lab) else np.nan})
    fdf = pd.DataFrame(rows)
    elig = fdf[(fdf["labelled"] >= MIN_LABELLED) & (fdf["labelled_HT"] > 0) & (fdf["labelled_SNAP"] > 0)].copy()
    for col in ("classified", "accuracy"):
        q1, med, q3 = elig[col].quantile([0.25, 0.5, 0.75])
        elig[f"{col}_dev"] = (elig[col] - med).abs() / (q3 - q1)
    elig["score"] = elig["classified_dev"] + elig["accuracy_dev"]
    fdf = fdf.merge(elig[["fov", "classified_dev", "accuracy_dev", "score"]], on="fov", how="left").sort_values("score")
    fdf["eligible"] = fdf["score"].notna()
    fdf.to_csv(a.out / "candidates.csv", index=False)
    pick = a.fovs or list(fdf.dropna(subset=["score"])["fov"].head(a.n))
    med = elig[["classified", "accuracy"]].median()
    print(f"{len(elig)} eligible FOVs of {len(fdf)}; medians: {med['classified']:.0f} classified, accuracy {med['accuracy']:.3f}")
    print(fdf.head(max(a.n, len(pick)) + 2).to_string(index=False))

    # images, one display range per channel for all rendered FOVs
    imgs = {fv: dye_images(cfg, fovs.loc[fv], k390_488, k520_405, off) for fv in pick}
    vmax = {d: float(np.percentile(np.concatenate([imgs[fv][d].ravel() for fv in pick]), 99.7)) for d in ("ATTO390", "ATTO520")}

    sys.path.insert(0, str(LOCBPLOTS))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    from scripts.plot_standards import apply_rcparams, save_figure, save_source_data
    apply_rcparams()
    bam = np.loadtxt(LOCBPLOTS / "ScientificColourMaps8" / "bam" / "bam.txt")
    full = ListedColormap(bam)
    cmap = ListedColormap(full(np.linspace(LO, HI, 256)))          # p_SNAP 0..1 -> bam 0.125..0.875
    tint = {"ATTO390": np.array(full(LO)[:3]), "ATTO520": np.array(full(HI)[:3])}
    tint = {k: v / v.max() for k, v in tint.items()}                 # the end hues at full brightness

    def composite(fv):
        rgb = sum(np.clip(imgs[fv][d] / vmax[d], 0, 1)[..., None] * tint[d] for d in tint)
        return np.clip(rgb, 0, 1)

    def draw(ax, fv, lw=1.0):
        ax.imshow(composite(fv), interpolation="nearest")
        g = c[c["fov"] == fv]
        halo = [pe.Stroke(linewidth=lw + 1.0, foreground="black"), pe.Normal()]
        for r in g.itertuples():
            xy = (r.col, r.row)
            if not r.filtered:
                ax.add_patch(Circle(xy, RADIUS_PX, fill=False, ec="0.6", lw=lw * 0.8, ls=(0, (1, 1.5))))
            else:
                ax.add_patch(Circle(xy, RADIUS_PX, fill=False, ec=cmap(r.p_SNAP), lw=lw,
                                    ls="-" if r.kept else (0, (3, 2)), path_effects=halo))
        h, w = composite(fv).shape[:2]
        bar = BAR_UM / PX_UM
        ax.plot([w - 8 - bar, w - 8], [h - 10, h - 10], color="white", lw=2, solid_capstyle="butt")
        ax.text(w - 8 - bar / 2, h - 14, f"{BAR_UM:.0f} µm", color="white", ha="center", va="bottom", fontsize=6)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_axis_off()
        return g

    def legend_and_cbar(fig, cax):
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax)
        cb.set_ticks([0, 0.5, 1])
        cb.set_ticklabels(["HT", "undecided", "SNAP"])
        cb.set_label("Call of the classifier (ensemble p$_{SNAP}$)")
        cb.outline.set_linewidth(0.4)
        return [Line2D([], [], color="0.4", lw=1, label="classified, kept by the uncertainty filter"),
                Line2D([], [], color="0.4", lw=1, ls=(0, (3, 2)), label="classified, removed by the uncertainty filter"),
                Line2D([], [], color="0.6", lw=0.8, ls=(0, (1, 1.5)), label="failed the quality filters"),
                Line2D([], [], ls="none", marker="s", ms=5, color=tint["ATTO390"], label="ATTO390 vesicle (HT)"),
                Line2D([], [], ls="none", marker="s", ms=5, color=tint["ATTO520"], label="ATTO520 vesicle (SNAP)")]

    for fv in pick:
        fig = plt.figure(figsize=(4.6, 4.0))
        ax = fig.add_axes([0.02, 0.14, 0.80, 0.81])
        cax = fig.add_axes([0.85, 0.27, 0.03, 0.55])
        g = draw(ax, fv)
        r = fdf.set_index("fov").loc[fv]
        ax.set_title(f"DFK851 mixed slide, {fv}: {int(r['circles'])} paper-IN localizations, {int(r['classified'])} classified, "
                     f"{int(r['kept'])} kept; accuracy {r['accuracy']:.2f} on {int(r['labelled'])} single-dye", fontsize=6, color="black")
        fig.legend(handles=legend_and_cbar(fig, cax), frameon=False, loc="lower center", ncol=2, fontsize=6,
                   bbox_to_anchor=(0.45, 0.0))
        save_figure(fig, a.out, f"fov_{fv}_P.png", dpi=600)
        plt.close(fig)
        save_source_data(g[["fov", "extraction_key", "uniqueID", "row", "col", "filtered", "in_channels", "class", "truth",
                            "p_SNAP", "certainty", "kept", "call", "correct"]], a.out, f"fov_{fv}_P_circles.csv")
    fig = plt.figure(figsize=(7.0, 3.0))
    n = len(pick)
    for i, fv in enumerate(pick):
        ax = fig.add_axes([0.01 + i * 0.88 / n, 0.15, 0.88 / n - 0.01, 0.76])
        draw(ax, fv, lw=0.7)
        r = fdf.set_index("fov").loc[fv]
        ax.set_title(f"{fv}: {int(r['classified'])} classified, accuracy {r['accuracy']:.2f}", fontsize=6, color="black")
    cax = fig.add_axes([0.91, 0.24, 0.012, 0.6])
    fig.legend(handles=legend_and_cbar(fig, cax), frameon=False, loc="lower center", ncol=3, fontsize=6, bbox_to_anchor=(0.45, -0.02))
    save_figure(fig, a.out, "candidates.png", dpi=450)
    plt.close(fig)
    (a.out / "caption.txt").write_text(
        f"Representative fields of view of the DFK851 mixed slide. Vesicle channels: ATTO390 (HT vesicles, 405 nm "
        f"excitation) in magenta, ATTO520 (SNAP vesicles, 488 nm) in green, two-colour vesicles white; mean of 10 frames, "
        f"background subtracted, bleed-through corrected (ATTO390 into 488 nm {k390_488:.3f}, ATTO520 into 405 nm "
        f"{k520_405:.3f} of the home-channel signal), registered to the 640 nm frame, one linear display range per "
        f"channel for all fields shown. Circles (radius {RADIUS_PX:.0f} px, {RADIUS_PX * PX_UM:.2f} um): protein "
        f"localizations (640 nm) within 4 px of a vesicle in either channel (the paper's IN rule); grey dotted, failed "
        f"the trace quality filters; coloured, classified by the ensemble of {n_models} models trained on seven "
        f"sessions' single-protein slides (point 17's P), colour = ensemble p_SNAP on the inner 75 % of Crameri bam "
        f"(magenta: HT called with certainty; white: undecided; green: SNAP), solid = kept and dashed = removed by the "
        f"uncertainty filter (mean certainty above the mean validation threshold, {thr:.3f}). A circle whose colour differs "
        f"from its vesicle's is a wrong call. Fields chosen as the closest to the slide's median in the number of "
        f"classified paper-IN traces and in the per-field accuracy ({len(elig)} eligible fields, candidates.csv). "
        f"Scale bar {BAR_UM:.0f} um.\n")
    shutil.copy(__file__, a.out)
    print(f"rendered {pick} into {a.out}")


if __name__ == "__main__":
    main()
