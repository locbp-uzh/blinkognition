#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Sensitivity, limits of detection and quantitation of the HT:SNAP ratio on DFK851's mixed slide (slide 3)
(TRAINING_NOTES.md point 15, design fixed before any limit was computed). CLSI EP17 with the composition as
the analyte.

Molecules: the filtered single-dye traces of the 112 FOVs of point 14 (at least 10 single-dye vesicles).
Per molecule p = the ensemble MC-mean p_SNAP (mean of the five runs, point 14); a sample's response T = the
mean p over its n molecules, read from k FOVs. Two-component variance model per class, by the method of
moments (one-way random-effects ANOVA, unbalanced): FOV effects (variances F_HT, F_SNAP, covariance c across
FOVs) and within-FOV variances (w_HT, w_SNAP); class means a' (HT), b' (SNAP) over molecules. For k FOVs with
n/k molecules each, each molecule SNAP with probability s (the material share):
    E[T]   = a' + s d,  d = b' - a'
    Var[T] = [(1-s) w_HT + s w_SNAP + s(1-s) d^2] / n            molecule term (with the binomial composition)
           + [(1-s)^2 F_HT + s^2 F_SNAP + 2 s(1-s) c] / k         FOV term
           + s(1-s) (F_HT + F_SNAP - 2 c) / n                     cross term
Limits, normal theory, 5 % false positives and 5 % false negatives, one-sided:
    LoB = E[T|0] + 1.645 SD[T|0];  LoD = smallest s with E[T|s] - 1.645 SD[T|s] >= LoB   (SNAP in HT)
    HT in SNAP mirrored (blank s = 1), reported as an HT share
    minimum detectable difference at s0 (0.10, 0.50): power 0.80, two-sided alpha 0.05
    quantitation: bias and SD of the reported (clipped to [0, 1]) share, uncalibrated (pure-slide validation
    constants a, b of point 14) and calibrated (a', b' of the slide's own labeled traces, plus the calibrator's
    sampling error); total error |bias| + 2 SD <= 0.10 between the two LoDs; the LoQ is its lowest share.
Detection uses the unclipped T, so detection limits do not depend on the constants (a monotone map of T).

Uncertainty: FOV-cluster bootstrap (FOVs resampled with replacement; components re-estimated). Checks:
real whole-FOV samples against the model (z-scores; the pass criterion of point 15), the same with the model
fitted on half the FOVs, a brute-force simulation of the formula, and the constants check (pure-slide,
mixed-slide and per-sample constants on the same k = 6 samples).

Outputs (<out>): components.csv, grid.csv, diagonal_diff.csv, quant_curve.csv, model_check.csv,
model_check_split.csv, formula_check.csv, constants_check.csv; figures/ (locbplots standards: PNG + PDF, one
source-data CSV per panel, caption.txt, a copy of this script).

    python Revisions/DualColor/dfk851_lod.py [--passes <dir>] [--out <dir>] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dfk851_fov_ratio import DFK, KEY, LOCBPLOTS, MIN_VESICLES, RUNS, ensemble, load_run, truth_table  # noqa: E402

MODELS = ("B", "C")
NS = (16, 50, 100, 200, 500)
KS = (1, 3, 6, 12, 24)
DIAG = ((16, 1), (50, 3), (100, 6), (200, 12), (500, 24))       # about 16 molecules per FOV, as on the slide
Z = float(stats.norm.ppf(0.95))                                  # 1.645
Z_MDD = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.80))      # 2.80
GOAL = 0.10
S0_MDD = (0.10, 0.50)
S_GRID = np.round(np.linspace(0, 1, 201), 3)                     # step 0.005
N_CHECK, THIN, TAIL_OK = 5000, 0.25, (0.025, 0.075)
N_SPLIT, N_SPLIT_SAMPLES, SPLIT_KS = 50, 500, (1, 3, 6, 12)
TYPES = ("blank_HT", "blank_SNAP", "natural", "low_SNAP", "low_HT")
FORMULA_CELLS = ((16, 1), (48, 3), (96, 6), (192, 12), (480, 24))  # integer molecules per FOV
FORMULA_S, N_FORMULA = (0.0, 0.1, 0.5, 0.9, 1.0), 200_000
SEED = 20261118


# --- data ----------------------------------------------------------------------------------------------------

def load(passes: Path) -> dict:
    truth = truth_table()
    ens = {c: ensemble([load_run(c, n, passes) for n in names]) for c, names in RUNS.items()}
    if not (ens["B"]["keys"] == ens["C"]["keys"]).all().all():
        raise ValueError("B and C passes cover different traces")
    t = ens["B"]["keys"].merge(truth, on=KEY, how="left", validate="one_to_one")
    lab = t["y"].notna().to_numpy()
    ves = t[lab].groupby("fov")["vesicle_id"].nunique()
    inc = sorted(ves.index[ves >= MIN_VESICLES])
    m = lab & t["fov"].isin(inc).to_numpy()
    fov = pd.Categorical(t.loc[m, "fov"], categories=inc).codes.astype(int)
    cls = t.loc[m, "y"].to_numpy().astype(int)
    D = {"fov": fov, "cls": cls, "P": np.column_stack([ens[c]["p"][m] for c in MODELS]), "J": len(inc),
         "fovs": inc, "ab": {c: (float(ens[c]["a"]), float(ens[c]["b"])) for c in MODELS}}
    tot = np.bincount(fov, minlength=len(inc))
    facts = (len(inc), int((cls == 0).sum()), int((cls == 1).sum()), int(tot.min()), int(tot.max()))
    if facts != (112, 989, 839, 10, 26):   # the design facts point 15 was written on
        raise ValueError(f"data differ from point 15's design facts: {facts}")
    D["density"] = (int(tot.min()), int(tot.max()))
    D["members"] = [[np.flatnonzero((fov == j) & (cls == c)) for c in (0, 1)] for j in range(len(inc))]
    return D


def sufficient(D: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per FOV x class: count; per FOV x class x model: sum and sum of squares of p."""
    J = D["J"]
    cnt, S, Q = np.zeros((J, 2)), np.zeros((J, 2, len(MODELS))), np.zeros((J, 2, len(MODELS)))
    for c in (0, 1):
        sel = D["cls"] == c
        cnt[:, c] = np.bincount(D["fov"][sel], minlength=J)
        for mi in range(len(MODELS)):
            S[:, c, mi] = np.bincount(D["fov"][sel], weights=D["P"][sel, mi], minlength=J)
            Q[:, c, mi] = np.bincount(D["fov"][sel], weights=D["P"][sel, mi] ** 2, minlength=J)
    return cnt, S, Q


# --- variance model --------------------------------------------------------------------------------------------

def components(cnt: np.ndarray, S: np.ndarray, Q: np.ndarray, mi: int) -> dict:
    """Method of moments on the FOVs given (rows; a FOV drawn twice is two FOVs)."""
    out = {"mu": np.zeros(2), "w": np.zeros(2), "F": np.zeros(2), "F_raw": np.zeros(2)}
    for c in (0, 1):
        has = cnt[:, c] > 0
        m, s, q = cnt[has, c], S[has, c, mi], Q[has, c, mi]
        N, J = m.sum(), len(m)
        mu = s.sum() / N
        msw = (q - s ** 2 / m).sum() / (N - J)
        msb = (m * (s / m - mu) ** 2).sum() / (J - 1)
        n0 = (N - (m ** 2).sum() / N) / (J - 1)
        out["mu"][c], out["w"][c], out["F_raw"][c] = mu, msw, (msb - msw) / n0
    out["F"] = np.maximum(out["F_raw"], 0.0)
    both = (cnt[:, 0] > 0) & (cnt[:, 1] > 0)
    out["c_raw"] = float(np.cov(S[both, 0, mi] / cnt[both, 0], S[both, 1, mi] / cnt[both, 1], ddof=1)[0, 1])
    lim = float(np.sqrt(out["F"][0] * out["F"][1]))
    out["c"] = float(np.clip(out["c_raw"], -lim, lim))
    # the calibrator = these FOVs' labeled traces: sampling variances of a', b' and their covariance
    NH, NS = cnt[:, 0].sum(), cnt[:, 1].sum()
    out["var_a"] = out["w"][0] / NH + out["F"][0] * (cnt[:, 0] ** 2).sum() / NH ** 2
    out["var_b"] = out["w"][1] / NS + out["F"][1] * (cnt[:, 1] ** 2).sum() / NS ** 2
    out["cov_ab"] = out["c"] * (cnt[:, 0] * cnt[:, 1]).sum() / (NH * NS)
    return out


def var_T(cp: dict, s, n: float, k: float):
    s = np.asarray(s, float)
    d = cp["mu"][1] - cp["mu"][0]
    w, F, c = cp["w"], cp["F"], cp["c"]
    return (((1 - s) * w[0] + s * w[1] + s * (1 - s) * d ** 2) / n
            + ((1 - s) ** 2 * F[0] + s ** 2 * F[1] + 2 * s * (1 - s) * c) / k
            + s * (1 - s) * (F[0] + F[1] - 2 * c) / n)


def first_root(f, lo: float = 0.0, hi: float = 1.0) -> float:
    """Smallest x in [lo, hi] with f(x) >= 0 (f vectorized); nan if none."""
    x = np.linspace(lo, hi, 2001)
    i = np.flatnonzero(f(x) >= 0)
    if not len(i):
        return np.nan
    if i[0] == 0:
        return float(x[0])
    return float(optimize.brentq(lambda v: float(f(np.array(v))), x[i[0] - 1], x[i[0]], xtol=1e-7))


def detection(cp: dict, n: float, k: float) -> dict:
    a1, d = cp["mu"][0], cp["mu"][1] - cp["mu"][0]
    mean = lambda s: a1 + s * d                                  # noqa: E731
    sd = lambda s: np.sqrt(var_T(cp, s, n, k))                    # noqa: E731
    lob = mean(0.0) + Z * sd(0.0)
    lob_h = mean(1.0) - Z * sd(1.0)
    out = {"lobT_snap": float(lob), "lod_snap": first_root(lambda s: mean(s) - Z * sd(s) - lob),
           "lobT_ht": float(lob_h), "lod_ht": first_root(lambda h: lob_h - mean(1 - h) - Z * sd(1 - h))}
    for s0 in S0_MDD:
        v0 = var_T(cp, s0, n, k)
        out[f"mdd_{s0:.2f}"] = first_root(lambda x: x * d - Z_MDD * np.sqrt(v0 + var_T(cp, s0 + x, n, k)), 0.0, 1.0 - s0)
    return out


def clipped_moments(mu, sig) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD of clip(X, 0, 1), X ~ N(mu, sig^2)."""
    mu, sig = np.broadcast_arrays(np.asarray(mu, float), np.asarray(sig, float))
    al, be = -mu / sig, (1 - mu) / sig
    Pa, Pb, pa, pb = stats.norm.cdf(al), stats.norm.cdf(be), stats.norm.pdf(al), stats.norm.pdf(be)
    mid = Pb - Pa
    e1 = mu * mid + sig * (pa - pb)
    e2 = (mu ** 2 + sig ** 2) * mid + 2 * mu * sig * (pa - pb) + sig ** 2 * (al * pa - be * pb)
    m1, m2 = e1 + (1 - Pb), e2 + (1 - Pb)
    return m1, np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))


def quantitation(cp: dict, ab: tuple[float, float], n: float, k: float) -> dict:
    a, b = ab
    a1, d = cp["mu"][0], cp["mu"][1] - cp["mu"][0]
    s = S_GRID
    vT = var_T(cp, s, n, k)
    var_cal = ((1 - s) ** 2 * cp["var_a"] + s ** 2 * cp["var_b"] + 2 * s * (1 - s) * cp["cov_ab"]) / d ** 2
    out = {}
    for ruler, (mu, sig) in {"uncal": ((a1 + s * d - a) / (b - a), np.sqrt(vT) / (b - a)),
                             "cal": (s, np.sqrt(vT / d ** 2 + var_cal))}.items():
        m1, sd = clipped_moments(mu, sig)
        out[ruler] = {"mean": m1, "sd": sd, "bias": m1 - s, "te": np.abs(m1 - s) + 2 * sd}
    return out


def q_range(te: np.ndarray, lod_snap: float, lod_ht: float) -> tuple[float, float, bool]:
    ok = (te <= GOAL) & (S_GRID >= lod_snap) & (S_GRID <= 1 - lod_ht)   # nan LoD -> nothing quantifiable
    i = np.flatnonzero(ok)
    if not len(i):
        return np.nan, np.nan, False
    return float(S_GRID[i[0]]), float(1 - S_GRID[i[-1]]), bool((np.diff(i) == 1).all())


def cell(cp: dict, ab: tuple[float, float], n: int, k: int, curves: bool = False) -> dict:
    a, b = ab
    a1, d = cp["mu"][0], cp["mu"][1] - cp["mu"][0]
    r = detection(cp, n, k)
    r.update({"lob_snap_uncal": (r["lobT_snap"] - a) / (b - a), "lob_snap_cal": (r["lobT_snap"] - a1) / d,
              "lob_ht_uncal": 1 - (r["lobT_ht"] - a) / (b - a), "lob_ht_cal": 1 - (r["lobT_ht"] - a1) / d})
    q = quantitation(cp, ab, n, k)
    for ruler, cur in q.items():
        lo, hi, contig = q_range(cur["te"], r["lod_snap"], r["lod_ht"])
        det = (S_GRID >= r["lod_snap"]) & (S_GRID <= 1 - r["lod_ht"])   # within the detectable range only:
        r.update({f"loq_snap_{ruler}": lo, f"loq_ht_{ruler}": hi, f"loq_contiguous_{ruler}": contig,   # beyond it the
                  f"te_min_{ruler}": float(cur["te"][det].min()) if det.any() else np.nan})       # clip at 0/1 fakes a small TE
        for s in (0.10, 0.50, 0.90):   # descriptive: how far from the goal
            r[f"te_{ruler}_at_{s:.2f}"] = float(cur["te"][np.argmin(np.abs(S_GRID - s))])
    if curves:
        r["curves"] = q
    return r


# --- checks on real whole-FOV samples ----------------------------------------------------------------------

def draw_samples(D: dict, pool: np.ndarray, k: int, n_samp: int, rng: np.random.Generator) -> dict:
    """Per type: list of (per-copy HT counts, per-copy SNAP counts, sum of p per model); FOVs with replacement."""
    P, mem = D["P"], D["members"]
    out = {t: [] for t in TYPES}
    chosen = []
    for _ in range(n_samp):
        ch = rng.choice(pool, k, replace=True)
        chosen.append(ch)
        acc = {t: ([], [], np.zeros(P.shape[1])) for t in TYPES}
        for j in ch:
            H, Sn = mem[j]
            kS, kH = rng.random(len(Sn)) < THIN, rng.random(len(H)) < THIN   # a FOV drawn twice: thinned twice
            parts = {"blank_HT": (H, Sn[:0]), "blank_SNAP": (H[:0], Sn), "natural": (H, Sn),
                     "low_SNAP": (H, Sn[kS]), "low_HT": (H[kH], Sn)}
            for t, (h, s_) in parts.items():
                acc[t][0].append(len(h))
                acc[t][1].append(len(s_))
                acc[t][2][:] += P[h].sum(axis=0) + P[s_].sum(axis=0)
        for t in TYPES:
            out[t].append((np.array(acc[t][0]), np.array(acc[t][1]), acc[t][2].copy()))
    return {"samples": out, "chosen": chosen}


def zscores(samples: list, cp: dict, mi: int) -> np.ndarray:
    z = np.empty(len(samples))
    for i, (nh, ns, sp) in enumerate(samples):
        n = nh.sum() + ns.sum()
        e = (nh.sum() * cp["mu"][0] + ns.sum() * cp["mu"][1]) / n
        v = ((nh.sum() * cp["w"][0] + ns.sum() * cp["w"][1])
             + (nh ** 2 * cp["F"][0] + ns ** 2 * cp["F"][1] + 2 * nh * ns * cp["c"]).sum()) / n ** 2
        z[i] = (sp[mi] / n - e) / np.sqrt(v)
    return z


def tails(z: np.ndarray) -> dict:
    hi, lo = float((z > Z).mean()), float((z < -Z).mean())
    return {"mean_z": float(z.mean()), "sd_z": float(z.std(ddof=1)), "frac_above": hi, "frac_below": lo,
            "pass": TAIL_OK[0] <= hi <= TAIL_OK[1] and TAIL_OK[0] <= lo <= TAIL_OK[1]}


def model_check(D, cps, rng) -> pd.DataFrame:
    rows = []
    pool = np.arange(D["J"])
    for k in KS:
        smp = draw_samples(D, pool, k, N_CHECK, rng)["samples"]
        for mi, m in enumerate(MODELS):
            for t in TYPES:
                rows.append({"model": m, "k": k, "type": t, "n_samples": N_CHECK,
                             "median_n": float(np.median([a.sum() + b.sum() for a, b, _ in smp[t]])),
                             **tails(zscores(smp[t], cps[m], mi))})
    return pd.DataFrame(rows)


def split_check(D, cnt, S, Q, rng) -> pd.DataFrame:
    rows = []
    J = D["J"]
    for sp in range(N_SPLIT):
        perm = rng.permutation(J)
        fit, test = np.sort(perm[: J // 2]), np.sort(perm[J // 2:])
        cps = {m: components(cnt[fit], S[fit], Q[fit], mi) for mi, m in enumerate(MODELS)}
        for k in SPLIT_KS:
            smp = draw_samples(D, test, k, N_SPLIT_SAMPLES, rng)["samples"]
            for mi, m in enumerate(MODELS):
                for t in TYPES:
                    rows.append({"split": sp, "model": m, "k": k, "type": t, **tails(zscores(smp[t], cps[m], mi))})
    r = pd.DataFrame(rows)
    return (r.groupby(["model", "k", "type"])
             .agg(mean_frac_above=("frac_above", "mean"), q05_above=("frac_above", lambda x: x.quantile(0.05)),
                  q95_above=("frac_above", lambda x: x.quantile(0.95)), mean_frac_below=("frac_below", "mean"),
                  q05_below=("frac_below", lambda x: x.quantile(0.05)), q95_below=("frac_below", lambda x: x.quantile(0.95)),
                  mean_sd_z=("sd_z", "mean"), mean_mean_z=("mean_z", "mean"), splits_passing=("pass", "sum"))
             .reset_index())


def formula_check(cps: dict, rng) -> pd.DataFrame:
    """Brute force from the fitted model (normal FOV effects and errors, binomial classes) vs the formula."""
    rows = []
    for m, cp in cps.items():
        a1, b1 = cp["mu"]
        L = np.linalg.cholesky(np.array([[cp["F"][0], cp["c"]], [cp["c"], cp["F"][1]]]) + 1e-15 * np.eye(2))
        for n, k in FORMULA_CELLS:
            nf = n // k
            det = detection(cp, n, k)
            sims = {}
            for s in sorted(set(FORMULA_S) | ({round(det["lod_snap"], 6)} if np.isfinite(det["lod_snap"]) else set())
                            | ({round(1 - det["lod_ht"], 6)} if np.isfinite(det["lod_ht"]) else set())):
                u = rng.standard_normal((N_FORMULA, k, 2)) @ L.T
                ns = rng.binomial(nf, s, (N_FORMULA, k)).astype(float)
                nh = nf - ns
                e = (rng.standard_normal((N_FORMULA, k)) * np.sqrt(nh * cp["w"][0])
                     + rng.standard_normal((N_FORMULA, k)) * np.sqrt(ns * cp["w"][1]))
                T = (nh * (a1 + u[..., 0]) + ns * (b1 + u[..., 1]) + e).sum(axis=1) / n
                sims[s] = T
                rows.append({"model": m, "n": n, "k": k, "s": s, "var_formula": float(var_T(cp, s, n, k)),
                             "var_sim": float(T.var(ddof=1)), "mean_formula": a1 + s * (b1 - a1), "mean_sim": float(T.mean())})
            lob_sim = float(np.quantile(sims[0.0], 0.95))
            lob_h_sim = float(np.quantile(sims[1.0], 0.05))
            rows.append({"model": m, "n": n, "k": k, "s": "LoB_snap", "formula": det["lobT_snap"], "sim": lob_sim})
            rows.append({"model": m, "n": n, "k": k, "s": "LoB_ht", "formula": det["lobT_ht"], "sim": lob_h_sim})
            if np.isfinite(det["lod_snap"]):
                rows.append({"model": m, "n": n, "k": k, "s": f"detected_at_LoD_snap={det['lod_snap']:.4f}",
                             "formula": 0.95, "sim": float((sims[round(det["lod_snap"], 6)] > det["lobT_snap"]).mean())})
            if np.isfinite(det["lod_ht"]):
                rows.append({"model": m, "n": n, "k": k, "s": f"detected_at_LoD_ht={det['lod_ht']:.4f}",
                             "formula": 0.95, "sim": float((sims[round(1 - det["lod_ht"], 6)] < det["lobT_ht"]).mean())})
    return pd.DataFrame(rows)


def constants_check(D, cnt, S, ab, rng, k: int = 6) -> pd.DataFrame:
    """Empirical LoB and detection rates on the same whole-FOV samples with three sets of constants."""
    drawn = draw_samples(D, np.arange(D["J"]), k, N_CHECK, rng)
    smp, chosen = drawn["samples"], drawn["chosen"]
    rows = []
    for mi, m in enumerate(MODELS):
        a, b = ab[m]
        a1, b1 = S[:, 0, mi].sum() / cnt[:, 0].sum(), S[:, 1, mi].sum() / cnt[:, 1].sum()
        per = []
        for ch in chosen:   # per-sample calibration: the slide's other FOVs
            keep = np.ones(D["J"], bool)
            keep[np.unique(ch)] = False
            per.append((S[keep, 0, mi].sum() / cnt[keep, 0].sum(), S[keep, 1, mi].sum() / cnt[keep, 1].sum()))
        per = np.array(per)
        T = {t: np.array([sp[mi] / (nh.sum() + ns.sum()) for nh, ns, sp in smp[t]]) for t in TYPES}
        share = {t: np.array([ns.sum() / (nh.sum() + ns.sum()) for nh, ns, _ in smp[t]]) for t in TYPES}
        rulers = {"pure_slide": (np.full(N_CHECK, a), np.full(N_CHECK, b)),
                  "mixed_slide": (np.full(N_CHECK, a1), np.full(N_CHECK, b1)),
                  "per_sample": (per[:, 0], per[:, 1])}
        for name, (ra, rb) in rulers.items():
            est = {t: (T[t] - ra) / (rb - ra) for t in TYPES}
            lob = float(np.quantile(est["blank_HT"], 0.95))
            lob_h = float(np.quantile(est["blank_SNAP"], 0.05))
            rows.append({"model": m, "k": k, "constants": name, "lob_snap_share": lob, "lob_ht_share": 1 - lob_h,
                         "detected_low_SNAP": float((est["low_SNAP"] > lob).mean()),
                         "detected_natural": float((est["natural"] > lob).mean()),
                         "detected_low_HT": float((est["low_HT"] < lob_h).mean()),
                         "mean_true_share_low_SNAP": float(share["low_SNAP"].mean()),
                         "mean_true_share_low_HT": float(share["low_HT"].mean())})
    return pd.DataFrame(rows)


# --- bootstrap -----------------------------------------------------------------------------------------------

SCALARS = ("lob_snap_uncal", "lob_snap_cal", "lod_snap", "lob_ht_uncal", "lob_ht_cal", "lod_ht", "mdd_0.10",
           "mdd_0.50", "loq_snap_uncal", "loq_ht_uncal", "loq_snap_cal", "loq_ht_cal")
COMP = ("a_mixed", "b_mixed", "separation", "w_HT", "w_SNAP", "F_HT", "F_SNAP", "c", "rho", "c_raw", "rho_raw",
        "icc_HT", "icc_SNAP")


def comp_row(cp: dict) -> dict:
    F, w = cp["F"], cp["w"]
    pos = F[0] > 0 and F[1] > 0
    return {"a_mixed": cp["mu"][0], "b_mixed": cp["mu"][1], "separation": cp["mu"][1] - cp["mu"][0],
            "w_HT": w[0], "w_SNAP": w[1], "F_HT": F[0], "F_SNAP": F[1], "c": cp["c"],
            "rho": cp["c"] / np.sqrt(F[0] * F[1]) if pos else np.nan, "c_raw": cp["c_raw"],
            "rho_raw": cp["c_raw"] / np.sqrt(F[0] * F[1]) if pos else np.nan,   # before clipping to [-1, 1]
            "icc_HT": F[0] / (F[0] + w[0]), "icc_SNAP": F[1] / (F[1] + w[1])}


def ci(x: np.ndarray) -> tuple[float, float]:
    """95 % percentile interval; nan (limit not reached within [0, 1]) counts as +inf."""
    x = np.where(np.isnan(np.asarray(x, float)), np.inf, x)
    lo, hi = np.quantile(x, [0.025, 0.975], method="inverted_cdf")
    return float(lo), float(hi)


def bootstrap(D, cnt, S, Q, n_boot: int, rng) -> tuple[dict, dict]:
    cells = [(n, k) for n in NS for k in KS if n >= k]
    res = {m: {(n, k): {s: np.empty(n_boot) for s in SCALARS} for n, k in cells} for m in MODELS}
    comp = {m: {c: np.empty(n_boot) for c in COMP} for m in MODELS}
    for i in range(n_boot):
        idx = rng.integers(0, D["J"], D["J"])
        for mi, m in enumerate(MODELS):
            cp = components(cnt[idx], S[idx], Q[idx], mi)
            for c, v in comp_row(cp).items():
                comp[m][c][i] = v
            for n, k in cells:
                r = cell(cp, D["ab"][m], n, k)
                for s in SCALARS:
                    res[m][(n, k)][s][i] = r[s]
        if (i + 1) % 200 == 0:
            print(f"  bootstrap {i + 1}/{n_boot}", flush=True)
    return res, comp


# --- figures (locbplots standards) -------------------------------------------------------------------------

def figures(grid: pd.DataFrame, curves: pd.DataFrame, check: pd.DataFrame, out: Path) -> None:
    sys.path.insert(0, str(LOCBPLOTS))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from scripts.plot_standards import COLORS, apply_axis_standards, apply_rcparams, save_figure, save_source_data
    apply_rcparams()
    batlow = ListedColormap(np.loadtxt(LOCBPLOTS / "ScientificColourMaps8" / "batlow" / "batlow.txt"))
    kcol = {k: batlow(x) for k, x in zip(KS, np.linspace(0.05, 0.8, len(KS)))}
    lab = {"B": "B, trained on DFK851 alone", "C": "C, trained on six other sessions"}
    fdir = out / "figures"

    # limits of detection and minimum detectable difference vs molecules, one line per number of FOVs
    rows_def = (("lod_snap", "Limit of detection, SNAP in HT\n(SNAP share)"),
                ("lod_ht", "Limit of detection, HT in SNAP\n(HT share)"),
                ("mdd_0.50", "Minimum detectable difference\nat 0.50 (share)"))
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.6), sharex=True, sharey="row")
    letters = iter("abcdef")
    for r, (col, ylab) in enumerate(rows_def):
        for cidx, m in enumerate(MODELS):
            ax, letter = axes[r, cidx], next(letters)
            g = grid[grid["model"] == m]
            for k in KS:
                gk = g[g["k"] == k].sort_values("n")
                y, lo, hi = gk[col].to_numpy(), gk[f"{col}_lo"].to_numpy(), np.minimum(gk[f"{col}_hi"].to_numpy(), 1.0)
                x = gk["n"].to_numpy() * (1 + 0.035 * (KS.index(k) - 2))   # small dodge
                ax.plot(x, y, color=kcol[k], lw=1.0, ls="-" if gk["validated"].all() else "--")
                ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=kcol[k], elinewidth=0.6, capsize=0)
                ext = gk["density_extrapolated"].to_numpy()
                ax.scatter(x[~ext], y[~ext], s=14, color=kcol[k], zorder=3, edgecolor="none")
                ax.scatter(x[ext], y[ext], s=14, facecolor="white", edgecolor=kcol[k], linewidth=0.8, zorder=3)
            d = g.set_index(["n", "k"]).loc[list(DIAG)]
            ax.plot([n for n, _ in DIAG], d[col].to_numpy(), color=COLORS["black"], lw=0.6, ls=":", zorder=1)
            ax.set_xscale("log")
            ax.set_xticks(NS)
            ax.set_xticklabels([str(n) for n in NS])
            ax.minorticks_off()
            ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(lab[m], color="black")
            if r == 2:
                ax.set_xlabel("Molecules read (traces)")
            if cidx == 0:
                ax.set_ylabel(ylab)
            ax.text(-0.2 if cidx == 0 else -0.08, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
            apply_axis_standards(ax)
            save_source_data(g[["model", "n", "k", "n_per_fov", "density_extrapolated", "validated", col,
                                f"{col}_lo", f"{col}_hi"]], fdir, f"lod_grid_panel_{letter}.csv")
    handles = [Line2D([], [], color=kcol[k], lw=1.0, marker="o", ms=3, label=f"{k} FOV" + ("s" if k > 1 else ""))
               for k in KS]
    handles += [Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec=COLORS["black"],
                       label="molecules per FOV outside the slide's range (model extrapolation)"),
                Line2D([], [], color=COLORS["black"], lw=0.6, ls=":", label="about 16 molecules per FOV, as on the slide"),
                Line2D([], [], color=COLORS["black"], lw=1.0, ls="--", label="model check failed at this number of FOVs")]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, -0.05))
    save_figure(fig, fdir, "lod_grid.png")
    plt.close(fig)

    # quantitation: reported share vs true share, mean +- 2 SD, against the +-0.10 goal
    designs = (DIAG[0], DIAG[2], DIAG[4])
    rcol = {"uncal": COLORS["vermillion"], "cal": COLORS["blue"]}
    rlab = {"uncal": "pure-slide constants (as now)", "cal": "calibrated on the mixed slide"}
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.9), sharex=True, sharey=True)
    letters = iter("abcdef")
    for r, m in enumerate(MODELS):
        for cidx, (n, k) in enumerate(designs):
            ax, letter = axes[r, cidx], next(letters)
            for off in (-GOAL, GOAL):
                ax.plot([0, 1], [off, 1 + off], ls=":", lw=0.8, color=COLORS["black"])
            ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=COLORS["black"])
            g = curves[(curves["model"] == m) & (curves["n"] == n) & (curves["k"] == k)]
            gr = grid[(grid["model"] == m) & (grid["n"] == n) & (grid["k"] == k)].iloc[0]
            txt = []
            for ruler in ("uncal", "cal"):
                h = g[g["ruler"] == ruler]
                for sign in (-1, 1):   # band edges as thin lines: two overlapping fills would blend into grey
                    ax.plot(h["s"], h["mean"] + sign * 2 * h["sd"], color=rcol[ruler], lw=0.6, alpha=0.8)
                ax.plot(h["s"], h["mean"], color=rcol[ruler], lw=1.3)
                lo, hi = gr[f"loq_snap_{ruler}"], gr[f"loq_ht_{ruler}"]
                txt.append("none" if not np.isfinite(lo) else f"{lo:.2f}-{1 - hi:.2f}")
            ax.set_title(f"{n} molecules, {k} FOV{'s' if k > 1 else ''}\nquantifiable (SNAP share): "
                         f"as now {txt[0]}; calibrated {txt[1]}", color="black", fontsize=6)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.02, 1.02)
            if r == 1:
                ax.set_xlabel("True SNAP share (fraction)")
            if cidx == 0:
                ax.set_ylabel(f"{m}: reported SNAP share (fraction)")
            ax.text(-0.25 if cidx == 0 else -0.08, 1.12, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
            apply_axis_standards(ax)
            save_source_data(g, fdir, f"quantitation_panel_{letter}.csv")
    handles = [Line2D([], [], color=rcol[r], lw=1.3, label=f"{rlab[r]}, mean (thick) ± 2 SD (thin)") for r in ("uncal", "cal")]
    handles += [Line2D([], [], ls=":", color=COLORS["black"], lw=0.8, label=f"total-error goal, identity ± {GOAL:.2f}"),
                Line2D([], [], ls="--", color=COLORS["black"], lw=0.8, label="identity")]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, -0.06))
    save_figure(fig, fdir, "quantitation.png")
    plt.close(fig)

    # model check: tail fractions of real whole-FOV samples under the model
    tcol = dict(zip(TYPES, (COLORS["orange"], COLORS["sky_blue"], COLORS["green"], COLORS["vermillion"], COLORS["pink"])))
    tlab = {"blank_HT": "all HT (blank)", "blank_SNAP": "all SNAP (blank)", "natural": "natural mix",
            "low_SNAP": "low SNAP share", "low_HT": "low HT share"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, m, letter in zip(axes, MODELS, "ab"):
        ax.axhspan(*TAIL_OK, color=COLORS["black"], alpha=0.08, lw=0)
        ax.axhline(0.05, ls="--", lw=0.8, color=COLORS["black"])
        g = check[check["model"] == m]
        for i, t in enumerate(TYPES):
            h = g[g["type"] == t].sort_values("k")
            x = h["k"].to_numpy() * (1 + 0.04 * (i - 2))
            ax.plot(x, h["frac_above"], color=tcol[t], lw=0.8, marker="o", ms=3)
            ax.plot(x, h["frac_below"], color=tcol[t], lw=0.8, ls=":", marker="o", ms=3, mfc="white")
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.minorticks_off()
        ax.set_xlabel("FOVs per sample")
        if m == "B":
            ax.set_ylabel("Fraction of samples beyond\nthe model's 95 % bound")
        ax.set_title(lab[m], color="black")
        ax.text(-0.2 if m == "B" else -0.08, 1.08, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
        apply_axis_standards(ax)
        save_source_data(g, fdir, f"model_check_panel_{letter}.csv")
    handles = [Line2D([], [], color=tcol[t], lw=0.8, marker="o", ms=3, label=tlab[t]) for t in TYPES]
    handles += [Line2D([], [], color=COLORS["black"], lw=0.8, marker="o", ms=3, label="above upper bound"),
                Line2D([], [], color=COLORS["black"], lw=0.8, ls=":", marker="o", ms=3, mfc="white", label="below lower bound"),
                Line2D([], [], color=COLORS["black"], lw=6, alpha=0.08, label=f"pass range {TAIL_OK[0]}-{TAIL_OK[1]}")]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, -0.12))
    save_figure(fig, fdir, "model_check.png")
    plt.close(fig)

    (fdir / "caption.txt").write_text(
        "Limits of detection of the HT:SNAP composition on the DFK851 mixed slide. (a-d) Limit of detection "
        "(5 % false positives against a blank of the same design, 5 % false negatives; CLSI EP17, normal theory) "
        "of SNAP in HT (a, b) and of HT in SNAP (c, d) versus the number of molecules (single-dye traces) read, "
        "one line per number of fields of view (FOVs) they come from, for models trained on DFK851 alone (left) "
        "or on six other sessions (right). (e, f) Minimum detectable difference from a share of 0.50 between two "
        "samples of the same design (power 0.80, two-sided alpha 0.05). Values from a two-component variance "
        "model (FOV and within-FOV variance of the model's SNAP probability, per class) fitted to n = 1828 "
        "traces in n = 112 FOVs of one slide; bars, 95 % FOV-cluster bootstrap intervals (2000 resamples), "
        "upper ends above 1 drawn at 1; open symbols, designs with more or fewer molecules per FOV than the "
        "slide's FOVs hold (10-26); dotted line, designs with about 16 molecules per FOV; dashed lines, numbers "
        "of FOVs at which the model failed its check on real samples (its limits are optimistic there). Missing "
        "points: not detectable within the full range of shares. Within-slide limits: blank, samples and "
        "calibrator share one slide.\n\n"
        "Quantitation of the composition. Reported SNAP share (mean, thick line; ± 2 SD, thin lines) versus the true share "
        "for three designs, with the pure-slide constants used now and calibrated on the mixed slide's own "
        "labeled traces (calibrator sampling error included); dotted lines, total-error goal (|bias| + 2 SD <= "
        "0.10: a share is quantifiable where both thin lines lie between them, within the detection limits); "
        "dashed, identity. Same model and data as above; analytic, for a share clipped to [0, 1].\n\n"
        "Check of the variance model on real samples. Fraction of samples of k whole FOVs (drawn with "
        f"replacement from the 112; {N_CHECK} per k and type) whose mean SNAP probability lies above (filled) or "
        "below (open) the model's one-sided 95 % bound given the sample's per-FOV class counts; grey, the pass "
        f"range fixed before the check ({TAIL_OK[0]}-{TAIL_OK[1]}); dashed, the nominal 0.05.\n")
    shutil.copy(__file__, fdir)


# --- main --------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passes", type=Path, default=DFK / "mc_passes")
    ap.add_argument("--out", type=Path, default=DFK / "lod")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = {name: np.random.default_rng([SEED, i]) for i, name in
           enumerate(("check", "split", "formula", "constants", "boot"))}

    D = load(a.passes)
    cnt, S, Q = sufficient(D)
    cps = {m: components(cnt, S, Q, mi) for mi, m in enumerate(MODELS)}
    print(f"{D['J']} FOVs, {len(D['cls'])} traces ({(D['cls'] == 0).sum()} HT, {(D['cls'] == 1).sum()} SNAP); "
          f"molecules per FOV {D['density'][0]}-{D['density'][1]}")
    for m, cp in cps.items():
        print(f"{m}: " + ", ".join(f"{k} {v:.4f}" for k, v in comp_row(cp).items())
              + f" | raw F {cp['F_raw'].round(5)}, raw c {cp['c_raw']:.5f}")

    # 1. model check first (point 15: before any limit is read)
    check = model_check(D, cps, rng["check"])
    check.to_csv(a.out / "model_check.csv", index=False)
    valid = {(m, k): bool(check[(check["model"] == m) & (check["k"] == k)]["pass"].all()) for m in MODELS for k in KS}
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(check.to_string(index=False))
    print("validated:", valid, flush=True)
    split = split_check(D, cnt, S, Q, rng["split"])
    split.to_csv(a.out / "model_check_split.csv", index=False)
    fcheck = formula_check(cps, rng["formula"])
    fcheck.to_csv(a.out / "formula_check.csv", index=False)
    cons = constants_check(D, cnt, S, D["ab"], rng["constants"])
    cons.to_csv(a.out / "constants_check.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(fcheck.to_string(index=False))
        print(cons.to_string(index=False))

    # 2. limits on the grid, with bootstrap intervals
    boot, bcomp = bootstrap(D, cnt, S, Q, a.n_boot, rng["boot"])
    rows, crows, comps = [], [], []
    for mi, m in enumerate(MODELS):
        for c, v in comp_row(cps[m]).items():
            x = bcomp[m][c]
            fin = np.isfinite(x)   # rho is undefined in draws with a zero FOV variance
            lo, hi = np.quantile(x[fin], [0.025, 0.975]) if fin.any() else (np.nan, np.nan)
            comps.append({"model": m, "param": c, "estimate": v, "lo": lo, "hi": hi, "boot_undefined": int((~fin).sum())})
        comps += [{"model": m, "param": "a_pure", "estimate": D["ab"][m][0]},
                  {"model": m, "param": "b_pure", "estimate": D["ab"][m][1]}]
        for n in NS:
            for k in KS:
                if n < k:
                    continue
                r = cell(cps[m], D["ab"][m], n, k, curves=True)
                cur = r.pop("curves")
                row = {"model": m, "n": n, "k": k, "n_per_fov": n / k,
                       "density_extrapolated": not (D["density"][0] <= n / k <= D["density"][1]),
                       "validated": valid[(m, k)], **r}
                for s in SCALARS:
                    row[f"{s}_lo"], row[f"{s}_hi"] = ci(boot[m][(n, k)][s])
                    row[f"{s}_boot_not_reached"] = int(np.isnan(boot[m][(n, k)][s]).sum())
                rows.append(row)
                for ruler, q in cur.items():
                    crows.append(pd.DataFrame({"model": m, "n": n, "k": k, "ruler": ruler, "s": S_GRID, **q}))
    grid, curves = pd.DataFrame(rows), pd.concat(crows, ignore_index=True)
    pd.DataFrame(comps).to_csv(a.out / "components.csv", index=False)
    grid.to_csv(a.out / "grid.csv", index=False)
    curves.to_csv(a.out / "quant_curve.csv", index=False)

    drows = []
    for n, k in DIAG:
        for s in ("lod_snap", "lod_ht", "mdd_0.10", "mdd_0.50", "loq_snap_uncal", "loq_snap_cal"):
            pt = grid.query("model == 'B' and n == @n and k == @k")[s].iloc[0] - grid.query("model == 'C' and n == @n and k == @k")[s].iloc[0]
            dd = boot["B"][(n, k)][s] - boot["C"][(n, k)][s]
            fin = np.isfinite(dd)   # a limit not reached in a draw has no difference: no interval if > 2.5 % such draws
            lo, hi = np.quantile(dd[fin], [0.025, 0.975]) if fin.mean() >= 0.975 else (np.nan, np.nan)
            drows.append({"n": n, "k": k, "metric": s, "B_minus_C": pt, "lo": lo, "hi": hi,
                          "boot_not_finite": int((~fin).sum())})
    pd.DataFrame(drows).to_csv(a.out / "diagonal_diff.csv", index=False)

    show = ["model", "n", "k", "validated", "density_extrapolated", "lob_snap_uncal", "lod_snap", "lod_snap_lo",
            "lod_snap_hi", "lod_ht", "lod_ht_lo", "lod_ht_hi", "mdd_0.10", "mdd_0.50", "loq_snap_uncal",
            "loq_ht_uncal", "loq_snap_cal", "loq_ht_cal"]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(grid[show].to_string(index=False))
        print(pd.DataFrame(drows).to_string(index=False))
    if not a.no_figures:
        figures(grid, curves, check, a.out)


if __name__ == "__main__":
    main()
