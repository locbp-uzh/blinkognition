#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Raw physical features before deep learning (R3.2): do the proteins separate on the paper's kinetic features
(mean ON time, mean OFF time, duty cycle; config.yaml `features`) alone, and is that separation the protein's or
the measurement day's? Absolute photon numbers are not used (user, 2026-11-20: they carry day- and slide-specific
information; reported descriptively in R3.3a).

Per pair (config.yaml `pairs`), on all published filtered traces inside vesicles (features from the R3.3a
event table), with log10-transformed features standardized within the pair:
  - PCA (2 components): scores per molecule, explained variance, loadings;
  - unsupervised two-component Gaussian mixture (full covariance, gmm_n_init starts): cluster per molecule
    and its agreement with the proteins (adjusted Rand index, contingency table);
  - logistic regression on the features (class-balanced weights), stratified n_folds-fold cross-
    validation with the paper's seed, scaler fitted on the training folds, test folds balanced by
    subsampling the larger class as in the paper: AUC and balanced accuracy (threshold 0.5) per fold,
    mean +/- population SD; coefficients of a fit on all molecules;
  - the same cross-validation within each session in which both proteins have >= min_per_class_session
    molecules;
  - rank-biserial effect size per feature, pooled and per session: r = 2 U / (n1 n2) - 1 with U the
    Mann-Whitney U of the pair's first protein (r > 0: the first protein tends to larger values);
    no p-values;
  - the networks' cross-validated AUC and balanced accuracy behind Figs. 4C, 5B, 6B (GeneralData).
No hypothesis tests.

Outputs (Results/Revisions/PhysicalFeatures/analysis/run_NNN/): features.csv, pca_scores.csv,
pca_summary.csv, gmm.csv, gmm_summary.csv, lr_folds.csv, lr_summary.csv, lr_coefficients.csv,
effect_sizes.csv, dl_reference.csv, dropped.csv, manifest.yaml, config.yaml, code/.

Usage (repo root):
    python Revisions/PhysicalFeatures/features_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "PaperData"))
from paperdata import REPO_ROOT, load_config, make_run_dir, write_manifest  # noqa: E402


def load_topic_config(path: Path | None = None) -> dict:
    path = Path(path) if path else HERE / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg


def load_features(cfg: dict, frame_ms: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    mol = pd.read_csv(REPO_ROOT / cfg["events_run"] / "molecules.csv")
    df = mol[["construct", "uniqueID", "session", "movie"]].copy()
    for name, col in cfg["features"].items():
        df[name] = mol[col] * (frame_ms if col in cfg["frame_columns"] else 1.0)
    feats = list(cfg["features"])
    ok = np.isfinite(df[feats]).all(axis=1) & (df[feats] > 0).all(axis=1)
    dropped = df.loc[~ok].groupby("construct").size().rename("n_dropped").reset_index()
    return df.loc[ok].reset_index(drop=True), dropped


def balanced_subsample(idx: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Subsample the larger class of y[idx] to the size of the smaller (the paper's balance_test)."""
    a, b = idx[y[idx] == 0], idx[y[idx] == 1]
    n = min(a.size, b.size)
    return np.sort(np.concatenate([rng.choice(a, n, replace=False), rng.choice(b, n, replace=False)]))


def cv_logreg(X: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> list[dict]:
    rows = []
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(folds.split(X, y)):
        te = balanced_subsample(te, y, np.random.default_rng(seed + k))
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        p = lr.predict_proba(sc.transform(X[te]))[:, 1]
        rows.append({"fold": k + 1, "n_train": int(tr.size), "n_test": int(te.size),
                     "auc": float(roc_auc_score(y[te], p)),
                     "balanced_accuracy": float(100 * balanced_accuracy_score(y[te], p >= 0.5))})
    return rows


def rank_biserial(x_first: np.ndarray, x_second: np.ndarray) -> float:
    u = mannwhitneyu(x_first, x_second, alternative="two-sided").statistic
    return float(2 * u / (x_first.size * x_second.size) - 1)


def dl_reference(cfg: dict) -> pd.DataFrame:
    gd = Path(cfg["dl_reference"]["general_data"]) / "MLresults"
    rows = []
    for pair, r in cfg["dl_reference"]["runs"].items():
        runs = [p for p in gd.glob(r["glob"]) if not p.name.startswith("._")]
        if len(runs) != 1:
            raise FileNotFoundError(f"{pair}: {runs}")
        d = pd.read_csv(runs[0] / "augmentation_sweep" / "data.csv")
        d = d[(d["model"] == r["model"]) & (d["aug_factor"] == 0)].iloc[0]
        rows.append({"pair": pair, "figure": r["figure"], "model": r["model"], "run": runs[0].name,
                     "auc_mean": d["auc_mean"], "auc_sd": d["auc_std"], "balanced_accuracy_mean": d["bal_acc_mean"],
                     "balanced_accuracy_sd": d["bal_acc_std"]})
    return pd.DataFrame(rows)


def main() -> None:
    cfg, pcfg = load_topic_config(), load_config(local=True)
    seed, nf, feats = cfg["seed"], cfg["n_folds"], list(cfg["features"])
    run = make_run_dir(pcfg, "analysis", topic_root=cfg["output_root"])
    df, dropped = load_features(cfg, pcfg["frame_interval_ms"])
    df.to_csv(run / "features.csv", index=False, float_format="%.6g")
    dropped.to_csv(run / "dropped.csv", index=False)

    scores, pca_sum, gmm_rows, gmm_sum, lr_folds, coefs, effects = [], [], [], [], [], [], []
    for first, second in cfg["pairs"]:
        pair = f"{first}_{second}"
        d = df[df["construct"].isin([first, second])].reset_index(drop=True)
        y = (d["construct"] == second).to_numpy(int)
        X = np.log10(d[feats].to_numpy(float))
        Z = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2, random_state=seed).fit(Z)
        s = pca.transform(Z)
        scores.append(d[["construct", "uniqueID", "session"]].assign(pair=pair, pc1=s[:, 0], pc2=s[:, 1]))
        for i in range(2):
            pca_sum.append({"pair": pair, "component": f"PC{i + 1}",
                            "explained_variance_ratio": float(pca.explained_variance_ratio_[i]),
                            **{f"loading_{f}": float(v) for f, v in zip(feats, pca.components_[i])}})

        gm = GaussianMixture(n_components=2, covariance_type="full", n_init=cfg["gmm_n_init"],
                             random_state=seed).fit(Z)
        lab = gm.predict(Z)
        gmm_rows.append(d[["construct", "uniqueID", "session"]].assign(pair=pair, cluster=lab))
        ct = pd.crosstab(d["construct"], lab)
        gmm_sum.append({"pair": pair, "adjusted_rand_index": float(adjusted_rand_score(y, lab)),
                        **{f"{c}_in_cluster{k}": int(ct.loc[c, k]) if k in ct.columns else 0
                           for c in (first, second) for k in (0, 1)}})

        for r in cv_logreg(X, y, nf, seed):
            lr_folds.append({"pair": pair, "scope": "pooled", "session": "all", **r})
        lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Z, y)
        coefs.append({"pair": pair, "positive_class": second, **{f"coef_{f}": float(c) for f, c in zip(feats, lr.coef_[0])},
                      "intercept": float(lr.intercept_[0])})

        for ses, g in d.groupby("session"):
            n = g["construct"].value_counts()
            if min(n.get(first, 0), n.get(second, 0)) < cfg["min_per_class_session"]:
                continue
            idx = g.index.to_numpy()
            for r in cv_logreg(X[idx], y[idx], nf, seed):
                lr_folds.append({"pair": pair, "scope": "session", "session": ses, **r})
            for f in feats:
                effects.append({"pair": pair, "scope": "session", "session": ses, "feature": f,
                                f"n_{first}": int(n[first]), f"n_{second}": int(n[second]),
                                "rank_biserial_r": rank_biserial(g.loc[g["construct"] == first, f].to_numpy(),
                                                                 g.loc[g["construct"] == second, f].to_numpy())})
        for f in feats:
            effects.append({"pair": pair, "scope": "pooled", "session": "all", "feature": f,
                            "rank_biserial_r": rank_biserial(d.loc[d["construct"] == first, f].to_numpy(),
                                                             d.loc[d["construct"] == second, f].to_numpy())})

    pd.concat(scores).to_csv(run / "pca_scores.csv", index=False, float_format="%.5f")
    pd.DataFrame(pca_sum).to_csv(run / "pca_summary.csv", index=False, float_format="%.5f")
    pd.concat(gmm_rows).to_csv(run / "gmm.csv", index=False)
    pd.DataFrame(gmm_sum).to_csv(run / "gmm_summary.csv", index=False, float_format="%.5f")
    lf = pd.DataFrame(lr_folds)
    lf.to_csv(run / "lr_folds.csv", index=False, float_format="%.5f")
    summ = lf.groupby(["pair", "scope", "session"], sort=False).agg(
        n_folds=("fold", "size"), n_test_per_fold=("n_test", "median"),
        auc_mean=("auc", "mean"), auc_sd=("auc", lambda v: float(np.std(v, ddof=0))),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", lambda v: float(np.std(v, ddof=0)))).reset_index()
    summ.to_csv(run / "lr_summary.csv", index=False, float_format="%.4f")
    pd.DataFrame(coefs).to_csv(run / "lr_coefficients.csv", index=False, float_format="%.5f")
    pd.DataFrame(effects).to_csv(run / "effect_sizes.csv", index=False, float_format="%.5f")
    dl = dl_reference(cfg)
    dl.to_csv(run / "dl_reference.csv", index=False, float_format="%.5f")

    inputs = [REPO_ROOT / cfg["events_run"] / "molecules.csv"] + [
        p / "augmentation_sweep" / "data.csv" for r in cfg["dl_reference"]["runs"].values()
        for p in (Path(cfg["dl_reference"]["general_data"]) / "MLresults").glob(r["glob"]) if not p.name.startswith("._")]
    write_manifest(run, {**cfg, "frame_interval_ms": pcfg["frame_interval_ms"]}, Path(__file__), inputs)
    print(run)


if __name__ == "__main__":
    main()
