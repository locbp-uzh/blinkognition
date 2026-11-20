# PhysicalFeatures: raw physical features before deep learning (R3.2)

Reviewer 3, point 2: a clustering or low-dimensional plot of raw measurable properties (detected
photons, mean ON, mean OFF, duty cycle) to show the data separate before deep learning, as a ground
truth for the models. Plan and decisions in `../ReviewRound1/README.md`.

## Choices (confirmed by the user 2026-11-20)

- Features per molecule from the R3.3a event table (`../PhotonsFrames/events.py` run_002; the
  paper's ON classification, reproduces Figs. 5D and 6C exactly): median photons per ON event, mean
  ON, mean OFF (34.49 ms per frame), duty cycle; all published filtered traces inside vesicles, not
  only the networks' confident ones.
- log10, then standardization within the pair (within the training folds for the classifier).
- PCA (2 components) for the picture; unsupervised two-component Gaussian mixture and its adjusted
  Rand index against the proteins.
- Logistic regression on the four features, 4-fold stratified CV with the paper's seed (840410),
  class-balanced training weights and test folds balanced by subsampling: AUC and balanced accuracy,
  mean +/- population SD, next to the networks of Figs. 4C, 5B, 6B.
- The same within single sessions with both proteins (>= 20 molecules each); rank-biserial effect
  size per feature pooled and per session (r > 0: the pair's first protein larger); no p-values.
- No hypothesis tests. Optional overlay of the networks' confident traces not done (not requested).

## Files

- `config.yaml`, `features_analysis.py` (all computations; outputs in
  `Results/Revisions/PhysicalFeatures/analysis/run_NNN/`), `figure_R3.2.py` (figures R3.2a-e;
  `--build --analysis <run> --general-data <GeneralData> --out <folder>`, or replot from the CSVs).

Figure folder: `/Volumes/MediumBerth/SalomePaperData/Püntener2026/revisions_figures/R3.2/`
(R3.2a overview, R3.2b-d corner plots per pair, R3.2e all traces vs the Fig. 5D/6C sets; CSVs,
captions, provenance, script).

```bash
/Users/privera/opt/anaconda3/envs/locbplots/bin/python Revisions/PhysicalFeatures/features_analysis.py
/Users/privera/opt/anaconda3/envs/locbplots/bin/python Revisions/PhysicalFeatures/figure_R3.2.py --build --analysis Results/Revisions/PhysicalFeatures/analysis/run_003 --general-data "/Volumes/MediumBerth/SalomePaperData/Püntener2026/GeneralData" --out "/Volumes/MediumBerth/SalomePaperData/Püntener2026/revisions_figures/R3.2"
```

## Results (analysis run_003, 2026-11-20): kinetic features only

The user's decision (2026-11-20): proteins are separated only on the kinetic features the paper uses,
mean ON time, mean OFF time and duty cycle. Absolute photon numbers carry experiment-, day- and
slide-specific information (laser power, focus, slide quality), which is why the paper normalizes its
traces; they are reported descriptively (R3.3a), not used for separation, and the response argues
against photon-based features. Runs run_001 and run_002 (with median photons per event as a fourth
feature) are superseded; their numbers are in the log below.

No molecule was dropped (all three features finite and positive for all 19,271).

| Pair | PCA (PC1, PC2) | ARI (unsupervised) | Features AUC, pooled | Features AUC, per session | Network AUC |
|---|---|---|---|---|---|
| Halo-D106 vs SNAP-C148 | 65 %, 35 % | 0.29 | 0.899 +/- 0.002 | 0.82, 0.88, 0.77, 0.93 (Exp1-4) | 0.910 +/- 0.017 (CNN-GRU, 4C) |
| Halo-D106 vs Halo-K117 | 76 %, 24 % | 0.41 | 0.910 +/- 0.008 | 0.94, 0.88 (Exp1-2) | 0.923 +/- 0.007 (CNN-GRU, 5B) |
| scGrx1 vs scGrx1-AcK20 | 63 %, 36 % | 0.07 | 0.667 +/- 0.016 | 0.75, 0.61, 0.66, 0.71, 0.36, 0.58 (Exp5-10) | 0.798 +/- 0.018 (TCN, 6B) |

Balanced accuracy, features pooled: 82.0, 82.9, 64.4 %; networks: 83.3, 84.7, 71.5 %. (scGrx1 Exp9:
0.36 +/- 0.23 over folds, below chance with a large spread; few scGrx1-AcK20 molecules that day.)

Effect sizes (rank-biserial r; pooled, range over sessions):

| Pair | Mean ON | Mean OFF | Duty cycle |
|---|---|---|---|
| Halo-D106 vs SNAP-C148 | 0.71 (0.53-0.79) | 0.28 (-0.26-0.29) | 0.18 (0.04-0.48) |
| Halo-D106 vs Halo-K117 | 0.81 (0.73-0.87) | -0.17 (-0.16 to -0.11) | 0.53 (0.44-0.52) |
| scGrx1 vs scGrx1-AcK20 | -0.04 (-0.31-0.20) | 0.23 (-0.13-0.50) | -0.28 (-0.42 to -0.04) |

Reading:
- For the two protein-identity tasks, three kinetic features carry nearly as much information as the
  networks (AUC 0.90 and 0.91 vs 0.91 and 0.92; with photons per event they were 0.90 and 0.93), and
  an unsupervised split recovers the proteins partly (ARI 0.29, 0.41). The separation holds within
  single days (0.77-0.94), so it is the proteins', not the sessions'. Halo-D106 vs Halo-K117
  differences match the paper's Fig. 5D in direction on all traces and on both shared days (longer
  ON and higher duty cycle for Halo-D106; OFF similar).
- For acetylation the kinetic features carry clearly less (AUC 0.67 vs the TCN's 0.80; ARI 0.07;
  single days 0.36-0.75): the network uses information beyond these three summaries, consistent with
  Fig. S29.
- Conclusion-level finding (R3.2e, `R3.2_table_on_time_subsets.csv`; unchanged): the median mean-ON
  time per molecule is the same for scGrx1 and scGrx1-AcK20 on all filtered traces (52 vs 52 ms; 3400
  and 1260 molecules) and on all final-model test traces (52 vs 52 ms; 189 each). The difference in
  Fig. 6C (here 82 vs 155 ms) exists only in the 84 test traces the TCN classified confidently
  (WD >= 0.7). Selecting traces by the classifier's confidence and then comparing their features is
  circular: the network is confident where the classes look most different. Of the kinetic features,
  the population differs in duty cycle (scGrx1-AcK20 higher, r = -0.28, negative on every day); the
  OFF-time difference (r = 0.23) changes sign between days. (scGrx1-AcK20 also has more photons per
  event, r = -0.42, but by the decision above that is not a basis for interpretation.) Fig. 5D is not
  affected this way: its differences hold on all traces.
- Caveat: ON/OFF features come from the paper's frame classifier, which marks low-level stretches as
  ON in some traces (R2.1; scGrx1 in R3.3a).

Not done: the networks' confident traces overlaid on the PCA (optional, not requested); a test of
the sub-frame ON-time idea for acetylation (would need a choice by the user).

## Log

- 2026-11-20, `analysis` run_001 (four features incl. median photons per event, 0.053 photons per ADU):
  AUC 0.904, 0.925, 0.740; ARI 0.29, 0.41, 0.09; photons per event r 0.66, 0.83, -0.42. Superseded.
- 2026-11-20, `analysis` run_002: the same analysis on events run_002 (0.062 photons per ADU, user
  decision). Every AUC, balanced accuracy, ARI, PCA variance and effect size equals run_001 within
  0.0001 (rank-based and standardized log features are scale-free; 3 of 19,271 molecules changed ON
  classification by a few frames, see `../PhotonsFrames/README.md`). Figure R3.2 rebuilt from it
  (photon axes x 1.1625).
- 2026-11-20, `analysis` run_003: kinetic features only (user decision above); figure R3.2 rebuilt
  from it (PCA, clustering, classifier, effect sizes and corner plots on ON, OFF and duty cycle;
  panels d-f now mean OFF against mean ON).
