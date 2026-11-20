# PhysicalFeatures: raw physical features before deep learning (R3.2)

Reviewer 3, point 2: a clustering or low-dimensional plot of raw measurable properties (detected
photons, mean ON, mean OFF, duty cycle) to show the data separate before deep learning, as a ground
truth for the models. Plan and decisions in `../ReviewRound1/README.md`.

## Choices (confirmed by the user 2026-11-20)

- Features per molecule from the R3.3a event table (`../PhotonsFrames/events.py` run_001; the
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
/Users/privera/opt/anaconda3/envs/locbplots/bin/python Revisions/PhysicalFeatures/figure_R3.2.py --build --analysis Results/Revisions/PhysicalFeatures/analysis/run_001 --general-data "/Volumes/MediumBerth/SalomePaperData/Püntener2026/GeneralData" --out "/Volumes/MediumBerth/SalomePaperData/Püntener2026/revisions_figures/R3.2"
```

## Results (analysis run_001, 2026-11-20)

No molecule was dropped (all four features finite and positive for all 19,271).

| Pair | PCA (PC1, PC2) | ARI (unsupervised) | Features AUC, pooled | Features AUC, per session | Network AUC |
|---|---|---|---|---|---|
| Halo-D106 vs SNAP-C148 | 58 %, 37 % | 0.29 | 0.904 +/- 0.001 | 0.80, 0.88, 0.80, 0.93 (Exp1-4) | 0.910 +/- 0.017 (CNN-GRU, 4C) |
| Halo-D106 vs Halo-K117 | 73 %, 24 % | 0.41 | 0.925 +/- 0.006 | 0.96, 0.89 (Exp1-2) | 0.923 +/- 0.007 (CNN-GRU, 5B) |
| scGrx1 vs scGrx1-AcK20 | 55 %, 32 % | 0.09 | 0.740 +/- 0.012 | 0.83, 0.67, 0.65, 0.72, 0.60, 0.60 (Exp5-10) | 0.798 +/- 0.018 (TCN, 6B) |

Balanced accuracy, features pooled: 82.4, 84.2, 67.5 %; networks: 83.3, 84.7, 71.5 %.

Effect sizes (rank-biserial r; pooled, range over sessions):

| Pair | Photons per event | Mean ON | Mean OFF | Duty cycle |
|---|---|---|---|---|
| Halo-D106 vs SNAP-C148 | 0.66 (0.50-0.74) | 0.71 (0.53-0.79) | 0.28 (-0.26-0.29) | 0.18 (0.04-0.48) |
| Halo-D106 vs Halo-K117 | 0.83 (0.73-0.89) | 0.81 (0.73-0.87) | -0.17 (-0.16 to -0.11) | 0.53 (0.44-0.52) |
| scGrx1 vs scGrx1-AcK20 | -0.42 (-0.47 to -0.18) | -0.04 (-0.31-0.20) | 0.23 (-0.13-0.50) | -0.28 (-0.42 to -0.04) |

Reading:
- For the two protein-identity tasks, four physical features carry about as much information as the
  networks (AUC 0.90 and 0.93 vs 0.91 and 0.92), and an unsupervised split recovers the proteins
  partly (ARI 0.29, 0.41). The separation holds within single days (0.80-0.96), so it is the
  proteins', not the sessions'. Halo-D106 vs Halo-K117 differences match the paper's Fig. 5D in
  direction on all traces and on both shared days (longer ON, more photons, higher duty cycle for
  Halo-D106; OFF similar).
- For acetylation the features carry less (AUC 0.74 vs the TCN's 0.80; ARI 0.09), and less on most
  single days (0.60-0.72; Exp5 0.83); the network adds information, consistent with Fig. S29.
- Conclusion-level finding (R3.2e, `R3.2_table_on_time_subsets.csv`): the median mean-ON time per
  molecule is the same for scGrx1 and scGrx1-AcK20 on all filtered traces (52 vs 52 ms; 3400 and
  1260 molecules) and on all final-model test traces (52 vs 52 ms; 189 each). The difference in
  Fig. 6C (here 82 vs 155 ms) exists only in the 84 test traces the TCN classified confidently
  (WD >= 0.7). Selecting traces by the classifier's confidence and then comparing their features
  is circular: the network is confident where the classes look most different. Discarding dim
  events (>= 300 or >= 600 photons) does not create a population difference (54 vs 52, 64 vs 58 ms).
  What does differ on the population: scGrx1-AcK20 has more photons per event (r = -0.42) and a
  higher duty cycle (r = -0.28), consistently across days. Since most events are shorter than a
  frame (R3.3a), more photons per event could reflect longer sub-frame ON periods; not tested.
  Fig. 5D is not affected this way: its differences hold on all traces.
- Caveat: ON/OFF features come from the paper's frame classifier, which marks low-level stretches as
  ON in some traces (R2.1; scGrx1 in R3.3a); half of scGrx1's events of >= 3 frames are below 600
  photons per frame against 6 % for scGrx1-AcK20.

Not done: the networks' confident traces overlaid on the PCA (optional, not requested); a test of
the sub-frame ON-time idea for acetylation (would need a choice by the user).
