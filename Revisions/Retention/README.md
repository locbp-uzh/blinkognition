# Retention: traces kept by the filters (R3.4a)

Reviewer 3, point 4, first half: the share of the data kept after the confidence filters. Plan and decisions in `../ReviewRound1/README.md`; the
mixture half of R3.4 is the DualColor session's.

## What is computed

Nothing is retrained; every number comes from the paper's published outputs.

- Cross-validation behind the main-figure confusion matrices (matched to the published numbers,
  GeneralData `MLresults/CrossValidation/`): Fig. 4C and 5B CNN-GRU, Fig. 6B TCN; min-max traces,
  time reversal, no further augmentation; the background controls without time reversal.
  Per fold, the stored Wasserstein-distance (WD) sweep (`MCD_results/cv_wd_sweep.csv`, 50
  thresholds from 0 to the fold's largest WD) gives the kept fraction and balanced accuracy.
  At the paper's WD = 0.7: the two bracketing grid points bound the kept fraction (it can only
  fall as the threshold rises); the single value is the linear interpolation between them.
  Fold summaries are mean +/- population SD (n = 4 in the denominator), as in the paper.
  Test folds are balanced, so kept fractions are of balanced test sets, pooled over both proteins.
- Final models behind Figs. 5D and 6C (GeneralData `MLresults/FinalModels/`): per-trace WD in
  `traces_with_wasserstein.npz`, so exact per-protein kept fractions.
- Trace funnel per construct from step 0 (`../PaperData/`, check run_002): ROIs, single ROIs in
  vesicles, valid traces, quality-filtered traces; approximate yield = quality-filter pass rate x
  the pair's mean kept fraction at WD 0.7.

Statistical choices confirmed by the user 2026-11-20. No tests.

## Files

- `figure_R3.4a.py`: builds the plotted data and the figure. `--build --general-data <GeneralData>
  --step0 <check run> --out <figure folder>` reads the published outputs, writes the CSVs, tables,
  caption and provenance, and copies itself into the figure folder; without `--build` it replots
  from the CSVs next to it (the figure folder works on its own).

Figure folder (user's convention, 2026-11-20):
`/Volumes/MediumBerth/SalomePaperData/Püntener2026/revisions_figures/R3.4a/`: `R3.4a.pdf`,
`R3.4a_panels_a-c.csv`, `R3.4a_panels_d-f.csv`, `R3.4a_panels_g-h.csv`, `R3.4a_wd07_by_fold.csv`,
`R3.4a_table_cv.csv`, `R3.4a_table_final.csv`, `R3.4a_table_funnel.csv`, `R3.4a_caption.txt`,
`R3.4a_provenance.yaml`, `figure_R3.4a.py`.

Run with the locbplots environment:

```bash
/Users/privera/opt/anaconda3/envs/locbplots/bin/python Revisions/Retention/figure_R3.4a.py --build --general-data "/Volumes/MediumBerth/SalomePaperData/Püntener2026/GeneralData" --step0 Results/Revisions/PaperData/check/run_002 --out "/Volumes/MediumBerth/SalomePaperData/Püntener2026/revisions_figures/R3.4a"
```

## Results (2026-11-20)

Cross-validation, WD >= 0.7, mean +/- SD over 4 folds (bounds from the grid in brackets):

| Pair (figure, model) | Test traces per fold | Kept, protein | Kept, background | Balanced accuracy, all | Balanced accuracy, kept |
|---|---|---|---|---|---|
| Halo-D106 vs SNAP-C148 (4C, CNN-GRU) | 1290 | 61.9 +/- 2.6 % (61.2-62.9) | 0.0 % | 83.7 +/- 1.3 % | 94.2 +/- 2.0 % |
| Halo-D106 vs Halo-K117 (5B, CNN-GRU) | 1192 | 69.8 +/- 1.7 % (69.0-70.3) | 1.1 +/- 1.0 % | 85.4 +/- 0.6 % | 93.5 +/- 0.5 % |
| scGrx1 vs scGrx1-AcK20 (6B, TCN) | 630 | 18.1 +/- 1.2 % (16.6-18.8) | 0.0 % | 72.0 +/- 1.9 % | 88.9 +/- 1.6 % |

Background test folds: 4478-4480, 884-886 and 4360-4362 traces; balanced accuracy on all of
them 51.5, 51.9 and 50.0 %.

Final models, WD >= 0.7 (single test split, 15 % of traces, balanced):

| Final model | Protein | Kept | Recall among kept | Recall, all test traces |
|---|---|---|---|---|
| CNN-GRU (Fig. 5D) | Halo-D106 | 262/358 = 73.2 % | 90.5 % | 82.4 % |
| | Halo-K117 | 248/358 = 69.3 % | 94.0 % | 86.6 % |
| TCN (Fig. 6C) | scGrx1 | 59/189 = 31.2 % | 100 % | 75.1 % |
| | scGrx1-AcK20 | 25/189 = 13.2 % | 84.0 % | 72.0 % |

Funnel (traces inside vesicles) and approximate yield:

| Construct (pair) | ROIs, all | Single, in vesicles | Valid | Quality-filtered (pass) | Approx. yield after WD 0.7 |
|---|---|---|---|---|---|
| Halo-D106 (4C) | 19590 | 6152 | 6067 | 2580 (42.5 %) | 26.3 % |
| SNAP-C148 (4C) | 28481 | 15747 | 15382 | 9647 (62.7 %) | 38.8 % |
| Halo-D106 (5B) | 19590 | 6152 | 6067 | 2580 (42.5 %) | 29.7 % |
| Halo-K117 (5B) | 8646 | 4740 | 4740 | 2384 (50.3 %) | 35.1 % |
| scGrx1 (6B) | 48358 | 12737 | 12664 | 3400 (26.8 %) | 4.8 % |
| scGrx1-AcK20 (6B) | 19907 | 4336 | 4324 | 1260 (29.1 %) | 5.3 % |

Reading:
- For the two protein-identity tasks the filter keeps 62-70 % of test traces and raises balanced
  accuracy by 8-11 points; for acetylation it keeps 18 % and raises it by 17 points, and the
  final TCN keeps AcK20 less than half as often as scGrx1.
- The filter keeps essentially no background traces (0-1 %), so confident calls come from
  protein blinking, not from noise.
- Without the filter, the TCN's cross-validated balanced accuracy for acetylation (72 %) is close
  to the random forest's with its own uncertainty filter (Fig. S29, 74 %); with the filter the
  TCN reaches 89 %.

Checks: the final models' accuracies on all test traces (84.5 %, 73.5 %) equal
`initial_accuracy` in their `mc_dropout_metrics.json`; the WD >= 0.7 counts equal the n of
Figs. 5D and 6C; replotting from the figure folder's CSVs alone gives a pixel-identical render.
Not verified: kept fractions at exactly 0.7 in cross-validation (the fold models were not saved;
values are bracketed by the grid).
- 2026-11-20 (user: answer only what is asked): R3.4a is now one table (`R3.4a_table.csv`, rendered as
  `R3.4a.pdf`; `figure_R3.4a.py`, commit c623014): quality-filter pass, kept at WD >= 0.7 (cross-validation
  and final models), approximate yield, balanced accuracy all and kept. The threshold-sweep figure is in
  `revisions_figures/not_in_revision/R3.4a_full_20261120/` and git history (e4dc7fb).
