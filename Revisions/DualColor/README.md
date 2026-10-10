# Dual-color datasets: HT vs SNAP from single-protein slides, tested on mixtures

The reviewers' experiment (two proteins in two spectrally distinct vesicle types) with all
dual-color datasets measured for the revision. Started 2026-10-09; the DFK785-only analysis
and its history are in `../DFK785/README.md`, whose code this folder reuses.

Goal (user, 2026-10-08): extract the traces of every dataset as before (on Daint), train a
model only on the single-protein slides of these datasets (nothing from the paper's data),
and see how well it classifies the vesicle-labeled traces of the mixed slides.

## Data

Raw data: `/Volumes/MediumBerth/Agent/dual_color/<dataset>/` (DFK785 moved there from
`Agent/20260928_DFK785`, so `Revisions/DFK785/config.yaml` input_root is stale). Copy on Daint:
`$SCRATCH/dual_color/` (`/ritom/scratch/cscs/priverafuentes`, purged after 30 days untouched),
checked file by file (count and bytes). 640 nm at 12 mW (DFK785) and 14-15 mW (DFK788/789),
"weaker than usual" in all three readmes.

| Dataset | Slide | Folder | Content | FOVs |
|---|---|---|---|---|
| DFK785 | 1 | SNAP520 | SNAP in ATTO520 | 31 |
| | 2 | HT390 | HT in ATTO390 | 31 |
| | 3 | HT390_SNAP520 | mix (7:1 by volume) | 21 (rest dried) |
| | 4 | HT390_SNAP520/new | mix, next morning | 35 (rest dried) |
| DFK788 | 1 | SNAP520/day1 | SNAP | 26 |
| | 2 | HT390/day1 | HT | 26 |
| | 3 | HT390-SNAP520 | mix (1:1) | 70 |
| | 4 | SNAP520/day2 | SNAP, next morning | 40 |
| | 5 | HT390/day2 | HT, next morning | 40 |
| DFK789 | 1 | SNAP520 | SNAP | 40 |
| | 2 | HT390/day1 | HT, poor immobilization | 66 |
| | 3 | HT390-SNAP520/day2a | mix (1:1), very few vesicles | 50 |
| | 4 | HT390/day2 | HT (6 % biotin) | 30 |
| | 5 | HT390-SNAP520/day2b | mix (1:1, 6 % biotin) | 100 |

DFK789 slide 4: the readme line says "HT390+SNAP520 (200 uL of HT390 ...)" but lists only
HT390, and the files are HT390. The vesicles confirm a single-protein slide: none of its
objects shows the ATTO520 515/488 ratio (below).

## Layout

- `base.yaml`: analysis settings shared by all datasets (copied from the DFK785 config).
- `datasets/<ID>.yaml`: slides, folders, extraction keys; merged with base by `dc.load_config`.
- `dc.py`: config loading, file discovery, run lookup, manifests with the reused DFK785 code.
- `stage_inputs.py`: per-slide links to the 640 movies for the extraction, with a link map
  relative to the data root (a staging made on Daint maps onto the local data).
- `extraction/config_640only_<ID>.yaml`: the paper pipeline, single-channel mode; copies of
  the DFK785 640-only config except folders, slide keys and n_workers.
- `vesicles.py`, `unmix.py`: the DFK785 code run on a dataset config.
- `assign_traces.py`: traces to labeled vesicles; radius by slide type; 640 registration
  estimated from the data.
- `train_pure.py`, `ml/*.yaml`: training on the single-protein slides, test on the mixtures.
- `evaluate.py`: metrics with FOV-cluster bootstrap intervals.
- `compare_runs.py`: table of several models' metrics with the validation-AUC selection rule.
- `background.py`: the filtered background traces with slide, FOV and class (gap >= 3 px to
  every protein box), for the background controls.
- `classify_background.py`: the background-inference control (a protein model classifies
  background traces); train_pure.py options `label_scramble` and `trace_source: background`
  are the other two controls (TRAINING_NOTES.md point 5).
- `plot_sample_traces.py`, `round3_summary.py`: sample traces per dataset and slide type
  (Results/Revisions/SampleTraces/); the round-3 tables (Results/Revisions/DualColor/comparisons/r3_*.csv).
- `paper_in.py`, `added_traces_check.py`, `round4_summary.py`: the paper's IN rule on the
  single-protein slides (picasso-env); the check of the traces it adds; the round-4 tables
  (comparisons/r4_*.csv).
- `kfold_summary.py`, `ml/kfold/*.yaml`: the round-5 FOV-grouped repeated k-fold
  (train_pure.py option `kfold`; TRAINING_NOTES.md point 9) and its tables (comparisons/r5_*.csv).
- `daint/extract.sbatch`, `daint/train.sbatch`: Daint jobs (picasso-env, blink2-cuda).
- `daint/run_train_queue.sh`, `daint/queues/*.txt`: submit a list of train.sbatch jobs to
  the debug partition as slots free.

Outputs: `Results/Revisions/DualColor/<dataset>/<stage>/run_NNN/` (manifest, config, code),
models in `Results/Revisions/DualColor/models/<timestamp>_<run_name>/`. On Daint `Results/` and
`Inputs/DualColor_640only` are links into `$SCRATCH/dualcolor/`.

## Decisions (2026-10-09)

- Trace assignment radius: 4 px on the single-protein slides (training; the paper's radius,
  no vesicle of the other protein exists there), 2 px on the mixed slides (test). The trace is
  the same 5 x 5 box sum either way; what differs is which traces count as "in a vesicle".
  Real protein-vesicle pairs lie within 2.5 px (DFK785 chance analysis), so the 2-4 px ring of
  the training set adds traces that are mostly not in a vesicle (about 13-20 % of 4 px IN
  calls on the DFK785 single-protein slides were chance). The user's caveat: a training set
  defined differently from the test set may differ in kind. So the primary model is also
  trained at 2 px (ml/pure_all_timeinv_2px.yaml, train_radius_px) and both are compared on
  the same test set.
- Training traces on single-protein slides: at a vesicle carrying the slide's own dye label
  (train_label_filter: own_dye), because on DFK789's HT slides most detected objects are not
  ATTO390 vesicles (below).
- Validation held out by FOV (20 % of the FOVs of every single-protein slide); early
  stopping, checkpoint and the Wasserstein threshold use it only.
- Augmentation: the paper's mirror, plus (primary model) random circular shifts and time
  reversal, since absolute timing differed between slides for non-protein reasons on DFK785.
  The paper's recipe alone is trained as a comparison.
- Everything fixed in `ml/*.yaml` before the mixed slides are predicted; each model is
  applied to them once.
- Cross-session check: each dataset in turn left out of training; its single-protein slides
  (exact labels) are the test.

## Vesicles and labels (vesicles run_001, unmixing run_001)

The DualColor code reproduces the DFK785 vesicles, blanks, labels and signatures exactly
(25282 vesicles, zero difference).

| Dataset | Slide | Vesicles / FOV | ATTO390 | ATTO520 | dual | no label |
|---|---|---|---|---|---|---|
| DFK785 | 1 SNAP | 193 | 54 | 5870 | 62 | 3 |
| | 2 HT | 186 | 4938 | 280 | 514 | 33 |
| | 3 mix | 296 | 1739 | 2697 | 1783 | 4 |
| | 4 mix | 209 | 3267 | 2445 | 1127 | 466 |
| DFK788 | 1 SNAP | 93 | 157 | 2129 | 90 | 37 |
| | 2 HT | 160 | 3183 | 259 | 472 | 234 |
| | 3 mix | 155 | 3133 | 3982 | 3738 | 5 |
| | 4 SNAP | 80 | 36 | 3123 | 26 | 25 |
| | 5 HT | 225 | 8011 | 169 | 657 | 172 |
| DFK789 | 1 SNAP | 84 | 78 | 3070 | 95 | 100 |
| | 2 HT | 41 | 735 | 622 | 52 | 1287 |
| | 3 mix | 69 | 763 | 1492 | 481 | 715 |
| | 4 HT | 78 | 580 | 601 | 312 | 840 |
| | 5 mix | 104 | 2107 | 4254 | 1560 | 2517 |

Real ATTO520 vesicles have 515/488 = 2.95-3.23 on every slide (median of those with 488 SNR
> 10) and a median 488 SNR of 150-220 on DFK788/789. The objects labeled ATTO520 on the HT
slides have 515/488 = 0.23-0.40 and a median 488 SNR of 7-9: not ATTO520 vesicles (dim objects
with some 488 signal; on DFK789 slides 2 and 4 they are as numerous as the ATTO390 vesicles,
plus many unlabeled ones). On the mixed slides of DFK789 only about half of the ATTO520-labeled
objects reach 488 SNR 10 (818 / 1492 and 2211 / 4254), so part of the SNAP labels there may sit
on such objects. evaluate.py therefore also reports a 515-confirmed subset (SNAP: 515/488 in
2.0-4.5 with 488 SNR >= 10; HT: 515 SNR < 5).

## Extraction results (corrected chain, 2026-10-09 10:56-11:35)

Run folders (Daint scratch, fetched to `Results/Revisions/DualColor/<ID>/Extraction/` without
the per-movie folders and background traces): DFK785 *_optparam_003, DFK788 *_optparam_002,
DFK789 *_optparam_002. Every movie extracted (118, 202, 286; no failures). DFK785 reproduces
the local extraction of 2026-10-02 (filtered 742 / 512 / 395 / 310 against 742 / 512 / 394 / 310).

| Dataset | Gradient | Slide | Single ROIs | Filtered (pass rate) |
|---|---|---|---|---|
| DFK785 | 20000 | 1 SNAP / 2 HT / 3 mix / 4 mix | 1134 / 690 / 577 / 476 | 742 / 512 / 395 / 310 (66-75 %) |
| DFK788 | 20000 | 1 SNAP / 2 HT / 3 mix / 4 SNAP / 5 HT | 234 / 270 / 781 / 234 / 702 | 126 / 150 / 431 / 127 / 359 (51-56 %) |
| DFK789 | 10000 | 1 SNAP / 2 HT / 3 mix / 4 HT / 5 mix | 392 / 1106 / 379 / 543 / 2506 | 129 / 235 / 71 / 45 / 347 (8-33 %) |

## Trace assignment (DFK785 run_002, DFK788 run_001, DFK789 run_001)

The 640 registration estimated from the data agrees across datasets (405: dy 0.35-0.41,
dx -1.36 to -1.53 px; 488: 0.14-0.29, -0.35 to -0.58; 515: 0.25-0.40, -0.46 to -0.59); residual
offsets of the IN pairs are below 0.03 px everywhere. Filtered traces by class (4 px on the
single-protein slides, 2 px on the mixed slides):

| Dataset | Slide | IN_ATTO390 | IN_ATTO520 | IN_dual | IN_no label | ambiguous | OUT |
|---|---|---|---|---|---|---|---|
| DFK785 | 1 SNAP | 4 | 379 | 6 | 0 | 2 | 351 |
| | 2 HT | 221 | 18 | 67 | 4 | 13 | 187 |
| | 3 mix | 40 | 49 | 50 | 0 | 1 | 255 |
| | 4 mix | 45 | 29 | 15 | 4 | 0 | 217 |
| DFK788 | 1 SNAP | 0 | 73 | 3 | 0 | 0 | 50 |
| | 2 HT | 89 | 0 | 18 | 0 | 2 | 41 |
| | 3 mix | 61 | 100 | 155 | 0 | 2 | 113 |
| | 4 SNAP | 0 | 62 | 2 | 0 | 1 | 62 |
| | 5 HT | 200 | 2 | 21 | 1 | 7 | 128 |
| DFK789 | 1 SNAP | 1 | 56 | 1 | 2 | 0 | 69 |
| | 2 HT | 54 | 3 | 2 | 5 | 2 | 168 |
| | 3 mix | 15 | 11 | 4 | 2 | 0 | 39 |
| | 4 HT | 12 | 2 | 2 | 0 | 2 | 27 |
| | 5 mix | 28 | 34 | 10 | 4 | 0 | 270 |

Training data (single-protein slides, own-dye vesicle within 4 px): about 570 SNAP and 576
HT traces. Test data (mixed slides, single-dye vesicle within 2 px): 189 HT and 223 SNAP
(DFK785 85 / 78, DFK788 61 / 100, DFK789 43 / 45).

## Models (Daint jobs 5010462-3, 2026-10-09; Results/Revisions/DualColor/models/2026-10-09_11-*)

Six models fixed in ml/ before any mixed-slide prediction, each applied once. Test = mixed
slides, single-dye vesicle within 2 px: 189 HT, 223 SNAP traces in 172 FOVs. AUC: ranking of
p_SNAP; BA: balanced accuracy at the argmax; kept: fraction of the labeled traces kept by the
Wasserstein threshold selected on validation; intervals: 95 % FOV-cluster bootstrap.

| Model | Val AUC (pure, held-out FOVs) | Mixed AUC | Mixed BA | Kept | BA of kept |
|---|---|---|---|---|---|
| pure_all_timeinv (primary) | 0.80 [0.73-0.86] | 0.72 [0.66-0.77] | 0.64 [0.59-0.69] | 41 % | 0.77 [0.70-0.83] |
| pure_all_paperaug | 0.69 | 0.50 [0.44-0.56] | 0.52 | 54 % | 0.47 |
| pure_all_timeinv_2px | 0.77 | 0.70 [0.65-0.75] | 0.65 | 48 % | 0.74 |

Primary model by dataset (mixed): DFK785 AUC 0.65 (slide 3 0.77, slide 4 0.48: at chance),
DFK788 0.75, DFK789 0.75. Clean or 515-confirmed subsets change nothing (AUC 0.71).

Cross-session (single-protein slides of a dataset never seen in training): holdout DFK788
AUC 0.73, BA 0.63 (its threshold keeps 97 %); holdout DFK789 AUC 0.73, BA 0.68, BA of the 30 %
kept 0.86. Holdout DFK785 did not train (loss at ln 2 for all epochs; 431 traces, 274 HT / 157
SNAP): its numbers describe a collapsed model and are not reported.

### Check (2026-10-09, three agents; Results/Revisions/DualColor/checks/20261009_model_verification)

- No leakage in any model: sets disjoint by trace and by FOV, test only mixed slides of the
  configured datasets, holdout datasets absent from training, training traces exactly the
  stated rule (recomputed from the vesicle positions), threshold reproduced from the
  validation set alone (on test it would differ in 5 of 6 models). All 111 rows of the six
  metrics.csv files reproduced to 1e-16.
- The signal is in the traces, not in FOV or slide composition: among HT-SNAP pairs from the
  same FOV the primary model's AUC is 0.76 [0.69-0.82] (259 pairs, 67 FOVs); labels shuffled
  within FOVs give 0.50 +- 0.05 (p = 0.001); FOV- or slide-level scores reach at most 0.53-0.59.
  It cannot separate the protein from the vesicle population: HT is always in ATTO390 vesicles
  and SNAP in ATTO520 vesicles, prepared separately (inherent in the design).
- Label noise is not the limit: chance coincidences at 2 px are 7.4 % pooled, capping AUC and
  BA near 0.93-0.96.
- A physical-feature baseline does as well as the CNN on the same split: logistic regression
  on nine features (ON events, mean ON and OFF, duty cycle, span, first and last ON frame,
  relative ON level, OFF noise) AUC 0.72 [0.67-0.76], random forest 0.74 [0.69-0.79]; paired
  differences to the CNN include 0. The features that separate the proteins on both pure and
  mixed slides are the mean ON duration (HT 2.2-2.4 vs SNAP 1.5-1.6 frames) and the relative ON
  level; event count, span and last ON frame separate on the pure slides but weakly on the
  mixed ones (slide cues). Exploratory, not pre-declared: averaging CNN and random forest
  gives 0.745.
- The primary model still leans on the active-window length (Spearman -0.39 with p_SNAP);
  the time augmentation removes absolute position only.
- Frame interval of the DualColor 640 movies: 34.49 ms (ND2 timestamps; DFK785, DFK788,
  DFK789), as for the paper's movies.

## Round 2: augmentation sweep (Daint jobs 5013068, 5013069, 5013182; models 2026-10-09_15-*)

Design in TRAINING_NOTES.md points 1-4: 4 px training set, the paper's early stopping
(patience 10), balanced validation, the paper's augmentation at factors 0 / 3 / 5 on top of
mirror and time invariance, one seed. Per setup, the factor with the highest validation AUC
is the chosen one (rule fixed before the runs). Table:
`Results/Revisions/DualColor/comparisons/r2_aug_sweep.csv` (compare_runs.py). Mixed = mixed
slides of the training datasets; holdout = single-protein slides of the dataset left out.

| Setup | Factor | Epochs (best) | Val AUC | Mixed AUC | Mixed kept, BA of kept | Holdout AUC | Holdout kept, BA of kept |
|---|---|---|---|---|---|---|---|
| all | 0 (chosen) | 43 (33) | 0.81 [0.74-0.87] | 0.71 [0.66-0.76] | 52 %, 0.68 | | |
| | 3 | 36 (26) | 0.80 | 0.72 [0.67-0.77] | 48 %, 0.75 | | |
| | 5 | 28 (18) | 0.80 | 0.71 [0.66-0.76] | 44 %, 0.75 | | |
| holdout DFK785 | 0 (chosen) | 15 (5) | 0.61 [0.48-0.73] | 0.53 | 100 %, 0.50 | 0.66 | 100 %, 0.52 |
| | 3 | 16 (6) | 0.55 | 0.55 | 80 %, 0.52 | 0.69 | 85 %, 0.65 |
| | 5 | 17 (7) | 0.57 | 0.52 | 53 %, 0.51 | 0.69 | 66 %, 0.57 |
| holdout DFK788 | 0 | 38 (28) | 0.76 [0.65-0.86] | 0.63 [0.55-0.70] | 51 %, 0.65 | 0.69 [0.63-0.75] | 55 %, 0.68 |
| | 3 | 60 (50) | 0.82 | 0.68 [0.61-0.75] | 38 %, 0.68 | 0.73 [0.67-0.79] | 44 %, 0.70 |
| | 5 (chosen) | 62 (52) | 0.84 [0.72-0.92] | 0.65 [0.57-0.72] | 37 %, 0.68 | 0.72 [0.66-0.77] | 43 %, 0.72 |
| holdout DFK789 | 0 | 71 (61) | 0.79 | 0.66 [0.60-0.72] | 44 %, 0.67 | 0.68 [0.57-0.80] | 33 %, 0.73 |
| | 3 (chosen) | 63 (53) | 0.80 [0.72-0.87] | 0.67 [0.61-0.73] | 49 %, 0.66 | 0.68 [0.59-0.79] | 40 %, 0.65 |
| | 5 | 41 (31) | 0.79 | 0.65 [0.58-0.71] | 66 %, 0.63 | 0.68 [0.57-0.79] | 65 %, 0.68 |

- Primary setup: augmentation leaves the mixed AUC at 0.71-0.72. Factors 3 and 5 keep fewer
  traces at a higher accuracy, a coverage trade at each model's own threshold rather than a
  better ranking. Validation differs by 0.01 against intervals 0.13 wide.
- Holdout DFK788: factors 3 and 5 are ahead of 0 on validation, mixed and holdout (+0.02 to
  +0.08 AUC), intervals overlapping. Holdout DFK789: no effect.
- Holdout DFK785 fails at every factor (best epoch 5-7, validation near chance on 35 + 35
  traces); factor 0 collapses to p_SNAP about 0.57 for every trace. Not reported as transfer.
- The rule picks a different factor in every setup (0, 0, 5, 3): validation cannot tell the
  factors apart.
- Run-to-run spread is as large as the augmentation effect. Round 1 and round 2 factor 0
  differ only in patience (15 vs 10) and validation balancing, yet holdout DFK788 moves from
  0.73 to 0.69 (mixed 0.68 to 0.63) and holdout DFK789 from 0.73 to 0.68 (mixed 0.70 to 0.66).

## Round 3: ResNet1D, TCN and the paper's controls (Daint jobs 5013893-5014450, models 2026-10-09_16-50 to 17-51)

Design and decision rule: TRAINING_NOTES.md point 5 (fixed before the runs; independent design
review in checks/20261009_round3_design_review). Tables: `Results/Revisions/DualColor/comparisons/
r3_*.csv` (round3_summary.py). Every point value was recomputed independently from the
predictions (about 260 values, no mismatch > 0.005) and every split checked (zero train /
validation FOV overlap in all 20 runs; the ResNet1D and TCN protein runs use r2_all_aug0's exact
sets): checks/20261009_round3_results_check. One training seed per condition; intervals are
95 % FOV-cluster bootstrap and cover test sampling only, not training variance (round 2 saw
run-to-run shifts of 0.04-0.05).

Protein models (mixed slides: 189 HT / 223 SNAP traces, 172 FOVs; paired = both models
resampled over the same FOVs):

| Model | Val AUC | Mixed AUC | Mixed DFK785 / 788 / 789 | Mixed, paired difference to the CNN-GRU |
|---|---|---|---|---|
| CNN-GRU (r2_all_aug0) | 0.81 [0.74-0.87] | 0.71 [0.66-0.76] | 0.65 / 0.75 / 0.75 | |
| ResNet1D | 0.86 [0.79-0.92] | 0.68 [0.63-0.74] | 0.63 / 0.69 / 0.74 | -0.031 [-0.082, 0.016] |
| TCN | 0.83 [0.75-0.90] | 0.72 [0.67-0.77] | 0.68 / 0.75 / 0.73 | +0.009 [-0.025, 0.044] |

- No difference to the CNN-GRU is detected on the mixed slides (not shown equal: a ResNet1D
  deficit up to 0.08 or a TCN gain up to 0.04 is not excluded). Within-dataset AUCs equal the
  pooled ones. The three models' mixed-slide scores correlate (Spearman 0.62-0.79).
- Validation: the pooled paired differences include 0 (ResNet1D +0.05 [-0.02, 0.12], TCN +0.02
  [-0.03, 0.07]); only ResNet1D on DFK785 is ahead beyond noise (+0.10 [+0.02, +0.19], 7 + 7
  FOVs, one slide per class). Every model drops from validation to mixed on DFK785 (CNN-GRU 0.84
  to 0.65, TCN 0.89 to 0.68, ResNet1D 0.94 to 0.63).
- DFK785 mixed slide 4 is at chance for all three (0.48-0.56); DFK785's mixed AUC rests on
  slide 3 (0.72-0.78).

Label scrambling (five permutations per architecture, same FOV split and training seed):

| Model | Val AUC, scrambled labels | Mixed AUC, true labels | Real model, SD above the permutation mean |
|---|---|---|---|
| CNN-GRU | 0.51-0.57 (mean 0.54) | 0.43-0.58 (0.53 +- 0.06) | 3.0 |
| ResNet1D | 0.52-0.56 (0.54) | 0.41-0.64 (0.53 +- 0.10) | 1.5 |
| TCN | 0.54-0.57 (0.55) | 0.43-0.66 (0.55 +- 0.10) | 1.8 |

- No scrambled model fit its labels (training loss at ln 2, stopped at epochs 11-20): they are
  near-initialization draws. They show that labels do not reach the scores through the pipeline
  or the evaluation; leakage that only a memorizing model could use is excluded by the direct
  split checks, not by this control. Five permutations cannot give p below 1/6 alone.
- Scored against the true labels, the scrambled ResNet1D and TCN models separate the validation
  classes at 0.31-0.71 (CNN-GRU 0.42-0.54), and this tracks their mixed AUC (r = 0.92): untrained
  functions of these two architectures already read generic trace statistics that differ
  between the proteins, which widens their null spread.

Background training (same recipe on background traces matched to the protein sets; AUC on the
held-out background of the validation FOVs):

| Model | DFK785 | DFK788 | DFK789 | Training fit |
|---|---|---|---|---|
| CNN-GRU | 0.60 [0.46-0.76] | 0.60 [0.49-0.70] | 0.30 [0.09-0.52] | none (loss at ln 2 for all 15 epochs) |
| ResNet1D | 0.87 [0.69-0.99] | 0.54 [0.34-0.73] | 0.24 [0.05-0.52] | train accuracy 0.69 |
| TCN | 0.89 [0.75-0.99] | 0.56 [0.38-0.72] | 0.24 [0.07-0.50] | train accuracy 0.70 (after 16 flat epochs) |

- On DFK785 ResNet1D and TCN separate the HT from the SNAP slide more than the CNN-GRU (paired
  +0.27 [+0.08, +0.43] and +0.29 [+0.12, +0.44]; 14 FOVs), but DFK785 has no same-protein pair,
  and the CNN-GRU run did not train, so its 0.60 is an untrained reference rather than evidence
  that the architecture cannot read slide cues. On DFK788 and DFK789 the three are within noise
  of each other.

Background inference (protein model on background, deterministic pass, FOVs not trained on):

| Model | Called SNAP | DFK785 | DFK788 | DFK789 |
|---|---|---|---|---|
| CNN-GRU | 0 % (mean p_SNAP 0.027) | 0.47 [0.29-0.63] | 0.39 [0.29-0.51] | 0.45 [0.35-0.55] |
| ResNet1D | 63 % | 0.11 [0.03-0.24] | 0.53 [0.35-0.72] | 0.81 [0.69-0.90] |
| TCN | 0 % (mean p_SNAP 0.0004) | 0.79 [0.64-0.93] | 0.43 [0.30-0.55] | 0.40 [0.33-0.47] |

- By the paper's criterion (collapse into one class) the CNN-GRU and TCN pass, ResNet1D fails.
  The CNN-GRU's per-dataset AUCs lie within its same-protein slide pairs (0.36-0.61). TCN's DFK785
  ranking (0.79) is 0.4 logit between slides at an output about 6 logits from any protein call.
  ResNet1D's background outputs fall inside its protein decision range, and on DFK789 its
  HT-vs-SNAP slide pairs (0.79, 0.84) exceed its same-protein pair (0.54): the one place where
  the same-protein reference flags a model.

What the background slide differences are (checks/20261009_round3_results_check, reproduced
locally): a movie-wide common mode. The skewness of the
filtered background alone separates DFK785's slides at 0.92 and DFK789's at 0.17-0.19, and the
lag-1 autocorrelation separates slides of one protein (DFK788 slides 2 vs 5: 0.92). After
removing each movie's common mode (per trace, the leave-one-out mean of the other background
traces of the movie, regressed out), every slide pair lies at 0.41-0.59. It does not depend on
the distance to the nearest protein box (not local protein residue), and the background-trained
ResNet1D and TCN outputs track the skewness. A movie-wide component is shared by both classes
of a mixed FOV, so it cannot raise the mixed-slide AUC; it can raise the validation AUC.

Decision (rule of point 5): neither ResNet1D nor TCN replaces the CNN-GRU. The first condition
("mixed-slide AUC higher") is read as higher beyond the paired FOV bootstrap, a reading made at
readout (taken literally, TCN is +0.009); it fails for both. ResNet1D is rejected more firmly
(lower point estimate, no collapse, DFK789 inference beyond its reference); TCN is level with
the CNN-GRU, and its background excess is only on DFK785, where there is no reference and the
CNN-GRU comparator did not train. The outcome agrees with the paper's choice of the CNN-GRU;
with one seed and a matched background budget it is weaker evidence than the paper's.

Not done: a vesicle-population control (background boxes next to single-dye vesicles on the
mixed slides, compared within FOV after removing the movie common mode); training with the
movie common mode removed from the inputs; repeated runs (TRAINING_NOTES "Open").

## Trace yield and the paper's IN rule (2026-10-09)

Why the dual-color datasets give few traces, measured stage by stage against the paper:
Results/Revisions/DualColor/checks/20261009_trace_yield/README.md. Up to the filter (defined
identically on both sides) DFK785 + DFK788 give as many traces per FOV as the paper (HT 10.5 vs
10.4, SNAP 10.3 vs 11.7 without SP_Exp4, which alone gave 8453 of the 9647 published SNAP
traces); the largest loss is the vesicle-assignment rule (the paper kept 0.82-0.94 of its
filtered traces as IN, the vesicle table 0.24-0.61). The training slides now use the paper's IN
rule (TRAINING_NOTES.md point 7; paper_in.py, Extraction_gt405 / Extraction_gt488 runs):
1829 traces in the pool against 1165, chance share of IN calls 2-14 % per slide after the
vesicle-channel gradient rule for DFK789 405 nm (checks/20261009_paper_in_gradient).

## Round 4: table pool vs paper-IN pool (Daint jobs 5015392, 5015393, 5015427; models 2026-10-09_21-*)

Design and rule: TRAINING_NOTES.md point 8 (fixed before the runs). CNN-GRU recipe of
r2_all_aug0, five seeds per pool (the seed sets the FOV split, initialization and sampling;
r2_all_aug0 is the table pool's seed 840410), the same 412 mixed-slide test traces. Tables:
comparisons/r4_*.csv (round4_summary.py). All values and splits recomputed independently
(checks/20261009_round4_check): test rows identical in all 10 runs, train / validation FOVs
disjoint, both pools rebuilt exactly from their sources.

| Pool | Training traces | Mixed AUC, seeds 840410 / 1 / 2 / 3 / 4 | Mean (SD) |
|---|---|---|---|
| Vesicle table (own dye, 4 px) | 899-921 | 0.713 / 0.657 / 0.664 / 0.716 / 0.711 | 0.692 (0.029) |
| Paper IN | 1391-1471 | 0.701 / 0.718 / 0.740 / 0.711 / 0.715 | 0.717 (0.014) |

- Paired difference (paper-IN minus table, same seed): +0.025 [-0.025, +0.075] (95 % t
  interval, 4 df); DFK785 -0.031 [-0.075, +0.013], DFK788 +0.073 [-0.009, +0.155], DFK789
  +0.025 [-0.021, +0.071]. By the rule of point 8 the paper-IN pool is the default: not shown
  to be worse, not shown to be better (with five seeds the rule could only reject a deficit of
  about 0.07).
- The mean gain comes from two table-pool runs (seeds 1 and 2) that stopped early (best epochs
  12 and 25, the lowest validation AUCs of the ten runs); on the other three seeds the pools
  agree within 0.012, and the five-seed ensembles differ by +0.007. The lower seed spread of
  the paper-IN runs is not established (F test p = 0.19) and is confounded with training
  length: patience counts epochs, about 290 optimizer steps on the table pool against 450.
- The two arms also differ in composition (697 added traces, 479 of them DFK785, 45 % of the
  paper-IN DFK785 pool), in selection relative to the table-defined test set, and in the
  validation set, so the comparison cannot separate size from these.
- DFK785 drops in 4 of 5 seeds, on mixed slide 3 (the slide where the models work; slide 4 is at
  chance in all runs). Not explained by purity: the paper-IN DFK785 chance share (14 %) is no
  higher than the table's at 4 px (13-20 %).
- Are the added traces protein traces (added_traces_check.py; checks/20261009_round4_added_traces)?
  Scored by the five table-pool models on the FOVs each never trained on, the added traces
  separate HT from SNAP at a within-dataset AUC of 0.73 (0.61-0.80 over the models) against 0.81
  (0.75-0.85) for the table traces of the same FOVs (DFK785 0.74 vs 0.85; the table traces are
  those models' validation set, a slight advantage). They carry protein information, but less.
- Single-run differences of about 0.05 (rounds 2-3) are within the table pool's seed spread
  (about 0.04 for the difference of two single runs).

Balanced accuracy before and after the MC-dropout filter (comparisons/r4_ba.csv,
r4_ba_summary.csv, r4_ba_paired.csv; added to round4_summary.py 2026-10-10). Before: argmax of
the mean over 100 MC passes, all traces. After: the traces kept at the run's own Wasserstein
threshold (selected on its validation set, trace loss at most 50 %). Mixed slides, mean (SD)
over the five seeds:

| Pool | Dataset | BA all | Kept fraction | BA kept | Gain |
|---|---|---|---|---|---|
| Vesicle table | pooled | 0.632 (0.018) | 0.520 (0.055) | 0.689 (0.040) | +0.056 |
| Vesicle table | DFK785 / 788 / 789 | 0.618 / 0.620 / 0.656 | 0.50 / 0.52 / 0.55 | 0.645 / 0.697 / 0.725 | |
| Paper IN | pooled | 0.654 (0.014) | 0.491 (0.022) | 0.753 (0.018) | +0.099 |
| Paper IN | DFK785 / 788 / 789 | 0.590 / 0.689 / 0.677 | 0.49 / 0.47 / 0.53 | 0.643 / 0.808 / 0.817 | |

On validation (each run's own held-out single-protein FOVs, so different sets per pool): table
0.689 to 0.785 at 0.57 kept, paper IN 0.730 to 0.846 at 0.53 kept.

- Paired by seed (paper IN minus table, 95 % t interval, 4 df), pooled mixed slides: BA all
  +0.022 [+0.007, +0.036] (5 of 5 seeds); BA kept at own threshold +0.064 [-0.006, +0.135];
  kept fraction -0.029 [-0.108, +0.049]; BA of the 50 % most certain traces (matched coverage)
  +0.053 [-0.010, +0.117] (4 of 5). Per dataset, the matched-coverage difference is DFK785
  -0.025, DFK788 +0.092 [+0.028, +0.156] (5 of 5), DFK789 +0.078 (5 of 5).
- Why BA moves when AUC does not: the table runs' decision boundary swings between the classes
  (mixed recall HT / SNAP from 0.77 / 0.48 to 0.44 / 0.83 across seeds); the paper-IN runs stay
  balanced (0.59-0.67 / 0.61-0.73). AUC ignores the operating point; BA at the argmax does not.
- The filter gain is larger for the paper-IN pool partly because its thresholds are higher
  (0.39-0.56 against 0.10-0.42) and so keep slightly fewer traces; the matched-coverage row
  removes that. The confounds listed above (training length, composition, validation set)
  apply to these differences too.
- Verified: the BA function of round4_summary.py reproduces evaluate.py's all / kept BA and kept
  fraction in all 120 run x dataset comparisons (maximum difference 6e-17).

## Round 5: FOV-grouped repeated k-fold (Daint jobs 5021067, 5021068, 5021105, 5021136; models 2026-10-10_10-*)

Design: TRAINING_NOTES.md point 9 (fixed before the runs, reviewed in checks/20261010_round5_review).
Tables: comparisons/r5_*.csv (kfold_summary.py; log r5_summary.log). The 15 Daint splits equal the
local dry runs (checks/20261010_round5_dryrun) trace for trace, with the pinned inputs.

6 of the 15 runs did not train: their training loss never left the initial plateau (ln 2; below
0.65 in no epoch), the best checkpoint was a validation-AUC spike at epochs 2-8 and patience ran
out at epochs 12-18. Their mixed AUC is 0.46-0.58, against 0.67-0.72 for the nine others. The
plateau ended at epochs 8-21 in every run that left it (rounds 4 and 5), later on smaller
training sets; the failure and the fix are TRAINING_NOTES.md point 10, rerun as round 5b.

Declared readouts (all 15 runs), mixed slides: AUC 0.633 (SD 0.096), balanced accuracy 0.591 all /
0.648 kept at 0.55 kept; ensembles of the five fold models of a repeat 0.700-0.721, of all 15 0.713
(the untrained models' near-constant scores barely move a mean). Gap d (test-fold AUC minus
mixed AUC, same model): overall +0.090 [+0.038, +0.145]; DFK785 +0.190 (slide 3 +0.091, slide 4
+0.311), DFK788 -0.015, DFK789 +0.080; without DFK785 slide 4 +0.024 [-0.028, +0.075].

Post hoc, the nine trained runs (r5_*_trained.csv; not declared, superseded by round 5b): mixed AUC
0.705 (SD 0.020; DFK785 0.629, DFK788 0.739, DFK789 0.746), balanced accuracy 0.649 all / 0.725
kept at 0.49 kept; test-fold AUC within datasets 0.808. Gap overall +0.090 [+0.032, +0.149],
carried by DFK785 slide 4 (+0.373 [+0.230, +0.506]); without it +0.012 [-0.051, +0.072]; DFK788
-0.013, DFK789 +0.035.

Out-of-fold slide pairs (all runs): HT vs SNAP slides 0.57-0.80, same-protein pairs 0.41-0.62
(separation 0.02-0.12), so the held-out FOVs of DFK788 and DFK789 separate by protein more than
by slide. The untrained runs still rank the held-out DFK785 FOVs at AUC 0.67-0.78 while near chance
on the mixed slides (0.49-0.58): a simple early feature tells the two DFK785 single-protein slides apart
without carrying over to the mixtures (compare the round-3 background models, 0.87-0.89 on DFK785).

## Paper code (checked 2026-10-10)

The revisions run on the `revisions` branch, on top of `main` (6f93bb0, untouched). Outside
`Revisions/` they change four things, none of which changes a paper run:

- `ML/utils.py` `train_model(..., warmup_epochs=0)` (TRAINING_NOTES.md point 10): with the default
  both added lines are no-ops; the paper's `ML/train.py` does not pass it and `ML/crossval.py` has
  its own loop. `tests/test_warmup.py` reproduces the train_model of 83efc7c (before the option)
  exactly (losses, AUCs, saved checkpoint).
- `Extraction/run_pipeline.py`: the step scripts are taken from the folder of run_pipeline.py
  instead of the config's folder; the same folder for the paper's `Extraction/config.yaml`.
- `Extraction/utils.py` `rel_under`: `os.path.abspath` instead of `Path.resolve()`; the same output
  folders for real files (also under a symlinked root), different only for a movie that is itself
  a symlink to outside the input root, where the old version wrote to the wrong folder.
- New files only: `ML/classify.py`, `ML/config_classify.yaml`, `docs/classify.md` (inference with a
  trained model; not called by the paper's code).

## Status

- Extraction on Daint: DFK785, DFK788, DFK789 submitted (jobs 5009197, 5009198, 5009538);
  all three copies verified against the local data (DFK785 410 files, DFK788 808, DFK789 1144;
  identical byte totals).
- Models fixed in ml/ (six): pure_all_timeinv (primary), pure_all_paperaug (paper's
  augmentation only), pure_all_timeinv_2px (training at 2 px), pure_holdout_<ID>_timeinv x3
  (cross-session).
- Trace assignment tested on DFK785 with the old local extraction (trace_assignment run_001;
  superseded once the Daint extraction is in): the data-driven registration gives 405 (0.34,
  -1.50), 488 (0.29, -0.35), 515 (0.25, -0.59) px, matching the occupancy offsets of DFK785
  with the 405 x correction found in its check (-1.14 - 0.33); residuals now about 0 px.
