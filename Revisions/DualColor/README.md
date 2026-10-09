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
- `daint/extract.sbatch`, `daint/train.sbatch`: Daint jobs (picasso-env, blink2-cuda).

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
