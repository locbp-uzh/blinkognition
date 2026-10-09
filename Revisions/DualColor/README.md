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
