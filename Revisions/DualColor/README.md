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
  defined differently from the test set may differ in kind. Checked by also training at 2 px
  if the 4 px model looks suspicious (not run yet).
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

## Status

- Extraction on Daint: DFK785 and DFK788 submitted (jobs 5009197, 5009198); DFK789 after the
  copy of its last two folders.
- Trace assignment tested on DFK785 with the old local extraction (trace_assignment run_001;
  superseded once the Daint extraction is in): the data-driven registration gives 405 (0.34,
  -1.50), 488 (0.29, -0.35), 515 (0.25, -0.59) px, matching the occupancy offsets of DFK785
  with the 405 x correction found in its check (-1.14 - 0.33); residuals now about 0 px.
