# DFK785: single-label and mixed ATTO390 / ATTO520 vesicle slides

Dataset acquired 2026-09-28/29, now at `/Volumes/MediumBerth/Agent/20260928_DFK785`
(moved there by the user on 2026-10-02; `20260926_readme.txt` inside describes the
slides). Work lives on the `revisions` branch only. Shared code (detection,
config, provenance) comes from `../Bleedthrough/common.py`.

| Slide | Folder | Sample | FOVs | Channels |
|---|---|---|---|---|
| 1 | SNAP520 | SNAP-HMSiR-IA in ATTO520 vesicles | 31 | 405, 488, 640 |
| 2 | HT390 | HT7-HMSiR-HTL in ATTO390 vesicles | 31 | 405, 488, 640 |
| 3 | HT390_SNAP520 | mix 350 uL HT390 + 50 uL SNAP520, overnight | 61 | 405, 488, 515, 640 |
| 4 | HT390_SNAP520/new | same mix, next morning | 60 | 405, 488, 515, 640 |

One ND2 per channel and FOV, 230 x 230 px at 0.13 um; 405/488/515 are 10-frame
movies, 640 (HMSiR) 6000 frames. All raw ND2 files except the two test images carry
the macOS lock flag (`uchg`): they can be read but not renamed or deleted.

## Goals (user, 2026-10-02)

1. Bleed-through and unmixing of the vesicles to get their labels: the HT7 protein
   is in ATTO390 vesicles, the SNAP protein in ATTO520 vesicles.
2. The fraction of each vesicle type that holds at least one protein (640 signal
   or not), from the 640 time traces, as % per vesicle / protein type.

Decided 2026-10-02: labels by per-object unmixing with 405 + 488 (slides 1-2 have
no 515; 515 is a check on the mixed slides); occupancy by score A (robust peak
score of the 640 trace) with score B (GMM ON-frame count) as cross-check, a blank
calibrated threshold and a chance correction.

## Running

```bash
python Revisions/DFK785/fov_qc.py           # vesicle counts per FOV in acquisition order
python Revisions/DFK785/move_dried.py       # dry run of moving dried acquisitions to bad_data/
python Revisions/DFK785/vesicles.py         # detection and photometry per FOV (+ blanks)
python Revisions/DFK785/unmix.py            # signatures from slides 1-2, labels for all vesicles
python Revisions/DFK785/occupancy.py        # 640 traces, scores A and B, occupancy per label
```

Outputs in `Results/Revisions/DFK785/<stage>/run_NNN/` with manifest, effective
config and code copy. The ND2 metadata here does not record per-channel settings:
every file of a slide carries the same microscope state (slides 1-2: 515 laser and
the dual 442/514 filter, slides 3-4: 405 laser and Continuous STORM, for 405, 488
and 640 files alike), so settings come from the file names and the data readme.

## FOV pairing

FOVs are matched by acquisition time (ND2 metadata), not by file number: on slide 3
the numbering is shifted between channels (the first FOV is 405/488/640 file _0031
with the unnumbered 515 file, and for most FOVs 515 file N goes with 405/488/640
file N-1). Within every FOV the channels are recorded within 5-11 s; FOVs are about
3.6 min apart.

## Drying QC (fov_qc run_002, 2026-10-02)

Vesicles detected per FOV (10-frame mean, the detection of
Revisions/Bleedthrough/segment.py, SNR >= 6):

| Slide | Good FOVs (acquisition order) | Dried | Counts in good FOVs (405 / 488 / 515) |
|---|---|---|---|
| 1 | all 31 | - | 78-126 / 160-212 / - |
| 2 | all 31 | - | 156-193 / 45-77 / - |
| 3 | 1-21 (20:57-22:09) | 22-61, from 22:12 | 131-156 / 150-184 / 152-187 |
| 4 | 1-35 (10:58-13:00) | 36-60, from 13:04 | 100-130 / 70-105 / 82-117 |

- The drop is abrupt on both mixed slides (slide 3: FOV 21 has 135/167/171, FOV 22
  has 0/0/1; slide 4: FOV 35 has 105/81/86, FOV 36 has 0/0/17).
- Slide 3 FOVs 41-42 (23:20, 23:24) show 58/30/31 and 5/8/8: a brief remnant after
  drying, counted as dried.
- On dried slide 4 FOVs, 515 keeps 0-20 spots while 405 and 488 are at 0; probably
  residue, not vesicles.
- The data readme says slide 4 dried after about 25 FOVs; the counts put it at 35.
- Per-channel counts are not label-specific (each dye is also detected in the other
  channels when bright), so they only say whether vesicles are present.

Rule used by `move_dried.py` (config `dried`): a slide is dried from the first FOV
whose 405 + 488 count is below 0.1 x the median of the earlier FOVs (at least 5),
and every later FOV is dried too.

## Move to bad_data/ (done 2026-10-02)

At the user's request all raw files were unlocked (`chflags nouchg` on all 671
locked files, the dataset's own files plus one stray copy) and the 260 files of
the dried FOVs (slide 3 FOVs 22-61, slide 4 FOVs 36-60, 65 FOVs x 4 channels) were
moved with `move_dried.py --apply`. They keep their relative paths under
`bad_data/`; `bad_data/moved_files.csv` lists every file with its FOV's acquisition
order, time and counts, and `bad_data/README.md` explains the move.

Checked from the ND2 timestamps after the move: the kept files end at 22:08:55
(slide 3) and 13:00:21 (slide 4), the moved ones start at 22:12:24 and 13:03:50.
Remaining usable data: slide 3, 21 FOVs (84 files); slide 4, 35 FOVs (140 files);
slides 1 and 2 untouched.

The first attempt (before the lock check) had left one identical copy of
`HT390_SNAP520_405nm_TIRF2x_23pr_100ms_230x230_0020.nd2` in `bad_data/HT390_SNAP520/`;
it was renamed to `<name>.duplicate` (not deleted) and can be removed by the user.

The files are no longer locked. `fov_qc.py` now only sees the kept FOVs; the
reference counts for all FOVs stay in fov_qc run_002.

## Vesicles (vesicles run_002)

Detection per channel with the peak lists merged across channels (2.5 px). The
first version (run_001) used the per-pixel maximum of the channels' SNR maps, as in
Revisions/Bleedthrough, and missed 35-47 % of the 405 vesicles on the mixed slides:
a dim ATTO390 vesicle on the wing of a bright ATTO520 one in 515 is not a local
maximum of the combined map. After the fix at most 3.6 % of per-channel peaks lack
a detection within 2 px (pairs of peaks 2-2.5 px apart that were merged).

| Slide | FOVs | Vesicles | per FOV | crowded (< 1.04 um) | nonlinear |
|---|---|---|---|---|---|
| 1 | 31 | 5989 | 193 | 30 % | 32 % |
| 2 | 31 | 5765 | 186 | 30 % | 3 % |
| 3 | 21 | 6223 | 296 | 61 % | 13 % |
| 4 | 35 | 7305 | 209 | 45 % | 4 % |

Nonlinear (an aperture pixel >= 30,000 counts in some frame) is mostly bright ATTO520
vesicles in 488; it compresses their home flux by at most ~25 %, irrelevant for the labels.

## Labels (unmixing run_001)

Signatures (mean +- SD across the 31 FOVs of each single-label slide): ATTO390
puts 1.20 +- 0.19 % of its 405 flux into 488; ATTO520 puts 1.64 +- 0.07 % of its
488 flux into 405.

| Slide | ATTO390 | ATTO520 | dual | no label |
|---|---|---|---|---|
| 1 (ATTO520) | 54 (0.9 %) | 5870 (98.0 %) | 62 (1.0 %) | 3 |
| 2 (ATTO390) | 4938 (85.7 %) | 280 (4.9 %) | 514 (8.9 %) | 33 |
| 3 (mix) | 1739 (27.9 %) | 2697 (43.3 %) | 1783 (28.7 %) | 4 |
| 4 (mix) | 3267 (44.7 %) | 2445 (33.5 %) | 1127 (15.4 %) | 466 (6.4 %) |

- 515 check on the mixed slides: 99.5-100 % of ATTO520-labeled vesicles have 515
  signal (median 515/488 = 2.6 on slide 3, 2.9 on slide 4); ATTO390-labeled ones
  have almost none (median 515/405 below 0.7 %); dual ones have ATTO520-like 515.
- Dual is partly crowding (52-79 % of vesicles with a neighbor within 4 px are
  dual), but among isolated vesicles (no neighbor within 12 px) it is still 7 %
  on slide 2, 8 % on slide 4 and 18 % on slide 3. The 7 % on the pure ATTO390 slide
  matches the ATTO390 vesicles with an extra green emitter seen in the revision
  test; the extra ~10 % on slide 3 (imaged in the evening) would be vesicles with
  both labels. Interpretation (dye exchange, fusion) is open.
- The 280 'ATTO520' objects on the pure ATTO390 slide are all 488-only objects:
  contamination or impurities.
- The 466 'no label' objects on slide 4 are 515-only (5-25 noise SDs); slide 4's
  dried FOVs also kept 0-20 such spots per FOV without any vesicles, so they are
  most likely residue, not dim ATTO520 vesicles; 515 is therefore left out of the
  labels.

## Protein occupancy (occupancy run_003; scores and traces from run_001)

Method (agreed 2026-10-02): for every vesicle and blank, the 640 trace over all
6000 frames is the aperture sum (r = 3 px) minus n_aperture x the annulus median
(5-8 px) in each frame, which removes the FOV-wide drift of the 640 baseline.
Score A = (max - median) / MAD of the trace; score B = ON frames from
Extraction/utils.gmm_classify_frames (posterior 0.9, minimum separation 3 noise
SDs; without the separation guard the GMM splits pure noise). Each threshold lets
at most 1 % of all blanks pass: A >= 15.5, B >= 1 ON frame. Occupancy per FOV =
(p_v - p_b) / (1 - p_b), with p_b the slide's blank positive rate (blinks that
do not belong to a vesicle: protein on the glass, noise), mean +- SD across FOVs;
crowded vesicles (neighbor within 1.04 um) excluded.

Registration: the 640 signal is offset from the vesicle position by a chromatic
shift that depends on the channel the position came from, measured from the ON
frames of clearly positive vesicles (pooled over slides, consistent between
slides): 405 (0.23, -1.14) px, 488 (0.21, -0.34) px, 515 (0.26, -0.55) px (dy, dx);
applied per vesicle.

| Vesicles (protein) | Slide | n (FOVs) | Occupancy A | Occupancy B | Raw positive / blanks | Incl. crowded (A) |
|---|---|---|---|---|---|---|
| ATTO520 (SNAP) | 1, alone | 4119 (31) | 8.4 +- 3.3 % | 6.1 +- 2.8 % | 10.0 % / 1.8 % | 8.7 +- 3.5 % |
| ATTO390 (HT7) | 2, alone | 3590 (31) | 6.3 +- 2.8 % | 5.1 +- 2.6 % | 7.0 % / 0.8 % | 6.7 +- 2.6 % |
| ATTO390 (HT7) | 3, mix | 840 (21) | 7.0 +- 5.9 % | 5.6 +- 5.2 % | 7.9 % / 1.0 % | 6.9 +- 4.5 % |
| ATTO520 (SNAP) | 3, mix | 1206 (21) | 3.3 +- 2.9 % | 2.4 +- 2.1 % | 4.3 % / 1.0 % | 3.7 +- 2.0 % |
| ATTO390 (HT7) | 4, mix | 1955 (35) | 2.5 +- 2.7 % | 1.7 +- 2.0 % | 3.0 % / 0.5 % | 3.0 +- 2.2 % |
| ATTO520 (SNAP) | 4, mix | 1450 (35) | 3.0 +- 3.2 % | 1.9 +- 2.4 % | 3.4 % / 0.5 % | 2.9 +- 2.3 % |

- Occupancy rises with vesicle brightness (a proxy for size): dim / middle /
  bright tertiles 4.5 / 8.0 / 12.5 % (slide 1) and 2.4 / 5.4 / 11.2 % (slide 2).
- Threshold sensitivity (A from 8 to 30): slide 1 10.3 -> 5.8 %, slide 2 7.5 ->
  4.5 %, mixed slides 1.0-7.6 %; the ranking between groups does not change.
- B, which needs a separated ON population, gives 70-80 % of A: single sparse
  blinks are missed by the GMM. Audit: of 1122 vesicle and 858 blank traces below
  B's pre-screen (A < 7), none had an ON frame.
- Dual vesicles are occupied more often (8-26 %), consistent with their being
  larger objects or overlaps.
- Excluding crowded vesicles (30-61 % of them) changes nothing beyond 0.5 points.

Not verified / limits: one slide per condition; FOVs are repeated measurements of
it. A protein that never blinks or bleaches early in the 3 min is missed, so these
are lower bounds on occupancy. Protein identity comes only from the vesicle label.
### Check (2026-10-02, two agents)

- Independent recomputation from the saved scores (without reading occupancy.py):
  thresholds, every positive call and every cell of the per-FOV and summary
  tables reproduced exactly; score A recomputed from the traces of 3 FOVs agrees
  to float32 precision; traces recomputed from the raw 640 movies of 2 FOVs
  (about 45 vesicles, 40 blanks) are identical to the saved ones, and differ by up
  to 91 % without the registration offset.
- Code review of vesicles.py, unmix.py, occupancy.py and the FOV pairing: no bug
  that changes the current numbers. It confirmed the pairing by colocalization
  (slide 1 FOV10: 11.4 % of vesicles positive in their own 640 movie against 4.0 %
  and 3.5 % in the previous and next FOV's movies; slide 3 FOV05: 5.1 % against
  1.7 %). Fixed afterwards: FOVs without vesicles now still contribute their blanks;
  --rescore refuses to run if the scoring parameters differ from the source run;
  a clear error if a trace window would cross the image edge (the current margin
  is exactly zero). Not fixed, documented: the +-3 px registration window
  underestimates the offsets by about 0.2 px (positives change by at most 1 per
  FOV); the pairing relies on the ND2 'date' stamps clustering per FOV.
- Not verified: score B was recomputed for 20 traces only; detection and snapshot
  photometry were not redone from raw data in the check; slides 1 and 3 raw traces.

Run history: occupancy run_001 (scores, traces; figure with impurity groups),
run_002 (rescore, fixed figure), run_003 (rescore with the review fixes,
identical tables to run_002, reference). vesicles run_001 (max-map detection,
missed 405 vesicles in the mixes) superseded by run_002.
