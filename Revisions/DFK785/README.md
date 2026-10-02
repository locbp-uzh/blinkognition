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

# Paper pipeline on the 640 movies (from Extraction/, picasso-env)
python run_pipeline.py -c ../Revisions/DFK785/extraction/config_slide1_SNAP520.yaml   # paper ground-truth mode
python run_pipeline.py -c ../Revisions/DFK785/extraction/config_slide2_HT390.yaml
python run_pipeline.py -c ../Revisions/DFK785/extraction/config_640only.yaml          # all slides, no ground truth
python Revisions/DFK785/assign_traces.py    # traces of config_640only -> vesicle labels
python Revisions/DFK785/brightness.py       # photons per localization, paper vs DFK785
python Revisions/DFK785/overview.py         # usable FOVs, crowding, provenance of the filtered traces
```

Outputs in `Results/Revisions/DFK785/<stage>/run_NNN/` with manifest, effective
config and code copy. The ND2 metadata here does not record per-channel settings:
every file of a slide carries the same microscope state (slides 1-2: 515 laser and
the dual 442/514 filter, slides 3-4: 405 laser and Continuous STORM, for 405, 488
and 640 files alike), so settings come from the file names and the data readme.

## FOV pairing

FOVs are matched by acquisition time (ND2 metadata), not by file number: on slide 3
the numbering is shifted between channels (the unnumbered file counts as 0: FOV01 is
405/488/640 file _0031 with 515 file 0, FOV02 is 640 _0032 with 405/488 0 and 515 _0001,
and from FOV03 on 515 file N goes with 405/488 file N-1 and 640 file N-2). Colocalization
of the 640 ROIs with the vesicles confirms this pairing (0.53 of single ROIs within 4 px
of a vesicle against 0.29 with the neighboring FOVs' vesicles) for every slide 3 FOV
except FOV01 (640 _0031), whose ROIs are at chance level for a reason not found; it
contributes no filtered IN_ATTO390 or IN_ATTO520 trace (check of 2026-10-02 below). Within every FOV the channels are recorded within 5-11 s; FOVs are about
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

## Protein occupancy (occupancy run_004; scores and traces from run_001)

Method (agreed 2026-10-02): for every vesicle and blank, the 640 trace over all
6000 frames is the aperture sum (r = 3 px) minus n_aperture x the annulus median
(5-8 px) in each frame, which removes the FOV-wide drift of the 640 baseline.
Score A = (max - median) / MAD of the trace; score B = ON frames from
Extraction/utils.gmm_classify_frames (posterior 0.9, minimum separation 3 noise
SDs; without the separation guard the GMM splits pure noise). Each threshold lets
at most 1 % of all blanks pass: A >= 15.5, B >= 1 ON frame. Occupancy per FOV =
(p_v - p_b) / (1 - p_b), with p_b the slide's blank positive rate (blinks that
do not belong to a vesicle: protein on the glass, noise), mean +- SD across FOVs;
vesicles with a neighbor closer than 5 px (0.65 um) excluded (decision 2026-10-02;
first 8 px = 1.04 um, see "Crowding" below).

Registration: the 640 signal is offset from the vesicle position by a chromatic
shift that depends on the channel the position came from, measured from the ON
frames of clearly positive vesicles (pooled over slides, consistent between
slides): 405 (0.23, -1.14) px, 488 (0.21, -0.34) px, 515 (0.26, -0.55) px (dy, dx);
applied per vesicle.

| Vesicles (protein) | Slide | n (FOVs) | Occupancy A | Occupancy B | at 8 px (n) |
|---|---|---|---|---|---|
| ATTO520 (SNAP) | 1, alone | 5781 (31) | 8.5 +- 3.6 % | 6.2 +- 2.8 % | 8.4 +- 3.3 % (4119) |
| ATTO390 (HT7) | 2, alone | 4800 (31) | 6.5 +- 2.6 % | 5.3 +- 2.6 % | 6.3 +- 2.8 % (3590) |
| ATTO390 (HT7) | 3, mix | 1640 (21) | 6.7 +- 4.7 % | 5.0 +- 3.5 % | 7.0 +- 5.9 % (840) |
| ATTO520 (SNAP) | 3, mix | 2363 (21) | 3.4 +- 2.1 % | 2.4 +- 1.6 % | 3.3 +- 2.9 % (1206) |
| ATTO390 (HT7) | 4, mix | 2989 (35) | 3.1 +- 2.4 % | 2.1 +- 1.6 % | 2.5 +- 2.7 % (1955) |
| ATTO520 (SNAP) | 4, mix | 2230 (35) | 2.9 +- 2.5 % | 1.8 +- 2.0 % | 3.0 +- 3.2 % (1450) |

Blank positive rates (chance hits) for A: slide 1 1.8 %, slide 2 0.8 %, slide 3 1.0 %,
slide 4 0.5 %.

- Occupancy rises with vesicle brightness (a proxy for size): dim / middle /
  bright tertiles 4.5 / 8.0 / 12.5 % (slide 1) and 2.4 / 5.4 / 11.2 % (slide 2).
- Threshold sensitivity (A from 8 to 30): slide 1 10.3 -> 5.8 %, slide 2 7.5 ->
  4.5 %, mixed slides 1.0-7.6 %; the ranking between groups does not change.
- B, which needs a separated ON population, gives 70-80 % of A: single sparse
  blinks are missed by the GMM. Audit: of 1122 vesicle and 858 blank traces below
  B's pre-screen (A < 7), none had an ON frame.
- Dual vesicles are occupied more often (8-26 %), consistent with their being
  larger objects or overlaps.
- Excluding crowded vesicles changes little: with all vesicles included, or with
  the 8 px cutoff (30-61 % excluded), occupancy moves by at most 0.6 points.

### Crowding

A vesicle is measured in a 3 px aperture with its background from a 5-8 px
annulus. For a neighbor at distance d (PSF sigma 1.6 px), the share of its light in
this vesicle's aperture / annulus is 50 % / 9 % at 2.5 px, 19 % / 32 % at 4 px,
7 % / 53 % at 5 px, 2 % / 65 % at 6 px and 0 % / 43 % at 8 px. Below 2.5 px two
vesicles are merged into one object (unresolvable); at 2.5-5 px they are resolved
but a neighbor's blink or label leaks into the aperture (false protein hits,
'dual' labels); at 5-8 px the aperture is clean and the neighbor only enters the
annulus, whose median is robust to it. Hence the 5 px cutoff for occupancy
(2 / 5 / 20 / 15 % of vesicles excluded on slides 1-4, against 30 / 30 / 61 / 45 % at
8 px). The signatures (unmix.py) still use the 8 px flag. With the radius set back
to 8 px the code reproduces run_003 exactly.

### Confident detections (absolute numbers, 5 px, occupancy run_004)

`confident_detections.csv` in the run. Standard: score A >= 15.5 in a vesicle with
its own label (not dual), not crowded. Strict: A >= 31 and B >= 1 ON frame, own label
>= 10 noise SDs with < 1 SD of the other dye, not crowded, not nonlinear. 'Chance' is
the number of hits the slide's blank rate predicts at the same criterion. On the
single-label slides the label is verified by the slide; on the mixed slides it is
the unmixing assignment.

| Tier | Protein | Detections | Chance | Net | single-label slide (chance) |
|---|---|---|---|---|---|
| standard | SNAP (ATTO520) | 770 | 139 | 631 | 591 (104) |
| standard | HT7 (ATTO390) | 582 | 68 | 514 | 348 (37) |
| strict | SNAP (ATTO520) | 145 | 20 | 125 | 115 (15) |
| strict | HT7 (ATTO390) | 187 | 15 | 172 | 146 (11) |

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
  FOV; 0.33 px in x for 405, measured later from the trace-assignment residuals); the pairing relies on the ND2 'date' stamps clustering per FOV.
- Not verified: score B was recomputed for 20 traces only; detection and snapshot
  photometry were not redone from raw data in the check; slides 1 and 3 raw traces.

Run history: occupancy run_001 (scores, traces; figure with impurity groups),
run_002 (rescore, fixed figure), run_003 (rescore with the review fixes,
identical tables to run_002), run_004 (rescore with the 5 px occupancy crowding
radius, reference). vesicles run_001 (max-map detection,
missed 405 vesicles in the mixes) superseded by run_002.

## Paper-pipeline extraction (2026-10-02)

The 640 movies go through Extraction/run_pipeline.py unchanged (Picasso MLE, linking,
5 x 5 ROI, GMM background, minmax, the paper's filters), so the traces are the kind
the classifier was trained on. Configs in `extraction/` are copies of
Extraction/config.yaml with only the input/output folders, protein names and ground-truth
channel changed. Inputs are staged as per-file links in `Inputs/` (gitignored):
`Inputs/DFK785_640only/link_map.tsv` maps each renamed link to its raw file. Two fixes
to shared code were needed: run_pipeline.py found its step scripts only next to the
config (6823f86), and Extraction/utils.py `rel_under` resolved symlinks, so extract.py
looked for localizations where localize.py had not written them.

| Run (Results/Revisions/DFK785/Extraction/) | Slides | Mode | Optimized gradient 640 / GT |
|---|---|---|---|
| SNAP520_optparam_001 | 1 | ground truth 488 | 20000 / 7000 |
| HT390_optparam_001 | 2 | ground truth 405 | 20000 / 1000 |
| Slide1SNAP_Slide2HT_Slide3Mix_Slide4Mix_optparam_001 | 1-4 | 640 only | 20000 |

Paper ground-truth mode, filtered IN traces: slide 1 SNAP 629 of 953 (66 %), slide 2
HT 440 of 576 (76 %). The single-ROI sets of the 640-only run match these runs in size
(1134 and 690 single ROIs on slides 1 and 2) and box for box on slide 2; on slide 1 two
boxes differ each way (movies 0017, 0028, 0030) and uniqueIDs are not aligned between
the runs (681 of 1126 shared IDs point to the same box), so cross-match by movie and
top-left corner, never by uniqueID. Slides 3 and 4 add 575 and 476.

Artifacts of the extraction (both runs): 54 single ROIs at (0, 0) or with a negative
corner (16 / 16 / 8 / 14 on slides 1-4), all with empty traces and none filtered, and
identical boxes in the same movie that are labeled single instead of overlapping (slide 2:
uniqueIDs 280/301 and 703/731, both pairs filtered, and 574/594 at (0, 0); slide 4: 192/199
at (0, 0)). Root cause not determined (get_overlapping_rois and the index-0 dummy of
get_and_link_locs in Extraction/utils.py are the suspects). The paper's own filtered IN
sets carry exact duplicate traces too: HaloD106 2 of 2580, HaloK117 1 of 2384, SNAPC148 0
of 9647, scGrx1AcK20 2 of 1260, scGrx1 87 of 3400 (not investigated further).

## Brightness vs the paper (brightness run_002)

Same Picasso settings and one gradient (20000) for all movies. Paper: three 640 movies
(_0005, _0010, _0020) per protein in each of SP_Exp1-4, 12 per protein (AllMovies; HTHTL =
HaloD106, snap = SNAPC148); DFK785: all 31 movies of slides 1 and 2. Photons per
localization, pooled median; ratio to the paper set of the same protein in the same frame
window.

| Set | Frames | Photons / loc | Ratio | Movie medians | Background / px | Ratio bg |
|---|---|---|---|---|---|---|
| paper HaloD106 | 0-6000 | 2826 | 1 | 1480-3318 | 114 | 1 |
| DFK785 slide 2 HT | 0-6000 | 1095 | 0.39 | 895-1458 | 67 | 0.59 |
| paper HaloD106 | 0-1000 | 3113 | 1 | 1660-3453 | 116 | 1 |
| DFK785 slide 2 HT | 0-1000 | 1133 | 0.36 | 869-2044 | 67 | 0.58 |
| paper HaloD106 | 3000-6000 | 1811 | 1 | 1275-2524 | 109 | 1 |
| DFK785 slide 2 HT | 3000-6000 | 1069 | 0.59 | 910-1614 | 68 | 0.62 |
| paper SNAPC148 | 0-6000 | 1526 | 1 | 1399-1952 | 108 | 1 |
| DFK785 slide 1 SNAP | 0-6000 | 1222 | 0.80 | 1060-1424 | 72 | 0.67 |
| paper SNAPC148 | 3000-6000 | 1409 | 1 | 572-4757 | 108 | 1 |
| DFK785 slide 1 SNAP | 3000-6000 | 1139 | 0.81 | 952-1596 | 71 | 0.66 |

run_001 used one paper movie (_0005) per experiment and frames 0-6000 only (HT 0.35, SNAP
0.79); superseded. Findings, from run_002 and the independent check of 2026-10-02
(checks/20261002_trace_assignment_verification/verify_brightness):
- Background at 0.6-0.7x in every window is consistent with the readme's weaker 640 laser
  (12 mW), but localizations cannot show it, and the ND2 files record 90 % and no power.
- The paper movies front-load bright, short-lived sites: 51 % of paper HT localizations fall
  in frames 0-500 (DFK785 27 %); paper HT sites first seen before frame 1000 have a median of
  2765 photons, later ones 1509. No site dims within itself (late / early 0.97 in all sets),
  and DFK785 has no early bright population. At frames 3000-6000 the ratios are 0.59 (HT)
  and 0.81 (SNAP), the paper's HT / SNAP is 1.29x (1.85x over all frames), and the per-movie
  ranges overlap. The cause of the missing early sites (640 pre-exposure while focusing, more
  proteins per vesicle in the paper samples, or else) cannot be told from the localizations.
- Overlapping spots explain none of it (under 2 % of localizations have a same-frame
  neighbor within 12 px).
- A fixed gradient drops the dimmest events: DFK785 sits closer to the cut (13.6 % of HT
  localizations within 10 % of it, paper 3.3 %), so its surviving median is pushed up; a
  scaling model puts the true ratios nearer 0.30 (HT) and 0.50 (SNAP). Direction solid,
  size model-dependent.
HT lost more than SNAP in every frame window, so the laser alone does not explain it.
Minmax normalization removes absolute brightness but not the noise relative to the blink
amplitude: the paper-trained model sees noisier HT traces than it was trained on. Not checked: the trace level (the paper's FinalTraces keep only
minmax and zscored traces).

## Trace assignment (trace_assignment run_003, 2 px)

assign_traces.py places each single ROI of the 640-only run on the vesicle table (unmixing
run_001 labels, positions shifted into the 640 frame by the occupancy run_004
registration). Protein position = ROI top-left + 2.5: the extraction sets top-left =
round(x - 2.5) on Picasso coordinates, which, like the vesicle centroids, put pixel centers
on integers. IN_<label>: nearest vesicle within 2 px and no second one; ambiguous: two
within 2 px. The radius is 2 px, not the paper's 4 px (user decision, 2026-10-02): in the
mixture a protein must not be assigned to the wrong vesicle, and at 4 px 30-53 % of the
mixed-slide IN calls are chance (below), while nearly all real pairs lie within 2.5 px. The
paper's own training and test data keep Extraction's 4 px and are not changed: there no
vesicle of the wrong kind exists, and the training needs the data. The artifact ROIs and the repeated slide 2 boxes above are
dropped first (dropped.csv: 54 artifacts, 2 duplicates, the 2 duplicates filtered).
Residual protein - vesicle offset of IN pairs: under 0.1 px for 488 and 515 vesicles,
+0.09 / -0.33 px (y / x) for 405 vesicles, the 405 registration underestimate.

Superseded runs: run_001 (top-left + 2, which put every protein 0.5 px up-left of its
localization, residuals -0.4 / -0.5 px, and kept the artifacts; 3 % of the ROIs change
class with the fix) and run_002 (fixed, 4 px; filtered IN_ATTO390 / IN_ATTO520 on slides 1-4:
4 / 377, 223 / 19, 63 / 67, 56 / 42).

| Slide | Set | IN_ATTO390 (HT) | IN_ATTO520 (SNAP) | IN_dual | IN_no label | ambiguous | OUT |
|---|---|---|---|---|---|---|---|
| 1 SNAP | single ROIs | 5 | 444 | 8 | 0 | 0 | 661 |
| 2 HT | single ROIs | 269 | 20 | 75 | 3 | 2 | 303 |
| 3 mix | single ROIs | 71 | 66 | 74 | 0 | 2 | 354 |
| 4 mix | single ROIs | 63 | 47 | 25 | 5 | 0 | 322 |
| 1 SNAP | filtered | 4 | 310 | 6 | 0 | 0 | 422 |
| 2 HT | filtered | 201 | 18 | 66 | 3 | 1 | 221 |
| 3 mix | filtered | 44 | 48 | 50 | 0 | 0 | 252 |
| 4 mix | filtered | 45 | 29 | 17 | 4 | 0 | 215 |

Mixed slides, filtered, per FOV: slide 3 (21 FOVs) 2.1 HT and 2.3 SNAP, slide 4 (35 FOVs)
1.3 HT and 0.8 SNAP. Slide 3 FOV01 (chance-level colocalization, see FOV pairing)
contributes one filtered IN_dual trace and nothing else; it is kept.

Label errors on the single-label slides, filtered IN traces: on slide 2 (all ATTO390)
6.3 % sit in vesicles labeled ATTO520 and 23 % in dual ones (9 % of all slide 2 vesicles
are dual); on slide 1 (all ATTO520) 1.3 % ATTO390 and 1.9 % dual. The unmixing threshold
is still to be tuned.

The OUT class includes traces in the 10 px border band where the table has no vesicles
(filtered: 95 of 422 OUT on slide 1, 57 / 221 on slide 2, 52 / 252 on slide 3, 44 / 215 on
slide 4; center within 10 px of the edge), so OUT is not "outside a vesicle".

### Chance coincidences (null.csv)

The same filtered ROIs are assigned to the vesicles of the previous and next FOV of the
slide. Model: a fraction f of ROIs is in a vesicle and always near one; the rest lie
independently of the vesicle pattern and are near one with the null probability q. Then
observed near fraction = f + (1 - f) q, and the expected chance hits of a class are
(1 - f) x its null count. On a mixed slide a chance hit carries the label of a random
vesicle, whatever the protein is.

| Slide | Radius | HT obs | HT chance | SNAP obs | SNAP chance |
|---|---|---|---|---|---|
| 3 | 2.0 px | 44 | 4.8 (11 %) | 48 | 8.6 (18 %) |
| 3 | 2.5 px | 50 | 8.6 (17 %) | 58 | 13.5 (23 %) |
| 3 | 4.0 px | 63 | 21.1 (34 %) | 67 | 35.4 (53 %) |
| 4 | 2.0 px | 45 | 3.6 (8 %) | 29 | 4.7 (16 %) |
| 4 | 2.5 px | 47 | 7.8 (17 %) | 29 | 6.3 (22 %) |
| 4 | 4.0 px | 56 | 16.8 (30 %) | 42 | 15.0 (36 %) |

f reaches its plateau at 2.5 px on every slide (0.42, 0.55, 0.34, 0.26-0.29 on slides 1-4)
and is 92-97 % of its 4 px value at 2 px, so nearly all real pairs lie within 2.5 px and
the paper's 4 px adds mostly coincidences. Obs - chance, the real vesicle-assigned traces,
is about the same at every radius: about 80 HT and 59-67 SNAP on slides 3 and 4 together
(2 px: 81 HT and 64 SNAP of 89 and 77 called).
On the single-label slides the chance share at 4 px is 20 % (slide 1) and 13 % (slide 2);
there it does not change the label. Assumptions not tested: proteins outside table
vesicles are placed independently of the vesicle pattern; dim vesicles suppressed next
to a bright one (crowded: 61 % on slide 3) break that, so the chance estimate is a lower
bound. Chance hits and unmixing errors add up.

### Paper ground-truth mode against the table

Cross-matched by movie and top-left corner (run_002, artifacts and duplicates dropped),
the table calls fewer ROIs IN than the paper's ground-truth mode: slide 1 553 against 943,
slide 2 406 against 570. Of the 402 slide 1 ROIs that are paper-IN but not table-IN, 113
lie in the 10 px border band where the table has no vesicles (slide 2: 54 of 183). The
rest, from the independent check of 2026-10-02 (all 31 FOVs, aperture photometry with an
annulus background, Picasso re-clustering):
- The paper's ground-truth clusters are 2.6x (slide 1: 15538 against 5989 table vesicles)
  and 3.3x (slide 2: 19046 against 5765) more numerous. 95-99 % of table vesicles have a
  cluster within 2 px (99.9 % of those detected in the ground-truth channel itself).
  Clusters within 4 px cover 39 % (slide 1) and 47 % (slide 2) of the FOV, the table's
  IN zones 18 % and 17 %.
- Slide 2 (405, gradient 1000, the lower bound of the paramfinder search range): about two
  thirds of the 13558 unmatched clusters are noise (the 9107 interior ones below table SNR
  6 have a localization in 4 of 10 frames and held-out-frame SNR -0.9, median). About 1800
  are real objects suppressed by a brighter neighbor within 3 px, and the 2365 in the border
  band are mixed. Paper-mode IN is only slightly inflated: 74 % of the 121 interior ROIs that
  are paper-IN but table-OUT sit on real 405 signal (SNR above 3, against 15 % for random
  positions of the same class), so about 40-60 of the 576 paper-mode IN calls (7-10 %) are
  chance; most of the gap is dim ATTO390 objects the table drops (border, SNR 6, neighbor
  suppression).
- Slide 1 (488, gradient 7000): the unmatched clusters are real, dimmer 488 objects (10 of
  10 frames, 3.7x fewer photons, aperture SNR about 14-36). 72 % of the interior ones fall
  below the table's SNR threshold of 6 (measured against the clipped spread of the filtered
  map, not pixel noise), 28 % are suppressed by a brighter neighbor (min_separation_px 3).
- data_rois.csv stores the clusters as box top-left corners, round(center - 2.5): add 2.5,
  not 2, to compare them with anything else.

The border and the SNR threshold of the vesicle table are kept as they are (user decision,
2026-10-02). Not checked: whether the dim 405 objects at slide 2 protein positions are
ATTO390 vesicles or protein signal leaking into the 405 snapshot (their 405/488 ratio fits
ATTO390); the paper's own SP_Exp1-4 ground-truth calls (gradients 1000-5000); slides 3-4.

### Check (2026-10-02, five agents)

Workflow of four independent verifiers and a critic; scripts, outputs and the full
result in `Results/Revisions/DFK785/checks/20261002_trace_assignment_verification/`.
- counts.csv and residual_offset.csv of run_001 recomputed from scratch without reading
  assign_traces.py: all 48 cells identical. uniqueIDs are unique per extraction key and
  every filtered trace correlates r = 1.000 with its stored raw trace. The code review
  predicted the run_002 counts with the center fix; run_002 matches them.
- Code review: the center bias (fixed in run_002); registration sign and frame correct;
  FOVs with 0, 1 or several vesicles handled; the rel_under change gives identical
  results for absolute, relative, trailing-slash and symlinked-root inputs and fixes the
  symlinked-file case; brightness.py reproduces from the locs files and its parameter
  check rejects a wrong gradient, box size or fit method.
- Refutation attempts on the ground-truth-cluster claims and the brightness comparison:
  results folded into the sections above.
- Not verified: the vesicle table itself (detection, labels) on slides 3-4; the protein
  identity of OUT traces; the final model.

## Data overview (overview run_001, 2 px)

| Slide | Experiment | FOVs acquired | Dried (bad_data/) | Usable |
|---|---|---|---|---|
| 1 | pure: SNAP in ATTO520 | 31 | 0 | 31 |
| 2 | pure: HT in ATTO390 | 31 | 0 | 31 |
| 3 | mixed | 61 | 40 | 21 |
| 4 | mixed | 60 | 25 | 35 |

118 usable FOVs: 62 pure, 56 mixed.

Crowding (vesicle table; nn = nearest other vesicle):

| Slide | Vesicles | Per FOV (median, range) | nn median | nn < 5 px | nn < 8 px | ATTO390 / ATTO520 / dual / none |
|---|---|---|---|---|---|---|
| 1 | 5989 | 193 (164-218) | 9.3 px | 1.6 % | 30 % | 1 / 98 / 1 / 0 % |
| 2 | 5765 | 186 (173-210) | 9.4 px | 4.5 % | 30 % | 86 / 5 / 9 / 1 % |
| 3 | 6223 | 297 (275-313) | 7.3 px | 20 % | 61 % | 28 / 43 / 29 / 0 % |
| 4 | 7305 | 210 (184-223) | 8.4 px | 15 % | 45 % | 45 / 34 / 15 / 6 % |

Filtered traces by provenance (trace_assignment run_003). clean: in a single-dye vesicle
with no other vesicle within 5 px; crowded: the same with a neighbor within 5 px; wrong
label: pure slides only, a vesicle labeled with the other dye; dual: a vesicle labeled with
both dyes; ambiguous: two vesicles within 2 px; OUT border: no vesicle, within the 10 px band
the table does not cover.

| Slide | Protein | clean | crowded | wrong label | dual | no label | ambiguous | OUT border | OUT interior | total |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 pure | SNAP | 302 | 8 | 4 | 6 | 0 | 0 | 95 | 327 | 742 |
| 2 pure | HT | 192 | 9 | 18 | 66 | 3 | 1 | 57 | 164 | 510 |
| 3 mixed | HT (ATTO390) | 42 | 2 | - | - | - | - | - | - | 44 |
| 3 mixed | SNAP (ATTO520) | 42 | 6 | - | - | - | - | - | - | 48 |
| 3 mixed | unknown | - | - | - | 50 | 0 | 0 | 52 | 200 | 302 |
| 4 mixed | HT (ATTO390) | 42 | 3 | - | - | - | - | - | - | 45 |
| 4 mixed | SNAP (ATTO520) | 25 | 4 | - | - | - | - | - | - | 29 |
| 4 mixed | unknown | - | - | - | 17 | 4 | 0 | 44 | 171 | 236 |

On the pure slides the protein is known for every trace; the vesicle only says whether it is
encapsulated. On the mixed slides only the single-dye classes identify the protein.

Crowding goes into the dual label: on the mixed slides 45-47 % of dual vesicles have a
neighbor within 5 px, against 6-12 % of single-dye ones, so two different vesicles closer
than 5 px read as one dual vesicle. Within each single-dye label the crowded share of the
traces is what chance predicts (binomial CDF 0.46-0.89). On slide 2 most dual vesicles are
not crowded (86 % have nn >= 5 px): they are ATTO390 vesicles with real 488 signal (cf. the
~7 % with an ATTO525-like extra emitter in Revisions/Bleedthrough). Vesicles holding more
than one protein ROI never appear here: the extraction drops overlapping ROIs.

Expected wrong labels on the mixed slides (mixed_expected_errors.csv): chance coincidences
(null.csv, an upper bound on wrong: a chance hit is right if the protein happens to match)
plus the other protein's traces in mislabeled vesicles, at the single-label slides' rates
(HT traces in ATTO520-labeled vesicles: 18 / 201 = 9.0 % of correct ones; SNAP in ATTO390:
4 / 310 = 1.3 %).

| Slide | Label | Traces | Chance | Unmixing | Purity (min) |
|---|---|---|---|---|---|
| 3 | HT | 44 | 4.8 | 0.5 | 88 % |
| 3 | SNAP | 48 | 8.6 | 3.5 | 75 % |
| 4 | HT | 45 | 3.6 | 0.3 | 91 % |
| 4 | SNAP | 29 | 4.7 | 3.7 | 71 % |

The unmixing rates come from slides with 30 % crowding (8 px), the mixed slides have 45-61 %,
and the chance estimate is itself a lower bound under crowding: the purities are estimates,
not bounds. Check: an independent recompute without reading overview.py matched every cell
of the four tables (checks/20261002_trace_assignment_verification/verify_overview). The
labels.csv column 'crowded' uses 8 px, the overview's category 5 px.

## Final model (S3IT, 2026-10-02)

The paper reports only cross-validation, which saves no model. The Fig. 4C setting
(CNN-GRU, minmax, mirror augmentation, 70/15/15, 6000 frames, MC dropout 100) is
retrained on the paper's FinalTraces with `ml/config_train_HaloD106_SNAPC148.yaml`,
submitted from a clone of the revisions branch at `~/blinkognition_revisions` (commit
6823f86; FinalTraces rsynced to `Inputs/FinalTraces`) as job 6823630, outputs in
`~/blinkognition_revisions/Results/Train/`.
