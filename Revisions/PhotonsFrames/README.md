# PhotonsFrames: photons per ON event and frame-time effects (R3.3)

Reviewer 3, point 3: Halo-K117's mean ON time (35 ms) is close to the frame time, so events may be
split across frames, lose SNR and be discarded by the SNR < 2 gate; show photons per ON event against
ON time, and report photon numbers. Plan and decisions in `../ReviewRound1/README.md`.

Parts: a) photons per event against ON time and photon numbers (done); b) which filter criterion
removes which traces, SNR gate isolated (next); c) optional simulation of short events against the
frame timing (needs the camera mode).

## Files

- `config.yaml`: event settings (the paper's: posterior >= 0.9, runs of >= 1 frame), padding,
  inner-frame rule, output root.
- `events.py`: every ON event of every published filtered trace in vesicles, with photons
  (`events.csv.gz`) and per-molecule summaries (`molecules.csv`); reused by R3.2.
- `check_fig5d6c.py`: the event definition against the published Figs. 5D and 6C.
- `figure_R3.3.py` (first `figure_R3.3a.py`): figure R3.3 from an events run (`--build --events <run> --out <folder>`), or
  replot from the CSVs next to it.
- `filter_audit.py` (+ `s3it/filter_audit.sbatch`): replay of the paper's quality filter with the
  first failing criterion per trace, IN or background traces, local mirror or S3IT (`--remote`).
- `bright_frames.py` (+ `s3it/bright_frames.sbatch`): frames above median + k sigma per unfiltered
  trace, IN and background.
- `molecule_like.py`: session thresholds from background, molecule-like flags for SNR-rejected
  traces. `flagged_features.py`: events of the flagged traces against the passed ones.
- `figure_R3.3b.py`: figure R3.3b from those runs (`--build`, defaults to the reference runs), or
  replot from the CSVs next to it. Figure folder
  `/Volumes/MediumBerth/SalomePaperData/Püntener2026/revisions_figures/R3.3b/`.

Outputs: `Results/Revisions/PhotonsFrames/events/run_NNN/`, `check_fig5d6c/run_NNN/` (manifests);
figure folder `/Volumes/MediumBerth/SalomePaperData/Püntener2026/revisions_figures/R3.3a/`.

```bash
/Users/privera/opt/anaconda3/envs/locbplots/bin/python Revisions/PhotonsFrames/events.py
/Users/privera/opt/anaconda3/envs/locbplots/bin/python Revisions/PhotonsFrames/figure_R3.3.py --build --events Results/Revisions/PhotonsFrames/events/run_002 --out "/Volumes/MediumBerth/SalomePaperData/Püntener2026/revisions_figures/R3.3"
```

## R3.3a choices (confirmed by the user 2026-11-20)

All filtered traces in vesicles; the paper's ON events; photons summed over each event (and with
one extra frame on each side as a check); duration = frames x 34.49 ms (an n-frame event lasted
between n - 1 and n + 1 frames); full-frame emission from inner frames of events >= 3 frames;
medians and IQR, per molecule first, then over molecules (events nested in molecules nested in
sessions); per-session values; Halo-D106 vs Halo-K117 only within the shared sessions; no tests.

## Log

- 2026-11-20, `check_fig5d6c` run_001: the event definition reproduces every published value of
  Figs. 5D and 6C (mean OFF, mean ON, duty cycle; 262 + 248 and 59 + 25 traces; max difference
  4e-12 after sorting), and the min-max input of the final models equals the published intensities'
  min-max over 6000 frames, first 4000 frames (3e-8, float32).
- 2026-11-20, `events` run_001: 19,271 molecules, 447,930 events, 6 min on 16 cores.
- 2026-11-20, figure R3.3a. A first build placed histogram cells by their lower photon edge after a
  CSV round trip and shifted some by one bin (white gaps); the CSV now carries the bin index.
- 2026-11-20, `filter_audit` run_001 (local): the replay of the paper's filter passed 2 traces the
  published run rejected (1 Halo-D106, 1 scGrx1-AcK20, both far from every threshold). Cause:
  `Extraction/filter.py` lines 212-215 drop traces whose z-scored version has an exact 0 before the
  cascade (4 traces in all have a frame exactly at their mean). Added as criterion `zscore_zero`.
- 2026-11-20, `filter_audit` run_002 (local): reproduces the published filtered set exactly for all
  five constructs. Caveat: it ran commit 2896d13, but its manifest names 4803d3f (the script was
  committed while it ran; the change only added options, the IN replay is the same). The S3IT run
  (job 8602367, original environment) is the reference.
- 2026-11-20, S3IT `filter_audit` run_001 (job 8602367, picasso-env, sklearn 1.7.2), the reference:
  reproduces the published filtered set exactly; identical criteria and separations to the local
  run_002 (sklearn 1.8.0). Fetched to `Results/Revisions/PhotonsFrames/s3it/filter_audit/run_001`.
- 2026-11-20, S3IT `filter_audit_background` run_001 (job 8602368): the same cascade on all unfiltered
  background traces (random positions >= 5 px from protein ROIs, same movies). 97.5-99.9 % have
  separation < 2 and 0.1-2.1 % pass the whole filter per session, except scGrx1 Exp5 (8.0 % pass,
  15 % separation >= 2): that slide carried blinking emitters away from detected proteins. Sessions
  attached from step 0 run_003 (the audit's own rows predate it).
- Same-day separation of IN traces: Halo-K117 below 2 in 52.6 % (Exp1) and 34.7 % (Exp2) against
  Halo-D106 11.4 % and 20.9 %; in the 2-3 band 4.7 / 3.8 % against 0.2 / 0.7 %.
- Exploratory (400 traces per group): per-frame noise of a 5 x 5 box is 54-62 photons (1.4826 x MAD);
  a Halo-K117 single-frame event (median 441 photons) is about 7 sigma. Halo-K117 traces rejected at
  separation < 3 have a median of 8 frames above median + 5 sigma (82 % have >= 4); rejected
  Halo-D106 traces 1 frame (21 %). Rule to flag molecule-like rejected traces proposed to the user.
- 2026-11-20, `bright_frames` run_001 (S3IT job 8602983, 1 min; fetched to `s3it/bright_frames/`) and
  `molecule_like` run_001 (local): the confirmed rule. Results under "R3.3b results" below.
- Found while replaying: `filter.py` reads `gmm_min_separation` with a default of 3.0 and
  `Extraction/config.yaml` does not set it, so traces with GMM separation below 3 get no ON frames
  and are rejected as having too few peaks. The effective SNR cut is 3, not the 2 the SI states.

## R3.3a results

Photons are detected photons at 0.062 per ADU (measured, `../PaperData/`; user decision 2026-11-20):
events run_002 and the figures built from it. Log entries before that decision (events run_001,
flagged_features run_001 and the exploratory notes) used 16.0 / 300 = 0.053, so their photon values
are 1.1625 times lower; nothing else differs (see the log).

Per construct (all sessions pooled; medians over molecules of per-molecule values, IQR in the
figure folder's tables):

| Construct | Molecules | Events | 1-frame events | Mean ON per molecule | Photons per event | Photons per fully-ON frame | Photons per molecule |
|---|---|---|---|---|---|---|---|
| Halo-D106 | 2580 | 47176 | 51.5 % | 3.3 frames (115 ms) | 3374 | 2886 (2239 molecules) | 78,754 |
| SNAP-C148 | 9647 | 250064 | 81.3 % | 1.2 frames (41 ms) | 763 | 1598 (4849) | 23,022 |
| Halo-K117 | 2384 | 83288 | 91.2 % | 1.1 frames (37 ms) | 546 | 2175 (919) | 23,801 |
| scGrx1 | 3400 | 51893 | 69.4 % | 1.5 frames (52 ms) | 761 | 1106 (2176) | 18,081 |
| scGrx1-AcK20 | 1260 | 15513 | 65.5 % | 1.5 frames (52 ms) | 1233 | 1992 (776) | 21,418 |

Short events against fully-ON frames (median photons of events of that length / full-frame value):

| Construct | 1 frame | 2 frames | 3 frames |
|---|---|---|---|
| Halo-D106 | 29 % | 47 % | 68 % |
| SNAP-C148 | 40 % | 59 % | 79 % |
| Halo-K117 | 24 % | 38 % | 65 % |
| scGrx1 | 46 % | 74 % | 80 % |
| scGrx1-AcK20 | 36 % | 59 % | 79 % |

Halo-D106 and Halo-K117 measured the same day (photons per fully-ON frame, median [IQR], molecules):
Exp1 2923 [2130-3633] (460) vs 1701 [976-2651] (155); Exp2 2650 [1777-3252] (599) vs 2286
[1079-3219] (764).

Reading:
- Photons grow in proportion to duration and approach the full-frame line from about three frames
  on: while a molecule is ON it emits at a steady rate.
- One- and two-frame events carry far fewer photons than whole ON frames; for Halo-K117, 91 % of
  events are single frames carrying a median 24 % of a fully-ON frame's photons. Most K117 ON
  periods are therefore shorter than one frame, and the 35 ms of the paper (37 ms at 34.49 ms per
  frame) is the frame limit, not a resolved ON time. Adding one frame on each side raises the
  median photons per event by 4-13 % (546 to 581 for K117), so little signal hides in neighboring
  frames below the ON threshold.
- Halo-K117's fully-ON frames are dimmer than Halo-D106's on the same days. With 34.49 ms frames this
  cannot separate a lower emission rate from switching faster than the frame within an ON stretch.
- scGrx1: long events fall below the full-frame line, and its full-frame value (1106) is the lowest;
  consistent with low-level stretches classified as ON (seen in R2.1). Not yet quantified.
- Across constructs, photons per event and per molecule mix construct and session (laser power
  differed between sessions); only the within-session comparison is clean.

Not done: tests (none requested); R3.3c. Camera mode (user, 2026-11-20): no frame transfer, so
each 34.49 ms frame is 30 ms of exposure plus 4.49 ms of readout with no light recorded. A
fully-ON frame's photons are 30 ms of emission (Halo-D106 2886 photons, about 96 per ms; Halo-K117
2175, about 73 per ms), and ON periods inside a readout gap are invisible.

## R3.3b results (2026-11-20)

The paper's filter, replayed (`filter_audit.py`; S3IT reference run and local run agree on every
trace), reproduces the published filtered set exactly once two rules outside the documented cascade
are included: an exact-zero z-score drops 4 traces, and `gmm_min_separation` = 3.0 (default, not in
the config) makes the effective SNR cut 3, not 2.

Fate of the unfiltered traces inside vesicles:

| Construct | Traces | SNR < 2 | SNR 2-3 | Too few peaks | Timing criteria | Passed |
|---|---|---|---|---|---|---|
| Halo-D106 | 6067 | 32.2 % | 0.6 % | 20.4 % | 4.2 % | 42.5 % |
| Halo-K117 | 4740 | 40.9 % | 4.1 % | 2.3 % | 2.3 % | 50.3 % |
| SNAP-C148 | 15382 | 28.5 % | 1.0 % | 4.7 % | 3.0 % | 62.7 % |
| scGrx1 | 12664 | 62.3 % | 1.0 % | 5.7 % | 4.2 % | 26.8 % |
| scGrx1-AcK20 | 4324 | 52.4 % | 0.1 % | 15.1 % | 3.2 % | 29.1 % |

Noise reference (background traces, same movies): 97.5-99.9 % below SNR 2 and 0.1-2.1 % pass the
whole filter per session, except scGrx1 Exp5 (8.0 % pass): random positions on that slide carry
blinking emitters.

Molecule-like rejected traces (rejected at separation < 3 with more frames above median + 5 sigma
than 99 % of the session's background traces; background exceeds its threshold 0.8-1.1 % per
session by construction). Thresholds vary between sessions from 1 to 79 bright frames: where the
background itself contains emitters (dense slides), the threshold is high and the flag conservative.

| Construct | Rejected at SNR < 3 | Molecule-like (k = 5) | Expected false flags | Gain over passed (upper bound) | k = 4 / 6 |
|---|---|---|---|---|---|
| Halo-D106 | 1987 | 154 (7.8 %) | 19.4 | 6.0 % | 5.1 / 11.8 % |
| Halo-K117 | 2136 | 99 (4.6 %) | 20.8 | 4.2 % | 8.2 / 2.7 % |
| SNAP-C148 | 4533 | 366 (8.1 %) | 43.5 | 3.8 % | 7.7 / 8.4 % |
| scGrx1 | 8014 | 1728 (21.6 %) | 73.4 | 50.8 % | 22.3 / 22.8 % |
| scGrx1-AcK20 | 2270 | 506 (22.3 %) | 21.7 | 40.2 % | 18.4 / 24.7 % |

Same-day Halo-D106 against Halo-K117 (k = 5): Exp1 0 of 97 vs 41 of 944 rejected traces
molecule-like (expected false flags 1.0 vs 8.8); Exp2 2 of 331 vs 58 of 1192 (3.2 vs 12.0). As a
share of the passed traces: Halo-K117 6.4 % (Exp1) and 3.3 % (Exp2); Halo-D106 0 and 0.3 %.

Reading:
- The reviewer's concern holds in direction and is small in size for the comparison it was about:
  the SNR gate removes more molecule-like Halo-K117 traces than Halo-D106 traces on the same days,
  but they would add at most 3-6 % to the Halo-K117 traces analyzed.
- The 400-trace exploratory sample (82 % of rejected Halo-K117 traces with >= 4 frames above
  5 sigma) overstated this: background traces of the same sessions have as many bright frames
  (thresholds 22-24 frames). Without the background calibration the conclusion would have been
  wrong.
- The gate matters more for the scGrx1 and scGrx1-AcK20 sessions (DK, cleaner background,
  thresholds 1-15 frames): 21-22 % of their rejected traces are molecule-like, as many as 40-51 % of
  the passed traces (more than the passed ones in some sessions). The fraction is similar for both
  proteins, so the gate does not favor one of them, but the analyzed traces are the subset with a
  clean bright mode. Halo-D106 Exp3 is similar (152 of 1098).
- The gain is an upper bound: flagged traces were not tested against the later peak-number and
  timing criteria.

Bias check (`flagged_features.py`, run_002; run_001 with 0.053 photons per ADU): events of every molecule-like rejected trace (the
paper's classifier without the separation gate, as in R3.3a) against the passed traces of the same
construct and session; medians over molecules, sessions with >= 20 flagged traces:

| Construct, session | Group | Molecules | Events | 1-frame events | Mean ON (frames) | Photons per event | Mean OFF (frames) |
|---|---|---|---|---|---|---|---|
| Halo-K117 Exp1 | passed | 638 | 24 | 95 % | 1.05 | 535 | 128 |
| | flagged | 41 | 46 | 98 % | 1.02 | 397 | 113 |
| Halo-K117 Exp2 | passed | 1746 | 30 | 93 % | 1.08 | 551 | 113 |
| | flagged | 58 | 45 | 97 % | 1.03 | 399 | 117 |
| scGrx1 Exp9 | passed | 768 | 11 | 86 % | 1.18 | 733 | 290 |
| | flagged | 712 | 157 | 96 % | 1.05 | 164 | 37 |
| scGrx1-AcK20 Exp10 | passed | 324 | 8 | 75 % | 1.33 | 1024 | 249 |
| | flagged | 363 | 194 | 94 % | 1.07 | 157 | 30 |

(all sessions in `flagged_features/run_002/comparison.csv`; Halo-D106 Exp3, SNAP-C148 Exp1-3 and the
other scGrx1 / scGrx1-AcK20 sessions look like the last two rows.)

- Halo-K117: the flagged traces are the same kind of molecule as the passed ones, fast single-frame
  blinkers with the same OFF times and duty cycle, about 27 % dimmer per event. Including them would
  move Halo-K117 toward shorter and dimmer events, i.e. widen the difference from Halo-D106; at
  3-6 % of the passed traces they would barely move its medians.
- Everywhere else the flagged traces are unlike the analyzed molecules: hundreds of tiny "events" of
  about 160 photons (about 2.5 x the per-frame noise of 62-72 photons), short OFF gaps, a few frames
  above 5 sigma. Noise-dominated traces, not lost molecules of the analyzed kind. Where the
  background threshold is low (1-8 frames) the flag is not specific, so the 22 % "molecule-like"
  share of scGrx1 and scGrx1-AcK20 overstates the loss.
- 2026-11-20, photon conversion changed to the measured 0.062 photons per ADU (user decision;
  `../PaperData/README.md`): `events` run_002 (8 min, 16 cores) and `flagged_features` run_002 replace
  run_001 for every photon number; figures R3.3a and R3.3b rebuilt. Check against run_001: every
  photon column is exactly 1.1625 x the old value, every other column identical, except 3 of 19,271
  passed molecules (Halo-K117 5671, 6247; scGrx1 6914) and 33 of 2853 flagged traces whose ON
  classification moved by a few frames (median 0.8 % of their events): min-max normalizing x 1.1625
  changes the last floating-point bit, which flips frames that sit at the 0.9 posterior cutoff. Event
  totals +4 (447,934); every reported median unchanged in the non-photon columns. run_002's code
  copies equal commit 5bf026c (it was started a minute before that commit; the manifest records the
  commit).
