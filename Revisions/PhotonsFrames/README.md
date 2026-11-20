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
- `figure_R3.3a.py`: figure R3.3a from an events run (`--build --events <run> --out <folder>`), or
  replot from the CSVs next to it.

Outputs: `Results/Revisions/PhotonsFrames/events/run_NNN/`, `check_fig5d6c/run_NNN/` (manifests);
figure folder `/Volumes/MediumBerth/SalomePaperData/Püntener2026/revisions_figures/R3.3a/`.

```bash
/Users/privera/opt/anaconda3/envs/locbplots/bin/python Revisions/PhotonsFrames/events.py
/Users/privera/opt/anaconda3/envs/locbplots/bin/python Revisions/PhotonsFrames/figure_R3.3a.py --build --events Results/Revisions/PhotonsFrames/events/run_001 --out "/Volumes/MediumBerth/SalomePaperData/Püntener2026/revisions_figures/R3.3a"
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

## R3.3a results

Per construct (all sessions pooled; medians over molecules of per-molecule values, IQR in the
figure folder's tables):

| Construct | Molecules | Events | 1-frame events | Mean ON per molecule | Photons per event | Photons per fully-ON frame | Photons per molecule |
|---|---|---|---|---|---|---|---|
| Halo-D106 | 2580 | 47176 | 51.5 % | 3.3 frames (115 ms) | 2903 | 2483 (2239 molecules) | 67,745 |
| SNAP-C148 | 9647 | 250064 | 81.3 % | 1.2 frames (41 ms) | 656 | 1375 (4849) | 19,804 |
| Halo-K117 | 2384 | 83286 | 91.2 % | 1.1 frames (37 ms) | 469 | 1871 (919) | 20,474 |
| scGrx1 | 3400 | 51891 | 69.4 % | 1.5 frames (52 ms) | 655 | 952 (2176) | 15,554 |
| scGrx1-AcK20 | 1260 | 15513 | 65.5 % | 1.5 frames (52 ms) | 1060 | 1714 (776) | 18,424 |

Short events against fully-ON frames (median photons of events of that length / full-frame value):

| Construct | 1 frame | 2 frames | 3 frames |
|---|---|---|---|
| Halo-D106 | 29 % | 47 % | 68 % |
| SNAP-C148 | 40 % | 59 % | 79 % |
| Halo-K117 | 24 % | 38 % | 65 % |
| scGrx1 | 46 % | 74 % | 80 % |
| scGrx1-AcK20 | 36 % | 59 % | 79 % |

Halo-D106 and Halo-K117 measured the same day (photons per fully-ON frame, median [IQR], molecules):
Exp1 2514 [1832-3125] (460) vs 1463 [840-2280] (155); Exp2 2279 [1529-2797] (599) vs 1966
[928-2769] (764).

Reading:
- Photons grow in proportion to duration and approach the full-frame line from about three frames
  on: while a molecule is ON it emits at a steady rate.
- One- and two-frame events carry far fewer photons than whole ON frames; for Halo-K117, 91 % of
  events are single frames carrying a median 24 % of a fully-ON frame's photons. Most K117 ON
  periods are therefore shorter than one frame, and the 35 ms of the paper (37 ms at 34.49 ms per
  frame) is the frame limit, not a resolved ON time. Adding one frame on each side raises the
  median photons per event by 4-13 % (469 to 500 for K117), so little signal hides in neighboring
  frames below the ON threshold.
- Halo-K117's fully-ON frames are dimmer than Halo-D106's on the same days. With 34.49 ms frames this
  cannot separate a lower emission rate from switching faster than the frame within an ON stretch.
- scGrx1: long events fall below the full-frame line, and its full-frame value (952) is the lowest;
  consistent with low-level stretches classified as ON (seen in R2.1). Not yet quantified.
- Across constructs, photons per event and per molecule mix construct and session (laser power
  differed between sessions); only the within-session comparison is clean.

Not done: tests (none requested); the SNR-gate question (R3.3b); the camera mode (frame transfer
or not) and anything that depends on it (R3.3c).
