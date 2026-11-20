# FrameZero: molecules ON in the first frame

Found 2026-11-20 in R3.1 (`../BlankSurface/`) and quantified on the paper's data
(`../PaperData/README.md`, `edge_groups` run_001; MANUSCRIPT_FIXES item 14): the published extraction
never includes a molecule that is ON in the movie's first frame. `Extraction/utils.py`
get_and_link_locs calls Picasso's `postprocess.link` with its default `remove_ambiguous_lengths=True`,
which removes every linked group starting in frame 0; with 4000-frame dark-time linking a group is nearly
a whole molecule. Single ROIs inside vesicles lost: Halo-D106 17.7 % (same days 8.6 % and 27 %),
Halo-K117 2.6 % (0.4 %, 3.8 %), SNAP-C148 4.5 %, scGrx1 1.4 %, scGrx1-AcK20 2.3 %.

The user asked to test whether this could change a conclusion (2026-11-20, "Yes, test them").

## Design

1. Rerun the paper's extraction (extract, combine, filter; `Extraction/` unchanged, parameters of
   `Extraction/config.yaml`) on the published localizations, with one change: `keep_edges/sitecustomize.py`
   makes `postprocess.link` keep groups touching the first or last frame (`FRAMEZERO_KEEP_EDGE_GROUPS=1`).
   Control images off; everything else as published. New run folder on S3IT scratch; the published run and
   the raw movies are only read.
2. `compare.py`: match every ROI to the published run by movie and box position. Where nothing changed
   (same ROI, single and inside a vesicle in both), the rerun must reproduce the published raw and
   background-removed traces and the filter fate exactly (check). Count the recovered molecules, how many
   pass the quality filter, and published ROIs that a recovered neighbor now overlaps.
3. `classify_recovered.py`: the paper's final models on the recovered passing traces, with their true
   labels (accuracy with Wilson interval, kept fraction at WD 0.7), against the final models' published
   test performance (`../Retention/README.md`).
4. `features.py`: do the recovered molecules differ from the published ones in the kinetic features
   of Figs. 5D and 6C (mean ON, mean OFF, duty cycle; no photons, user 2026-11-20), and would including
   them change the pair differences? Medians and rank-biserial r, pooled and within shared sessions;
   sensitivity without the recovered molecules' first ON event (cut short by the start of recording).

## Files

- `config.yaml`: published run, rerun folder, overrides.
- `keep_edges/sitecustomize.py`: the one change (tested: on the empty-ATTO425 field whose only spot
  Picasso had dropped, 0 linked groups without the patch, 1 with it).
- `stage.py`: the rerun folder (run_info.yaml copied, every published `*_locs.*` linked, run config).
- `s3it/rerun.sbatch`: stage, extract, combine, filter (picasso-env).
- `compare.py`, `s3it/compare.sbatch`: rerun against published.
- `classify_recovered.py`: final models on the recovered traces (local, torch).
- `features.py`: kinetic features of the recovered molecules and the pair effect sizes (local).

## Log

- 2026-11-20: rerun submitted (S3IT job 8609846, run folder
  `/scratch/pabriv/framezero/Grx1_HTHTL_HTIA_K20Ac_snap_keepedges_001`).
- 2026-11-20, rerun finished (23 min: extract 11, combine 8, filter 4; 60 GB on scratch).
- 2026-11-20, `compare` run_001 overcounted ROIs that share a box corner within a movie (merged many
  to many); run_002 (S3IT job 8610143) matches them one to one and checks every count against both
  runs' own tables. Results below are run_002.
- 2026-11-20, `classify` run_001 (local MPS) and `features` run_001 (local).

## Results (2026-11-20)

Reproduction (`compare` run_002, `checks.json`): every ROI common to both runs that is single and inside
a vesicle in both (5966 Halo-D106, 4702 Halo-K117, 15245 SNAP-C148, 12612 scGrx1, 4301 scGrx1-AcK20) has
bit-identical raw and background-removed traces and the same filter fate as in the published run; no
published ROI is missing; all counts equal the runs' own tables. The one change does only what it should.

Dataset change (traces inside vesicles passing the quality filter):

| Construct | Published | Recovered single IN (passing) | Published lost to overlap (had passed) | Rerun | Change |
|---|---|---|---|---|---|
| Halo-D106 | 2580 | 1086 (515) | 101 (60) | 3035 | +17.6 % |
| Halo-K117 | 2384 | 125 (88) | 38 (31) | 2441 | +2.4 % |
| SNAP-C148 | 9647 | 701 (408) | 141 (110) | 9946 | +3.1 % |
| scGrx1 | 3400 | 183 (106) | 54 (34) | 3472 | +2.1 % |
| scGrx1-AcK20 | 1260 | 99 (41) | 23 (9) | 1292 | +2.5 % |

Same days, recovered passing traces against the published passing ones: Halo-D106 Exp1 8 %, Exp2
25 %, Exp3 13 %, Exp4 25 %; Halo-K117 Exp1 0.3 %, Exp2 5 %. 99.8 % of the recovered molecules have
their first ON event in frame 0 (the rest within 4 frames).

Kinetic features (`features` run_001; recovered against published molecules of the same construct,
medians; r > 0: recovered larger):

| Construct | Mean ON (ms) | Mean OFF (ms) | Duty cycle |
|---|---|---|---|
| Halo-D106 | 164 vs 114 (r 0.31) | 1733 vs 2873 (-0.24) | 0.087 vs 0.041 (0.32) |
| Halo-K117 | 55 vs 37 (0.65) | 3740 vs 4014 (0.03) | 0.013 vs 0.010 (0.17) |
| SNAP-C148 | 52 vs 41 (0.44) | 1283 vs 1642 (-0.16) | 0.044 vs 0.027 (0.28) |
| scGrx1 | 86 vs 52 (0.40) | 10660 vs 10083 (0.03) | 0.009 vs 0.006 (0.22) |
| scGrx1-AcK20 | 145 vs 52 (0.53) | 6650 vs 6923 (-0.10) | 0.022 vs 0.011 (0.40) |

Without the first (cut-short) ON event the recovered molecules still have longer ON times (r 0.21-0.54)
and higher duty cycles (r 0.04-0.30). Molecules ON in the first frame are the more active ones of each
construct (longer ON, higher duty cycle), as expected when sampling by "ON at a given instant".

Pair differences (rank-biserial r, published dataset -> rerun dataset; pooled):

| Pair | Mean ON | Mean OFF | Duty cycle |
|---|---|---|---|
| Halo-D106 vs Halo-K117 | 0.807 -> 0.818 | -0.171 -> -0.214 | 0.528 -> 0.571 |
| Halo-D106 vs SNAP-C148 | 0.707 -> 0.732 | 0.277 -> 0.240 | 0.175 -> 0.230 |
| scGrx1 vs scGrx1-AcK20 | -0.039 -> -0.049 | 0.227 -> 0.237 | -0.277 -> -0.290 |

Within the shared days the Halo-D106 vs Halo-K117 differences grow the same way (Exp1 ON 0.870 ->
0.877, duty 0.524 -> 0.546; Exp2 ON 0.727 -> 0.751, duty 0.441 -> 0.500); without the recovered
molecules' first event the numbers are the same to 0.003. scGrx1 vs scGrx1-AcK20 changes by at most
0.04 on any day.

Final models on the recovered passing traces (`classify` run_001; never seen in training; true labels;
Wilson 95 % intervals), against the published test set (`../Retention/README.md`):

| Recovered | n | Accuracy, all | Kept (WD >= 0.7) | Accuracy, kept | Published test: all / kept / kept accuracy |
|---|---|---|---|---|---|
| Halo-D106 | 515 | 93.6 % (91.1-95.4) | 82 % | 98.8 % (97.3-99.5) | 82.4 % / 73 % / 90.5 % |
| Halo-K117 | 88 | 64.8 % (54.4-73.9) | 41 % | 58 % (42-73; n 36) | 86.6 % / 69 % / 94.0 % |
| scGrx1 | 106 | 79.2 % (70.6-85.9) | 39 % | 95 % (84-99; n 41) | 75.1 % / 31 % / 100 % |
| scGrx1-AcK20 | 41 | 68.3 % (53.0-80.4) | 17 % | 71 % (36-92; n 7) | 72.0 % / 13 % / 84.0 % |

Reading:
- The exclusion is construct-dependent (Halo-D106 lost a sixth of its passing traces, the others
  2-3 %) and removed the most active molecules of every construct.
- It does not change any conclusion. With the molecules put back, the Halo-D106 vs Halo-K117 kinetic
  differences grow slightly (pooled and on both shared days), and the acetylation pair is unchanged.
  The final model classifies the recovered Halo-D106 molecules better than its published test set
  and the recovered Halo-K117 ones worse (they have longer ON times, closer to Halo-D106); with 515
  against 88 traces, a rough recall weighted over published plus recovered traces would move
  Halo-D106 from 82.4 % to 84.3 % and Halo-K117 from 86.6 % to 85.8 % (not a retrained model).
  The acetylation model behaves on the recovered traces as on its test set.
- Not tested: models retrained or cross-validated with the recovered traces included; SNAP-C148 has
  no final model (its kinetic differences from Halo-D106 grow slightly as well).
