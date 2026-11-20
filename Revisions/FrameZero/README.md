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
4. Then, after the user confirms the statistics: do the recovered molecules differ from the published
   ones in the features of Figs. 5D and 6C, and would including them change the population difference?

## Files

- `config.yaml`: published run, rerun folder, overrides.
- `keep_edges/sitecustomize.py`: the one change (tested: on the empty-ATTO425 field whose only spot
  Picasso had dropped, 0 linked groups without the patch, 1 with it).
- `stage.py`: the rerun folder (run_info.yaml copied, every published `*_locs.*` linked, run config).
- `s3it/rerun.sbatch`: stage, extract, combine, filter (picasso-env).
- `compare.py`, `s3it/compare.sbatch`: rerun against published.
- `classify_recovered.py`: final models on the recovered traces (local, torch).

## Log

- 2026-11-20: rerun submitted (S3IT job 8609846, run folder
  `/scratch/pabriv/framezero/Grx1_HTHTL_HTIA_K20Ac_snap_keepedges_001`).
