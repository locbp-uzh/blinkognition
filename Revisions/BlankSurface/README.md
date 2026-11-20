# BlankSurface: blank coverslips and empty vesicles (R3.1)

Reviewer 3, point 1: show fully prepared coverslips imaged without any sample (no vesicles, no
protein), to exclude that surface impurities produce protein-like blinking. Plan and decisions in
`../ReviewRound1/README.md`.

## Data (S3IT, `/shares/locbp.chem.uzh/dkossm/Blinkognition`, read in place)

| Slide (pipeline name) | Session | Folder | Content | FOVs | 640 nm |
|---|---|---|---|---|---|
| Blank | DFK786 | blank | coated (PLL-PEG + 2 % biotin, NeutrAvidin), 400 uL PB only | 10 | 13 mW |
| EmptyATTO520 | DFK786 | ATTO520 | empty ATTO520 vesicles (files named blank_*) | 25 | 13 mW |
| EmptyATTO390 | DFK786 | ATT390 | empty ATTO390 vesicles | 25 | 13 mW |
| Blank | DFK787 | blank | coated, 600 uL PB only | 20 | 15 mW |
| EmptyATTO390 | DFK787 | ATTO390 | empty ATTO390 vesicles | 20 | 15 mW |
| EmptyATTO425 | DFK787 | ATTO425 | empty ATTO425 vesicles | 20 | 15 mW |
| SNAP520 | DFK788 | SNAP520/day1 | same-power control: SNAP-HMSiR-IA in ATTO520 vesicles | 26 | 14-15 mW |
| HT390 | DFK788 | HT390/day1 | same-power control: HT7-HMSiR-HTL in ATTO390 vesicles | 26 | 14-15 mW |

6000 frames at 640 nm (30 ms exposure), 10-frame snapshots at 405, 488, 515 nm. The paper's 2025
sessions were imaged at 20-21 mW.

## Design (confirmed by the user 2026-11-20)

The paper's extraction (`Extraction/`, unchanged) from the localize step, with fixed gradients
instead of the parameter finder: protein 10000 (main; the paper's most permissive), 20000 and 40000
(sensitivity); vesicle channel 488 nm 2000, 405 nm 4000. Slides with ATTO520 vesicles (and the blank
slides) use the 488 nm channel as vesicle reference, as the paper; ATTO390 / ATTO425 slides the 405 nm
channel. Per field of view: spots, single ROIs, traces passing the quality filter; for vesicle slides,
vesicles and passing traces inside vesicles; an upper bound of impurity traces counted as protein.
Descriptive; no tests. Passing blank traces, if any, classified by the final models.

## Files

- `config.yaml`: slides, gradients, staging, overrides of the paper's config.
- `stage.py`: links one vesicle channel's slides into the pipeline's layout on S3IT scratch and writes
  the run config (`<stage_root>/configs/config_gt<ch>_g<gradient>.yaml`).
- `s3it/extract.sbatch <ch> <gradient>`: staging plus `Extraction/run_pipeline.py --start-from
  localize`; copies the run config and link map into the pipeline's run folder.
- `picasso_sync/sitecustomize.py`: synchronous Picasso MLE fits (see below).
- `dye_check.py`: which vesicle dye is on which slide, from the snapshots.
- `count.py`: per-FOV and per-trace counts of the six runs, references (same-power protein slides, the
  paper's passing IN traces per movie), checks (docstring lists them).
- `projections.py`: per-field images (640 nm maximum projection and mean, vesicle snapshot).
- `classify_blank.py`: the paper's two final models on every passing trace (local, `ML/classify.py`).
- `figure_R3.1.py`: figure R3.1 (a: fields and counts; b: passing traces and model calls).
- `s3it/run_cpu.sbatch <script>`: count.py or projections.py on an S3IT CPU node.

Outputs: `Results/Revisions/BlankSurface/Extraction_gt<ch>_g<gradient>/<run>/` (S3IT clone),
`dye_check/run_NNN/`.

## Log

- 2026-11-20, Picasso: the published run's Picasso 0.9.3 (picasso-env) has the same race as 0.8.8
  (Revisions/DualColor, memory note): `python -m picasso localize` reads the MLE fits once the last
  spot is handed to a thread (`__main__.py` `while current[0] < n_spots`; `gaussmle_async` ends with
  `executor.shutdown(wait=False)`), so fits still running (at most one per thread, up to 60) are read
  as zeros and dropped. For the paper's dense movies this is at most a few dozen of tens of thousands
  of localizations per movie, negligible for the traces (box sums at linked ROI positions); for
  sparse blank movies it could remove a large share. These runs use synchronous fits.
- 2026-11-20, `dye_check` run_001 (S3IT srun job 8604817): bright objects above background
  (99.9th percentile minus median, median over fields, ADU) at 405 / 488 / 515 nm:

  | Slide | 405 nm | 488 nm | 515 nm |
  |---|---|---|---|
  | DFK786 Blank | 526 | 281 | 136 |
  | DFK786 ATTO520 folder | 1049 | 52096 | 47784 |
  | DFK786 ATT390 folder | 7822 | 360 | 240 |
  | DFK787 Blank | 508 | 275 | 134 |
  | DFK787 ATTO390 | 10420 | 362 | 213 |
  | DFK787 ATTO425 | 10267 | 495 | 149 |
  | DFK788 HT390 (reference) | 7851 | 337 | 210 |
  | DFK788 SNAP520 (reference) | 611 | 28811 | 55600 |

  The ATTO520 folder holds ATTO520 vesicles and the ATT390 folder ATTO390 vesicles. Acquisition times
  (ND2 dates, `projections` run_001) order the slides blank 16:57-17:29, ATTO520 folder 18:18-19:44,
  ATT390 folder 20:06-21:32: slide 2 is the ATTO520 folder, so the readme's "ATTO390" for slide 2 and
  the `blank_` file names in that folder are labeling slips; slide 3, with the readme's note of
  background at 640 nm, is the ATT390 folder (EmptyATTO390, DFK786). (Corrected 2026-11-20: an earlier
  version named slide 3 as the slip.) The
  blank slides show no vesicles in any channel. (488 and 515 nm on ATTO520 vesicles are close to
  the camera's saturation, about 56,800 ADU per pixel; irrelevant for detection.)
- 2026-11-20: six extraction jobs submitted (S3IT 8604886, 8604887, 8604889 for 488 nm at gradients
  10000, 20000, 40000; 8604890, 8604892, 8604893 for 405 nm).
- 2026-11-20, first six runs (8604886-8604893, all COMPLETED in 5-10 min) are invalid for the blank and
  empty-vesicle slides, for two reasons found before counting anything:
  1. `Extraction/extract.py` skips a movie without any 640 nm localization ("No localizations",
     `FileLists/failed_file_list.pkl`): at gradient 10000, 3 of 10 DFK786 blank fields, 14 of 20 DFK787
     blank fields and 1 empty-ATTO520 field had none. Nothing is lost: count.py counts them as fields with
     zero of everything (and checks that every skipped movie has zero localizations and vice versa).
  2. `Extraction/combine.py` assigns per-movie traces to a protein by matching the protein name in the file
     name, case-sensitively and bounded by underscores. The raw files are named `blank_*`, `ATTO390_*`,
     `ATTO425_*`, so it found no files for Blank, EmptyATTO520, EmptyATTO390 and EmptyATTO425 ("No trace
     files found", 358-byte empty tables); only SNAP520 and HT390 went through the filter. Fixed in my
     staging, not in the pipeline (commit eaef51a): links are named `<slide>_<original file name>` and
     checked against combine's own patterns; the pipeline's channel pairing (first name token plus field
     number) is unchanged. Tested on a mock tree with the real file names (pairing and matching per slide).
     The paper's files carried the construct key as prefix, so the published run is not affected.
  New staging root `/scratch/pabriv/blanksurface/stage2`; the `_001` run folders stay as they are and are
  not used (count.py takes the newest run staged from the configured root).
- 2026-11-20: six extraction jobs resubmitted (S3IT 8606234-8606236 for 488 nm at 10000, 20000, 40000;
  8606237-8606239 for 405 nm).
- 2026-11-20, reruns 8606234-8606239 (5-8 min each): combine.py now finds every slide (Blank 13 movies,
  EmptyATTO520 24, EmptyATTO390 42, EmptyATTO425 17 at gradient 10000; the others have no linked
  emitter). `projections` run_001 (S3IT job 8606489, 3 min): images of all 172 fields.
- 2026-11-20, `count` run_001 and run_002 failed checks that caught two errors of mine, fixed before any
  number was used: run_001 matched skipped movies by file name, but DFK786 and DFK787 share file names
  (commit b346e45); run_002 showed one movie skipped despite 4 localizations, because Picasso's
  `postprocess.link` (default `remove_ambiguous_lengths=True`) drops every linked group starting in
  frame 0; count.py now reports linked groups with and without that removal (commit 12a3309). The
  same removal applies to the paper's data (`../PaperData/README.md`, `edge_groups`; MANUSCRIPT_FIXES 14).

## Results (count run_003, 2026-11-20; all checks pass)

DFK786's ATTO390 slide (folder ATT390, readme slide 3) is not used: the user reports an experimental
mix-up on that slide (2026-11-20). It was extracted and counted with the others (no cross-slide
dependence in the pipeline: combine, background and filter act per trace), and is left out of every
number below, the figure and the response.

`Results/Revisions/BlankSurface/s3it/count/run_003/` (S3IT job 8608552). Every 640 nm movie has its
localizations; skipped movies are exactly those without a linked emitter (counted as zero); the filter
replay reproduces the pipeline's filtered set trace by trace; the run's filter parameters equal the
published ones; no blank field has a vesicle cluster.

Per field at gradient 10000:

| Slide | Fields | Without any emitter | Localizations (median) | Emitters (mean) | Vesicles (mean) | IN area | Single IN | Single OUT |
|---|---|---|---|---|---|---|---|---|
| Blank DFK786 | 10 | 3 | 1.5 | 1.7 | 0 | 0 | 0 | 1.7 |
| Blank DFK787 | 20 | 14 | 0 | 0.35 | 0 | 0 | 0 | 0.35 |
| Empty ATTO520 DFK786 | 25 | 1 | 19 | 5.8 | 595 | 0.45 | 2.6 | 3.0 |
| Empty ATTO390 DFK787 | 20 | 3 | 4.5 | 2.6 | 361 | 0.29 | 1.0 | 1.5 |
| Empty ATTO425 DFK787 | 20 | 3 | 5.5 | 3.0 | 212 | 0.17 | 0.65 | 2.2 |
| SNAP-tag DFK788 (protein) | 26 | 0 | 195 | 13.0 | 108 | 0.09 | 5.3 | 5.5 |
| HaloTag7 DFK788 (protein) | 26 | 0 | 450 | 15.4 | 180 | 0.15 | 8.1 | 4.5 |

(IN area: share of the field within 4 px of a vesicle center, where a spot counts as inside.) The paper's
movies have 117-375 vesicles per field on average (`edge_groups` run_001; scGrx1-AcK20 117, scGrx1 214,
Halo-D106 232, Halo-K117 253, SNAP-C148 375).

Traces passing the quality filter (all fields; IN / OUT) at gradients 10000, 20000, 40000:

| Slide | 10000 | 20000 | 40000 |
|---|---|---|---|
| Blank DFK786 | 0 / 0 | 0 / 0 | 0 / 0 |
| Blank DFK787 | 0 / 0 | 0 / 0 | 0 / 0 |
| Empty ATTO520 DFK786 | 1 / 0 | 2 / 0 | 1 / 0 |
| Empty ATTO390 DFK787 | 0 / 0 | 0 / 0 | 0 / 0 |
| Empty ATTO425 DFK787 | 0 / 0 | 0 / 1 | 0 / 0 |
| SNAP-tag DFK788 | 54 / 29 | 71 / 37 | 55 / 30 |
| HaloTag7 DFK788 | 97 / 10 | 114 / 13 | 105 / 11 |

Passing IN traces per field: blank 0; empty ATTO520 0.04-0.08; empty ATTO390 and ATTO425 0; same-power
protein 2.1-2.7 (SNAP-tag) and 3.7-4.4 (HaloTag7); the paper's movies 3.9 (scGrx1-AcK20) to 41
(Halo-K117) on average (Halo-D106 9.1, scGrx1 9.6, SNAP-C148 34).

Reading:
- Coated coverslips without vesicles carry almost no emitters (median 0-1.5 localizations per field;
  17 of 30 fields without a single linked emitter) and none passes the filter at any gradient.
- Empty vesicles: three emitters in 65 fields pass at any gradient (two inside ATTO520 vesicles, one
  outside vesicles on the ATTO425 slide). The ATTO520 slide, the paper's vesicle dye, gives 0.04-0.08
  passing IN traces per field: 2-3 % of the same-power SNAP-tag slide's rate although it has 5.5 times
  as many vesicles per field, and at most 2 % of the sparsest paper dataset (scGrx1-AcK20, 3.9 per
  movie). Two of the three look unlike protein blinking (R3.1b: a 25-s plateau, a burst then nothing).
- Caveat: these slides were imaged at 13-15 mW, the paper at 20-21 mW; the same-power protein slides
  give the like-for-like reference.

Not reported (user, 2026-11-20): how the final models call these traces. `classify_blank.py` run_001
exists (the Halo model calls blank, empty and SNAP-tag traces Halo-D106 with confidence, as a
two-class model must; HaloTag7 with the HaloTag ligand, the paper's Halo-D106 labeling, is called
D106), but the paper claims only the distinction between its two classes, so out-of-distribution
behavior is not discussed. Photon numbers are reported descriptively only.

Figure: `revisions_figures/R3.1/` (R3.1a: fields and counts; R3.1b: every passing blank or empty trace
next to one protein trace per same-power slide; caption, CSVs, script; standalone replot checked).

Not done: no test of the gradient the paper used per session (all three gradients reported
instead), and the laser power differs (13-15 against 20-21 mW).
- 2026-11-20 (user: answer only what is asked): figure R3.1 trimmed to the two blank coverslips and one
  same-power SNAP-tag field with the counts in the caption (`figure_R3.1.py`, commit 05b8bf1); the full
  version (empty vesicles, counts against the paper, passing traces) is in
  `revisions_figures/not_in_revision/R3.1_full_20261120/` and git history (4ddac80). The empty-vesicle
  results above stay as a record; they are not part of the answer.
