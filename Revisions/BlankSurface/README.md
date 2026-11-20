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

  The ATTO520 folder holds ATTO520 vesicles and the ATT390 folder ATTO390 vesicles; the readme's
  "ATTO390" for slide 3 and the `blank_` file names in the ATTO520 folder are labeling slips. The
  blank slides show no vesicles in any channel. (488 and 515 nm on ATTO520 vesicles are close to
  the camera's saturation, about 56,800 ADU per pixel; irrelevant for detection.)
- 2026-11-20: six extraction jobs submitted (S3IT 8604886, 8604887, 8604889 for 488 nm at gradients
  10000, 20000, 40000; 8604890, 8604892, 8604893 for 405 nm).
