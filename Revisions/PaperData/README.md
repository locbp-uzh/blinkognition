# PaperData: the paper's published extraction, checked and indexed

Shared foundation (step 0) of the first-round review topics; plan and decisions in
`../ReviewRound1/README.md`. Reads the paper's data in place and never writes to them; changes no
published analysis or parameter.

## Sources

- Published extraction (S3IT):
  `/shares/locbp.chem.uzh/pabriv/AllRawMovies/GT/FinalTraces/Grx1_HTHTL_HTIA_K20Ac_snap_optparam_001`
  (2026-05-09, Picasso 0.9.3 MLE). Its normalized filtered IN pickles are byte-identical to
  `Inputs/FinalTraces` and the Zenodo FinalTraces. It also holds raw and background-removed
  intensities (filtered and unfiltered), OUT and background traces, and UniqueIDs.
- Raw movies and localizations: `/shares/locbp.chem.uzh/pabriv/AllRawMovies/GT/<session>/`.
- Model outputs (local): Zenodo GeneralData,
  `/Volumes/MediumBerth/SalomePaperData/Püntener2026/GeneralData` (`MLresults/`).

Construct names: HaloD106 = HTHTL, HaloK117 = HTIA, SNAPC148 = snap, scGrx1 = Grx1,
scGrx1AcK20 = K20Ac.

## Files

- `config.yaml`: paths, construct keys, published md5s, camera constants, frame interval.
- `paperdata.py`: loader (`load_traces`, `load_ids`, `to_photons`, `frame_s`, `nd2_camera`) and run
  helpers (`make_run_dir`, `write_manifest`) for all review topics.
- `check_data.py`: step 0 checks and per-trace index (docstring lists every check).
- `s3it/check_data.sbatch`: the CPU job on S3IT (picasso-env).

## Local mirror

`Inputs/PaperData/Grx1_HTHTL_HTIA_K20Ac_snap_optparam_001/` (gitignored), same layout as the
published run, read with `load_config(local=True)`. Holds only what has been copied, each file
checked against the sha256 in check run_002's manifest:

- 2026-11-20: `ProteinTracesIN/Filtered/*_IN_filtered_bg_rm.pkl` (5 files, 0.93 GB; R2.1).

`load_index()` reads step 0's per-trace index CSVs (check run_002), so the UniqueIDs pickles are
not needed locally.

## Units

- Detected photons per frame = bg_rm (ADU) x 16.0 / 300, Picasso's convention (no QE division), with
  the Picasso Localize parameters of the published run (baseline 79, sensitivity 16.0 e-/ADU, EM
  gain 300, QE 0.93, 130 nm pixels). EM gain 300 is confirmed by the ND2 headers; the sensitivity is
  not stored there (readout speed and conversion gain are indices) and is used as entered in Picasso.
- Time: 34.49 ms per frame (ND2 timestamps; 30 ms exposure plus readout), not the 30 ms exposure.

## Outputs

`Results/Revisions/PaperData/check/run_NNN/`: `checks.json`, `funnel.csv` (traces before and after
the quality filter per construct and IN / OUT / background), `sessions.csv` (the same per session),
`rois.csv` (ROIs by overlap and IN/OUT before filtering), `index_<where>_<construct>.csv`
(uniqueID, session, folder, movie, x, y, passed_filter), `camera.csv` (ND2 headers of every 640 nm
movie), `manifest.yaml`, `config.yaml`, `code/`.

## Log

- 2026-11-20: written; tested locally on the 2026-07-20 re-extraction (same layout), Halo-K117:
  structure checks pass (normalizations reproduced to 2e-16, raw minus bg_rm constant per trace,
  filtered values identical to unfiltered ones, IDs complete), md5 checks fail as they should on a
  different extraction. ND2 reader tested on a DFK788 movie: EM gain 300, 30 ms exposure, 34.49 ms
  frame interval, 640 nm at 90 % (read from the per-frame events; the text header's laser line
  belonged to another channel).
- 2026-11-20, check run_001 (S3IT job 8598013, 5 min, 45 GB peak memory): every data check
  passed; the "IDs equal unfiltered traces" check failed for 4 of 5 constructs. Cause:
  `Extraction/combine.py` drops traces with any zero or NaN frame. The check was changed to count
  and flag those IDs (commit 04028d1).
- 2026-11-20, check run_002 (S3IT job 8598156, 6 min, 50 GB peak memory), the reference run:
  all checks pass. Dropped IDs are all traces of 10 whole movies (Halo-D106: Exp4 movies 0029,
  0032; SNAP-C148: Exp4 0048, 0055, 0067, 0105; scGrx1: Exp7 0014, Exp8 0014, Exp9 0031;
  scGrx1-AcK20: Exp6 0006) plus single OUT and background traces in a few others; most likely
  a zero frame in each movie (not verified on the movies). 1303 movie headers read, all with
  EM gain 300, readout index 2, conversion-gain index 0, sensor-mode index 1, 30 ms exposure,
  640 nm at 90 %, 34.493 ms per frame; Exp1-6 have 8000 frames, Exp7-10 6000. Per-session counts
  are in `../ReviewRound1/README.md`.
