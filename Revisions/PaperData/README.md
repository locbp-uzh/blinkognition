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
  frame interval, 640 nm at 90 %.
