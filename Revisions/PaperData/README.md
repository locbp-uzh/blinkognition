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

## Photon conversion check (conversion run_001, 2026-11-20)

`conversion_check.py` (choices confirmed by the user): mean-variance analysis of 20 raw movies (two
per session): 640 nm background pixels (frames 4000-6000, > 5 px from every published
localization) and 488 nm snapshot pixels below 20,000 ADU, noise variance from consecutive-frame
differences, binned weighted linear fit; photons per ADU = 2 / slope (EM excess noise factor F^2 = 2).

| Pixels | Photons per ADU, median of 20 movies (range) | Implied sensitivity at gain 300 |
|---|---|---|
| all (main) | 0.062 (0.059-0.065) | 18.6 e-/ADU |
| 640 nm background | 0.062 (0.058-0.063) | 18.5 |
| 488 nm snapshots | 0.059 (0.057-0.062) | 17.6 |
| config (Picasso) 16.0 / 300 | 0.053 | 16.0 |

The binned pixels of every movie lie on one line from about 600 to 20,000 ADU
(`conversion_fits.pdf`); stable across 2023-2025 sessions (SP 0.060-0.062, DK 0.060-0.065, all
pixels). The empirical factor is 10-16 % above the configured one: detected photon numbers computed
with 16.0 / 300 (R2.1, R3.3, R3.2 photons) would be 10-16 % higher with the empirical factor; ratios,
rankings and all non-photon results are unaffected. The fits cross zero variance near 530 ADU, the
camera's per-pixel offset (Picasso's baseline is 79); irrelevant for the bg_rm photon numbers, which
subtract each trace's own background. Caveat: F^2 = 2 is the high-gain limit; clock-induced charge
and any departure of F^2 shift the factor by a few percent.

Decision (user, 2026-11-20): the revision reports detected photons with the measured 0.062 photons
per ADU (`config.yaml` camera.photons_per_adu, used by `to_photons`), not Picasso's 16.0 / 300. The
paper itself reports no photon numbers and no camera values, so only the revision's new numbers
change (all photon values x 1.1625 against the first versions of R2.1, R3.3a, R3.3b and R3.2).

## Local mirror

`Inputs/PaperData/Grx1_HTHTL_HTIA_K20Ac_snap_optparam_001/` (gitignored), same layout as the
published run, read with `load_config(local=True)`. Holds only what has been copied, each file
checked against the sha256 in check run_002's manifest:

- 2026-11-20: `ProteinTracesIN/Filtered/*_IN_filtered_bg_rm.pkl` (5 files, 0.93 GB; R2.1).

`load_index()` reads step 0's per-trace index CSVs (check run_002), so the UniqueIDs pickles are
not needed locally.

## Units

- Detected photons per frame = bg_rm (ADU) x 0.062, the photons per ADU measured from the raw movies'
  noise (see Photon conversion check; chosen by the user 2026-11-20), no QE division. The Picasso
  Localize parameters of the published run (baseline 79, sensitivity 16.0 e-/ADU, EM gain 300, QE
  0.93, 130 nm pixels) give 0.053; they stay in `config.yaml` for the record. EM gain 300 is confirmed
  by the ND2 headers; the sensitivity is not stored there (readout speed and conversion gain are
  indices). Until 2026-11-20 (events run_001 and the first figure versions) the revision used 0.053.
- Time: 34.49 ms per frame (ND2 timestamps; 30 ms exposure plus readout), not the 30 ms exposure.
  The camera did not run in frame-transfer mode (user, 2026-11-20): each frame is 30 ms of exposure
  followed by 4.49 ms of readout with no light recorded. Photons per fully-ON frame are photons per
  30 ms of emission; ON periods that fall entirely within a readout gap are not seen.

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
- 2026-11-20, check run_003 (S3IT job 8602497), now the reference (`config.yaml` step0_run): run_002
  left background traces without a session, because background UniqueIDs keep the source path in
  `file_origin` (their `origin` is just "background"). `load_ids` now reads `file_origin` when
  present, and check_data warns on IDs without a session. All checks pass; only the background
  index files and sessions.csv differ from run_002 (IN and OUT indices byte-identical).
- 2026-11-20, `edge_groups` run_001 (S3IT job 8608997, 1 min): molecules the published extraction
  dropped for touching the movie's first frame. `Extraction/utils.py` get_and_link_locs calls
  Picasso's `postprocess.link` with its default `remove_ambiguous_lengths=True`, which removes every
  linked group starting in frame 0 (or ending at or after the frame count; none did). With
  max_darktime 4000 a group is nearly a whole molecule, so a molecule ON in frame 0 is never
  extracted. Found in R3.1, where it removed a blank-surface field's only spot. Replication exact:
  linked groups equal the published ROIs per construct and session, and the recomputed single IN
  ROIs equal step 0's (`checks.json`). Dropped groups (all start in frame 0) and the single IN
  ROIs among them (overlap recomputed with the dropped groups put back), against the published
  single IN ROIs:

  | Construct | Groups dropped | Of all groups | Single IN dropped | Of published single IN |
  |---|---|---|---|---|
  | Halo-D106 | 1624 | 7.7 % | 1086 | 17.7 % |
  | Halo-K117 | 215 | 2.4 % | 125 | 2.6 % |
  | SNAP-C148 | 1242 | 4.2 % | 705 | 4.5 % |
  | scGrx1 | 701 | 1.4 % | 183 | 1.4 % |
  | scGrx1-AcK20 | 379 | 1.9 % | 99 | 2.3 % |

  Same days, Halo-D106 against Halo-K117: Exp1 8.6 % vs 0.4 %, Exp2 27.3 % vs 3.8 % (D106 Exp4
  23 %). The loss depends on the construct: Halo-D106 molecules are more often ON in the first frame
  (longer ON, higher duty cycle), so its published population lacks some of its most-ON molecules.
  Direction: probably conservative for D106 vs K117 (the missing molecules are the most D106-like),
  but not tested; how many would pass the quality filter and how the models would call them is
  unknown without extracting their traces. Reported to the user; nothing changed.
