# Bleed-through between ATTO390 and ATTO525 vesicle labels

Revision experiment for the blinkognition paper. Question: in a future sample that
mixes ATTO390-labeled and ATTO525-labeled small unilamellar vesicles (SUVs), can
each vesicle be assigned to its label from a single four-channel TIRF snapshot,
given spectral bleed-through between channels?

Work lives on the `revisions` branch only. Nothing from here goes to `main` until
the revisions are merged deliberately.

## Data

Raw data are read in place from `/Volumes/MediumBerth/SalomePaperData/Revisions`
(absolute path in `config.yaml`; nothing is copied). See `20260923_readme.txt` there.

| Slide | Folder | Vesicle label (membrane) | Protein | FOVs |
|---|---|---|---|---|
| ATTO390 | HT7-ATTO390 | ATTO390 | HT7-HMSiR-HTL | 013-017 |
| ATTO525 | SNAP-ATTO525 | ATTO525 | SNAP-HMSiR-IA | 008-012 |

Each ND2 is one 512 x 512 frame, 4 channels, 0.13 um pixels, TIRF 2x:

| Key | ND2 name | Excitation / emission | Laser (ND2 metadata) | Laser (data readme) | Exposure |
|---|---|---|---|---|---|
| 405 | 405 TIRF 2x | 405 / 432 nm | 28.3 % | 23.3 %, 0.7 mW | 100 ms |
| 488 | 488 TIRF 2x | 488 / 525 nm | 6.1 % | 6.1 %, 0.570 mW | 30 ms |
| 640 | 638LP TIRF 2x | 640 / 705 LP | 90.2 % | 90.2 %, 9.1 mW | 30 ms |
| 515 | 515 TIRF 2x | 515 / 525 nm | 5.0 % | 5 %, 0.250 mW | 30 ms |

The 405 power differs between the ND2 metadata (28.3 % in all ten files) and the
data readme (23.3 %); which one the 0.7 mW belongs to is unknown (open question for
the experimenter). Camera: Andor DU-888 (iXon Ultra 888 EMCCD), EM gain 300, 10 MHz
readout, conversion gain 1, 1x1 binning, identical in every file. Every flux ratio
below is specific to these laser powers and exposures; `segment.py` records
exposure, EM gain and lasers per FOV in `acquisition.csv` and warns if they differ.

The camera is not linear to 65,535 counts: pixels read about 5 % low at
30,000-40,000 counts, about 14 % low above 40,000, and clip near 56,800 (two
ATTO525 FOVs reach 56,857 and 56,119). Objects with any aperture pixel at or above
30,000 counts are flagged `nonlinear` (`camera.nonlinear_above_counts`).

Slides were not washed. Each slide is one sample: its FOVs are repeated
measurements of the same sample, not independent replicates.

Excluded: ATTO390/HT7-ATTO390_017 (preliminary test, FOV not measured carefully:
half the vesicle density, ATTO390 about 2.5x dimmer, many aggregates). Listed with
its reason under `exclude_fovs` in `config.yaml` (keys are `<slide>/<file stem>`),
which every stage honors. Decision 2026-09-24; similar FOVs in the main datasets
will be handled when they arise.

## Approach

1. Channel QC (`qc_channels.py`): cross-channel offsets and spot width.
2. Segmentation (`segment.py`): detect vesicles once per FOV on the per-pixel
   maximum of the noise-normalized, matched-filtered 405 / 488 / 515 images, then
   measure every vesicle in all four channels at the same position (aperture flux
   minus local annulus median). Segmenting each channel on its own would only find
   the ATTO525 vesicles bright enough to bleed into 405, and the bleed-through
   estimate would then depend on the detection threshold. Empty apertures (blanks)
   are measured the same way to calibrate the zero point and noise.
3. Bleed-through coefficients (`coefficients.py`, analysis step 1).
4. Classification: fit a mixture model to the pooled vesicles of both slides (an
   in-silico mix whose true labels are known from the slide), and score the
   assignments against the slide labels. Pending, see "Open decisions".

Vesicle brightness varies with vesicle size (membrane dye scales with membrane
area), so a single-channel intensity cannot separate the labels: a large ATTO525
vesicle leaking into 405 can match a small ATTO390 vesicle. Bleed-through is a
fixed fraction of the emitting dye's signal, so the size-independent feature is the
channel ratio.

## Running

Environment: the existing `blink2env` (numpy, scipy, scikit-learn, pandas,
matplotlib, nd2, pyyaml). No extra packages. Run from the repo root:

```bash
python Revisions/Bleedthrough/qc_channels.py
python Revisions/Bleedthrough/segment.py
python Revisions/Bleedthrough/coefficients.py
python Revisions/Bleedthrough/features.py
```

`segment.py` and `coefficients.py` accept `--set key.path=value` overrides (the
value keeps the type of the config entry it replaces); `coefficients.py` reads the
segmentation run named by `bleedthrough.segmentation_run`, or `--seg-run DIR`. Each
run writes `Results/Revisions/Bleedthrough/<stage>/run_NNN/` with a `manifest.yaml`
(git commit and dirty flag, SHA-256 of every input and of every module), the
effective config and a copy of all modules in `code/`.

Reference runs: segmentation run_005, coefficients run_005, features run_002,
qc_channels run_003.
Earlier runs are kept as records and are superseded (see "Run history").

### vesicles.csv columns

| Column | Meaning |
|---|---|
| slide, dye, fov, vesicle_id | slide, its dye (ground-truth label), ND2 file stem, index within FOV |
| y_px, x_px | subpixel position (centroid in the channel that detected it) |
| detected_by | detection channel with the highest SNR at the peak |
| snr_detect, snr_405/488/515 | matched-filter SNR at the peak (combined and per channel) |
| nn_dist_px, crowded | distance to nearest vesicle; flag if < 8 px (1.04 um) |
| flux_<ch> | aperture sum (r = 3 px) minus n_aperture x annulus median (r = 5-8 px), camera counts |
| flux_err_<ch> | annulus robust sigma x sqrt(n aperture px), background noise only (underestimates, see below) |
| bg_<ch> | annulus median (counts per pixel) |
| peak_raw_<ch> | highest raw pixel in the aperture |
| nonlinear | any aperture pixel >= camera.nonlinear_above_counts in any channel |

`blanks.csv` has the same photometry columns for 300 random positions per FOV at
least 8 px from any detected vesicle (per-FOV seeded RNG keyed on slide and FOV, so
positions do not change when other FOVs are excluded). `acquisition.csv` has
exposure, EM gain and lasers per FOV and channel. FOVs are identified by
(slide, fov) everywhere, since file stems can repeat between slide folders.

## Results

### Channel QC (qc_channels run_003)

- Spot sigma 1.5-1.8 px (195-235 nm); `psf_sigma_px` set to 1.6.
- 488 vs 515: offset below 0.15 px on both slides. No correction needed.
- 405 vs 488 / 515: dx about -0.75 to -0.9 px on the ATTO390 slide (every FOV),
  but only -0.1 to -0.2 px on the ATTO525 slide. The offset follows the emitting
  dye, not the slide position: lateral chromatic shift between the 432 nm and
  ~525 nm emission bands (the ATTO525 light reaching the 405 channel is presumably
  long-wavelength leakage). The 5x5 centroid slightly pulls positions toward whole
  pixels (about 0.05-0.1 px at these offsets), so the offsets are approximate.
- 640 has too few spots to register (5-23 per FOV).

### Populations (exploratory, segmentation run_001, includes FOV 017, no zero point)

Kept because it describes what is on the slides; the numbers are superseded.

| Slide | Population | n | F405 | F488 | F515 |
|---|---|---|---|---|---|
| ATTO390 | main ATTO390 vesicles | 2361 | 33 800 | 535 | 229 |
| ATTO525 | main ATTO525 vesicles | 520 | 3 780 | 24 700 | 58 000 |
| ATTO390 | dim, 515-only | 161 | ~0 | 478 | 667 |
| ATTO525 | dim, 515-only | 75 | ~0 | 343 | 1 040 |
| ATTO525 | 405-only | 34 | 8 900 | 644 | 207 |

1. The per-vesicle 405/515 ratio of the two main populations differs by a factor
   of about 2500 (3.4 decades).
2. The limiting problem for the mixing experiment is not bleed-through but
   off-population objects on each slide: about 25 dim 515-positive, 405-negative
   objects per ATTO390 FOV (6 %), which would be called ATTO525 in a mix, and 4-9
   405-positive, 515-negative objects per ATTO525 FOV (5 %), which would be called
   ATTO390. Most likely impurities from the unwashed slides (user, 2026-09-24).
3. 640 (protein) shows almost no signal at vesicles: HMSiR is a spontaneously
   blinking dye, mostly dark in any single frame.

### Empty-aperture control (segmentation run_003)

Median over FOVs of the per-FOV blank statistics (camera counts):

| Slide | Channel | Blank median flux | Blank robust SD | Predicted flux_err | SD / flux_err |
|---|---|---|---|---|---|
| ATTO390 | 405 | -387 | 1931 | 1401 | 1.38 |
| ATTO390 | 488 | 186 | 309 | 252 | 1.22 |
| ATTO390 | 515 | 130 | 103 | 59 | 1.76 |
| ATTO390 | 640 | 147 | 771 | 610 | 1.26 |
| ATTO525 | 405 | 198 | 1567 | 1237 | 1.27 |
| ATTO525 | 488 | 143 | 332 | 263 | 1.26 |
| ATTO525 | 515 | 162 | 205 | 131 | 1.56 |
| ATTO525 | 640 | 204 | 682 | 585 | 1.17 |

1. Zero point. Empty apertures do not read zero. The positive offsets are the
   EMCCD skew: n_aperture x (annulus mean - annulus median) at the blank positions
   reproduces them (for example 137 vs 130 counts in 515 on the ATTO390 slide).
   The negative 405 offset on the ATTO390 slide is neighbor light in the annuli of
   the dense field. Every analysis subtracts the per-FOV, per-channel blank median.
2. Noise. flux_err underestimates the real scatter of blank fluxes by 1.2-1.8x, so
   the SNR values in vesicles.csv are inflated. Analyses use the per-FOV blank
   robust SD instead.

### Step 1: bleed-through coefficients (coefficients run_005, segmentation run_005)

Method (agreed 2026-09-24, options A and A): fluxes zero-point corrected with the
per-FOV, per-channel blank median; vesicles with home-channel signal >= 10 blank
SDs, not crowded, not nonlinear (1587 of 2417 ATTO390, 502 of 706 ATTO525). Per
FOV, k = median of per-vesicle ratios; reported as mean +- SD across FOVs (n = 4
ATTO390, 5 ATTO525). Least-squares slope through the origin (LS) as a cross-check.

| Dye | Into | k (mean +- SD of FOVs) | LS cross-check |
|---|---|---|---|
| ATTO390 | 488 | 0.86 +- 0.10 % | 1.049 +- 0.070 % |
| ATTO390 | 515 | 0.225 +- 0.023 % | 0.394 +- 0.052 % |
| ATTO390 | 640 | 0.01 +- 0.12 % | 0.10 +- 0.12 % |
| ATTO525 | 405 | 6.13 +- 0.78 % | 6.21 +- 0.83 % |
| ATTO525 | 488 | 42.7 +- 1.8 % | 43.9 +- 1.7 % |
| ATTO525 | 640 | 0.143 +- 0.096 % | 0.214 +- 0.068 % |

Figure: `coefficients/run_005/bleedthrough.pdf` (caption draft in caption.txt;
vesicle composition, dye mol%, buffer and temperature still to be added).

- Proportionality holds: binned medians of the ratio are flat across the home
  brightness range for every pair.
- Into 640, k cannot be separated from genuine protein signal, which also scales
  with vesicle size; both are at most about 0.1-0.2 % of the label signal.
- ATTO525 into 405: FOV 008 is at 7.4 %, the other four at 5.4-6.2 %.
- Aperture: a 4 px aperture raised the ATTO390 terms by about 5 % and moved the
  ATTO525 terms by under 1 % (on segmentation run_003/run_004), within the
  FOV-to-FOV SD, so the 3 px aperture stays (rule agreed in advance).

Why LS disagrees with the median for ATTO390 (verification, 2026-09-24): about
7 % of selected ATTO390 vesicles (115 of 1564 in run_003, 26-32 per FOV) carry an
extra co-localized emitter with an ATTO525-like spectrum (excess 488/515 about 0.57,
like the free 515-only objects, against 3.9 for ATTO390 bleed). It adds a roughly
constant 900-1500 counts in 515 regardless of vesicle brightness and is enriched on
bright vesicles, which LS weights by F405^2. Not camera nonlinearity (clean bright
objects show no rise), not aggregates (normal spot width), not photometry (annulus
and aperture variants give the same), not crowding. The ATTO525 into 488
disagreement in run_002 came from two clipped, very bright objects and disappears
once `nonlinear` objects are excluded. LS is therefore a diagnostic here, not a
coefficient.

Systematic uncertainty (report, do not correct): the 3 px aperture ratio
underestimates the total-flux ATTO390 coefficients by about 5 % (chromatic shift)
plus 0-9 % (the bleed channels have a wider PSF); the mixed ATTO390 objects push
the ATTO390 k up by at least 3 % (488) and 9 % (515). Net about +-10 % on the
ATTO390 terms and a few percent on the ATTO525 terms. Negligible for
classification, where the populations are 3.4 decades apart.

### Step 2: model features (features run_002, segmentation run_005)

Method (agreed 2026-09-24, options A and A): channels 405, 488 and 515 (640 is the
protein, not a label); per object and channel z = (flux - blank median) / blank
robust SD, per FOV, and u = asinh(z / 5). Every detected object is kept (3123:
2417 ATTO390 slide, 706 ATTO525 slide), with its crowded and nonlinear flags.
Checked against its definition from the raw segmentation outputs (bit-identical,
no missing values). Figure: `features/run_002/feature_space.pdf`.

- The expected pure-dye curves from the step 1 coefficients run through the middle
  of both populations, so the feature space behaves as the physics predicts.
- In 405 vs 515 the two labels form separate bands, apart from objects near the
  noise.
- Three groups the model will have to deal with: dim 515-only objects from both
  slides at 3-5 noise SDs in 515, next to the empty-aperture cloud; the mixed
  ATTO390 objects spraying toward higher 515 and 488; and dim ATTO390 vesicles
  whose 405 signal is within a few noise SDs.

### Verification of step 1 (workflow, 2026-09-24)

Run on coefficients run_002 / segmentation run_003 (before the fixes below).

1. Independent recomputation from the method description, without reading
   coefficients.py: all 19,014 numbers reproduced to the bit.
2. Known-answer injection: synthetic vesicles with known k injected into the real
   images (real background, EMCCD noise on the injected photons, g = 14 counts per
   photoelectron from the mean-variance slope), measured with the project's own
   code including full detection. The median-ratio estimator recovers k within
   about 2 % for all four terms; the chain itself creates neither the A/B gap nor a
   brightness trend. Without the zero-point correction ATTO390 into 515 is +110 %
   and into 488 +42 %, so the correction is essential. The chromatic shift costs
   5.3 % of the shifted channel's flux at 3 px (2.2 % at 4 px).
3. A vs B investigation: see above.
4. Code review with adversarial check of each finding: 16 findings, 15 confirmed.
   Fixed: coefficients ignored `exclude_fovs`; FOV identity was the bare file stem
   (clashes across slides, shared blank RNG); slide and dye were one key (replicate
   slides silently dropped); the caption was hard-coded; the manifest copied only
   the entry script; `--set` could turn '405' into an int; zero-vesicle or
   zero-blank FOVs crashed; single-FOV summaries reported NaN SD as "agree"; the
   saturation flag never fired (camera clips near 56,800); laser power was not
   recorded; blanks could sit 7.3 px from a vesicle; border 0 cleared the whole
   mask; the annulus could exceed the border; figure issues (clipped marks, wrong
   off-axis counts, legend styles, sub-5 pt text, width over 180 mm). Not fixed,
   judged minor: the 5x5 centroid's pull toward whole pixels.
5. Side finding, fixed: the detection noise estimate (MAD of the filtered image)
   grew with object density, so dense FOVs had a higher effective threshold (450
   injected spots doubled it). Detection now uses an iteratively clipped MAD (+7 %
   for the same test); run_005 finds 3123 vesicles instead of 2944.

Scripts and outputs of the verification are in the session scratchpad and not
kept in the repo; their conclusions are recorded here.

## Decisions

- 2026-09-24: FOV HT7-ATTO390_017 excluded (see Data).
- 2026-09-24: the classification model must allow for objects that are neither
  ATTO390 nor ATTO525 vesicles. The odd objects here are most likely impurities
  from the unwashed slides, but similar objects may appear in cleaner datasets.
- 2026-09-24: step 1 uses the per-FOV median of per-vesicle ratios, summarized as
  mean +- SD across FOVs (options A and A); 3 px aperture kept after the 4 px check.
- 2026-09-24: step 2 features are asinh(z / 5) of 405, 488 and 515, all detected
  objects kept with flags (options A and A).

## Open decisions

- Classification analysis, agreed one step at a time before any code runs:
  1. bleed-through coefficients and their uncertainty (done)
  2. features and scaling fed to the model (done)
  3. model structure (mixture with odd-object components, unassigned option).
     Must also handle the mixed ATTO390 objects (ATTO390 plus an ATTO525-like
     emitter) found in step 1, which will look dual-labeled in a real mix.
  4. evaluation on the in-silico mix
- Which 405 laser power is correct (28.3 % in the metadata, 23.3 % in the readme).

## Run history

- segmentation run_001 (includes FOV 017), run_002/run_003 (FOV 017 excluded, blanks;
  run_003 adds acquisition.csv), run_004 (4 px aperture): superseded by run_005
  (clipped detection noise, nonlinear flag, (slide, fov) keys, laser record).
- coefficients run_001 (panel order bug), run_002 (verified), run_003 (4 px): superseded
  by run_005 (run_004 is identical to run_005 except for the legend layout).
- qc_channels run_001/run_002: superseded by run_003 (clipped detection noise).
- features run_001: axes extended past the data; superseded by run_002.

## Not verified

- The fixes after the verification were checked with targeted tests (each failure
  case now passes) and a full rerun, not by a second independent review.
- Photometry treats the background as the only noise; bright vesicles are noisier
  than the blank SD says.
- No camera linearity calibration (photon transfer or exposure series); the
  30,000-count limit and the clipping level are inferred from the data.
- The identity of the 515 emitter on ATTO390 vesicles and of the odd objects.
- All numbers come from one slide per label; slide-to-slide variation is unmeasured.
