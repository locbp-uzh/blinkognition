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

| Key | ND2 name | Excitation / emission | Laser |
|---|---|---|---|
| 405 | 405 TIRF 2x | 405 / 432 nm | 23.3 %, 0.7 mW |
| 488 | 488 TIRF 2x | 488 / 525 nm | 6.1 %, 0.570 mW |
| 640 | 638LP TIRF 2x | 640 / 705 LP | 90.2 %, 9.1 mW |
| 515 | 515 TIRF 2x | 515 / 525 nm | 5 %, 0.250 mW |

Camera (from the ND2 metadata): Andor DU-888 (iXon Ultra 888 EMCCD), EM gain 300,
10 MHz readout, 1x1 binning, identical in every file. Exposure is 100 ms for 405 and
30 ms for 488, 640 and 515. Every flux ratio below is specific to these laser powers
and exposures; `segment.py` writes them to `acquisition.csv` and warns if FOVs differ.

Slides were not washed. Each slide is one sample: its FOVs are repeated
measurements of the same sample, not independent replicates.

Excluded: HT7-ATTO390_017 (preliminary test, FOV not measured carefully: half the
vesicle density, ATTO390 about 2.5x dimmer, many aggregates). Listed with its reason
under `exclude_fovs` in `config.yaml`, which every stage honors. Decision 2026-09-24;
similar FOVs in the main datasets will be handled when they arise.

## Approach

1. Channel QC (`qc_channels.py`): cross-channel offsets and spot width.
2. Segmentation (`segment.py`): detect vesicles once per FOV on the per-pixel
   maximum of the noise-normalized, matched-filtered 405 / 488 / 515 images, then
   measure every vesicle in all four channels at the same position (aperture flux
   minus local annulus median). Segmenting each channel on its own would only find
   the ATTO525 vesicles bright enough to bleed into 405, and the bleed-through
   estimate would then depend on the detection threshold.
3. Bleed-through coefficients: flux ratios of each dye into the "wrong" channels.
4. Classification: fit a mixture model to the pooled vesicles of both slides (an
   in-silico mix whose true labels are known from the slide), and score the
   assignments against the slide labels. Pending, see "Open decisions".

Vesicle brightness varies with vesicle size (membrane dye scales with membrane
area), so a single-channel intensity cannot separate the labels: a large ATTO525
vesicle leaking into 405 can match a small ATTO390 vesicle. Bleed-through is a
fixed fraction of the emitting dye's signal, so the size-independent feature is the
channel ratio. On log axes the two labels form parallel diagonal bands.

## Running

Environment: the existing `blink2env` (numpy, scipy, scikit-learn, pandas,
matplotlib, nd2, pyyaml). No extra packages. Run from the repo root:

```bash
python Revisions/Bleedthrough/qc_channels.py
python Revisions/Bleedthrough/segment.py
```

Each run writes `Results/Revisions/Bleedthrough/<stage>/run_NNN/` with a
`manifest.yaml` (git commit and dirty flag, package versions, SHA-256 of every
input ND2) and copies of the config and script.

### vesicles.csv columns

| Column | Meaning |
|---|---|
| slide, fov, vesicle_id | ground-truth label (slide), FOV, index within FOV |
| y_px, x_px | subpixel position (centroid in the channel that detected it) |
| detected_by | detection channel with the highest SNR at the peak |
| snr_detect, snr_405/488/515 | matched-filter SNR at the peak (combined and per channel) |
| nn_dist_px, crowded | distance to nearest vesicle; flag if < 8 px |
| flux_<ch> | aperture sum (r = 3 px) minus 28 px x annulus median (r = 5-8 px), camera counts |
| flux_err_<ch> | annulus robust sigma x sqrt(n aperture px), background noise only |
| bg_<ch> | annulus median (counts per pixel) |
| saturated | any aperture pixel at the camera maximum in any channel |

`blanks.csv` has the same flux, flux_err and bg columns for 300 random positions per
FOV at least 8 px from any detected vesicle (per-FOV seeded RNG, so positions do
not change when other FOVs are excluded). `acquisition.csv` has exposure and EM gain
per FOV and channel.

## Results so far (2026-09-24)

### Channel QC (qc_channels run_002)

- Spot sigma 1.5-1.8 px (195-235 nm); `psf_sigma_px` set to 1.6.
- 488 vs 515: offset below 0.15 px on both slides. No correction needed.
- 405 vs 488 / 515: dx = -0.75 to -1.05 px on the ATTO390 slide (every FOV), but only
  -0.1 to -0.3 px on the ATTO525 slide. The offset follows the emitting dye, not
  the slide position, which points to lateral chromatic shift between the 432 nm
  and ~525 nm emission bands (the ATTO525 light reaching the 405 channel is then
  presumably long-wavelength leakage). With a 3 px aperture and sigma 1.6 px, a
  0.9 px shift loses roughly 5 % of the flux in the shifted channel: irrelevant for
  classification, a small bias on the bleed coefficients. Not yet checked with a
  larger aperture.
- 640 has too few spots to register (5-23 per FOV).

### Segmentation, first look (segmentation run_001, includes FOV 017)

3242 vesicles: 2593 on the ATTO390 slide (about 570 per FOV, except 298 in FOV 017)
and 649 on the ATTO525 slide (107-143 per FOV). None saturated.

The fluxes and ratios in this subsection are NOT zero-point corrected and are
superseded by the empty-aperture control below; they overstate the small bleed
terms. Kept because they describe the populations that exist.

Median values per population (fluxes in camera counts; the ratio column is the
median of the per-vesicle ratio):

| Slide | Population | n | F405 | F488 | F515 | F515/F405 or F405/F515 |
|---|---|---|---|---|---|---|
| ATTO390 | main ATTO390 vesicles | 2361 | 33 800 | 535 | 229 | F515/F405 = 0.006 |
| ATTO525 | main ATTO525 vesicles | 520 | 3 780 | 24 700 | 58 000 | F405/F515 = 0.062 |
| ATTO390 | dim, 515-only | 161 | ~0 (z 0.3) | 478 | 667 | not ATTO390 |
| ATTO525 | dim, 515-only | 75 | ~0 | 343 | 1 040 | dim tail of ATTO525? |
| ATTO525 | 405-only | 34 | 8 900 | 644 | 207 | F515/F405 = 0.02 |

Population cuts here are exploratory (asinh-scaled flux boxes), only to describe
what is there; they are not the classifier.

Observations:

1. Bleed-through between the two main populations is small compared with their
   separation. ATTO525 puts about 6 % of its 515 flux into 405; ATTO390 puts about
   0.6 % of its 405 flux into 515 and 1.6 % into 488 (at the laser settings above).
   The per-vesicle 405/515 ratio of the two main populations differs by a factor
   of about 2500 (3.4 decades).
2. The limiting problem for the mixing experiment is not bleed-through but
   off-population objects on each slide:
   - ATTO390 slide: about 25 dim 515-positive, 405-negative objects per FOV (6 % of
     objects). They cannot be ATTO390 bleed-through (no 405 signal) and would be
     called ATTO525 in a mix. Their 515 flux overlaps the dim ATTO525 objects.
   - ATTO525 slide: 4-9 405-positive, 515-negative objects per FOV (5 %), about 4x
     dimmer in 405 than typical ATTO390 vesicles. They would be called ATTO390.
   Whether these are cross-contamination, dye aggregates or fluorescent debris is
   unknown from these data.
3. FOV HT7-ATTO390_017 is unlike the other four ATTO390 FOVs: half the vesicles,
   the ATTO390 population about 2.5x dimmer in 405, and many more 515-positive and
   mixed objects (plus bright broadband aggregates in the QC image).
4. 640 (protein) shows almost no signal at vesicles (see the empty-aperture control
   below). Consistent with HMSiR being a spontaneously blinking dye mostly in its
   dark form in a single frame; the protein cannot be assessed from these
   snapshots. There is no detectable ATTO390 or ATTO525 bleed into 640.

### Empty-aperture control (segmentation run_003, FOV 017 excluded)

2944 vesicles (2295 ATTO390 slide, 649 ATTO525 slide) and 2700 blanks. Vesicles and
blanks are identical to run_002; run_003 only adds `acquisition.csv`.

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

1. Zero point. Empty apertures do not read zero. The positive offsets fit the
   EMCCD: at EM gain 300 the pixel distribution is right-skewed, so the annulus
   median sits below the mean background and every aperture gains a positive
   offset. The negative 405 offset on the ATTO390 slide fits annulus contamination
   in the dense field (blank annuli overlap nearby vesicles). The offset matters
   only where the signal is small: the uncorrected ATTO390 bleed into 515 (median
   229 counts) is more than half offset (130), and into 488 (535) about a third
   (186). Rough corrected values: ATTO390 into 515 about 0.3 % and into 488 about
   1.0 % of its 405 flux; ATTO525 into 405 still about 6 % of its 515 flux. To be
   recomputed properly in analysis step 1.
2. Noise. flux_err (annulus sigma x sqrt(n)) underestimates the real scatter of
   blank fluxes by 1.2-1.8x (correlated EMCCD noise and background structure), so
   the SNR values in vesicles.csv are inflated. The blanks give the honest
   per-channel noise.
3. 640. Vesicles sit barely above blanks: 3.2 % (ATTO390 slide) and 5.4 %
   (ATTO525 slide) of vesicles exceed 3x flux_err in 640, against 1.1 % and 2.0 %
   of blanks. A faint protein signal on a few percent of vesicles at most.

Planned use: subtract the per-FOV, per-channel blank median from every vesicle
flux, and use the per-FOV blank robust SD as the noise of each channel.

## Decisions

- 2026-09-24: FOV HT7-ATTO390_017 excluded (see Data).
- 2026-09-24: the classification model must allow for objects that are neither
  ATTO390 nor ATTO525 vesicles. The odd objects here are most likely impurities
  from the unwashed slides, but similar objects may appear in cleaner datasets.

## Open decisions

- Classification analysis, agreed one step at a time before any code runs:
  1. bleed-through coefficients and their uncertainty
  2. features and scaling fed to the model
  3. model structure (mixture with odd-object components, unassigned option)
  4. evaluation on the in-silico mix

## Not verified

- Flux sensitivity to aperture size (5 % chromatic-shift bias estimate is analytic).
- Photometry treats the background as the only noise (no shot noise in flux_err);
  the blank SD is also background-only, so bright vesicles are noisier than it says.
- Blank positions are at least 8 px from detected vesicles, so undetected dim
  objects can sit in blank apertures; the median is used to limit their effect.
- All numbers come from one slide per label; slide-to-slide variation is unmeasured.
