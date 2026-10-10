# Training discussion notes (DualColor)

Step-by-step discussion with the user about how to train and evaluate the HT vs SNAP model
on the dual-color datasets. One entry per agreed point, with its reasoning; nothing here is
run until the user gives the green light. Results of the first round of six models (Daint
jobs 5010462-3, 2026-10-09) are in README.md, "Models".

## Agreed

1. Training-set radius: 4 px (2026-10-09). Traces on the single-protein slides count as
   in a vesicle within 4 px of an own-dye vesicle; the mixed-slide test set stays at 2 px.
   The 2 px training variant is dropped. Evidence (one run each, so small differences are
   within noise): mixed AUC 0.72 [0.66-0.77] at 4 px vs 0.70 [0.65-0.75] at 2 px; within-FOV
   AUC 0.76 vs 0.71; balanced accuracy of the kept traces 0.77 vs 0.74; validation AUC 0.80
   vs 0.77; the 2 px model is skewed toward SNAP (recall HT 0.47, SNAP 0.83). 4 px gives
   912 training traces against 782.

2. Early stopping exactly as in the paper (2026-10-09): patience 10 epochs (SI, "Combined
   early stopping"; published CV config patience_limit 10), the paper's combined criterion
   (checkpoint when AUC improves >= 0.005 with loss <= 1.10 x best, or AUC within 0.003 of
   best with loss improved >= 0.005), max 500 epochs. The first round used patience 15
   (from the repo's ML/config_train.yaml, which disagrees with the SI).

3. Validation balanced as in the paper (2026-10-09): the larger class of the validation set
   is subsampled (ML/utils._balance_dataset, as balance_val: true); the traces left out are
   marked 'val_unused' in data_split.csv and are neither trained on nor validated.
   Primary setup: 134 HT / 119 SNAP -> 119 / 119; holdout DFK785: 93 / 35 -> 35 / 35.

4. Augmentation sweep as in the paper, time invariance kept (2026-10-09): factors 0, 3, 5
   with the paper's published CV parameters (time_warp_sigma 0.5, noise_sigma 0.5,
   magnitude_jitter 0.5; warp and jitter clipped to +-5 %; mirror), for the primary setup and
   the three holdouts (12 runs, ml/aug_sweep/r2_*.yaml), one seed (840410) to see first
   whether the effect is worth several seeds. Training traces: 912 -> 1824 / 3648 / 5472.
   Rule fixed before the runs: per setup, the factor with the highest validation AUC (MC
   mean, balanced validation set of held-out FOVs) is the chosen one; all factors are
   reported on the mixed slides and holdout slides.

   Result (Daint jobs 5013068, 5013069, 5013182; README.md "Round 2"): the rule chooses
   factor 0 (all), 0 (holdout DFK785, a collapsed model), 5 (holdout DFK788), 3 (holdout
   DFK789). Mixed AUC of the primary setup 0.71 / 0.72 / 0.71 at factors 0 / 3 / 5; only
   holdout DFK788 gains (+0.02 to +0.08 AUC on all sets), and the round-1 to round-2 shift at
   factor 0 (patience and validation balancing only) is as large (-0.04 to -0.05). Holdout
   DFK785 does not train at any factor. One seed; the effect is within run-to-run spread.

5. The paper's other two models and the paper's controls (requested 2026-10-09, before the
   k-fold; the CNN-GRU controls agreed the same day; the details are Claude's reading of the
   paper, revised after an independent review, Results/Revisions/DualColor/checks/
   20261009_round3_design_review). Configs: ml/models_controls/r3_*.yaml.

   Models. ResNet1D and TCN with the model zoo's defaults (the paper's CV configs only name
   the model), trained with the recipe chosen in round 2 (r2_all_aug0: 4 px, patience 10,
   balanced validation, mirror and time invariance, factor 0), primary setup only, so that
   the architecture is the only change. Receptive fields: TCN 61 frames (causal), ResNet1D
   133 frames, both then averaged over the trace; the CNN-GRU integrates the whole trace. A
   difference to the CNN-GRU is therefore not a pure capacity effect.

   Controls, for all three architectures (the CNN-GRU as the reference, as in the paper):
   - Label scrambling (SI): the classes of the training pool (training and validation
     traces) permuted before the FOV split; class counts kept; FOV split and training seed
     unchanged; the mixed-slide truth not scrambled. Five permutations per architecture
     (scramble_seed 1-5), since one scrambled model is one draw of an arbitrary function of
     the traces and its mixed-slide AUC can lie far from 0.5 (untrained TCNs: 0.36-0.61).
     Readout: the scrambled validation AUC (selection-biased upward by the checkpoint rule)
     and mixed-slide AUC across the five permutations; the real model's mixed AUC against
     that spread. A trace-level shuffle tests the pipeline (leakage, evaluation), not
     slide-level confounds: after it each slide is 40-58 % HT.
   - Background training (the paper's "noise-only training", Figure S13, on which it rejected
     ResNet and TCN for Halo-D106 vs SNAP-C148): the same model and recipe on background
     traces labeled with the slide's protein. Background traces: random 5 x 5 boxes with a
     gap of >= 3 px to every protein box of the movie (the extraction's 5 px buffer lets boxes
     touch, and touching boxes show the neighbor's blinking), spike-free (robust score <= 6),
     minmax. Matched to the protein run: same FOV split, and per dataset, slide and set as
     many traces as protein traces (912 train, 119 / 119 validation). Readout on the held-out
     background of the validation FOVs (val_unused, not used for the checkpoint): per-dataset
     and within-dataset AUC, two-sided, and the slide pairs (below); the scrambled runs'
     validation AUCs as the empirical null of "best epoch on these FOVs".
   - Background inference (the paper's "noise classification with pre-trained model",
     Figure S24B): the protein model classifies every background trace, one deterministic
     pass (the paper's; MC mean as secondary). Readout: the collapse (fraction called SNAP,
     mean p_SNAP per slide; the paper's criterion), per-dataset and within-dataset AUC
     two-sided, on the FOVs whose protein traces the model never trained on (the training
     FOVs' background sits in the OFF frames of their protein traces), and the slide pairs.

   Expected values. 0.5 is not the null for either background control: slides differ in
   their background even when they carry the same protein (lag-1 autocorrelation separates
   same-protein slides of DFK788 at AUC 0.82-0.92; skewness separates the DFK785 slides at
   0.92, in the opposite direction on DFK789), and a ResNet1D protein model trained for two
   epochs (local smoke test; checks/20261009_round3_design_review/smoke_2epoch_resnet1d)
   already classified background at 0.82 (DFK785) and 0.21 (DFK789). Pooled AUCs also mix in dataset identity (the class mix
   differs by dataset; a dataset-only score gets 0.64 on validation). So the readout is
   per dataset and within datasets, two-sided, and compared with the same-protein slide pairs
   of the same model (DFK788: slides 1 vs 4 SNAP, 2 vs 5 HT; DFK789: 2 vs 4 HT; DFK785 has
   none): a model responds to something protein-specific in the background only if it
   separates HT from SNAP slides clearly more than slides of one protein. Residual
   sub-threshold protein events in the background (the spike filter keeps weak ones) remain
   a possible reading of any background learning.

   Decision rule, fixed before the runs. Architectures are compared on the mixed-slide AUC
   (per dataset and pooled, FOV bootstrap); the validation AUC is reported but not used to
   choose (it is the slide-confounded metric). ResNet1D or TCN replaces the CNN-GRU only if
   its mixed-slide AUC is higher and its background controls separate HT from SNAP slides no
   more than the CNN-GRU's do, relative to the same-protein slide pairs. The scrambled runs
   must sit at chance on validation (within the selection bias) for the pipeline to count
   as sound.

   Deviations from the paper: its background CV trained on all ~18,600 background traces
   with trace-level folds (FOVs shared across folds; about 15x the budget here), no mirror,
   factors 0 / 3 / 5; here matched budget, FOV split, the protein recipe. A null result here
   is therefore weaker evidence than the paper's and not directly comparable with its
   rejection of ResNet and TCN. Its scrambling also scrambled the test labels and used one
   permutation; its CV used 4000 frames (here 6000 throughout DualColor).

   Result (2026-10-09; README.md "Round 3"; numbers recomputed independently,
   checks/20261009_round3_results_check): neither ResNet1D nor TCN replaces the CNN-GRU. Mixed
   AUC 0.71 / 0.68 / 0.72 (CNN-GRU / ResNet1D / TCN), paired differences to the CNN-GRU
   -0.031 [-0.082, 0.016] and +0.009 [-0.025, 0.044]: the first condition, read at readout as
   "higher beyond the paired FOV bootstrap" (literally TCN is +0.009), fails for both. ResNet1D
   also fails the background inference (no collapse; DFK789 beyond its same-protein reference);
   TCN collapses as the CNN-GRU does and its only background excess is on DFK785 (no reference;
   the CNN-GRU background run did not train). The scrambled models did not fit their labels
   (near-initialization draws; the pipeline passes no labels, memorization-type leakage is
   excluded by the split checks only). The background slide differences are a movie-wide common
   mode (gone after removing it: slide pairs 0.41-0.59), which cannot raise the mixed-slide AUC.

6. Input normalization: minmax only, one channel (2026-10-09, kept for now). Every DualColor
   model so far reads ProteinTraces/Filtered/<key>_filtered_minmax.pkl (and the background
   controls BackgroundTraces/Filtered/<key>_background_filtered_minmax.pkl), as the paper's
   final CNN-GRU (Figure 3C); the paper saw ResNet and TCN learn more from background with
   dual-channel or z-scored input (Figure S13). The z-scored traces exist locally and on
   Daint if a dual-channel condition is wanted later.

7. Training-set IN rule: the paper's (2026-10-09, user decision; supersedes, for the
   training set only, the 2026-10-02 decision to keep the vesicle table's border and SNR
   threshold). The trace-yield deep dive (Results/Revisions/DualColor/checks/
   20261009_trace_yield) found that extraction and filtering match the paper per FOV, and that
   the vesicle table is the largest single loss: every paper session kept 0.82-0.94 of its
   filtered traces as IN, the table keeps 0.24-0.61, and on the same DFK785 movies the paper
   rule keeps 440 HT and 629 SNAP filtered traces against 225 and 381. On the single-protein
   slides a trace is IN as in the paper (Extraction/extract.py): the vesicle-channel snapshot
   (405 nm on HT / ATTO390 slides, 488 nm on SNAP / ATTO520 slides) localized at the gradient
   the paper's paramfinder picks (ground-truth mode, 640 nm gradient fixed at the dataset's
   value so the protein ROIs stay the same), clustered (2.5 px, >= 3 localizations), IN if a
   cluster lies within 4 px of the ROI. The chance rate is measured per slide with the
   neighbor-FOV null before training. The mixed-slide test set is unchanged (vesicle table,
   single-dye vesicle within 2 px), so results stay comparable with rounds 1-3.

   Result (2026-10-09; paper_in runs: DFK785 run_001, DFK788 run_001, DFK789 run_002; checks/
   20261009_paper_in_gradient). paper_in.py reproduces extract.py's ground-truth calls on DFK785
   (670/670, 1114/1114 ROIs). With the 640 gradient fixed, the paramfinder's ground-truth
   objective is flat (the same score for every ground-truth gradient), so its choice is its
   first trial: 405 nm 1000 / 4000 / 1000 and 488 nm 7000 / 4000 / 8000 (DFK785 / 788 / 789).
   On DFK789's HT slides 1000 gives about 920 clusters per FOV, mostly noise, and the rule stops
   selecting (estimated chance share of IN calls 68 % and 30 %); everywhere else the rule is
   insensitive to the gradient. Rule adopted (before any training): keep the paramfinder's
   gradient unless the chance share (1 - f) * null / in exceeds 15 % (about the vesicle table's
   own chance level on DFK785) on a slide of the run; then the lowest gradient, in 1000 steps,
   at which every slide is at or below 15 %. Only DFK789 405 nm changes, to 3000.
   Filtered traces in the pool (paper IN / filtered, chance share): DFK785 SNAP 629 / 742 (14 %),
   HT 441 / 510 (14 %); DFK788 SNAP 80 / 126 and 80 / 127 (7 %, 2 %), HT 133 / 150 and 318 / 359
   (3 %, 7 %); DFK789 SNAP 68 / 129 (8 %), HT 61 / 234 and 19 / 45 (4 %, 3 %). Training set
   (seed 840410 split): 734 HT / 695 SNAP (1429) against 458 / 454 (912) with the table;
   validation 162 / 162 against 119 / 119.

8. Round 4, does the larger training pool help (2026-10-09; fixed before the runs; configs
   ml/paper_in/r4_*.yaml). The CNN-GRU recipe of r2_all_aug0, trained on the vesicle-table pool
   (own dye within 4 px; r2_all_aug0 is its seed 840410) and on the paper-IN pool (point 7),
   five seeds each (840410, 1, 2, 3, 4; the seed sets the FOV split, the initialization and
   the sampling). The mixed-slide test set is the same 412 traces for every run.
   Readout: per seed, the mixed-slide AUC of each pool and their paired difference (paper-IN
   minus table, same seed); the mean difference over the five seeds with a 95 % t interval
   (4 degrees of freedom) from the seed-to-seed spread, which carries the training and split
   variance that a single FOV bootstrap does not; per-run FOV-bootstrap intervals alongside.
   The validation AUCs are reported, not compared (the two pools validate on different sets).
   Rule: the paper-IN pool becomes the default training set unless the interval of the mean
   difference lies entirely below 0 (it is the paper's definition and the larger set, so it
   is kept unless it is shown to be worse).

   Result (README.md "Round 4"; checks/20261009_round4_check): mixed AUC table 0.692 (SD 0.029),
   paper-IN 0.717 (SD 0.014); paired difference +0.025 [-0.025, +0.075]. The paper-IN pool is the
   default by the rule: not shown to be worse, not shown to be better. The mean gain comes from
   two table runs that stopped early; on the other three seeds the pools agree within 0.012
   (ensembles +0.007). The added traces are protein-like but less separable (AUC 0.73 vs 0.81 on
   never-trained FOVs). With rounds 3 and 4, the mixed-slide AUC sits at 0.69-0.74 whatever the
   architecture or the training pool.

   Balanced accuracy, added after the fact (2026-10-10; not part of the rule, so descriptive):
   mixed BA before / after the MC-dropout filter, table 0.632 / 0.689 at 0.52 kept, paper-IN
   0.654 / 0.753 at 0.49 kept. Paired: BA all +0.022 [+0.007, +0.036] (5 of 5 seeds), BA of the
   50 % most certain traces +0.053 [-0.010, +0.117]. The BA gain without an AUC gain is the
   operating point: the table runs' argmax boundary swings between HT-biased and SNAP-biased
   across seeds, the paper-IN runs' does not. Same confounds as the AUC comparison.

9. Round 5, FOV-grouped repeated k-fold on the paper-IN pool (requested 2026-10-10; fixed
   before the runs; configs ml/kfold/r5_*.yaml, summary kfold_summary.py). The question: how
   well do the models do on single-protein FOVs they never saw, every FOV counted once, how much
   does that differ from the mixed slides, and how much do the mixed-slide results move from
   one training set to the next. CNN-GRU only (round 3: neither other architecture replaces
   it); the folds are fixed by the partition seed and the pool (the configs pin the
   trace_assignment and paper_in runs of round 4), so ResNet1D or TCN can be added later paired
   by (repeat, fold), checking that kfold_fold in data_split.csv is unchanged per trace.
   Recipe: r4_paperin (paper-IN pool, patience 10, balanced validation, mirror + time
   invariance, factor 0, fp32, MC dropout 100 passes, trace loss at most 50 %).
   Folds: in every single-protein slide, the FOVs with pool traces are shuffled (partition
   seed) and dealt in turn into 5 folds from a random starting fold, so each fold holds
   floor(n/5) or ceil(n/5) of every slide's FOVs and a FOV lies in one fold only.
   Run (repeat r, fold k): test fold k ('test_oof', every pool trace of those FOVs, not
   balanced); validation fold (k + 1) mod 5 (early stopping, checkpoint, Wasserstein
   threshold; balanced, the excess 'val_unused', as before); training the other three folds.
   The test fold never touches the model or its threshold (nested), so its scores are not
   inflated by the checkpoint selection; the price is training on 60 % of the FOVs (1053-1131
   traces) instead of 80 % (round 4 paper-IN: 1391-1471). Round 4 did not vary the training-set
   size alone (its pools differ in composition too; paper-IN minus table +0.025 [-0.025,
   +0.075]), so the cost of the smaller training set is not known; round-5 absolute numbers
   describe models trained on 60 % of the FOVs.
   Repeats: 3, partition seeds 840410, 1, 2; run seed 10 x partition seed + k (initialization,
   sampling, validation balancing). 15 runs; mixed-slide test the same 412 traces.
   Readouts (all on the 'all' subset):
   a. Out-of-fold (per repeat, the five test folds together cover every pool trace once): AUC
      within datasets (at the pool's own pair weights, DFK785 about 0.8) and per dataset, balanced accuracy of all traces and of the kept ones
      (each trace at its own fold model's threshold), kept fraction, 95 % FOV-cluster bootstrap
      intervals (evaluate.py metrics); slide pairs (HT slide vs SNAP slide, and same-protein
      pairs as the reference; evaluate.py slide_pairs).
   b. Mixed slides: per run AUC (pooled and within datasets), balanced accuracy all / kept,
      kept fraction; mean and SD over the 15 runs (the same 412 traces: training variance
      alone) and the SD of the 3 repeat means (2 degrees of freedom: reported, not read); ensembles
      (mean p_SNAP over the five fold models of a repeat, and over all 15): AUC and balanced
      accuracy at the argmax (no kept set: each model has its own threshold).
   c. Held-out-FOV vs mixed-slide gap, per model, dataset and mixed slide: d = (AUC on its test
      fold, per dataset) - (AUC on the mixed slides of that dataset, or on one mixed slide), the
      same model on both sides; per dataset and per mixed slide the mean over the 15 models, and
      overall the per-dataset means weighted by the mixed set's HT x SNAP pair counts (DFK785
      0.45, DFK788 0.42, DFK789 0.13), so that the pool's heavier DFK785 share does not set the
      comparison (the overall row gives both sides at these weights). Sensitivity: the overall d
      without DFK785 mixed slide 4 (at chance for every model so far: round-4 AUC 0.44-0.55,
      against about 0.74 on slide 3), weights recomputed. Interval: 95 % FOV-cluster bootstrap
      with the models fixed (pool FOVs and mixed FOVs resampled within their slide, the same
      draw for all 15 models). The SD of d over the 15 models, reported next to it, holds
      training variance and the differences between the models' test folds; the SD of the
      mixed AUC over the runs is training variance alone.
   Reading (declared now; no automatic change to the recipe). d is read per dataset and per
   mixed slide first, then overall. If the interval of the overall d lies above 0, the AUC on
   held-out FOVs of the single-protein slides is higher than the AUC on the mixed slides
   against their vesicle-table truth; validation numbers are then not quoted as expected
   mixed-slide performance, and choices stay read on the mixed slides, as now. The reading
   names the datasets and slides that carry the gap; a gap that rests on DFK785 slide 4 (the
   sensitivity row) is reported as a property of that slide, not of held-out FOVs. A positive
   d does not by itself show that the models use slide-specific cues: it also holds the lower
   label purity of the mixed-slide truth (a protein within 2 px of a vesicle by chance carries
   a random class on the mixed slides and the right one on the single-protein slides; from the
   trace-assignment null about 0.01-0.03 of d overall, 0.02-0.05 on DFK785) and any other
   difference between single-protein and mixed slides. DFK785 has one single-protein slide per
   class and no same-protein reference, so its d cannot separate slide cues from protein
   signal; the same-protein slide pairs of readout a (DFK788, DFK789) can. If the interval
   includes 0, round 5 shows no gap, and its upper end bounds it (the 412 mixed traces keep
   the interval near +-0.06 whatever the number of repeats). The mixed-slide SD over the 15
   runs (training sets sharing 33-67 % of their traces within a repeat, 60 % across) is the
   yardstick for margins in later comparisons on these folds, which pair by (repeat, fold) and
   use the SD of the paired differences. Round-5 mixed numbers are compared with round 4 only
   descriptively (the training fraction differs).
   Pre-submission review (2026-10-10; checks/20261010_round5_review, four reviewers and one
   skeptic per finding): no blocker. Folds, sets, pool and mixed test verified from the 15 dry
   runs (checks/20261010_round5_dryrun); the round-4 splits reproduce exactly with the new
   code; the partitions are the same under Daint's numpy 1.26.4. kfold_summary.py tested end
   to end on synthetic runs with planted gaps (point estimates exact, 95 % interval coverage
   0.94-0.95 over 1000 synthetic worlds; duplicated or missing test-fold traces raise). The
   eight minor findings are fixed above and in kfold_summary.py (wording of c and the reading,
   per-slide and sensitivity rows, both sides of the overall row, pinned inputs, OOF coverage
   check). Estimated runtime: about 13 min per 4-run debug job.

## Open

- Architectures on the round-5 folds (ResNet1D, TCN paired with the CNN-GRU by repeat and fold,
  with a margin declared in advance), if wanted after round 5.
- Controls that would answer what round 3 could not: background training to a fixed number of
  epochs (the CNN-GRU background run stalled at ln 2), with random-initialization and skewness
  baselines; scrambled runs that fit their labels; a vesicle-population control (background
  boxes next to single-dye vesicles on the mixed slides, compared within FOV).
- Input option: remove the movie common mode (the leave-one-out mean of the movie's background
  traces) from the traces before training; expected to shrink the validation-mixed gap without
  lowering the mixed AUC.
