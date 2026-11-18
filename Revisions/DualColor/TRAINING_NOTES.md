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

2. Early stopping exactly as in the paper (2026-10-09; amended by point 10, warm-up of 25 epochs): patience 10 epochs (SI, "Combined
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

   Result (2026-10-10, Daint jobs 5021067, 5021068, 5021105, 5021136; README.md "Round 5"): 6
   of the 15 runs stopped on the initial plateau and did not train (point 10); the declared
   readouts are dominated by them and are superseded by round 5b.

10. Early stopping with a warm-up; round 5 rerun as round 5b (user decision 2026-10-10; fixed
   before the runs; configs ml/kfold_warmup/r5b_*.yaml).
   The failure: every CNN-GRU run starts on a plateau where the training loss stays near
   ln 2 = 0.693 (0.665-0.70) and the validation AUC swings from epoch to epoch (0.35-0.77 in
   rounds 4 and 5, about 2.3 times the sampling SD of a no-signal AUC on these sets, 0.03): the
   spikes are transient weak cues, not noise (the six failed checkpoints rank their own test
   folds at 0.62-0.73 and the mixed slides at 0.46-0.58). The plateau ended (training loss
   below 0.65) at epochs 8-21 in the 19 runs of rounds 4 and 5 that left it (10 of 10 in round
   4, 9 of 15 in round 5). When a spike in the first epochs becomes the best checkpoint and the
   plateau outlasts it by the patience (10 epochs), the run stops untrained: 6 of 15 round-5
   runs (best epochs 2-8, censored at epochs 12-18, mixed AUC 0.46-0.58, against 0.67-0.72 for
   the other nine). In round 4 the table seed 1 checkpoint (epoch 12) lies just before its
   plateau end (epoch 14), but that model had started to learn (validation loss 0.63, mixed
   AUC 0.66).
   The fix: optimization.early_stopping.warmup_epochs W (ML/utils.train_model, default 0 = the
   paper's behavior, unchanged): the first W epochs only train; the paper's rule, with all its
   state (best AUC, best loss, patience), starts at epoch W + 1, which is therefore always a
   checkpoint (if its AUC is finite), and the earliest stop is epoch W + 1 + patience. W = 25,
   above the latest plateau end among the runs that left the plateau (21). A run still fails
   if its plateau outlasts about W + 10 epochs (spliced round-4/5 logs: 10 % of runs with a
   34-epoch plateau, 56 % at 41), which a censored lognormal fit to the round-5 plateau ends
   puts at about 0.3 % of runs (longer patience instead, 13-20 epochs, fails 6-25 %). Round 5b
   therefore uses the paper's early stopping from epoch 26, a deviation from point 2.
   Everything else as round 5 (same 15 configs, folds, seeds and inputs), so round 5b pairs
   with round 5 by (repeat, fold). The training itself is not reproduced: same-seed runs on the
   same training set (rounds 1 and 2) agree at epoch 1 and diverge from epoch 2 (GPU
   nondeterminism); one pair's plateau ended at epoch 12 vs 17, and their mixed AUC differed by
   -0.044 to +0.011. A paired difference therefore holds training variance as well as the
   effect of the warm-up.
   Declared now: a run whose training loss never falls below 0.65 is reported as not trained
   (kfold_summary.py left_plateau). The readouts of point 9 are computed on all 15 runs; if
   any run is not trained, readouts b and c are repeated on the trained runs
   (mixed_summary_trained, ensembles_trained for the ensemble of all trained runs,
   gap_trained). Readout a needs all five fold models of a repeat and stays on all runs,
   naming the untrained folds. Expected cost: the runs that peaked before epoch 26 (three
   trained round-5 runs: p840410_f1 22, p1_f1 22, p1_f2 17) now take their checkpoint at epoch
   26 or later; in the 11 round-4/5 runs whose W = 25 replay ends inside their logs, the
   checkpoint is the one the paper's rule chose. r5b_paired.csv gives per (repeat, fold) the
   round-5 and round-5b numbers and is read separately for the nine pairs trained in round 5
   (mean and SD of the paired differences: the cost of the warm-up plus training variance) and
   for the six that were not (the rescue).

   Result (2026-10-10; README.md "Round 5b"; checks/20261010_round5b_check): all 15 runs trained
   (plateau ends 8-21, checkpoints at epochs 26-81). Mixed AUC 0.706 (SD 0.014), balanced
   accuracy 0.643 all / 0.725 kept; out-of-fold AUC within datasets 0.80-0.82. Gap d per dataset:
   DFK785 +0.218 [+0.129, +0.316] (mixed slide 3 +0.097 [+0.009, +0.204], within what label noise
   accounts for at that slide; slide 4 +0.370 [+0.234, +0.499]), DFK788 -0.004 [-0.084, +0.083],
   DFK789 +0.022 [-0.100, +0.136]. Overall +0.100 [+0.046, +0.158], carried by DFK785 mixed slide
   4; without it +0.021 [-0.034, +0.078], almost all of it from DFK785 slide 3. Warm-up cost on the
   nine runs trained in both rounds: mixed AUC +0.001 (SD 0.027; 95 % t interval [-0.020,
   +0.021]). Decision (user, 2026-10-10): warmup_epochs 25 stays in the CNN-GRU recipe for later
   rounds, amending point 2 (otherwise the paper's early stopping). W = 25 was calibrated on
   CNN-GRU plateau ends, so another architecture needs its own check first.

11. Round 6: the paper-IN pool of all six sessions, three training designs (user request
   2026-11-17; fixed before the runs; configs ml/round6/, made by ml/round6/make_configs.py;
   summary round6_summary.py). Revisions are under time pressure: no new slides; the existing
   six sessions are used as fully as possible.

   Pool. The paper's IN rule (point 7) on the single-protein slides of all six sessions. Old
   sessions: round 4's runs, pinned (trace_assignment DFK785 run_002, DFK788 run_001, DFK789
   run_001; paper_in DFK785 run_001, DFK788 run_001, DFK789 run_002). New sessions:
   trace_assignment run_001; ground-truth-mode runs Extraction_gt405 / Extraction_gt488 made on the
   Mac (extraction/run_local.sh gt mode: paramfinder and localize, 640 gradient fixed at the
   640-only runs' 10000, as the old sessions' Daint runs); paper_in run_001 with point 7's
   gradient rule (chance share of IN calls at most 15 %), the rule evaluated first
   (checks/20261117_round6_paper_in/gradient_rule.py) so that the compliant run is run_001.
   Amendment to point 7 for round 6 (pre-submission review): the 15 % is checked on the filtered
   ROIs of the FOVs that enter the pool, so that excluded slides and FOVs (below) do not set the
   gradient of the ones used. Two exclusions, based on the first look (checks/
   20261116_new_sessions_first_look) and declared before any round-6 training: DFK846 Slide1HT
   (background below the session's level, ramping over its first 9 FOVs) and DFK846 Slide3SNAP
   FOVs 13-40 (the ATTO520 label fades to 1-3 % of its start and the 640 yield falls by 70 %: the
   sample changes during the acquisition, and the paper-IN truth depends on that label).
   Everything else is kept, the dim-HT slides and DFK843 Slide4SNAP included. The first look also
   suggested dropping DFK846 Slide2HT (39 traces, falling yield); kept (user, 2026-11-17): a falling
   yield alone has not been an exclusion reason, and the sessions are to be used as fully as possible.
   Correction (2026-11-17, after the runs): the reason also given then, that its ATTO390 label does not
   fade, was wrong (late/early vesicle flux 0.51); its paper-IN chance share (0.037) is what separates it
   from the excluded Slide3SNAP block (about 0.22). The experimenter's readme notes insufficient
   immobilization for DFK846 slides 1-2 (checks/20261117_round6_results/acquisition_issues.md). Mixed-slide test sets unchanged (vesicle table,
   single-dye vesicle within 2 px): DFK785 slides 3 and 4, DFK788 slide 3, DFK789 slides 3 and 5,
   DFK842 slide 3, DFK843 slide 3, DFK846 slide 4.

   Recipe: round 5b's (CNN-GRU, patience 10 after a 25-epoch warm-up, balanced validation, mirror
   and time invariance, factor 0, fp32, MC dropout 100 passes, trace loss at most 50 %); only the
   split differs between designs. The mirror's per-trace GMM runs in parallel
   (augmentation.mirror_n_jobs; output identical to apply_augmentation, tests/
   test_mirror_parallel.py), since the six-session pool would otherwise spend most of a 30-min
   debug job on it.

   Designs (every run also predicts all eight mixed slides; the test named here is the design's
   primary one, the others are secondary):
   A. Leave one experiment out (6 sessions x 3 seeds = 18 runs; val_datasets). Training: the pool
      of the other five sessions, every FOV. Validation: the whole pool of the left-out session's
      single-protein slides (balanced): early stopping, checkpoint, Wasserstein threshold and the
      equal-recall threshold below. Test: the left-out session's mixed slide(s). The model never
      trains on the left-out session, but its stopping point and thresholds come from that
      session's single-protein slides, as they would for any new session recorded with its own
      single-protein controls. Seeds 840410, 1, 2.
   B. Within experiment (6 x 5 = 30 runs). Training and FOV-grouped validation (val_fov_fraction
      0.2 per slide, as rounds 1-4) from one session's pool; test: its own mixed slide(s). Seeds
      840410, 1, 2, 3, 4. Pools are small (DFK789 about 150 traces), so these runs are expected to
      be noisy; secondary readout: the session-to-session transfer matrix.
   C. Pooled (5 runs). All six pools, FOV-grouped validation 0.2 per slide, all eight mixed
      slides. Seeds 840410, 1, 2, 3, 4. FOV groups rather than a random trace split, as in rounds
      1-5: traces of one FOV share focus, laser and sample, and a trace-level split would leak them
      into validation.

   Readouts (round6_summary.py), on the single-dye traces of the mixed slides, per mixed slide
   and per session:
   - Per run: AUC of p_SNAP; within-FOV AUC (HT-SNAP pairs of the same FOV only); balanced
     accuracy and recalls at the argmax (p_SNAP 0.5); balanced accuracy at the run's
     equal-recall threshold (the p_SNAP cut at which HT and SNAP recall are equal on the run's own
     validation set; new in round 6, because the transfer check found the operating point moving
     between sessions); kept fraction and balanced accuracy of the kept traces at the Wasserstein
     threshold; trained (training loss below 0.65 at some epoch, point 10).
   - Per design cell (A: left-out session; B: session; C) and unit: mean and SD over all seeds and
     over the trained seeds (as point 10; a run without a log is reported and counted as trained,
     as in kfold_summary.py); ensemble (mean p_SNAP over the trained seeds): AUC and within-FOV AUC
     with 95 % FOV-cluster bootstrap intervals (FOVs resampled within each slide; a FOV drawn twice
     counts as two FOVs), balanced accuracy at 0.5 and at the median of the members' equal-recall
     thresholds.
   - Main table: session x {A, B, C, round-5b ensemble (old sessions in-session, new sessions
     transferred)}; differences between designs per session as a paired FOV bootstrap of the
     ensembles on the same mixed traces (test sampling only), with the trained-seed SD of both
     cells on that unit next to it (the training variance) and the reading rule below applied
     (paired.csv read_as_real).
   - DFK785 slide 4 (at chance in every model so far) per slide, and pooled numbers with and
     without it. A mixed slide with fewer than 50 traces of a class (DFK846: 32 SNAP) is reported
     but not read on its own.

   Reading, declared now. No recipe is chosen from these numbers. C is the paper-consistent
   analysis (the user's description) and is the main result; A is the estimate for a session the
   model never trained on that comes with single-protein controls; B says whether a session's own
   controls are enough. All three are reported whatever they show. A difference between designs on
   a session is read as real only if its paired interval excludes 0 and it exceeds twice the larger
   seed SD of the two cells. Limits: A validates on the left-out session (no independent
   validation set; its numbers are not free of selection on that session's single-protein
   slides); B's pools are small; C's validation FOVs come from all sessions in proportion to their
   FOVs, so its checkpoint can favor the larger sessions.

   Result (2026-11-17; Daint jobs 5377591-5378278; README.md "Round 6"; checks/20261117_round6_results,
   every number reproduced by an independent recomputation). C on all eight mixed slides: AUC 0.749
   [0.722, 0.777], within-FOV 0.771, balanced accuracy 0.68; per run 0.733 (SD 0.017); DFK785 slide 4
   at chance in every design. A equals C on every session (C - A -0.027 to -0.003, no interval
   excludes 0; all six negative, mean -0.012 [-0.027, +0.004]): training on a session's own
   single-protein slides does not help its mixed slide. B fails (B - A -0.11 to -0.18, all
   intervals exclude 0; 10 of 30 runs never left the plateau, DFK846 0 of 5), largely because the
   recipe's warm-up and patience count epochs and B has 4-28 optimizer steps per epoch (A and C
   107-136); not tested with a step-based warm-up. C equals the round-5b models per session (-0.037
   to +0.020) and beats them on the eight slides pooled (+0.029 [+0.012, +0.048]) by calibration.

12. Round 6 design A with clipmax input (user request 2026-11-17; fixed before the runs; configs
   ml/round6_clipmax/r6c_loeo_<DS>_s<seed>.yaml, made by ml/round6/make_configs.py; readout
   round6_variant_compare.py). A quick check: design A of point 11 again, with the trace
   normalization `clipmax` instead of `minmax`, everything else identical (sessions, pool, exclusions,
   seeds 840410 / 1 / 2, splits, recipe, mixed test traces; the configs differ only in run_name and
   input_channel; the mirror's GMM runs on the clipmax trace, the same code path).
   clipmax (clipmax.py; ported from blinkognition2 clarissa-local 64f1a48, combine.py, identical output,
   tests/test_clipmax_port.py): per background-removed trace, the GMM signal mask of
   Extraction/utils.gmm_classify_frames at the pipeline's own peak picking (gmm_proba_threshold 0.9,
   min_separation 2.0, the trace filter's settings; blinkognition2 used 0.6 / 2), non-signal and negative
   frames set to 0, divided by the trace maximum: blinks in (0, 1], background exactly 0. In the
   blinkognition2 investigation (LoD/HTvsCL_confound_investigation.md) clipmax beat minmax across
   sessions (HTvSNAP leave-one-session-out 0.768 vs 0.702) and lost within sessions.
   Readout, per session and pooled (all six; without DFK785 slide 4): ensemble AUC, within-FOV AUC,
   balanced accuracy at 0.5 and at the session's own equal-recall cut, clipmax minus minmax on the same
   resampled FOVs (95 % FOV-cluster bootstrap); per run AUC, balanced accuracy, MC-dropout kept fraction
   and balanced accuracy of the kept. Reading, as point 11: a session's difference counts only if its
   interval excludes 0 and it exceeds twice the larger seed SD; the pooled differences are read on
   their intervals. No recipe change follows automatically; the user decides.

   Result (2026-11-17; Daint jobs 5379420-5379829; checks/20261117_round6_results, recomputed
   independently). Per-session AUC clipmax - minmax -0.009 to +0.047, none passing the rule (DFK785
   +0.047 [+0.009, +0.085] < 2 x 0.029). Pooled, both declared differences exclude 0: AUC 0.752 ->
   0.773 (+0.021 [+0.008, +0.034]) and balanced accuracy at 0.5 0.687 -> 0.719 (+0.032 [+0.014,
   +0.051]); post hoc, the AUC gain sits in pairs across sessions (same-slide pairs +0.002 [-0.011,
   +0.015]; +0.005 to +0.010 with session location removed or sessions weighted equally) and the
   balanced-accuracy gain is an operating-point effect that is gone at the sessions' own cuts (0.698 vs
   0.699). Per run a small borderline gain (+0.017, 13 of 18). Not a reason to replace minmax; reported
   as a sensitivity analysis.

13. DFK851, a seventh session, in design B (user request 2026-11-18; fixed before any DFK851 trace
   was extracted; configs ml/dfk851/r6_within_DFK851_s<seed>.yaml, made by ml/dfk851/make_configs.py;
   readout dfk851_summary.py). "Full extraction, training and testing as before for a single session (B)."

   Data (datasets/DFK851.yaml; raw data dual_color/20261116_DFK851, dual_color/SOURCES.md): slide 1
   SNAP (100 FOVs), slide 2 HT (130), slide 3 mix (120; 15 uL HT + 5 uL SNAP fractions), slide 4 SNAP
   (150), 16-18 Nov, the DFK842-846 acquisition settings. New: vesicles loaded with 8 uM protein
   (DFK842-846 2 uM), so more proteins per vesicle; traces per vesicle are reported.

   Extraction as DFK842/843/846, on this Mac (extraction/run_local.sh; Picasso 0.8.8 with the MLE patch,
   the version of every other session; S3IT's picasso-env has 0.9.3): 640-only run, vesicles and
   unmixing run_001, trace_assignment run_001, ground-truth-mode gt405 / gt488 runs, paper_in with point
   7's gradient rule on the pool FOVs (round-6 amendment), so that the compliant run is pinned. A movie
   that Picasso fails on is rerun unpatched with the same command (as round 6).
   Changed before any extraction output was used (user, 2026-11-18): the 640-only run, the slow step, runs
   on S3IT (s3it/extract.sbatch: run_local.sh unchanged in conda env picasso088, Picasso 0.8.8 with the Mac
   package versions; S3IT's picasso-env has 0.9.3), reading the raw data in place; the download continues
   for the local steps. Equivalence check on two DFK846 movies (gradient 10000): every converged
   localization bit-identical to the Mac's; 1-4 per movie differ, all collapsed fits (sigma 0.01-0.06 px,
   architecture-dependent end points of non-converged MLE iterations). DFK851's paramfinder chose the 640
   gradient 20000 (32.6 vs 26.5 at 10000; every other session got the search's floor, 10000); not
   overridden. The gt configs fix the 640 gradient at the 640-only run's, so at 20000 for DFK851. The gt runs
   also run on S3IT, staged from a vesicles run made there for the FOV pairing only (VESICLES_RUN); the
   canonical vesicles and unmixing runs are made on the Mac (blink2env, as DFK842-846) and their fovs.csv
   is checked against the S3IT one.

   Pool exclusions, rules fixed now and applied mechanically after the extraction; anything else seen in
   the first look is reported and kept:
   - Fading label: on a single-protein slide, a run of at least 5 consecutive FOVs (acquisition order)
     whose median home-channel vesicle flux (vesicles.csv, all detected vesicles of the FOV) is below
     10 % of the median of the slide's first five FOVs excludes every FOV from the first of that run to
     the end of the slide (the paper-IN truth depends on the label). Applied to the six sessions this
     rule gives DFK846 Slide3SNAP FOVs 12-40 (round 6 excluded 13-40) and nothing else.
   - Focus: a single-protein slide whose median single-frame spot sigma (sf_sigma_med: Gaussian fits on
     the raw frames of each own-dye trace's 10 brightest frames, the first look's
     checks/20261116_new_sessions_first_look/verify/c4c5_ht_brightness/s7_psf_frames.py; in focus
     1.25-1.27 px, DFK846 Slide1HT 1.43, DFK842 Slide2HT 1.65) exceeds 1.40 px is excluded
     (acquisition_issues.md's proposed QC threshold).
   - Chance share: point 7's rule (gradient raised until the slide's chance share is at most 0.15).
   - A FOV missing its 640 movie or a snapshot stops vesicles.py (fov_qc.fov_groups), so the copy is
     verified against S3IT (sizes, a dry-run rsync, sampled MD5) before anything runs.

   Training: design B of point 11 unchanged (round 5b recipe; val_fov_fraction 0.2 per slide; seeds
   840410, 1, 2, 3, 4; every run predicts the mixed slides of all seven sessions). Known limit (point
   11): the warm-up and patience count epochs, and B failed in round 6 where the pool gave 4-28
   optimizer steps per epoch; DFK851's steps per epoch are reported. No recipe change.
   Changed before any training (user, 2026-11-18; Daint in scheduled maintenance that day): the five runs
   and the C-model reference run on UZH S3IT instead of Daint, as the tasks of one GPU job
   (s3it/train.sbatch; conda env blink2-cuda, torch 2.5.1 as the Daint container; inputs sent with
   daint/sync_inputs.sh, SYNC_DEST=cluster:<clone>/, the five C model folders copied from local). The only
   difference is the GPU type (S3IT's, recorded in each run's config_full.json; round 6 ran on GH200),
   with fp32 and TF32 off as before. The sets are checked to be identical to the local dry run's
   (checks/20261118_dfk851/dryrun_check.py) before training.
   Outcome (2026-11-18; README "DFK851 in design B", checks/20261118_dfk851/README.md). Deviations found by
   the independent verification and corrected before reporting: the C-reference prediction first ran with
   cuDNN TF32 on (apply_models.py had not set the flags; rerun with TF32 off, job 8519664; the numbers moved by
   at most 0.003); paired.csv printed the AUC's seed SD on every metric row (now each metric's own; the rule
   stays on the AUC, as declared); transfer_B.csv lacked point 11's per-slide rows and 50-per-class floor
   (added); the readout folder dfk851 is the DFK851 data folder on a case-insensitive disk (now dfk851_b).
   Added on request after the results: balanced accuracy before and after the paper's MC-dropout filter
   (dfk851_ba_filter.py; post hoc). Ensemble filter, the user's rule (2026-11-18): the mean of the members'
   Wasserstein certainties against the mean of their thresholds (each the paper's 50 % rule on the member's
   validation set), so the ensemble keeps the paper's trace-retention rule; for future ensembles too.
   Balanced accuracy is a result (user, 2026-11-18, decided after the DFK851 results were seen): B - C
   differences in balanced accuracy (at 0.5, at the cut, and after the paper's filter) are read with point 11's
   rule, each with that metric's own trained-seed SD; the within-FOV AUC stays reported. On DFK851 slide 3 this
   reads BA at 0.5 as real (C higher, -0.028 [-0.047, -0.009] against 2 x 0.0135; margin 0.0007, on an SD from
   five seeds), BA at the cut not (-0.031 against 2 x 0.0167), the filtered BA not (either threshold); early
   half real at both cuts, late half not.

   Readouts on the single-dye traces of DFK851 slide 3 (dfk851_summary.py), as point 11:
   - Primary: the B ensemble (mean p_SNAP of the trained seeds): AUC and within-FOV AUC with 95 %
     FOV-cluster bootstrap intervals, balanced accuracy at 0.5 and at the median of the members'
     equal-recall thresholds; per run AUC, balanced accuracy, kept fraction and balanced accuracy of the
     kept, trained flag, epochs and steps per epoch.
   - Reference (no training): the five round-6 C models (trained on the six other sessions, never on
     DFK851) applied unchanged to the same traces with train_pure's prediction (MC dropout, 100 passes;
     their own Wasserstein and equal-recall thresholds); B minus this reference on the same resampled
     FOVs, read as real only if the interval excludes 0 and the difference exceeds twice the larger seed
     SD (point 11). The prediction script is first checked on DFK846's mixed slide against the C runs'
     own predictions.
   - Secondary: early and late halves of slide 3 (split at its median FOV in acquisition order; the
     late-FOV drop of acquisition_issues.md item 5, declared here rather than post hoc); the DFK851 B
     ensemble on the other sessions' mixed slides (transfer row).
   - Slide 3 counts per class are reported; fewer than 50 traces of a class make it a reported, not a
     read, result (as point 11).

14. Per-FOV HT:SNAP ratio on DFK851's mixed slide, with the per-trace uncertainty (user request and
   decisions 2026-11-18; fixed before any per-FOV estimate was computed; script dfk851_fov_ratio.py, outputs
   Results/Revisions/DualColor/dfk851_b/fov_ratio/). Question: for each FOV of the mixed slide, does the
   model recover the share of HT and SNAP traces?

   Seen before deciding (design facts only, no model output on the mixed slide): 118 FOVs with single-dye
   traces, median 16 (IQR 13-19); the true SNAP share per FOV (0.23-0.64, SD 0.130) varies no more than
   binomial sampling from one mix would (pooled 0.455; binomial SD at n = 16 0.124; homogeneity chi2 113 on
   117 df, p = 0.58), so a FOV is a small sample of one composition; the vesicle-label share correlates 0.30
   with the trace share across FOVs (not a usable truth); on B's validation sets the paper's filter keeps
   53-59 % of HT and 43-50 % of SNAP traces, and B over-calls SNAP on the mixed slide (point 13).

   Decisions (user): truth = the SNAP share among the FOV's single-dye traces (class IN_ATTO390 -> HT,
   IN_ATTO520 -> SNAP); FOVs with at least 10 single-dye vesicles carrying a filtered trace (112 of 118; the
   same set as at least 10 traces); primary estimator = probabilistic counting; designed mixtures added.

   Models: the B ensemble (five r6_within_DFK851 runs, primary) and the C ensemble (five round-6 r6_pooled
   runs, reference). The saved predictions keep only each trace's MC mean and SD, so the 100 MC-dropout passes
   are regenerated for slide 3 (apply_models.py --save-passes; seeded with the run's seed, TF32 off, as the
   recorded predictions; checked against them). A pass is one dropout-sampled model applied to every trace,
   so drawing passes keeps the model uncertainty correlated across the traces of a FOV; drawing each trace
   independently would make the per-trace spread irrelevant (a Bernoulli draw with a random p is a Bernoulli
   draw with its mean).

   Estimators per FOV (labeled traces; ensemble = the user's averaging over models, point 13):
   - Primary, adjusted probabilistic count (PACC): (mean over the FOV's traces of the ensemble MC-mean p_SNAP
     - a) / (b - a), clipped to [0, 1]; a, b = means over the five runs of each run's mean p_SNAP over its own
     validation traces of true class HT (a) and SNAP (b) (B: DFK851 held-out FOVs; C: its six sessions).
     Never fitted on mixed-slide labels.
   - Its 95 % interval, simulated (4000 draws per FOV): draw a run and an MC pass (shared by all traces of the
     FOV), draw each trace's class from that pass's p_SNAP, share -> (share - a_run) / (b_run - a_run),
     clipped; 2.5 / 97.5 percentiles. By construction at least as wide as PACC's own sampling error (it adds
     the vote randomness), so expected conservative if the validation constants transfer.
   - Comparison, the paper's filter: ensemble kept = mean certainty > mean threshold (point 13), call = mean
     p_SNAP >= 0.5; the kept calls corrected for class-dependent retention and confusion (validation rates of
     each run's own kept / calls, averaged over runs; 2 x 2 inversion), clipped. Point estimate only.
   - Naive classify and count (share called SNAP at 0.5), uncorrected.
   - Baselines: the slide-level PACC (all included FOVs pooled) predicted for every FOV (no per-FOV
     information); the true slide share 0.455 (an oracle constant).
   Readouts: per estimator, bias, mean absolute error (MAE), RMSE, Pearson r and OLS slope of estimate on
   truth across FOVs, for PACC interval coverage and mean width; MAE of each estimator minus each baseline with
   95 % FOV-cluster bootstrap intervals (2000 draws); B and C side by side. Reported, no truth: PACC on all
   filtered traces of each FOV (dual, no-label and outside traces included).

   Designed mixtures: traces of the included FOVs pooled by class; SNAP shares 0.1 ... 0.9 (step 0.1) at
   sizes 16, 50 and 200 traces, 500 mixtures each, drawn without replacement; the same estimators (PACC
   interval with 400 draws). Readouts per size: mean estimate per designed share (calibration curve), OLS
   slope and intercept of estimate on designed share (1 and 0 = follows the ratio faithfully), RMSE, PACC
   coverage. Limits: the pool mixes FOVs (a mixture has no single-FOV background); the validation constants
   come from slides where the model separates the classes better than on the mixed slide (BA 0.79 vs 0.72),
   so PACC is expected to under-correct (slope below 1). Not read with a significance rule; reported.
   Outcome and corrections (2026-11-18; independent verification, checks/20261118_dfk851/verify_fov; README there).
   Every number reproduced from the raw passes (two recomputations). Corrections to the text above: (1) the
   premise that a pass is one dropout-sampled model applied to every trace is false: mc_dropout_predict draws a
   dropout mask per trace (ML/utils.py, ML/models.py LockedDropout), passes are uncorrelated across traces, so the
   per-trace MC spread adds nothing at the FOV level; the interval is the simulated class votes (85 % / 94 % of the
   variance, B / C) plus the choice of run, not a sampling interval of PACC; (2) it is not wider "by construction"
   (only while a class's variance of p is below E[p(1 - p)]) and not conservative in practice: PACC's analytic
   sampling interval covers 75 % (B) / 88 % (C) of FOVs, because the model's p varies between FOVs within a class
   (B SNAP ICC 0.29, variance ratio 2.76); B's 97 % coverage is coincidence, C's 100 % comes from 72 of 112
   intervals being [0, 1]; (3) the regenerated passes are an independent MC draw, statistically consistent with
   the recorded predictions (mean |diff| 0.0045-0.0061, as expected for independent masks), not a reproduction;
   (4) the expected under-correction holds for B (slope 0.76, predicted 0.762 from the slide's class means of p)
   but not for C (slope 1.05-1.08; C over-reads SNAP because SNAP traces score higher on the mixed slide than on
   its validation sets); (5) the oracle used is 0.459 (the 112 included FOVs), "0.23-0.64" is the 5-95 % range
   (full range 0.09-0.73 over the included FOVs), and the mixture slope is fitted on the realized share
   (round(s x n) / n; differs from the designed share at 16 traces only).

15. Sensitivity, limits of detection and quantitation of the HT:SNAP ratio on DFK851's mixed slide (user
   request and decisions 2026-11-18; fixed before any limit was computed; script dfk851_lod.py, outputs
   Results/Revisions/DualColor/dfk851_b/lod/). Question: point 14 found the slide-wide ratio good and a single
   FOV (~16 traces) poor; how many molecules (traces), read from how many FOVs, does a ratio measurement need,
   and how small a minority does it detect and quantify? Framework: CLSI EP17 (limits of blank, detection and
   quantitation), with the composition as the analyte.

   Seen before deciding (design facts only, no limit computed): the 112 included FOVs of point 14 hold 10-26
   single-dye traces (mean 16.3; HT 3-17, SNAP 1-16; 989 HT, 839 SNAP in all). Class means of the ensemble
   p_SNAP on the mixed slide vs the pure-slide validation constants of point 14: B HT 0.358 vs 0.305, SNAP
   0.691 vs 0.741 (separation 0.332 vs 0.436); C HT 0.343 vs 0.334, SNAP 0.650 vs 0.613 (0.306 vs 0.279).
   With the pure-slide constants a true share s reads as 0.123 + 0.762 s (B) and 0.033 + 1.098 s (C).

   Decisions (user): EP17's error rates, 5 % false positives and 5 % false negatives, one-sided (1.645 SD);
   design grid n = 16, 50, 100, 200, 500 molecules x k = 1, 3, 6, 12, 24 FOVs (realistic for an experiment);
   B primary, C reference; quantitation goal: total error at most 0.10 in absolute share; two optional checks
   later (B's false positives against a blank from a pure-HT slide; constants from early FOVs applied to late
   ones). A correction to the proposal discussed with the user: the scale compression does not raise the
   limit of detection (below); calibration matters for quantitation and for where the blank comes from.

   Response and estimators. Per molecule p = the ensemble MC-mean p_SNAP of point 14 (mean of the five runs);
   a sample's response T = mean p over its n molecules. Uncalibrated share = (T - a) / (b - a), a and b the
   pure-slide validation constants of point 14. Calibrated share = (T - a') / (b' - a'), a' and b' the mean p
   of the HT and SNAP traces of the slide's other included FOVs (best case: calibrator and sample on one
   slide). Reported shares are clipped to [0, 1]. Detection decisions use the unclipped T: any fixed pair of
   constants is a monotone map of T, so detection limits in true-share units do not depend on the
   constants, and clipping would discard information at the edges.

   Variance model, two components per class. p = mu_c + u_jc + e: FOV effects (u_HT, u_SNAP) with variances
   s2F_HT, s2F_SNAP and covariance cF across FOVs, within-FOV variances s2w_HT, s2w_SNAP. Estimated by the
   method of moments (one-way random-effects ANOVA, unbalanced, Searle's n0); the covariance from the
   unweighted covariance of the FOVs' two class means (unbiased: the within-FOV errors of different molecules
   are independent); mu_c = the class mean over molecules (a', b'). A sample: k FOVs, n/k molecules each,
   each molecule SNAP with probability s, the material share (the slide's composition is binomial across
   FOVs, point 14). Then E[T] = a' + s (b' - a') and
     Var[T] = [(1 - s) s2w_HT + s s2w_SNAP + s (1 - s) (b' - a')^2] / n
            + [(1 - s)^2 s2F_HT + s^2 s2F_SNAP + 2 s (1 - s) cF] / k
            + s (1 - s) (s2F_HT + s2F_SNAP - 2 cF) / n.
   The first line is the molecule term (shrinks with n; includes the binomial composition), the second the
   FOV term (shrinks only with k), the third a small cross term. Normal theory for T (EP17's parametric
   route).

   Detection, both directions. SNAP in HT: limit of blank LoB = E[T | 0] + 1.645 SD[T | 0] (95th percentile of
   all-HT samples; reported in share units with both sets of constants); limit of detection LoD = smallest s
   with E[T | s] - 1.645 SD[T | s] >= LoB (5 % false negatives at the LoD). HT in SNAP: mirrored, blank s = 1,
   reported as an HT share. Minimum detectable difference between two samples of the same design: smallest
   d with power 0.80 in a two-sided test at alpha 0.05, d (b' - a') = 2.80 sqrt(Var[T | s0] + Var[T | s0 + d]),
   at s0 = 0.10 and 0.50.
   Quantitation: per true share s (step 0.005), bias and SD of the reported (clipped) share, from the clipped
   normal; the calibrated SD adds the calibrator's sampling error (delta method, calibrator = the counts of all
   112 FOVs). Total error TE = |bias| + 2 SD; quantifiable range = the shares with TE <= 0.10 that lie between
   the SNAP LoD and 1 - the HT LoD; its lower end is the limit of quantitation (LoQ). Uncalibrated and
   calibrated.
   Grid cells with n < k are skipped. Cells whose molecules per FOV (n / k) lie outside the slide's 10-26 are
   reported but flagged as density extrapolation (the model assumes FOV effects do not depend on density).

   Uncertainty: FOV-cluster bootstrap, 2000 draws (the 112 FOVs resampled with replacement, duplicates kept
   as separate FOVs; means and components re-estimated; negative variance estimates set to 0, correlation
   clipped to [-1, 1]); 95 % percentile intervals for every limit. The pure-slide constants are held fixed.
   B vs C: paired bootstrap differences on the realistic diagonal (n / k about 16), reported, not read with a
   decision rule.

   Model check, before any limit is read, with the pass criterion fixed here. Real whole-FOV samples, no
   model: k FOVs drawn with replacement (k = 1, 3, 6, 12, 24; 5000 samples per k; with replacement so the
   drawn FOVs are independent, as the model assumes, and no finite-population correction is needed: the unit
   drawn is a whole FOV with its traces, so a correction would apply to the whole variance and, for the
   thinned types, only in part; a FOV drawn twice is thinned independently), five types: all HT
   traces (blank), all SNAP traces (the other blank), all labeled traces (the natural mix, about 0.46), all HT
   + each SNAP trace kept with probability 0.25 (low SNAP share), all SNAP + each HT trace kept with 0.25 (low
   HT share). Per sample z = (T - model mean given its per-FOV class counts) / (model SD given those
   counts). Pass: for each type and k, the
   fractions of z above 1.645 and below -1.645 each within 0.025-0.075 (the model's one-sided 95 % bounds
   hold to within half their nominal rate). A k that fails flags its grid cells "model not validated"; no
   other estimator is substituted. Reported, not criteria: the same check with the model fitted on a random
   half of the FOVs and samples drawn from the other half (k <= 12, 50 splits); a brute-force simulation from
   the fitted model at a few cells against the formula (a check of the algebra).
   Constants check (promised to the user): on the k = 6 whole-FOV samples, the empirical LoB and detection
   rates with the pure-slide constants, the mixed-slide constants, and a calibration per sample from the
   slide's other FOVs (expected: identical for the first two, slightly worse for the third).

   Limits: one session and one mixed slide; the molecules are the slide's 1828 labeled traces; within-slide
   only (blanks, samples and calibrator share the slide), so slide-to-slide variation, of which the shift
   between pure and mixed slides is one example, is in no limit here, and comparisons between slides will be
   worse than the minimum detectable difference; the traces are those near a single-dye vesicle (an unknown
   sample would use every filtered trace); the trace length (the true re-read analog) is fixed.
   Result (2026-11-18; Revisions/DualColor/README.md "DFK851 limits of detection"): on the designs with about 16
   molecules per FOV (16/1, 50/3, 100/6, 200/12, 500/24 molecules / FOVs) the LoD of SNAP in HT is B not
   detectable, 0.64, 0.45, 0.31, 0.21 and C 0.79, 0.48, 0.35, 0.25, 0.16; of HT in SNAP B not detectable,
   0.68, 0.49, 0.35, 0.24 and C 0.71, 0.38, 0.26, 0.18, 0.11. Model check: C passes at every k, B fails at
   k = 1, 3, 6. Total error <= 0.10 is reached only by C calibrated with 500 molecules (SNAP 0.545-0.885 at 24
   FOVs, 0.76-0.87 at 12). At these densities the FOV term is 47-90 % of Var[T] for B and 23-38 % for C.
   Outcome and corrections (2026-11-18; independent verification, three agents, checks/20261118_dfk851/
   verify_lod; README there). Every number reproduced: a blind recomputation from this text alone (components
   identical, all grid cells to 2e-8; ML and REML FOV variances inside the bootstrap intervals, limits within
   0.015 on the diagonal), an independent reimplementation (Var[T] against exact binomial enumeration to 2e-14),
   and a model-free test of detection power on real whole-FOV samples (C's 0.95 crossing within 0.002 of the
   model at k = 6, 12, 24; B's 0.009-0.018 share higher: B's model is slightly optimistic). Clarifications of
   the text above: the MDD multiplier is 1.960 + 0.842 = 2.8016, with both samples' variances, searched upward
   from s0 only (downward differs, e.g. C 100/6 at 0.50: 0.413 vs 0.363); te_min (descriptive) is the minimum
   TE between the two LoDs (beyond them the clip at 0 or 1 fakes a small TE). Corrections to the reading:
   (1) B's model check fails at k = 1 through heavy lower tails (SNAP blanks below the bound 0.117, natural mix
   0.101), at k = 3 and 6 through too light upper tails (SNAP blanks 0.009, 0.018), which lie on the
   conservative side for the limits; near 95 % power B's model is 0.01-0.02 share optimistic at every k; at
   k = 12 B passes by the rule's 5000 samples (0.0252 above the bound) but not by 10^6 draws (0.0237), on a tail
   no limit uses. (2) The limits are oracle limits: the blank's mean and SD are taken as known. Estimating the
   blank mean from a calibrator raises the false-positive rate (112 FOVs: 0.063-0.065 at 200/12, about 0.08 at
   500/24; 24 FOVs: 0.16 at 500/24); most of the split-half check's excess tails (about 1.5 x nominal at k = 12)
   is that. A blank or calibrator needs many more FOVs than the sample. (3) B's deficit against C is its FOV
   variance alone (component swap: B with C's FOV components reaches C's limits), not the within-FOV variance.
   (4) The FOV effects of the two classes are strongly correlated for B but not a common shift (SD of the SNAP
   effect 1.6 x the HT effect's, 95 % CI 1.16-3.26); for C the correlation is undetermined. The raw correlation
   above 1 comes from mixing count-weighted variance estimates with an unweighted covariance; clipping it at 1
   lowers the limits by 0.001-0.011; with unweighted variances C's limits rise by up to 0.09 (16/1), with REML
   by at most 0.015 on the diagonal (0.044 at 500/1): C's small FOV variance is poorly determined (intervals
   reach 0). (5) More FOVs beat more molecules: confirmed on real traces at fixed n (1-3 molecules per FOV,
   within 0.02 of the model; the gain is small for C); the plateau at one FOV rests only on density-extrapolated
   cells. (6) Quantitation within 0.10 is at the edge at 500 molecules: C calibrated reaches it (LoQ 95 %
   interval 0.13 to not reached), B with the pure-slide constants misses by 0.007 at 500/24 (TE 0.107; within
   the goal in 25 % of bootstrap draws), so "never without calibration" is not robust. (7) Post hoc, one
   checker: within the slide's range C's per-FOV SNAP mean rises with the FOV's HT count (Spearman 0.26,
   p = 0.006 unadjusted, about 0.07 over 12 tests); extrapolated to no HT it points the way, and roughly the
   size, of the pure vs mixed shift. Either a composition effect (then blanks built from a mixed slide's HT
   traces do not represent an all-HT sample: the deferred pure-HT-slide check) or FOV quality tracking the
   counts; the data cannot separate them.

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
