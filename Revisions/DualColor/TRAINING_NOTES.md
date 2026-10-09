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

## Open

- Repeated runs (FOV-grouped k-fold within the single-protein slides, several seeds) to
  estimate the variance of the results, which a single split cannot. Proposal from the round-3
  check, to discuss: 5 folds grouped by FOV and stratified by slide (every single-protein FOV
  validated once; DFK785 31 + 31 validation FOVs instead of 7 + 7), 3 seeds, the mixed slides as
  the fixed test; architectures compared paired by (fold, seed) with a margin declared in
  advance; out-of-fold validation minus mixed AUC as the test of the validation inflation.
- Controls that would answer what round 3 could not: background training to a fixed number of
  epochs (the CNN-GRU background run stalled at ln 2), with random-initialization and skewness
  baselines; scrambled runs that fit their labels; a vesicle-population control (background
  boxes next to single-dye vesicles on the mixed slides, compared within FOV).
- Input option: remove the movie common mode (the leave-one-out mean of the movie's background
  traces) from the traces before training; expected to shrink the validation-mixed gap without
  lowering the mixed AUC.
