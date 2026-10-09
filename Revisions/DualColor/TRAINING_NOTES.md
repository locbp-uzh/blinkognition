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

## Open

- Repeated runs (FOV-grouped k-fold within the single-protein slides, several seeds) to
  estimate the variance of the results, which a single split cannot.
