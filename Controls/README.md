# Controls

This folder contains five control experiments that validate and interrogate the main
Grx1 vs K20Ac TCN classifier reported in Püntener et al. 2026. Each experiment has
its own subdirectory with a Python script and a YAML config file. Results are written
to `Results/Controls/<experiment>/`.

---

## Dataset

| Item | Path |
|------|------|
| Protein traces (Grx1, K20Ac) | `/Volumes/MediumBerth/March27_NewRepo/Grx1_HTHTL_HTIA_K20Ac_snap_optparam_002/ProteinTracesIN/Filtered` |
| Background traces | `/Volumes/MediumBerth/March27_NewRepo/Grx1_HTHTL_HTIA_K20Ac_snap_optparam_002/BackgroundTraces/Filtered` |
| UniqueIDs PKL files | `/Volumes/MediumBerth/March27_NewRepo/Grx1_HTHTL_HTIA_K20Ac_snap_optparam_002/UniqueIDs` |
| Pre-trained model | `/Users/privera/Desktop/Puentener2026_ML/2026-04-14_17-26-33_tcn_noaug_both_mirrored_training_Grx1_K20Ac` |
| MCD results NPZ | `<pre-trained model dir>/MCD_results/traces_with_wasserstein.npz` |

Pickle files follow the naming convention `<Protein>_IN_filtered_<channel>.pkl` where
channel is one of `raw`, `minmax`, `zscored`, `bg_rm`. The model was trained on
`minmax` + `zscored` channels.

All training hyperparameters (unless noted otherwise) match `ML/config_train.yaml`.

---

## Experiment 1 — Label Scrambling

**Script:** `label_scrambling/scramble_train.py`  
**Config:** `label_scrambling/config_scramble.yaml`  
**Run locally (MPS/CPU):**
```bash
cd Controls/label_scrambling
conda run -n blink2env python scramble_train.py -c config_scramble.yaml
```

### Motivation
A classifier trained on randomly shuffled labels cannot learn meaningful structure and
should perform at chance level (≈50% balanced accuracy, AUC ≈ 0.5 for a binary
problem). This experiment confirms that the performance of the real model is not an
artifact of the training procedure, data loading, or evaluation pipeline.

### Implementation
`scramble_train.py` is a near-identical copy of `ML/train.py` with one modification:
immediately after the full dataset `(X, y)` is loaded by `build_dataset_from_keys`,
the label vector `y` is shuffled in place using `np.random.default_rng(seed).shuffle(y)`
**before** the train/val/test split. This ensures that all three splits receive
randomly relabelled traces, so the model cannot exploit any label information.
Everything else — model architecture (TCN), augmentation (mirroring), optimisation,
MC Dropout evaluation, UMAP, and output structure — is identical to the main training
pipeline.

### Expected output
- Confusion matrices near the diagonal random baseline (≈50%/50%).
- AUC ≈ 0.5, loss curves that do not converge.
- MCD Wasserstein distances uniformly low (model is uncertain on all traces).

---

## Experiment 2 — Noise Classification from Pre-trained Model

**Script:** `noise_classification/classify_noise.py`  
**Config:** `noise_classification/config_noise.yaml`  
**Run locally (MPS/CPU):**
```bash
cd Controls/noise_classification
conda run -n blink2env python classify_noise.py -c config_noise.yaml
```

### Motivation
The TCN classifier was trained exclusively on protein (signal) traces. Background
(noise) traces should, in principle, not resemble either Grx1 or K20Ac blinking
patterns. Classifying them with the pre-trained model tests whether the classifier
has inadvertently learned to distinguish noise from protein signal rather than
protein identity. If background traces are predicted with high confidence and split
roughly equally between Grx1 and K20Ac, the classifier is likely responding to
genuine protein features. If they cluster into one class or are all filtered out by
MC Dropout uncertainty, it suggests the model is sensitive to signal quality.

### Implementation
Inference-only script — no training occurs.

1. Reads `config_full.json` from the pre-trained run directory to reconstruct the
   exact model architecture and class mapping.
2. Loads `best_model.pth` onto the available device.
3. Loads background traces from `BackgroundTraces/Filtered` for both Grx1 and K20Ac
   (`minmax` + `zscored` channels, matching training).
4. Runs a deterministic forward pass to obtain class probabilities.
5. Runs MC Dropout (100 forward passes) to compute per-trace Wasserstein distances,
   mirroring the MCD evaluation in `ML/train.py`.
6. Outputs:
   - Bar chart: fraction of background traces predicted as Grx1 vs K20Ac.
   - Normalized confusion matrix: true background label (rows) vs predicted class (cols).
   - Wasserstein distance distribution histogram.
   - WD sweep plot (balanced accuracy vs retained fraction).
   - JSON metrics file.

### Expected output
Background traces should either be filtered out by high MCD uncertainty (low
Wasserstein distances) or predicted without a strong class preference, indicating
the model is not generalising meaningless noise patterns.

---

## Experiment 3 — Leave-One-Out Cross-Validation (Nested, Experiment-Based)

**Script:** `leave_one_out_cv/loocv.py`  
**Config:** `leave_one_out_cv/config_loocv.yaml`  
**Submit script (HPC/SLURM):** `leave_one_out_cv/submit_loocv.sh`  
**Run on HPC:**
```bash
cd Controls/leave_one_out_cv
sbatch submit_loocv.sh
# or directly:
python loocv.py -c config_loocv.yaml --n-gpus 4
```

> **Note:** This experiment is designed for HPC (CUDA, multiple GPUs). Running it
> locally would be impractical given the 20 training jobs required.

### Motivation
The protein traces come from five independent acquisition sessions (DK experiments).
Standard cross-validation mixes traces from all sessions in train and val, which can
overestimate generalisation if there is session-level batch variation. A leave-one-out
design that respects experiment boundaries (one session held out as test, one as
validation, three as training) provides a conservative estimate of how well the
classifier generalises to entirely unseen acquisitions.

### DK Experiments
| Experiment | Date |
|------------|------|
| `20250713_DK_Exp5` | 2025-07-13 |
| `20250714_DK_Exp6` | 2025-07-14 |
| `20250729_DK_Exp7` | 2025-07-29 |
| `20250730_DK_Exp8` | 2025-07-30 |
| `20251127_DK_Exp9` | 2025-11-27 |

### Implementation
Each trace is mapped to its DK experiment using the `origin` column of the UniqueIDs
PKL files (regex match on `YYYYMMDD_DK_ExpN`).

The design is a **nested leave-one-out**:

- **Outer loop (5 iterations):** one experiment is held out as the **TEST** set.
- **Inner loop (4 iterations):** for each remaining experiment taken as the **VAL**
  set, a TCN is trained on the other three experiments (TRAIN).

This yields **20 training runs** total. Each run produces a test confusion matrix
evaluated on the outer held-out experiment. The 4 test matrices within each outer
fold are averaged (mean ± std) to give a per-held-out-experiment result. The 5
per-experiment results are then aggregated into a final overall confusion matrix.

The 20 tasks are distributed across available GPUs using the same dynamic work-stealing
queue already implemented in `ML/crossval.py` (`_distribute_tasks_to_gpus` /
`_gpu_worker_loop` pattern). Each task is a `(test_exp, val_exp)` pair.

Model: **TCN only**. No augmentation. No trace trimming (`trim_end: 0`).
`max_traces_per_class: 0` (all traces used).

### Outputs
- `cv_folds_metrics.csv`: per-run metrics (AUC, balanced accuracy, epochs, timing).
- `confmat_<test_exp>.pdf`: mean ± std confusion matrix for each held-out experiment.
- `confmat_overall.pdf`: confusion matrix aggregated over all 20 runs.
- `loss_curve_<test_exp>_<val_exp>.pdf`: one loss curve per task.
- MCD Wasserstein sweep per task.

### Expected output
If the model generalises across experiments, per-held-out-experiment confusion
matrices should approach the performance of the main model. Large variance across
held-out experiments would indicate session-level batch effects.

---

## Experiment 4 — Blinking Features of Misclassified K20Ac Traces

**Script:** `misclassified_features/plot_misclassified_features.py`  
**Config:** `misclassified_features/config_misclassified.yaml`  
**Run locally:**
```bash
cd Controls/misclassified_features
conda run -n blink2env python plot_misclassified_features.py -c config_misclassified.yaml
```

### Motivation
MC Dropout uncertainty filtering (Wasserstein distance ≥ 0.61) retains only the
traces the model classifies with high confidence. Among those, some K20Ac traces are
nonetheless misclassified as Grx1. Examining the biophysical blinking features of
these traces tests whether they are genuinely atypical K20Ac traces (resembling Grx1
in their blinking dynamics) or whether misclassification arises despite normal K20Ac
blinking behaviour.

### Implementation
1. Load `traces_with_wasserstein.npz` from the MCD results of the pre-trained model.
   NPZ layout: `traces (N, 2, T)`, `labels (N,)`, `predictions (N,)`,
   `wasserstein_distances (N,)`, `class_names (2,)`, `unique_ids (N,)`.
2. Filter: `wasserstein_distances ≥ 0.61` AND `labels == 1` (K20Ac) AND
   `predictions == 0` (Grx1).
3. For each retained trace, run `gmm_classify_frames` (from `Extraction/utils.py`)
   on channel 0 (minmax) to detect ON/OFF state transitions.
4. Compute the same six features used in `Features/blink_features.py` (pooled mode):
   - Mean off-time (ms)
   - Mean on-time (ms)
   - Blinking rate (peaks s⁻¹, active window)
   - Duty cycle (fraction of active window in ON state)
   - CV of on-times
   - CV of off-times
5. Plot a 2×3 violin panel (one panel per feature). Save feature table as CSV.

### Expected output
If these misclassified traces are atypical K20Ac molecules with Grx1-like dynamics,
their feature distributions will overlap more with Grx1 than with the broader K20Ac
population. If they look like typical K20Ac, the misclassification is not explained
by simple biophysical differences.

---

## Experiment 5 — Random Forest Classifier on Handcrafted Blinking Features

**Script:** `random_forest/rf_classifier.py`  
**Config:** `random_forest/config_rf.yaml`  
**Run locally:**
```bash
cd Controls/random_forest
conda run -n blink2env python rf_classifier.py -c config_rf.yaml
```

### Motivation
The TCN operates on raw time-series and learns features implicitly. A random forest
trained on six hand-engineered blinking features tests how much discriminative
information is captured by simple summary statistics alone, without any learned
representation. This serves as a classical-ML baseline for the binary Grx1 vs K20Ac
classification task.

### Implementation
1. Load **all** Grx1 + K20Ac filtered traces from `ProteinTracesIN/Filtered`
   (`zscored` channel only, one channel per trace).
2. For each trace, run `gmm_classify_frames` from `Extraction/utils.py` to detect
   ON/OFF state transitions.
3. Compute six per-trace features from the active window:
   - Mean ON time
   - Mean OFF time
   - Standard deviation of ON times
   - Standard deviation of OFF times
   - Total ON time
   - Total OFF time
4. Stratified 80/20 train/test split (seeded).
5. Train `sklearn.ensemble.RandomForestClassifier` (default hyperparameters; 500
   trees, seeded).
6. Evaluate on test set. Outputs:
   - Normalised confusion matrix (PDF).
   - Feature importances bar chart (PDF).
   - Feature table CSV (all traces, with protein label and computed features).
   - JSON classification report (precision, recall, F1, AUC).

### Expected output
A random forest on six summary statistics will likely underperform the TCN, confirming
that raw time-series classifiers capture information not accessible to hand-engineered
features. However, if the random forest achieves competitive performance, it suggests
the blinking dynamics are largely captured by these six statistics.

---

## Running order recommendation

| # | Experiment | Where | Estimated time |
|---|-----------|-------|---------------|
| 1 | Label scrambling | Local | ~same as main training |
| 2 | Noise classification | Local | ~5–10 min |
| 4 | Misclassified features | Local | ~10–20 min (GMM per trace) |
| 5 | Random forest | Local | ~20–60 min (GMM on full dataset) |
| 3 | Leave-one-out CV | HPC | Hours (20 TCN training runs) |
