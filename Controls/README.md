# Controls

Five control experiments that validate and interrogate the main Grx1 vs K20Ac TCN
classifier reported in Püntener et al. 2026. Each experiment has its own
subdirectory with a Python script and a YAML config file. Results are written to
`Results/Controls/<experiment>/`.

---

## Shared dataset paths

| Item | Path |
|------|------|
| Protein traces (Grx1, K20Ac) | `/Volumes/MediumBerth/March27_NewRepo/Grx1_HTHTL_HTIA_K20Ac_snap_optparam_002/ProteinTracesIN/Filtered` |
| Background traces | `/Volumes/MediumBerth/March27_NewRepo/Grx1_HTHTL_HTIA_K20Ac_snap_optparam_002/BackgroundTraces/Filtered` |
| UniqueIDs PKL files | `/Volumes/MediumBerth/March27_NewRepo/Grx1_HTHTL_HTIA_K20Ac_snap_optparam_002/UniqueIDs` |
| Pre-trained model | `/Users/privera/Desktop/Puentener2026_ML/2026-04-14_17-26-33_tcn_noaug_both_mirrored_training_Grx1_K20Ac` |
| MCD results NPZ | `<pre-trained model dir>/MCD_results/traces_with_wasserstein.npz` |

Pickle files follow the naming convention `<Protein>_IN_filtered_<channel>.pkl` where
channel is one of `raw`, `minmax`, `zscored`, `bg_rm`. The TCN was trained on
`minmax` + `zscored` channels.

All TCN hyperparameters (unless noted otherwise) match `ML/config_train.yaml`.

---

## Experiment 1 — Label Scrambling (TCN)

**Script:** `label_scrambling/scramble_train.py`
**Config:** `label_scrambling/config_scramble.yaml`
**Run locally (MPS/CPU):**
```bash
cd Controls/label_scrambling
conda run -n blink2env python scramble_train.py -c config_scramble.yaml
```

### Motivation
A TCN trained on randomly shuffled labels cannot learn meaningful structure and must
perform at chance level (≈50% balanced accuracy, AUC ≈ 0.5). This confirms that the
real model's performance is not an artifact of the training procedure, data loading,
or evaluation pipeline.

### Implementation
`scramble_train.py` is a near-identical copy of `ML/train.py` with one modification:
immediately after the full dataset `(X, y)` is assembled by `build_dataset_from_keys`,
the label vector `y` is shuffled in place with `np.random.default_rng(seed).shuffle(y)`
**before** the train/val/test split. This ensures all three splits receive randomly
relabelled traces, so no label information can be exploited.

Everything else is identical to the main pipeline: TCN architecture, mirroring
augmentation (`aug_factor=1`, `include_mirror=True`, no noise or warp), AdamW
optimisation, MC Dropout (100 passes), UMAP, and the full output directory structure.

Config highlights:
- `max_traces_per_class: 0` (all traces used)
- `trim_end: 0`
- Augmentation enabled with mirroring only (matches main training)

### Outputs
Same directory structure as `ML/train.py`:
- Loss curves, confusion matrix, UMAP embeddings
- `MCD_results/`: Wasserstein histogram, WD sweep, MCD-filtered metrics
- `loocv_folds_metrics.csv` is not produced (no LOO CV here)

### Expected result
Confusion matrix near the random diagonal (≈50%/50%), AUC ≈ 0.5, loss curves that
fail to converge. MCD Wasserstein distances uniformly low (model uncertain on all
traces).

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
The TCN was trained exclusively on protein (signal) traces. Background (noise) traces
should not resemble either Grx1 or K20Ac blinking patterns. Classifying them with the
pre-trained model tests whether the classifier has inadvertently learned to distinguish
noise from signal rather than protein identity. If background traces are predicted with
high confidence and split roughly equally between the two classes, the model is likely
responding to genuine protein features.

### Implementation
Inference-only — no training occurs.

1. Reads `config_full.json` from `pretrained_model_dir` to reconstruct the exact model
   architecture and class mapping, then loads `best_model.pth`.
2. Loads background traces from `BackgroundTraces/Filtered` for both Grx1 and K20Ac
   (`minmax` + `zscored` channels, matching training inputs).
3. Deterministic forward pass → class probabilities.
4. MC Dropout: 100 forward passes per trace → per-trace Wasserstein distance to the
   nearest competing class, auto-threshold sweep (≤50% trace loss budget), filtered
   confusion matrix and balanced accuracy.
5. UMAP on the penultimate-layer embeddings (same as `ML/train.py`).

Config keys:
- `pretrained_model_dir`: path to the saved training run
- `background_traces_path`: `BackgroundTraces/Filtered`
- `proteins: ["Grx1", "K20Ac"]`, `channels: ["minmax", "zscored"]`
- `mc_dropout.n_mc: 100`, `mc_dropout.trace_loss: 50.0`

### Outputs
```
<run_dir>/
  predicted_class_distribution.pdf    bar chart: fraction predicted as Grx1 vs K20Ac
  confusion_matrix/                   evaluate_model outputs
  test_metrics.json
  umap_embeddings/
  MCD_results/
    wasserstein_histogram.pdf
    wd_sweep.pdf
    traces_with_wasserstein.npz
  mc_dropout_metrics.json
  config_full.json
  <run_name>.log
```

### Expected result
Background traces should either be filtered out by high MCD uncertainty (low WD) or
predicted without a clear class preference, indicating the model does not generalise
to noise.

---

## Experiment 3 — Nested Leave-One-Out CV (TCN, Experiment-Based)

**Script:** `leave_one_out_cv/loocv.py`
**Config:** `leave_one_out_cv/config_loocv.yaml`
**Submit script (SLURM/HPC):** `leave_one_out_cv/submit_loocv.sh`
**Run on HPC:**
```bash
cd Controls/leave_one_out_cv
sbatch submit_loocv.sh
# or directly:
python loocv.py -c config_loocv.yaml --n-gpus 4
```

> **Note:** Designed for HPC (CUDA, 4 GPUs). Running locally would be impractical
> given the 20 parallel training jobs required.

### Motivation
Protein traces come from five independent DK acquisition sessions. Standard
cross-validation mixes traces from all sessions in train and val, which can
overestimate generalisation if there is session-level batch variation. A leave-one-out
design that respects experiment boundaries (one session held out as test, one as
validation, three as training) gives a conservative estimate of how well the classifier
generalises to entirely unseen acquisitions.

### DK Experiments
| Experiment | Date |
|------------|------|
| `20250713_DK_Exp5` | 2025-07-13 |
| `20250714_DK_Exp6` | 2025-07-14 |
| `20250729_DK_Exp7` | 2025-07-29 |
| `20250730_DK_Exp8` | 2025-07-30 |
| `20251127_DK_Exp9` | 2025-11-27 |

Each trace is mapped to its DK experiment via a regex (`YYYYMMDD_DK_ExpN`) applied to
the `origin` column of the UniqueIDs PKL files.

### Design (20 training runs)
- **Outer loop (5 iterations):** one experiment held out as the **TEST** set.
- **Inner loop (4 iterations per outer fold):** for each remaining experiment taken as
  the **VAL** set, a TCN is trained on the other three (TRAIN).

Each of the 20 `(test_exp, val_exp)` tasks produces one test confusion matrix evaluated
on the outer held-out experiment. The 4 matrices within each outer fold are averaged
(mean ± std) → per-experiment result. The 5 per-experiment results are then aggregated
into the overall confusion matrix.

The 20 tasks are distributed across GPUs using the same dynamic work-stealing queue
as `ML/crossval.py`. Deadlock protection: `result_queue.get(timeout=60 s)` with a
worker liveness check — if all workers have exited, collection aborts cleanly with
partial results.

### Training details
- Model: **TCN only**. No augmentation. No trace trimming (`trim_end: 0`).
- `max_traces_per_class: 0` (all traces used).
- All three splits are balanced by subsampling to the minority class.
- Early stopping: patience 10, AUC primary criterion, alternate checkpoint on loss
  improvement.
- Channels: `minmax` + `zscored` (2-channel input, matches main model).
- Optimiser: AdamW, lr=4×10⁻⁴, weight decay=4×10⁻⁴, label smoothing=0.001,
  gradient clip=1.0, max 500 epochs.

### MC Dropout per task
100 forward passes on the test set. Per-trace Wasserstein distance to the nearest
competing class. Auto-threshold sweep (50 points, ≤50% trace loss budget):
- Saves `wd_sweep_<test_exp>_<val_exp>.pdf` per task (dual-axis: balanced accuracy
  and retained fraction vs WD threshold, red dashed line at selected threshold).
- Applies the best threshold → `mcd_bal_acc_filtered`, MCD-filtered confusion matrix.

### Outputs
```
<run_dir>/
  loocv_folds_metrics.csv               all 20 runs (AUC, bal_acc, MCD, epochs, timing)
  confmat_<test_exp>.pdf                mean ± std CM averaged over 4 inner val folds
  confmat_<test_exp>_mcd.pdf            MCD-filtered version of the above
  confmat_overall.pdf                   aggregated over all 5 outer folds
  confmat_overall_mcd.pdf               MCD-filtered overall
  loss_<test_exp>_<val_exp>.pdf         one loss curve per task (20 total)
  wd_sweep_<test_exp>_<val_exp>.pdf     WD sweep per task (20 total)
  config_snapshot.json
  <run_name>_loocv.log
```

### Printed summary
Per test experiment: AUC mean ± std, balanced accuracy mean ± std, MCD acc mean ± std,
mean epochs (all averaged over the 4 inner validation folds). Overall: same metrics
aggregated across all 20 runs.

### Expected result
Per-held-out-experiment confusion matrices approaching the main model's performance
indicate the classifier generalises across acquisitions. Large variance across held-out
experiments indicates session-level batch effects.

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
MC Dropout filtering (WD ≥ 0.61) retains only high-confidence traces. Among those,
some K20Ac traces are still misclassified as Grx1. Examining their biophysical blinking
features tests whether misclassified traces are genuinely atypical (resembling Grx1 in
their blinking dynamics) or whether misclassification arises despite normal K20Ac
blinking behaviour.

### Implementation
1. Loads `traces_with_wasserstein.npz` from the pre-trained model's MCD results.
   NPZ fields: `traces (N, 2, T)`, `labels (N,)`, `predictions (N,)`,
   `wasserstein_distances (N,)`, `class_names (2,)`, `unique_ids (N,)`.
2. Filters: `wasserstein_distances ≥ 0.61` AND `labels == 1` (K20Ac) AND
   `predictions == 0` (Grx1 — misclassified).
3. Same filter applied to correctly classified high-WD Grx1 and K20Ac traces, used
   as reference populations.
4. For each retained trace, runs `gmm_classify_frames` (from `Extraction/utils.py`)
   on channel 0 (minmax) to detect ON/OFF state transitions.
5. Computes six per-trace features (same as `Features/blink_features.py` pooled mode):
   - Mean ON time (ms)
   - Mean OFF time (ms)
   - Blinking rate (peaks s⁻¹, over the active window)
   - Duty cycle (fraction of active window in ON state)
   - CV of ON times
   - CV of OFF times
6. Violin panel (2×3 layout, one panel per feature) comparing misclassified K20Ac
   against correct Grx1 and K20Ac populations. Mann–Whitney U test annotations.
   Feature table exported as CSV.

Config keys:
- `npz_path`: path to `traces_with_wasserstein.npz`
- `wasserstein_min: 0.61`
- `gmm.proba: 0.9`, `gmm.min_peak_width: 1`, `gmm.frame_interval_ms: 30.0`

### Outputs
```
<run_dir>/
  features_violin.pdf           2×3 violin panel
  features_misclassified.csv    per-trace feature table
  config_snapshot.json
```

### Expected result
If misclassified K20Ac traces have Grx1-like blinking dynamics, their feature
distributions will overlap more with the Grx1 population than with K20Ac. If they
look like typical K20Ac, biophysical features do not explain the misclassification.

---

## Experiment 5 — Random Forest Classifier (Handcrafted Blinking Features)

**Script:** `random_forest/rf_pipeline.py`
**Config:** `random_forest/config_rf.yaml`
**Run locally:**
```bash
cd Controls/random_forest
conda run -n blink2env python rf_pipeline.py -c config_rf.yaml
```

### Motivation
The TCN operates on raw time-series and learns features implicitly. A random forest
trained on six hand-engineered blinking features tests how much discriminative
information is captured by simple summary statistics alone, without any learned
representation. It also provides a classical-ML baseline and an internal
label-scrambling and LOO CV control that mirror the TCN controls.

### Design
Features are extracted **once** from the full dataset and reused across all three
analyses in the same run:

| Section | What it does |
|---------|-------------|
| **Classifier** | Standard RF on an 80/20 train/test split |
| **Label scrambling** | Same RF after globally shuffling labels; expected ≈50% |
| **LOO CV** | Experiment-based leave-one-out CV (one fold per DK experiment) |

LOO CV requires `uid_path` in the config. If omitted, only the classifier and scramble
sections run.

### Feature extraction
- Traces channel: `zscored` (single channel per trace).
- For each trace, `gmm_classify_frames` (from `Extraction/utils.py`) detects ON/OFF
  transitions using a 2-component GMM (posterior probability threshold 0.9, minimum
  peak width 1 frame, frame interval 30 ms). Traces with fewer than 2 detected ON
  events are discarded.
- Six features per trace:
  - Mean ON time (ms)
  - Mean OFF time (ms)
  - Std of ON times (ms)
  - Std of OFF times (ms)
  - Total ON time (ms)
  - Total OFF time (ms)
- Extraction is parallelised across all available CPU cores with `joblib.Parallel`.
- Experiment labels are derived from the UniqueIDs PKL files: the `origin` field
  encodes the full path to the source PKL, and `Path(origin).parent.parent.name` gives
  the experiment directory name. This is regex-free and works for any naming convention
  (DK, SP, etc.).

### Section 1 — Classifier
- Stratified 80/20 train/test split (seeded).
- Both splits are undersampled to the minority class before training.
- `RandomForestClassifier`: 500 trees, `n_jobs=-1`, seeded.
- **Vote margin filtering:** the confidence proxy is
  `|p(class0) − p(class1)|` (margin). A sweep over 50 thresholds between 0 and 1
  finds the threshold that maximises balanced accuracy subject to ≤50% trace removal
  (`trace_loss`). The best threshold, filtered confusion matrix, and sweep scatter plot
  (retained % vs balanced accuracy, coloured by threshold) are saved to
  `classifier/margin_results/`.

Outputs:
```
classifier/
  confusion_matrix.pdf          normalised CM on the unfiltered test set
  feature_importances.pdf       horizontal bar chart (mean decrease in impurity)
  metrics.json                  AUC, balanced accuracy, margin filtering summary
  margin_results/
    margin_histogram.pdf        distribution of vote margins
    margin_sweep.pdf            scatter: retained % vs balanced accuracy vs threshold
    confusion_matrix_filtered.pdf
    traces_with_margin.npz
    margin_metrics.json
```

### Section 2 — Label scrambling
- The full label vector `y` is shuffled with `np.random.default_rng(seed).shuffle(y)`
  **before** train/test split, so no class information is available to the classifier.
- Same RF and undersampling as Section 1.
- Saves a confusion matrix and metrics JSON; no margin sweep (not meaningful with
  random labels).

Outputs:
```
scrambled/
  confusion_matrix.pdf
  metrics.json
```

### Section 3 — Experiment-based LOO CV
A simple (non-nested) leave-one-out: each experiment is held out as the test fold once,
with all remaining experiments used for training. One RF is trained per fold.

- Undersampling applied to both train and test folds.
- Same margin filtering as Section 1 is applied per fold: best threshold found under
  ≤50% trace loss budget; margin histogram, sweep, and filtered CM saved per fold.
- Per-fold confusion matrices are averaged into an aggregate mean ± std CM.
- Margin-filtered CMs are likewise averaged separately.
- Feature distributions (violin plots per experiment and protein) saved before folding.

Outputs:
```
loocv/
  feature_distributions.pdf     2×3 violin panel: feature × experiment, coloured by protein
  per_fold/
    <experiment>/
      confusion_matrix.pdf      unfiltered CM for this fold
      metrics.json
      margin_results/           same structure as classifier/margin_results/
  aggregated/
    loocv_metrics.json          per-fold and aggregate metrics (mean, std, AUC)
    confusion_matrix_mean_std.pdf
    margin_results/
      confusion_matrix_margin_mean_std.pdf
```

### Config keys (`config_rf.yaml`)
```yaml
traces_path:  <path to ProteinTracesIN/Filtered>
uid_path:     <path to UniqueIDs>          # optional; enables LOO CV
proteins:     ["Grx1", "K20Ac"]
channel:      "zscored"
output_root:  "../../Results/Controls/RandomForest"

gmm:
  proba:             0.9
  min_peak_width:    1
  frame_interval_ms: 30.0

random_forest:
  n_estimators: 500
  test_size:    0.20
  seed:         840410
  trace_loss:   50.0    # max % traces removable by margin filtering
```

### Expected result
The RF will likely underperform the TCN, confirming that the raw time-series contains
information beyond what these six statistics capture. The scramble section should give
≈50% balanced accuracy and AUC ≈ 0.5. The LOO CV provides an experiment-robust
estimate of RF generalisation for comparison with the TCN LOO CV.

---

## Running order

| # | Experiment | Script | Where | Estimated time |
|---|------------|--------|-------|----------------|
| 1 | Label scrambling (TCN) | `scramble_train.py` | Local (MPS/CPU) | Same as main training |
| 2 | Noise classification | `classify_noise.py` | Local (MPS/CPU) | ~5–10 min |
| 4 | Misclassified features | `plot_misclassified_features.py` | Local | ~10–20 min (GMM per trace) |
| 5 | Random forest pipeline | `rf_pipeline.py` | Local | ~20–60 min (GMM + 3 analyses) |
| 3 | TCN leave-one-out CV | `loocv.py` | HPC (4 GPUs) | Several hours (20 training runs) |
