# Control Experiments

Three experiments validate that a trained TCN classifier learns genuine protein-specific blinking features rather than artifacts of the training pipeline, the data distribution, or the choice of architecture. All three use the same protein traces, the same preprocessing, and the same evaluation metrics as the main model.

Scripts live in `Controls/`, results are written to `Results/Controls/`.

---

## 1. Label Scrambling

**Script:** `Controls/label_scrambling/scramble_train.py`
**Config:** `Controls/label_scrambling/config_scramble.yaml`

```bash
cd Controls/label_scrambling
python scramble_train.py -c config_scramble.yaml
```

### Rationale

A TCN trained on randomly shuffled labels cannot learn meaningful structure and must perform at chance level. If it still achieves above-chance performance, the result implies the model is exploiting some artifact of the training procedure, data loading, or evaluation pipeline rather than protein-specific signal.

### Design

`scramble_train.py` is structurally identical to `ML/train.py`. The only modification is that immediately after the full dataset (X, y) is assembled, the label vector y is shuffled in place with `np.random.default_rng(seed).shuffle(y)` before the train/val/test split. All three splits therefore receive randomly relabelled traces with no label information surviving into any fold.

All other settings — model architecture, augmentation, optimizer, early stopping, MC Dropout, and dataset balancing — match the main training run exactly.

### Expected outcome

Validation AUC oscillates around 0.5 throughout training and early stopping triggers quickly. Test accuracy, balanced accuracy, and AUC-PR are all near chance. Confusion matrices are near the random diagonal.

### Outputs

Same directory structure as `ML/train.py`: loss curves, confusion matrix, `MCD_results/` (Wasserstein histogram, WD sweep, `traces_with_wasserstein.npz`), `test_metrics.json`.

---

## 2. Noise Classification

**Script:** `Controls/noise_classification/classify_noise.py`
**Config:** `Controls/noise_classification/config_noise.yaml`

```bash
cd Controls/noise_classification
python classify_noise.py -c config_noise.yaml
```

### Rationale

The TCN is trained exclusively on protein (signal) traces. Background traces from the same acquisitions — camera noise, autofluorescence, non-specific binding events — should not resemble either protein class. Classifying them with the pre-trained model tests whether the classifier has inadvertently learned to separate signal from noise rather than to discriminate between protein identities.

Two outcomes are acceptable: (1) background traces classified with low confidence (high MCD uncertainty), indicating the model recognises they do not match either learned pattern; or (2) background traces classified with high confidence but split roughly evenly across classes, consistent with chance-level assignment of non-informative inputs. A strong systematic bias toward one class would be a red flag.

### Design

Inference only — no training occurs. The script loads `config_full.json` from the pretrained model directory to reconstruct the exact TCN architecture, then runs a deterministic forward pass followed by MC Dropout (100 passes). The Wasserstein-distance threshold sweep and all output plots match the `train.py` output format.

An additional bar chart (`predicted_class_distribution.pdf`) shows the mean predicted probability for each class broken down by background source, with standard error bars.

### Expected outcome

Background traces classified near chance (mean max probability ≈ 0.5–0.6) with no strong systematic class bias.

### Outputs

```
<run_dir>/
  predicted_class_distribution.pdf
  confusion_matrix/
  test_metrics.json
  MCD_results/
    wasserstein_sweep.*
    traces_with_wasserstein.npz
  mc_dropout_metrics.json
  config_full.json
```

---

## 3. Random Forest on Handcrafted Blinking Features

**Script:** `Controls/random_forest/rf_pipeline.py`
**Config:** `Controls/random_forest/config_rf.yaml`

```bash
cd Controls/random_forest
python rf_pipeline.py -c config_rf.yaml
```

### Rationale

The TCN operates on raw time-series and learns its own internal representation. A random forest trained on six hand-engineered blinking statistics tests how much discriminative information is already captured by simple summary statistics. This provides a classical-ML baseline and an internal label-scrambling control at the feature level.

### Feature extraction

Traces are loaded from the `minmax` channel. A 2-component GMM classifies every frame as ON or OFF (posterior threshold 0.9, minimum peak width 1 frame). Traces with fewer than 2 detected ON events are discarded. Six features are computed per trace:

| Feature | Description |
|---------|-------------|
| Mean ON time | Mean ON-state dwell time (ms) |
| Mean OFF time | Mean OFF-state dwell time (ms) |
| Std ON time | Standard deviation of ON dwell times (ms) |
| Std OFF time | Standard deviation of OFF dwell times (ms) |
| Total ON time | Sum of all ON dwell times (ms) |
| Total OFF time | Sum of all OFF dwell times (ms) |

### Analysis sections

The pipeline runs two analyses in sequence on the same feature matrix:

**Classifier.** Stratified 80/20 train/test split, both halves undersampled to the minority class. `RandomForestClassifier` with 500 trees. A vote-margin filter (|p(class 0) − p(class 1)|) sweeps 50 thresholds to find the value that maximises balanced accuracy subject to ≤50% trace removal.

**Label scrambling.** The full label vector is globally shuffled before the train/test split. Same training procedure as above. Expected result: balanced accuracy ≈ 50%, AUC ≈ 0.5.

### Expected outcome

The RF achieves above-chance but lower balanced accuracy and AUC than the TCN. A large performance gap confirms that the raw temporal structure carries information beyond simple dwell-time statistics. The label-scrambling section gives ≈50% accuracy.

### Outputs

```
<run_dir>/
  features.csv
  classifier/
    confusion_matrix.pdf
    feature_importances.pdf
    metrics.json
    margin_results/
      margin_histogram.pdf
      margin_sweep.pdf
      confusion_matrix_filtered.pdf
      traces_with_margin.npz
      margin_metrics.json
  scrambled/
    confusion_matrix.pdf
    predictions.npz
    metrics.json
```

---

## Running order

| # | Experiment | Script | Estimated time |
|---|------------|--------|----------------|
| 1 | Label scrambling (TCN) | `scramble_train.py` | ~30–60 min |
| 2 | Noise classification | `classify_noise.py` | ~10–20 min |
| 3 | Random forest pipeline | `rf_pipeline.py` | ~20–40 min |
