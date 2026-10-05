# Classification of New Traces

`ML/classify.py` applies a trained model to traces it has not seen, without retraining:
the traces of a new experiment, a mixture, or a held-out set. It was added for the
revisions (two-vesicle / two-protein mixture, `Revisions/DFK785/`); `ML/train.py`,
`ML/crossval.py` and the paper's results are unchanged.

**Script:** `ML/classify.py`
**Config:** `ML/config_classify.yaml` (template)

```bash
cd ML
python classify.py -c config_classify.yaml
```

---

## What it does

1. Loads `best_model.pth` and `config_full.json` of a run of `ML/train.py` and rebuilds the
   model with train.py's own `build_model_from_config` (same architecture, dropout and
   kwargs). Classes and channels come from the training config.
2. Loads each trace set: filtered trace pickles as written by the extraction pipeline
   (one column per trace, column name = UniqueID), found with
   `utils.discover_protein_files` by file-name prefix and channel. Traces must have at least
   the training length (read from the training log, or `n_frames`); longer ones are trimmed.
3. Runs one deterministic pass and MC Dropout (`n_mc` passes, `utils.mc_dropout_predict`,
   as in training). The prediction is the argmax of the MC-mean probabilities.
4. Computes the Wasserstein uncertainty of every trace with the definition of
   `utils.evaluate_uncertainty_filtered`: the distance between the MC distribution of the
   predicted class's probability and that of the closest other class. A trace is kept if
   the distance exceeds the threshold.

### The threshold is fixed in advance

By default (`wasserstein_threshold: "model"`) the threshold is the one the training run
auto-selected on its own test set (`MCD_results/mc_dropout_metrics.json`,
`selected_threshold`); a number in the config fixes it otherwise. It is never re-tuned on
the traces being classified. Training auto-selects the threshold that maximizes the
balanced accuracy of the test set while removing at most `trace_loss` % of it; doing the
same on new data would fit the threshold to the labels of that data, and on unlabeled or
uncertainly labeled data (a mixture) it is not possible at all. The fraction of traces kept
on new data is therefore an outcome, not a setting: it need not be 50 %.

### Labels

A set may carry a `label` (one of the model's classes), e.g. a single-protein slide.
Labeled sets get accuracy with a Wilson 95 % interval (all traces and kept traces), and the
labeled sets pooled get confusion matrices and the Wasserstein sweep
(`utils.evaluate_uncertainty_filtered` with the fixed threshold; the sweep is a
diagnostic of how accuracy would change with the threshold, not a choice of it).

---

## Config

| Key | Meaning |
|---|---|
| `model_dir` | Training run folder (`Results/Train/<run>/`) |
| `run_name`, `output_root` | Output folder `<output_root>/<timestamp>_<run_name>/` |
| `trace_sets` | List of `{name, path, key, label?}`: folder, file-name prefix, optional true class |
| `channels` | Optional; default: the channels of the training config |
| `n_frames` | Optional; default: the training log's "Dataset loaded ... timesteps" |
| `mc_dropout.n_mc` | MC Dropout passes (100 as in training) |
| `mc_dropout.wasserstein_threshold` | `"model"` (default) or a number |
| `system.seed` | Seed for the MC Dropout masks |

## Outputs

```
<output_root>/<timestamp>_<run_name>/
├── predictions.csv      one row per trace: set, uniqueID, label, p_det_<class>, p_mc_<class>,
│                        p_mc_sd_<class>, prediction, wasserstein, kept
├── summary.csv          per set: n, n_kept, kept_frac, frac_pred_<class> (all, kept);
│                        with a label: accuracy and ci95_lo/hi (all, kept)
├── labeled/             pooled labeled sets: confusion_matrix/, filtered_confusion_matrix/,
│                        wasserstein_sweep/, mc_dropout_metrics.json
├── config_full.json, config_summary.json
└── <run_name>.log
```

## Check, and run on the training hardware

Classifying the 774-trace test set saved by the training run
(`MCD_results/traces_with_wasserstein.npz`) on Apple MPS (fp32) reproduces the training's
predictions for 97.8 % of the traces, initial accuracy 84.2 % (training 84.9 %) and
Wasserstein distances with r = 0.97. But the distances are systematically lower than on
the H100 the model was trained on (CUDA, bfloat16 autocast): by 0.037 on average (SE
0.002), and 14 % of the test traces lie within 0.05 of the threshold, so 45 % of the traces
are kept instead of 52 % (filtered accuracy 96.3 % against 95.5 %). The threshold is only
meaningful on the precision it was selected on: classify on the same kind of GPU as the
training run. The check is in `Revisions/DFK785/README.md` ("Final model").

Inference is about 1 s per trace with 100 MC passes on Apple MPS (M4 Max) and 0.025 s on an
H100.
