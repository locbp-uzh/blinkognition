#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH), Andreas Biri (ETHZ) and Roman Briskine (UZH).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

# train.py
import sys, os, json, argparse, random, re
from datetime import datetime
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import yaml

from utils import (
    build_dataset,
    build_dataset_from_keys,
    create_dataloaders,
    create_dataloaders_with_augmentation,
    train_model,
    plot_losses,
    evaluate_model,
    estimate_batch_size,
    mc_dropout_predict,
    evaluate_uncertainty_filtered,
    extract_embeddings,
    plot_umap_embeddings,
)

from utils import (
    detect_accelerator,
    setup_precision_and_flags,
    dataloader_kwargs_for,
    maybe_compile,
    batch_size_hint,
)

# model registry + factory
from models import build_model as build_from_zoo


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _slug(s: str) -> str:
    s = s or "run"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def make_run_dir(output_root: str, run_name: str, dataset_keys) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"{ts}_{_slug(run_name)}_training_{_slug('_'.join(dataset_keys))}"
    run_dir = os.path.join(output_root, dir_name)
    os.makedirs(os.path.join(run_dir, "MCD_results"), exist_ok=True)
    return run_dir


def build_model_from_config(model_name: str, in_channels: int, num_classes: int, dropout: float, model_kwargs: dict, device):
    """
    Build model using hardcoded defaults from model zoo with user overrides.

    Args:
        model_name: Name of the model to build
        in_channels: Number of input channels
        num_classes: Number of output classes
        dropout: Global dropout rate (optional override)
        model_kwargs: User-provided parameter overrides (from config)
        device: Target device
    """
    # Base parameters that all models need
    kwargs = {"in_channels": in_channels, "num_classes": num_classes}

    # Add dropout if specified (overrides hardcoded defaults)
    if dropout is not None and dropout > 0:
        kwargs["dropout"] = dropout

    # Handle TCN special case
    if model_name.lower() == "tcn":
        kwargs.pop("in_channels", None)
        kwargs["num_inputs"] = in_channels

    # Apply user overrides (from config)
    if model_kwargs:
        kwargs.update(model_kwargs)

    # Build model (defaults are applied automatically in build_from_zoo)
    model = build_from_zoo(model_name, **kwargs)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # sectioned config with backward-compatible fallbacks
    io_cfg     = cfg.get("io", {})
    data_cfg   = cfg.get("data", {})
    model_cfg  = cfg.get("model", {})
    opt_cfg    = cfg.get("optimization", cfg)
    uncert_cfg = cfg.get("uncertainty", {})
    system_cfg = cfg.get("system", cfg)

    # seed
    seed_val = system_cfg.get("seed", cfg.get("seed"))
    if seed_val is not None:
        set_seed(int(seed_val))

    # I/O and dataset
    output_root = io_cfg.get("output_root", cfg.get("output_root", "../Results/Train"))
    dataset     = data_cfg.get("dataset", cfg.get("dataset"))
    if dataset is None:
        raise ValueError("No dataset specified in config.")
    dataset_keys = list(dataset.keys())
    run_name    = io_cfg.get("run_name", cfg.get("run_name") or cfg.get("model") or "run")
    run_dir     = make_run_dir(output_root, run_name, dataset_keys)
    mcd_dir     = os.path.join(run_dir, "MCD_results")

    # logging + snapshot
    log_path = os.path.join(run_dir, f"{_slug(run_name)}_training_{_slug('_'.join(dataset_keys))}.log")
    log_file = open(log_path, "w")
    sys.stdout = sys.stderr = Tee(sys.__stdout__, log_file)
    print(f"Logging to: {log_path}")

    # Save full config for reference
    full_config_path = os.path.join(run_dir, "config_full.json")
    with open(full_config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote full config to: {full_config_path}")

    model_save_path  = os.path.join(run_dir, "best_model.pth")
    checkpoint_path  = os.path.join(run_dir, "checkpoint.pth")
    loss_plot_dir    = os.path.join(run_dir, "loss_curve")
    conf_matrix_dir  = os.path.join(run_dir, "confusion_matrix")
    umap_dir         = os.path.join(run_dir, "umap_embeddings")

    # data params
    trim_end = int(data_cfg.get("trim_end", cfg.get("trim_end", 0))) or None
    max_traces_per_class = int(data_cfg.get("max_traces_per_class", cfg.get("max_traces_per_class", 0))) or None
    balance_train = bool(data_cfg.get("balance_train", cfg.get("balance_train", True)))
    balance_test = bool(data_cfg.get("balance_test", cfg.get("balance_test", False)))
    balance_val = bool(data_cfg.get("balance_val", cfg.get("balance_val", False)))
    traces_path = data_cfg.get("traces_path", "../Data/traces")

    print("Loading dataset...")
    # Check if dataset contains file paths (old format) or protein keys (new format)
    first_key = next(iter(dataset))
    first_value = dataset[first_key]

    if isinstance(first_value, (list, str)) and (
        (isinstance(first_value, str) and first_value.endswith('.pkl')) or
        (isinstance(first_value, list) and len(first_value) > 0 and first_value[0].endswith('.pkl'))
    ):
        # Old format: file paths specified directly
        print("Using legacy dataset format with explicit file paths")
        X, y, class_map, in_channels, unique_ids = build_dataset(
            dataset, trim_end=trim_end, max_traces_per_class=max_traces_per_class, random_seed=seed_val
        )
    else:
        # New format: protein keys with file discovery
        print(f"Using new dataset format with protein keys, discovering files in: {traces_path}")
        X, y, class_map, in_channels, unique_ids = build_dataset_from_keys(
            dataset, traces_path, trim_end=trim_end, max_traces_per_class=max_traces_per_class, random_seed=seed_val
        )

    # Dataset information logging
    class_names = [class_map[i] for i in sorted(class_map)]
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} channels, {X.shape[2]} timesteps")
    print(f"Classes ({len(class_names)}): {class_names}")
    class_counts = {class_names[i]: int((y == i).sum()) for i in range(len(class_names))}
    print(f"Class distribution: {class_counts}")

    # accelerator + precision
    accel = detect_accelerator()
    device = accel["device"]
    print(f"Using device: {accel['name']} ({accel['type']})")

    amp_dtype, autocast_ctx, scaler = setup_precision_and_flags(accel)

    model_name = model_cfg.get("name", cfg.get("model") or "ResNet1DClassifier")

    # Augmentation configuration
    use_augmentation = data_cfg.get("augmentation", {}).get("enabled", False)
    if use_augmentation:
        augmentation_cfg = data_cfg.get("augmentation", {})
        print(f"\nData augmentation enabled (training set only):")
        print(f"  Time warping: {augmentation_cfg.get('time_warp_sigma', 0.03)}")
        print(f"  Gaussian noise: {augmentation_cfg.get('noise_sigma', 0.02)}")
        print(f"  Magnitude jitter: {augmentation_cfg.get('magnitude_jitter', 0.02)}")
        print(f"  Augmentation factor: {augmentation_cfg.get('aug_factor', 3)}x")
        print(f"  Mirror (time-reversal): {augmentation_cfg.get('include_mirror', False)}")

    # dataloaders
    print("\nSplitting into train/val/test...")
    batch_size = int(opt_cfg.get("batch_size", cfg.get("batch_size", 64)))
    loader_kwargs = dataloader_kwargs_for(accel)

    if use_augmentation:
        # Use augmented dataloaders (augmentation applied ONLY to training set)
        train_loader, val_loader, test_loader, _ytrain = create_dataloaders_with_augmentation(
            X, y,
            batch_size=batch_size,
            balance_train=balance_train,
            balance_test=balance_test,
            balance_val=balance_val,
            random_seed=seed_val,
            augment_train=True,
            aug_factor=augmentation_cfg.get("aug_factor", 3),
            time_warp_sigma=augmentation_cfg.get("time_warp_sigma", 0.03),
            noise_sigma=augmentation_cfg.get("noise_sigma", 0.02),
            magnitude_jitter=augmentation_cfg.get("magnitude_jitter", 0.02),
            include_mirror=augmentation_cfg.get("include_mirror", False),
            **loader_kwargs
        )
    else:
        # Standard dataloaders without augmentation
        train_loader, val_loader, test_loader, _ytrain = create_dataloaders(
            X, y, batch_size=batch_size, balance_train=balance_train, balance_test=balance_test,
            balance_val=balance_val, random_seed=seed_val, **loader_kwargs
        )

    # Save split indices for TRM to use (prevents data leakage)
    # Note: These are the ORIGINAL stratified split indices before any balancing
    # Recreate the split to get indices
    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=y, random_state=seed_val
    )
    y_temp = y[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y_temp, random_state=seed_val
    )
    split_indices_path = os.path.join(run_dir, "split_indices.npz")
    np.savez(split_indices_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
             balance_test=balance_test)
    print(f"Saved split indices to: {split_indices_path}")
    print(f"  Original split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    if balance_test:
        actual_test_size = len(test_loader.dataset)
        print(f"  Balanced test set used for evaluation: {actual_test_size} samples")

    # model configuration (simplified - defaults are in model zoo)
    num_classes = len(class_map) if class_map else len(np.unique(y))

    user_kwargs = {}
    if "kwargs" in model_cfg:
        user_kwargs.update(model_cfg["kwargs"])
    if "per_model" in model_cfg and model_name in model_cfg["per_model"]:
        user_kwargs.update(model_cfg["per_model"][model_name])
    dropout = model_cfg.get("dropout")
 
    print(f"Initializing model: {model_name}")
    try:
        import torch._dynamo as _dynamo
        if accel["type"] == "mps":
            _dynamo.config.suppress_errors = True
    except Exception:
        pass

    model = build_model_from_config(
        model_name=model_name,
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
        model_kwargs=user_kwargs,
        device=device,
    )

    # Save model build kwargs for later use (e.g., embedding extraction)
    # optimization parameters
    lr  = float(opt_cfg.get("lr",  cfg.get("lr",  1e-3)))
    wd  = float(opt_cfg.get("weight_decay", cfg.get("weight_decay", 0.0)))
    label_smoothing = float(opt_cfg.get("label_smoothing", 0.05))

    # compile (opt-in or SM80+ default)
    compile_enabled = bool(system_cfg.get("compile", accel["type"] == "cuda" and accel.get("cap", (0, 0))[0] >= 8))
    model = maybe_compile(model, accel, enabled=compile_enabled)

    # verbosity setting
    verbose = bool(system_cfg.get("verbose", False))
    max_epochs = int(opt_cfg.get("max_epochs", cfg.get("max_epochs", 100)))
    patience_limit = int(opt_cfg.get("patience_limit", cfg.get("patience_limit", 10)))
    threshold_metric = str(opt_cfg.get("threshold_metric", "argmax"))

    # Early stopping configuration
    early_stop_cfg = opt_cfg.get("early_stopping", {})
    auc_min_delta = float(early_stop_cfg.get("auc_min_delta", 0.005))
    loss_mode = str(early_stop_cfg.get("loss_mode", "relative"))
    loss_tolerance = float(early_stop_cfg.get("loss_tolerance", 1.10))
    # Alternative checkpoint: save if AUC within tolerance AND loss improves significantly
    auc_tolerance = early_stop_cfg.get("auc_tolerance")
    if auc_tolerance is not None:
        auc_tolerance = float(auc_tolerance)
    loss_min_delta = early_stop_cfg.get("loss_min_delta")
    if loss_min_delta is not None:
        loss_min_delta = float(loss_min_delta)

    clip_grad_norm = float(opt_cfg.get("clip_grad_norm", 1.0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    print("Starting training...")
    print(f"Early stopping: auc_min_delta={auc_min_delta}, loss_mode={loss_mode}, loss_tolerance={loss_tolerance}")
    if auc_tolerance is not None and loss_min_delta is not None:
        print(f"Alternative checkpoint: auc_tolerance={auc_tolerance}, loss_min_delta={loss_min_delta}")
    train_losses, val_losses, val_aucs, optimal_threshold = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_classes=num_classes,
        max_epochs=max_epochs,
        patience_limit=patience_limit,
        model_save_path=model_save_path,
        checkpoint_path=checkpoint_path,
        autocast_ctx=autocast_ctx,
        scaler=scaler,
        clip_grad_norm=clip_grad_norm,
        verbose=verbose,
        threshold_metric=threshold_metric,
        auc_min_delta=auc_min_delta,
        loss_mode=loss_mode,
        loss_tolerance=loss_tolerance,
        auc_tolerance=auc_tolerance,
        loss_min_delta=loss_min_delta,
    )

    print(f"Training complete. Saving loss plot to '{loss_plot_dir}'...")
    plot_losses(train_losses, val_losses, save_path=loss_plot_dir)

    print(f"Loading best model from '{model_save_path}' for evaluation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)

    print(f"Evaluating on test set. Saving confusion matrix to '{conf_matrix_dir}'...")
    label_names = dataset_keys
    test_metrics = evaluate_model(
        model, test_loader, device,
        label_names=label_names,
        save_path=conf_matrix_dir,
        autocast_ctx=autocast_ctx,
        optimal_threshold=optimal_threshold
    )

    # Save test metrics to JSON
    test_metrics_path = os.path.join(run_dir, "test_metrics.json")
    with open(test_metrics_path, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {}
        for k, v in test_metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                metrics_json[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, dict):
                metrics_json[k] = {str(kk): float(vv) if not np.isnan(vv) else None for kk, vv in v.items()}
            else:
                metrics_json[k] = v
        json.dump(metrics_json, f, indent=2)
    print(f"Test metrics saved to '{test_metrics_path}'")

    # Generate UMAP embeddings visualization
    print(f"Generating UMAP visualization of test set embeddings to '{umap_dir}'...")
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device=device, autocast_ctx=autocast_ctx)

    # Compute UMAP once for consistent visualization across plots
    import umap
    print("Computing UMAP coordinates for test embeddings...")
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=seed_val)
    test_umap_coords = umap_reducer.fit_transform(test_embeddings)

    plot_umap_embeddings(
        embeddings=test_embeddings,
        labels=test_labels,
        class_names=label_names,
        save_path=umap_dir,
        umap_coords=test_umap_coords,
        random_state=seed_val
    )

    # MC Dropout
    mcd_cfg = uncert_cfg.get("mc_dropout", cfg.get("mc_dropout", {"enabled": True, "n_mc": 100, "trace_loss": 50.0}))
    if mcd_cfg.get("enabled", True):
        bs_hint = batch_size_hint(batch_size, accel)
        est_bs = estimate_batch_size(model_name=model_name) if (accel["type"] == "cuda" and bs_hint is None) else (bs_hint or batch_size)
        print(f"Running MC Dropout with batch size: {est_bs}")

        # Build test tensors on CPU
        X_list, y_list = [], []
        for Xb, yb in test_loader:
            X_list.append(Xb.cpu())
            y_list.append(yb.cpu())
        X_test = torch.cat(X_list, dim=0).contiguous()
        y_test = torch.cat(y_list, dim=0).numpy()

        # Match X_test traces back to UniqueIDs via fingerprint on first channel, first 10 values.
        # X is cast to float32 to match the DataLoader's tensor conversion.
        X_fp = X.astype(np.float32)
        fp_lookup = {tuple(X_fp[i, 0, :10].tolist()): unique_ids[i] for i in range(len(X))}
        unique_ids_test = np.array(
            [fp_lookup.get(tuple(X_test[i, 0, :10].numpy().tolist()), -1) for i in range(len(X_test))],
            dtype=np.int64,
        )
        n_unmatched = int((unique_ids_test == -1).sum())
        if n_unmatched:
            print(f"Warning: {n_unmatched}/{len(X_test)} test traces could not be matched to a UniqueID.")
        else:
            print(f"UniqueID matching: all {len(X_test)} test traces matched.")

        with torch.no_grad():
            probs_mc = mc_dropout_predict(
                model, X_test,
                n_mc=int(mcd_cfg.get("n_mc", 100)),
                batch_size=est_bs,
                device=device,
                autocast_ctx=autocast_ctx,
            )

        # Get Wasserstein threshold: either fixed value or auto-select via trace_loss
        wasserstein_threshold = mcd_cfg.get("wasserstein_threshold", None)
        if wasserstein_threshold is not None:
            wasserstein_threshold = float(wasserstein_threshold)

        mcd_metrics = evaluate_uncertainty_filtered(
            probs_mc,
            y_true=y_test,
            threshold=wasserstein_threshold,
            class_names=label_names,
            show_plots=False,
            save_dir=mcd_dir,
            trace_loss=float(mcd_cfg.get("trace_loss", 50.0)),
            verbose=verbose,
            optimal_threshold=optimal_threshold,
            embeddings=test_embeddings,
            umap_coords=test_umap_coords,
            traces=X_test,
            unique_ids=unique_ids_test,
            random_state=seed_val,
        )

        # Save MC dropout metrics
        mcd_metrics_path = os.path.join(mcd_dir, "mc_dropout_metrics.json")
        with open(mcd_metrics_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            mcd_json = {}
            for k, v in mcd_metrics.items():
                if isinstance(v, (np.floating, np.integer)):
                    mcd_json[k] = float(v) if not np.isnan(v) else None
                elif isinstance(v, (np.ndarray, list)):
                    if isinstance(v, np.ndarray):
                        mcd_json[k] = v.tolist()
                    else:
                        mcd_json[k] = v
                else:
                    mcd_json[k] = v
            json.dump(mcd_json, f, indent=2)

        print("Finished MC Dropout evaluation.")

    print("All done.")
    summary = {
        "run_dir": run_dir,
        "model": model_name,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": wd,
        "max_epochs": max_epochs,
        "patience_limit": patience_limit,
        "device": str(device),
        "accel_type": accel["type"],
        "amp_dtype": str(amp_dtype) if amp_dtype is not None else "fp32",
        "compile": bool(compile_enabled),
    }
    with open(os.path.join(run_dir, "config_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
