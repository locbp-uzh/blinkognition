#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH), Andreas Biri (ETHZ) and Roman Briskine (UZH).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
utils.py
Helper functions for Blinkognition2
"""

import os
import re
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    balanced_accuracy_score, roc_auc_score, classification_report, recall_score,
    matthews_corrcoef, f1_score, precision_recall_curve, auc as sklearn_auc
)
from sklearn.preprocessing import label_binarize
from torchmetrics.functional import accuracy
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from pathlib import Path
from collections import Counter

# Crameri scientific colormaps bundled with this repo
_CMAP_DIR = Path(__file__).parent.parent / "assets" / "colormaps"

def _load_cmap(name: str):
    """Load a Crameri colormap from bundled assets, with viridis fallback."""
    lut = _CMAP_DIR / f"{name}.txt"
    if lut.exists():
        return mcolors.LinearSegmentedColormap.from_list(name, np.loadtxt(lut))
    return mpl.colormaps.get_cmap("viridis")

_DAVOS_R = _load_cmap("davos").reversed()
_LAPAZ   = _load_cmap("lapaz")

# Okabe-Ito color-blind friendly palette
COLORS = {
    'orange':    '#E69F00',
    'sky_blue':  '#56B4E9',
    'green':     '#009E73',
    'yellow':    '#F0E442',
    'blue':      '#0072B2',
    'vermillion':'#D55E00',
    'pink':      '#CC79A7',
    'black':     '#000000',
}

# Font size hierarchy
FONTSIZE_LABEL  = 7
FONTSIZE_TICK   = 6
FONTSIZE_TITLE  = 7
FONTSIZE_LEGEND = 6


def apply_rcparams():
    """Apply global matplotlib rcParams to match locbplots standards."""
    mpl.rcParams['font.family']        = 'sans-serif'
    mpl.rcParams['font.sans-serif']    = ['Helvetica', 'Arial', 'DejaVu Sans']
    mpl.rcParams['font.size']          = FONTSIZE_LABEL
    mpl.rcParams['axes.titlesize']     = FONTSIZE_TITLE
    mpl.rcParams['axes.labelsize']     = FONTSIZE_LABEL
    mpl.rcParams['xtick.labelsize']    = FONTSIZE_TICK
    mpl.rcParams['ytick.labelsize']    = FONTSIZE_TICK
    mpl.rcParams['legend.fontsize']    = FONTSIZE_LEGEND
    mpl.rcParams['figure.dpi']         = 150
    mpl.rcParams['savefig.dpi']        = 450
    mpl.rcParams['savefig.bbox']       = 'tight'
    mpl.rcParams['lines.linewidth']    = 1.2
    mpl.rcParams['patch.linewidth']    = 0.8
    mpl.rcParams['axes.grid']          = False
    mpl.rcParams['pdf.fonttype']       = 42
    mpl.rcParams['svg.fonttype']       = 'none'


apply_rcparams()


def apply_axis_standards(ax):
    """Apply standard axis formatting for all plots."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICK,
                   length=4, width=0.8, direction='out')
    ax.tick_params(axis='both', which='minor', length=2, width=0.6, direction='out')


from scipy.stats import wasserstein_distance
import datetime

from models import MCDropout, LockedDropout



# --- Data Loading ---
def load_series(path):
    obj = pd.read_pickle(path)

    if isinstance(obj, pd.DataFrame):
        cols_to_keep = [c for c in obj.columns]
        arr = obj.loc[:, cols_to_keep].values.T  # (n_traces, T)

    elif isinstance(obj, pd.Series):
        name = obj.name or ""
        if name.endswith("_m"):
            raise ValueError(f"Dropped mirrored series '{name}'—cannot load.")
        arr = obj.values[np.newaxis, :]  # (1, T)

    else:
        arr = np.asarray(obj)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (n_traces, T), got {arr.shape}")

    return arr


def _load_col_names(path):
    """Return the column names (UniqueIDs) from a traces pkl file as an int64 array."""
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        return np.array(list(obj.columns), dtype=np.int64)
    elif isinstance(obj, pd.Series):
        return np.array([obj.name], dtype=np.int64)
    else:
        arr = np.asarray(obj)
        n = arr.shape[0] if arr.ndim >= 2 else 1
        return np.arange(n, dtype=np.int64)


def discover_protein_files(traces_path, protein_key, channels=None):
    """
    Discover pickle files for a given protein key in the traces directory.

    Args:
        traces_path (str): Path to traces directory
        protein_key (str): Protein name to match files against
        channels (list, optional): List of channel identifiers to find specific files.
                                  If None, returns all matching files.

    Returns:
        list: File paths in the order specified by channels (if provided) or sorted alphabetically
    """
    if not os.path.exists(traces_path):
        raise ValueError(f"Traces path does not exist: {traces_path}")

    files_in_dir = os.listdir(traces_path)

    if channels is None:
        # Auto-discovery mode: find all matching files
        matching_files = []
        for filename in files_in_dir:
            if not filename.endswith('.pkl'):
                continue
            if filename.lower().startswith(protein_key.lower()):
                matching_files.append(os.path.join(traces_path, filename))

        if not matching_files:
            raise ValueError(f"No .pkl files found for protein key '{protein_key}' in {traces_path}")

        return sorted(matching_files)

    else:
        # Channel specification mode: find files matching each channel
        channel_files = []
        for channel in channels:
            channel_file = None

            for filename in files_in_dir:
                if not filename.endswith('.pkl'):
                    continue
                if not filename.lower().startswith(protein_key.lower()):
                    continue

                # Look for channel identifier in filename
                if channel.lower() in filename.lower():
                    channel_file = os.path.join(traces_path, filename)
                    break

            if channel_file is None:
                raise ValueError(f"No .pkl file found for protein '{protein_key}' with channel '{channel}' in {traces_path}")

            channel_files.append(channel_file)

        return channel_files


def build_dataset_from_keys(dataset_keys, traces_path, trim_end=0, max_traces_per_class=None, random_seed=42):
    """
    Build dataset from protein keys by discovering files in traces_path.

    Args:
        dataset_keys (dict): Dictionary with protein names as keys and channel specifications as values.
                           Values can be:
                           - {} (empty dict): Auto-discover all files
                           - {"channels": ["minmax", "zscored"]}: Explicit channel specification
        traces_path (str): Path to directory containing trace pickle files
        trim_end (int): Number of timepoints to trim from end
        max_traces_per_class (int): Maximum traces per class
        random_seed (int): Random seed for reproducibility

    Returns:
        tuple: (X, y, class_map, in_channels)
    """
    # Convert keys to file paths
    dataset_dict = {}
    for protein_key, config in dataset_keys.items():
        if isinstance(config, dict) and "channels" in config:
            # Explicit channel specification
            channels = config["channels"]
            dataset_dict[protein_key] = discover_protein_files(traces_path, protein_key, channels)
        else:
            # Auto-discovery mode (backward compatibility with {} or None)
            dataset_dict[protein_key] = discover_protein_files(traces_path, protein_key)

    return build_dataset(dataset_dict, trim_end, max_traces_per_class, random_seed)  # includes unique_ids


def build_dataset(dataset_dict, trim_end=0, max_traces_per_class=None, random_seed=42):
    sample_names = list(dataset_dict.keys())
    X_list, y_list, uid_list = [], [], []

    rng = np.random.default_rng(seed=random_seed)

    # global min length across all inputs
    def _iter_paths(dct):
        for v in dct.values():
            if isinstance(v, str):
                yield v
            else:
                for p in v:
                    yield p

    min_T = min(load_series(p).shape[-1] for p in _iter_paths(dataset_dict))

    # per-class load, align, trim, subsample
    for class_idx, name in enumerate(sample_names):
        paths = dataset_dict[name]
        if isinstance(paths, str):
            paths = [paths]

        channel_arrays = [load_series(p) for p in paths]
        shapes = [arr.shape for arr in channel_arrays]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(f"Shape mismatch across channels for '{name}': {shapes}")

        X_i = np.stack(channel_arrays, axis=1)  # (n_traces, channels, T)

        # UniqueIDs from the first channel file (columns are shared across channels)
        uid_i = _load_col_names(paths[0])

        # trim to min_T
        if X_i.shape[2] > min_T:
            X_i = X_i[:, :, :min_T]

        # optional extra trimming from end
        if trim_end and trim_end > 0:
            T = X_i.shape[2]
            if trim_end >= T:
                raise ValueError(f"Cannot trim {trim_end} from total length {T}")
            X_i = X_i[:, :, :-trim_end]

        # optional subsample — apply same indices to uid_i
        if max_traces_per_class is not None and X_i.shape[0] > max_traces_per_class:
            idx = rng.choice(X_i.shape[0], max_traces_per_class, replace=False)
            X_i  = X_i[idx]
            uid_i = uid_i[idx]

        n_traces = X_i.shape[0]
        y_i = np.full(n_traces, class_idx, dtype=int)

        X_list.append(X_i)
        y_list.append(y_i)
        uid_list.append(uid_i)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    unique_ids = np.concatenate(uid_list, axis=0)
    class_map = {i: name for i, name in enumerate(sample_names)}
    first_val = dataset_dict[next(iter(dataset_dict))]
    in_channels = len(first_val) if isinstance(first_val, list) else 1

    return X, y, class_map, in_channels, unique_ids


# --- Dataset (tensorized, multiprocessing-safe) ---
def _balance_dataset(X, y, random_seed=42):
    """
    Balance a dataset by subsampling majority classes to match the minority class size.

    Args:
        X: Data array of shape (N, ...)
        y: Labels array of shape (N,)
        random_seed: Random seed for reproducibility

    Returns:
        X_balanced, y_balanced: Balanced arrays
    """
    rng = np.random.default_rng(seed=random_seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    balanced_indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > min_count:
            # Subsample to match minority class
            selected = rng.choice(cls_indices, size=min_count, replace=False)
        else:
            selected = cls_indices
        balanced_indices.extend(selected)

    balanced_indices = np.array(balanced_indices)
    rng.shuffle(balanced_indices)

    return X[balanced_indices], y[balanced_indices]


def _make_tensor_dataset(X, y):
    X_t = torch.as_tensor(X, dtype=torch.float32)
    X_t = torch.nan_to_num(X_t, nan=0.0, posinf=1e6, neginf=-1e6).contiguous()
    y_t = torch.as_tensor(y, dtype=torch.long).contiguous()
    return TensorDataset(X_t, y_t)

# --- Dataloaders ---
def create_dataloaders(X, y, batch_size, balance_train=True, balance_test=False, balance_val=False,
                       random_seed=42, **dl_kwargs):
    """
    Create train/val/test dataloaders.

    Args:
        X: Data array of shape (N, C, T)
        y: Labels array of shape (N,)
        batch_size: Batch size for dataloaders
        balance_train: Whether to use weighted sampling for training
        balance_test: Whether to balance test set by subsampling majority classes
        balance_val: Whether to balance val set by subsampling majority classes
        random_seed: Random seed for reproducibility
        **dl_kwargs: Additional arguments for DataLoader

    Returns:
        train_loader, val_loader, test_loader, y_train
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    # Balance val set if requested
    if balance_val:
        original_val_size = len(y_val)
        X_val, y_val = _balance_dataset(X_val, y_val, random_seed=random_seed)
        print(f"Balanced val set: {original_val_size} -> {len(y_val)} samples")
        print(f"  Val set class distribution: {Counter(y_val)}")

    # Balance test set if requested
    if balance_test:
        original_test_size = len(y_test)
        X_test, y_test = _balance_dataset(X_test, y_test, random_seed=random_seed)
        print(f"Balanced test set: {original_test_size} -> {len(y_test)} samples")
        print(f"  Test set class distribution: {Counter(y_test)}")

    train_ds = _make_tensor_dataset(X_train, y_train)
    val_ds   = _make_tensor_dataset(X_val,   y_val)
    test_ds  = _make_tensor_dataset(X_test,  y_test)

    # persistent_workers only valid when num_workers > 0
    if dl_kwargs.get("persistent_workers", False) and (dl_kwargs.get("num_workers", 0) == 0):
        dl_kwargs = dict(dl_kwargs)
        dl_kwargs["persistent_workers"] = False
    # prefetch_factor only valid when num_workers > 0
    if dl_kwargs.get("num_workers", 0) == 0 and "prefetch_factor" in dl_kwargs:
        dl_kwargs = dict(dl_kwargs)
        dl_kwargs.pop("prefetch_factor", None)

    # prefer fork context to avoid pickling quirks
    extra = {}
    if dl_kwargs.get("num_workers", 0) > 0:
        try:
            import multiprocessing as mp
            extra["multiprocessing_context"] = mp.get_context("fork")
        except Exception:
            pass

    if balance_train:
        class_weights = 1.0 / (np.bincount(y_train) + 1e-6)
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **dl_kwargs, **extra)
        # distribution preview without spinning workers
        try:
            idxs = list(sampler)
            sampled_labels = y_train[np.asarray(idxs)]
            print("Original train set class distribution:", Counter(y_train))
            print("Sampled-by-sampler class distribution:", Counter(sampled_labels.tolist()))
        except Exception as e:
            print(f"Sampler preview skipped: {e}")
    else:
        print("Original train set class distribution:", Counter(y_train))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs, **extra)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, **dl_kwargs, **extra)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dl_kwargs, **extra)

    return train_loader, val_loader, test_loader, y_train


def create_dataloaders_with_augmentation(X, y, batch_size, balance_train=True, balance_test=False,
                                         balance_val=False, random_seed=42, augment_train=False,
                                         aug_factor=2, time_warp_sigma=0.03, noise_sigma=0.02,
                                         magnitude_jitter=0.02, include_mirror=False, **dl_kwargs):
    """
    Create train/val/test dataloaders with augmentation applied ONLY to training set.

    This prevents data leakage by ensuring augmented versions of a trace only appear
    in the training set, never in validation or test sets.

    Args:
        X, y: Full dataset
        batch_size: Batch size for dataloaders
        balance_train: Whether to use weighted sampling for training
        balance_test: Whether to balance test set by subsampling majority classes
        balance_val: Whether to balance val set by subsampling majority classes
        random_seed: Random seed for reproducibility
        augment_train: Whether to augment training set
        aug_factor: Number of augmented versions to generate (total = original + augmented)
        time_warp_sigma: Sigma for time warping augmentation
        noise_sigma: Sigma for Gaussian noise augmentation
        magnitude_jitter: Magnitude jitter factor
        include_mirror: If True, add one time-reversed copy of every training trace
        **dl_kwargs: Additional arguments for DataLoader

    Returns:
        train_loader, val_loader, test_loader, y_train
    """
    # Split FIRST, before any augmentation
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    print(f"\nOriginal split sizes:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # Balance val set if requested
    if balance_val:
        original_val_size = len(y_val)
        X_val, y_val = _balance_dataset(X_val, y_val, random_seed=random_seed)
        print(f"\nBalanced val set: {original_val_size} -> {len(y_val)} samples")
        print(f"  Val set class distribution: {Counter(y_val)}")

    # Balance test set if requested
    if balance_test:
        original_test_size = len(y_test)
        X_test, y_test = _balance_dataset(X_test, y_test, random_seed=random_seed)
        print(f"\nBalanced test set: {original_test_size} -> {len(y_test)} samples")
        print(f"  Test set class distribution: {Counter(y_test)}")

    # Apply augmentation ONLY to training set
    if augment_train:
        print(f"\nApplying {aug_factor}x augmentation to TRAINING SET ONLY...")
        print(f"  Time warping: {time_warp_sigma}")
        print(f"  Gaussian noise: {noise_sigma}")
        print(f"  Magnitude jitter: {magnitude_jitter}")
        if include_mirror:
            print(f"  Mirror (time-reversal): enabled")

        X_train_orig_size = X_train.shape[0]
        X_train, y_train = apply_augmentation(
            X_train, y_train,
            aug_factor=aug_factor,
            time_warp_sigma=time_warp_sigma,
            noise_sigma=noise_sigma,
            magnitude_jitter=magnitude_jitter,
            include_mirror=include_mirror,
            random_seed=random_seed
        )
        print(f"  Training set: {X_train_orig_size} → {X_train.shape[0]} samples")
        print(f"  Val/Test sets: UNCHANGED (no data leakage)")

    # Create tensor datasets
    train_ds = _make_tensor_dataset(X_train, y_train)
    val_ds   = _make_tensor_dataset(X_val,   y_val)
    test_ds  = _make_tensor_dataset(X_test,  y_test)

    # Handle DataLoader kwargs
    if dl_kwargs.get("persistent_workers", False) and (dl_kwargs.get("num_workers", 0) == 0):
        dl_kwargs = dict(dl_kwargs)
        dl_kwargs["persistent_workers"] = False
    if dl_kwargs.get("num_workers", 0) == 0 and "prefetch_factor" in dl_kwargs:
        dl_kwargs = dict(dl_kwargs)
        dl_kwargs.pop("prefetch_factor", None)

    # Multiprocessing context
    extra = {}
    if dl_kwargs.get("num_workers", 0) > 0:
        try:
            import multiprocessing as mp
            extra["multiprocessing_context"] = mp.get_context("fork")
        except Exception:
            pass

    # Create train loader with optional weighted sampling
    if balance_train:
        class_weights = 1.0 / (np.bincount(y_train) + 1e-6)
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **dl_kwargs, **extra)

        # Show distribution
        try:
            idxs = list(sampler)
            sampled_labels = y_train[np.asarray(idxs)]
            print("\nTraining set class distribution:")
            print(f"  Original: {Counter(y_train)}")
            print(f"  Sampled:  {Counter(sampled_labels.tolist())}")
        except Exception as e:
            print(f"Sampler preview skipped: {e}")
    else:
        print(f"\nTraining set class distribution: {Counter(y_train)}")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs, **extra)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, **dl_kwargs, **extra)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dl_kwargs, **extra)

    return train_loader, val_loader, test_loader, y_train


# --- CM helper ---
def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    normalize="true",
    save_path=None,
    show=False,
    title="Confusion matrix",
    figsize=(7, 7),
    fontsize_scale=1.6,
    axes_fontscale=1.6,
):
    """
    Plot confusion matrix with optional data export.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        save_path: Path to save. If ends with .png, saves there directly.
                   Otherwise treated as directory with plot.png + data.csv.
        show: Whether to display interactively
        title: Plot title (sentence case recommended)
        figsize: Figure size tuple
        fontsize_scale: Scale factor for cell text
        axes_fontscale: Scale factor for axes labels
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # stable label set
    if class_names:
        labels = list(range(len(class_names)))
        disp_labels = class_names
    else:
        if len(y_true) or len(y_pred):
            labels = sorted(set(np.unique(y_true)).union(set(np.unique(y_pred))))
        else:
            labels = [0, 1]
        disp_labels = labels

    cm = confusion_matrix(y_true, y_pred, normalize=normalize, labels=labels)

    base_fs = 10
    fs_cells = base_fs * float(fontsize_scale)   # inside boxes
    fs_axes  = base_fs * float(axes_fontscale)   # axes + colorbar

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
    disp.plot(
        cmap=_DAVOS_R,
        ax=ax,
        colorbar=False,
        values_format=".2f",
        im_kw={"vmin": 0.0, "vmax": 1.0},
    )

    # colorbar sized to match the matrix height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    cb = fig.colorbar(disp.im_, cax=cax)
    cb.ax.tick_params(labelsize=fs_axes, length=4, width=0.8)
    cb.outline.set_linewidth(0.4)

    # axes fonts
    ax.set_title(title, fontsize=fs_axes)
    ax.set_xlabel(ax.get_xlabel(), fontsize=fs_axes)
    ax.set_ylabel(ax.get_ylabel(), fontsize=fs_axes)
    ax.tick_params(axis="both", which='major', labelsize=fs_axes, length=4, width=0.8, direction='out')
    ax.tick_params(axis="both", which='minor', length=2, width=0.6, direction='out')
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    # cell annotation: font size + white/black based on value vs 0.5 midpoint
    texts = getattr(disp, "text_", None)
    if texts is not None:
        try:
            it = texts.ravel()
        except Exception:
            it = texts if isinstance(texts, (list, tuple)) else [texts]
        for t in it:
            t.set_fontsize(fs_cells)
            val = float(t.get_text().replace('±', ''))
            t.set_color('white' if val > 0.5 else 'black')

    ax.grid(False)
    plt.tight_layout()

    if save_path:
        # Determine output paths
        if save_path.endswith('.png'):
            # Legacy mode: save directly to the specified path (as PDF)
            plot_path = save_path.replace('.png', '.pdf')
            data_path = save_path.replace('.png', '_data.csv')
        else:
            # Folder mode: save plot.pdf and data.csv in directory
            os.makedirs(save_path, exist_ok=True)
            plot_path = os.path.join(save_path, "plot.pdf")
            data_path = os.path.join(save_path, "data.csv")

        plt.savefig(plot_path, dpi=450, bbox_inches='tight')
        plt.close()

        # Save confusion matrix data
        cm_df = pd.DataFrame(cm, index=disp_labels, columns=disp_labels)
        cm_df.index.name = 'true_label'
        cm_df.to_csv(data_path)
    elif show:
        plt.show()
        plt.close()


def plot_confusion_matrix_with_std(
    mean_cm: np.ndarray,
    std_cm: np.ndarray,
    class_names: list,
    save_path: str = None,
    show: bool = False,
    title: str = "Confusion matrix",
    figsize: tuple = (7, 7),
    fontsize_scale: float = 1.6,
    axes_fontscale: float = 1.6,
):
    """
    Plot a confusion matrix showing mean ± std in each cell.

    Args:
        mean_cm: Mean confusion matrix (C x C), normalized values in [0, 1].
        std_cm: Standard deviation confusion matrix (C x C).
        class_names: List of class names.
        save_path: Path to save the plot (.png).
        show: Whether to display interactively.
        title: Plot title.
        figsize: Figure size tuple.
        fontsize_scale: Scale factor for cell text.
        axes_fontscale: Scale factor for axis labels.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    base_fs = 10
    fs_cells = base_fs * float(fontsize_scale)
    fs_axes = base_fs * float(axes_fontscale)

    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the heatmap using mean values for color
    im = ax.imshow(mean_cm, cmap=_DAVOS_R, vmin=0.0, vmax=1.0)

    # Add text annotations with mean ± std; white text above 0.5 midpoint
    for i in range(n_classes):
        for j in range(n_classes):
            mean_val = mean_cm[i, j]
            std_val = std_cm[i, j]
            text_color = "white" if mean_val > 0.5 else "black"
            ax.text(j, i, f"{mean_val:.3f}\n±{std_val:.3f}",
                    ha="center", va="center",
                    color=text_color, fontsize=fs_cells, linespacing=1.3)

    # Set ticks and labels
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Colorbar sized to match the matrix height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=fs_axes, length=4, width=0.8)
    cb.outline.set_linewidth(0.4)

    # Axes labels and title
    ax.set_title(title, fontsize=fs_axes)
    ax.set_xlabel("Predicted label", fontsize=fs_axes)
    ax.set_ylabel("True label", fontsize=fs_axes)
    ax.tick_params(axis="both", which='major', labelsize=fs_axes, length=4, width=0.8, direction='out')
    ax.tick_params(axis="both", which='minor', length=2, width=0.6, direction='out')
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    ax.grid(False)
    plt.tight_layout()

    if save_path:
        pdf_path = str(save_path).replace('.png', '.pdf') if str(save_path).endswith('.png') else str(save_path)
        plt.savefig(pdf_path, dpi=450, bbox_inches='tight')
        plt.close()
    elif show:
        plt.show()
        plt.close()


# --- Threshold Optimization ---
def optimize_binary_threshold(y_true, y_probs, metric='mcc'):
    """
    Find optimal threshold for binary classification.

    Args:
        y_true: True labels (0 or 1)
        y_probs: Predicted probabilities for class 1
        metric: Metric to optimize ('mcc', 'f1', 'youden', 'balanced_accuracy', 'min_recall')

    Returns:
        tuple: (optimal_threshold, metric_value)
    """
    thresholds = np.linspace(0, 1, 101)
    scores = []

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)

        if metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'youden':
            # Youden's J statistic: Sensitivity + Specificity - 1
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.sum() == 0:
                score = 0
            else:
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == 'min_recall':
            # Maximize minimum recall across both classes
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.sum() == 0:
                score = 0
            else:
                tn, fp, fn, tp = cm.ravel()
                recall_class0 = tn / (tn + fp) if (tn + fp) > 0 else 0
                recall_class1 = tp / (tp + fn) if (tp + fn) > 0 else 0
                score = min(recall_class0, recall_class1)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

    optimal_idx = np.argmax(scores)
    optimal_thresh = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]

    return optimal_thresh, optimal_score


def optimize_multiclass_thresholds(y_true, y_probs, metric='macro_f1', mode='global'):
    """
    Find optimal thresholds for multiclass classification.

    Args:
        y_true: True labels (shape: N)
        y_probs: Predicted probabilities (shape: N x C)
        metric: Metric to optimize ('macro_f1', 'balanced_accuracy')
        mode: 'global' for single confidence threshold, 'per_class' for per-class thresholds

    Returns:
        If mode='global': float (single threshold)
        If mode='per_class': numpy array of shape (C,) with per-class thresholds
    """
    n_classes = y_probs.shape[1]

    if mode == 'global':
        # Find single confidence threshold
        # Require max probability to exceed threshold
        thresholds = np.linspace(0, 1, 51)
        scores = []

        for thresh in thresholds:
            max_probs = y_probs.max(axis=1)
            valid_mask = max_probs >= thresh

            if valid_mask.sum() == 0:
                scores.append(0)
                continue

            y_pred = y_probs[valid_mask].argmax(axis=1)
            y_true_valid = y_true[valid_mask]

            if metric == 'macro_f1':
                score = f1_score(y_true_valid, y_pred, average='macro', labels=list(range(n_classes)), zero_division=0)
            elif metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_true_valid, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            scores.append(score)

        optimal_idx = np.argmax(scores)
        optimal_thresh = thresholds[optimal_idx]

        return optimal_thresh

    elif mode == 'per_class':
        # Find per-class thresholds using one-vs-rest approach
        thresholds = np.zeros(n_classes)

        for c in range(n_classes):
            # Treat as binary problem: class c vs rest
            y_binary = (y_true == c).astype(int)
            y_probs_c = y_probs[:, c]

            # Optimize MCC for this class
            thresh, _ = optimize_binary_threshold(y_binary, y_probs_c, metric='mcc')
            thresholds[c] = thresh

        return thresholds

    else:
        raise ValueError(f"Unknown mode: {mode}")


def apply_threshold(y_probs, threshold, num_classes):
    """
    Apply threshold to probability predictions.

    Args:
        y_probs: Predicted probabilities (shape: N x C for multiclass, N for binary)
        threshold: Threshold value (float for binary/global, array for per-class)
        num_classes: Number of classes

    Returns:
        y_pred: Predicted labels
    """
    if num_classes == 2:
        # Binary classification
        if y_probs.ndim == 2:
            y_probs = y_probs[:, 1]  # Use probability of class 1
        return (y_probs >= threshold).astype(int)
    else:
        # Multiclass
        if isinstance(threshold, (int, float)):
            # Global confidence threshold
            max_probs = y_probs.max(axis=1)
            y_pred = y_probs.argmax(axis=1)
            # Set to -1 (uncertain) if below threshold
            y_pred[max_probs < threshold] = -1
            return y_pred
        else:
            # Per-class thresholds
            y_pred = y_probs.argmax(axis=1)
            for i, pred_class in enumerate(y_pred):
                if y_probs[i, pred_class] < threshold[pred_class]:
                    y_pred[i] = -1  # uncertain
            return y_pred


# --- Training loop (val AUC early stop) ---
def train_model(
    model, train_loader, val_loader, criterion, optimizer,
    device, num_classes, max_epochs, patience_limit,
    model_save_path, checkpoint_path,
    autocast_ctx=None, scaler=None, clip_grad_norm=1.0, verbose=False,
    threshold_metric="balanced_accuracy",
    auc_min_delta=0.005, loss_mode="relative", loss_tolerance=1.10,
    auc_tolerance=None, loss_min_delta=None
):
    """
    Train model with combined AUC + loss early stopping.

    Early stopping logic:
    - Primary condition: Checkpoint when AUC improves by at least auc_min_delta
      AND loss satisfies the loss constraint (based on loss_mode)
    - Alternative condition (if auc_tolerance and loss_min_delta are set):
      Checkpoint when AUC is within auc_tolerance of best AND loss improves
      by at least loss_min_delta. This allows saving checkpoints with slightly
      lower AUC but significantly better loss, which can yield higher accuracy.

    Args:
        auc_min_delta: Minimum AUC improvement required for primary checkpoint
        loss_mode: "relative", "absolute", or "none" for loss constraint
        loss_tolerance: Loss constraint threshold (multiplier for relative, additive for absolute)
        auc_tolerance: (Optional) Max AUC drop from best to allow alternative checkpoint
        loss_min_delta: (Optional) Min loss improvement required for alternative checkpoint
    """
    autocast_ctx = autocast_ctx or (lambda: nullcontext())

    best_val_auc = -float("inf")
    best_val_loss = float("inf")
    best_checkpoint_loss = float("inf")  # Loss at last checkpoint (for alternative condition)
    best_epoch = -1
    patience = 0
    train_losses, val_losses, val_aucs = [], [], []

    # Check if alternative checkpoint condition is enabled
    alt_checkpoint_enabled = (auc_tolerance is not None) and (loss_min_delta is not None)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = train_acc = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                logits = model(Xb)
                loss = criterion(logits, yb)

            # Skip non-finite batches to avoid poisoning weights
            if not torch.isfinite(loss):
                print("[warn] non-finite loss encountered; skipping batch")
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                # Only unscale if the scaler has this method (not available on _NoopScaler for MPS)
                if hasattr(scaler, 'unscale_'):
                    scaler.unscale_(optimizer)
                if clip_grad_norm is not None and clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_grad_norm is not None and clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

            train_loss += loss.item() * Xb.size(0)
            train_acc  += accuracy(logits.softmax(dim=1), yb, task="multiclass", num_classes=num_classes).item() * Xb.size(0)

        model.eval()
        val_loss = val_acc = 0.0
        all_val_labels, all_val_probs = [], []

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                with autocast_ctx():
                    logits = model(Xb)
                    loss = criterion(logits, yb)

                if torch.isnan(loss):
                    continue

                val_loss += loss.item() * Xb.size(0)
                val_acc  += accuracy(logits.softmax(dim=1), yb, task="multiclass", num_classes=num_classes).item() * Xb.size(0)

                probs = F.softmax(logits, dim=1).to(torch.float32).detach().cpu().numpy()
                all_val_probs.append(probs)
                all_val_labels.append(yb.cpu().numpy())

        Ntr = len(train_loader.dataset); Nva = len(val_loader.dataset)
        train_loss /= max(1, Ntr); val_loss /= max(1, Nva)
        train_acc  /= max(1, Ntr); val_acc  /= max(1, Nva)

        all_val_labels = np.concatenate(all_val_labels) if len(all_val_labels) else np.array([])
        all_val_probs  = np.concatenate(all_val_probs)  if len(all_val_probs)  else np.array([])

        try:
            if num_classes == 2:
                val_auc = roc_auc_score(all_val_labels, all_val_probs[:, 1])
            else:
                y_true_bin = label_binarize(all_val_labels, classes=np.arange(num_classes))
                val_auc = roc_auc_score(y_true_bin, all_val_probs, average="macro")
        except Exception as e:
            print(f"[AUC Warning] Epoch {epoch}: {e}")
            val_auc = np.nan

        if num_classes == 2 and all_val_probs.size:
            probs_pos = all_val_probs[:, 1]
            acc_argmax = accuracy_score(all_val_labels, np.argmax(all_val_probs, axis=1))
            acc_thr05  = accuracy_score(all_val_labels, (probs_pos >= 0.5).astype(int))
            ts = np.linspace(0, 1, 501)
            accs = [accuracy_score(all_val_labels, (probs_pos >= t).astype(int)) for t in ts]
            t_best = float(ts[int(np.argmax(accs))]); acc_best = float(np.max(accs))
            cm_best = confusion_matrix(all_val_labels, (probs_pos >= t_best).astype(int), labels=[0, 1])
            if verbose:
                print(f"[SANITY] argmax_acc={acc_argmax:.3f} | thr05_acc={acc_thr05:.3f} | best_t={t_best:.3f} | best_acc={acc_best:.3f}")
                print(f"[SANITY] cm@best_t:\n{cm_best}")

        train_losses.append(train_loss); val_losses.append(val_loss); val_aucs.append(val_auc)
        print(f"Epoch {epoch:03d} — train loss: {train_loss:.4f}, acc: {train_acc:.3f} | val loss: {val_loss:.4f}, acc: {val_acc:.3f}, auc: {val_auc:.3f}")

        # Track best loss independently (for loss constraint check)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Combined early stopping: AUC improvement + loss constraint
        auc_improved = np.isfinite(val_auc) and (val_auc >= best_val_auc + auc_min_delta)

        # Check loss constraint based on mode
        if loss_mode == "relative":
            loss_ok = val_loss <= best_val_loss * loss_tolerance
        elif loss_mode == "absolute":
            loss_ok = val_loss <= best_val_loss + loss_tolerance
        else:  # "none" - no loss constraint
            loss_ok = True

        # Primary checkpoint condition: AUC improved significantly AND loss constraint satisfied
        primary_checkpoint = auc_improved and loss_ok

        # Alternative checkpoint condition: AUC within tolerance AND loss improved significantly
        # This allows saving checkpoints with slightly lower AUC but better loss (often better accuracy)
        alt_checkpoint = False
        if alt_checkpoint_enabled and np.isfinite(val_auc) and best_val_auc > -float("inf"):
            auc_within_tolerance = val_auc >= best_val_auc - auc_tolerance
            loss_improved_significantly = val_loss <= best_checkpoint_loss - loss_min_delta
            alt_checkpoint = auc_within_tolerance and loss_improved_significantly

        if primary_checkpoint or alt_checkpoint:
            checkpoint_reason = "AUC improved" if primary_checkpoint else "loss improved (AUC within tolerance)"
            # Update best AUC only if it actually improved (not for alternative checkpoints)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
            best_checkpoint_loss = val_loss
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), model_save_path)
            if verbose:
                print(f"  -> New checkpoint ({checkpoint_reason}): AUC={val_auc:.4f}, loss={val_loss:.4f}")
        else:
            patience += 1
            if verbose and auc_improved and not loss_ok:
                print(f"  -> AUC improved but loss too high (loss={val_loss:.4f}, best={best_val_loss:.4f}, threshold={best_val_loss * loss_tolerance if loss_mode == 'relative' else best_val_loss + loss_tolerance:.4f})")
            if patience >= patience_limit:
                print(f"Early stopping at epoch {epoch} (best AUC {best_val_auc:.3f} at epoch {best_epoch})")
                break

    # Compute optimal threshold on validation set after training
    model.eval()
    all_val_labels, all_val_probs = [], []

    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            with autocast_ctx():
                logits = model(Xb)
            probs = F.softmax(logits, dim=1).to(torch.float32).detach().cpu().numpy()
            all_val_probs.append(probs)
            all_val_labels.append(yb.cpu().numpy())

    all_val_labels = np.concatenate(all_val_labels) if len(all_val_labels) else np.array([])
    all_val_probs  = np.concatenate(all_val_probs)  if len(all_val_probs)  else np.array([])

    # Optimize threshold on validation set (skip if argmax is selected)
    optimal_threshold = None
    if threshold_metric.lower() == "argmax":
        print("Using argmax for predictions (no threshold optimization)")
    elif num_classes == 2 and all_val_probs.size:
        # Binary: optimize using configured metric
        optimal_threshold, metric_score = optimize_binary_threshold(
            all_val_labels, all_val_probs[:, 1], metric=threshold_metric
        )
        print(f"Optimal threshold found on validation set: {optimal_threshold:.3f} ({threshold_metric.upper()}: {metric_score:.3f})")
    elif num_classes > 2 and all_val_probs.size:
        # Multiclass: use global confidence threshold optimizing macro F1
        optimal_threshold = optimize_multiclass_thresholds(
            all_val_labels, all_val_probs, metric='macro_f1', mode='global'
        )
        print(f"Optimal global confidence threshold found on validation set: {optimal_threshold:.3f}")

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "optimal_threshold": optimal_threshold,
    }, checkpoint_path)

    return train_losses, val_losses, val_aucs, optimal_threshold



# --- Saving results ---

def plot_losses(train_losses, val_losses, save_path="loss_plot.pdf"):
    """
    Plot training and validation losses.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot. If ends with .png/.pdf, saves there directly.
                   Otherwise treated as directory and saves plot.pdf + data.csv inside.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(train_losses) + 1)
    # Scatter plot with thin connecting lines
    ax.plot(epochs, train_losses, color=COLORS['orange'], linewidth=0.8, zorder=1)
    ax.scatter(epochs, train_losses, color=COLORS['orange'], s=20, label="Train loss", zorder=2)
    ax.plot(epochs, val_losses, color=COLORS['sky_blue'], linewidth=0.8, zorder=1)
    ax.scatter(epochs, val_losses, color=COLORS['sky_blue'], s=20, label="Val loss", zorder=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    ax.set_title("Training and validation loss")
    apply_axis_standards(ax)
    plt.tight_layout()

    # Determine output paths
    if save_path.endswith('.png') or save_path.endswith('.pdf'):
        # Legacy mode: save directly to the specified path (as PDF)
        plot_path = str(save_path).replace('.png', '.pdf')
        data_path = str(save_path).replace('.png', '_data.csv').replace('.pdf', '_data.csv')
    else:
        # Folder mode: save plot.pdf and data.csv in directory
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, "plot.pdf")
        data_path = os.path.join(save_path, "data.csv")

    plt.savefig(plot_path, dpi=450, bbox_inches='tight')
    plt.close()

    # Save data
    df = pd.DataFrame({
        'epoch': list(epochs),
        'train_loss': train_losses,
        'val_loss': val_losses,
    })
    df.to_csv(data_path, index=False)


# --- Evaluation ---
def evaluate_model(model, test_loader, device, label_names=None, save_path=None,
                   autocast_ctx=None, optimal_threshold=None):
    """
    Evaluate model on test set with optional threshold-based predictions.

    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: Device for inference
        label_names: List of class names
        save_path: Path to save confusion matrix
        autocast_ctx: Autocast context for mixed precision
        optimal_threshold: Optimal threshold from validation set (None uses 0.5/argmax)

    Returns:
        dict: Evaluation metrics including accuracy, AUC, MCC (binary), etc.
    """
    autocast_ctx = autocast_ctx or (lambda: nullcontext())
    model.eval()
    y_true, y_probs = [], []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with autocast_ctx():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            probs = F.softmax(outputs, dim=1).to(torch.float32).detach().cpu().numpy()

            y_true.extend(labels.cpu().numpy())
            y_probs.extend(probs)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    test_loss = test_loss / len(test_loader.dataset) if len(test_loader.dataset) else float("nan")

    num_classes = y_probs.shape[1] if y_probs.size else 2

    # Generate predictions with argmax (baseline)
    y_pred_argmax = np.argmax(y_probs, axis=1)

    # Generate predictions with optimal threshold (if provided)
    if optimal_threshold is not None:
        y_pred = apply_threshold(y_probs, optimal_threshold, num_classes)
    else:
        y_pred = y_pred_argmax

    # always pass full label set to avoid "y_pred contains classes not in y_true"
    if label_names:
        labels_full = list(range(len(label_names)))
        target_names = label_names
    elif y_probs.size:
        labels_full = list(range(y_probs.shape[1]))
        target_names = None
    else:
        labels_full = sorted(np.unique(np.concatenate([y_true, y_pred]))) if len(y_true) else []
        target_names = None

    # Compute metrics
    acc_argmax = accuracy_score(y_true, y_pred_argmax) if len(y_true) else float("nan")
    acc_optimal = accuracy_score(y_true, y_pred) if len(y_true) and optimal_threshold is not None else float("nan")
    balanced_acc = balanced_accuracy_score(y_true, y_pred) if len(y_true) else float("nan")

    # Compute MCC for binary classification
    mcc = None
    if num_classes == 2 and len(y_true):
        mcc = matthews_corrcoef(y_true, y_pred)

    # Classification report
    report = (
        classification_report(
            y_true, y_pred,
            labels=labels_full,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        if len(y_true) else {}
    )

    acc_per_class = {}
    if report:
        if target_names:
            for name in target_names:
                if name in report:
                    acc_per_class[name] = report[name]["precision"]
        else:
            for lbl in labels_full:
                key = str(lbl)
                if key in report:
                    acc_per_class[key] = report[key]["precision"]

    # AUC metrics
    try:
        if len(np.unique(y_true)) > 1:
            auc_roc = roc_auc_score(y_true, y_probs, multi_class="ovr")
        else:
            auc_roc = np.nan
    except ValueError:
        auc_roc = np.nan

    # AUC-PR for binary classification
    auc_pr = None
    if num_classes == 2 and len(y_true):
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
            auc_pr = sklearn_auc(recall, precision)
        except ValueError:
            auc_pr = np.nan

    # Print summary
    print("\n=== Test Set Evaluation ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy (argmax): {acc_argmax:.4f}")
    if optimal_threshold is not None:
        print(f"Accuracy (optimal threshold={optimal_threshold:.3f}): {acc_optimal:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    if auc_pr is not None:
        print(f"AUC-PR: {auc_pr:.4f}")
    if mcc is not None:
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    # Save confusion matrices
    if save_path is not None and len(y_true):
        # Determine if save_path is a folder or file
        if save_path.endswith('.png'):
            # Legacy file-based mode
            main_path = save_path
            argmax_path = save_path.replace(".png", "_argmax.png")
        else:
            # Folder-based mode - save to confusion_matrix/ folder
            os.makedirs(save_path, exist_ok=True)
            main_path = save_path  # plot_confusion_matrix handles folder mode
            argmax_path = os.path.join(save_path, "argmax")  # separate subfolder

        # Confusion matrix with optimal threshold
        plot_confusion_matrix(
            y_true, y_pred,
            class_names=label_names,
            normalize="true",
            save_path=main_path,
            show=False,
            title=f"Confusion matrix (threshold={optimal_threshold:.3f})" if optimal_threshold else "Confusion matrix",
            figsize=(5, 5),
        )

        # Also save argmax confusion matrix for comparison
        if optimal_threshold is not None:
            plot_confusion_matrix(
                y_true, y_pred_argmax,
                class_names=label_names,
                normalize="true",
                save_path=argmax_path,
                show=False,
                title="Confusion matrix (argmax)",
                figsize=(5, 5),
            )

    return {
        "balanced_accuracy": balanced_acc,
        "accuracy_argmax": acc_argmax,
        "accuracy_optimal": acc_optimal,
        "accuracy_per_class": acc_per_class,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "mcc": mcc,
        "test_loss": test_loss,
        "optimal_threshold": optimal_threshold,
    }


def estimate_batch_size(device_id=0, model_name=None):
    """Conservative batch estimator. Much smaller for GRU-based models."""
    cap = torch.cuda.get_device_properties(device_id).total_memory // (1024**2)  # MB

    if model_name and ("gru" in model_name.lower()):
        if cap > 40000:
            return 256
        elif cap > 16000:
            return 64
        else:
            return 32
    else:
        if cap > 40000:
            return 1024
        elif cap > 16000:
            return 512
        else:
            return 128


# --- MC Dropout ---
@torch.no_grad()
def mc_dropout_predict(model, X_test, n_mc=10, batch_size=1280, device=None, autocast_ctx=None):
    if device is None:
        device = next(model.parameters()).device
    autocast_ctx = autocast_ctx or (lambda: nullcontext())

    model.eval()

    # Freeze BN; activate dropout; force LockedDropout stochastic
    bn_states, do_states, ld_states = [], [], []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_states.append((m, m.training)); m.eval()
        elif isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            do_states.append((m, m.training)); m.train()
        elif isinstance(m, LockedDropout):
            ld_states.append((m, m.mc_eval)); m.mc_eval = True

    N = len(X_test)
    current_bs = int(batch_size)
    min_bs = 8
    max_oom_retries = 8  # per step

    probs_mc = []
    start = datetime.datetime.now()
    for _ in range(n_mc):
        mc_run = []
        i = 0
        while i < N:
            try:
                xb = X_test[i:i + current_bs].to(device, non_blocking=True)
                with autocast_ctx():
                    logits = model(xb)
                    p = torch.softmax(logits, dim=1).to(torch.float32).cpu().numpy()
                mc_run.append(p)
                i += current_bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_bs = max(min_bs, current_bs // 2)
                if new_bs == current_bs:
                    raise RuntimeError(f"[mc_dropout] OOM persists at batch_size={current_bs} (min reached).")
                current_bs = new_bs
                print(f"[mc_dropout] OOM at index {i}. Reducing batch size to {current_bs}.")
                max_oom_retries -= 1
                if max_oom_retries < 0 and current_bs == min_bs:
                    raise RuntimeError("[mc_dropout] OOM despite minimum batch size.")
                continue
        probs_mc.append(np.concatenate(mc_run, axis=0))
    end = datetime.datetime.now()

    # Restore original states
    for m, st in do_states: m.train(st)
    for m, st in bn_states: m.train(st)
    for m, st in ld_states: m.mc_eval = st

    duration = (end - start).total_seconds() / max(1, N)
    print(f"MC time per sample: {duration:.3f} seconds")
    return np.stack(probs_mc, axis=0)


# --- Uncertainty filtering + plots ---
def evaluate_uncertainty_filtered(
    probs_mc,
    y_true,
    threshold=None,
    class_names=None,
    show_plots=True,
    save_dir="results",
    trace_loss=50.0,
    verbose=False,
    optimal_threshold=None,
    embeddings=None,
    umap_coords=None,
    traces=None,
    unique_ids=None,
    random_state=42,
):
    """
    Evaluate Monte Carlo dropout predictions with Wasserstein uncertainty filtering.

    Args:
        probs_mc: MC dropout probabilities (n_mc, N, C)
        y_true: True labels
        threshold: Wasserstein distance threshold for filtering. If None, auto-selects
            threshold that maximizes accuracy while keeping trace loss <= trace_loss.
            Set to a specific value (e.g., 0.1) to use a fixed threshold.
        class_names: List of class names
        show_plots: Whether to show plots
        save_dir: Directory to save results
        trace_loss: Maximum acceptable trace loss percentage when auto-selecting threshold.
            Only used when threshold=None.
        verbose: Verbose output
        optimal_threshold: Optimal prediction threshold from validation set
        embeddings: Optional embeddings array for UMAP (only used if umap_coords not provided)
        umap_coords: Pre-computed UMAP coordinates (N, 2) for consistent visualization
        traces: Optional traces array (N, C, T) for saving with Wasserstein distances
        random_state: Random state for UMAP (only used if umap_coords not provided)

    Returns:
        dict: Evaluation metrics with uncertainty filtering
    """
    mean_probs = probs_mc.mean(axis=0)            # (N, C)
    N, C = mean_probs.shape

    # Generate predictions using optimal threshold if provided
    if optimal_threshold is not None:
        pred_labels = apply_threshold(mean_probs, optimal_threshold, C)
        # For uncertain predictions (-1), fall back to argmax
        uncertain_mask = pred_labels == -1
        if uncertain_mask.any():
            pred_labels[uncertain_mask] = mean_probs[uncertain_mask].argmax(axis=1)
    else:
        pred_labels = mean_probs.argmax(axis=1)       # (N,)

    # Wasserstein uncertainty per sample
    w_dists = []
    for i, pred in enumerate(pred_labels):
        other = [wasserstein_distance(probs_mc[:, i, pred], probs_mc[:, i, c]) for c in range(C) if c != pred]
        w_dists.append(min(other) if other else 0.0)
    w_dists = np.array(w_dists)

    # Save traces with Wasserstein distances
    if traces is not None:
        os.makedirs(save_dir, exist_ok=True)
        traces_with_uncertainty_path = os.path.join(save_dir, "traces_with_wasserstein.npz")
        save_kwargs = dict(
            traces=traces.numpy() if hasattr(traces, 'numpy') else traces,
            labels=y_true,
            predictions=pred_labels,
            wasserstein_distances=w_dists,
            class_names=class_names if class_names else [],
        )
        if unique_ids is not None:
            save_kwargs['unique_ids'] = unique_ids
        np.savez(traces_with_uncertainty_path, **save_kwargs)
        print(f"Saved traces with Wasserstein distances to: {traces_with_uncertainty_path}")

        # Create quartile visualization plot: 4x4 grid (rows=quartiles, cols=traces)
        rng = np.random.default_rng(seed=random_state)
        quartiles = np.percentile(w_dists, [25, 50, 75])
        quartile_labels = ['Q1 (least certain)', 'Q2', 'Q3', 'Q4 (most certain)']
        quartile_ranges = [
            (w_dists.min(), quartiles[0]),
            (quartiles[0], quartiles[1]),
            (quartiles[1], quartiles[2]),
            (quartiles[2], w_dists.max() + 1e-9)
        ]

        unique_classes = np.unique(y_true)
        n_examples_per_class = 2
        traces_np = traces.numpy() if hasattr(traces, 'numpy') else traces

        # 4 rows (quartiles) x 4 columns (2 traces per class)
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))

        for q_idx, (q_min, q_max) in enumerate(quartile_ranges):
            q_mask = (w_dists >= q_min) & (w_dists < q_max)
            col_idx = 0  # Track column position

            for cls_idx, cls in enumerate(unique_classes):
                cls_mask = (y_true == cls) & q_mask
                cls_indices = np.where(cls_mask)[0]
                cls_name = class_names[cls] if class_names and cls < len(class_names) else f"Class {cls}"
                color = COLORS[list(COLORS.keys())[cls_idx % len(COLORS)]]

                # Select random examples
                n_select = min(n_examples_per_class, len(cls_indices))
                if len(cls_indices) > 0:
                    selected_indices = rng.choice(cls_indices, size=n_select, replace=False)
                else:
                    selected_indices = []

                # Plot each trace in its own subplot
                for i in range(n_examples_per_class):
                    ax = axes[q_idx, col_idx]
                    if i < len(selected_indices):
                        idx = selected_indices[i]
                        trace = traces_np[idx]
                        # If multi-channel, use first channel for visualization
                        if trace.ndim == 2:
                            trace = trace[0]
                        ax.plot(trace, color=color, linewidth=1.2)
                        ax.set_title(f"{cls_name}, W={w_dists[idx]:.3f}", fontsize=FONTSIZE_LABEL)
                    else:
                        ax.set_title(f"{cls_name}, N/A", fontsize=FONTSIZE_LABEL)
                        ax.text(0.5, 0.5, "No samples", ha='center', va='center', transform=ax.transAxes)

                    # Only add y-label on leftmost column
                    if col_idx == 0:
                        ax.set_ylabel(f"{quartile_labels[q_idx]}\nIntensity", fontsize=FONTSIZE_LABEL)
                    # Only add x-label on bottom row
                    if q_idx == 3:
                        ax.set_xlabel("Time", fontsize=FONTSIZE_LABEL)

                    apply_axis_standards(ax)
                    col_idx += 1

        plt.suptitle("Example traces by Wasserstein distance quartile", fontsize=FONTSIZE_TITLE, y=1.02)
        plt.tight_layout(h_pad=3.0, w_pad=3.0)

        if show_plots:
            plt.show()
        else:
            # Save to folder with plot.png and data.csv
            quartile_plot_dir = os.path.join(save_dir, "traces_by_wasserstein_quartile")
            os.makedirs(quartile_plot_dir, exist_ok=True)
            plt.savefig(os.path.join(quartile_plot_dir, "plot.pdf"), dpi=450, bbox_inches='tight')
            plt.close()
            # Save quartile data
            quartile_data = {
                'quartile': [], 'quartile_label': [], 'w_min': [], 'w_max': [],
                'n_samples': []
            }
            for q_idx, (q_min, q_max) in enumerate(quartile_ranges):
                q_mask = (w_dists >= q_min) & (w_dists < q_max)
                quartile_data['quartile'].append(q_idx + 1)
                quartile_data['quartile_label'].append(quartile_labels[q_idx])
                quartile_data['w_min'].append(q_min)
                quartile_data['w_max'].append(q_max)
                quartile_data['n_samples'].append(int(q_mask.sum()))
            pd.DataFrame(quartile_data).to_csv(os.path.join(quartile_plot_dir, "data.csv"), index=False)
            print(f"Saved quartile trace plot to: {quartile_plot_dir}")

    # UMAP visualization colored by Wasserstein distance with class-specific markers
    if umap_coords is not None or embeddings is not None:
        try:
            # Use pre-computed UMAP coordinates if provided, otherwise compute
            if umap_coords is None:
                import umap
                print("Computing UMAP coordinates...")
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=random_state)
                umap_coords_plot = reducer.fit_transform(embeddings)
            else:
                print("Using pre-computed UMAP coordinates for Wasserstein visualization...")
                umap_coords_plot = umap_coords

            # Define markers for different classes
            markers = ['o', '^', 's', 'D', 'v', 'p', 'h', '*', 'X', 'P']  # circle, triangle, square, diamond, etc.
            unique_classes = np.unique(y_true)

            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot each class with different marker, colored by Wasserstein distance
            scatter_handles = []
            for idx, cls in enumerate(unique_classes):
                mask_cls = y_true == cls
                marker = markers[idx % len(markers)]
                sc = ax.scatter(
                    umap_coords_plot[mask_cls, 0], umap_coords_plot[mask_cls, 1],
                    c=w_dists[mask_cls], cmap=_LAPAZ, s=35, alpha=0.7,
                    marker=marker, vmin=w_dists.min(), vmax=w_dists.max(),
                    edgecolors='black', linewidths=0.5
                )
                # Create legend handle with class name
                cls_name = class_names[cls] if class_names and cls < len(class_names) else f"Class {cls}"
                handle = plt.Line2D([0], [0], marker=marker, color='#808080', linestyle='',
                                    markersize=8, label=cls_name)
                scatter_handles.append(handle)

            ax.set_xlabel('UMAP component 1', fontsize=FONTSIZE_LABEL)
            ax.set_ylabel('UMAP component 2', fontsize=FONTSIZE_LABEL)
            ax.set_title('UMAP projection colored by Wasserstein distance', fontsize=FONTSIZE_TITLE)
            apply_axis_standards(ax)

            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Wasserstein distance (certainty)')

            # Add legend for class markers at top
            ax.legend(handles=scatter_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                     framealpha=0.95, fontsize=FONTSIZE_LEGEND, title='Classes', ncol=len(unique_classes))

            plt.tight_layout()

            if show_plots:
                plt.show()
            else:
                # Save to folder with plot.png and data.csv
                umap_uncertainty_dir = os.path.join(save_dir, "umap_wasserstein")
                os.makedirs(umap_uncertainty_dir, exist_ok=True)
                plt.savefig(os.path.join(umap_uncertainty_dir, "plot.pdf"), dpi=450, bbox_inches='tight')
                plt.close()
                # Save UMAP data with Wasserstein distances
                umap_data = pd.DataFrame({
                    'umap_1': umap_coords_plot[:, 0],
                    'umap_2': umap_coords_plot[:, 1],
                    'wasserstein_distance': w_dists,
                    'true_label': y_true,
                    'predicted_label': pred_labels
                })
                umap_data.to_csv(os.path.join(umap_uncertainty_dir, "data.csv"), index=False)
                print(f"Saved UMAP uncertainty plot to: {umap_uncertainty_dir}")
        except Exception as e:
            print(f"Could not generate UMAP uncertainty plot: {e}")

    # Threshold sweep
    thresholds = np.linspace(0, w_dists.max() if w_dists.size else 1.0, 50)
    bal_accs, removals = [], []

    for thr in thresholds:
        m = w_dists > thr  # keep mask
        if m.sum() == 0:
            bal_accs.append(np.nan)
            removals.append(100.0)
        else:
            recs = recall_score(y_true[m], pred_labels[m], labels=list(range(C)), average=None, zero_division=0)
            bal_accs.append(float(np.mean(recs)) * 100)
            removals.append((1 - m.sum() / len(y_true)) * 100)

    # Auto-select threshold: maximize accuracy while keeping trace loss under limit
    retained_pct = 100 - np.array(removals)  # Convert removal % to retained %
    if threshold is None:
        valid = [
            (thr, acc, rem) for thr, acc, rem in zip(thresholds, bal_accs, removals)
            if rem <= trace_loss and not np.isnan(acc)
        ]
        if valid:
            threshold, best_acc, best_rem = max(valid, key=lambda x: x[1])
            print(f"Auto-selected threshold = {threshold:.3f}")
            print(f"  Accuracy = {best_acc:.1f}%, trace loss = {best_rem:.1f}% (max allowed: {trace_loss}%)")
        else:
            print("No valid threshold found under trace-loss constraint. Skipping filtered confusion matrix.")
            threshold = 0.0
    else:
        print(f"Using provided threshold = {threshold:.3f}")

    # Apply chosen threshold
    mask = w_dists > threshold
    if mask.sum() > 0:
        # Save to folder with plot.png and data.csv
        filtered_cm_dir = os.path.join(save_dir, "filtered_confusion_matrix")
        plot_confusion_matrix(
            y_true[mask], pred_labels[mask],
            class_names=class_names,
            normalize="true",
            save_path=filtered_cm_dir,
            show=False,
            title=f"Filtered confusion matrix (threshold={threshold:.2f})",
            figsize=(6, 6),
        )
    else:
        print("No samples exceed the uncertainty threshold.")

    # Sweep plot: X = retained %, Color = Wasserstein distance
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(retained_pct, bal_accs, c=thresholds, cmap=_LAPAZ)
    ax.set_xlabel("Retained traces (%)")
    ax.set_ylabel("Balanced accuracy of retained samples (%)")
    ax.set_title("Balanced accuracy vs. retained traces")
    ax.set_xlim(105, -5)  # Invert: 100% on left, decaying to right
    apply_axis_standards(ax)
    cbar = plt.colorbar(sc)
    cbar.set_label("Wasserstein distance (certainty)")
    plt.tight_layout()

    if show_plots:
        plt.show()
    else:
        # Save to folder with plot.png and data.csv
        sweep_dir = os.path.join(save_dir, "wasserstein_sweep")
        os.makedirs(sweep_dir, exist_ok=True)
        plt.savefig(os.path.join(sweep_dir, "plot.pdf"), dpi=450, bbox_inches='tight')
        plt.close()
        # Save sweep data
        sweep_data = pd.DataFrame({
            'threshold': thresholds,
            'retained_pct': retained_pct,
            'balanced_accuracy_pct': bal_accs,
        })
        sweep_data.to_csv(os.path.join(sweep_dir, "data.csv"), index=False)
        print(f"Saved Wasserstein sweep plot to: {sweep_dir}")

    # Summary
    init_acc = accuracy_score(y_true, pred_labels) * 100
    filtered_acc = accuracy_score(y_true[mask], pred_labels[mask]) * 100 if mask.sum() > 0 else np.nan
    removed_pct = (1 - mask.sum() / len(y_true)) * 100

    print(f"Initial accuracy: {init_acc:.2f}%")
    print(f"Filtered accuracy: {filtered_acc:.2f}% (removed {removed_pct:.1f}% of samples)")

    return {
        "initial_accuracy": init_acc,
        "filtered_accuracy": filtered_acc,
        "removed_percent": removed_pct,
        "balanced_accuracies": bal_accs,
        "thresholds": thresholds,
        "mask": mask,
        "selected_threshold": threshold,
    }




@torch.no_grad()
def extract_embeddings(model, data_loader, device=None, autocast_ctx=None):
    """
    Extract embeddings from a trained model.

    Args:
        model: Trained model with get_embedding() method
        data_loader: DataLoader providing input data
        device: Device to run inference on (defaults to model's device)
        autocast_ctx: Optional autocast context for mixed precision

    Returns:
        embeddings: numpy array of shape (N, embedding_dim)
        labels: numpy array of shape (N,) with true labels
        indices: list of sample indices (if available)
    """
    if device is None:
        device = next(model.parameters()).device
    autocast_ctx = autocast_ctx or (lambda: nullcontext())

    model.eval()
    embeddings_list = []
    labels_list = []

    for batch in data_loader:
        # Handle different batch formats
        if len(batch) == 2:
            inputs, labels = batch
        elif len(batch) == 3:
            inputs, labels, _ = batch  # Ignore indices for now
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        inputs = inputs.to(device)

        with autocast_ctx():
            emb = model.get_embedding(inputs)

        # Convert to float32 if needed (bfloat16 not supported by numpy)
        embeddings_list.append(emb.float().cpu().numpy())
        labels_list.append(labels.cpu().numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return embeddings, labels


def plot_umap_embeddings(embeddings, labels, class_names, save_path, max_per_class=10000, random_state=42, umap_coords=None):
    """
    Generate and plot UMAP projection of embeddings with balanced class sampling.

    Args:
        embeddings: numpy array of shape (N, embedding_dim)
        labels: numpy array of shape (N,) with class labels
        class_names: list of class names
        save_path: path to save the UMAP plot
        max_per_class: maximum number of samples per class (default: 10000)
        random_state: random seed for reproducibility
        umap_coords: optional pre-computed UMAP coordinates (N, 2). If provided,
                     uses these instead of computing UMAP, and skips subsampling.

    Returns:
        umap_data_path: path to saved CSV with UMAP coordinates
    """
    np.random.seed(random_state)

    # Get unique classes and their counts
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # If pre-computed UMAP coordinates provided, use all points
    if umap_coords is not None:
        print(f"Using pre-computed UMAP coordinates for {len(labels)} samples ({n_classes} classes)...")
        embeddings_subset = embeddings
        labels_subset = labels
        embeddings_2d = umap_coords
    else:
        # Import umap only when needed
        try:
            import umap
        except ImportError:
            print("WARNING: umap-learn not installed. Skipping UMAP visualization.")
            print("Install with: pip install umap-learn")
            return None

        # Sample balanced subset for visualization
        selected_indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            n_available = len(label_indices)

            # For the most populous class, use max_per_class
            # For other classes, scale proportionally
            if n_available > max_per_class:
                n_to_sample = max_per_class
            else:
                n_to_sample = n_available

            # Randomly sample
            if n_to_sample < n_available:
                sampled = np.random.choice(label_indices, size=n_to_sample, replace=False)
            else:
                sampled = label_indices

            selected_indices.extend(sampled)

        selected_indices = np.array(selected_indices)
        embeddings_subset = embeddings[selected_indices]
        labels_subset = labels[selected_indices]

        print(f"Running UMAP on {len(selected_indices)} samples ({n_classes} classes)...")

        # Fit UMAP
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=random_state,
            verbose=False
        )

        embeddings_2d = reducer.fit_transform(embeddings_subset)

    # Determine output paths
    if save_path.endswith('.png'):
        # Legacy mode: save as PDF regardless
        plot_path = save_path.replace('.png', '.pdf')
        data_path = save_path.replace('.png', '_data.csv')
    else:
        # Folder mode: save plot.pdf and data.csv in directory
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, "plot.pdf")
        data_path = os.path.join(save_path, "data.csv")

    # Save UMAP coordinates as CSV
    data_rows = []
    for i, (x, y) in enumerate(embeddings_2d):
        label_idx = labels_subset[i]
        class_name = class_names[label_idx] if label_idx < len(class_names) else f"Class_{label_idx}"
        data_rows.append([x, y, class_name, int(label_idx)])

    df_umap = pd.DataFrame(data_rows, columns=['UMAP1', 'UMAP2', 'class', 'label_idx'])
    df_umap.to_csv(data_path, index=False)
    print(f"  Saved UMAP data to: {os.path.basename(data_path)}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define color palette and markers (same as cluster.py)
    colors = ['dodgerblue', 'gold', 'lightcoral', 'firebrick', 'teal',
              'darkorange', 'orchid', 'forestgreen', 'palegreen', 'peru']
    markers = ['o', '^', 's', 'D', 'v', 'P', '*', 'X', 'p', 'h']

    # Extend colors and markers if we have more than 10 classes
    if n_classes > 10:
        colors = colors * ((n_classes // 10) + 1)
        markers = markers * ((n_classes // 10) + 1)

    for i, label in enumerate(unique_labels):
        mask = labels_subset == label
        class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
        n_samples = mask.sum()

        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[i],
            marker=markers[i],
            s=50,
            alpha=0.7,
            label=f'{class_name} (n={n_samples})',
            edgecolors='black',
            linewidths=0.3
        )

    ax.set_xlabel('UMAP component 1', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('UMAP component 2', fontsize=FONTSIZE_LABEL)
    ax.set_title('UMAP projection of trace embeddings', fontsize=FONTSIZE_TITLE)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.95, fontsize=FONTSIZE_LEGEND)
    # Apply axis standards (keeping original color scheme as requested)
    apply_axis_standards(ax)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=450, bbox_inches='tight')
    plt.close()

    print(f"  Saved UMAP plot to: {os.path.basename(plot_path)}")

    return data_path


# --- Data Augmentation ---
def apply_augmentation(X, y, aug_factor=2, time_warp_sigma=0.03, noise_sigma=0.02,
                       magnitude_jitter=0.02, include_mirror=False, random_seed=42):
    """
    Apply conservative augmentation to time-series data.

    Args:
        X: Input data of shape (N, C, T)
        y: Labels of shape (N,)
        aug_factor: Augmentation multiplier (2 = double dataset size, 3 = triple, etc.)
        time_warp_sigma: Std dev for time warping (default 0.03 = ±3%)
        noise_sigma: Std dev for Gaussian noise (default 0.02 = 2% of signal range)
        magnitude_jitter: Std dev for magnitude scaling (default 0.02 = ±2%)
        include_mirror: If True, append one time-reversed copy of every training trace.
            Applied after all warp/noise/jitter versions, before shuffling.
        random_seed: Random seed for reproducibility

    Returns:
        X_aug: Augmented data of shape (N * aug_factor [+ N if mirror], C, T)
        y_aug: Augmented labels of shape (N * aug_factor [+ N if mirror],)
    """
    from scipy.interpolate import interp1d

    rng = np.random.default_rng(seed=random_seed)
    N, C, T = X.shape

    print(f"  Original dataset: {N} samples")
    print(f"  Augmentation factor: {aug_factor}x")

    # Start with original data
    X_list = [X]
    y_list = [y]

    # Generate (aug_factor - 1) augmented versions
    for aug_idx in range(1, aug_factor):
        X_aug_version = np.zeros_like(X)

        for i in range(N):
            for c in range(C):
                trace = X[i, c, :]

                # 1. Time warping (temporal distortion)
                if time_warp_sigma > 0:
                    # Create slightly distorted time axis
                    warp_factor = 1.0 + rng.normal(0, time_warp_sigma)
                    warp_factor = np.clip(warp_factor, 0.95, 1.05)  # Limit to ±5%

                    original_indices = np.arange(T)
                    warped_indices = np.linspace(0, T - 1, int(T * warp_factor))

                    # Interpolate to warped time axis
                    if len(warped_indices) > 1:
                        interp_func = interp1d(original_indices, trace, kind='linear',
                                             bounds_error=False, fill_value='extrapolate')
                        warped_trace = interp_func(warped_indices)

                        # Resample back to original length
                        resample_indices = np.linspace(0, len(warped_trace) - 1, T)
                        resample_func = interp1d(np.arange(len(warped_trace)), warped_trace,
                                               kind='linear', bounds_error=False, fill_value='extrapolate')
                        trace = resample_func(resample_indices)
                    else:
                        trace = trace  # Skip if warping failed

                # 2. Gaussian noise (measurement variability)
                if noise_sigma > 0:
                    noise = rng.normal(0, noise_sigma * np.std(trace), T)
                    trace = trace + noise

                # 3. Magnitude jitter (intensity variation)
                if magnitude_jitter > 0:
                    jitter_factor = 1.0 + rng.normal(0, magnitude_jitter)
                    jitter_factor = np.clip(jitter_factor, 0.95, 1.05)  # Limit to ±5%
                    trace = trace * jitter_factor

                X_aug_version[i, c, :] = trace

        X_list.append(X_aug_version)
        y_list.append(y)

        print(f"    Generated augmentation version {aug_idx}/{aug_factor - 1}")

    # Mirror augmentation: one time-reversed copy of every original trace.
    # Added after all warp/noise/jitter versions so it is easy to reason about
    # the total dataset size (N * aug_factor + N if mirror).
    if include_mirror:
        X_mirrored = X[:, :, ::-1].copy()
        X_list.append(X_mirrored)
        y_list.append(y)
        print(f"    Added mirrored (time-reversed) version: {X_mirrored.shape[0]} traces")

    # Concatenate all versions
    X_augmented = np.concatenate(X_list, axis=0)
    y_augmented = np.concatenate(y_list, axis=0)

    print(f"  Augmented dataset: {X_augmented.shape[0]} samples")

    # Shuffle augmented dataset
    shuffle_idx = rng.permutation(len(y_augmented))
    X_augmented = X_augmented[shuffle_idx]
    y_augmented = y_augmented[shuffle_idx]

    return X_augmented, y_augmented


# --- Accelerator ---

def detect_accelerator():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)  # (major, minor)
        return {"type": "cuda", "device": dev, "name": name, "cap": cap}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"type": "mps", "device": torch.device("mps"), "name": "Apple MPS", "cap": None}
    return {"type": "cpu", "device": torch.device("cpu"), "name": "CPU", "cap": None}

class _NoopScaler:
    def scale(self, x): return x
    def step(self, opt): opt.step()
    def update(self): pass
    def __bool__(self): return False

def setup_precision_and_flags(accel):
    """
    Return (amp_dtype, autocast_ctx, scaler) with new torch.amp API. Safe on MPS/CPU.

    Args:
        accel: Accelerator dict from detect_accelerator()
    """
    atype = accel["type"]

    # defaults
    amp_dtype = None
    autocast_ctx = nullcontext
    scaler = _NoopScaler()

    if atype == "cuda":
        major, _ = accel["cap"]
        # TF32 + bf16 on Ampere+; fp16 on pre-Ampere
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            amp_dtype = torch.bfloat16
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            amp_dtype = torch.float16

        autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        torch.backends.cudnn.benchmark = True

    elif atype == "mps":
        # MPS fp16 disabled: deep models get NaN losses with fp16 on MPS
        amp_dtype = None
        autocast_ctx = nullcontext

    else:
        # CPU: fp32 only
        amp_dtype = None
        autocast_ctx = nullcontext
        scaler = _NoopScaler()

    return amp_dtype, autocast_ctx, scaler

def _slurm_cpus():
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        return int(v) if v else None
    except Exception:
        return None

def dataloader_kwargs_for(accel):
    """
    Clamp workers to allocated CPUs, keep prefetch low, and avoid pinning on non-CUDA.
    This prevents RAM spikes under Slurm.
    """
    alloc = _slurm_cpus()
    host = os.cpu_count() or 1
    # target cores we can actually use
    usable = alloc if alloc is not None else min(host, 8)

    # conservative workers for tuning
    if usable <= 2:
        nw = 0
    elif usable <= 4:
        nw = 2
    else:
        nw = min(4, usable - 2)

    pin = (accel["type"] == "cuda")
    base = dict(
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=False,  # safer for RAM during many short trials
    )
    if nw > 0:
        base["prefetch_factor"] = 1
    return base

def maybe_compile(model, accel, enabled=True):
    """Compile only when safe. Default: CUDA SM80+; never on MPS."""
    if not enabled:
        return model
    try:
        if hasattr(torch, "compile"):
            if accel["type"] == "cuda" and accel.get("cap", (0, 0))[0] >= 8:
                # A100/H100 etc.
                return torch.compile(model, mode="max-autotune")
            # MPS or older CUDA -> skip
    except Exception as e:
        print(f"torch.compile skipped: {e}")
    return model

def batch_size_hint(default_bs, accel):
    # Keep your estimate_batch_size() for CUDA. Give a safe floor elsewhere.
    if accel["type"] == "cuda":
        return None  # signal to use your estimate_batch_size()
    if accel["type"] == "mps":
        return max(128, default_bs // 2)  # Apple GPUs like larger batches
    return 256  # CPU fallback
