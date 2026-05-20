#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH), Andreas Biri (ETHZ).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Extract intensity traces from ND2 movies using Picasso localizations.

This script:
- Pairs protein and ground truth channel movies (or processes single-channel)
- Links/clusters localizations
- Extracts intensity time series
- Labels traces as IN/OUT based on ground truth colocalization (if applicable)
- Saves per-movie trace pickle files
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import warnings
import yaml
from pathlib import Path

import nd2
import numpy as np
from scipy import spatial

# Set matplotlib to non-interactive backend for thread safety in parallel processing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D

from utils import (
    setup_logging,
    get_num_workers,
    rel_under,
    discover_all_data,
    print_discovery_summary,
    get_and_link_locs,
    get_and_cluster_locs,
    get_overlapping_rois,
    get_traces_new_incl_ov,
    extract_background_traces,
    colorize,
    draw_max_projection,
)

# Suppress pandas deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Series.swapaxes.*")


def load_config(config_path: Path) -> dict:
    """Load and validate configuration file."""
    with config_path.open("r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    required = [
        "boxsize",
        "overlap_threshold",
        "max_distance",
        "max_darktime",
        "movie_length",
    ]

    # Add ground truth-specific parameters only if has_ground_truth is True
    if cfg.get("has_ground_truth", True):
        required.extend([
            "max_distance_ground_truth",
            "min_on_ground_truth",
            "max_dist_closest_ground_truth",
        ])

    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    return cfg


def load_run_info(run_folder: Path) -> dict:
    """Load run info (metadata) from run folder."""
    info_file = run_folder / "run_info.yaml"

    if not info_file.exists():
        raise FileNotFoundError(
            f"Run info not found: {info_file}\n"
            "Make sure you're pointing to a valid localization run folder."
        )

    with info_file.open("r") as f:
        info = yaml.load(f, Loader=yaml.SafeLoader)

    return info


def process_movie_pair(
    protein_file: Path,
    ground_truth_file: Path | None,
    cfg: dict,
    input_root: Path,
    run_folder: Path,
) -> bool:
    """
    Process a protein/ground truth movie pair (or single protein channel) to extract traces.

    Args:
        protein_file: Path to protein channel ND2 file.
        ground_truth_file: Path to ground truth channel ND2 file (None if no ground truth).
        cfg: Configuration dictionary.
        input_root: Input root directory (where ND2 files are).
        run_folder: Localization run folder (where locs and traces are saved).

    Returns:
        True if successful, False otherwise.
    """
    has_ground_truth = cfg.get("has_ground_truth", True)

    if has_ground_truth and ground_truth_file:
        logging.info(f"Processing pair: {protein_file.name} + {ground_truth_file.name}")
    else:
        logging.info(f"Processing: {protein_file.name}")

    # Output directory in run folder (mirror input structure)
    out_folder = run_folder / rel_under(input_root, protein_file)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Output base name
    out_base = out_folder / protein_file.stem

    # Load movies
    try:
        movie_protein = nd2.imread(str(protein_file))
        movie_ground_truth = nd2.imread(str(ground_truth_file)) if (has_ground_truth and ground_truth_file) else None
    except Exception as e:
        logging.error(f"Failed to load movies: {e}")
        return False

    # Check protein movie length
    MOVIE_LENGTH = int(cfg["movie_length"])
    nframes = movie_protein.shape[0]

    if nframes < MOVIE_LENGTH:
        logging.warning(
            f"Skipping {protein_file.name}: too short ({nframes} < {MOVIE_LENGTH})"
        )
        return False
    elif nframes > MOVIE_LENGTH:
        movie_protein = movie_protein[:MOVIE_LENGTH]
        logging.info(f"Cropped {protein_file.name} from {nframes} to {MOVIE_LENGTH} frames")

    # Max projections
    max_proj_protein = np.max(movie_protein, axis=0)
    max_proj_ground_truth = np.max(movie_ground_truth, axis=0) if movie_ground_truth is not None else None

    # Masks for overlap detection
    binary_mask = np.zeros(max_proj_protein.shape)
    label_mask = np.zeros(max_proj_protein.shape)

    # Load localization files from run folder (created by localize.py)
    locs_protein = out_folder / f"{protein_file.stem}_locs.hdf5"
    locs_ground_truth = out_folder / f"{ground_truth_file.stem}_locs.hdf5" if ground_truth_file else None

    if not locs_protein.exists():
        logging.error(f"Missing localizations: {locs_protein}")
        return False
    if has_ground_truth and locs_ground_truth and not locs_ground_truth.exists():
        logging.warning(f"Missing ground truth localizations: {locs_ground_truth}")
        # Can still process without ground truth (all traces labeled OUT)

    # Process protein channel localizations
    try:
        x_pix, y_pix, x_pix_t, y_pix_t, n_rois = get_and_link_locs(
            str(locs_protein),
            box_size=int(cfg["boxsize"]),
            max_distance=float(cfg["max_distance"]),
            max_off_time=int(cfg["max_darktime"]),
        )

        if len(x_pix) == 0:
            logging.warning(f"No localizations in {protein_file.name}")
            return False

        # Strip the dummy element at index 0 that get_and_link_locs prepends,
        # so ROI indices are 0-based over real localizations only.
        x_pix_t = x_pix_t[1:]
        y_pix_t = y_pix_t[1:]
        n_rois = len(x_pix_t)

        # Find overlapping ROIs
        overlapping_rois, binary_mask, label_mask = get_overlapping_rois(
            x_pix_t,
            y_pix_t,
            binary_mask,
            label_mask,
            box_size=int(cfg["boxsize"]),
            overlap_threshold=int(cfg["overlap_threshold"]),
        )

        non_overlapping_rois = list(set(range(n_rois)).difference(overlapping_rois))

        # Extract traces
        df = get_traces_new_incl_ov(
            movie_protein,
            non_overlapping_rois,
            overlapping_rois,
            x_pix_t,
            y_pix_t,
            box_side_length=int(cfg["boxsize"]),
        )

        logging.info(
            f"  Extracted {len(non_overlapping_rois)} single + {len(overlapping_rois)} overlapping ROIs"
        )

    except Exception as e:
        logging.error(f"Failed to process protein localizations: {e}")
        return False

    # Process ground truth localizations (only if has_ground_truth)
    ground_truth_positions = np.empty((0, 2))

    if has_ground_truth and locs_ground_truth and locs_ground_truth.exists():
        try:
            _, _, x_gt_t, y_gt_t, _, n_ground_truth = get_and_cluster_locs(
                str(locs_ground_truth),
                box_size=int(cfg["boxsize"]),
                max_distance=float(cfg.get("max_distance_ground_truth", 2.5)),
                min_locs=int(cfg.get("min_on_ground_truth", 3)),
            )

            # Remove dummy 0
            ground_truth_positions = np.stack((x_gt_t[1:], y_gt_t[1:]), axis=1)

            logging.info(f"  Found {n_ground_truth} ground truth clusters")

        except Exception as e:
            logging.warning(f"Failed to process ground truth: {e}")

    # Label traces as IN/OUT based on nearest ground truth (only if has_ground_truth)
    if has_ground_truth:
        if ground_truth_positions.size == 0:
            df["ground_truth"] = ["OUT"] * len(df)
        else:
            tree = spatial.KDTree(ground_truth_positions)
            max_dist = float(cfg.get("max_dist_closest_ground_truth", 2.0))

            gt_labels = []
            for x, y in zip(df["x"], df["y"]):
                dist, _ = tree.query((x, y))
                gt_labels.append("IN" if dist < max_dist else "OUT")

            df["ground_truth"] = gt_labels

    # Add origin column to protein traces
    df["origin"] = "protein"

    # Extract background traces if enabled
    bg_x, bg_y = None, None  # Initialize for max projection visualization
    n_background = cfg.get("n_background_traces", 0)
    if n_background > 0:
        try:
            df_background = extract_background_traces(
                movie_protein,
                binary_mask,
                boxsize=int(cfg["boxsize"]),
                n_traces=n_background,
                min_edge_distance=5,
                seed=42,
            )

            if len(df_background) > 0:
                # Background traces don't have ground truth labeling
                df_background["ground_truth"] = "background"

                # Store background positions for max projection visualization
                bg_x = df_background["x"].values
                bg_y = df_background["y"].values

                logging.info(f"  Extracted {len(df_background)} background traces")

                # Save background traces separately
                bg_out_file = str(out_base) + "_background_traces.pkl"
                df_background.to_pickle(bg_out_file)
                logging.info(f"  Saved background: {bg_out_file}")
            else:
                logging.warning("  No valid background positions found")

        except Exception as e:
            logging.warning(f"Failed to extract background traces: {e}")

    # Save protein trace file
    out_file = str(out_base) + "_traces.pkl"
    df.to_pickle(out_file)

    logging.info(f"  Saved: {out_file}")

    # =========================================================================
    # Visualization and QC
    # =========================================================================

    # Max projection control: protein channel with ROI overlays
    if cfg.get("draw_maxprojection_control", False):
        try:
            output_path = str(out_base) + "_max_proj_rect.tif"
            draw_max_projection(
                x_pix_t,
                y_pix_t,
                boxsize=int(cfg["boxsize"]),
                overlap=overlapping_rois,
                output_file=output_path,
                max_projection=max_proj_protein,
                bg_x=bg_x,
                bg_y=bg_y,
            )
            logging.info(f"  Saved max projection: {output_path}")
        except Exception as e:
            logging.warning(f"Failed to draw max projection: {e}")

    # Colocalization control: RGB composite (protein=red, ground_truth=cyan) or grayscale if no ground truth
    if cfg.get("draw_coloc_control", False):
        try:
            import pandas as pd

            # Create RGB composite: protein=red, GT=cyan
            im_red = colorize(max_proj_protein, (1, 0, 0))
            if max_proj_ground_truth is not None:
                im_cyan = colorize(max_proj_ground_truth, (0, 1, 1))
                im_comp = np.clip(im_red + im_cyan, 0, 1)
            else:
                im_cyan = None
                im_comp = im_red

            h_im, w_im = im_comp.shape[:2]
            box = int(cfg["boxsize"])

            # --- Composite image ---
            composite_dir = out_base.parent / (out_base.name + "_control_composite_GT")
            composite_dir.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(im_comp)

            # Ground truth boxes (cyan, dotted)
            # Offset by -0.5 because imshow places pixel centers at integer coordinates.
            if has_ground_truth and ground_truth_positions.size > 0:
                for x_gt, y_gt in ground_truth_positions:
                    ax.add_patch(patches.Rectangle(
                        (x_gt - 0.5, y_gt - 0.5), box, box,
                        linewidth=1, edgecolor="cyan", facecolor="none", linestyle=":",
                    ))

            # Protein trace boxes
            if has_ground_truth and "ground_truth" in df.columns:
                for x, y, gt, ov in zip(df["x"], df["y"], df["ground_truth"], df["overlap"]):
                    if gt == "IN" and ov == "single":
                        edge, style = "red", "-"
                    elif gt == "IN" and ov == "overlapping":
                        edge, style = "orange", "-"
                    elif gt == "OUT" and ov == "single":
                        edge, style = "grey", "-"
                    else:
                        continue  # OUT, overlapping: not plotted
                    ax.add_patch(patches.Rectangle(
                        (x - 0.5, y - 0.5), box, box,
                        linewidth=1, edgecolor=edge, facecolor="none", linestyle=style,
                    ))
            else:
                for x, y, ov in zip(df["x"], df["y"], df["overlap"]):
                    edge = "red" if ov == "single" else "orange"
                    ax.add_patch(patches.Rectangle(
                        (x - 0.5, y - 0.5), box, box,
                        linewidth=1, edgecolor=edge, facecolor="none", linestyle="-",
                    ))

            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            ax.set_title("Colocalization composite (protein=red, GT=cyan)", pad=9)
            ax.tick_params(labelsize=6)

            if has_ground_truth:
                legend_elements = [
                    Line2D([0], [0], color="cyan",   linewidth=2, linestyle=":", label="Ground truth"),
                    Line2D([0], [0], color="red",    linewidth=2, linestyle="-", label="IN, single"),
                    Line2D([0], [0], color="orange", linewidth=2, linestyle="-", label="IN, overlapping"),
                    Line2D([0], [0], color='#808080', linewidth=2, linestyle="-", label="OUT, single"),
                ]
            else:
                legend_elements = [
                    Line2D([0], [0], color="red",    linewidth=2, linestyle="-", label="Single"),
                    Line2D([0], [0], color="orange", linewidth=2, linestyle="-", label="Overlapping"),
                ]
            ax.legend(handles=legend_elements, loc="upper right", frameon=False)
            ax.grid(False)
            plt.tight_layout()
            # Re-apply after tight_layout: on older matplotlib, tight_layout with
            # aspect='equal' can silently reset the axis limits.
            ax.set_xlim(-0.5, w_im - 0.5)
            ax.set_ylim(h_im - 0.5, -0.5)
            plt.savefig(composite_dir / "plot.pdf", dpi=450, bbox_inches="tight")
            plt.close("all")

            # Save composite ROI data
            roi_rows = []
            if has_ground_truth and ground_truth_positions.size > 0:
                for x_gt, y_gt in ground_truth_positions:
                    roi_rows.append({"roi_type": "ground_truth", "label": "", "overlap": "", "x": int(x_gt), "y": int(y_gt)})
            for _, row in df.iterrows():
                roi_rows.append({"roi_type": "protein", "label": row.get("ground_truth", ""), "overlap": row["overlap"], "x": row["x"], "y": row["y"]})
            pd.DataFrame(roi_rows).to_csv(composite_dir / "data_rois.csv", index=False)

            logging.info(f"  Saved composite: {composite_dir}/")

            # --- Ground truth-only image ---
            if has_ground_truth and im_cyan is not None:
                gt_dir = out_base.parent / (out_base.name + "_ground_truth_only")
                gt_dir.mkdir(parents=True, exist_ok=True)

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(im_cyan)

                if ground_truth_positions.size > 0:
                    for x_gt, y_gt in ground_truth_positions:
                        ax.add_patch(patches.Rectangle(
                            (x_gt - 0.5, y_gt - 0.5), box, box,
                            linewidth=1, edgecolor="cyan", facecolor="none", linestyle=":",
                        ))

                ax.set_xlabel("X (pixels)")
                ax.set_ylabel("Y (pixels)")
                ax.set_title("Ground truth channel (488 nm)", pad=9)
                ax.tick_params(labelsize=6)
                ax.legend(
                    handles=[Line2D([0], [0], color="cyan", linewidth=2, linestyle=":", label=f"Ground truth (n={len(ground_truth_positions)})")],
                    loc="upper right", frameon=False,
                )
                ax.grid(False)
                plt.tight_layout()
                ax.set_xlim(-0.5, w_im - 0.5)
                ax.set_ylim(h_im - 0.5, -0.5)
                plt.savefig(gt_dir / "plot.pdf", dpi=450, bbox_inches="tight")
                plt.close("all")

                # Save GT ROI data
                gt_rows = [{"x": int(x_gt), "y": int(y_gt)} for x_gt, y_gt in ground_truth_positions]
                pd.DataFrame(gt_rows).to_csv(gt_dir / "data_rois.csv", index=False)

                logging.info(f"  Saved ground truth: {gt_dir}/")

            # --- IN proteins overlay ---
            if has_ground_truth and "ground_truth" in df.columns:
                in_dir = out_base.parent / (out_base.name + "_IN_proteins")
                in_dir.mkdir(parents=True, exist_ok=True)

                df_in_single = df[(df["ground_truth"] == "IN") & (df["overlap"] == "single")]

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(im_comp)

                for _, row in df_in_single.iterrows():
                    ax.add_patch(patches.Rectangle(
                        (row["x"] - 0.5, row["y"] - 0.5), box, box,
                        linewidth=1, edgecolor="red", facecolor="none", linestyle="-",
                    ))

                ax.set_xlabel("X (pixels)")
                ax.set_ylabel("Y (pixels)")
                ax.tick_params(labelsize=6)
                ax.grid(False)
                plt.tight_layout()
                ax.set_xlim(-0.5, w_im - 0.5)
                ax.set_ylim(h_im - 0.5, -0.5)
                plt.savefig(in_dir / "plot.pdf", dpi=450, bbox_inches="tight")
                plt.close("all")

                in_rows = [{"x": row["x"], "y": row["y"]}
                           for _, row in df_in_single.iterrows()]
                pd.DataFrame(in_rows).to_csv(in_dir / "data_rois.csv", index=False)

                logging.info(f"  Saved IN proteins: {in_dir}/")

        except Exception as e:
            logging.warning(f"Failed to draw colocalization control: {e}")

    return True


def extract_worker(args: tuple) -> tuple[Path, bool, str, str | None, str | None]:
    """
    Worker function for parallel trace extraction from movie pairs.

    Args:
        args: Tuple of (protein_file, ground_truth_file, cfg, input_root, run_folder)

    Returns:
        Tuple of (protein_file, success, error_message, trace_file_path, background_file_path)
    """
    protein_file, ground_truth_file, cfg, input_root, run_folder = args

    try:
        success = process_movie_pair(protein_file, ground_truth_file, cfg, input_root, run_folder)
        if success:
            out_base = run_folder / rel_under(input_root, protein_file)
            out_file = str(out_base / protein_file.stem) + "_traces.pkl"

            # Check if background file was created
            bg_file = str(out_base / protein_file.stem) + "_background_traces.pkl"
            bg_file_path = bg_file if Path(bg_file).exists() else None

            return (protein_file, True, "", out_file, bg_file_path)
        else:
            return (protein_file, False, "process_movie_pair returned False", None, None)
    except Exception as e:
        return (protein_file, False, str(e), None, None)


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract traces from ND2 movies using Picasso localizations"
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_path",
        required=True,
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "-r",
        "--run-folder",
        dest="run_folder",
        required=True,
        type=str,
        help="Path to localization run folder (e.g., Results/Extract/HaloD106_SNAPC148_grad640-12000_grad488-4000_001)",
    )
    args = parser.parse_args()

    # Setup
    setup_logging("INFO")
    cfg = load_config(Path(args.config_path))

    # Load run info
    run_folder = Path(args.run_folder).expanduser()
    run_info = load_run_info(run_folder)

    # Get input folder and proteins from run info
    input_root = Path(run_info["input_data"]["input_folder"]).expanduser()
    proteins = run_info["input_data"]["proteins"]
    protein_channel = run_info["input_data"].get("protein_channel", "640")
    ground_truth_channel = run_info["input_data"].get("ground_truth_channel", "488")
    has_ground_truth = run_info["input_data"].get("has_ground_truth", True)

    # Add has_ground_truth to cfg for use in process_movie_pair
    cfg["has_ground_truth"] = has_ground_truth

    # Get parallel processing configuration
    n_workers_config = cfg.get("n_workers", -1)
    num_workers = get_num_workers(n_workers_config)

    logging.info(f"Run folder: {run_folder}")
    logging.info(f"Input folder: {input_root}")
    logging.info(f"Proteins: {', '.join(proteins)}")
    logging.info(f"Has ground truth: {has_ground_truth}")

    # Discover paired movies using structured organization
    if has_ground_truth:
        logging.info("Discovering paired protein/ground truth movies...")
        protein_data = discover_all_data(
            input_root,
            proteins,
            protein_channel,
            ground_truth_channel
        )
    else:
        logging.info("Discovering protein channel movies...")
        # In single-channel mode, discover_all_data should return pairs with None as ground_truth_file
        protein_data = discover_all_data(
            input_root,
            proteins,
            protein_channel,
            None  # No ground truth channel
        )

    print_discovery_summary(protein_data)

    # Flatten to list of all pairs
    pairs = []
    for protein, protein_pairs in protein_data.items():
        pairs.extend(protein_pairs)

    if not pairs:
        if has_ground_truth:
            logging.error("No paired protein/ground truth movies found")
        else:
            logging.error("No protein movies found")
        sys.exit(1)

    logging.info(f"\nFound {len(pairs)} movie{'s' if not has_ground_truth else ' pairs'} to process")

    # Prepare work items for parallel/sequential processing
    work_items = [
        (protein_file, ground_truth_file, cfg, input_root, run_folder)
        for protein_file, ground_truth_file in pairs
    ]

    # Process movie pairs (parallel or sequential based on num_workers)
    trace_files_list = []
    background_files_list = []
    failed_files = []

    if num_workers == 1:
        # Sequential processing
        for work_item in work_items:
            protein_file = work_item[0]
            result = extract_worker(work_item)
            protein_file, success, error_msg, trace_file, bg_file = result
            if success and trace_file:
                trace_files_list.append(trace_file)
                if bg_file:
                    background_files_list.append(bg_file)
            else:
                logging.error(f"Error processing {protein_file.name}: {error_msg}")
                failed_files.append(str(protein_file))
    else:
        # Parallel processing
        import multiprocessing as mp
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(extract_worker, work_items)

        # Process results
        for protein_file, success, error_msg, trace_file, bg_file in results:
            if success and trace_file:
                trace_files_list.append(trace_file)
                if bg_file:
                    background_files_list.append(bg_file)
            else:
                logging.error(f"Error processing {protein_file.name}: {error_msg}")
                failed_files.append(str(protein_file))

    # Save file lists
    file_lists_dir = run_folder / "FileLists"
    file_lists_dir.mkdir(exist_ok=True)
    with open(file_lists_dir / "trace_file_list.pkl", "wb") as f:
        pickle.dump(trace_files_list, f)

    with open(file_lists_dir / "background_file_list.pkl", "wb") as f:
        pickle.dump(background_files_list, f)

    with open(file_lists_dir / "failed_file_list.pkl", "wb") as f:
        pickle.dump(failed_files, f)

    # Summary
    logging.info("=" * 60)
    logging.info(f"Extraction complete: {len(trace_files_list)} succeeded, {len(failed_files)} failed")
    if background_files_list:
        logging.info(f"Background traces: {len(background_files_list)} files")
    logging.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        logging.error(f"Fatal error: {ex}")
        sys.exit(1)
