#!/usr/bin/env python3
"""
Plot random traces from a filtered traces pkl file.
"""
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 450
mpl.rcParams['pdf.fonttype'] = 42

COLORS = {
    'blue': '#0072B2',
}

def plot_random_traces(pkl_path: Path, n: int = 30, seed: int = 42) -> None:
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)

    rng = np.random.default_rng(seed)
    cols = rng.choice(df.columns, size=min(n, len(df.columns)), replace=False)

    fig, axes = plt.subplots(5, 6, figsize=(21, 17))
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        ax.plot(df.index, df[col], '-', color=COLORS['blue'], linewidth=0.8)
        ax.set_xlabel('Frames')
        ax.set_ylabel('Intensity')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.1)
        ax.spines['bottom'].set_linewidth(1.1)

    stem = pkl_path.stem
    out_dir = pkl_path.parent / f"{stem}_random_traces"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.tight_layout(h_pad=3.0, w_pad=2.0)
    plt.savefig(out_dir / 'plot.pdf', dpi=450, bbox_inches='tight')
    plt.close()

    pd.DataFrame({'trace_index': cols}).to_csv(out_dir / 'data.csv', index=False)
    print(f"Saved to {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl', type=Path, help='Path to filtered traces pkl file')
    parser.add_argument('--n', type=int, default=30, help='Number of traces to plot')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    plot_random_traces(args.pkl, n=args.n, seed=args.seed)
