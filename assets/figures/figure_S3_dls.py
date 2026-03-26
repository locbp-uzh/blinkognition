#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH), Andreas Biri (ETHZ).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Figure S3 — Vesicle Size Stability (DLS)

Thin wrapper around Characterization/dls_analysis.py for Puentener 2026.
Samples SP513, SP515, SP516.

Run from blinkognition2/ root:
    conda activate picasso-env
    python Papers/Puentener2026/figure_S3_dls.py
"""
import sys
from pathlib import Path

# Allow import from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Characterization.dls_analysis import (
    load_config, load_sample, plot_stability, setup_logging,
)

CONFIG = Path(__file__).parent / 'config_S3_dls.yaml'


def main() -> None:
    cfg = load_config(CONFIG)
    setup_logging(cfg.get('log_level', 'INFO'))

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs_by_condition = {}
    for condition_name, sample_dir in cfg['samples'].items():
        df = load_sample(Path(sample_dir))
        df.to_csv(output_dir / f'dls_{condition_name}.csv', index=False)
        dfs_by_condition[condition_name] = df

    plot_stability(
        dfs_by_condition,
        output_dir,
        temperatures=cfg.get('temperatures'),
    )
    print(f"Done. Results in {output_dir}")


if __name__ == '__main__':
    main()
