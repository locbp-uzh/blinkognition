#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Figure S6 — Vesicle Occupation (Photobleaching)

Thin wrapper around Experiments/VesicleAnalysis/occupation.py
for Puentener 2026. Samples SP517 (main), SP512 (replicate).

Reads the KV step-detection result CSVs that were produced by the
original analysis pipeline and makes publication-quality plots.

Run from blinkognition2/ root:
    conda activate picasso-env
    python Papers/Puentener2026/figure_S6_occupation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Experiments.VesicleAnalysis.occupation import (
    load_config, load_concentration_series,
    plot_occupation_histograms, plot_mean_occupation,
    plot_occupation_across_replicates,
    setup_logging,
)

CONFIG = Path(__file__).parent / 'config_S6_occupation.yaml'


def main() -> None:
    cfg = load_config(CONFIG)
    setup_logging(cfg.get('log_level', 'INFO'))

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_n = cfg.get('max_proteins', 10)

    replicates = {}
    for rep_name, rep_cfg in cfg['replicates'].items():
        print(f"Loading replicate {rep_name}...")
        traces_dir     = Path(rep_cfg['traces_dir'])
        concentrations = rep_cfg.get('concentrations', None)
        rep_data = load_concentration_series(
            traces_dir,
            concentrations=concentrations,
            good_flag_only=cfg.get('good_flag_only', True),
        )
        replicates[rep_name] = rep_data

        # Print summary
        for conc, df in rep_data.items():
            from Experiments.VesicleAnalysis.occupation import mean_occupation
            print(f"  {rep_name} {conc}: {len(df)} vesicles, "
                  f"mean occupation = {mean_occupation(df):.2f}")

        rep_out = output_dir / rep_name
        plot_occupation_histograms(rep_data, rep_out, max_n=max_n)
        plot_mean_occupation(rep_data, rep_out)

    if len(replicates) > 1:
        plot_occupation_across_replicates(replicates, output_dir, max_n=max_n)

    print(f"\nDone. Results in {output_dir}")


if __name__ == '__main__':
    main()
