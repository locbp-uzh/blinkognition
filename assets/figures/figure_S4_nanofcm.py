#!/usr/bin/env python3
"""
Figure S4 — Vesicle Membrane Mixing (NanoFCM)

Thin wrapper around Characterization/fcs_analysis.py for Puentener 2026.
Samples SP522, SP523, SP528.

Run from blinkognition2/ root:
    conda activate picasso-env
    python Papers/Puentener2026/figure_S4_nanofcm.py

Output replaces the previously hard-coded mixing percentages in
Figure_S4/Vesicle_mixing_SP522_SP523.ipynb.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Characterization.fcs_analysis import (
    load_config, analyze_sample, plot_mixing_kinetics, plot_bar_chart,
    setup_logging,
)

CONFIG = Path(__file__).parent / 'config_S4_nanofcm.yaml'


def main() -> None:
    cfg = load_config(CONFIG)
    setup_logging(cfg.get('log_level', 'INFO'))

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    fitc_ch  = cfg.get('fitc_channel', 'FITC-H')
    pc5_ch   = cfg.get('pc5_channel', 'PC5-H')
    ss_ch    = cfg.get('ss_channel', 'SS-H')
    ss_min   = cfg.get('ss_gate_min_percentile', 5.0)
    ss_max   = cfg.get('ss_gate_max_percentile', 99.5)
    ctrl_pct = cfg.get('control_percentile', 99.0)

    results_by_sample = {}
    for sample_name, dirs in cfg['samples'].items():
        print(f"Analysing {sample_name}...")
        sample_dirs = [Path(d) for d in (dirs if isinstance(dirs, list) else [dirs])]
        df = analyze_sample(
            sample_dirs,
            fitc_channel=fitc_ch,
            pc5_channel=pc5_ch,
            ss_channel=ss_ch,
            ss_min_pct=ss_min,
            ss_max_pct=ss_max,
            control_percentile=ctrl_pct,
        )
        df.to_csv(output_dir / f'{sample_name}_fcs_results.csv', index=False)
        results_by_sample[sample_name] = df

        # Print summary (mirrors the old hardcoded table)
        print(f"  {sample_name} mixing percentages:")
        for _, row in df.iterrows():
            print(f"    {row['timepoint']:12s}: {row['mixing_pct']:.2f}%")

    plot_mixing_kinetics(results_by_sample, output_dir)
    plot_bar_chart(results_by_sample, output_dir)
    print(f"\nDone. Results in {output_dir}")


if __name__ == '__main__':
    main()
