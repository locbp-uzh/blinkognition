#!/usr/bin/env python3
"""
Figure S5 — Cargo Exchange Kinetics

Thin wrapper around Experiments/VesicleAnalysis/cargo_exchange.py
for Puentener 2026. Samples SP520, SP521.

Run from blinkognition2/ root:
    conda activate picasso-env
    python Papers/Puentener2026/figure_S5_cargo_exchange.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Experiments.VesicleAnalysis.cargo_exchange import (
    load_config, analyze_experiment, plot_kinetics, setup_logging,
)

CONFIG = Path(__file__).parent / 'config_S5_cargo_exchange.yaml'


def main() -> None:
    cfg = load_config(CONFIG)
    setup_logging(cfg.get('log_level', 'INFO'))

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_dist = cfg.get('max_dist', 3.0)

    results_by_replicate = {}
    for rep_name, rep_cfg in cfg['replicates'].items():
        print(f"Analysing replicate {rep_name}...")
        picasso_base = Path(rep_cfg['picasso_dir'])
        tp_map_raw   = rep_cfg.get('slide_timepoints', None)
        tp_map = {k: (None if v is None else float(v))
                  for k, v in tp_map_raw.items()} if tp_map_raw else None

        df = analyze_experiment(
            picasso_base,
            slide_timepoint_map=tp_map,
            max_dist=max_dist,
            c_protein=cfg.get('channel_protein', 'C1'),
            c_vesicle_a=cfg.get('channel_vesicle_a', 'C3'),
            c_vesicle_b=cfg.get('channel_vesicle_b', 'C4'),
        )
        df.to_csv(output_dir / f'cargo_exchange_{rep_name}.csv', index=False)
        results_by_replicate[rep_name] = df

        print(f"  {rep_name} results:")
        mixed = df[~df['is_control']].sort_values('timepoint_h')
        print(mixed[['slide', 'timepoint_h', 'n_protein',
                      'pct_in_A520', 'pct_in_A425', 'pct_free']].to_string(index=False))

    plot_kinetics(results_by_replicate, output_dir)
    print(f"\nDone. Results in {output_dir}")


if __name__ == '__main__':
    main()
