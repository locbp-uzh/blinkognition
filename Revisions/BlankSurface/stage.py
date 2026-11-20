#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
Stage one BlankSurface extraction (R3.1): symlinks to the raw movies in the pipeline's layout, and a run
config derived from the paper's Extraction/config.yaml.

For a vesicle channel (--gt 488 or 405) and a protein gradient (--gradient), every slide of config.yaml with
that vesicle channel is linked into <stage_root>/gt<gt>_g<gradient>/<session>/<slide name>/ (only its
640 nm movies and its vesicle-channel snapshots; Picasso writes its _locs files next to the links). Each link
is named <slide name>_<original file name>: Extraction/combine.py assigns per-movie traces to a protein by
matching the protein name in the file name (case-sensitive, bounded by underscores), and the pairing of the
two channels keys on the first name token and the field number, so the slide name must lead the file name
(the raw files are named blank_*, ATTO390_*, ...; without the prefix combine finds no files for them). Every
link name is checked against combine's own patterns before anything is written. Then
<stage_root>/configs/config_gt<gt>_g<gradient>.yaml is written: the paper's config with config.yaml
`overrides`, input_folder = the staging folder, output_folder_base = Results/Revisions/BlankSurface/
Extraction_gt<gt>_g<gradient> in this clone, proteins = the slide names, ground_truth_channel, and the fixed
gradients (gradient_protein, gradient_ground_truth); the localize step uses them when no optimized
parameter file exists, which this staging never creates. A link map (link_map.tsv) is written next to it.

Refuses to reuse an existing staging folder (a previous run's _locs files would be mixed in).

Usage (clone root, on S3IT):
    python Revisions/BlankSurface/stage.py --gt 488 --gradient 10000
Prints the config path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "Extraction"))
from combine import build_compound_patterns  # noqa: E402  (the pipeline's own file-to-protein matcher)


def channel_token(ch: str) -> str:
    return f"_{ch}nm_"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, choices=["488", "405"])
    ap.add_argument("--gradient", required=True, type=int)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.gradient not in cfg["gradient_protein"]:
        sys.exit(f"gradient {args.gradient} is not one of config.yaml gradient_protein {cfg['gradient_protein']}")

    tag = f"gt{args.gt}_g{args.gradient}"
    stage = Path(cfg["stage_root"]) / tag
    if stage.exists():
        sys.exit(f"{stage} exists; use a fresh staging folder (old _locs files would be reused)")
    slides = [s for s in cfg["slides"] if s["gt"] == args.gt]
    patterns = build_compound_patterns(sorted({s["name"] for s in slides}))
    plan, links = [], []
    for s in slides:
        src = Path(cfg["source_root"]) / s["folder"]
        dst = stage / s["session"] / s["name"]
        movies = sorted(p for p in src.iterdir() if p.suffix.lower() == ".nd2" and not p.name.startswith("._")
                        and p.name.startswith(s["prefix"] + "_")
                        and (channel_token("640") in p.name or channel_token(args.gt) in p.name))
        n640 = sum(channel_token("640") in p.name for p in movies)
        ngt = sum(channel_token(args.gt) in p.name for p in movies)
        if not n640 or n640 != ngt:
            sys.exit(f"{src}: {n640} movies at 640 nm and {ngt} at {args.gt} nm")
        for p in movies:
            name = f"{s['name']}_{p.name}"
            hits = sorted(c for c, pat in patterns.items() if pat.search(name))
            if hits != [s["name"]] or name.split("_")[0] != s["name"]:
                sys.exit(f"link name {name} matches proteins {hits}, not only {s['name']}")
            plan.append((dst / name, p))
            links.append(f"{s['session']}\t{s['name']}\t{name}\t{p}")
    for link, target in plan:          # nothing is written until every slide and name has been checked
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)

    with open(REPO_ROOT / cfg["base_config"]) as f:
        run_cfg = yaml.safe_load(f)
    run_cfg.update(cfg["overrides"])
    run_cfg.update({
        "input_folder": str(stage),
        "output_folder_base": str(REPO_ROOT / cfg["output_root"] / f"Extraction_{tag}"),
        "proteins": sorted({s["name"] for s in slides}),
        "protein_channel": "640",
        "ground_truth_channel": args.gt,
        "gradient_protein": args.gradient,
        "gradient_ground_truth": cfg["gradient_ground_truth"][args.gt],
        "resume_run_folder": None,
    })
    cdir = Path(cfg["stage_root"]) / "configs"
    cdir.mkdir(parents=True, exist_ok=True)
    out = cdir / f"config_{tag}.yaml"
    with open(out, "w") as f:
        f.write(f"# BlankSurface run config ({tag}), generated by Revisions/BlankSurface/stage.py from "
                f"{cfg['base_config']} and Revisions/BlankSurface/config.yaml\n")
        yaml.safe_dump(run_cfg, f, sort_keys=False)
    (stage / "link_map.tsv").write_text("session\tslide\tlink\ttarget\n" + "\n".join(links) + "\n")
    print(out)


if __name__ == "__main__":
    main()
