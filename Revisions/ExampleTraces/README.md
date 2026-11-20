# ExampleTraces: single-molecule traces of every construct (R2.1)

Reviewer 2, point 1: example single-molecule traces for every construct. Plan and decisions in `../ReviewRound1/README.md`.

## What is shown

For each construct (Halo-D106, SNAP-C148, Halo-K117, scGrx1, scGrx1-AcK20), 3 traces drawn at
random (numpy `default_rng(42)`, one generator for all constructs in that order) from all
traces inside vesicles that passed the paper's quality filter (the published filtered IN set,
2580 / 9647 / 2384 / 3400 / 1260 traces). No selection by appearance or by model output.

- Left: the full trace, 6000 frames at 34.49 ms per frame.
- Right: the 10 s window (290 frames) with the most ON frames, earliest if tied; time from the
  window start. ON frames as the paper classifies them for Figs. 5D and 6C
  (`Extraction/utils.py` `gmm_classify_frames` on the min-max trace, posterior >= 0.9).
- Intensity: the extraction's bg_rm traces (5 x 5 pixel sum minus the per-trace GMM
  background), in detected photons per frame (x 16.0 / 300, `../PaperData/`); one common
  intensity scale for all panels.

No statistics.

## Files

- `figure_R2.1.py`: `--build [--out <figure folder>]` selects and extracts the traces from the
  local mirror of the published extraction (`../PaperData/`, `load_config(local=True)`), writes
  the CSVs, caption and provenance, and copies itself into the figure folder; without `--build`
  it replots from the CSVs next to it.

Figure folder: `/Volumes/MediumBerth/SalomePaperData/Püntener2026/revisions_figures/R2.1/`:
`R2.1.pdf`, `R2.1_traces.csv`, `R2.1_selection.csv`, `R2.1_caption.txt`, `R2.1_provenance.yaml`,
`figure_R2.1.py`.

```bash
/Users/privera/opt/anaconda3/envs/locbplots/bin/python Revisions/ExampleTraces/figure_R2.1.py --build --out "/Volumes/MediumBerth/SalomePaperData/Püntener2026/revisions_figures/R2.1"
```

## Log

- 2026-11-20: first build. The paper's `Extraction/utils.py` sets DejaVu Sans first in the
  font list on import, so the figure style is applied again inside `plot()` (the PDF embeds
  Helvetica, checked with pdffonts). Selected traces: Halo-D106 Exp1/Exp4/Exp4, SNAP-C148
  Exp3/Exp4/Exp4, Halo-K117 Exp2 x 3, scGrx1 Exp5/Exp6/Exp9, scGrx1-AcK20 Exp6/Exp6/Exp7 (session
  shares follow the pools, e.g. 88 % of SNAP-C148 traces are Exp4).
- Seen in the traces, for later points: most ON events last one to a few frames; a few traces
  show plateaus of 0.5-1 s (Halo-D106 trace 469, scGrx1-AcK20 trace 7018). In traces with few
  bright events the per-trace classifier also marks low-level fluctuations as ON (e.g. SNAP-C148
  trace 24676, Halo-K117 trace 5246), which matters for the ON and OFF times of Figs. 5D and 6C
  (R3.2, R3.3).
