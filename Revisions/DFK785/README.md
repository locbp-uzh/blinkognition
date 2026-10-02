# DFK785: single-label and mixed ATTO390 / ATTO520 vesicle slides

Dataset acquired 2026-09-28/29, now at `/Volumes/MediumBerth/Agent/20260928_DFK785`
(moved there by the user on 2026-10-02; `20260926_readme.txt` inside describes the
slides). Work lives on the `revisions` branch only. Shared code (detection,
config, provenance) comes from `../Bleedthrough/common.py`.

| Slide | Folder | Sample | FOVs | Channels |
|---|---|---|---|---|
| 1 | SNAP520 | SNAP-HMSiR-IA in ATTO520 vesicles | 31 | 405, 488, 640 |
| 2 | HT390 | HT7-HMSiR-HTL in ATTO390 vesicles | 31 | 405, 488, 640 |
| 3 | HT390_SNAP520 | mix 350 uL HT390 + 50 uL SNAP520, overnight | 61 | 405, 488, 515, 640 |
| 4 | HT390_SNAP520/new | same mix, next morning | 60 | 405, 488, 515, 640 |

One ND2 per channel and FOV, 230 x 230 px at 0.13 um; 405/488/515 are 10-frame
movies, 640 (HMSiR) 6000 frames. All raw ND2 files except the two test images carry
the macOS lock flag (`uchg`): they can be read but not renamed or deleted.

## Running

```bash
python Revisions/DFK785/fov_qc.py           # vesicle counts per FOV in acquisition order
python Revisions/DFK785/move_dried.py       # dry run of moving dried acquisitions to bad_data/
```

## FOV pairing

FOVs are matched by acquisition time (ND2 metadata), not by file number: on slide 3
the numbering is shifted between channels (the first FOV is 405/488/640 file _0031
with the unnumbered 515 file, and for most FOVs 515 file N goes with 405/488/640
file N-1). Within every FOV the channels are recorded within 5-11 s; FOVs are about
3.6 min apart.

## Drying QC (fov_qc run_002, 2026-10-02)

Vesicles detected per FOV (10-frame mean, the detection of
Revisions/Bleedthrough/segment.py, SNR >= 6):

| Slide | Good FOVs (acquisition order) | Dried | Counts in good FOVs (405 / 488 / 515) |
|---|---|---|---|
| 1 | all 31 | - | 78-126 / 160-212 / - |
| 2 | all 31 | - | 156-193 / 45-77 / - |
| 3 | 1-21 (20:57-22:09) | 22-61, from 22:12 | 131-156 / 150-184 / 152-187 |
| 4 | 1-35 (10:58-13:00) | 36-60, from 13:04 | 100-130 / 70-105 / 82-117 |

- The drop is abrupt on both mixed slides (slide 3: FOV 21 has 135/167/171, FOV 22
  has 0/0/1; slide 4: FOV 35 has 105/81/86, FOV 36 has 0/0/17).
- Slide 3 FOVs 41-42 (23:20, 23:24) show 58/30/31 and 5/8/8: a brief remnant after
  drying, counted as dried.
- On dried slide 4 FOVs, 515 keeps 0-20 spots while 405 and 488 are at 0; probably
  residue, not vesicles.
- The data readme says slide 4 dried after about 25 FOVs; the counts put it at 35.
- Per-channel counts are not label-specific (each dye is also detected in the other
  channels when bright), so they only say whether vesicles are present.

Rule used by `move_dried.py` (config `dried`): a slide is dried from the first FOV
whose 405 + 488 count is below 0.1 x the median of the earlier FOVs (at least 5),
and every later FOV is dried too.

## Move to bad_data/ (done 2026-10-02)

At the user's request all raw files were unlocked (`chflags nouchg` on all 671
locked files, the dataset's own files plus one stray copy) and the 260 files of
the dried FOVs (slide 3 FOVs 22-61, slide 4 FOVs 36-60, 65 FOVs x 4 channels) were
moved with `move_dried.py --apply`. They keep their relative paths under
`bad_data/`; `bad_data/moved_files.csv` lists every file with its FOV's acquisition
order, time and counts, and `bad_data/README.md` explains the move.

Checked from the ND2 timestamps after the move: the kept files end at 22:08:55
(slide 3) and 13:00:21 (slide 4), the moved ones start at 22:12:24 and 13:03:50.
Remaining usable data: slide 3, 21 FOVs (84 files); slide 4, 35 FOVs (140 files);
slides 1 and 2 untouched.

The first attempt (before the lock check) had left one identical copy of
`HT390_SNAP520_405nm_TIRF2x_23pr_100ms_230x230_0020.nd2` in `bad_data/HT390_SNAP520/`;
it was renamed to `<name>.duplicate` (not deleted) and can be removed by the user.

The files are no longer locked. `fov_qc.py` now only sees the kept FOVs; the
reference counts for all FOVs stay in fov_qc run_002.
