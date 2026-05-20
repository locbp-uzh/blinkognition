#!/usr/bin/env python3
"""
download_data.py — Download input data from Zenodo into Inputs/.

Downloads FinalTraces (filtered ML-ready traces) and/or SampleMovies
(raw ND2 files for re-running the extraction pipeline) from the Zenodo
record associated with this repository.

Usage
-----
    python download_data.py --dataset traces    # ~5.2 GB
    python download_data.py --dataset movies    # sample ND2 movies
    python download_data.py --dataset all       # both

After downloading, data lands in Inputs/:
    Inputs/FinalTraces/   — use with ML/config_train.yaml and ML/config_cv.yaml
    Inputs/SampleMovies/  — use with Extraction/config.yaml

NOTE: Replace ZENODO_RECORD_ID and the md5 checksums below once the
Zenodo record is published (available on the record's files tab).
"""

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Zenodo record settings — update after publishing
# ---------------------------------------------------------------------------
ZENODO_RECORD_ID = "XXXXXXX"  # TODO: replace with the actual Zenodo record ID
_BASE = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"

DATASETS = {
    "traces": {
        "url":         f"{_BASE}/FinalTraces.tar.gz?download=1",
        "filename":    "FinalTraces.tar.gz",
        "md5":         None,   # TODO: fill in after upload (from Zenodo files tab)
        "description": "Filtered protein and background traces (~5.2 GB)",
    },
    "movies": {
        "url":         f"{_BASE}/SampleMovies.tar.gz?download=1",
        "filename":    "SampleMovies.tar.gz",
        "md5":         None,   # TODO: fill in after upload
        "description": "Sample ND2 movies for extraction pipeline (~XX GB)",
    },
}

REPO_ROOT  = Path(__file__).resolve().parent
INPUTS_DIR = REPO_ROOT / "Inputs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_progress(downloaded: int, total: int) -> None:
    if total <= 0:
        print(f"\r  {downloaded / 1e6:.1f} MB downloaded", end="", flush=True)
        return
    pct  = downloaded / total * 100
    done = int(pct / 2)
    bar  = "#" * done + "-" * (50 - done)
    print(
        f"\r  [{bar}] {pct:5.1f}%  {downloaded / 1e9:.2f}/{total / 1e9:.2f} GB",
        end="",
        flush=True,
    )


def _download(url: str, dest: Path) -> None:
    print(f"  Downloading → {dest.name}")
    with urllib.request.urlopen(url) as resp:
        total     = int(resp.headers.get("Content-Length", 0))
        chunk     = 1 << 20  # 1 MiB
        received  = 0
        with open(dest, "wb") as f:
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                f.write(data)
                received += len(data)
                _print_progress(received, total)
    print()  # newline after progress bar


def _verify_md5(path: Path, expected: str) -> None:
    print("  Verifying checksum …", end=" ", flush=True)
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != expected:
        path.unlink(missing_ok=True)
        raise ValueError(
            f"MD5 mismatch for {path.name}\n"
            f"  expected: {expected}\n"
            f"  got:      {h.hexdigest()}"
        )
    print("OK")


def _extract(archive: Path, dest_dir: Path) -> None:
    print(f"  Extracting → {dest_dir.name}/")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest_dir)
    print("  Done.")


def download_dataset(name: str) -> None:
    meta = DATASETS[name]
    print(f"\n[{name}] {meta['description']}")

    archive = INPUTS_DIR / meta["filename"]
    extract_dir = INPUTS_DIR / Path(meta["filename"]).stem.split(".")[0]

    if extract_dir.exists():
        print(f"  Already present: {extract_dir.relative_to(REPO_ROOT)} — skipping.")
        return

    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        _download(meta["url"], archive)
    except Exception as exc:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {name}: {exc}") from exc

    if meta["md5"]:
        _verify_md5(archive, meta["md5"])
    else:
        print("  (No checksum configured — skipping verification)")

    _extract(archive, INPUTS_DIR)

    archive.unlink()
    print(f"  Cleaned up archive.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download blinkognition input data from Zenodo into Inputs/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python download_data.py --dataset traces\n"
            "  python download_data.py --dataset movies\n"
            "  python download_data.py --dataset all\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["traces", "movies", "all"],
        required=True,
        help=(
            "traces — FinalTraces (filtered PKL files for ML/Features/Controls); "
            "movies — SampleMovies (raw ND2 files for Extraction); "
            "all — both datasets"
        ),
    )
    args = parser.parse_args()

    if ZENODO_RECORD_ID == "XXXXXXX":
        print(
            "ERROR: Zenodo record ID has not been set.\n"
            "       Edit ZENODO_RECORD_ID in download_data.py after publishing the record.",
            file=sys.stderr,
        )
        sys.exit(1)

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    print(f"Downloading to: {INPUTS_DIR.relative_to(REPO_ROOT)}/")

    for name in targets:
        download_dataset(name)

    print("\nAll done. Update config paths to point to Inputs/ as needed.")


if __name__ == "__main__":
    main()
