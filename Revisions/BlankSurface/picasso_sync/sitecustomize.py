# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
"""
Synchronous MLE fitting in Picasso's command-line localization, for the BlankSurface runs only.

`python -m picasso localize` (0.8.8 and 0.9.3; __main__.py: `while current[0] < n_spots`) starts
gaussmle.gaussmle_async with min(60, 0.75 x cpu_count) threads and reads the fits once the shared
counter reaches the number of spots, i.e. once the last spot is handed to a thread, not fitted. Fits
still running are read as zeros and dropped. Dense movies lose at most the few spots in flight;
sparse movies, such as blank coverslips, can lose many or all.

With BLANKSURFACE_PICASSO_SYNC_MLE=1 and this folder on PYTHONPATH (Python imports sitecustomize at
start-up in every process, including the picasso subprocesses), gaussmle.gaussmle_async is replaced by
the synchronous gaussmle.gaussmle, which fits every spot before returning. Same approach as
Revisions/DualColor/daint/picasso_sync/sitecustomize.py (not imported from there; that folder belongs to
another analysis).
"""
import os
import sys

if os.environ.get("BLANKSURFACE_PICASSO_SYNC_MLE") == "1":
    try:
        import picasso.gaussmle as _gm

        def _gaussmle_sync(spots, eps, max_it, method="sigmaxy"):
            thetas, crlbs, likelihoods, iterations = _gm.gaussmle(spots, eps, max_it, method=method)
            return [len(spots)], thetas, crlbs, likelihoods, iterations

        _gm.gaussmle_async = _gaussmle_sync
        print("[BlankSurface] picasso gaussmle_async -> synchronous gaussmle", file=sys.stderr)
    except Exception as e:  # noqa: BLE001  (never block start-up)
        print(f"[BlankSurface] picasso sync patch not applied: {e!r}", file=sys.stderr)
