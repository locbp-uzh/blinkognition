# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
"""
Fix a race in Picasso 0.8.8's CLI MLE fitting, for the DualColor extraction jobs only.

`picasso localize` (__main__._localize) starts gaussmle.gaussmle_async, which hands spots to
min(60, 0.75 x cpu_count) threads from a shared counter, and then waits only until the counter
reaches the number of spots, i.e. until the last spot has been HANDED OUT, not fitted. Fits
still running then are read as zeros and lost. On a 288-core Daint node (60 threads) a movie
with fewer than 60 spots can lose every spot (6 of the 118 DFK785 movies came back empty; their
local localizations had 13-51 spots); on large movies a few of the last spots are lost.

With DUALCOLOR_PICASSO_SYNC_MLE=1 this module (put on PYTHONPATH, so Python imports it at
start-up in every process of the job, including the `python -m picasso localize` subprocesses)
replaces gaussmle.gaussmle_async by the synchronous gaussmle.gaussmle, which runs the same
per-spot fit for every spot before returning. Nothing else changes; the container image and
the paper's code are untouched.
"""
import os
import sys

if os.environ.get("DUALCOLOR_PICASSO_SYNC_MLE") == "1":
    try:
        import picasso.gaussmle as _gm

        def _gaussmle_sync(spots, eps, max_it, method="sigma"):
            thetas, crlbs, likelihoods, iterations = _gm.gaussmle(spots, eps, max_it, method=method)
            return [len(spots)], thetas, crlbs, likelihoods, iterations

        _gm.gaussmle_async = _gaussmle_sync
    except Exception as exc:  # never break an unrelated Python process
        print(f"[dualcolor sitecustomize] picasso patch not applied: {exc}", file=sys.stderr)
