# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
"""
Keep linked groups that touch the movie's first or last frame, for the FrameZero rerun only.

Extraction/utils.py get_and_link_locs calls picasso.postprocess.link(locs, info, r_max=..., max_dark_time=...)
with Picasso's default remove_ambiguous_lengths=True, which drops every linked group whose first frame is 0
(or whose last frame is not before the frame count). With FRAMEZERO_KEEP_EDGE_GROUPS=1 and this folder on
PYTHONPATH (Python imports sitecustomize at start-up in every process, including the pipeline's step
subprocesses and joblib workers), postprocess.link is wrapped so that remove_ambiguous_lengths is False.
Nothing else changes. Same mechanism as Revisions/BlankSurface/picasso_sync.
"""
import os
import sys

if os.environ.get("FRAMEZERO_KEEP_EDGE_GROUPS") == "1":
    try:
        import picasso.postprocess as _pp

        _link_original = _pp.link

        def _link_keep_edges(*args, **kwargs):
            kwargs["remove_ambiguous_lengths"] = False
            return _link_original(*args, **kwargs)

        _link_keep_edges.__wrapped__ = _link_original
        _pp.link = _link_keep_edges
        print("[FrameZero] picasso postprocess.link: remove_ambiguous_lengths=False", file=sys.stderr)
    except Exception as e:  # noqa: BLE001  (never block start-up)
        print(f"[FrameZero] keep-edges patch not applied: {e!r}", file=sys.stderr)
