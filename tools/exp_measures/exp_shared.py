"""Measures every experiment of exp06-15 may read: the growth auditor, and cell counts over a run.

    shared.growth_audit   exp 3's growth auditor (`tools/growth_audit.py`), 0-10: the caps' "wrecked run"
    shared.cell_count     live cells first / last / min / max over the run
"""
from __future__ import annotations

import os

import numpy as np

from .common import cells, finite, register_run


def _spec_name(T):
    from plexus.paths import graphs_data_path
    rel = os.path.relpath(T.dir, graphs_data_path())
    return rel.replace(os.sep, "/")


def growth_audit(T, every=20, window=60, jump_every=1, **_):
    """exp 3's auditor on this run: score 0-10, band, and the reason that set it. Mesh runs only.
    The vertex-jump and finiteness tests run on every frame (`jump_every` 1, since 2026-09-27)."""
    import growth_audit as GA
    try:
        r = GA.audit(_spec_name(T), every=every, window=window, jump_every=jump_every)
    except Exception as e:                                                   # noqa: BLE001
        return {"available": False, "why": f"{type(e).__name__}: {e}"}
    return {"available": True, "score": finite(r.get("score")), "band": r.get("band"), "why": r.get("reason"), "growth": finite(r.get("growth")),
            "jump_max": finite(r.get("jump_max")), "jump_at": r.get("jump_at"), "jumps_over": r.get("jumps_over"), "wrecked_at": r.get("wrecked_at")}


def cell_count(T, **_):
    n = [len(cells(T, t)) for t in range(T.n_rows())]
    return {"first": n[0], "last": n[-1], "min": int(min(n)), "max": int(max(n)),
            "last_over_first": finite(n[-1] / max(n[0], 1))}


register_run("shared.growth_audit", growth_audit, None, "exp 3's growth auditor, 0-10")
register_run("shared.cell_count", cell_count, "count", "live cells over the run")
