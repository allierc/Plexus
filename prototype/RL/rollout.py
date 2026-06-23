"""rollout.py -- the RL environment step: u -> simulation -> trajectory.

A single function that turns a parameter vector into a forward rollout using the
*codebase* engine and registered operators (no reimplementation). Returns the
recorded cell trajectory (positions per frame + static per-cell type). Velocity is
recovered downstream as pos[t+1]-pos[t] (the recorder stores positions only).
"""
from __future__ import annotations

import numpy as np
from plexus.engine import run


def rollout(mech, u, seed=0, device="cpu"):
    """(pos:(F,N,2), node_type:(N,)) or None on instability/blow-up."""
    try:
        spec, _ = mech.apply_u(u, seed=seed)
        _, out = run(spec, out_path=None, device=device)
    except Exception:
        return None
    s = out["sets"][mech.set_name]
    pos = np.asarray(s["pos"], dtype=np.float32)
    nt = s["node_type"]
    nt = np.zeros(pos.shape[1], dtype=int) if nt is None else np.asarray(nt)
    if not np.all(np.isfinite(pos)):
        return None
    return pos, nt
