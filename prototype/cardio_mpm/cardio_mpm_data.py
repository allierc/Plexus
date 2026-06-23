#!/usr/bin/env python
"""cardio_mpm_data.py -- map the REAL cardiomyocyte trajectories onto the MPM sheet.

The real data (`cardio_real.npz`, 137^2 tracked nodes, normalised ~[0,1]) is mapped into
the MPM sheet domain [DOM_LO,DOM_HI]^2 (the block-filled elastic sheet), resampled to the
run length, and indexed per MPM particle (nearest real node by rest position). Provides:

  real_disp[T, N, 2]  -- the real displacement (from rest) carried by each MPM particle
                         (so anchored = particle_rest + real_disp[t]).
  bnd[N] (bool)       -- the OUTER BAND of the sheet (within `bwidth` of the domain border),
                         the Dirichlet ring pinned to the real data; the interior is predicted.
"""
from __future__ import annotations
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "..", "cardio", "cardio_real.npz")
DOM_LO, DOM_HI = 0.15, 0.85          # MPM sheet domain (matches the spec's block: [0.15,0.15,0.85,0.85])
DOM = DOM_HI - DOM_LO


def load_real(rest_pos, bwidth=0.06, npz=NPZ):
    """Beat-aware, NON-resampled mapping (1 model frame = 1 real frame) for the inverse fit.
    Returns (real_disp [F, N, 2] per particle (mapped to domain, displacement from real frame 0),
    bnd [N], onsets [list of beat-onset frames], period int)."""
    from scipy.spatial import cKDTree
    from scipy.signal import find_peaks
    P = np.load(npz)["pos"].astype(np.float32)               # [F, 137^2, 2] in ~[0,1]
    Pm = DOM_LO + DOM * P                                     # map real -> sheet domain
    node = cKDTree(Pm[0]).query(np.asarray(rest_pos, np.float32))[1]
    real_disp = (Pm[:, node] - Pm[0, node]).astype(np.float32)            # [F, N, 2]
    spd = np.linalg.norm(np.diff(P, axis=0), axis=2).mean(1)              # mean nodal speed per frame
    pk, _ = find_peaks(spd, height=spd.mean(), distance=20)               # beat onsets
    period = int(round(float(np.diff(pk).mean()))) if len(pk) > 1 else 50
    rp = np.asarray(rest_pos, np.float32)
    bnd = ((rp[:, 0] < DOM_LO + bwidth) | (rp[:, 0] > DOM_HI - bwidth)
           | (rp[:, 1] < DOM_LO + bwidth) | (rp[:, 1] > DOM_HI - bwidth))
    return real_disp, bnd, [int(p) for p in pk], period


def load_real_for(rest_pos, n_frames, bwidth=0.06, npz=NPZ):
    """rest_pos [N,2] MPM particle REST positions (in the sheet domain). Returns
    (real_disp [n_frames+1, N, 2], bnd [N] bool)."""
    from scipy.spatial import cKDTree
    P = np.load(npz)["pos"].astype(np.float32)               # [F, 137^2, 2] in ~[0,1]
    Pm = DOM_LO + DOM * P                                     # map real -> sheet domain
    F = Pm.shape[0]; T = int(n_frames) + 1
    idx = np.linspace(0, F - 1, T)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, F - 1); f = (idx - lo)[:, None, None]
    Pr = (1 - f) * Pm[lo] + f * Pm[hi]                        # [T, M, 2] resampled real (sheet domain)
    # each MPM particle -> nearest real node by rest position (robust to ordering)
    node = cKDTree(Pr[0]).query(np.asarray(rest_pos, np.float32))[1]   # [N]
    real_disp = (Pr[:, node] - Pr[0, node]).astype(np.float32)         # [T, N, 2] displacement per particle
    rp = np.asarray(rest_pos, np.float32)
    bnd = ((rp[:, 0] < DOM_LO + bwidth) | (rp[:, 0] > DOM_HI - bwidth)
           | (rp[:, 1] < DOM_LO + bwidth) | (rp[:, 1] > DOM_HI - bwidth))
    return real_disp, bnd
