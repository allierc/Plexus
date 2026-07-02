"""embryo_metrics -- the Phase-1 SCIENTIFIC OBSERVABLES for the blastula loop.

Phase-1 objective (the user's brief): inner-core cell FLOW deforms the outer membrane, cells
NEVER collapse (a hard minimum cell-cell distance is maintained), motion stays bounded WITHOUT
relying on the velocity clamp (a parameter balance, not a hardcode), division progressively
deforms the blastula, cells keep FLOWING even at high density (collective migration emerges),
and two cell types PARTITION the blastula (e.g. left/right).

These are DIAGNOSTICS for hypothesis validation/falsification, not a single scalar loss. Read
from trajectory.npz (positions + occ + node_type). All lengths in world units.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def phase1_metrics(traj_path, r0=0.024, core_r=0.27, membrane_band=0.85):
    tr = np.load(traj_path)
    return phase1_from_arrays(tr["agent__pos"], tr["agent__occ"], tr["agent__node_type"],
                              tr["mpm_particle__pos"], r0=r0, membrane_band=membrane_band)


def phase1_from_arrays(ap, occ, nt, mp, r0=0.024, membrane_band=0.85):
    ap = np.asarray(ap); occ = np.asarray(occ); nt = np.asarray(nt); mp = np.asarray(mp)
    c = np.array([0.5, 0.5]); T = ap.shape[0]
    last = occ[-1] > 0
    P = ap[-1][last]; ntl = nt[last]

    # --- 1. COLLAPSE (hard constraint): min cell-cell distance must stay >= ~r0 ---
    if len(P) > 2:
        nn = cKDTree(P).query(P, k=2)[0][:, 1]
        nn_min = float(nn.min()); nn_mean = float(nn.mean())
        # truly-stacked cells (< 0.15*r0); loose enough that a just-divided daughter (offset ~0.2*r0)
        # is NOT flagged -- only real hydrodynamic collapse counts.
        collapsed = float((nn < 0.15 * r0).mean())
    else:
        nn_min = nn_mean = collapsed = 0.0

    # --- 2. MEMBRANE DEFORMATION driven by inner flow: outer-shell radial roughness vs t0 ---
    r0m = np.linalg.norm(mp[0] - c, axis=1); Rd = np.quantile(r0m, 0.99)
    mem = r0m > membrane_band * Rd                          # outer shell (membrane) particles, by identity
    def shell_profile(pos):
        rel = pos[mem] - c; ang = np.arctan2(rel[:, 1], rel[:, 0]); rr = np.linalg.norm(rel, axis=1)
        b = np.linspace(-np.pi, np.pi, 49); idx = np.digitize(ang, b)
        prof = np.array([rr[idx == k].mean() if (idx == k).any() else np.nan for k in range(1, 49)])
        return prof
    p0 = shell_profile(mp[0]); p1 = shell_profile(mp[-1])
    good = ~np.isnan(p0) & ~np.isnan(p1)
    deform = float(np.sqrt(np.nanmean((p1[good] - p0[good]) ** 2)))   # RMS membrane radial displacement

    # --- 3. FLOW / continuous motion (not jammed): mean cell speed over the last frames ---
    k = max(2, T // 10)
    v = np.diff(ap[-k:], axis=0)                            # [k-1, N, 2] per-frame displacement
    lm = occ[-1] > 0
    speed = float(np.linalg.norm(v[:, lm], axis=-1).mean())

    # --- 4. COLLECTIVE MIGRATION: polar order of the cell velocity field (last frames) ---
    vv = v[:, lm]; sp = np.linalg.norm(vv, axis=-1, keepdims=True)
    vhat = vv / np.clip(sp, 1e-9, None)
    migration = float(np.linalg.norm(vhat.mean(axis=(0, 1))))

    # --- 5. SEGREGATION (2 types -> left/right partition): |<x>_a - <x>_b| / R ---
    seg = 0.0
    if len(np.unique(ntl)) >= 2:
        xa = P[ntl == 0, 0].mean() if (ntl == 0).any() else 0.5
        xb = P[ntl == 1, 0].mean() if (ntl == 1).any() else 0.5
        seg = float(abs(xa - xb) / max(Rd, 1e-6))

    # --- 6. ACCELERATION (bounded WITHOUT the clamp?): 95th pct |a| over last frames ---
    if v.shape[0] >= 2:
        a = np.diff(v, axis=0)[:, lm]
        accel = float(np.percentile(np.linalg.norm(a, axis=-1), 95))
    else:
        accel = 0.0

    return dict(n_cells=int(last.sum()), collapsed=round(collapsed, 4), nn_min=round(nn_min, 4),
                nn_mean=round(nn_mean, 4), deform=round(deform, 4), flow=round(speed, 5),
                migration=round(migration, 4), segregation=round(seg, 4), accel=round(accel, 6),
                disc_R=round(float(Rd), 4))
