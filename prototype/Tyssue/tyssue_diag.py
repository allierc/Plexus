"""tyssue_diag -- geometric diagnostics for the 3D monolayer shell.

hollow_metric: detect "hollow" cells -- the ones that render as grey exposed lateral walls. A smoothly
tiled apical cap agrees with its edge-neighbours' orientation; a folded / tilted / inverted cap (or a
tiny degenerate daughter) is what shows its wall. The score is the angle between a face's outward
normal and its edge-neighbours' mean normal, plus a tiny-area flag. Built as a DIAGNOSTIC (validated
against the render: red score == visible grey) so fixes can be measured, not guessed.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from tyssue_topology_ops3d import rings_from_flat_3d


def face_normals(pos, rings, nF):
    """Unit (outward) Newell normal + area + centroid per face, from its ordered vertex ring."""
    n = np.zeros((nF, 3)); area = np.zeros(nF); cen = np.zeros((nF, 3))
    for f in range(nF):
        r = rings[f]
        if r is None or len(r) < 3:
            continue
        V = pos[r]; c = V.mean(0); cen[f] = c
        N = np.zeros(3)
        for i in range(len(r)):
            N += np.cross(V[i] - c, V[(i + 1) % len(r)] - c)
        area[f] = 0.5 * np.linalg.norm(N)
        n[f] = N / (np.linalg.norm(N) + 1e-12)
    return n, area, cen


def hollow_metric(pos, mesh):
    """Per-cell hollow diagnostics. Returns (dev, area, ndeg):
      dev  -- angle (radians) between the face normal and its edge-neighbours' mean normal
              (a folded/inverted cap deviates strongly; a smooth cap ~0);
      area -- face area (tiny == a degenerate / just-divided daughter);
      ndeg -- number of edge-neighbours.
    """
    es = np.asarray(mesh["E_srce"]); et = np.asarray(mesh["E_trgt"]); ef = np.asarray(mesh["E_face"])
    nF = int(mesh["nF"]); rings = rings_from_flat_3d(es, et, ef, nF)
    n, area, _ = face_normals(pos, rings, nF)
    nbr = defaultdict(set); byedge = defaultdict(list)
    for k in range(len(ef)):
        byedge[(min(int(es[k]), int(et[k])), max(int(es[k]), int(et[k])))].append(int(ef[k]))
    for faces in byedge.values():
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                if faces[i] != faces[j]:
                    nbr[faces[i]].add(faces[j]); nbr[faces[j]].add(faces[i])
    dev = np.zeros(nF); ndeg = np.zeros(nF, int)
    for f in range(nF):
        ns = list(nbr[f]); ndeg[f] = len(ns)
        if not ns:
            continue
        mn = n[ns].mean(0); mn /= (np.linalg.norm(mn) + 1e-12)
        dev[f] = np.arccos(np.clip(float(n[f] @ mn), -1.0, 1.0))
    return dev, area, ndeg


def hollow_flags(pos, mesh, dev_deg=50.0, tiny_frac=0.15):
    """Boolean per-cell 'hollow' + a [0,1] score for colouring. A cell is hollow if its cap is folded
    (dev > dev_deg), degenerate-small (area < tiny_frac x median), or under-connected (deg < 3)."""
    dev, area, ndeg = hollow_metric(pos, mesh)
    devd = np.degrees(dev)
    # TUBE-AWARE "tiny": compare each cell's area to its EDGE-NEIGHBOURS' mean (local scale), not the GLOBAL
    # median. A thin tube's cells are uniformly small (vs the big body cells) -> a global test flags the whole
    # wall as false-positive "hollow"; a local test flags only a genuine sliver (much smaller than its
    # neighbours). Vectorised via the half-edge twin map.
    es = np.asarray(mesh["E_srce"]); et = np.asarray(mesh["E_trgt"]); ef = np.asarray(mesh["E_face"]); nF = int(mesh["nF"])
    Nv = int(max(es.max(), et.max())) + 1
    key = es * Nv + et; order = np.argsort(key); ks = key[order]
    p = np.searchsorted(ks, et * Nv + es).clip(max=len(key) - 1)
    tw = np.where(ks[p] == et * Nv + es, ef[order[p]], ef)      # neighbour face across each half-edge
    nsum = np.zeros(nF); np.add.at(nsum, ef, area[tw]); ncnt = np.zeros(nF); np.add.at(ncnt, ef, (tw != ef).astype(float))
    locmean = np.where(ncnt > 0, nsum / np.maximum(ncnt, 1), area)
    tiny = area < tiny_frac * np.maximum(locmean, 1e-9)         # local (not global) tiny-cell test
    hollow = (devd > dev_deg) | tiny | (ndeg < 3)
    score = np.clip(devd / 70.0, 0.0, 1.0); score[tiny | (ndeg < 3)] = 1.0
    return hollow, score, dict(frac=float(hollow.mean()), n=int(hollow.sum()),
                               n_tiny=int(tiny.sum()), dev_mean=float(devd.mean()),
                               dev_p90=float(np.percentile(devd, 90)))
