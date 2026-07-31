"""tyssue_diag -- geometric diagnostics for the 3D monolayer shell.

hollow_metric: detect "hollow" cells -- the ones that render as grey exposed lateral walls. A smoothly
tiled apical cap agrees with its edge-neighbours' orientation; a folded / tilted / inverted cap (or a
tiny degenerate daughter) is what shows its wall. The score is the angle between a face's outward
normal and its edge-neighbours' mean normal, plus a tiny-area flag. Built as a DIAGNOSTIC (validated
against the render: red score == visible grey) so fixes can be measured, not guessed.

THREE FAILURE MODES, NOT ONE (mesh_faults).  `hollow_flags` used to OR three unrelated events into a
single fraction:

    hollow = (devd > dev_deg) | tiny | (ndeg < 3)

  * FOLDED   -- the cap's normal disagrees with its neighbours'. Geometry warping. The shell is still
                a shell; the physics is still integrable. Ugly, usually recoverable.
  * SLIVER   -- area far below the local neighbour mean. Frequently BENIGN: a cell that just divided,
                or one being squeezed by a T1, is briefly small and then relaxes.
  * BROKEN   -- the face is no longer a face. Under-connected (fewer than three edge-neighbours) or
                its vertex ring is not a valid polygon. This is what invalidates the physics.

Blending them produced numbers nobody could act on: an archived run reports hollow_frac_peak = 0.966
(log/okuda/r01_02_4af688) and the number cannot say whether 97% of the cells are slightly bent or 97%
are destroyed. Measured on a stressed 60-frame vesicle run, the blend reads 0.0094 at one frame
(100% slivers, nothing folded, nothing broken) and 0.1901 at another (100% folded, no slivers, nothing
broken) -- two completely different states reported on the same axis. The modes are now counted and
reported SEPARATELY; `hollow` is kept, bit-identical, purely so archived numbers stay comparable.
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


def ring_valid_3d(es, ef, nF, area, amin=1e-4):
    """Vectorised 3D counterpart of tyssue_topology_ops.ring_valid: a face is a VALID POLYGON iff it
    has >= 3 vertices, no repeated vertex, and positive area.

    WHY a port and not an import. `ring_valid` is the real validity test the 2D runners
    (run_tyssue_apoptosis / _flow / _growth / _division) have always used, and it was never wired
    into the 3D path -- the third guard found written and not installed where the danger was. But it
    CANNOT be imported as-is: its area comes from `ring_signed_area`, a shoelace over (x, y) only.
    Projected onto the xy-plane, half of a closed sphere's faces wind the other way and have NEGATIVE
    projected area. Measured on the healthy seed mesh here, the 2D predicate calls ~50% of faces
    invalid at every frame -- it would have replaced one meaningless number with a worse one. The
    predicate is therefore reproduced exactly with the 3D Newell area already computed by
    `face_normals`, which is orientation-correct on a closed shell.

    Vectorised because this runs on EVERY recorded frame: a bincount for the ring lengths and one
    sort for the repeated-vertex test, both O(E), against the O(nF) Python loops hollow_metric
    already pays.
    """
    ring_len = np.bincount(ef, minlength=nF)[:nF]               # len(ring) >= 3
    dup = np.zeros(nF, bool)                                    # len(set(ring)) == len(ring)
    if es.size:
        stride = int(es.max()) + 1
        key = np.sort(ef.astype(np.int64) * stride + es.astype(np.int64))
        if key.size > 1:
            rep = key[1:][key[1:] == key[:-1]]                  # a vertex visited twice by one face
            if rep.size:
                dup[np.unique(rep // stride)] = True
    return (ring_len >= 3) & (~dup) & (area > amin)             # + ring_signed_area > amin


def mesh_faults(pos, mesh, dev_deg=50.0, tiny_frac=0.15):
    """The three mesh-failure modes as SEPARATE boolean masks -- see the module docstring.

    Returns a dict of per-face masks:
      folded  -- cap normal deviates > dev_deg from its edge-neighbours' mean (geometry warping);
      sliver  -- area far below the LOCAL neighbour mean (often benign: a just-divided daughter);
      broken  -- under-connected (< 3 edge-neighbours) OR not a valid polygon ring. ONLY this one
                 invalidates the physics: the face is no longer a face;
      hollow  -- the FROZEN legacy blend, folded | sliver | (ndeg < 3). Derived, back-compat only:
                 it deliberately keeps the OLD under-connected term rather than the new `broken`
                 test, so every archived hollow_frac stays comparable to a fresh run.
    """
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
    sliver = area < tiny_frac * np.maximum(locmean, 1e-9)       # local (not global) tiny-cell test
    folded = devd > dev_deg
    under = ndeg < 3                                            # too few edge-neighbours to be a cell
    invalid = ~ring_valid_3d(es, ef, nF, area)                  # ring is not a polygon at all
    return dict(folded=folded, sliver=sliver, broken=under | invalid, under=under, invalid=invalid,
                hollow=folded | sliver | under,                 # FROZEN legacy blend -- do not "improve"
                dev=devd, area=area, ndeg=ndeg)


def hollow_flags(pos, mesh, dev_deg=50.0, tiny_frac=0.15):
    """Boolean per-cell 'hollow' + a [0,1] score for colouring + a stats dict.

    `hollow` / `score` / stats["frac"] / stats["n"] / stats["n_tiny"] are BIT-IDENTICAL to the
    pre-split behaviour so the archive and the renderers keep meaning what they meant. The stats
    dict now also carries the three modes separately (folded / sliver / broken); read THOSE. The
    blend cannot tell a fifth of the cells being slightly bent from a fifth being destroyed.
    """
    f = mesh_faults(pos, mesh, dev_deg=dev_deg, tiny_frac=tiny_frac)
    folded, sliver, broken, hollow, devd = f["folded"], f["sliver"], f["broken"], f["hollow"], f["dev"]
    nF = max(len(hollow), 1)
    score = np.clip(devd / 70.0, 0.0, 1.0); score[sliver | f["under"]] = 1.0
    return hollow, score, dict(frac=float(hollow.mean()), n=int(hollow.sum()),
                               n_tiny=int(sliver.sum()), dev_mean=float(devd.mean()),
                               dev_p90=float(np.percentile(devd, 90)),
                               # the three modes, separately -- this is what to act on
                               n_folded=int(folded.sum()), frac_folded=float(folded.sum()) / nF,
                               n_sliver=int(sliver.sum()), frac_sliver=float(sliver.sum()) / nF,
                               n_broken=int(broken.sum()), frac_broken=float(broken.sum()) / nF,
                               n_under=int(f["under"].sum()), n_invalid_ring=int(f["invalid"].sum()))

def mesh_genus(mesh):
    """Euler characteristic and genus of the closed shell. THE DISCOVERY-OR-BUG TEST.

    V - E + F = 2 - 2g.  A sphere has g=0 (chi=2); a torus has g=1 (chi=0).

    WHY THIS EXISTS. Cedric asked what happens if the loop produces something Okuda never
    described -- a torus, say. Tracing it: the elongation metric would read a torus as an
    unremarkable bump (p95(r)/median(r) ~ (R+r)/R), the tube counter would report zero, and the
    scoreboard had no row for it. Nothing anywhere computed the topology.

    That matters more than a missing row, because NO OPERATOR IN THIS SUBSTRATE CAN FUSE TWO
    SURFACES. Division, reconnection, apoptosis and growth all preserve genus. So a handle cannot
    be created legally -- and therefore a torus-shaped result is far more likely to be a CORRUPTED
    MESH than a discovery. Without this function, serendipity and corruption are indistinguishable,
    and the exciting interpretation is the one a tired reader will pick.

    So: measure it. If the genus changed, that is a bug report, not a phenotype -- unless and until
    an operator exists that is allowed to change it. If the genus is unchanged, the odd shape is a
    real deformation (a deep invagination that nearly closes looks like a torus and is still a
    sphere) and IS worth chasing.
    """
    es = np.asarray(mesh["E_srce"]); et = np.asarray(mesh["E_trgt"]); ef = np.asarray(mesh["E_face"])
    nF = int(mesh["nF"])
    live = ef < nF
    es, et = es[live], et[live]
    V = len(np.unique(np.concatenate([es, et])))
    undirected = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(es, et)}
    E = len(undirected)
    F = nF
    chi = V - E + F
    g2 = 2 - chi
    return {"V": int(V), "E": int(E), "F": int(F), "euler": int(chi),
            "genus": (g2 // 2 if g2 % 2 == 0 else None),
            "closed_sphere": bool(chi == 2),
            "verdict": ("sphere (as built)" if chi == 2 else
                        f"NOT a sphere: chi={chi}. No operator here can change genus, so this is "
                        f"a MESH BUG until proven otherwise -- not a new morphology.")}
