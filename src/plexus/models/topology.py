"""Half-edge ALGORITHMS on the closed 3D surface: rings, edge split, face division.

Moved out of `discovery_okuda/ops/topology_ops.py`, and moved to `models/` rather than `operators/`
because none of it is an operator: these are pure functions over the flat table that
`plexus.models.mesh.MeshTable` holds, and `cell_divide` / `edge_flip` are the operators that drive
them. Keeping them beside the table is what makes the table's central invariant checkable in one
place -- `rings_from_flat_3d` walks `E_face` IN TABLE ORDER and never sorts, so the ordering of the
flat table is the geometry.
"""
from __future__ import annotations
import numpy as np


# ==========================================================================================================
# FROM `discovery_okuda/ops/topology_ops.py` -- topology_ops -- 3D face DIVISION on the closed spherical half-edge mesh (the 3D sibling of
# ==========================================================================================================
def rings_from_flat_3d(es, et, ef, nF):
    """Flat half-edge table -> list of per-face vertex rings (ordered). build_sphere_mesh emits each
    face's half-edges contiguously in ring order, and divide preserves that, so grouping src vertices
    by face recovers the CCW ring."""
    rings = [[] for _ in range(nF)]
    for k in range(len(ef)):
        rings[int(ef[k])].append(int(es[k]))
    return rings


def flat_from_rings_3d(rings):
    """Compact (drop dead faces), renumber, and emit the flat half-edge table + the surviving-face
    index map (new_face -> old_face) so per-face targets can be carried over."""
    es, et, ef, keep = [], [], [], []
    for f, r in enumerate(rings):
        if r is None or len(r) < 3:
            continue
        nf = len(keep); keep.append(f)
        k = len(r)
        for i in range(k):
            es.append(r[i]); et.append(r[(i + 1) % k]); ef.append(nf)
    return (np.array(es, np.int64), np.array(et, np.int64), np.array(ef, np.int64),
            len(keep), np.array(keep, np.int64))


def _edge_face_map(rings):
    """Directed edge (u,v) -> face that traverses it. On a closed orientable surface the opposite
    directed edge (v,u) belongs to the one neighbouring face."""
    m = {}
    for f, r in enumerate(rings):
        if r is None or len(r) < 3:
            continue
        k = len(r)
        for i in range(k):
            m[(r[i], r[(i + 1) % k])] = f
    return m


def _insert_after(ring, u, v, w):
    """Insert vertex w between consecutive u,v in ring (in place). Returns True on success."""
    k = len(ring)
    for i in range(k):
        if ring[i] == u and ring[(i + 1) % k] == v:
            ring.insert(i + 1, w)
            return True
    return False


def divide_face_3d(rings, pos, f, project=True, ea=None, eb=None):
    """Divide face f by an edge-midpoint septum across edges `ea` and `eb` (pass the two edges the
    cell's SHORT axis crosses -> the septum runs perpendicular to the long axis, Hertwig's rule ->
    compact daughters). Defaults to roughly-opposite edges. Mutates `rings` (f -> two daughters,
    neighbours gain the midpoint vertices) and appends the two new vertex positions to `pos`.
    Returns (idx_daughterB, m1, m2) -- daughter A stays at index f, B is appended. None if it can't."""
    r = rings[f]
    k = len(r)
    if k < 4:                                   # need >=4 edges to split two non-adjacent ones cleanly
        return None
    if ea is None:
        ea, eb = 0, k // 2
    i0, i1 = sorted((ea % k, eb % k))
    if i1 - i0 < 2 or (i0 + k - i1) < 2:        # adjacent edges -> a daughter would be degenerate; use opposite
        i0, i1 = 0, k // 2
    a0, a1 = r[i0], r[(i0 + 1) % k]
    b0, b1 = r[i1], r[(i1 + 1) % k]
    emap = _edge_face_map(rings)
    nbrA = emap.get((a1, a0)); nbrB = emap.get((b1, b0))
    if nbrA is None or nbrB is None or nbrA == f or nbrB == f or nbrA == nbrB:
        return None                             # both split edges must have distinct interior neighbours

    m1 = len(pos); m2 = len(pos) + 1
    p_a = 0.5 * (np.asarray(pos[a0]) + np.asarray(pos[a1]))
    p_b = 0.5 * (np.asarray(pos[b0]) + np.asarray(pos[b1]))
    if project:                                 # push the midpoint back out to the local shell radius
        ra = 0.5 * (np.linalg.norm(pos[a0]) + np.linalg.norm(pos[a1]))
        rb = 0.5 * (np.linalg.norm(pos[b0]) + np.linalg.norm(pos[b1]))
        p_a = p_a * (ra / max(np.linalg.norm(p_a), 1e-9))
        p_b = p_b * (rb / max(np.linalg.norm(p_b), 1e-9))
    pos.append(p_a); pos.append(p_b)

    # split the two shared edges inside the neighbouring faces (keeps every edge shared by two faces)
    if not _insert_after(rings[nbrA], a1, a0, m1):
        return None
    if not _insert_after(rings[nbrB], b1, b0, m2):
        return None

    # daughter A: m1 -> (r[i0+1 .. i1]) -> m2, closed by the septum m2->m1
    seg_A = [m1] + [r[j % k] for j in range(i0 + 1, i1 + 1)] + [m2]
    # daughter B: m2 -> (r[i1+1 .. i0]) -> m1, closed by the septum m1->m2
    seg_B = [m2] + [r[j % k] for j in range(i1 + 1, i0 + 1 + k)] + [m1]
    rings[f] = seg_A
    rings.append(seg_B)
    return len(rings) - 1, m1, m2


# --------------------------------------------------------------------------------------------------
def face_collapse_3d(rings, pos, f):
    """T2 / cell extrusion: collapse triangular face `f` to a point, so the sheet closes over it.

    THE INVERSE OF `divide_face_3d`, and the operator family the algebra was missing. Plexus2 lists
    eight elementary families and Die -- "removal of biological entities" -- is one of them; every
    operator this vesicle owns deforms the sheet OUTWARD (growth inflates, division subdivides,
    the purse-string measured inert, extrusion is the disqualified forcing term), so there was no
    mechanism for inward deformation at all.

    WHY IT IS SOUND, and why the caller must shed neighbours first. On a closed trivalent surface,
    collapsing a TRIANGLE removes 2 vertices, 3 edges and 1 face, so

        chi = V - E + F  ->  (V-2) - (E-3) + (F-1) = V - E + F

    is unchanged and the sheet stays closed at genus 0. Collapsing anything with more than three
    sides would merge k > 3 vertices into one and leave a rosette -- a vertex of degree > 3 that
    the trivalent mechanics cannot represent. `edge_flip` is what walks a shrinking cell down
    to a triangle, one short edge at a time, exactly as the 2D `apoptosis` relies on
    `t1_transition`; this function refuses anything else rather than doing it badly.

    Mutates `rings` (f retired, every reference to its vertices rewired) and `pos` (the surviving
    vertex moved to the centroid). Returns True if the collapse happened AND left a valid closed
    surface -- on any failure `rings` and `pos` are left untouched, because a half-applied topology
    edit is worse than a refused one.
    """
    r = rings[f]
    if r is None or len(r) != 3:
        return False
    keep, drop = int(r[0]), {int(v) for v in r[1:]}
    c = pos[list(r)].mean(0)
    # WORK ON A COPY AND COMMIT ONLY IF CLOSED. `_check_closed` is cheap next to a frame of
    # mechanics, and a mesh that has lost its closure does not announce itself: the genus check is
    # combinatorial and cannot see a surface folded through its own centre, which is how premise
    # P11 came to be written.
    trial = [None if rg is None else list(rg) for rg in rings]
    trial[f] = None
    for g, rg in enumerate(trial):
        if rg is None:
            continue
        ng = [keep if int(u) in drop else int(u) for u in rg]
        ded = [ng[i] for i in range(len(ng)) if ng[i] != ng[(i - 1) % len(ng)]]
        # a neighbour that shared TWO of the collapsed vertices loses a side; below three it is
        # degenerate and the collapse is refused rather than silently deleting a second cell
        if len(ded) < 3:
            return False
        trial[g] = ded
    ok, _V, _E, _F, euler = _check_closed(trial)
    if not ok or euler != 2:
        return False
    for g in range(len(rings)):
        rings[g] = None if trial[g] is None else np.asarray(trial[g], dtype=np.int64)
    pos[keep] = c
    return True


def _check_closed(rings):
    """Validate a closed orientable surface: every directed edge appears once, its opposite exists,
    no duplicate vertices within a ring. Returns (ok, V, E, F, euler)."""
    seen = {}
    verts = set()
    F = 0
    for r in rings:
        if r is None or len(r) < 3:
            continue
        F += 1
        if len(set(r)) != len(r):
            return False, 0, 0, 0, 0
        k = len(r)
        for i in range(k):
            e = (r[i], r[(i + 1) % k])
            if e in seen:
                return False, 0, 0, 0, 0            # a directed edge used twice -> non-manifold
            seen[e] = True
            verts.add(r[i])
    for (u, v) in seen:
        if (v, u) not in seen:
            return False, 0, 0, 0, 0                # boundary edge -> not closed
    V = len(verts); E = len(seen) // 2
    return True, V, E, F, V - E + F


if __name__ == "__main__":                          # standalone soundness test (no engine)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mesh_ops import build_sphere_mesh

    verts, es, et, ef, nF = build_sphere_mesh(120, 5.0, 0.15, 0)
    rings = rings_from_flat_3d(es, et, ef, nF)
    pos = [v for v in verts]
    ok, V, E, F, eu = _check_closed(rings)
    print(f"start: closed={ok} V={V} E={E} F={F} euler={eu}")

    rng = np.random.default_rng(0)
    ndone = 0
    for _ in range(40):
        f = int(rng.integers(0, len(rings)))
        if rings[f] is None or len(rings[f]) < 4:
            continue
        res = divide_face_3d(rings, pos, f)
        if res is not None:
            ndone += 1
    ok, V, E, F, eu = _check_closed(rings)
    print(f"after {ndone} divisions: closed={ok} V={V} E={E} F={F} euler={eu}  (want euler=2)")
    es2, et2, ef2, nF2, keep = flat_from_rings_3d(rings)
    print(f"rebuilt flat: nF={nF2} half-edges={len(es2)} (each real edge twice -> {len(es2)//2} edges)")
