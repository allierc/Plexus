"""tyssue_t1_ops3d -- surface T1 NEIGHBOUR-EXCHANGE (reversible network reconnection) on the closed
spherical half-edge mesh (the epithelial vesicle). The 3D sibling of tyssue_topology_ops.t1_flip: the
topological rewire is identical on a surface (a local relabel of four face rings), only the geometry
(place the split vertices in the tangent plane, project back onto the shell) and the validity tests
(closed-surface manifold check + per-face SIMPLE test in 3D) differ.

T1 on an interior edge e=(u,v):  e is shared by A (...u->v...) and B (...v->u...). Around it four faces
meet: A, B, and the third faces C (at u) and D (at v).  A trivalent-vertex T1 collapses u,v to a point
(a transient 4-valent vertex whose faces are A,C,B,D in cyclic order) and re-splits perpendicular so
the new edge separates C and D (the previously non-adjacent cells).  On the ring representation:

    A loses v      A=[..p,u,v,s..] -> [..p,u,s..]      (edge u->s new)
    B loses u      B=[..t,v,u,q..] -> [..t,v,q..]      (edge v->q new)
    C gains v       C=[..q,u,p..]  -> [..q,v,u,p..]    (v inserted before u; C now v->u)
    D gains u       D=[..s,v,t..]  -> [..s,u,v,t..]    (u inserted before v; D now u->v)

so the new edge u-v is shared by C and D -> C,D become neighbours, A,B stop being neighbours. V,E,F are
ALL unchanged (the edge merely rotates 90 deg), so a T1 keeps Euler=2. The mirror chirality (insert
AFTER instead of before) is the other, geometrically opposite, T1; both are closed, so the flip TRIES
both chiralities x both perpendicular signs and commits the first that leaves all four faces simple and
outward-facing -- otherwise it refuses (no-op), which is what keeps autonomous runs from tangling.

Pure functions here; the plexus operator `reconnect_t1_3d` drives them (mirrors t1_transition in 2D).
"""
from __future__ import annotations

import numpy as np
import torch

from plexus.models.base import Rewire
from plexus.models.registry import register_operator

from tyssue_topology_ops3d import (rings_from_flat_3d, flat_from_rings_3d,
                                    _edge_face_map, _check_closed)


# --------------------------------------------------------------------------------------------------
#  ring helpers
# --------------------------------------------------------------------------------------------------
def _vertex_faces(rings):
    """vertex -> set of faces incident to it (used to find the third face at u / at v)."""
    vf = {}
    for f, r in enumerate(rings):
        if r is None or len(r) < 3:
            continue
        for w in r:
            vf.setdefault(w, set()).add(f)
    return vf


def _insert_before(ring, anchor, w):
    """New ring with w inserted immediately BEFORE the first occurrence of anchor (or None)."""
    out, done = [], False
    for x in ring:
        if x == anchor and not done:
            out.append(w); done = True
        out.append(x)
    return out if done else None


def _insert_after(ring, anchor, w):
    """New ring with w inserted immediately AFTER the first occurrence of anchor (or None)."""
    out, done = [], False
    for x in ring:
        out.append(x)
        if x == anchor and not done:
            out.append(w); done = True
    return out if done else None


def _ring_ok(r):
    return r is not None and len(r) >= 3 and len(set(r)) == len(r)


# --------------------------------------------------------------------------------------------------
#  local closed-surface (manifold) check on the 4 changed faces
# --------------------------------------------------------------------------------------------------
def _boundary_de(faces_map):
    """Directed edges of a face patch; returns the set of BOUNDARY directed edges (those whose reverse
    is not in the patch), or None if any directed edge repeats inside the patch (-> non-manifold)."""
    de = set()
    for r in faces_map.values():
        k = len(r)
        for i in range(k):
            e = (r[i], r[(i + 1) % k])
            if e in de:
                return None                                  # directed edge used twice -> non-manifold
            de.add(e)
    return set(e for e in de if (e[1], e[0]) not in de)


def _local_manifold_ok(old_map, new_map):
    """A T1 only touches the four faces {A,B,C,D}; every changed directed edge stays inside that patch.
    So the mesh stays closed iff the patch mates with the (unchanged) exterior exactly as before, i.e.
    the patch's boundary directed edges are unchanged -- and no directed edge repeats in the new patch."""
    b_old = _boundary_de(old_map); b_new = _boundary_de(new_map)
    return b_new is not None and b_old is not None and b_new == b_old


# --------------------------------------------------------------------------------------------------
#  3D per-face validity: non-degenerate, outward-facing, and SIMPLE (no self-crossing)
# --------------------------------------------------------------------------------------------------
def _seg_cross(p1, p2, p3, p4):
    """Proper 2D segment intersection (touching endpoints do NOT count)."""
    ccw = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = ccw(p3, p4, p1); d2 = ccw(p3, p4, p2); d3 = ccw(p1, p2, p3); d4 = ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _polygon_simple_2d(Q):
    """True iff the 2D polygon Q has no two NON-adjacent edges properly crossing (catches bow-ties)."""
    k = len(Q)
    for i in range(k):
        for j in range(i + 1, k):
            if j == i + 1 or (i == 0 and j == k - 1):
                continue                                     # adjacent edges share a vertex
            if _seg_cross(Q[i], Q[(i + 1) % k], Q[j], Q[(j + 1) % k]):
                return False
    return True


def _face_ok_3d(ring, getp):
    """Face is valid: >=3 verts, non-zero area, Newell normal points OUTWARD (dot with centroid > 0),
    and the polygon is SIMPLE when projected onto its own plane. `getp(i)` returns vertex i's 3-vector."""
    P = np.array([np.asarray(getp(i), float) for i in ring])
    k = len(P)
    if k < 3:
        return False
    c = P.mean(0)
    N = 0.5 * np.cross(P, np.roll(P, -1, 0)).sum(0)          # Newell area vector (|N| = area)
    a = np.linalg.norm(N)
    if a < 1e-9 or float(np.dot(N, c)) <= 0.0:               # degenerate, or inward-facing
        return False
    n = N / a                                                # project to the face plane, test simplicity
    e1 = P[0] - c; e1 = e1 - np.dot(e1, n) * n
    if np.linalg.norm(e1) < 1e-9:
        e1 = P[1] - c; e1 = e1 - np.dot(e1, n) * n
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n, e1)
    Q = np.stack([(P - c) @ e1, (P - c) @ e2], 1)
    return _polygon_simple_2d(Q)


# --------------------------------------------------------------------------------------------------
#  the T1 flip
# --------------------------------------------------------------------------------------------------
def t1_flip_3d(rings, pos, e_uv, new_len=None, emap=None, vf=None):
    """One surface T1 on interior edge e_uv=(u,v): rewire the four rings A,B,C,D and move u,v apart
    along the tangent-plane perpendicular (projected onto the shell). Mutates `rings` and pos[u],pos[v]
    in place and returns (u,v) on success; returns None (no-op) if the flip is impossible or would break
    an invariant: boundary edge, non-trivalent u/v, C==D, a face < 3 verts, a duplicated vertex in a
    ring, a broken closed-surface, or a non-simple/inward face for BOTH chiralities x BOTH signs.
    OPTIMISATION: pass `emap` (edge->face) and `vf` (vertex->faces) built ONCE by the caller; a successful
    flip updates them in place (only the 4 faces A,B,C,D change), avoiding an O(F) rebuild per candidate."""
    u, v = int(e_uv[0]), int(e_uv[1])
    if emap is None:
        emap = _edge_face_map(rings)
    A = emap.get((u, v)); B = emap.get((v, u))
    if A is None or B is None or A == B:
        return None                                          # boundary / degenerate interior edge
    if vf is None:
        vf = _vertex_faces(rings)
    Cs = vf.get(u, set()) - {A, B}; Ds = vf.get(v, set()) - {A, B}
    if len(Cs) != 1 or len(Ds) != 1:
        return None                                          # u or v not trivalent -> no clean T1
    C = Cs.pop(); D = Ds.pop()
    if C == D or len({A, B, C, D}) != 4:
        return None                                          # the two "far" cells must be distinct
    rA, rB, rC, rD = rings[A], rings[B], rings[C], rings[D]

    nA = [w for w in rA if w != v]                           # A loses v
    nB = [w for w in rB if w != u]                           # B loses u
    if not (_ring_ok(nA) and _ring_ok(nB)):
        return None                                          # would leave a face with < 3 verts

    # geometry: collapse u,v to their midpoint then reopen perpendicular in the local tangent plane
    pu = np.asarray(pos[u], float); pv = np.asarray(pos[v], float)
    mid = 0.5 * (pu + pv); rmid = np.linalg.norm(mid)
    d = pv - pu; L = np.linalg.norm(d)
    if L < 1e-12 or rmid < 1e-9:
        return None
    n = mid / rmid                                           # outward radial (shell normal)
    perp = np.cross(n, d / L); pn = np.linalg.norm(perp)
    if pn < 1e-9:                                            # edge is radial -> perpendicular undefined
        return None
    perp = perp / pn
    half = 0.5 * (new_len if new_len is not None else L)     # reopen at ~ new_len (default: same length)
    rm = 0.5 * (np.linalg.norm(pu) + np.linalg.norm(pv))     # target shell radius for both split verts

    old_map = {A: rA, B: rB, C: rC, D: rD}
    #   two chiralities: insert v/u BEFORE (v->u in C) or AFTER (u->v in C) -- both are closed T1s;
    #   geometry (simple + outward) picks the correct, non-folding one.
    for insert in (_insert_before, _insert_after):
        nC = insert(rC, u, v)                                # C gains v next to u
        nD = insert(rD, v, u)                                # D gains u next to v
        if not (_ring_ok(nC) and _ring_ok(nD)):
            continue
        new_map = {A: nA, B: nB, C: nC, D: nD}
        if not _local_manifold_ok(old_map, new_map):
            continue                                         # would break the closed surface
        for sign in (+1.0, -1.0):
            nu = mid - sign * perp * half; nv = mid + sign * perp * half
            nu = nu * (rm / (np.linalg.norm(nu) + 1e-12))    # back onto the shell
            nv = nv * (rm / (np.linalg.norm(nv) + 1e-12))
            getp = lambda i: (nu if i == u else nv if i == v else pos[i])
            if all(_face_ok_3d(r, getp) for r in (nA, nB, nC, nD)):
                for fid, ro, rn in ((A, rA, nA), (B, rB, nB), (C, rC, nC), (D, rD, nD)):
                    for i in range(len(ro)):                # keep the passed maps in sync: only A,B,C,D changed
                        emap.pop((ro[i], ro[(i + 1) % len(ro)]), None)
                    for i in range(len(rn)):
                        emap[(rn[i], rn[(i + 1) % len(rn)])] = fid
                    for w in ro:
                        s = vf.get(w)
                        if s is not None:
                            s.discard(fid)
                    for w in rn:
                        vf.setdefault(w, set()).add(fid)
                rings[A] = nA; rings[B] = nB; rings[C] = nC; rings[D] = nD
                pos[u] = nu; pos[v] = nv
                return (u, v)
    return None


# --------------------------------------------------------------------------------------------------
#  plexus operator
# --------------------------------------------------------------------------------------------------
@register_operator("reconnect_t1_3d", set="vertex", kind="rewire", family="topology")
class ReconnectT1_3D(Rewire):
    """Surface T1 reconnection on the vesicle: flip every interior edge shorter than the threshold by a
    local neighbour exchange, committing only valid flips. The 3D sibling of t1_transition -- explicit
    intercalation on the closed half-edge shell, the ingredient a re-tessellated (spherical-Voronoi)
    route cannot have. Keeps V,E,F (and Euler=2) fixed; only reconnects and repositions the two verts."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["T1_transition", "reversible_network_reconnection", "intercalation",
                      "neighbour_exchange", "vertex_model", "vesicle", "surface"]
    PARAM_ROLES = {"l_th": "reconnection_threshold_length (absolute; 0 -> use l_th_frac)",
                   "l_th_frac": "threshold as fraction of the mean edge length",
                   "max_flips": "cap on reconnections per call", "every": "call period (ticks)"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.l_th = float(params.get("l_th", 0.0))           # absolute; <=0 -> l_th_frac x mean edge
        self.l_th_frac = float(params.get("l_th_frac", 0.15))
        self.max_flips = int(params.get("max_flips", 20))
        from tyssue_ops3d import _engine_owns_clock
        self.every = _engine_owns_clock(params); self._k = 0

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        self._k += 1                    # monotonic tick only -- D1: the engine owns the period
        dev = lvl.state.device; dt = lvl.state.dtype
        Nv = int(m["Nv"])
        pos_np = lvl.get("pos")[:Nv].detach().cpu().numpy().astype(np.float64)
        es = m["E_srce"].detach().cpu().numpy(); et = m["E_trgt"].detach().cpu().numpy()
        ef = m["E_face"].detach().cpu().numpy(); nF = int(m["nF"])
        rings = rings_from_flat_3d(es, et, ef, nF)
        emap = _edge_face_map(rings); vf = _vertex_faces(rings)   # build the adjacency maps ONCE; t1_flip_3d
        #   updates them incrementally on each successful flip (was rebuilt O(F) per candidate = the hot spot)
        pos = [p.copy() for p in pos_np]
        length = np.linalg.norm(pos_np[et] - pos_np[es], axis=1)
        thr = self.l_th if self.l_th > 0 else self.l_th_frac * float(length.mean())
        order = np.argsort(length)                           # shortest interior edges first
        seen = set(); used = set(); ndone = 0                # `used` verts -> one flip per vertex / call
        for k in order:
            if ndone >= self.max_flips or length[k] >= thr:
                break
            a, b = int(es[k]), int(et[k]); key = (min(a, b), max(a, b))
            if key in seen or a in used or b in used:
                continue
            seen.add(key)
            if t1_flip_3d(rings, pos, (a, b), new_len=thr, emap=emap, vf=vf) is not None:
                used.add(a); used.add(b); ndone += 1
        if ndone == 0:
            return {}
        es2, et2, ef2, nF2, _ = flat_from_rings_3d(rings)    # T1 drops no face -> nF2 == nF, order kept
        m["E_srce"] = torch.as_tensor(es2, device=dev)
        m["E_trgt"] = torch.as_tensor(et2, device=dev)
        m["E_face"] = torch.as_tensor(ef2, device=dev)
        m["nF"] = nF2
        px0, px1 = lvl.state_schema["pos"]                   # write the two moved verts back (like divide_3d)
        st = lvl.state.clone()
        st[:Nv, px0:px1] = torch.as_tensor(np.asarray(pos), dtype=dt, device=dev)
        lvl.state = st
        m["n_t1"] = int(m.get("n_t1", 0)) + ndone
        return {}


# --------------------------------------------------------------------------------------------------
#  standalone self-test (no engine): build a jittered vesicle, run many T1 flips, assert it stays
#  CLOSED with Euler=2 and that a T1 keeps V,E,F constant.
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tyssue_ops3d import build_sphere_mesh

    verts, es, et, ef, nF = build_sphere_mesh(150, 5.0, 0.15, 0)
    rings = rings_from_flat_3d(es, et, ef, nF)
    pos = [v.copy() for v in verts]
    ok0, V0, E0, F0, eu0 = _check_closed(rings)
    print(f"start:  closed={ok0} V={V0} E={E0} F={F0} euler={eu0}")

    def undirected_edges(rr):
        s = set()
        for r in rr:
            if r is None or len(r) < 3:
                continue
            k = len(r)
            for i in range(k):
                a, b = r[i], r[(i + 1) % k]; s.add((min(a, b), max(a, b)))
        return s

    ndone = 0
    for sweep in range(6):                                   # several sweeps: each flip reopens the edge
        E = sorted(undirected_edges(rings),                  #   at ~mean length so it exits the short set
                   key=lambda ab: np.linalg.norm(pos[ab[1]] - pos[ab[0]]))
        me = float(np.mean([np.linalg.norm(pos[b] - pos[a]) for a, b in E]))
        thr = 0.9 * me                                       # target the shorter ~half of the edges
        used = set(); fired = 0
        for (a, b) in E:
            if np.linalg.norm(pos[b] - pos[a]) >= thr:
                break
            if a in used or b in used:
                continue
            if t1_flip_3d(rings, pos, (a, b), new_len=me) is not None:  # reopen at ~mean length
                used.add(a); used.add(b); fired += 1
        ndone += fired
        ok, V, E_, F, eu = _check_closed(rings)
        print(f"sweep {sweep}: fired {fired:3d}  closed={ok} V={V} E={E_} F={F} euler={eu}")
        assert ok and eu == 2, "mesh broke the closed-surface invariant"

    ok, V, E, F, eu = _check_closed(rings)
    print(f"after {ndone} T1 flips:  closed={ok} V={V} E={E} F={F} euler={eu}  (want euler=2)")
    assert ok and eu == 2
    assert (V, E, F) == (V0, E0, F0), "a T1 must keep V,E,F constant"    # dV=dE=dF=0
    es2, et2, ef2, nF2, keep = flat_from_rings_3d(rings)
    print(f"rebuilt flat: nF={nF2} half-edges={len(es2)} (each real edge twice -> {len(es2)//2} edges)")
    assert ndone >= 20, "expected a meaningful number of flips"
    print(f"SELF-TEST OK  ({ndone} flips, closed, euler=2, V/E/F unchanged)")
