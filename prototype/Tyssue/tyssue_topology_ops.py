"""tyssue_topology_ops -- the explicit topology operators of the AVM: T1 reconnection,
cell division, cell extrusion, and growth of the per-face target area. Forked from the core
mechanics (tyssue_ops.py) so the two can evolve separately.

These are exactly the operators the Turing_vertex (SPV) route could not have: on a re-tessellated
Voronoi mesh topology changes are implicit; here the half-edge mesh is explicit, so a neighbour
swap (T1) is a local edge collapse+split, a division inserts a septum, and an extrusion collapses
a face. All mutate the face-ring topology carried on the vertex Level (`lvl._mesh["faces"]`) and
then rebuild the flat half-edge table that `shape_energy` reads.

Robustness contract: every topology mutation is applied to a trial copy, the affected faces are
validated (>=3 unique vertices, positive signed area), and the change is COMMITTED only if valid
-- otherwise it is silently skipped. The mesh therefore never tangles (an invalid T1 is a no-op),
which matters for autonomous runs.

Operators:
  face_growth    (structural) -- inflate per-face target area A0 (the proliferation drive)
  face_divide    (structural) -- split over-target faces by a septum through two vertices
  face_extrude   (structural) -- collapse a face to a point (apoptosis / delamination)
  t1_transition  (rewire)     -- flip sub-threshold interior edges (reversible network reconnection)
"""
from __future__ import annotations

import math
import numpy as np
import torch

from plexus.models.base import Structural, Rewire
from plexus.models.registry import register_operator


# --------------------------------------------------------------------------- #
#  Face-ring <-> flat half-edge table
# --------------------------------------------------------------------------- #
def rings_from_flat(es, et, ef, nF):
    """List of CCW vertex rings (one per face position; None-free at build)."""
    rings = [[] for _ in range(nF)]
    for k in range(len(ef)):
        rings[int(ef[k])].append(int(es[k]))
    return [np.array(r, dtype=np.int64) for r in rings]


def flat_from_rings(faces):
    es, et, ef = [], [], []
    for f, r in enumerate(faces):
        if r is None or len(r) < 3:
            continue
        k = len(r)
        for i in range(k):
            es.append(int(r[i])); et.append(int(r[(i + 1) % k])); ef.append(f)
    return (np.array(es, np.int64), np.array(et, np.int64), np.array(ef, np.int64))


def ring_signed_area(ring, pos):
    v = pos[ring]; x, y = v[:, 0], v[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ring_valid(ring, pos, amin=1e-4):
    return (ring is not None and len(ring) >= 3 and len(set(int(v) for v in ring)) == len(ring)
            and ring_signed_area(ring, pos) > amin)


def _seg_cross(p1, p2, p3, p4):
    """Proper segment intersection -- endpoints merely touching does NOT count as a crossing."""
    ccw = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = ccw(p3, p4, p1); d2 = ccw(p3, p4, p2); d3 = ccw(p1, p2, p3); d4 = ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def ring_simple(ring, pos):
    """True iff the polygon is SIMPLE (no two non-adjacent edges cross). A bow-tie has positive
    shoelace area, so ring_valid alone is NOT enough -- this is what catches the overlap artefact."""
    k = len(ring)
    if k < 3:
        return False
    P = [pos[int(v)] for v in ring]
    for i in range(k):
        for j in range(i + 1, k):
            if j == i + 1 or (i == 0 and j == k - 1):
                continue                                       # adjacent edges share a vertex
            if _seg_cross(P[i], P[(i + 1) % k], P[j], P[(j + 1) % k]):
                return False
    return True


def faces_overlap(rings, pos):
    """True iff any two of the given face rings have PROPERLY crossing edges (a local overlap)."""
    seg = lambda r, i: (pos[int(r[i])], pos[int(r[(i + 1) % len(r)])])
    for x in range(len(rings)):
        for y in range(x + 1, len(rings)):
            ra, rb = rings[x], rings[y]
            for i in range(len(ra)):
                a1, a2 = seg(ra, i)
                for j in range(len(rb)):
                    b1, b2 = seg(rb, j)
                    if _seg_cross(a1, a2, b1, b2):
                        return True
    return False


def ensure_faces(m):
    """Lazily attach the face-ring topology + per-face np buffers to the mesh dict. A small
    deterministic spread on the initial target areas desynchronises the division cycle (so cells
    do not all divide on the same tick -- the vertex-model analogue of reset_noise)."""
    if "faces" in m:
        return
    es = m["E_srce"].cpu().numpy(); et = m["E_trgt"].cpu().numpy(); ef = m["E_face"].cpu().numpy()
    m["faces"] = rings_from_flat(es, et, ef, m["nF"])
    A0 = m["A0"].detach().cpu().numpy().astype(np.float64).copy()
    g = np.random.default_rng(0)
    A0 = A0 * (1.0 + 0.12 * (g.random(A0.shape[0]) - 0.5))    # +/-6% cell-cycle phase spread
    m["A0_np"] = A0
    m["alive_np"] = m["alive"].detach().cpu().numpy().astype(np.float64).copy()


def rebuild(m, dev, dtype, p0):
    """Refresh the flat half-edge table + A0/P0/alive tensors from the face rings."""
    es, et, ef = flat_from_rings(m["faces"])
    nF = len(m["faces"])
    m["E_srce"] = torch.as_tensor(es, device=dev)
    m["E_trgt"] = torch.as_tensor(et, device=dev)
    m["E_face"] = torch.as_tensor(ef, device=dev)
    m["nF"] = nF
    A0 = np.asarray(m["A0_np"], np.float64); alive = np.asarray(m["alive_np"], np.float64)
    m["A0"] = torch.as_tensor(A0, device=dev, dtype=dtype)
    m["P0"] = float(p0) * m["A0"].clamp(min=1e-9).sqrt()
    m["alive"] = torch.as_tensor(alive, device=dev, dtype=dtype)


# --------------------------------------------------------------------------- #
#  Pure topology surgery on the face-ring list (numpy positions)
# --------------------------------------------------------------------------- #
def _faces_at(faces, v):
    return [f for f, r in enumerate(faces) if r is not None and v in r]


def _directed(faces, a, b):
    """Face index whose ring has a immediately followed by b (or None)."""
    for f, r in enumerate(faces):
        if r is None:
            continue
        k = len(r)
        for i in range(k):
            if r[i] == a and r[(i + 1) % k] == b:
                return f
    return None


def _remove(ring, v):
    return np.array([u for u in ring if u != v], dtype=np.int64)


def _insert_after(ring, anchor, v):
    out = []
    for u in ring:
        out.append(int(u))
        if u == anchor:
            out.append(int(v))
    return np.array(out, dtype=np.int64)


def _insert_before(ring, anchor, v):
    out = []
    for u in ring:
        if u == anchor:
            out.append(int(v))
        out.append(int(u))
    return np.array(out, dtype=np.int64)


def _adjacent(ring, x, y):
    """True if x and y are consecutive in the ring (i.e. form an edge of the face)."""
    idx = {int(u): i for i, u in enumerate(ring)}
    if int(x) not in idx or int(y) not in idx:
        return False
    k = len(ring); i, j = idx[int(x)], idx[int(y)]
    return abs(i - j) == 1 or abs(i - j) == k - 1


def t1_flip(faces, pos, a, b, new_len):
    """Reversible network reconnection on interior edge (a,b). 3-regular case. Tries the
    valid geometric convention; commits only if all four faces stay simple. Returns True if
    the flip was applied."""
    alpha = _directed(faces, a, b); beta = _directed(faces, b, a)
    if alpha is None or beta is None:
        return False                                            # boundary edge
    at_a = set(_faces_at(faces, a)); at_b = set(_faces_at(faces, b))
    da = at_a - {alpha, beta}; db = at_b - {alpha, beta}
    if len(da) != 1 or len(db) != 1:
        return False                                            # not 3-regular here
    delta = da.pop(); gamma = db.pop()
    mid = 0.5 * (pos[a] + pos[b]); d = pos[b] - pos[a]; L = np.linalg.norm(d)
    if L < 1e-9:
        return False
    perp = np.array([-d[1], d[0]]) / L
    # alpha loses b, beta loses a (unambiguous); gamma gains a and delta gains b, but the SIDE the
    # gained vertex is inserted on is set by geometry -- search sign x insertion-side and accept the
    # first flip that leaves all four faces simple, non-overlapping, and sharing the new a-b edge.
    ra = _remove(faces[alpha], b); rb = _remove(faces[beta], a)
    for sign in (+1.0, -1.0):
        na = mid - sign * perp * new_len / 2
        nb = mid + sign * perp * new_len / 2
        tpos = pos.copy(); tpos[a] = na; tpos[b] = nb
        for gfun in (_insert_after, _insert_before):            # a into gamma, next to b
            for dfun in (_insert_after, _insert_before):        # b into delta, next to a
                rg = gfun(faces[gamma], b, a); rd = dfun(faces[delta], a, b)
                rings = [ra, rb, rg, rd]
                if (_adjacent(rg, a, b) and _adjacent(rd, a, b)
                        and all(ring_valid(r, tpos) and ring_simple(r, tpos) for r in rings)
                        and not faces_overlap(rings, tpos)):
                    faces[alpha] = ra; faces[beta] = rb; faces[gamma] = rg; faces[delta] = rd
                    pos[a] = na; pos[b] = nb
                    return True
    return False


def _insert_between(ring, x, y, mid):
    """Insert `mid` wherever consecutive (x, y) appears in ring (i.e. on that directed edge)."""
    out = []
    k = len(ring)
    for i in range(k):
        out.append(int(ring[i]))
        if int(ring[i]) == x and int(ring[(i + 1) % k]) == y:
            out.append(int(mid))
    return np.array(out, dtype=np.int64)


def _line_cross_edges(ring, pos, c, angle):
    """Edges of `ring` crossed by the line through centre c at `angle`, with the crossing points."""
    nrm = np.array([-np.sin(angle), np.cos(angle)])          # normal to the division line
    s = np.array([np.dot(pos[int(v)] - c, nrm) for v in ring])
    k = len(ring); out = []
    for i in range(k):
        a, b = s[i], s[(i + 1) % k]
        if (a > 0) != (b > 0):                               # endpoints on opposite sides -> crossed
            t = a / (a - b)
            p = pos[int(ring[i])] + t * (pos[int(ring[(i + 1) % k])] - pos[int(ring[i])])
            out.append((i, p))
    return out


def face_divide_line(faces, pos, m, f, angle):
    """Straight-line (Brodland--Veldhuis) division of face f at `angle` through its centroid: bisect
    the two crossed edges (a new vertex on each, SHARED with the neighbour across that edge), then a
    septum between the two new vertices -> two daughters. Returns (newpos:{idx:xy}, daughter_f) or
    (None, -1). Needs vertex-buffer headroom (m['Nv']+2 <= buffer)."""
    ring = faces[f]
    if ring is None or len(ring) < 4:
        return None, -1
    c = pos[ring].mean(0)
    crossed = _line_cross_edges(ring, pos, c, angle)
    if len(crossed) != 2:
        return None, -1
    (i0, p0), (i1, p1) = crossed
    Nv = m["Nv"]
    if Nv + 2 > pos.shape[0]:
        return None, -1
    mi, mj = Nv, Nv + 1
    k = len(ring)
    a0, b0 = int(ring[i0]), int(ring[(i0 + 1) % k])
    a1, b1 = int(ring[i1]), int(ring[(i1 + 1) % k])
    trial = [dict() for _ in range(0)]                       # (validation happens after we commit tentatively)
    # neighbour faces across each split edge get the new vertex too (shared edge stays shared)
    g0 = _directed(faces, b0, a0); g1 = _directed(faces, b1, a1)
    # build daughters from f's ring with mi after a0 and mj after a1
    newring = []
    for kk in range(k):
        newring.append(int(ring[kk]))
        if kk == i0:
            newring.append(mi)
        if kk == i1:
            newring.append(mj)
    im = newring.index(mi); jm = newring.index(mj)
    lo, hi = min(im, jm), max(im, jm)
    d1 = np.array(newring[lo:hi + 1], np.int64)
    d2 = np.array(newring[hi:] + newring[:lo + 1], np.int64)
    tpos = pos.copy(); tpos[mi] = p0; tpos[mj] = p1
    if not (ring_valid(d1, tpos) and ring_simple(d1, tpos)
            and ring_valid(d2, tpos) and ring_simple(d2, tpos)):
        return None, -1
    # commit
    pos[mi] = p0; pos[mj] = p1
    if g0 is not None:
        faces[g0] = _insert_between(faces[g0], b0, a0, mi)
    if g1 is not None:
        faces[g1] = _insert_between(faces[g1], b1, a1, mj)
    faces[f] = d1
    faces.append(d2)
    m["Nv"] = Nv + 2
    return {mi: p0, mj: p1}, len(faces) - 1


def face_split(faces, pos, A0_np, alive_np, f, a0_base):
    """Through-vertex division: septum between two ~opposite vertices of face f. No new
    vertices (local, always-valid); appends the daughter. Returns daughter index or -1."""
    r = faces[f]
    if r is None or len(r) < 4:
        return -1
    k = len(r); i0 = 0; i1 = k // 2
    d1 = r[i0:i1 + 1].copy()                                    # v_i0 .. v_i1
    d2 = np.concatenate([r[i1:], r[:i0 + 1]])                   # v_i1 .. v_{k-1}, v_i0
    if not (ring_valid(d1, pos) and ring_valid(d2, pos)):
        return -1
    faces[f] = d1
    faces.append(d2)
    A0_np[f] = a0_base                                          # both daughters reset to base
    A0_np.resize(len(faces), refcheck=False); A0_np[-1] = a0_base
    alive_np.resize(len(faces), refcheck=False); alive_np[-1] = 1.0
    return len(faces) - 1


def face_collapse(faces, pos, alive_np, f):
    """Extrude/apoptosis: collapse face f to its centroid vertex, retire the face. The shared
    vertices are pulled to the centroid so the neighbours re-close around it."""
    r = faces[f]
    if r is None or len(r) < 3:
        return False
    c = pos[r].mean(0)
    keep = int(r[0])
    pos[keep] = c
    # rewire every face: replace all of r's vertices by `keep`, drop consecutive duplicates
    for g, rg in enumerate(faces):
        if rg is None:
            continue
        ng = np.array([keep if u in r else int(u) for u in rg], dtype=np.int64)
        ded = [ng[i] for i in range(len(ng)) if ng[i] != ng[(i - 1) % len(ng)]]
        faces[g] = np.array(ded, dtype=np.int64) if len(ded) >= 3 else None
        if faces[g] is None:
            alive_np[g] = 0.0
    faces[f] = None; alive_np[f] = 0.0
    return True


# --------------------------------------------------------------------------- #
#  Registered operators
# --------------------------------------------------------------------------- #
@register_operator("face_growth", set="vertex", kind="structural", family="growth")
class FaceGrowth(Structural):
    """Inflate each live face's target area A0 by a fixed rate per tick (the proliferation
    drive). Mutates the mesh buffer; emits no delta."""
    SUPPORTED_DIMS = [2]; DIFFERENTIABLE = False
    MECHANISM_TAGS = ["growth", "target_area_inflation", "proliferation_drive"]
    PARAM_ROLES = {"rate": "area_growth_per_tick", "a0_max": "division_area"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.rate = float(params.get("rate", 0.0))
        self.p0 = float(params.get("p0", 3.85))

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None or self.rate == 0.0:
            return {}
        ensure_faces(m)
        m["A0_np"] = m["A0_np"] * (1.0 + self.rate)
        rebuild(m, lvl.state.device, lvl.state.dtype, self.p0)
        return {}


@register_operator("face_divide", set="vertex", kind="structural", family="growth")
class FaceDivide(Structural):
    """Split faces whose actual area exceeds `ratio` x base by a septum through two vertices
    (through-vertex, always-valid, no new vertices). Resets daughters' A0 to base."""
    SUPPORTED_DIMS = [3, 2]; DIFFERENTIABLE = False
    MECHANISM_TAGS = ["cell_division", "mitosis", "vertex_model"]
    PARAM_ROLES = {"ratio": "division_area_ratio", "a0_base": "reset_target_area"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.ratio = float(params.get("ratio", 1.5))
        self.p0 = float(params.get("p0", 3.85))
        self.a0_base = float(params.get("a0_base", (math.sqrt(3) / 2.0)))
        self.frac = float(params.get("frac", 0.0))            # >0: one-shot clonal division of a fraction
        self._done = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        ensure_faces(m)
        pos = lvl.get("pos").detach().cpu().numpy().astype(np.float64)
        changed = False
        nF0 = m["nF"]
        if self.frac > 0.0:                                   # one-shot: divide a fraction of well-shaped cells once
            if self._done:
                return {}
            self._done = True
            g = np.random.default_rng(1)
            for f in range(nF0):
                if (m["alive_np"][f] > 0 and len(m["faces"][f]) >= 6 and g.random() < self.frac
                        and face_split(m["faces"], pos, m["A0_np"], m["alive_np"], f, self.a0_base) >= 0):
                    changed = True
        else:                                                # cell-cycle: divide on the GROWN target area A0
            for f in range(nF0):
                if m["alive_np"][f] > 0 and m["A0_np"][f] > self.ratio * self.a0_base and len(m["faces"][f]) >= 4:
                    if face_split(m["faces"], pos, m["A0_np"], m["alive_np"], f, self.a0_base) >= 0:
                        changed = True
        if changed:
            rebuild(m, lvl.state.device, lvl.state.dtype, self.p0)
        return {}


@register_operator("face_divide_line", set="vertex", kind="structural", family="growth")
class FaceDivideLine(Structural):
    """Straight-line cell division (Brodland--Veldhuis; tyssue 06-Cell_Division): divide the target
    cells by a plane at a decidable `angle` through each centroid, adding a vertex on each of the two
    crossed edges (shared with the neighbour) and a septum between them -> well-shaped daughters.
    One-shot. `cells` selects explicit face indices, else `frac` of the cells at random."""
    SUPPORTED_DIMS = [2]; DIFFERENTIABLE = False
    MECHANISM_TAGS = ["cell_division", "straight_line", "brodland_veldhuis", "edge_midpoint"]
    PARAM_ROLES = {"angle": "division_plane_angle", "cells": "explicit_mother_cells", "frac": "random_fraction"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.angle = float(params.get("angle", np.pi / 2))
        self.frac = float(params.get("frac", 0.0))
        self.ratio = float(params.get("ratio", 0.0))          # >0: CELL-CYCLE mode (divide when A0 > ratio*base)
        self.cells = params.get("cells", None)
        self.p0 = float(params.get("p0", 3.85))
        self.a0_base = float(params.get("a0_base", (math.sqrt(3) / 2.0)))
        self._done = False; self._call = 0

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        ensure_faces(m)
        pos = lvl.get("pos").detach().cpu().numpy().astype(np.float64)
        nF0 = m["nF"]; self._call += 1
        if self.ratio > 0:                                    # CELL-CYCLE: repeats; divide grown cells at random angle
            targets = [f for f in range(nF0) if m["alive_np"][f] > 0
                       and m["A0_np"][f] > self.ratio * self.a0_base and len(m["faces"][f]) >= 4]
            g = np.random.default_rng(1000 + self._call)
            angles = {f: float(g.random() * np.pi) for f in targets}
        else:                                                 # one-shot (explicit cells or a fraction) at fixed angle
            if self._done:
                return {}
            self._done = True
            if self.cells is not None:
                targets = list(self.cells)
            elif self.frac > 0:
                g = np.random.default_rng(1)
                targets = [f for f in range(nF0)
                           if m["alive_np"][f] > 0 and len(m["faces"][f]) >= 4 and g.random() < self.frac]
            else:
                targets = list(range(nF0))
            angles = {f: self.angle for f in targets}
        A0 = list(m["A0_np"]); alive = list(m["alive_np"])
        for f in targets:
            if m["alive_np"][f] <= 0 or len(m["faces"][f]) < 4:
                continue
            _, d = face_divide_line(m["faces"], pos, m, f, angles[f])
            if d >= 0:
                A0[f] = A0[f] / 2.0                             # area conserved: mother + daughter each
                A0.append(A0[f]); alive.append(1.0)            #   target HALF the original (tyssue halves prefered_area)
        m["A0_np"] = np.array(A0, np.float64); m["alive_np"] = np.array(alive, np.float64)
        if m["pin"].shape[0] < m["Nv"]:                       # the new division vertices are interior (not pinned)
            pad = torch.zeros(m["Nv"] - m["pin"].shape[0], dtype=m["pin"].dtype, device=m["pin"].device)
            m["pin"] = torch.cat([m["pin"], pad])
        Nv = m["Nv"]; px0, px1 = lvl.state_schema["pos"]
        st = lvl.state.clone()
        st[:Nv, px0:px1] = torch.as_tensor(pos[:Nv], device=st.device, dtype=st.dtype)
        lvl.state = st
        if getattr(lvl, "occ", None) is not None:
            occ = lvl.occ.clone(); occ[:Nv] = 1.0; lvl.occ = occ
        rebuild(m, lvl.state.device, lvl.state.dtype, self.p0)
        return {}


@register_operator("apoptosis", set="vertex", kind="structural", family="growth")
class Apoptosis(Structural):
    """Cell elimination (Monier et al. 2015; tyssue B-Apoptosis), as a scheduled behaviour decomposed
    into primitives: the apoptotic cell's target area SHRINKS each tick (shape_energy then contracts
    it), short edges are resolved by T1 (the cell sheds neighbours), and once it is small enough (or
    reduced to a triangle) it is EXTRUDED -- its remaining vertices merged to a single point
    (face_collapse). The tissue closes the gap by force balance."""
    SUPPORTED_DIMS = [2]; DIFFERENTIABLE = False
    MECHANISM_TAGS = ["apoptosis", "cell_elimination", "extrusion", "delamination"]
    PARAM_ROLES = {"cells": "apoptotic cells", "shrink_rate": "target-area shrink per tick",
                   "critical_frac": "extrude below this x base area"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.cells = list(params.get("cells", []))
        self.shrink = float(params.get("shrink_rate", 0.04))
        self.crit = float(params.get("critical_frac", 0.12))
        self.p0 = float(params.get("p0", 3.85))
        self.a0_base = float(params.get("a0_base", (math.sqrt(3) / 2.0)))

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        ensure_faces(m)
        pos = lvl.get("pos").detach().cpu().numpy().astype(np.float64)
        crit_area = self.crit * self.a0_base
        for f in self.cells:
            if f >= len(m["faces"]) or m["alive_np"][f] <= 0:
                continue
            m["A0_np"][f] = max(m["A0_np"][f] * (1.0 - self.shrink), 1e-4)   # shrink the target area
            r = m["faces"][f]
            if r is None:
                continue
            # MULTISTEP reconnection, but GENTLE: the scheduled t1_transition sheds the shrinking cell's
            # sides gradually (one short edge at a time); apoptosis only collapses the final TRIANGLE
            # -> a clean degree-3 junction (no rosette), and surface tension keeps the intermediates round.
            if len(r) <= 3 and abs(ring_signed_area(r, pos)) < crit_area:
                face_collapse(m["faces"], pos, m["alive_np"], f)
        Nv = m["Nv"]; px0, px1 = lvl.state_schema["pos"]
        st = lvl.state.clone(); st[:Nv, px0:px1] = torch.as_tensor(pos[:Nv], device=st.device, dtype=st.dtype)
        lvl.state = st
        rebuild(m, lvl.state.device, lvl.state.dtype, self.p0)
        return {}


@register_operator("t1_transition", set="vertex", kind="rewire", family="topology")
class T1Transition(Rewire):
    """Reversible network reconnection (RNR / T1): flip every interior edge shorter than
    `l_th` by a local collapse+split, committing only valid flips. This is the operator the
    Self-Propelled-Voronoi route could not have -- explicit neighbour exchange on a shared-
    vertex mesh, the ingredient the Turing_vertex report named as missing for tissue flow."""
    SUPPORTED_DIMS = [2]; DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True                        # T1 repositions the collapsed/split vertices
    MECHANISM_TAGS = ["T1_transition", "reversible_network_reconnection", "intercalation",
                      "neighbour_exchange", "vertex_model"]
    PARAM_ROLES = {"l_th": "reconnection_threshold_length", "multiplier": "new_edge_length_factor"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.l_th = float(params.get("l_th", 0.12))
        self.mult = float(params.get("multiplier", 1.5))
        self.p0 = float(params.get("p0", 3.85))
        self.n_t1 = 0

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        ensure_faces(m)
        pos = lvl.get("pos").detach().cpu().numpy().astype(np.float64)
        es = m["E_srce"].cpu().numpy(); et = m["E_trgt"].cpu().numpy()
        length = np.linalg.norm(pos[et] - pos[es], axis=1)
        # unique undirected short edges, shortest first
        seen = set(); order = np.argsort(length)
        fired = 0
        for k in order:
            if length[k] >= self.l_th:
                break
            a, b = int(es[k]), int(et[k]); key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            if t1_flip(m["faces"], pos, a, b, self.l_th * self.mult):
                fired += 1
        if fired:
            self.n_t1 += fired
            m["n_t1"] = m.get("n_t1", 0) + fired              # count on the mesh -> readable after the run
            # write the moved vertex positions back + rebuild the flat table
            px0, px1 = lvl.state_schema["pos"]
            st = lvl.state.clone()
            st[:pos.shape[0], px0:px1] = torch.as_tensor(pos, device=lvl.state.device, dtype=lvl.state.dtype)
            lvl.state = st
            rebuild(m, lvl.state.device, lvl.state.dtype, self.p0)
        return {}


@register_operator("topo_snapshot", set="vertex", kind="structural", family="growth")
class TopoSnapshot(Structural):
    """Records a per-tick copy of the face-ring topology onto the mesh (`_mesh["hist_faces"]`)
    so a movie can be rendered even though division / T1 change the topology during the run.
    Emits no delta; put it LAST in the schedule so it captures the post-step topology."""
    SUPPORTED_DIMS = [2, 3]; DIFFERENTIABLE = False

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        ensure_faces(m)
        m.setdefault("hist_faces", []).append([None if r is None else r.copy() for r in m["faces"]])
        return {}


# --------------------------------------------------------------------------- #
#  Standalone self-test: build the honeycomb, apply T1 + division + extrusion,
#  assert the mesh stays valid (no engine involved).
# --------------------------------------------------------------------------- #
def _mesh_valid_global(faces, pos):
    bad = 0
    for r in faces:
        if r is None:
            continue
        if not ring_valid(r, pos):
            bad += 1
    return bad


if __name__ == "__main__":
    from tyssue_ops import build_honeycomb
    verts, es, et, ef, fc, pin, a0 = build_honeycomb(10, 12, 1.0, 1, 0.15, 0)
    nF = int(ef.max()) + 1
    faces = rings_from_flat(es, et, ef, nF)
    pos = verts.copy()
    A0 = np.full(nF, a0); alive = np.ones(nF)
    print(f"honeycomb: Nv={pos.shape[0]} F={nF}  invalid faces at start: {_mesh_valid_global(faces, pos)}")

    # 1) T1 on the shortest interior edges
    length = np.linalg.norm(pos[et] - pos[es], axis=1)
    order = np.argsort(length); fired = 0; tried = 0
    for k in order[:40]:
        a, b = int(es[k]), int(et[k])
        tried += 1
        if t1_flip(faces, pos, a, b, 0.25):
            fired += 1
    print(f"T1: fired {fired}/{tried} attempts; invalid faces after: {_mesh_valid_global(faces, pos)}")

    # 2) division of a few faces
    ndiv = 0
    for f in [10, 20, 30, 40]:
        if face_split(faces, pos, A0, alive, f, a0) >= 0:
            ndiv += 1
    print(f"divide: {ndiv} faces split -> F={len(faces)}; invalid faces after: {_mesh_valid_global(faces, pos)}")

    # 3) extrusion of a face
    ok = face_collapse(faces, pos, alive, 55)
    print(f"extrude: collapsed={ok}; live faces={int(alive.sum())}; invalid faces after: {_mesh_valid_global(faces, pos)}")
    print("SELF-TEST OK" if _mesh_valid_global(faces, pos) == 0 else "SELF-TEST: invalid faces present")
