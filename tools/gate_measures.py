#!/usr/bin/env python
"""One function per gate measure, and one facade over the two trajectory layouts.

WHY A FACADE. A gate is only worth running if the SAME code grades the promoted run and the okuda
reference; two measurement functions that agree on a good run and disagree on a bad one are worse
than none. So every `fn` below takes a `Traj` and never a filename, and there are two `Traj`
implementations -- one over the core's `trajectory.npz`, one over okuda's `traj.npz` -- with exactly
the same seven accessors.

WHAT IS IN A CORE TRAJECTORY, since every function here is bounded by it:

    <set>__pos              [T, buffer, D]      positions, the live prefix given by nF/Nv
    <set>__occ              [T, buffer]         bool
    <set>__<block>          [T, buffer, width]  every recorded state block (chem, area, cen, ...)
    <set>__mesh_offsets     [T+1]               HALF-EDGE row offsets
    <set>__mesh_face_offsets[T+1]               FACE row offsets -- a DIFFERENT ragged length
    <set>__mesh_nF/_Nv      [T]
    <set>__mesh_E_srce/_E_trgt/_E_face  concatenated int64
    <set>__mesh_<name>      concatenated float32, one entry per face per row (A0, age, ndiv, ...)

THE TWO OFFSET ARRAYS ARE NOT INTERCHANGEABLE and mixing them is the mistake this module is written
to make impossible: there are about six half-edges per face, so slicing the myosin with the half-edge
offsets returns a window six times too long, starting in the wrong frame, and the resulting number
looks entirely plausible.

EVERY FUNCTION RETURNS A SERIES, one entry per recorded row, and the row's `reduce:` collapses it.
That is deliberate: a gate that only ever looks at the last frame cannot see a tissue that tore at
frame 200 and healed, and `euler_characteristic` with `reduce: all` is exactly that case.

UNITS ARE APPLIED BY THE REDUCER, NEVER BY THE FUNCTION. An `fn` returns simulation units; a row whose
`unit:` is physical converts through the spec's `units:` block, which RAISES if none is declared.
That is the mechanism that stops a dimensionless run quoting a micrometre.
"""
from __future__ import annotations

import os

import numpy as np


# ============================================================================== the facade
class _Lazy:
    """`np.load`'s NpzFile DECOMPRESSES THE WHOLE ARRAY ON EVERY `z[key]`, and that is not a
    micro-optimisation to fix.

    `ParticleTraj.pos(t)` was written as `self.z["mpm_particle__pos"][t]` -- which re-reads all 778 MB
    of a 720-row MPM trajectory to take one row. Five measures x 720 rows is 2,800 GB of decompression
    for a table that should take seconds, and the grading ran past ten minutes with nothing to show.
    One dict, read once per key.
    """

    def __init__(self, z):
        self._z, self._c = z, {}

    def __contains__(self, k): return k in self._z.files

    @property
    def files(self): return self._z.files

    def __getitem__(self, k):
        if k not in self._c:
            self._c[k] = self._z[k]
        return self._c[k]


class Traj:
    """Seven accessors. Everything else in this file is written against these."""

    def n_rows(self) -> int: raise NotImplementedError
    def nF(self, t) -> int: raise NotImplementedError
    def nV(self, t) -> int: raise NotImplementedError
    def pos(self, t): raise NotImplementedError          # [nV, 3] live prefix
    def half_edges(self, t): raise NotImplementedError   # (E_srce, E_trgt, E_face) for row t
    def face_col(self, name, t): raise NotImplementedError   # [nF] or None
    def state(self, block, t): raise NotImplementedError     # [nF, width] or None
    # PER-VERTEX, AND SEPARATE FROM `state` ON PURPOSE. `state` reads the CELL set and crops to nF;
    # a per-vertex block is a different array of a different length -- on gate_00's last row, 13,824
    # vertices against 6,914 cells -- so reusing `state` would return the first 6,914 rows of a
    # 13,824-row array, which is not a subset of anything and would look like data. Added for the
    # apico-basal separation `sep`, which is the first per-vertex block that is not `pos`.
    def vertex_block(self, name, t): return None             # [nV, width] or None
    def occ(self, set_name, t): raise NotImplementedError    # [buffer] bool
    def scalar(self, name, t): return None                   # an operator's own counter, or None
    def edge_col(self, name, t): return None                 # per-HALF-EDGE state, or None


class CoreTraj(Traj):
    """The core's `trajectory.npz`."""

    def __init__(self, path, set_name=None, cell_set=None):
        self.z = _Lazy(np.load(path))
        f = self.z.files
        if set_name is None:
            c = [k[: -len("__mesh_nF")] for k in f if k.endswith("__mesh_nF")]
            set_name = c[0] if c else None
        if set_name is None:
            raise ValueError(f"{path} has no mesh set -- this gate's measures need one")
        self.s = set_name
        if cell_set is None:
            c = [k[: -len("__occ")] for k in f
                 if k.endswith("__occ") and not k.startswith(set_name)]
            cell_set = c[0] if c else None
        self.c = cell_set
        self._nF = self.z[f"{set_name}__mesh_nF"]
        self._Nv = self.z[f"{set_name}__mesh_Nv"]
        self._off = self.z[f"{set_name}__mesh_offsets"]
        self._foff = self.z[f"{set_name}__mesh_face_offsets"]
        self.path = path

    def n_rows(self): return int(len(self._nF))
    def nF(self, t): return int(self._nF[t])
    def nV(self, t): return int(self._Nv[t])
    def pos(self, t): return np.asarray(self.z[f"{self.s}__pos"][t][: self.nV(t)], float)

    def half_edges(self, t):
        a, b = int(self._off[t]), int(self._off[t + 1])
        return (self.z[f"{self.s}__mesh_E_srce"][a:b],
                self.z[f"{self.s}__mesh_E_trgt"][a:b],
                self.z[f"{self.s}__mesh_E_face"][a:b])

    def face_col(self, name, t):
        k = f"{self.s}__mesh_{name}"
        if k not in self.z.files:
            return None
        a, b = int(self._foff[t]), int(self._foff[t + 1])
        return np.asarray(self.z[k][a:b], float)

    def state(self, block, t):
        k = f"{self.c}__{block}"
        if self.c is None or k not in self.z.files:
            return None
        return np.asarray(self.z[k][t][: self.nF(t)], float)

    def vertex_block(self, name, t):
        k = f"{self.s}__{name}"
        if k not in self.z.files:
            return None
        a = np.asarray(self.z[k][t][: self.nV(t)], float)
        return a[:, None] if a.ndim == 1 else a

    def occ(self, set_name, t):
        k = f"{set_name}__occ"
        return np.asarray(self.z[k][t], bool) if k in self.z.files else None

    def scalar(self, name, t):
        k = f"{self.s}__mesh_scalar_{name}"
        return float(self.z[k][t]) if k in self.z.files else None

    def edge_col(self, name, t):
        """A per-HALF-EDGE column, sliced with ITS OWN offsets.

        A third ragged length, and the reason it carries its own offsets rather than sharing
        `mesh_offsets`: `junction_myosin` writes `myo` for the arrays as they were BEFORE the
        frame's topology operators, so on a frame with a division or a flip the array is a few
        entries short of the half-edge table. Sharing offsets would slice the next frame's myosin
        into this one, silently. `myosin_array_aligned_with_half_edges` is the row that asserts the
        lengths agree; this accessor must not paper over the case it tests.
        """
        k = f"{self.s}__mesh_e_{name}"
        ko = k + "_offsets"
        if k not in self.z.files or ko not in self.z.files:
            return None
        o = self.z[ko]
        return np.asarray(self.z[k][int(o[t]):int(o[t + 1])], float)


class ParticleTraj(Traj):
    """A core trajectory with NO MESH: an MPM set and its parent, which is gates 02 and 03.

    `CoreTraj` raises on a file with no `__mesh_nF`, and rightly -- eleven of gate 00's rows would
    silently become uncomputable rather than failing. A particle gate needs a different reader, not
    a `CoreTraj` with the mesh accessors returning None, because "the mesh is empty" and "there is
    no mesh" are different facts and only one of them is a defect.
    """

    def __init__(self, path, set_name=None):
        self.z = _Lazy(np.load(path))
        if set_name is None:
            c = [k[: -len("__pos")] for k in self.z.files if k.endswith("__pos")]
            # the biggest positional set is the material; the parent is a single centroid
            set_name = max(c, key=lambda k: self.z[f"{k}__pos"].shape[1]) if c else None
        if set_name is None:
            raise ValueError(f"{path} has no positional set")
        self.s = set_name
        self.parent = next((k[: -len("__pos")] for k in self.z.files
                            if k.endswith("__pos") and k[: -len("__pos")] != set_name), None)
        self.path = path

    def n_rows(self): return int(self.z[f"{self.s}__pos"].shape[0])
    def nF(self, t): return int(self.occ(self.s, t).sum())
    def nV(self, t): return self.nF(t)

    def pos(self, t):
        o = self.occ(self.s, t)
        p = self.z[f"{self.s}__pos"][t]
        return np.asarray(p[o] if o is not None else p, float)

    def half_edges(self, t):
        raise KeyError("this gate's trajectory has no mesh -- a topology row does not belong in it")

    def face_col(self, name, t): return None
    def state(self, block, t):
        k = f"{self.s}__{block}"
        return np.asarray(self.z[k][t], float) if k in self.z.files else None

    def occ(self, set_name, t):
        k = f"{set_name}__occ"
        return np.asarray(self.z[k][t], bool) if k in self.z.files else None

    def scalar(self, name, t): return None

    def parent_pos(self, t):
        if self.parent is None:
            return self.pos(t).mean(0)
        return np.asarray(self.z[f"{self.parent}__pos"][t][0], float)


class OkudaTraj(Traj):
    """okuda's `traj.npz`: `pos_i`, a pickled `mesh_i`, `act_i`, and `ticks`.

    IT IS DECIMATED, and that matters for a gate. okuda keeps about 60 of the run's rows -- the ones
    its movie draws -- so a `window:` that falls between kept rows cannot be answered. `row_of(tick)`
    raises rather than picking the nearest, because a per-tick ledger silently evaluated across a
    31-frame gap is a green row that tested nothing.
    """

    def __init__(self, path):
        self.z = np.load(path, allow_pickle=True)
        self.ticks = np.asarray(self.z["ticks"]).tolist() if "ticks" in self.z.files else None
        self._n = sum(1 for k in self.z.files if k.startswith("pos_"))
        self._m = [None] * self._n
        self.path = path

    def _mesh(self, t):
        if self._m[t] is None:
            v = self.z[f"mesh_{t}"]
            self._m[t] = v.item() if hasattr(v, "item") else v
        return self._m[t]

    def n_rows(self): return self._n
    def nF(self, t): return int(self._mesh(t)["nF"])
    def nV(self, t): return int(self._mesh(t)["Nv"])
    def pos(self, t): return np.asarray(self.z[f"pos_{t}"], float)

    def half_edges(self, t):
        m = self._mesh(t)
        return (np.asarray(m["E_srce"]), np.asarray(m["E_trgt"]), np.asarray(m["E_face"]))

    def face_col(self, name, t):
        v = self._mesh(t).get(name)
        return None if v is None else np.asarray(v, float).ravel()[: self.nF(t)]

    def state(self, block, t):
        if block == "chem" and f"act_{t}" in self.z.files:
            return np.asarray(self.z[f"act_{t}"], float)[:, None]
        return None

    def occ(self, set_name, t):
        return None                                # okuda records the live prefix, not the mask

    def scalar(self, name, t):
        v = self._mesh(t).get(name)
        return None if v is None or np.ndim(v) != 0 else float(v)

    def edge_col(self, name, t):
        v = self._mesh(t).get(name)
        return None if v is None or np.ndim(v) == 0 else np.asarray(v, float).ravel()

    def row_of(self, tick):
        if self.ticks is None:
            return int(tick)
        if int(tick) not in self.ticks:
            raise KeyError(f"tick {tick} is not one of the {len(self.ticks)} rows okuda kept "
                           f"(nearest {min(self.ticks, key=lambda x: abs(x - tick))}); a per-tick "
                           f"measure cannot be evaluated across the gap")
        return self.ticks.index(int(tick))


def _core(path):
    """`CoreTraj` if the file carries a mesh, `ParticleTraj` if it does not."""
    z = np.load(path)
    return (CoreTraj(path) if any(k.endswith("__mesh_nF") for k in z.files)
            else ParticleTraj(path))


def open_traj(path_or_dir):
    """A `Traj` over whichever layout is at this path."""
    p = path_or_dir
    if os.path.isdir(p):
        if os.path.exists(os.path.join(p, "traj.npz")):
            return OkudaTraj(os.path.join(p, "traj.npz"))
        for root, _d, files in os.walk(p):
            if "trajectory.npz" in files:
                return _core(os.path.join(root, "trajectory.npz"))
        raise FileNotFoundError(f"no trajectory under {p}")
    return OkudaTraj(p) if os.path.basename(p) != "trajectory.npz" else _core(p)


# ============================================================================== helpers
def _n_edges(T, t):
    """Undirected edges of a CLOSED surface: every edge is exactly two half-edges."""
    es, _et, _ef = T.half_edges(t)
    return len(es) // 2


def _live_face_mask(T, t):
    es, et, ef = T.half_edges(t)
    return ef < T.nF(t)


# ============================================================================== the measures
def cell_count(T, **kw):
    return [T.nF(t) for t in range(T.n_rows())]


def cell_count_delta(T, **kw):
    c = cell_count(T)
    return [c[i] - c[i - 1] for i in range(1, len(c))] or [0]


def vertex_count(T, **kw):
    return [T.nV(t) for t in range(T.n_rows())]


def occ_vs_mesh(T, vertex_set="vertex", cell_set="cell", **kw):
    """|live(occ) - nF| + |live(vertex occ) - Nv| per row.

    okuda records the live PREFIX and no mask, so there is nothing to disagree with and the row is
    identically zero there. That is honest -- the check exists for the core's reservoir, which is the
    side that has a mask to get wrong -- and it is stated rather than hidden, because a row that is
    structurally zero on one side is a row that cannot fail there.
    """
    out = []
    for t in range(T.n_rows()):
        ov, oc = T.occ(vertex_set, t), T.occ(cell_set, t)
        d = 0
        if ov is not None:
            d += abs(int(ov.sum()) - T.nV(t))
        if oc is not None:
            d += abs(int(oc.sum()) - T.nF(t))
        out.append(d)
    return out


def topology_ledger(T, **kw):
    """|dV - 2*dF| + |dE - 3*dF| per recorded tick.

    A septum adds 2 vertices, 3 edges, 1 face; a T1 adds none of the three. On a DECIMATED reference
    the differences span many ticks and the identity does not hold per row, so this returns the
    per-row residual only when consecutive rows are consecutive ticks.
    """
    ticks = getattr(T, "ticks", None)
    out = []
    for t in range(1, T.n_rows()):
        if ticks is not None and ticks[t] - ticks[t - 1] != 1:
            continue
        dV = T.nV(t) - T.nV(t - 1)
        dF = T.nF(t) - T.nF(t - 1)
        dE = _n_edges(T, t) - _n_edges(T, t - 1)
        out.append(abs(dV - 2 * dF) + abs(dE - 3 * dF))
    return out or [0]


def nonfinite_count(T, blocks=("chem", "area", "cen"), **kw):
    out = []
    for t in range(T.n_rows()):
        n = int((~np.isfinite(T.pos(t))).sum())
        for b in blocks:
            v = T.state(b, t)
            if v is not None:
                n += int((~np.isfinite(v)).sum())
        out.append(n)
    return out


def reservoir_fraction(T, **kw):
    """Live vertices as a fraction of the allocated buffer. Core only -- okuda's file does not
    carry the buffer size, so the row reports 0 there rather than guessing."""
    z = getattr(T, "z", None)
    s = getattr(T, "s", None)
    if z is None or s is None or f"{s}__occ" not in getattr(z, "files", []):
        return [0.0]
    buf = z[f"{s}__occ"].shape[1]
    return [T.nV(t) / float(buf) for t in range(T.n_rows())]


# ============================================================================== apico-basal
# THE FOUR ROWS OF R2, and all four exist only because R1(d) gave the facade `vertex_block`. Every
# one reads `sep` -- the apico-basal HALF separation, so apical = pos + sep and basal = pos - sep --
# which is a per-VERTEX state block cropped by nV, not a per-cell one cropped by nF.


def _sep(T, t, name="sep"):
    """The separation at row t, or None. One place, because four measures need it and a second
    spelling of "which block holds the separation" is a second thing to keep in step."""
    return T.vertex_block(name, t)


def _vertex_normals_np(pos, es, et, ef, nF, nV):
    """Outward unit normal per vertex: the normalised sum of incident face area vectors.

    THE SAME CONSTRUCTION `monolayer_shells` USES, in numpy because a measure reads a saved run and
    never has torch tensors. It is a reader, not a second definition of the model's geometry: if the
    two ever disagree the gate is measuring something the run did not do, which is why AB-B1 tests
    the SIGN against this normal and never the magnitude against it.
    """
    cr = np.cross(pos[es], pos[et])
    Nf = np.zeros((nF, 3)); np.add.at(Nf, ef, cr); Nf *= 0.5
    vn = np.zeros((nV, 3)); np.add.at(vn, es, Nf[ef])
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def _cell_polyhedron_volume(a, b, es, et, ef, nF, origin):
    """Signed volume of each cell's polyhedron about `origin[f]`, by the divergence theorem.

    V_f = (1/6) * sum over the cell's triangles of (p-o) . ((q-o) x (r-o)), over the closed surface

        apical cap   the ring on `a`, fanned from the cap's own centroid, outward winding
        basal cap    the ring on `b`, REVERSED, because it faces the other way
        lateral wall one quad per ring edge (a_s, a_t, b_t, b_s), split into two triangles

    which is exactly the surface `cell_geometry[implementation: polyhedral]` will integrate at R7 --
    so a disagreement between this and that is a disagreement about the same object, not about two
    conventions.
    """
    o = origin[ef]                                             # [E, 3] the owning cell's origin
    ca = np.zeros((nF, 3)); cb = np.zeros((nF, 3)); cnt = np.zeros(nF)
    np.add.at(ca, ef, a[es]); np.add.at(cb, ef, b[es]); np.add.at(cnt, ef, 1.0)
    k = np.maximum(cnt, 1.0)[:, None]
    ca /= k; cb /= k                                           # cap centroids, per cell
    vol = np.zeros(nF)

    def acc(p, q, r):
        np.add.at(vol, ef, np.einsum("ij,ij->i", p - o, np.cross(q - o, r - o)))

    A0, A1, B0, B1 = a[es], a[et], b[es], b[et]
    acc(ca[ef], A0, A1)                                        # apical cap, outward
    acc(cb[ef], B1, B0)                                        # basal cap, reversed
    # THE LATERAL WALL, WOUND OUTWARD. Written (a_s, a_t, b_t) first, which is the order the quad
    # reads in, and that is INWARD: for a ring CCW seen from outside, (a_t - a_s) is tangential and
    # (b_t - a_s) points basally, and their cross product faces into the cell. The closure test did
    # not catch it -- an inward-wound wall is still CLOSED, so the two origins still agreed exactly
    # -- it only showed up against a hexagonal prism of known volume, which came back -0.866 for a
    # cell of +2.598. Consistency is not correctness, which is why this function is checked against
    # an analytic solid and not only against itself.
    acc(A0, B0, B1)                                            # lateral quad, triangle 1
    acc(A0, B1, A1)                                            # lateral quad, triangle 2
    return vol / 6.0


def apicobasal_span_invalid_count(T, name="sep", **kw):
    """AB-B1 -- ring-referenced vertices whose apico-basal span is non-finite or points INWARD.

    THE POLARITY AXIS IS DERIVED AND NOTHING ENFORCES ITS SIGN. `sep/|sep|` is the cell's
    apical-basal direction and no operator owns it: the design declines to write a `cell_polarity`
    operator, because a normalisation of another operator's state is not a mechanism, and the cost of
    that decision is exactly this row. A shell that inverts through itself keeps a perfectly
    plausible energy while apical and basal have swapped.

    OUTWARD IS TESTED AGAINST THE MID-SURFACE NORMAL, NOT THE RADIUS. A radial test only works on a
    star-shaped shell about the origin and would score every buckle as a failure -- which is the
    shape this promotion exists to produce.

    ONLY RING-REFERENCED VERTICES COUNT, and that mask is load-bearing rather than tidy: `cell_die`
    rewrites `nF` and never `Nv`, so a vertex orphaned by an extrusion keeps a stale span that no
    face reads and no reader should score.
    """
    out = []
    for t in range(T.n_rows()):
        sep = _sep(T, t, name)
        if sep is None:
            out.append(float("nan")); continue
        es, et, ef = (np.asarray(x, int) for x in T.half_edges(t))
        pos = T.pos(t); nF = T.nF(t); nV = pos.shape[0]
        n = _vertex_normals_np(pos, es, et, ef, nF, nV)
        used = np.zeros(nV, bool); used[es] = True
        bad = ~np.isfinite(sep).all(axis=1)
        inward = (sep * n).sum(axis=1) <= 0.0
        out.append(float(np.count_nonzero((bad | inward) & used)))
    return out


def apicobasal_span_recorded_fraction(T, name="sep", **kw):
    """AB-B7 -- 1.0 on a row that carries a finite, non-zero span; 0.0 otherwise.

    THE ROW THAT SAYS THE APICOBASAL PATH ACTUALLY RAN. `run_gates` never reads a gate's
    `operators_exercised:` key -- preflight reads id, arms, measures and the per-row keys only -- so
    a gate can NAME a variant and prove nothing about which one it got. A quantity the operator
    itself writes is the only evidence, and it is free: `sep` reaches the trajectory through the
    generic per-set recording path.
    """
    out = []
    for t in range(T.n_rows()):
        sep = _sep(T, t, name)
        ok = (sep is not None and np.isfinite(sep).all()
              and float(np.linalg.norm(sep, axis=1).max(initial=0.0)) > 0.0)
        out.append(1.0 if ok else 0.0)
    return out


def polyhedron_volume_closure(T, name="sep", **kw):
    """AB-B2 -- is each cell's polyhedron CLOSED? Two origins, one volume.

    A closed oriented surface encloses the same volume from ANY origin; an unclosed one, or one with
    a face wound the wrong way, does not. So the entire apico-basal geometry -- both caps, one
    lateral quad per ring edge, the fan triangulation and the orientation convention -- is exercised
    by a single number: the relative discrepancy between the volume taken about the cell's own
    centroid and the volume taken about the world origin.

    It is the only row that would catch a lateral quad wound the wrong way, which is the failure the
    maintained C++ reference needs an explicit `polygonDirections_` array to avoid. Returns the WORST
    cell per row, relative to that cell's own volume.
    """
    out = []
    for t in range(T.n_rows()):
        sep = _sep(T, t, name)
        if sep is None:
            out.append(float("nan")); continue
        es, et, ef = (np.asarray(x, int) for x in T.half_edges(t))
        pos = T.pos(t); nF = T.nF(t)
        a, b = pos + sep, pos - sep
        cen = np.zeros((nF, 3)); cnt = np.zeros(nF)
        np.add.at(cen, ef, pos[es]); np.add.at(cnt, ef, 1.0)
        cen /= np.maximum(cnt, 1.0)[:, None]
        v_o = _cell_polyhedron_volume(a, b, es, et, ef, nF, np.zeros((nF, 3)))
        v_c = _cell_polyhedron_volume(a, b, es, et, ef, nF, cen)
        d = np.abs(v_o - v_c) / np.maximum(np.abs(v_c), 1e-12)
        out.append(float(np.nanmax(d)) if d.size else 0.0)
    return out


def cap_area_ratio(T, name="sep", **kw):
    """AB-C3 -- the mean apical:basal cap-area ratio.

    On a spherical shell of radius R and thickness h the caps scale with their own radii, so the
    closed form is ((R + h/2) / (R - h/2))^2 -- 1.1736111 at R = 5, h = 0.4. It is a CONTROL at R2,
    where `sep` is frozen at (h0/2)n and the caps are therefore exactly what `monolayer_shells`
    would build; it becomes a result at R5, when `sep` is integrated and the ratio is solved rather
    than constructed. Both rows exist so that the difference between them is visible.
    """
    out = []
    for t in range(T.n_rows()):
        sep = _sep(T, t, name)
        if sep is None:
            out.append(float("nan")); continue
        es, et, ef = (np.asarray(x, int) for x in T.half_edges(t))
        pos = T.pos(t); nF = T.nF(t)
        a, b = pos + sep, pos - sep
        # Newell area of each cap ring: |1/2 sum (p x q)| over the directed ring edges
        Aa = np.zeros((nF, 3)); Ab = np.zeros((nF, 3))
        np.add.at(Aa, ef, np.cross(a[es], a[et]))
        np.add.at(Ab, ef, np.cross(b[es], b[et]))
        na = np.linalg.norm(Aa, axis=1); nb = np.linalg.norm(Ab, axis=1)
        r = na / np.maximum(nb, 1e-12)
        out.append(float(np.nanmean(r)) if r.size else float("nan"))
    return out


def euler_closed(T, **kw):
    return [T.nV(t) - _n_edges(T, t) + T.nF(t) for t in range(T.n_rows())]


def valence_fraction(T, valence=3, **kw):
    """Fraction of LIVE vertices with exactly `valence` incident half-edges.

    Counted over `E_srce` of the live faces: on a closed trivalent surface every vertex is the source
    of exactly three half-edges. Orphaned vertices -- the ones `cell_die` leaves inside `Nv` without
    touching the table -- appear with valence 0 and would drag the fraction down, so they are excluded
    by counting only vertices that appear at all.
    """
    out = []
    for t in range(T.n_rows()):
        es, _et, ef = T.half_edges(t)
        live = ef < T.nF(t)
        v, c = np.unique(np.asarray(es)[live], return_counts=True)
        out.append(float((c == valence).mean()) if len(c) else 1.0)
    return out


def mean_neighbours_residual(T, **kw):
    """|2E/F - (6 - 12/F)|, Euler's theorem for a trivalent closed cellular surface."""
    out = []
    for t in range(T.n_rows()):
        F, E = T.nF(t), _n_edges(T, t)
        out.append(abs(2.0 * E / max(F, 1) - (6.0 - 12.0 / max(F, 1))))
    return out


def _radii(T, t):
    p = T.pos(t)
    return np.linalg.norm(p - p.mean(0), axis=1)


def apical_radius(T, **kw):
    """MEDIAN vertex radius about the vertex centroid -- the reference's own definition.

    Not the mean, and the difference is not cosmetic: `tissue.py:329` is
    `float(np.median(np.linalg.norm(v, axis=1)))`, and a just-divided sliver sitting briefly outside
    the surface moves a mean and not a median. The first version of this row used the mean and came
    out at 3.85781 against a threshold of 3.8607 derived from the median -- a FAILING row that was
    measuring a different quantity from the one its threshold described. Two estimators of "apical
    radius" is exactly the way a gate lies.

    CENTROID-REFERENCED, for the reason `apical_map` gives: nothing pins the vesicle to the origin,
    so an origin-referenced radius reads the vesicle's DRIFT as growth.
    """
    return [float(np.median(_radii(T, t))) for t in range(T.n_rows())]


def apical_radius_fold(T, **kw):
    r = apical_radius(T)
    return [x / max(r[0], 1e-12) for x in r]


def aspect_ratio(T, **kw):
    """Equatorial over axial semi-axis, from the gyration tensor's eigenvalues.

    THE REFERENCE'S OWN ESTIMATOR (`tissue.py:336-337`): the 98th percentile of hypot(x, y) over the
    98th percentile of |z|, both about the vertex centroid. The 98th rather than the max because one
    stray vertex should not set the shape of a tissue and a just-divided sliver can sit briefly
    outside the surface.

    THIS REPLACED A GYRATION TENSOR, and the swap is the point rather than a detail. The tensor is a
    perfectly good aspect estimator -- it is more robust than a percentile -- but it is a DIFFERENT
    NUMBER, and it read 1.00787 against a threshold of 1.0193 taken from the percentile pair. A row
    that fails because its threshold and its measurement disagree about what they mean is worse than
    no row: it looks like a finding.
    """
    out = []
    for t in range(T.n_rows()):
        p = T.pos(t)
        v = p - p.mean(0)
        r_eq = float(np.percentile(np.hypot(v[:, 0], v[:, 1]), 98))
        r_ax = float(np.percentile(np.abs(v[:, 2]), 98))
        out.append(r_eq / max(r_ax, 1e-9))
    return out


def scalar_col(T, name, **kw):
    """One of the operators' own cumulative counters, per recorded row (`n_t1`, `n_apop`, ...).

    THE OPERATOR'S NUMBER, NOT A RECONSTRUCTION OF IT. `t1_total_inferred` below reconstructs the
    same quantity from the topology and disagrees by about a factor of two, because a 3D reversible
    network reconnection rewires more than the one edge it is named for. Both are right about
    different things, which is why a gate must read the counter the operator keeps.
    """
    out = []
    for t in range(T.n_rows()):
        v = T.scalar(name, t)
        out.append(0.0 if v is None else float(v))
    return out


def renumber_did_not_act(T, **kw):
    """`renumber_failed`, cumulative -- THE SENTINEL THAT WAS INSTALLED AND NEVER WIRED UP.

    When a cell dies or a T1 drops a face the cells are renumbered, and every per-cell array must be
    permuted to match or the chemistry ends up on the wrong cells. `Hierarchy.renumber_set` does
    that and returns False if it could not. On 23 August it returned False on EVERY call -- its
    guard tested `hasattr(self.levels, "get")` and `levels` is an `nn.ModuleDict`, which has no
    `.get` -- and both call sites discarded the bool. Nineteen twin rows stayed green while the
    chemistry of every run that killed a cell was scrambled; one run went half-NaN from frame 889
    and stamped itself `valid_evidence: True`.

    The fix put this counter in `MeshTable.SCALAR_RECORD` for one stated reason: "so a gate can
    assert it is 0 instead of a human having to notice a printed line". THAT ASSERTION WAS NEVER
    WRITTEN. The counter has been recorded, and unread, ever since: `MEASURES` exposes four thin
    wrappers over `scalar_col` and `scalar_col` itself is not in the table, so a row naming it fails
    preflight. This is the fifth wrapper.
    """
    return scalar_col(T, "renumber_failed")


def t1_total(T, **kw):
    """`edge_flip`'s own accepted-reconnection counter, cumulative."""
    return scalar_col(T, "n_t1")


def n_apop(T, **kw):
    """`cell_die`'s own extrusion counter, cumulative."""
    return scalar_col(T, "n_apop")


def divisions_refused(T, **kw):
    """`cell_divide`'s count of divisions refused for want of buffer, per frame."""
    return scalar_col(T, "div_blocked")


def apop_spill(T, **kw):
    """Material a death could not bequeath without pushing a neighbour out of the integrator's
    basin. It is dropped and counted rather than injected, and it must stay ~0."""
    return scalar_col(T, "apop_spill")


def t1_total_inferred(T, **kw):
    """Cumulative NEW EDGES BETWEEN PRE-EXISTING VERTICES, inferred from the topology.

    NOT THE SAME QUANTITY AS `t1_total`, and the gap is the finding: this reads 2,890 where
    `edge_flip`'s own counter reads 1,499 on gate 00, i.e. about two new old-old edges per accepted
    reconnection. Kept as a separate measure rather than reconciled, because it is the only T1
    diagnostic available on a trajectory recorded before `SCALAR_RECORD` existed.

    A T1 changes the half-edge table WITHOUT changing V, E or F, so the Euler ledger cannot see it;
    what it changes is WHICH VERTEX PAIRS are edges.

    THE DIVISIONS HAVE TO BE SUBTRACTED, AND NOT BY COUNTING THEM. The first version of this took
    "pairs present now, absent before, minus 3 per new face", on the grounds that a septum adds three
    edges. It returned 16,169 against the reference's 1,499. A division does not add three pairs: it
    SPLITS two existing edges -- each split deletes one pair and creates two -- and then adds the
    septum, so five pairs appear and two disappear for a net of three. Subtracting three left two per
    division, about 13,000 of them, and the row read as a ten-fold T1 excess that did not exist.

    SO THE DIVISIONS ARE EXCLUDED BY GEOMETRY RATHER THAN BY ARITHMETIC. Every pair a division
    creates involves at least one vertex that did not exist a tick ago -- the two edge midpoints. A
    T1's new pair joins two vertices that were both already there. Counting only new pairs whose
    endpoints are both below the previous tick's `Nv` therefore counts flips and nothing else, with
    no coefficient to get wrong.
    """
    ticks = getattr(T, "ticks", None)
    total, out = 0, []
    prev, prev_nv = None, 0
    for t in range(T.n_rows()):
        es, et, ef = T.half_edges(t)
        live = ef < T.nF(t)
        a, b = np.asarray(es)[live], np.asarray(et)[live]
        cur = set(map(tuple, np.sort(np.stack([a, b], 1), axis=1).tolist()))
        step_ok = t > 0 and (ticks is None or ticks[t] - ticks[t - 1] == 1)
        if prev is not None and step_ok:
            total += sum(1 for (x, y) in (cur - prev) if x < prev_nv and y < prev_nv)
        prev, prev_nv = cur, T.nV(t)
        out.append(total)
    return out


def t1_rail_fraction(T, max_flips=30, **kw):
    """Fraction of the `edge_flip` calls that DID something and hit the cap while doing it.

    Read off the operator's own cumulative counter, so a call's delivery is exact. A call that hit
    the rail wanted more reconnections than it got, so the T1 rate on that frame is a property of
    `max_flips` and not of the tissue -- which is the whole reason this row exists beside
    `t1_total`.
    """
    tt = t1_total(T)
    per = [tt[i] - tt[i - 1] for i in range(1, len(tt))]
    calls = [x for x in per if x > 0]
    if not calls:
        return [0.0]
    return [float(sum(1 for x in calls if x >= max_flips) / len(calls))]


def doubling_time_hours(T, **kw):
    """Mean cell-cycle length. NEEDS THE `units:` BLOCK -- the reducer converts, and raises without
    one; this returns the cycle in FRAMES."""
    c = cell_count(T)
    d = np.log2(max(c[-1], 1) / max(c[0], 1))
    n = getattr(T, "ticks", None)
    span = (n[-1] - n[0]) if n else (len(c) - 1)
    return [float(span / max(d, 1e-9))]


def mean_cell_diameter(T, **kw):
    """sqrt of the mean apical area per cell, in simulation length units.

    Area from the closed surface, not from `cell__area`: the state block is recorded before the
    frame's divisions are applied, so on a dividing frame it is a few entries short -- which is the
    same off-by-a-few that made the renderer's blue flicker.

    THE TRIANGLE FAN IS ABOUT EACH FACE'S OWN CENTROID, not about the body's. The first version
    fanned every half-edge from the global centre, which measures the volume swept rather than the
    surface: it returned 50.9 um for a cell that is 7.7 um across, a factor of 6.6, and it would have
    been read as a model that grows by inflating its cells.
    """
    out = []
    for t in range(T.n_rows()):
        p, (es, et, ef) = T.pos(t), T.half_edges(t)
        nF = T.nF(t)
        live = ef < nF
        a, b, f = np.asarray(es)[live], np.asarray(et)[live], np.asarray(ef)[live]
        # each face's centroid, as the mean of its own half-edge sources
        cen = np.zeros((nF, 3))
        cnt = np.zeros(nF)
        np.add.at(cen, f, p[a])
        np.add.at(cnt, f, 1.0)
        cen /= np.maximum(cnt, 1)[:, None]
        tri = 0.5 * np.linalg.norm(np.cross(p[a] - cen[f], p[b] - cen[f]), axis=1)
        area = np.zeros(nF)
        np.add.at(area, f, tri)
        out.append(float(np.sqrt(area.mean())))
    return out


def spheroid_diameter(T, **kw):
    return [2.0 * r for r in apical_radius(T)]


# ============================================================================== junction measures
def _edges(T, t):
    """(vi, vj, length) over the LIVE half-edges of row t, and the half-edge index they came from."""
    p, (es, et, ef) = T.pos(t), T.half_edges(t)
    live = np.asarray(ef) < T.nF(t)
    a, b = np.asarray(es)[live], np.asarray(et)[live]
    return a, b, np.linalg.norm(p[b] - p[a], axis=1), live


def myosin_aligned(T, name="myo", **kw):
    """|len(myo) - len(E_srce)| per row. A BOOKKEEPING row, and the one that made it necessary.

    `junction_myosin` writes `myo` for the half-edge arrays as they were, and `edge_flip` and
    `cell_divide` then rewire and lengthen them within the same tick. On the 401-frame nominal, 56
    of 200 archived snapshots carried a myosin array 6 to 1,356 entries short of the edge arrays,
    and every reader indexes it positionally -- so each of those frames coloured, averaged and
    thresholded the wrong junctions. `junction_sync` re-keys it by vertex pair; this asserts the
    re-keying happened.
    """
    out = []
    for t in range(T.n_rows()):
        v = T.edge_col(name, t)
        es, _et, _ef = T.half_edges(t)
        out.append(0 if v is None else abs(len(v) - len(es)))
    return out


def myosin_mean(T, name="myo", **kw):
    out = []
    for t in range(T.n_rows()):
        v = T.edge_col(name, t)
        _a, _b, _l, live = _edges(T, t)
        out.append(0.0 if v is None or len(v) < len(live)
                   else float(np.asarray(v)[live].mean()))
    return out


def myosin_dispersion(T, name="myo", pct=98, **kw):
    """p98 / mean over the live junctions -- how UNEVEN the myosin is.

    The mean is pinned near `activity` by construction, so it says nothing; a belt that is doing
    something has a tail. This is the number that separates "myosin is present" from "myosin is
    localised", and the two have completely different mechanics.
    """
    out = []
    for t in range(T.n_rows()):
        v = T.edge_col(name, t)
        _a, _b, _l, live = _edges(T, t)
        if v is None or len(v) < len(live):
            out.append(0.0); continue
        w = np.asarray(v)[live]
        m = float(w.mean())
        out.append(float(np.percentile(w, pct) / m) if m > 1e-12 else 0.0)
    return out


def hot_junction_fraction(T, name="myo", above=1.5, **kw):
    out = []
    for t in range(T.n_rows()):
        v = T.edge_col(name, t)
        _a, _b, _l, live = _edges(T, t)
        if v is None or len(v) < len(live):
            out.append(0.0); continue
        w = np.asarray(v)[live]
        m = float(w.mean())
        out.append(float((w > above * m).mean()) if m > 1e-12 else 0.0)
    return out


def myosin_length_correlation(T, name="myo", **kw):
    """Pearson r between a junction's myosin and its LENGTH, over the live junctions.

    THE ROW THAT DISTINGUISHES THE TWO BELTS, and it is independent of the dispersion row in a way
    `hot_junction_fraction` was not. `junction_myosin` has two keyings: `length` accumulates myosin
    on SHORT junctions, which homogenises lengths and suppresses intercalation; `tension` is the
    destabilising one the germband-extension literature describes and produces T1s. Both raise the
    dispersion identically -- so a "myosin is localised" row cannot tell them apart, and this can:
    the length-keyed belt gives a NEGATIVE correlation, the tension-keyed one does not.

    WHY IT REPLACED `hot_junction_fraction`. That row asked for at least 2% of junctions above 1.5x
    the mean while its sibling asserted p98/mean >= 1.20. The two are not independent, they are in
    CONTRADICTION: p98 = 1.42 means, by the definition of a 98th percentile, that exactly 2% of
    junctions are above 1.42 -- so at most 2% can be above 1.5, and the pair could never both hold.
    It measured 0.0079 against a demand for 0.02. A gate row that cannot pass while its neighbour
    passes is not a strict row, it is a wrong one.
    """
    out = []
    for t in range(T.n_rows()):
        v = T.edge_col(name, t)
        _a, _b, L, live = _edges(T, t)
        if v is None or len(v) < len(live):
            out.append(0.0); continue
        w = np.asarray(v)[live]
        if w.std() < 1e-12 or L.std() < 1e-12:
            out.append(0.0); continue
        out.append(float(np.corrcoef(w, L)[0, 1]))
    return out


def mean_junction_length(T, **kw):
    return [float(_edges(T, t)[2].mean()) for t in range(T.n_rows())]


def mean_junction_length_fold(T, **kw):
    L = mean_junction_length(T)
    return [x / max(L[0], 1e-12) for x in L]


def junction_persistence(T, lag=1, **kw):
    """Fraction of a row's junctions that were also junctions `lag` recorded rows earlier.

    A tissue that intercalates loses junctions; one that only grows keeps them and adds more. This
    separates the two, which cell count and radius cannot.
    """
    out, prev = [], None
    for t in range(T.n_rows()):
        a, b, _l, _live = _edges(T, t)
        cur = set(map(tuple, np.sort(np.stack([a, b], 1), axis=1).tolist()))
        out.append(1.0 if prev is None else float(len(cur & prev) / max(len(prev), 1)))
        prev = cur
    return out


def pos_max_delta(A, B, **kw):
    """max |pos_A - pos_B| over every row both sides recorded -- the two-arm neutrality row.

    Refuses rather than truncates when the two arms disagree in length or in live count: two runs
    that are not the same length are not a controlled comparison, and silently comparing the first
    N rows of each is how a neutrality claim comes to be about a prefix.
    """
    n = min(A.n_rows(), B.n_rows())
    if A.n_rows() != B.n_rows():
        raise ValueError(f"the two arms recorded {A.n_rows()} and {B.n_rows()} rows")
    worst = 0.0
    for t in range(n):
        pa, pb = A.pos(t), B.pos(t)
        if pa.shape != pb.shape:
            raise ValueError(f"row {t}: {pa.shape} live vertices against {pb.shape}")
        worst = max(worst, float(np.abs(pa - pb).max()))
    return [worst]


def t1_rate_delta(A, B, **kw):
    """(T1 per cell per frame in A) - (in B). Negative means A suppresses intercalation."""
    def rate(T):
        n = T.n_rows()
        cells = float(np.mean([T.nF(t) for t in range(n)]))
        return t1_total(T)[-1] / max(cells * max(n - 1, 1), 1e-9)
    return [rate(A) - rate(B)]


def t1_rate_per_cell_per_frame(T, **kw):
    n = T.n_rows()
    cells = float(np.mean([T.nF(t) for t in range(n)]))
    return [t1_total(T)[-1] / max(cells * max(n - 1, 1), 1e-9)]


# ============================================================================== particle measures
def particle_count(T, **kw):
    return [T.nF(t) for t in range(T.n_rows())]


def out_of_box(T, lo=0.03125, hi=0.96875, **kw):
    """Particles outside the clamp `mpm_gather` enforces, `[2dx, 1-2dx]` at n_grid 64.

    THE CLAMP IS THE OPERATOR'S OWN, so a particle outside it means the clamp did not run, not that
    the material moved: this is a bookkeeping row about the code, not a statement about the block.
    """
    out = []
    for t in range(T.n_rows()):
        p = T.pos(t)
        out.append(int(((p < lo) | (p > hi)).any(axis=1).sum()))
    return out


def centroid_height(T, axis=1, **kw):
    """The block's centre of mass along `axis`. Every particle carries the same mass here (one
    `p_vol` from a uniform seed), so the mean position IS the centre of mass."""
    return [float(T.pos(t)[:, axis].mean()) for t in range(T.n_rows())]


def free_fall_acceleration(T, dt=0.0032, axis=1, frames=60, **kw):
    """The acceleration of the centroid over the opening frames, in box units per second squared.

    THE CLOSED-FORM ROW OF THIS GATE, in the paper's exact sense: does the implementation reproduce
    the physics it was GIVEN? The spec hands `gravity` g = 2.5 and nothing else acts until the block
    reaches the floor, so the centroid must follow y0 - g t^2 / 2 and a quadratic fit must return
    -2.5. It is not a statement about matrices; it is a statement about whether a force declared once
    per tick and consumed by sixteen substeps is applied once or sixteen times -- which is precisely
    the error the paper names, "a force applied once per substep where it should be applied once per
    step", and which would show up here as -40.0 rather than -2.5.

    `frames` is bounded well before the first impact (frame ~287 in the archived runs).
    """
    y = np.asarray(centroid_height(T, axis=axis)[:frames], float)
    t = np.arange(len(y)) * float(dt)
    if len(y) < 3:
        return [0.0]
    c = np.polyfit(t, y, 2)                      # y = c0 t^2 + c1 t + c2  ->  a = 2 c0
    return [float(2.0 * c[0])]


def min_centroid_height(T, axis=1, **kw):
    return [min(centroid_height(T, axis=axis))]


def strand_length(T, per_strand=60, **kw):
    """Mean end-to-end length of the seeded fibres, in box units.

    `ecm_seed` lays `n_fibres` strands of `per_strand` CONTIGUOUS particles, so strand i is the slice
    [i*per_strand : (i+1)*per_strand] and its end-to-end length is the distance between the two ends.
    Measured at frame 0 this is the seeded fibre length; later it is what the deformation did to it.
    """
    out = []
    for t in range(T.n_rows()):
        p = T.pos(t)
        n = (len(p) // per_strand) * per_strand
        q = p[:n].reshape(-1, per_strand, 3)
        out.append(float(np.linalg.norm(q[:, -1] - q[:, 0], axis=1).mean()))
    return out


def strand_length_um(T, per_strand=60, **kw):
    return strand_length(T, per_strand=per_strand)


def min_radius_from_centre(T, centre=(0.5, 0.5, 0.5), **kw):
    """The closest any material point comes to the domain centre, per frame.

    THE NON-PENETRATION STATEMENT, in the one form a prescribed surface allows. `mesh_contact`
    replays a tissue whose radius is known per frame, so "no matrix inside the epithelium" is
    exactly "min |p - centre| >= the tissue's radius". It needs no stress, no contact counter and no
    module global -- only the positions the trajectory already holds.
    """
    c = np.asarray(centre, float)
    return [float(np.linalg.norm(T.pos(t) - c, axis=1).min()) for t in range(T.n_rows())]


def contact_penetration_um(T, surface_radius=0.15, centre=(0.5, 0.5, 0.5), **kw):
    """How far the nearest material point sits INSIDE the prescribed surface, at the last frame.

    THE NUMBER THAT ONLY MEANS SOMETHING IN MICROMETRES. As a fraction of a unit box a penetration of
    0.0146 reads as a rounding error; at this calibration it is 17 um, which is two cell diameters.
    That conversion is the paper's own worked example of why the measurement tier needs a `units:`
    block -- the same quantity was once reported as "0.82 grid cells", sounded small, and was
    described as an improvement.

    Returned in SIMULATION units; the row's `unit: um` converts it through the spec's `units:`.
    """
    c = np.asarray(centre, float)
    t = T.n_rows() - 1
    return [max(0.0, float(surface_radius) - float(np.linalg.norm(T.pos(t) - c, axis=1).min()))]


def median_displacement(T, **kw):
    """Median |p(t) - p(0)| over the material points that are present at both ends.

    A MEDIAN, not a mean: a handful of points flicked out of a contact zone move by a box width and
    would carry a mean on their own. The failure this guards against is a matrix that is being
    FLICKED rather than pushed -- measured once at k = 4000, where the median displacement collapsed
    from 0.085 to 0.0005 while the peak grew, because a huge acceleration on the contact layer
    accelerates that layer out of the zone before it can transmit anything to the material behind.
    """
    p0 = T.pos(0)
    out = []
    for t in range(T.n_rows()):
        p = T.pos(t)
        n = min(len(p), len(p0))
        out.append(float(np.median(np.linalg.norm(p[:n] - p0[:n], axis=1))))
    return out


def final_stress_p99_Pa(T, band_scale=2.0, bands=8, pct=99, **kw):
    """p99 of the final frame's von Mises stress, from the recorded colour band.

    THE BAND IS WHAT REACHES DISK, and it is a lossy readout: `ecm_stress` writes
    `round(clip(vm / band_scale, 0, 1) * (bands - 1))` into `node_type`, so the stress is recovered
    as `band * band_scale / (bands - 1)` and is quantised to `band_scale / (bands - 1)` = 0.286
    stress units. That is coarse and it is stated rather than hidden: this row can distinguish 100 Pa
    from 1000 Pa, which is what its literature interval asks, and it cannot resolve 10%.

    IT IS THE FINAL FRAME ONLY, because `node_type` is a per-node buffer written once into the npz
    rather than a per-frame array -- the run's last state. A stress TRACE would need `ecm_stress` to
    publish into a recorded state block instead of into a colour channel and a module global.
    """
    z, sname = getattr(T, "z", None), getattr(T, "s", None)
    k = f"{sname}__node_type"
    if z is None or k not in z.files:
        raise KeyError("no node_type in this trajectory -- ecm_stress did not run, or the writer "
                       "did not record it")
    band = np.asarray(z[k], float)
    return [float(np.percentile(band, pct) * band_scale / max(bands - 1, 1))]


MEASURES = {
    "myosin_aligned": myosin_aligned,
    "myosin_mean": myosin_mean,
    "myosin_dispersion": myosin_dispersion,
    "hot_junction_fraction": hot_junction_fraction,
    "myosin_length_correlation": myosin_length_correlation,
    "mean_junction_length": mean_junction_length,
    "mean_junction_length_fold": mean_junction_length_fold,
    "junction_persistence": junction_persistence,
    "pos_max_delta": pos_max_delta,
    "t1_rate_delta": t1_rate_delta,
    "t1_rate_per_cell_per_frame": t1_rate_per_cell_per_frame,
    "particle_count": particle_count,
    "out_of_box": out_of_box,
    "centroid_height": centroid_height,
    "free_fall_acceleration": free_fall_acceleration,
    "min_centroid_height": min_centroid_height,
    "strand_length": strand_length,
    "strand_length_um": strand_length_um,
    "min_radius_from_centre": min_radius_from_centre,
    "contact_penetration_um": contact_penetration_um,
    "median_displacement": median_displacement,
    "final_stress_p99_Pa": final_stress_p99_Pa,
    "cell_count": cell_count,
    "cell_count_delta": cell_count_delta,
    "vertex_count": vertex_count,
    "occ_vs_mesh": occ_vs_mesh,
    "cap_area_ratio": cap_area_ratio,
    "polyhedron_volume_closure": polyhedron_volume_closure,
    "apicobasal_span_recorded_fraction": apicobasal_span_recorded_fraction,
    "apicobasal_span_invalid_count": apicobasal_span_invalid_count,
    "renumber_did_not_act": renumber_did_not_act,
    "topology_ledger": topology_ledger,
    "nonfinite_count": nonfinite_count,
    "reservoir_fraction": reservoir_fraction,
    "euler_closed": euler_closed,
    "valence_fraction": valence_fraction,
    "mean_neighbours_residual": mean_neighbours_residual,
    "apical_radius": apical_radius,
    "apical_radius_fold": apical_radius_fold,
    "aspect_ratio": aspect_ratio,
    "t1_total": t1_total,
    "t1_total_inferred": t1_total_inferred,
    "n_apop": n_apop,
    "divisions_refused": divisions_refused,
    "apop_spill": apop_spill,
    "t1_rail_fraction": t1_rail_fraction,
    "doubling_time_hours": doubling_time_hours,
    "mean_cell_diameter_um": mean_cell_diameter,
    "spheroid_diameter_um": spheroid_diameter,
}

# WHICH MEASURES ARE PHYSICAL, and what they are measured in. The reducer converts through the
# spec's `units:` block and RAISES if none is declared -- which is what stops a dimensionless run
# from quoting a micrometre. A name absent from here is dimensionless by declaration.
PHYSICAL = {
    "strand_length_um": ("length", "um"),
    "final_stress_p99_Pa": ("stress", "Pa"),
    "contact_penetration_um": ("length", "um"),
    "doubling_time_hours": ("time", "hours"),
    "mean_cell_diameter_um": ("length", "um"),
    "spheroid_diameter_um": ("length", "um"),
}

REDUCERS = {
    "all": lambda s: s,
    "first": lambda s: s[0],
    "last": lambda s: s[-1],
    "min": lambda s: min(s),
    "max": lambda s: max(s),
    "mean": lambda s: float(np.mean(s)),
    "count": lambda s: len(s),
    "series": lambda s: s,
}


def _all(v, pred):
    return all(pred(x) for x in v) if isinstance(v, (list, tuple)) else pred(v)


ASSERTS = {
    "eq": lambda v, a: _all(v, lambda x: x == a),
    "ne": lambda v, a: _all(v, lambda x: x != a),
    "ge": lambda v, a: _all(v, lambda x: x >= a),
    "le": lambda v, a: _all(v, lambda x: x <= a),
    "gt": lambda v, a: _all(v, lambda x: x > a),
    "lt": lambda v, a: _all(v, lambda x: x < a),
    "within": lambda v, a: _all(v, lambda x: abs(x - a[0]) <= a[1]),
    "interval": lambda v, a: _all(v, lambda x: a[0] <= x <= a[1]),
}
