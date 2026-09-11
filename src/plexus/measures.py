"""One entry point for every quantity read off a run.

THREE CONSUMERS USED TO COMPUTE THE SAME NUMBERS SEPARATELY: the movie's curve panel
(`live_movie._curve_row`, seven quantities on a live or replayed level), the gates
(`tools/gate_measures.MEASURES`, sixty-one series over a trajectory reader) and the regression
fingerprints (`tools/regression_lib.checkpoint_metrics`, twenty metrics at a frame). A cell count
was coded three times and a shell radius twice, with two definitions (mean, median). This module
holds every one of them, moved verbatim, and the three consumers read it.

WHAT IS HERE
  readers    `Traj` -- the eleven accessors every row measure is written against -- and its
             implementations over the core `trajectory.npz` (`CoreTraj`, `ParticleTraj`) and the
             archived okuda `traj.npz` (`OkudaTraj`); `open_traj` picks one from a path.
  curve      `curve_row(H, lvl, q, ntype, nt, cell_cols)` -- one row [nt, 2] of (mean, sd) for a
             curve quantity on a level, live (engine tensors) or replayed (`_ReplayLevel`). The
             body is the renderer's, unchanged; `count:<set>` is the one addition.
  frame      `checkpoint_metrics(d, fr)` -- the fingerprint's metrics at one recorded frame.
  registry   `MEASURES`: name -> Measure(kind, fn, dim). `kind` is "curve" (one row on a level),
             "row" (a series over a Traj, the gate form) or "frame" (a dict at one recorded
             frame). `dim` is the quantity's dimension in the vocabulary of `plexus.units`
             ("count", "length", "area", "volume", "fraction", or None when undeclared), so a
             consumer converts at display time and never inside a measure.

BIT-IDENTITY IS THE CONTRACT OF THE MOVE. `tools/quantities_baseline.py` captures what the three
consumers compute on the registered working points before and after; the move is right when the
two captures are equal in every bit. Each consumer keeps its own array dtypes (the curve panel
reads float32 off the engine, the gates read float64 off the file), which is why the readers
are adapters and the bodies are untouched.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# ============================================================================== registry
@dataclass(frozen=True)
class Measure:
    name: str
    kind: str                    # "curve" | "row" | "frame"
    fn: Optional[Callable]       # None for a curve quantity, which `curve_row` dispatches on by name
    dim: Optional[str] = None    # plexus.units dimension, or None when the quantity has none
    doc: str = ""


MEASURES: dict[str, Measure] = {}


def register(name: str, kind: str, fn=None, dim=None, doc=""):
    if name in MEASURES and MEASURES[name].fn is not fn:
        raise ValueError(f"measure {name!r} is already registered ({MEASURES[name].kind})")
    MEASURES[name] = Measure(name, kind, fn, dim, doc or (getattr(fn, "__doc__", "") or "").strip().split("\n")[0])
    return fn


def register_rows(table: dict, dims: dict | None = None):
    """Register a gate-style table {name: fn(T, **kw) -> series} as row measures."""
    dims = dims or {}
    for n, f in table.items():
        register(n, "row", f, dims.get(n))


def dims_of(kind: str | None = None) -> dict[str, str]:
    return {n: m.dim for n, m in MEASURES.items() if m.dim and (kind is None or m.kind == kind)}


def to_physical(name: str, value, units):
    """A measured number in physical units, through the run's declared `units:` block; the value
    itself when the measure declares no dimension or the run declares no scale."""
    m = MEASURES.get(name)
    if m is None or m.dim is None or not units:
        return value
    from plexus.units import to_physical as _tp
    s = _tp(1.0, m.dim, units)
    return value if s is None else value * float(s)


# ============================================================================== readers
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




class _MeshView:
    """A read-only mesh table with the cell set's per-cell blocks overlaid.

    WHY A VIEW AND NOT A DICT COPY. The live path's `m` is a `MeshTable` holding CUDA tensors and
    the replay path's is a plain dict; both are read the same two ways, `m["nF"]` and
    `m.get(name)`, and nothing in the renderer writes to either. Wrapping keeps one code path for
    both and copies nothing.

    THE MESH WINS EVERY NAME CLASH. An overlay is consulted only for a name the table does not
    have, so a cell-set block called `area` -- and there is one -- can never shadow a face column
    or `nF`. The overlay is what the mesh no longer carries, never a second opinion about what it
    does.
    """

    __slots__ = ("_m", "_c")

    def __init__(self, m, cell_cols):
        self._m = m
        self._c = cell_cols or {}

    def __getitem__(self, k):
        try:
            return self._m[k]
        except KeyError:
            return self._c[k]

    def __contains__(self, k):
        return k in self._m or k in self._c

    def get(self, k, default=None):
        v = self._m.get(k, None)
        return self._c.get(k, default) if v is None else v

    def __getattr__(self, a):                 # `reindex_faces`, `snapshot`, ... stay reachable
        return getattr(self._m, a)


# ============================================================================== curve quantities
# WHAT EACH CURVE IS, SO THE PANEL CAN CONVERT IT. Until this existed the volume panel appended
# `um^3` to a raw simulation number and was wrong by `length_um ** 3`. The unit is DERIVED from the
# same declaration that converts the number. `myosin` is deliberately absent: it is a per-junction
# activity with no declared dimension, and UNKNOWN prints bare.
CURVE_DIMS = {"area": "area", "volume": "volume", "radius": "length",
              "cells": "count", "phase": "fraction", "cycle_progress": "fraction"}
CURVE_QUANTITIES = ("cells", "area", "volume", "radius", "myosin", "phase", "cycle_progress")
for _q in CURVE_QUANTITIES:
    register(_q, "curve", None, CURVE_DIMS.get(_q))
register("count", "curve", None, "count", "the live count of a named set: `count:<set>`")


def curve_row(H, lvl, q, ntype, nt, cell_cols):
    """[nt, 2] of (mean, sd) for `q` at the level's CURRENT frame -- one row of `_curve_series`.

    THE SAME BODY SERVES THE REPLAY AND THE LIVE PASS. The replay sets `lvl.t` and calls this per
    recorded frame with numpy arrays behind `lvl.get`; the live pass calls it once per frame with
    CUDA tensors behind the same accessors, so everything that feeds numpy or a CPU geometry call
    goes through `_np`, a no-op on an array and `.detach().cpu().numpy()` on a tensor.
    """
    import torch
    def _np(v):
        return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    row = np.full((nt, 2), np.nan)
    if isinstance(q, str) and q.startswith("count:"):
        # THE LIVE COUNT OF A NAMED SET, mesh or not: `quantity: count:integrin`. Live, `occ` is the
        # engine's occupancy vector; on replay `_ReplayLevel._occ` is [T, n] and `t` picks the row.
        name = q[len("count:"):]
        species = None
        if ":" in name:                                  # `count:protein:integrin` -- one species of a typed set
            name, species = name.split(":", 1)
        try:
            lv = H.level(name)
        except Exception:                                # noqa: BLE001
            return row
        occ = getattr(lv, "_occ", None)
        frame = int(getattr(lvl, "t", getattr(lv, "t", 0)))
        if occ is not None and getattr(occ, "ndim", 1) == 2:
            # the replay advances `t` on the level the curves read (the mesh set); a counted set
            # follows that frame, not its own `t`, which the series loop never touches
            occ = occ[frame]
        if occ is None:
            occ = getattr(lv, "occ", None)
        if occ is None:
            n = float(getattr(lv, "n", 0))
        else:
            live = _np(occ).astype(bool)
            if species is not None:
                names = list(getattr(lv, "type_names", None) or [])
                if species not in names:
                    raise ValueError(f"count:{name}:{species}: {name!r} has no species {species!r} (types: {names})")
                # THE TYPE COLUMN AT THE SAME FRAME AS THE OCCUPANCY. A reserve slot is typed 0
                # until an operator wakes it with its species; `lv.node_type` reads the level's OWN
                # `t`, which the series loop never advances, so every slot born after frame 0 was
                # counted under type 0: 7,910 "nuclei" for 670 cells, the mitochondria born by
                # organelle_express. The per-frame column, when the archive kept one, at `frame`.
                ntt = getattr(lv, "_node_type_t", None)
                nt = ntt[frame] if ntt is not None else getattr(lv, "node_type", None)
                if nt is None:
                    return row
                live = live & (_np(nt).astype(int) == names.index(species))
            n = float(live.sum())
        row[0] = (n, 0.0)
        return row
    m = getattr(lvl, "mesh", None)
    if m is None or not int(m.get("nF", 0) or 0):
        return row
    _es, _et, _ef = (torch.as_tensor(_np(m[k])) for k in ("E_srce", "E_trgt", "E_face"))
    nF = int(m["nF"])
    # THE CELL SET'S BLOCKS, on the curve path too. `phase` is a block on the cell set and
    # not a face column, and this is the only reader of it outside `_mesh_face_rgb`; the
    # panel is the one that says what fraction of the tissue is in G1/S/G2/M, so without
    # the overlay it would draw four empty series and look like a run with no cycle.
    m = _MeshView(m, cell_cols(H, lvl, nF))
    k = (np.zeros(nF, int) if ntype is None
         else np.asarray(ntype)[np.clip(np.arange(nF), 0, len(ntype) - 1)].astype(int))
    if q == "cells":
        for j in range(nt):
            row[j] = (float((k == j).sum()), 0.0)
        return row
    if q == "phase":
        # PERCENT OF THE POPULATION IN EACH PHASE, four series in one panel. A cycle is
        # read as a DISTRIBUTION -- what fraction is where -- and the count of cells does
        # not show it: a tissue cycling steadily and one frozen in G1 both just grow. The
        # four fractions sum to 100 at every frame, so the panel is also its own check.
        #
        # It ignores `ntype`: the partition here is the PHASE, and splitting phases by cell
        # type as well would be sixteen series in one panel, which is a different plot.
        v = m.get("phase")
        if v is None:
            return row
        a = np.rint(np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach")
                               else v, float).ravel()[:nF]).astype(int)
        for j in range(min(nt, 4)):
            row[j] = (100.0 * float(np.mean(a == j)) if a.size else np.nan, 0.0)
        return row
    if q == "cycle_progress":
        v = m.get("cycle_progress")
        if v is None:
            return row
        vv = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v,
                        float).ravel()[:nF]
        for j in range(nt):
            sel = k == j
            if sel.any():
                row[j] = (float(np.nanmean(vv[sel])), float(np.nanstd(vv[sel])))
        return row
    if q == "myosin":
        v = m.get("e_myo")
        if v is None:
            return row
        ef = _np(m["E_face"]); live = ef < nF
        vv = np.asarray(v, float)[live]
        ke = k[np.clip(ef[live].astype(int), 0, nF - 1)]
        for j in range(nt):
            sel = ke == j
            if sel.any():
                row[j] = (float(np.nanmean(vv[sel])), float(np.nanstd(vv[sel])))
        return row
    import torch
    if q == "radius":
        nv = int(m["Nv"])
        P = _np(lvl.get("pos")[:nv])
        r = np.linalg.norm(P - P.mean(0), axis=1)
        for j in range(nt):                   # per VERTEX, so the type split is by face
            row[j] = (float(np.mean(r)), float(np.std(r)))
        return row
    from plexus.operators.vertex_ops import face_geometry_3d, wedge_apex
    nv = int(m["Nv"])
    pos = torch.as_tensor(_np(lvl.get("pos")[:nv]), dtype=torch.float64)
    a, _p, _c, _v = face_geometry_3d(pos, _es, _et, _ef, nF, apex=wedge_apex(m, pos))
    a = a.numpy()
    # `area` FOLLOWS `volume`'S RULE, AND FOR THE SAME REASON. What `face_geometry_3d`
    # returns is the MID-SURFACE area, and on an apico-basal run the mid-surface is not a
    # boundary of anything: the cell's surface is the polyhedron's -- two caps and one wall
    # per ring edge -- and that is the `S` the energy's `kappa_s` integrates. A panel
    # labelled "cell area" showing the area of a surface the cell does not have is the same
    # defect the volume branch below documents, one term along in the same functional.
    #
    # Falls back to the mid-surface when the run carries no `sep`, where it IS the cell.
    if q == "area":
        _sa = lvl.get("sep")
        if _sa is not None and int(_sa.shape[0]) >= nv:
            from plexus.operators.vertex_ops import apicobasal_geometry_3d
            _ss = torch.as_tensor(np.asarray(_sa[:nv].detach().cpu()
                                             if hasattr(_sa, "detach") else _sa[:nv]),
                                  dtype=torch.float64)
            a = apicobasal_geometry_3d(pos, _ss, _es, _et, _ef, nF)[1].numpy()
    if q == "volume":
        # THE VOLUME THE ENERGY DEFENDS, WHICH ON AN APICO-BASAL RUN IS NOT `_v`.
        # `face_geometry_3d` returns the origin-referenced WEDGE volume -- the cone from
        # the world origin out to the cell's mid-surface ring -- and that is the quantity
        # `cell_grow` scales and `cell_divide` triggers on, but it is NOT the cell. It
        # rises when the SHELL's radius rises, with the cell unchanged, which is how a
        # tissue comes to divide without growing. `cell_mechanics[model: apicobasal]`
        # defends the polyhedron: two caps and one wall per ring edge, by the divergence
        # theorem, and that is what a plot labelled "cell volume" has to show.
        #
        # Falls back to the wedge when the run carries no `sep`, so a mid-surface spec
        # asking for this curve still gets the only volume it has.
        try:
            _sp = lvl.get("sep")
        except Exception:                                # noqa: BLE001
            _sp = None
        if _sp is not None and int(_sp.shape[0]) >= nv:
            from plexus.operators.vertex_ops import apicobasal_geometry_3d
            _s = torch.as_tensor(np.asarray(_sp[:nv].detach().cpu()
                                            if hasattr(_sp, "detach") else _sp[:nv]),
                                 dtype=torch.float64)
            a = apicobasal_geometry_3d(pos, _s, _es, _et, _ef, nF)[0].numpy()
        else:
            a = _v.numpy()
    for j in range(nt):
        sel = k == j
        if sel.any():
            row[j] = (float(np.nanmean(a[sel])), float(np.nanstd(a[sel])))
    return row


# ============================================================================== frame metrics (fingerprints)
def _occ(d, key, fr):
    o = d.get(key)
    return None if o is None else o[fr].astype(bool)


def checkpoint_metrics(d, fr: int) -> dict:
    """Everything the fingerprint records at one frame. Keys absent from the run are absent here."""
    m: dict = {}
    files = set(d.files)
    # ---- a vertex tissue --------------------------------------------------------------------
    if "vertex__pos" in files:
        occ = _occ(d, "vertex__occ", fr)
        P = d["vertex__pos"][fr]
        P = P[occ] if occ is not None else P
        if P.shape[0]:
            c = P.mean(0)
            r = np.linalg.norm(P - c, axis=1)
            m["r_med"] = float(np.median(r))
            m["r_p10"] = float(np.percentile(r, 10))
            m["r_p90"] = float(np.percentile(r, 90))
            m["roughness"] = float(r.std() / max(r.mean(), 1e-12))
            m["centroid"] = [float(v) for v in c]
            m["vertices"] = int(P.shape[0])
            # out-of-plane spread, the sheet's own number; on a shell it is just the radius
            m["z_sd"] = float(P[:, -1].std())
        if "cell__occ" in files:
            m["cells"] = int(d["cell__occ"][fr].sum())
        if "vertex__mesh_offsets" in files:
            off = d["vertex__mesh_offsets"]
            if fr + 1 < off.shape[0]:
                m["half_edges"] = int(off[fr + 1] - off[fr])   # live half-edges, from the mesh table
        if "cell__area" in files and "cell__occ" in files:
            a = d["cell__area"][fr].reshape(-1)
            co = d["cell__occ"][fr].astype(bool)
            a = a[co] if a.shape[0] == co.shape[0] else a
            if a.size:
                m["area_mean"] = float(a.mean())
        for k in ("cell__ndiv", "cell__age"):
            if k in files and "cell__occ" in files:
                v = d[k][fr].reshape(-1)
                co = d["cell__occ"][fr].astype(bool)
                v = v[co] if v.shape[0] == co.shape[0] else v
                if k == "cell__ndiv" and v.size:
                    m["ndiv_sum"] = int(v.sum())   # sum of per-cell generation counts, monotone
        for k in ("cell__chem", "vertex__chem"):
            if k in files:
                v = d[k][fr]
                oc = _occ(d, k.split("__")[0] + "__occ", fr)
                v = v[oc] if oc is not None and v.shape[0] == oc.shape[0] else v
                if v.size and float(np.abs(v).max()) > 0.0:   # an all-zero chem block is unused
                    m["chem_max"] = float(v[:, 0].max())
                    m["chem_min"] = float(v[:, 0].min())
    # ---- an MPM body -------------------------------------------------------------------------
    mp = [k[:-5] for k in files if k.endswith("__pos") and k != "vertex__pos"]
    if mp:
        mset = "mpm_particle" if "mpm_particle" in mp else sorted(mp)[0]
        m["mpm_set"] = mset
        occ = _occ(d, mset + "__occ", fr)
        X = d[mset + "__pos"][fr]
        X = X[occ] if occ is not None else X
        if X.shape[0]:
            m["mpm_n"] = int(X.shape[0])
            m["mpm_spread"] = float(X.std(0).mean())
            m["mpm_bbox"] = float((X.max(0) - X.min(0)).mean())
            m["mpm_centroid"] = [float(v) for v in X.mean(0)]
            if "vertex__pos" in files and "r_med" in m:
                c = np.asarray(m["centroid"])
                m["mpm_inside"] = int((np.linalg.norm(X - c, axis=1) < m["r_med"]).sum())
    return m


def deaths_by(d, fr: int) -> int | None:
    """Cells that have left the live set since frame 0, when the run has a cell set."""
    if "cell__occ" not in d.files:
        return None
    return int(max(0, int(d["cell__occ"][0].sum()) - int(d["cell__occ"][fr].sum())))

for _k in ("cells", "half_edges", "vertices", "r_med", "r_p10", "r_p90", "roughness", "z_sd", "area_mean",
           "ndiv_sum", "deaths_or_net_loss", "chem_max", "chem_min", "mpm_n", "mpm_spread", "mpm_bbox", "mpm_inside"):
    _dim = {"cells": "count", "half_edges": "count", "vertices": "count", "r_med": "length", "r_p10": "length",
            "r_p90": "length", "z_sd": "length", "area_mean": "area", "mpm_n": "count", "mpm_inside": "count",
            "ndiv_sum": "count", "deaths_or_net_loss": "count", "mpm_bbox": "length", "mpm_spread": "length"}.get(_k)
    if "fp:" + _k not in MEASURES:
        register("fp:" + _k, "frame", checkpoint_metrics, _dim, "fingerprint checkpoint metric")


# THE GATE ROWS REGISTER THEMSELVES on import of the package's copy, so the registry is complete
# whichever consumer imports this module first.
from plexus import measures_rows as _rows  # noqa: E402,F401
