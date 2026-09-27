"""What every experiment ruler of exp06-15 shares: opening a run, reading cells, finding neighbours.

WHY THIS IS NOT A SECOND FRAMEWORK. `plexus.measures` already reads every trajectory layout through
one facade (`open_traj` -> `CoreTraj` for a mesh, `ParticleTraj` for a point set) and keeps one
registry of named quantities. The experiment rulers are registered INTO that registry with a new
kind, "run": a function of a whole trajectory, `fn(T, **kw) -> dict of named numbers`, because the
questions these experiments ask -- how fast a front moved, whether a bud grew on one axis and not
the other, how clone sizes are distributed -- are about a run, not about one row. Nothing under
`src/` is edited: registration happens when this package is imported.

THE TWO CELL LAYOUTS, and the one trap. On a mesh run (`CoreTraj`) a cell is a face; its blocks come
from `T.state(block, t)`, cropped to the live prefix `nF`, and its liveness from `occ(cell_set)`.
On a point run (`ParticleTraj`) a cell is a particle; `T.pos(t)` is ALREADY masked by occupancy but
`T.state(block, t)` is NOT -- it returns every buffer slot. `cells(T, t)` hides that difference:
it returns positions and a block reader that are aligned row for row, whichever the layout.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plexus.measures import CoreTraj, ParticleTraj, open_traj, register  # noqa: E402


# ============================================================================ opening a run
def run_dir(spec: str) -> str:
    """`group/name` under graphs_data, or a directory path as given."""
    if os.path.isdir(spec):
        return spec
    from plexus.paths import graphs_data_path
    group, name = spec.split("/", 1)
    return graphs_data_path(group, name)


def open_run(spec: str):
    """A `Traj` over the run, with `.dir` and `.spec` (the spec as run, or {}) attached."""
    d = run_dir(spec)
    T = open_traj(d)
    T.dir = d
    try:
        import yaml
        T.spec = yaml.safe_load(open(os.path.join(d, "spec.yaml"))) or {}
    except Exception:                                                        # noqa: BLE001
        T.spec = {}
    return T


def spec_op(T, name: str) -> dict:
    """The first operator or seed entry of the run's spec whose `op` is `name`, or {}."""
    for sec in ("operators", "seed"):
        for o in (T.spec.get(sec) or []):
            if isinstance(o, dict) and o.get("op") == name:
                return o
    return {}


def n_rows(T) -> int:
    return T.n_rows()


# ============================================================================ cells
class Cells:
    """The live cells of one row: positions `x` [n, D], their buffer slots `slot` [n], and `block()`."""

    def __init__(self, T, t, x, slot):
        self.T, self.t, self.x, self.slot = T, t, x, slot

    def __len__(self):
        return len(self.slot)

    def block(self, name):
        """[n, width] for the live cells, or None if the run did not record the block."""
        a = self.T.state(name, self.t)
        if a is None:
            return None
        a = np.asarray(a, float)
        a = a[:, None] if a.ndim == 1 else a
        if len(a) <= int(self.slot.max(initial=-1)):
            return None
        return a[self.slot]


def cells(T, t) -> Cells:
    """Live cells at row t, aligned across positions and blocks (see the module docstring)."""
    if isinstance(T, CoreTraj):
        nF = T.nF(t)
        occ = T.occ(T.c, t) if T.c else None
        live = np.ones(nF, bool) if occ is None else np.asarray(occ[:nF], bool)
        cen = T.state("centroid", t)
        if cen is None:
            cen = face_centroids(T, t)
        slot = np.flatnonzero(live)
        return Cells(T, t, np.asarray(cen, float)[slot], slot)
    occ = T.occ(T.s, t)
    slot = np.flatnonzero(occ) if occ is not None else np.arange(T.nF(t))
    x = np.asarray(T.z[f"{T.s}__pos"][t], float)[slot]
    return Cells(T, t, x, slot)


def face_centroids(T, t) -> np.ndarray:
    """Mean of each face's vertices, from the half-edge table (for runs that record no centroid)."""
    es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
    P = T.pos(t)
    nF = T.nF(t)
    s = np.zeros((nF, P.shape[1]))
    np.add.at(s, ef, P[es])
    n = np.bincount(ef, minlength=nF).astype(float)
    return s / np.maximum(n, 1)[:, None]


# ============================================================================ neighbours
def neighbour_pairs(T, t, c: Cells | None = None, cut: float = 2.0) -> np.ndarray:
    """[m, 2] index pairs INTO `cells(T, t)` of cells that touch.

    On a mesh: two faces touch when they share an edge -- half-edge (a, b) of one face and (b, a) of
    the other. On a point set: the Delaunay triangulation, dropping edges longer than `cut` x the
    median edge (the hull's long spurious edges)."""
    c = c or cells(T, t)
    if isinstance(T, CoreTraj):
        es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
        big = int(max(es.max(initial=0), et.max(initial=0))) + 1
        key, twin = es * big + et, et * big + es
        o = np.argsort(key)
        k_sorted = key[o]
        j = np.searchsorted(k_sorted, twin)
        j = np.clip(j, 0, len(k_sorted) - 1)
        ok = k_sorted[j] == twin
        fa, fb = ef[ok], ef[o[j[ok]]]
        m = fa < fb
        pairs = np.unique(np.stack([fa[m], fb[m]], 1), axis=0)
        # faces -> rows of `c` (dead faces dropped)
        row = -np.ones(int(max(pairs.max(initial=0), c.slot.max(initial=0))) + 1, np.int64)
        row[c.slot] = np.arange(len(c.slot))
        r = row[pairs]
        return r[(r >= 0).all(1)]
    from scipy.spatial import Delaunay
    x = c.x
    if len(x) < 4:
        return np.zeros((0, 2), np.int64)
    tri = Delaunay(x[:, :2] if x.shape[1] == 2 or np.ptp(x[:, -1]) < 1e-9 else x)
    s = tri.simplices
    e = np.concatenate([s[:, [i, j]] for i in range(s.shape[1]) for j in range(i + 1, s.shape[1])])
    e = np.unique(np.sort(e, 1), axis=0)
    L = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)
    return e[L <= cut * np.median(L)]


# ============================================================================ registry and scoring
def register_run(name: str, fn, dim=None, doc=""):
    """A whole-run measure `fn(T, **kw) -> dict`, in the same registry as the gate rows."""
    return register(name, "run", fn, dim, doc)


def partial(value, full, zero) -> float:
    """Fraction of a gate's points, linear from the `zero` line (0) to the `full` line (1), clipped.
    Works both ways: `full > zero` means larger is better, `full < zero` smaller is better."""
    if value is None or not np.isfinite(value) or full == zero:
        return 0.0
    return float(np.clip((value - zero) / (full - zero), 0.0, 1.0))


def finite(x):
    """A JSON-safe float (None for nan/inf)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
