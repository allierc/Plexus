"""exp09 adhesion_sorting -- the rulers: does a random two-type mixture demix, and who ends inside?

    exp09.sorting   the demixing index over the run, its half-time, the inside index, the aggregate's integrity
    exp09.layers    the radial order of the types' median radii at one row (Toda 2018's programmed layers)
    exp09.architecture  Cerchiari 2015's architecture class I-V of the last row (a seed mean is the paper's frequency)

THE DEMIXING INDEX. Over the touching pairs of live cells (shared edges on a mesh,
`common.neighbour_pairs`; Delaunay edges no longer than one fixed length on a point set, below), f_same is the fraction of pairs whose two cells have the
same type. A random mixture with type fractions phi_k has f_rand = sum_k phi_k^2 in expectation, so

    demix = (f_same - f_rand) / (1 - f_rand)

reads 0 for any random mixture whatever its fractions, 1 when no pair is heterotypic, and below 0 for a
checkerboard. It is the "same-type neighbour fraction" of the experiment's G-sort and G-identity,
rescaled so the two arms of different fractions share one zero.

THE INSIDE INDEX. On the largest connected cluster of the touching graph, each cell's radius from the
cluster's centroid is divided by R = rms radius * sqrt((D+2)/D) -- the edge of a uniform D-ball with
that rms, so R is the aggregate's radius whatever its size. Then

    inside = (mean rho_B - mean rho_A) / perfect(phi_A, D)

with perfect(phi, D) = D/(D+1) * [(1 - phi^((D+1)/D)) / (1 - phi) - phi^(1/D)], the same difference for
a perfect core of A (fraction phi_A of the cluster) inside a shell of B. So +1 = A perfectly engulfed
by B, 0 = no radial order, about -1 = B inside A. "A" is `types[0]`, "B" `types[1]` (default 0 and 1).
THE AGGREGATE MUST NOT WRAP: on a periodic box a cluster across the boundary has a meaningless
centroid, so the arms that read `inside` run in an open (non-periodic) world.

THE SURFACE SHARE. Graner & Glazier 1992 Fig. 2b reads who is inside from the EDGE boundary: the
dark (cohesive) cells' contact with the medium falls from 0.027 of all boundary to 0.00 by 300 MCS
(Monte Carlo steps) while the light cells' rises to 0.057. The point-set form: the aggregate's surface
cells are the vertices of the cut Delaunay complex's boundary facets (a facet of exactly one kept
simplex; a simplex is kept when all its edges are within `cut` x the median edge), on a mesh the faces
owning a half-edge with no twin. surface_A = (A's share of the surface cells) / (A's share of the
cluster): 1 for a random mixture, 0 when no A cell touches the medium.

THE LOGARITHMIC PACE. Graner & Glazier 1992 Fig. 2a: the homotypic boundary grows LINEARLY IN log t
(fits between 5 and 4000 MCS, R^2 > 0.97). log_r2 is the R^2 of demix against log10(frame) over the
rise, from frame `fit_from` to the first row at 95 % of the plateau; log_slope its slope, demix per
decade of frames -- a number independent of the time unit, so it compares with the paper's without a
calibration (log_decades says how many decades the fit spans). Rows are read every `every` frames
AND at `n_log` log-spaced frames, so the early decades are sampled.

THE SURFACE FRACTION (Phase 2, the talk's number). surface_frac_A = the fraction of the aggregate's
surface cells that are A -- absolute, not relative to A's share: 0 when A is fully inside, 1 when A
covers the whole rim. With `surface_series: true` it is read at every row, and t90 is the first row
at which it has made 90 % of its first-to-last change (t90_hours through the spec's declared
`general.units.time_s` and `dt`).

VALIDITY. On a point run, where no cell can turn inside out, valid_frac_min is the smallest share of
cells in the largest touching cluster over the rows read (the aggregate stays whole). ON A MESH, inverted_max = the largest fraction of live cells, over the rows read, whose
signed area is <= 0 (turned inside out). exp 3's growth auditor is not used for it: it scores a
tissue that does not grow at 2 ("smooth but no growth"), and these aggregates are not meant to grow.

THE HALF-TIME. t_half is the first recorded frame at which demix reaches demix_first + 0.5 x
(demix_plateau - demix_first), demix_plateau being the mean over the last 10 % of the rows read; None
when the index never rises. Frames, not minutes: turning it into the papers' time is Stage 0's one
calibrated number.

TYPES. A point run records `<set>__node_type` [N], and `<set>__node_type_t` [T, N] when a signal
changed a cell's type during the run (read first); a mesh run a face block named by `type_block` (default `node_type`). A run
whose two types are two SETS (`sets: [A, B]`) -- the only way a per-type-PAIR adhesion table exists
with `pair_potential`, whose well depth is one number per pair of sets -- is read with type = the
set's index in that list.

TOUCHING IS A FIXED LENGTH, NOT A FRACTION OF THE CURRENT SPACING. On a point run two cells touch
when they share a Delaunay edge no longer than L = `cut` x the median Delaunay edge of ROW 0 (or
`contact`, a length in world units, when given), the same L at every row, because cells in this
experiment neither divide nor grow. Batch 1 is why (Findings 2-3): a cut relative to the CURRENT
median followed the cells when they collapsed onto each other (median edge 0.0071 -> 0.0025) and
kept a gas of evaporated cells "connected", so a condensate in a gas read as a sorted aggregate.
`compression` = median touching length at the last row over row 0's: 1 for cells that keep their
size, well below 1 when the law has no hard core and the cells pile onto each other.

ROWS ARE NOT FRAMES. The engine strides the trajectory (`record_cap`, default 10,000 rows), so row r
is frame r x frames_per_row, frames_per_row = general.n_frames / (rows - 1). t_half is reported in
both; the logarithmic fit is unchanged by the stride (a constant factor in time shifts log t).
"""
from __future__ import annotations

import numpy as np

from .common import CoreTraj, cells, finite, neighbour_pairs, register_run


def _rows(T, every, n_log=0):
    n = T.n_rows()
    r = set(list(range(0, n, max(1, int(every)))) + [n - 1])
    if n_log and n > 2:
        r |= set(np.unique(np.rint(np.logspace(0, np.log10(n - 1), int(n_log))).astype(int)).tolist())
    return sorted(r)


def _types(T, t, c, type_block="node_type"):
    """Integer type label of each live cell of `c` (aligned with c.x)."""
    if isinstance(T, CoreTraj):
        # A MESH RUN RECORDS ITS CELL TYPES AS THE CELL SET'S `node_type` (per row as `node_type_t` when
        # they change) -- the same two columns a point run records -- read directly, because the static
        # column is [N], not [T, N], and the block reader indexes it by row. A face block named
        # `type_block` is the fallback for a run that stores types as state.
        if type_block == "node_type":
            for k in (f"{T.c}__node_type_t", f"{T.c}__node_type"):
                if T.c and k in T.z.files:
                    a = np.asarray(T.z[k])
                    a = a[t] if a.ndim == 2 else a
                    return np.asarray(a, np.int64)[c.slot]
        a = c.block(type_block)
        if a is None:
            raise KeyError(f"mesh run records neither {T.c}__node_type nor a face block {type_block!r}")
        return np.rint(a[:, 0]).astype(np.int64)
    k = f"{T.s}__node_type_t" if f"{T.s}__node_type_t" in T.z.files else f"{T.s}__node_type"
    if k not in T.z.files:
        raise KeyError(f"point run records no {k}")
    a = np.asarray(T.z[k])
    a = a[t] if a.ndim == 2 else a
    return np.asarray(a, np.int64)[c.slot]


def _frame(T, t, sets=None, type_block="node_type"):
    """(x [n, D], type [n], Cells or None) at row t -- from one typed set, a mesh, or two sets."""
    if sets:
        xs, ty = [], []
        for k, name in enumerate(sets):
            p = np.asarray(T.z[f"{name}__pos"][t], float)
            if f"{name}__occ" in T.z.files:
                p = p[np.asarray(T.z[f"{name}__occ"][t], bool)]
            xs.append(p)
            ty.append(np.full(len(p), k, np.int64))
        return np.concatenate(xs), np.concatenate(ty), None
    c = cells(T, t)
    return c.x, _types(T, t, c, type_block), c


def _delaunay_edges(x):
    from scipy.spatial import Delaunay
    D = x.shape[1]
    if len(x) < D + 2:
        return np.zeros((0, 2), np.int64), np.zeros(0)
    s = Delaunay(x).simplices
    e = np.unique(np.sort(np.concatenate([s[:, [i, j]] for i in range(D + 1) for j in range(i + 1, D + 1)]), 1), axis=0)
    return e, np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)


def touching(T, t, x, c, L):
    """[m, 2] pairs of touching cells (rows of x): shared edges on a mesh, Delaunay edges <= L on points."""
    if isinstance(T, CoreTraj):
        return neighbour_pairs(T, t, c)
    e, le = _delaunay_edges(x)
    return e[le <= L]


def demix_index(typ, pairs):
    """(f_same - f_rand) / (1 - f_rand) over the touching pairs; nan with no pairs or one type."""
    if len(pairs) == 0:
        return float("nan")
    f_same = float(np.mean(typ[pairs[:, 0]] == typ[pairs[:, 1]]))
    _, n = np.unique(typ, return_counts=True)
    phi = n / n.sum()
    f_rand = float((phi ** 2).sum())
    return (f_same - f_rand) / (1.0 - f_rand) if f_rand < 1.0 else float("nan")


def largest_cluster(n, pairs):
    """Indices of the largest connected component of the touching graph on n cells."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if n == 0:
        return np.zeros(0, np.int64)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    _, lab = connected_components(g, directed=False)
    return np.flatnonzero(lab == np.bincount(lab).argmax())


def surface_cells(T, t, x, c=None, L=None, cut=2.0):
    """Boolean [n]: the cells of `x` on the aggregate's free surface (see the module docstring).
    A simplex is kept when all its edges are <= L (default `cut` x the median Delaunay edge of x)."""
    if isinstance(T, CoreTraj):
        es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
        big = int(max(es.max(initial=0), et.max(initial=0))) + 1
        has_twin = np.isin(es * big + et, et * big + es)
        faces = np.unique(ef[~has_twin])
        return np.isin(c.slot, faces)
    from scipy.spatial import Delaunay
    n, D = x.shape
    if n < D + 2:
        return np.ones(n, bool)
    s = Delaunay(x).simplices
    e = np.concatenate([s[:, [i, j]] for i in range(D + 1) for j in range(i + 1, D + 1)])
    L_e = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1).reshape(-1, len(s)).T    # [simplex, edge]
    Lu = np.linalg.norm(np.diff(x[np.unique(np.sort(e, 1), axis=0)], axis=1)[:, 0], axis=1)
    s = s[(L_e <= (L if L is not None else cut * np.median(Lu))).all(1)]
    f = np.concatenate([np.delete(s, k, axis=1) for k in range(D + 1)])            # every facet
    f = np.sort(f, 1)
    u, cnt = np.unique(f, axis=0, return_counts=True)
    out = np.zeros(n, bool)
    out[u[cnt == 1].ravel()] = True
    out[np.setdiff1d(np.arange(n), s.ravel())] = True                              # in no kept simplex
    return out


def surface_share(surf, typ, k):
    """Type k's share of the surface cells over its share of all cells: 1 random, 0 buried."""
    phi = np.mean(typ == k)
    if phi == 0 or not surf.any():
        return float("nan")
    return float(np.mean(typ[surf] == k) / phi)


def log_fit(ts, dm, fit_from=1):
    """(R^2, slope per decade, decades spanned) of demix against log10(frame), over the rise."""
    ts, dm = np.asarray(ts, float), np.asarray(dm, float)
    ok = np.isfinite(dm) & (ts >= max(1, fit_from))
    if ok.sum() < 4:
        return None, None, None
    tail = max(1, int(round(0.1 * len(dm))))
    plateau, first = np.nanmean(dm[-tail:]), dm[0]
    if not plateau > first:
        return None, None, None
    hit = np.flatnonzero(dm >= first + 0.95 * (plateau - first))
    end = ts[hit[0]] if len(hit) else ts[-1]
    ok &= ts <= end
    if ok.sum() < 4:
        return None, None, None
    X, Y = np.log10(ts[ok]), dm[ok]
    p = np.polyfit(X, Y, 1)
    res = Y - np.polyval(p, X)
    ss = np.sum((Y - Y.mean()) ** 2)
    return (1.0 - np.sum(res ** 2) / ss if ss > 0 else None), float(p[0]), float(X.max() - X.min())


def inverted_fraction(T, t):
    """Mesh runs: the fraction of live cells whose signed area (shoelace over its half-edges, in the
    sheet's xy plane, CCW seen from +z positive) is <= 0 -- a cell turned inside out. None off a mesh."""
    if not isinstance(T, CoreTraj):
        return None
    es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
    P = T.pos(t)
    nF = T.nF(t)
    a = np.zeros(nF)
    np.add.at(a, ef, 0.5 * (P[es, 0] * P[et, 1] - P[et, 0] * P[es, 1]))
    occ = T.occ(T.c, t) if T.c else None
    live = np.ones(nF, bool) if occ is None else np.asarray(occ[:nF], bool)
    return float(np.mean(a[live] <= 0)) if live.any() else None


def perfect_gap(phi, D):
    """mean rho_B - mean rho_A for a perfect core of A (fraction phi) in a shell of B, in a uniform D-ball of radius 1."""
    if not 0.0 < phi < 1.0:
        return float("nan")
    return D / (D + 1.0) * ((1.0 - phi ** ((D + 1.0) / D)) / (1.0 - phi) - phi ** (1.0 / D))


def norm_radius(x):
    """Each point's distance from the centroid over the radius of the uniform D-ball with the same rms."""
    D = x.shape[1]
    r = np.linalg.norm(x - x.mean(0), axis=1)
    R = np.sqrt(np.mean(r ** 2)) * np.sqrt((D + 2.0) / D)
    return r / max(R, 1e-30)


def inside_index(x, typ, A=0, B=1):
    """(mean rho_B - mean rho_A) / perfect_gap(phi_A): +1 A engulfed, 0 no order, ~-1 B engulfed."""
    a, b = typ == A, typ == B
    if a.sum() < 2 or b.sum() < 2:
        return float("nan")
    ab = a | b
    rho = norm_radius(x[ab])
    phi = a[ab].mean()
    return (rho[b[ab]].mean() - rho[a[ab]].mean()) / perfect_gap(phi, x.shape[1])


def sorting(T, every=10, types=(0, 1), cut=2.0, contact=None, sets=None, type_block="node_type",
            n_log=16, fit_from=1, surface_series=False, **_):
    """Demixing index, inside index, surface shares, largest-cluster share and compression at every
    `every`-th row, `n_log` log-spaced rows and the last. Touching is one fixed length L (module
    docstring): `contact` (world units) if given, else `cut` x the median Delaunay edge of row 0."""
    A, B = int(types[0]), int(types[1])
    ts = _rows(T, every, n_log)
    x0, _, c0 = _frame(T, ts[0], sets, type_block)
    if contact is not None:
        L = float(contact)
    elif isinstance(T, CoreTraj):
        L = None
    else:
        L = cut * float(np.median(_delaunay_edges(x0)[1]))
    dm, ins, big, med, sfa, inv = [], [], [], [], [], []
    for t in ts:
        x, typ, c = _frame(T, t, sets, type_block)
        p = touching(T, t, x, c, L)
        dm.append(demix_index(typ, p))
        k = largest_cluster(len(x), p)
        big.append(len(k) / max(len(x), 1))
        ins.append(inside_index(x[k], typ[k], A, B))
        med.append(float(np.median(np.linalg.norm(x[p[:, 0]] - x[p[:, 1]], axis=1))) if len(p) else float("nan"))
        inv.append(inverted_fraction(T, t))
        if surface_series:
            _sf = surface_cells(T, t, x[k], c=_Sub(c, k) if c is not None else None, L=L, cut=cut)
            sfa.append(float(np.mean(typ[k][_sf] == A)) if _sf.any() else float("nan"))
    surf = surface_cells(T, ts[-1], x[k], c=_Sub(c, k) if c is not None else None, L=L, cut=cut)
    sA, sB = surface_share(surf, typ[k], A), surface_share(surf, typ[k], B)
    r2, slope, dec = log_fit(ts, dm, fit_from)
    dm = np.asarray(dm, float)
    tail = max(1, int(round(0.1 * len(ts))))
    plateau = float(np.nanmean(dm[-tail:]))
    first = float(dm[0])
    t_half = None
    if np.isfinite(plateau) and np.isfinite(first) and plateau > first:
        hit = np.flatnonzero(dm >= first + 0.5 * (plateau - first))
        t_half = int(ts[hit[0]]) if len(hit) else None
    gen = ((T.spec or {}).get("general") or {}) if hasattr(T, "spec") else {}
    nf = gen.get("n_frames")
    fpr = float(nf) / max(T.n_rows() - 1, 1) if nf else None
    # HOURS FROM THE SPEC'S OWN DECLARED UNITS (`general.units.time_s` per unit of sim time, `dt` per
    # frame), never assumed; None when the spec declares no time unit.
    hpf = (float(gen.get("dt", 1.0)) * float((gen.get("units") or {}).get("time_s")) / 3600.0
           if (gen.get("units") or {}).get("time_s") else None)
    sf_last = float(np.mean(typ[k][surf] == A)) if surf.any() else None
    t90 = None
    if surface_series and len(sfa) > 1 and np.isfinite(sfa[0]) and np.isfinite(sfa[-1]) and abs(sfa[-1] - sfa[0]) > 1e-9:
        prog = (np.asarray(sfa) - sfa[0]) / (sfa[-1] - sfa[0])
        hit = np.flatnonzero(prog >= 0.9)
        t90 = int(ts[hit[0]]) if len(hit) else None
    last_in = finite(ins[-1])
    return {"demix_first": finite(first), "demix_last": finite(dm[-1]), "demix_plateau": finite(plateau),
            "demix_rise": finite(plateau - first), "t_half": t_half,
            "t_half_frames": None if (t_half is None or fpr is None) else finite(t_half * fpr),
            "frames_per_row": finite(fpr),
            "inside_first": finite(ins[0]), "inside_last": last_in,
            "inside": None if last_in is None else ("A" if last_in > 0 else "B"),
            "surface_A_last": finite(sA), "surface_B_last": finite(sB),
            "surface_frac_A_last": finite(sf_last),
            "surface_frac_A_series": [finite(v) for v in sfa] if surface_series else None,
            "t90_row": t90,
            "t90_hours": None if (t90 is None or fpr is None or hpf is None) else finite(t90 * fpr * hpf),
            "share_A": finite(np.mean(typ[k] == A)),
            "inverted_max": finite(max(v for v in inv if v is not None)) if any(v is not None for v in inv) else None,
            "valid_frac_min": (finite(1.0 - max(v for v in inv if v is not None)) if any(v is not None for v in inv)
                               else finite(min(big))),
            "log_r2": finite(r2), "log_slope": finite(slope), "log_decades": finite(dec),
            "largest_cluster_last": finite(big[-1]), "largest_cluster_min": finite(min(big)),
            "contact_length": finite(L) if L is not None else None,
            "compression": finite(med[-1] / med[0]) if med[0] else None,
            "demix_series": [finite(v) for v in dm], "inside_series": [finite(v) for v in ins],
            "cluster_series": [finite(v) for v in big], "rows": ts}


class _Sub:
    """The cells of `c` restricted to rows `k` (their slots), for the mesh surface test."""

    def __init__(self, c, k):
        self.slot = c.slot[k]


def layers(T, t=-1, target=None, min_share=0.02, sep_min=0.8, cut=2.0, contact=None, sets=None,
           type_block="node_type", **_):
    """Radial layer order of the largest cluster at row t, centre outwards.

    Each type holding at least `min_share` of the cluster is placed at its MEDIAN normalised radius
    rho (module docstring); `order` lists the types by that median, centre first. `separation` is,
    over consecutive types in that order, the smallest probability that a cell of the inner type lies
    closer to the centre than a cell of the outer one (Mann-Whitney AUC: 0.5 = interleaved, 1 = fully
    nested). `match` is 1 when `order` equals `target` (the paper's order, declared in gates.yaml)
    AND `separation` >= `sep_min` (0.8, model choice), else 0.

    WHY MEDIANS AND NOT SHELLS (2026-09-26). The first version cut the cluster into equal-count
    shells and took each shell's majority. Toda et al. 2018's red middle layer is ONE cell thick
    (fig. S2C: green core to 4.3 cell radii, red 4.3-5.3, blue 5.3-7.8): around a 40-cell 2D core
    that is ~22 cells, never the majority of a 40-cell shell, so the shells could not see the
    structure the gate is about. A type's median radius sees a layer of any thickness."""
    t = T.n_rows() - 1 if t is None or t < 0 else int(t)
    x, typ, c = _frame(T, t, sets, type_block)
    if contact is not None:
        L = float(contact)
    else:
        x0 = _frame(T, 0, sets, type_block)[0]
        L = cut * float(np.median(_delaunay_edges(x0)[1])) if c is None or not isinstance(T, CoreTraj) else None
    p = touching(T, t, x, c, L)
    k = largest_cluster(len(x), p)
    x, typ = x[k], typ[k]
    rho = norm_radius(x)
    kinds, counts = np.unique(typ, return_counts=True)
    keep = kinds[counts >= min_share * len(typ)]
    med = {int(q): float(np.median(rho[typ == q])) for q in keep}
    order = sorted(med, key=med.get)
    seps = []
    for a_, b_ in zip(order, order[1:]):
        ra, rb = rho[typ == a_], rho[typ == b_]
        seps.append(float(np.mean(ra[:, None] < rb[None, :])))
    sep = min(seps) if seps else None
    out = {"order": "".join(str(o) for o in order), "n_layers": len(order), "separation": finite(sep),
           "median_rho": {str(q): finite(v) for q, v in med.items()},
           "shares": {str(int(q)): finite(n / len(typ)) for q, n in zip(kinds, counts)}, "row": t, "dim": x.shape[1]}
    if target is not None:
        out["match"] = 1.0 if (order == [int(v) for v in target] and sep is not None and sep >= sep_min) else 0.0
    return out


def architecture(T, types=(0, 1), cut=2.0, contact=None, sets=None, type_block="node_type",
                 mixed_below=0.3, ring=0.75, core=0.25, **_):
    """Cerchiari et al. 2015's five tissue architectures (Fig. 1D, 3F, 3I), read off the last row.

    With A the tracked type (Cerchiari's MEP) and s its surface fraction (A's share of the aggregate's
    surface cells, absolute):
        V    mixed         demix < `mixed_below` (0.3)                  salt and pepper, no layers
        I    A outside     s >= `ring` (0.75)                           "correct": A rings the other type
        II   A mostly out  0.5 <= s < `ring`                            A-majority rim, not closed
        III  A mostly in   `core` < s < 0.5                             the other type's rim, not closed
        IV   A inside      s <= `core` (0.25)                           "inverted": the other type rings A
    The thresholds are model choices; the paper classifies by eye from the images. `is_I` .. `is_V`
    (1 or 0) are returned so that a seed mean IS the paper's frequency of that architecture."""
    A = int(types[0])
    t = T.n_rows() - 1
    x, typ, c = _frame(T, t, sets, type_block)
    if contact is not None:
        L = float(contact)
    elif isinstance(T, CoreTraj):
        L = None
    else:
        L = cut * float(np.median(_delaunay_edges(_frame(T, 0, sets, type_block)[0])[1]))
    p = touching(T, t, x, c, L)
    k = largest_cluster(len(x), p)
    d = demix_index(typ, p)
    surf = surface_cells(T, t, x[k], c=_Sub(c, k) if c is not None else None, L=L, cut=cut)
    s = float(np.mean(typ[k][surf] == A)) if surf.any() else float("nan")
    if not np.isfinite(d) or d < mixed_below:
        cls = "V"
    elif s >= ring:
        cls = "I"
    elif s >= 0.5:
        cls = "II"
    elif s > core:
        cls = "III"
    else:
        cls = "IV"
    out = {"class": cls, "surface_frac_A": finite(s), "demix": finite(d)}
    for k_ in ("I", "II", "III", "IV", "V"):
        out[f"is_{k_}"] = 1.0 if cls == k_ else 0.0
    return out


register_run("exp09.sorting", sorting, None, "demixing index, half-time, inside index, aggregate integrity")
register_run("exp09.layers", layers, None, "radial layer order of the aggregate, centre outwards")
register_run("exp09.architecture", architecture, None, "Cerchiari 2015 architecture class I-V of the last row")
