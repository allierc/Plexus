"""exp08 planar_polarity -- the rulers: does junctional exchange order the sheet, and do clones and flow turn it?

    exp08.polarity   per-cell polarity vectors from the two junctional complexes; global order, local
                     order, alignment with the cue axis, and the mean axis's turn, over the run
    exp08.clone      rings of wild-type cells around a declared clone: how far out, and which way, the
                     neighbours' polarity is turned toward or away from the clone

THE POLARITY OF A CELL IS READ FROM ITS JUNCTION SIDES, not from a block the model writes about itself.
Each half-edge h (cell f's side of one junction, from vertex s to vertex t) carries the amounts of the
two complexes on that side, `e_<a>` and `e_<b>` (Frizzled-like and Van Gogh-like: a is the side the
hair grows from). With l_h the edge length and n_h the unit normal of the edge IN THE SHEET PLANE,
oriented away from the cell's centroid,

    p_f = sum over h in f of (a_h - b_h) * l_h * n_h          the cell's polarity vector (points to its a-side)
    s_f = sum over h in f of (a_h + b_h) * l_h                the cell's total junctional complex
    asym_f = |p_f| / s_f                                       0 unpolarised, 1 all a on one side, all b opposite

A run that records no half-edge columns may name a per-cell vector block instead (`vector_block`);
the ruler then reads p_f from it and `asym` is not available.

THE ORDER PARAMETERS, on the unit vectors u_f = p_f / |p_f| of the cells that are not on the sheet's
free boundary (a boundary cell has sides with no neighbour to exchange with; `drop_boundary`):

    order   = | mean over cells of u_f |                       global: 1 all parallel, ~sqrt(pi / 4N) random
    local   = mean over touching pairs of u_i . u_j            domains: 1 all parallel, 0 random
    cue     = mean over cells of u_f . c                       signed alignment with the declared cue axis c
    axis_deg        the angle of the mean vector sum_f u_f in the sheet plane, degrees from the plane's first axis
    turn_deg        how far the mean arrow turned from row `turn_from` (or `turn_from_frac` of the way
                    through the rows) to the last row, UNWRAPPED along the sampled series: an arrow
                    that goes round by 200 degrees reads 200, not -160
    turn_axial_deg  the same for the AXIAL (head-tail blind) mean angle, half the angle of sum_f exp(2 i theta_f),
                    unwrapped on the doubled angle -- for a paper that reads the axis as a line. THE
                    ENDPOINT DIFFERENCE WRAPPED TO +-90, which this replaced, read an axis that crossed
                    the +-90 seam once as 180 degrees off (-93 stored as +87; exp08 round-5 judge). The
                    series is sampled every `every` rows; a turn faster than 90 (axial) or 180 (arrow)
                    degrees between two samples cannot be unwrapped and would be misread

    rc      = Burak & Shraiman 2009's correlation radius (their Fig. 5C legend), in cell spacings:
              rc = sum_k r_k s_k / sum_k s_k,   s_k = mean of u_i . u_j over pairs whose centroids are
              r_k = k cell spacings apart (bins of one spacing; k = 0 is a cell with itself, s_0 = 1),
              summed up to the first bin where s_k <= 0 or `rc_max` spacings. 0 for an uncorrelated
              field, rc_max / 2 for a perfectly aligned one. The cut at the first zero is a model
              choice: the paper integrates over a periodic lattice where s(r) decays monotonically,
              and on a free-edged sheet the negative tail of a random field would otherwise divide
              a noise by a noise. The cell spacing is the median centroid distance of touching cells.

`order_chance` = sqrt(pi / (4 N)) is the expected `order` of N independent random unit vectors in the
plane (the mean length of a 2D random walk of N unit steps, over N), so a reader can see chance at the
run's own size. `local` has chance 0 and a spread of about sqrt(1 / (2 n_pairs)).

THE CLONE is a declared per-cell block (`clone_block`, default `mutant`, > 0.5 = in the clone). Its
wild-type neighbours are sorted into rings by contact-graph distance (ring 1 touches the clone). In ring
k, toward_k = mean over its cells of u_f . r_f, r_f the unit vector from the cell to the nearest clone
cell: +1 all point AT the clone, -1 all point AWAY, ~0 unaffected (a sheet aligned by the cue reads ~0
round a ring that encloses the clone, because r_f turns through 360 degrees).

THE REVERSED ROWS, the papers' own reading (Amonlirdviman 2005 Fig. 2F/2G: fz clones reverse the
wild-type cells DISTAL to them, Vang clones those PROXIMAL). A ring cell is distal when the vector
from its nearest clone cell to it lies within 60 degrees of the cue axis c, proximal within 60 degrees
of -c. `rev_distal` is the number of consecutive rings from ring 1 whose distal cells have a mean
u_f . c below 0 -- they point proximally, against the cue, which is what reversed means there --
and `rev_proximal` the same on the proximal side.

THE PLANE is `plane_axis` (the axis normal to the sheet, 0/1/2), or when not given the axis along
which row 0's vertices spread least. The rulers are for a flat sheet; on a curved one the plane
projection folds and they say nothing.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .common import cells, finite, neighbour_pairs, register_run


# ============================================================================ geometry
def _plane(T, plane_axis):
    if plane_axis is None:
        plane_axis = int(np.argmin(np.ptp(T.pos(0), axis=0)))
    return [i for i in range(3) if i != int(plane_axis)], int(plane_axis)


def _twin_mask(es, et):
    """True where half-edge (s, t) has a twin (t, s) -- an interior junction side."""
    big = int(max(es.max(initial=0), et.max(initial=0))) + 1
    key = np.sort(es * big + et)
    tw = et * big + es
    j = np.clip(np.searchsorted(key, tw), 0, len(key) - 1)
    return key[j] == tw


def cell_vectors(T, t, a="fz", b="vang", plane_axis=None, vector_block=None):
    """(Cells, p [n, 2], s [n] or None, boundary [n] bool) at row t, rows aligned with `cells(T, t)`."""
    c = cells(T, t)
    ax, _ = _plane(T, plane_axis)
    if vector_block is not None:
        v = c.block(vector_block)
        if v is None:
            raise KeyError(f"run records no per-cell block {vector_block!r}")
        p = v[:, ax] if v.shape[1] >= 3 else v[:, :2]
        return c, p, None, np.zeros(len(c), bool)
    A, B = T.edge_col(a, t), T.edge_col(b, t)
    es, et, ef = (np.asarray(x, np.int64) for x in T.half_edges(t))
    if A is None or B is None:
        raise KeyError(f"run records no half-edge columns e_{a} / e_{b} (a `<set>__mesh_e_<name>` pair)")
    if len(A) != len(es) or len(B) != len(es):
        raise ValueError(f"row {t}: e_{a}/e_{b} have {len(A)}/{len(B)} entries, the half-edge table {len(es)}")
    P = T.pos(t)[:, ax]
    nF = T.nF(t)
    cen = np.zeros((nF, 2))
    np.add.at(cen, ef, P[es])
    cen /= np.maximum(np.bincount(ef, minlength=nF), 1)[:, None]
    d = P[et] - P[es]
    L = np.linalg.norm(d, axis=1)
    n = np.stack([d[:, 1], -d[:, 0]], 1) / np.maximum(L, 1e-30)[:, None]
    mid = 0.5 * (P[et] + P[es])
    n *= np.sign(np.einsum("ij,ij->i", mid - cen[ef], n))[:, None]
    pf = np.zeros((nF, 2))
    np.add.at(pf, ef, ((A - B) * L)[:, None] * n)
    sf = np.bincount(ef, weights=(A + B) * L, minlength=nF)
    bnd = np.zeros(nF, bool)
    bnd[ef[~_twin_mask(es, et)]] = True
    return c, pf[c.slot], sf[c.slot], bnd[c.slot]


def _unit(p):
    r = np.linalg.norm(p, axis=1)
    return p / np.maximum(r, 1e-30)[:, None], r > 1e-30


def _wrap(x, half):
    return (x + half) % (2 * half) - half if x is not None else None


def _rows(T, every):
    n = T.n_rows()
    return sorted(set(list(range(0, n, max(1, int(every)))) + [n - 1]))


# ============================================================================ exp08.polarity
def _rc(x, u, spacing, rc_max):
    """Burak & Shraiman's correlation radius, in cell spacings (see the module docstring)."""
    from scipy.spatial import cKDTree
    if len(x) < 2 or not np.isfinite(spacing) or spacing <= 0:
        return None
    pr = cKDTree(x).query_pairs((rc_max + 0.5) * spacing, output_type="ndarray")
    if not len(pr):
        return None
    k = np.rint(np.linalg.norm(x[pr[:, 0]] - x[pr[:, 1]], axis=1) / spacing).astype(int)
    c = np.einsum("ij,ij->i", u[pr[:, 0]], u[pr[:, 1]])
    ok = (k >= 1) & (k <= rc_max)
    sk = np.bincount(k[ok], weights=c[ok], minlength=rc_max + 1) / np.maximum(np.bincount(k[ok], minlength=rc_max + 1), 1)
    sk[0] = 1.0
    stop = next((i for i in range(1, rc_max + 1) if sk[i] <= 0), rc_max + 1)
    r = np.arange(stop)
    return float((r * sk[:stop]).sum() / sk[:stop].sum())


def polarity(T, a="fz", b="vang", cue_axis=None, plane_axis=None, vector_block=None,
             drop_boundary=True, every=10, turn_from=0, turn_from_frac=None, rc_max=10, **_):
    """Global order, local order, cue alignment and axis angle of the per-cell polarity, over rows."""
    ax, _ = _plane(T, plane_axis)
    cue = None
    if cue_axis is not None:
        cue = np.asarray(cue_axis, float)[ax]
        cue = cue / np.linalg.norm(cue)
    ts = _rows(T, every)
    if turn_from_frac is not None:                     # the row a declared fraction into the run
        turn_from = int(round(float(turn_from_frac) * (T.n_rows() - 1)))
    if turn_from not in ts:
        ts = sorted(set(ts + [int(turn_from)]))
    ser = {k: [] for k in ("order", "local", "cue", "asym", "axis_deg", "axial_deg", "rc")}
    n_used = []
    for t in ts:
        c, p, s, bnd = cell_vectors(T, t, a, b, plane_axis, vector_block)
        u, ok = _unit(p)
        keep = ok & (~bnd if drop_boundary else True)
        allp = neighbour_pairs(T, t, c)
        pairs = allp[keep[allp[:, 0]] & keep[allp[:, 1]]] if len(allp) else allp
        uk = u[keep]
        m = uk.mean(0) if len(uk) else np.full(2, np.nan)
        th = np.arctan2(uk[:, 1], uk[:, 0])
        z2 = np.exp(2j * th).mean() if len(th) else np.nan
        ser["order"].append(finite(np.linalg.norm(m)))
        ser["local"].append(finite(np.einsum("ij,ij->i", u[pairs[:, 0]], u[pairs[:, 1]]).mean()) if len(pairs) else None)
        ser["cue"].append(finite((uk @ cue).mean()) if cue is not None and len(uk) else None)
        ser["asym"].append(finite(np.median(np.linalg.norm(p[keep], axis=1) / np.maximum(s[keep], 1e-30)))
                           if s is not None and keep.any() else None)
        ser["axis_deg"].append(finite(np.degrees(np.arctan2(m[1], m[0]))))
        ser["axial_deg"].append(finite(0.5 * np.degrees(np.angle(z2))))
        sp = np.median(np.linalg.norm(c.x[allp[:, 0]][:, ax] - c.x[allp[:, 1]][:, ax], axis=1)) if len(allp) else np.nan
        ser["rc"].append(finite(_rc(c.x[keep][:, ax], uk, sp, int(rc_max))))
        n_used.append(int(keep.sum()))
    i0 = ts.index(int(turn_from))

    def _unwrapped_turn(series, period):
        v = [x for x in series[i0:] if x is not None]
        if len(v) < 2:
            return None
        u = np.unwrap(np.radians(np.asarray(v, float)) * (360.0 / period)) * (period / 360.0)
        return finite(np.degrees(u[-1] - u[0]))
    out = {"rows": ts, "n_cells_last": n_used[-1],
           "order_chance": finite(np.sqrt(np.pi / (4 * max(n_used[-1], 1))))}
    for k, v in ser.items():
        out[f"{k}_first"], out[f"{k}_last"] = v[0], v[-1]
        out[f"{k}_series"] = v
    out["turn_deg"] = _unwrapped_turn(ser["axis_deg"], 360.0)
    out["turn_axial_deg"] = _unwrapped_turn(ser["axial_deg"], 180.0)
    return out


# ============================================================================ exp08.clone
def clone(T, a="fz", b="vang", clone_block="mutant", cue_axis=(1, 0, 0), plane_axis=None, vector_block=None,
          row=-1, n_rings=6, **_):
    """toward_k per ring of wild-type cells round the clone, and the reversed rows distal and proximal, at one row."""
    t = T.n_rows() + row if row < 0 else row
    c, p, _s, _b = cell_vectors(T, t, a, b, plane_axis, vector_block)
    lab = c.block(clone_block)
    if lab is None:
        return {"available": False, "why": f"run records no per-cell block {clone_block!r}"}
    inc = lab[:, 0] > 0.5
    if not inc.any():
        return {"available": False, "why": "the clone is empty"}
    ax, _ = _plane(T, plane_axis)
    x = c.x[:, ax]
    u, _ok = _unit(p)
    pairs = neighbour_pairs(T, t, c)
    adj = [[] for _ in range(len(c))]
    for i, j in pairs:
        adj[i].append(j)
        adj[j].append(i)
    dist = np.full(len(c), -1)
    q = deque(np.flatnonzero(inc).tolist())
    dist[inc] = 0
    while q:
        i = q.popleft()
        for j in adj[i]:
            if dist[j] < 0:
                dist[j] = dist[i] + 1
                q.append(j)
    cue = np.asarray(cue_axis, float)[ax]
    cue = cue / np.linalg.norm(cue)
    xc = x[inc]
    out = {"available": True, "n_clone": int(inc.sum())}
    toward, dist_side, prox_side = [], [], []
    for k in range(1, int(n_rings) + 1):
        r = np.flatnonzero(dist == k)
        if not len(r):
            toward.append(None); dist_side.append(None); prox_side.append(None)
            continue
        d = xc[np.argmin(((x[r, None, :] - xc[None]) ** 2).sum(-1), axis=1)] - x[r]
        d /= np.maximum(np.linalg.norm(d, axis=1), 1e-30)[:, None]
        toward.append(finite(np.einsum("ij,ij->i", u[r], d).mean()))
        side = -(d @ cue)                                   # cos of (clone -> cell) with the cue
        uc = u[r] @ cue
        dist_side.append(finite(uc[side > 0.5].mean()) if (side > 0.5).any() else None)
        prox_side.append(finite(uc[side < -0.5].mean()) if (side < -0.5).any() else None)
        out[f"n_ring{k}"] = int(len(r))

    def _run(v):
        n = 0
        for x_ in v:
            if x_ is None or x_ >= 0:
                break
            n += 1
        return n
    out["toward"], out["toward_1"] = toward, toward[0]
    out["cue_distal"], out["cue_proximal"] = dist_side, prox_side
    out["rev_distal"], out["rev_proximal"] = _run(dist_side), _run(prox_side)
    return out


register_run("exp08.polarity", polarity, None, "order, local order, cue alignment, axis turn of junctional polarity")
register_run("exp08.clone", clone, "count", "rings of wild-type cells reoriented round a clone, and which way")


# ============================================================================ the results table's source
def write_results(number=8):
    """`specs/exp08/polarity.jsonl` and `audit.jsonl` -- one line per run, DERIVED from the scorer's
    cache `measures.jsonl` (the last value of each run x measure) -- then the markdown's Results table
    (`tools/exp_results_table.py`). Run after `tools/exp_gate_score.py 8`:

        PYTHONPATH=src:tools python -c "import exp_measures.exp08 as m; m.write_results()"

    (not `python -m exp_measures.exp08`: that executes this file a second time as `__main__` and its
    measures would be registered twice)
    """
    import json
    import os
    from .common import ROOT
    d = os.path.join(ROOT, "experiments", "specs", f"exp{number:02d}")
    last = {}
    for line in open(os.path.join(d, "measures.jsonl")):
        r = json.loads(line)
        last[(r["run"], r["measure"])] = r["value"] or {}
    pol, aud = {}, {}
    for (run, meas), v in last.items():
        if meas == "exp08.polarity":
            pol.setdefault(run, {}).update({k: v.get(k) for k in ("order_last", "local_last", "rc_last",
                                                                 "cue_last", "asym_last", "turn_axial_deg")})
        elif meas == "exp08.celsr":
            pol.setdefault(run, {}).update({k: v.get(k) for k in ("MP_last", "ang_P_deform_deg_last", "ME_peak", "ME_last")})
        elif meas == "exp08.clone" and v.get("available"):
            pol.setdefault(run, {}).update({k: v.get(k) for k in ("toward_1", "rev_distal", "rev_proximal")})
        elif meas == "shared.growth_audit" and v.get("available"):
            aud[run] = None
    with open(os.path.join(d, "polarity.jsonl"), "w") as f:
        for run, v in sorted(pol.items()):
            f.write(json.dumps({"spec": run, **v}) + "\n")
    import growth_audit as GA                     # exp 3's full record, the shape exp_record reads
    with open(os.path.join(d, "audit.jsonl"), "w") as f:
        for run in sorted(aud):
            f.write(json.dumps({"spec": run, **GA.audit(run)}, default=float) + "\n")
    import sys
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import exp_results_table
    exp_results_table.main(number)



# ============================================================================ exp08.celsr (Phase 2)
def celsr(T, a="fz", b="vang", deform_axis=(0, 1, 0), plane_axis=None, drop_boundary=True, every=10, **_):
    """Aw et al. 2016's readouts, from the recorded junction sides: the Celsr1-like nematic polarity,
    the cell-elongation nematic, and the angle of each to the imposed deformation's axis.

    CELSR1 SITS ON BOTH SIDES OF A PCP JUNCTION (it binds homophilically across it), so its intensity on
    cell i's side h is read as I_h = a_h + b_h, the total complex there. Aw quantified "nematic order of
    the integrated fluorescence intensity values of Celsr1 at cell borders" (Results; Aigouy 2010's Q):
        q_i  = sum_h I_h int_{phi_a}^{phi_b} exp(2 i phi) dphi / sum_h I_h (phi_b - phi_a)
               phi_a, phi_b the angles of side h's two ends about the cell's centroid -- Aigouy's
               Q = int I(phi) exp(2 i phi) dphi over the ANGLE round the cell, with I constant along
               a border. A uniform intensity reads exactly 0 on any cell shape: MP is enrichment,
               not shape (a length weighting would call an elongated, uniformly stained cell polar)
        MP   = | mean over cells of q_i |      "magnitude of average polarity" (Aw Fig. 1B), 0 random, 1 all
                                               cells' complex on one pair of opposite borders
        P_axis_deg = half the angle of mean q_i, the axis of the enriched borders' POSITIONS
        Mcell = mean over cells of | q_i |     each cell's OWN enrichment, whatever its axis: MP / Mcell is
                                               the cells' agreement on the axis (1 all agree)
    Cell elongation, the same way from each cell's shape (the second moment of its vertices about the
    centroid, eigenvalues l1 >= l2): e_i = (l1 - l2) / (l1 + l2) exp(2 i theta_i); ME = | mean e_i |,
    E_axis_deg its axis ("magnitude of average cell elongation", Aw Fig. 1B).
    `ang_P_deform_deg` is the angle between the polarity axis and `deform_axis` (the imposed extension),
    folded to [0, 90]: 90 = perpendicular, Aw's result (88.9 deg, Fig. 5B); 45 = no relation.
    """
    ax, _ = _plane(T, plane_axis)
    d = np.asarray(deform_axis, float)[ax]
    d_ang = np.arctan2(d[1], d[0])
    ts = _rows(T, every)
    ser = {k: [] for k in ("MP", "Mcell", "P_axis_deg", "ME", "E_axis_deg", "ang_P_deform_deg")}
    for t in ts:
        c = cells(T, t)
        A, B = T.edge_col(a, t), T.edge_col(b, t)
        es, et, ef = (np.asarray(x, np.int64) for x in T.half_edges(t))
        if A is None or B is None or len(A) != len(es):
            raise KeyError(f"row {t}: no aligned half-edge columns e_{a} / e_{b}")
        P = T.pos(t)[:, ax]
        nF = T.nF(t)
        cnt = np.maximum(np.bincount(ef, minlength=nF), 1)
        cen = np.zeros((nF, 2)); np.add.at(cen, ef, P[es]); cen /= cnt[:, None]
        ra, rb = P[es] - cen[ef], P[et] - cen[ef]
        pa, pb = np.arctan2(ra[:, 1], ra[:, 0]), np.arctan2(rb[:, 1], rb[:, 0])
        dphi = (pb - pa + np.pi) % (2 * np.pi) - np.pi              # the signed angle side h subtends
        pb = pa + dphi
        I = A + B
        num = np.zeros(nF, complex); np.add.at(num, ef, I * (np.exp(2j * pb) - np.exp(2j * pa)) / 2j)
        den = np.bincount(ef, weights=I * dphi, minlength=nF)
        q = num / np.where(np.abs(den) > 1e-30, den, 1e-30)
        rel = P[es] - cen[ef]
        Sxx = np.bincount(ef, weights=rel[:, 0] ** 2, minlength=nF)
        Syy = np.bincount(ef, weights=rel[:, 1] ** 2, minlength=nF)
        Sxy = np.bincount(ef, weights=rel[:, 0] * rel[:, 1], minlength=nF)
        tr = np.maximum(Sxx + Syy, 1e-30)
        e = ((Sxx - Syy) + 2j * Sxy) / tr                     # (l1 - l2)/(l1 + l2) exp(2 i theta)
        bnd = np.zeros(nF, bool)
        bnd[ef[~_twin_mask(es, et)]] = True
        keep = np.zeros(nF, bool); keep[c.slot] = True
        if drop_boundary:
            keep &= ~bnd
        mq, me = q[keep].mean(), e[keep].mean()
        pax = 0.5 * np.angle(mq)
        ser["MP"].append(finite(abs(mq))); ser["Mcell"].append(finite(np.abs(q[keep]).mean()))
        ser["P_axis_deg"].append(finite(np.degrees(pax)))
        ser["ME"].append(finite(abs(me))); ser["E_axis_deg"].append(finite(0.5 * np.degrees(np.angle(me))))
        dd = np.degrees(abs(((pax - d_ang) + np.pi / 2) % np.pi - np.pi / 2))
        ser["ang_P_deform_deg"].append(finite(dd))
    out = {"rows": ts}
    for k, v in ser.items():
        out[f"{k}_last"], out[f"{k}_series"] = v[-1], v
    me = [x for x in ser["ME"] if x is not None]
    out["ME_peak"] = finite(max(me)) if me else None
    return out


register_run("exp08.celsr", celsr, None, "Aw 2016: Celsr1-like nematic MP, cell elongation ME, polarity axis vs the deformation")
