"""exp12 tumor_invasion -- the rulers: is the spheroid layered the way a consumed, diffusing nutrient
layers it, and do strands leave it?

    exp12.spheroid   spheroid radius, starved-core radius, proliferating-rim thickness, the nutrient
                     profile's fit to the diffusion-consumption solution, its critical radius, growth
    exp12.strands    invading strands: connected clusters of cells beyond the body's radius
    exp12.mesh_sanity  `tools/mesh_sanity.py` on the run: is every cell still a cell of the surface?

THE BODY IS A SOLID BALL (`dim: 3`) OR A FLAT DISC (`dim: 2`, the radius taken in the plane normal to
`plane_axis`). A monolayer SHELL has no radial layers -- every cell sits at the same radius and its
"core" is the lumen -- so `spheroid` reports `shell_ratio` = median cell radius / R (0.79 for a
filled ball, 0.71 for a filled disc, ~1 for a shell) and gates.yaml caps a hollow body.

EVERY LENGTH IS IN MICROMETRES (the spec's `units.length_um` per model unit, or `um_per_unit`) and
in CELL DIAMETERS, d = the median distance between touching cells (`common.neighbour_pairs`).

THE DEFINITIONS, each declared here so a band read from a paper is compared with the same thing:

  centre      mean position of the live cells.
  R           spheroid radius = the 98th percentile of the cells' radii + d/2 (cell centres sit half a
              cell inside the surface). For n cells filling a ball the 98th percentile is R 0.98^(1/3),
              0.7 % inside, below one cell diameter at every size this experiment runs.
  starved     a cell whose nutrient is below `c_starve`, or whose `apop_flag` is set. With no declared
              `c_starve` the run's own death rule is read: cell_die[chem_low] marks a cell when its
              chem is below `a_sw` x the live tissue's maximum, so c_starve = a_sw x max(nutrient),
              and the channel is that operator's `chan`.
  cycling     IN CYCLE NOW, the ruler's stand-in for a Ki-67 stain (which marks every cell in G1-S-G2-M
              and none in G0): a live, unstarved cell born within the last cycle (`age` below
              `cycling_age`) whose nutrient still holds its growth gate at least half open (>= `c_cycle`,
              by default the run's own `cell_grow` `a_sw`). Born-lately alone overcounts once division
              is continuous: cells born days ago have drifted inward and stopped growing (exp12 round 8,
              rim ~30 um too thick); Grimes likewise placed the Ki-67 boundary at the 10 mmHg isobar. `age` counts
              `cell_divide` CALLS since birth (vertex_ops.py, "age in division-calls since birth"),
              not frames, so `cycling_age` is in those units: one cycle's frames / the divide
              operator's `every`. With no `age` block, nutrient >= `c_cycle`.
  r_core      the step boundary that best separates starved (inside) from fed (outside) cells: the
              radius maximising (#starved inside) + (#fed outside). 0 with no starved cell.
  rim         R minus the step boundary separating non-cycling (inside) from cycling (outside).
  profile     over ALL cells, least squares of Grimes' full solution: c(r) = c0 - beta g(r) outside
              the zero-flux radius r_n, and the flat plateau c0 - beta g(r_n) inside it (a core that
              consumes nothing has no gradient) -- the steady diffusion with zero-order consumption a
              (per volume) and diffusivity D (Grimes et al. 2014, eq. 2.4):
                  3D   g(r) = (R^2 - r^2) - 2 r_n^3 (1/r - 1/R),     beta = a / (6 D)
                  2D   g(r) = (R^2 - r^2) - 2 r_n^2 ln(R / r),       beta = a / (4 D)
              r_n IS FITTED FROM THE FIELD (`rn_fit`, a search over 120 radii), NOT TAKEN FROM THE DEATH
              LABELS: the two differ whenever the dead core was fixed at an earlier geometry or the
              uptake is Michaelis-Menten-limited near it, and forcing the label radius into g bent the
              fitted curvature to 0.77 of the operator's own a / 4D on exp12 round 2's `v2_r4` (1.06
              with the field's radius). `r_core` (the labels) stays the core the layers are scored on,
              as Grimes measured his r_n by staining.
              fit_r2 is the fit's coefficient of determination; r_crit = sqrt((c0 - c_starve) / beta)
              is the radius at which a spheroid with no core first starves at its centre (the onset).
  r_m, r_l    the fitted field's GEOMETRY-FREE lengths: r_m = sqrt(2 D c0 / a) = sqrt(c0 / (dim beta)),
              the rim a huge spheroid tends to (Grimes 2014 eq. 2.10), and r_l = sqrt(3) r_m, their
              spherical diffusion limit (eq. 2.6; 233 +- 22 um for DLD1, Fig 5). A disc and a ball with
              the same a / D share them, which is what lets a disc be compared with the paper.
  *_ref       GRIMES' OWN CURVE AT THIS RUN'S RADIUS, given their r_l (`r_l_ref_um`, 233 um, Fig 5):
              the anoxic core r_n (eq. 2.9, the root of 2 r_c^3 - 3 R r_c^2 + R r_l^2 = 0, r_c = R - r_n)
              and the Ki-67 boundary r_p where p = `p_cycle_frac` p_o (10 of 100 mmHg, Fig 6), from their
              eq. 2.4 with the core. A spheroid that grows during a run is thus compared at the radius it
              reached, not at the one it was seeded at; the curve against the paper's data is Grimes'
              Figs 6-7 (every spheroid of r_o 372-588 um within 1 s.d. of it). `core_err` and
              `rim_err_um` are the run's distance from it.
"""
from __future__ import annotations

import numpy as np

from .common import cells, finite, neighbour_pairs, register_run


# ============================================================================ geometry
def _um(T, um_per_unit):
    s = um_per_unit or ((T.spec.get("general") or {}).get("units") or {}).get("length_um")
    return float(s) if s else 1.0


def _row_days(T):
    """Model time per recorded row, in days: dt x (frames per row) x the spec's `units.time_s`."""
    g = T.spec.get("general") or {}
    dt = float(g.get("dt", 1.0))
    per_row = float(g.get("n_frames", T.n_rows())) / max(T.n_rows(), 1)
    ts = float((g.get("units") or {}).get("time_s", 0.0))
    return dt * per_row * ts / 86400.0 if ts else None


def _radii(x, dim, plane_axis):
    c = x.mean(0)
    v = x - c
    if dim == 2:
        v = np.delete(v, int(plane_axis), axis=1)
    return np.linalg.norm(v, axis=1)


def _diameter(T, t, c):
    p = neighbour_pairs(T, t, c)
    if len(p) == 0:
        return float("nan")
    return float(np.median(np.linalg.norm(c.x[p[:, 0]] - c.x[p[:, 1]], axis=1)))


def step_boundary(r, inside):
    """Radius maximising (#inside-labelled cells below it) + (#others at or above it); 0 if none inside."""
    if not np.any(inside):
        return 0.0
    o = np.argsort(r)
    rs, lab = r[o], inside[o].astype(int)
    # a boundary just after sorted cell k puts cells 0..k inside
    score = np.cumsum(lab) + (np.sum(1 - lab) - np.cumsum(1 - lab))
    k = int(np.argmax(score))
    return float(rs[k] if k == len(rs) - 1 else 0.5 * (rs[k] + rs[k + 1]))


def _g(r, R, rn, dim):
    r = np.maximum(r, 1e-12)
    if dim == 3:
        return (R ** 2 - r ** 2) - 2.0 * rn ** 3 * (1.0 / r - 1.0 / R)
    return (R ** 2 - r ** 2) - 2.0 * rn ** 2 * np.log(R / r)


def fit_profile(r, c, R, rn, dim):
    """(c0, beta, r2) of c = c0 - beta g(max(r, rn)) by least squares: the fed profile outside rn and
    its flat plateau inside."""
    A = np.stack([np.ones_like(r), -_g(np.maximum(r, rn), R, rn, dim)], 1)
    (c0, beta), *_ = np.linalg.lstsq(A, c, rcond=None)
    res = c - A @ np.array([c0, beta])
    ss = np.sum((c - c.mean()) ** 2)
    return float(c0), float(beta), float(1.0 - np.sum(res ** 2) / ss) if ss > 0 else float("nan")


def fit_profile_free_rn(r, c, R, dim, lo, hi, n=40):
    """(c0, beta, r2, rn) with rn searched over n radii in [lo, hi]: the zero-flux radius the FIELD has.

    THE SEARCH IS BOUNDED BY THE PHYSICS, NOT OPEN. The zero-flux radius cannot lie INSIDE the dead
    core (dead cells consume nothing, so there is no gradient anywhere in it), and it lies at most a few
    cells outside it (cells there are nutrient-limited, not dead). An open search let a plateau imitate
    a first-order profile -- c ~ sinh(r / L) / r, flat inside and steep at the rim -- and G-profile could
    no longer fail (the planted first-order test)."""
    best = None
    for rn in np.linspace(lo, max(lo, hi), n):
        c0, beta, r2 = fit_profile(r, c, R, rn, dim)
        if np.isfinite(r2) and (best is None or r2 > best[2]):
            best = (c0, beta, r2, float(rn))
    return best


def grimes_layers(R, r_l, p_frac=0.1):
    """Grimes et al. 2014's spherical layers at spheroid radius R for diffusion limit r_l (same units):
    (r_n, r_p) = the anoxic radius (eq. 2.9) and the radius where p = p_frac p_o (eq. 2.4 with the core)."""
    from scipy.optimize import brentq
    R, r_l = float(R), float(r_l)
    rn = 0.0 if R <= r_l else brentq(lambda x: R ** 2 - 3 * x ** 2 + 2 * x ** 3 / R - r_l ** 2, 0.0, R)
    p = lambda r: 1.0 - ((R ** 2 - r ** 2) - 2 * rn ** 3 * (1 / r - 1 / R)) / r_l ** 2 - p_frac
    lo = max(rn, 1e-9 * R)
    rp = lo if p(max(lo, 1e-9)) >= 0 else brentq(p, max(lo, 1e-9), R)
    return rn, rp


# ============================================================================ the rulers
def _nutrient(c, block, chan):
    b = c.block(block)
    return None if b is None else b[:, int(chan)]


def _death_rule(T):
    """The run's cell_die[chem_low] entry, or {}."""
    for o in T.spec.get("operators") or []:
        if isinstance(o, dict) and o.get("op") == "cell_die" and o.get("model") == "chem_low":
            return o
    return {}


def _layers(T, t, dim, plane_axis, c_starve, c_cycle, cycling_age, block, chan, s, r_l_ref_um=None,
            p_cycle_frac=0.1):
    c = cells(T, t)
    if len(c) < 8:
        return None
    r = _radii(c.x, dim, plane_axis)
    d = _diameter(T, t, c)
    R = float(np.quantile(r, 0.98) + 0.5 * d)
    rule = _death_rule(T)
    if chan is None:
        chan = rule.get("chan", 0)
    nut = _nutrient(c, block, chan)
    if c_starve is None and nut is not None and rule.get("a_sw") is not None:
        c_starve = float(rule["a_sw"]) * float(np.nanmax(nut))
    apop = c.block("apop_flag")
    starved = np.zeros(len(c), bool)
    if nut is not None and c_starve is not None:
        starved |= nut < float(c_starve)
    if apop is not None:
        starved |= apop[:, 0] > 0
    age = c.block("age")
    if c_cycle is None and nut is not None:
        g = next((o for o in (T.spec.get("operators") or [])
                  if isinstance(o, dict) and o.get("op") == "cell_grow" and not o.get("a_sw_rel")), None)
        if g is not None and g.get("a_sw") is not None and float(g.get("rho", 0.0)) == 0.0:
            c_cycle = float(g["a_sw"])            # the run's own growth gate: half-open at this nutrient
    if age is not None and cycling_age is not None:
        cycling = age[:, 0] < float(cycling_age)
        if nut is not None and c_cycle is not None:
            cycling &= nut >= float(c_cycle)      # still growing: in cycle NOW, not merely born lately
    elif nut is not None and c_cycle is not None:
        cycling = nut >= float(c_cycle)
    else:
        cycling = None
    if cycling is not None:
        # A DEAD OR STARVED CELL IS NEVER CYCLING (Ki-67 negative), however young. Once division runs
        # through the whole run (exp12 round 8, asynchronous phases), daughters born within the window
        # are pushed inward and die in the core; counted as cycling they collapsed the rim's step
        # boundary to the centre (`v8_r2` read a 376 um rim on a 386 um disc).
        cycling = cycling & ~starved
    rn = step_boundary(r, starved)
    out = {"n": len(c), "c_starve": finite(c_starve) if c_starve is not None else None, "d_um": d * s, "R_um": R * s, "R_cd": R / d, "shell_ratio": float(np.median(r) / R),
           "r_core_um": rn * s, "core_over_R": rn / R, "starved_frac": float(starved.mean())}
    if cycling is not None:
        ri = step_boundary(r, ~cycling)
        out.update(rim_um=(R - ri) * s, rim_cd=(R - ri) / d, cycling_frac=float(cycling.mean()))
    if r_l_ref_um:
        Rum = R * s
        rn_ref, rp_ref = grimes_layers(Rum, r_l_ref_um, p_cycle_frac)
        out.update(core_over_R_ref=rn_ref / Rum, core_err=abs(rn / R - rn_ref / Rum), rim_ref_um=Rum - rp_ref)
        if "rim_um" in out:
            out["rim_err_um"] = abs(out["rim_um"] - (Rum - rp_ref))
    if nut is not None:
        hi = min(rn + 3.0 * d, 0.97 * R) if rn > 0 else 0.0      # no dead cell, no non-consuming core
        c0, beta, r2, rn_fit = fit_profile_free_rn(r, nut, R, dim, rn, hi)
        out.update(c0=c0, beta=beta, fit_r2=r2, rn_fit_um=rn_fit * s)
        if beta > 0 and c0 > 0:
            rm = np.sqrt(c0 / (dim * beta))
            out.update(r_m_um=rm * s, r_l_um=np.sqrt(3.0) * rm * s)
        if beta > 0 and c_starve is not None and c0 > float(c_starve):
            rc = np.sqrt((c0 - float(c_starve)) / beta)
            out.update(r_crit_um=rc * s, r_crit_cd=rc / d, R_over_rcrit=R / rc)
    return {k: (finite(v) if isinstance(v, float) else v) for k, v in out.items()}


def _rows(T, every):
    n = T.n_rows()
    return sorted(set(list(range(0, n, max(1, int(every)))) + [n - 1]))


def _R_at_row(T, t, dim, plane_axis):
    c = cells(T, t)
    if len(c) < 8:
        return None
    r = _radii(c.x, dim, plane_axis)
    return float(np.quantile(r, 0.98) + 0.5 * _diameter(T, t, c))


def spheroid(T, dim=3, plane_axis=2, c_starve=None, c_cycle=None, cycling_age=None, block="chem", chan=None,
             every=20, um_per_unit=None, r_l_ref_um=None, p_cycle_frac=0.1, R_ref_um=None, **_):
    """The layers at the last row (`<key>`), the radius series, and the growth rate over the run.

    GROWTH KINETICS, `R_ref_um: {day: radius_um}`: the spheroid's radius at those days since the run's
    first row (the row nearest each day, from the spec's `dt`, frames per row and `units.time_s`), as
    `R_day<d>_um`, and `growth_err_um` = the mean |R(d) - reference(d)| -- Grimes et al. 2014's Fig 4 sizes
    by day, read in gates.yaml."""
    s = _um(T, um_per_unit)
    ts = _rows(T, every)
    last = _layers(T, ts[-1], dim, plane_axis, c_starve, c_cycle, cycling_age, block, chan, s, r_l_ref_um,
                   p_cycle_frac)
    if last is None:
        return {"available": False, "why": "fewer than 8 live cells at the last row"}
    series = []
    for t in ts:
        c = cells(T, t)
        if len(c) < 8:
            series.append(None)
            continue
        r = _radii(c.x, dim, plane_axis)
        series.append(finite((np.quantile(r, 0.98) + 0.5 * _diameter(T, t, c)) * s))
    out = {"available": True, **last, "R_um_series": series, "rows": ts}
    days = _row_days(T)
    if R_ref_um and days:
        errs = []
        for d_key, ref in dict(R_ref_um).items():
            d = float(d_key)
            row = int(np.clip(round(d / days), 0, T.n_rows() - 1))
            Rd = _R_at_row(T, row, dim, plane_axis)
            if Rd is not None:
                out[f"R_day{d:g}_um"] = finite(Rd * s)
                errs.append(abs(Rd * s - float(ref)))
        out["growth_err_um"] = finite(np.mean(errs)) if errs else None
    ok = [(t, v) for t, v in zip(ts, series) if v is not None]
    if days and len(ok) >= 3:
        tt = np.array([t for t, _ in ok], float) * days
        out["dRdt_um_per_day"] = finite(np.polyfit(tt, np.array([v for _, v in ok], float), 1)[0])
        n0, n1 = len(cells(T, ts[0])), last["n"]
        out["doublings_per_day"] = finite(np.log2(max(n1, 1) / max(n0, 1)) / ((ts[-1] - ts[0]) * days))
    return out


def strands(T, dim=3, plane_axis=2, beyond_cd=3.0, link=1.5, min_cells=3, every=20, um_per_unit=None, body="filled",
            **_):
    """Clusters of cells more than `beyond_cd` cell diameters outside the body's radius, linked when
    closer than `link` diameters; clusters of at least `min_cells` count as strands.

    A SHELL (`body: shell`) -- the apico-basal spheroid of Phase 2, one cell layer round a lumen -- has
    every cell at one radius, so its body radius is the median cell radius itself.

    THE LINE IS THREE CELLS OUT, NOT ONE. A contained disc's own rim is bumpy: exp12's `v7_r4` has its
    outermost cells 1.4-1.5 cell diameters past the equal-area radius (14-15 um, cells compressed to
    ~10 um), and the Step-0 line of one diameter and two cells read those bumps as 2 "strands" while the
    disc settled. Cheung et al. 2013's strands are multicellular protrusions that extend well beyond the
    body (Fig 6C), so a strand must reach three diameters out with at least three linked cells.

    THE BODY'S RADIUS IS READ FROM ITS BULK, NOT ITS EDGE -- an edge percentile would be dragged
    outward by the very strands it is meant to find. On a disc with a recorded cell `area` it is the
    radius of the disc of the same area, R_body = sqrt(sum A / pi): a few strand cells add little area.
    Otherwise R_body = median(r) / 0.5^(1/dim), exact for a UNIFORMLY filled ball or disc -- and wrong
    when density varies: exp12 batch 1's r4 had compressed core cells and large rim cells, the median
    fell inside, and a contained disc read 19 strands. The area form is immune to that."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    s = _um(T, um_per_unit)
    ts = _rows(T, every)
    counts, reach = [], []
    for t in ts:
        c = cells(T, t)
        if len(c) < 8:
            counts.append(None); reach.append(None)
            continue
        r = _radii(c.x, dim, plane_axis)
        d = _diameter(T, t, c)
        area = c.block("area") if (dim == 2 and body != "shell") else None
        if body == "shell":
            Rb = float(np.median(r))
        else:
            Rb = (float(np.sqrt(np.nansum(area[:, 0]) / np.pi)) if area is not None
                  else float(np.median(r) / 0.5 ** (1.0 / dim)))
        out_i = np.flatnonzero(r > Rb + beyond_cd * d)
        n = 0
        if len(out_i) >= min_cells:
            G = cKDTree(c.x[out_i]).sparse_distance_matrix(cKDTree(c.x[out_i]), link * d)
            _, lab = connected_components(G, directed=False)
            n = int(np.sum(np.bincount(lab) >= min_cells))
        counts.append(n)
        reach.append(finite((r.max() - Rb) / d))
    last = counts[-1]
    return {"strands_last": last, "strands_max": max((v for v in counts if v is not None), default=None),
            "reach_last_cd": reach[-1], "series": counts, "rows": ts, "um_per_unit": s}


def mesh_sanity(T, every=5, **_):
    """`tools/mesh_sanity.py` (exp 14's per-cell geometry check) over the run: `sane` 1.0 when no sampled
    row breaks a line -- Euler characteristic, longest edge 8x the median, largest cell 10x the median
    area, a vertex 4 edges off a closed shell -- else 0.0, with the first broken row and the worst
    values. Imported, not rewritten, so its lines stay its own. G-sanity reads it in place of exp 3's
    growth auditor, whose x1.5 growth line and slot-keyed wobble do not fit a spheroid that grows only
    at its rim (human's call, 2026-09-26)."""
    import mesh_sanity as MS
    rows = [MS.row(T, t) for t in range(0, T.n_rows(), max(1, int(every)))]
    for r in rows:
        # THE TOOL'S EULER COUNT ASSUMES A CLOSED SURFACE: E = half-edges // 2, true only when every edge
        # has a twin. A disc's rim edges have one half-edge each, so every exp12 disc read chi = 305
        # (= 1 + 608 rim half-edges / 2) on every row. Recounted here with undirected edges; the flag is
        # dropped only when the true chi is 1 (disc) or 2 (shell). Every other line is the tool's own.
        if any(b.startswith("Euler") for b in r["bad"]):
            es, et, ef = (np.asarray(x, np.int64) for x in T.half_edges(r["row"]))
            E = len(np.unique(np.sort(np.stack([es, et], 1), 1), axis=0))
            chi = len(np.unique(np.concatenate([es, et]))) - E + int(T.nF(r["row"]))
            r["euler"] = int(chi)
            if chi in (1, 2):
                r["bad"] = [b for b in r["bad"] if not b.startswith("Euler")]
    first = next((r for r in rows if r["bad"]), None)
    w = {k: finite(max((r.get(k, 0) or 0) for r in rows)) for k in ("radial", "edge", "area", "orphans")}
    return {"sane": 0.0 if first else 1.0, "rows": len(rows), "broken_rows": int(sum(bool(r["bad"]) for r in rows)),
            "first_broken": first["row"] if first else None, "why": "; ".join(first["bad"]) if first else "",
            "euler_last": rows[-1].get("euler"), **{f"worst_{k}": v for k, v in w.items()}}


register_run("exp12.spheroid", spheroid, None, "spheroid radius, starved core, cycling rim, nutrient profile fit")
register_run("exp12.strands", strands, "count", "invading strands beyond the body's radius")
register_run("exp12.mesh_sanity", mesh_sanity, None, "tools/mesh_sanity.py: every cell still a cell")
