#!/usr/bin/env python
"""test_geometries -- do the metrics report the shape that is actually there?

CEDRIC, 5 AUGUST: *"can we test the metrics thoroughly against test geometries to be generated with a
dedicated generator?"*

EVERY OTHER TEST IN THIS CAMPAIGN CHECKS THAT A METRIC EXISTS, IS ADMITTED, HAS A PRODUCER AND REACHES
A SUMMARY. None of them checks that it is RIGHT. That gap is not hypothetical: the instrument gate
measured four metrics to lie outright (`ta_aspect_len_over_diam`, `ta_tube_len_final`, `retention`,
`autocorr_hops_uncalibrated`, all rejected as F15/F16), `corr_act_rad` returned a confident 0.294 on a
field whose entire spread was 8.4e-05, and the morphology classifier called `coral_gate` *branched*
where the eye saw a lobed sphere with `n_tubes` 0. In each case the metric was present, produced,
documented and admitted -- and wrong.

THE GENERATOR, AND WHY IT DEFORMS RATHER THAN BUILDS. A metric needs a mesh with valid topology: a
closed trivalent sheet where every edge is shared by exactly two faces. Building that for an arbitrary
shape is a meshing problem. Building it for a SPHERE is already solved -- `build_sphere_mesh` does a
spherical Voronoi of a jittered Fibonacci sphere -- so every geometry here is that mesh with its
VERTICES moved. The connectivity is untouched, so `euler` and `genus` stay exactly what a sphere gives,
and any change they report is a bug in them rather than in the fixture.

That also makes the fixtures honest about one thing: a deformed sphere is not a grown tissue. These
test the RULER, not the biology. A tube built by pushing a polar cap outward has the shape of a tube
and none of its history, which is precisely what you want when asking whether `protr` can see a tube.

WHAT EACH FIXTURE PINS. The assertions are one-sided wherever the true value is a bound rather than a
number: a sphere's `protr` is 1.0 by construction and its `gyr_prolate` is 1.0 to within the jitter,
but a tube's `protr` has no analytic value, so what is asserted is that it EXCEEDS the sphere's and
crosses the 1.3 the campaign treats as the tube threshold. A test that pins a number nobody derived is
how the alpha ceiling put Okuda's own value outside the searchable space.

RUN: python test_geometries.py
"""
from __future__ import annotations

import io
import contextlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TYSSUE = os.path.join(HERE, "ops")
for _p in (HERE, os.path.join(HERE, "agents"), TYSSUE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FAIL = []
N_CELLS = 260               # enough that a p95 tail statistic has a tail to measure
JITTER = 0.015              # small: the fixtures are about shape, not about mesh noise


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


# ================================================================ the generator

def base_sphere(n=N_CELLS, jitter=JITTER, seed=0):
    """The one mesh everything else deforms. Radius 1, centred on the origin."""
    from mesh_ops import build_sphere_mesh
    verts, es, et, ef, nF = build_sphere_mesh(n, r=1.0, jitter=jitter, seed=seed)
    return np.asarray(verts, float), {"E_srce": es, "E_trgt": et, "E_face": ef, "nF": nF}


def sphere(**kw):
    """A sphere. protr = 1, gyr_prolate = 1, reduced_volume = 1, one ray crossing everywhere."""
    return base_sphere(**kw)


def prolate(aspect=2.0, **kw):
    """Stretched along z. ELONGATED WITHOUT A PROTRUSION -- every cell moves, none sticks out.

    This is the fixture that separates `gyr_prolate` from `protr`, which is the distinction Okuda's
    phenotype axis needs and nothing in the bank could make before the gyration tensor was added: a
    tube and a rugby ball both have a long axis, and only one of them has a tip.
    """
    v, mt = base_sphere(**kw)
    v = v.copy()
    v[:, 2] *= float(aspect)
    return v, mt


def oblate(aspect=2.5, **kw):
    """Flattened along z -- a vesicle collapsing into a disc. A failure mode that reads as "not a
    tube" and needs a number of its own (`gyr_oblate`), or it is invisible."""
    v, mt = base_sphere(**kw)
    v = v.copy()
    v[:, 2] /= float(aspect)
    return v, mt


def tubed(length=2.4, cap=0.82, waist=0.30, **kw):
    """One CYLINDRICAL finger from the north pole: a sphere with a real tube.

    MY FIRST VERSION WAS A TEARDROP AND CEDRIC SAW IT IMMEDIATELY. It scaled xy by a smoothstep and
    stretched z, which tapers continuously from the equator to the pole -- a pear. A tube is not a
    taper: it has a NECK where it leaves the body and a roughly CONSTANT WIDTH along its length, and
    those two features are exactly what `protrusion_aspect_max` and `tube_diameter` are built to
    measure. A fixture that lacks them cannot test them, and a metric that passes on a pear tells you
    nothing about a tube.

    So the polar cap is MAPPED ONTO A CYLINDER rather than stretched. A vertex at polar parameter t
    (0 at the cap rim, 1 at the pole) goes to radius `waist` -- the same radius for every t, which is
    what makes it a cylinder -- and to height `cap + t * length`. The rim itself stays put, so the
    tube joins the body at a neck instead of blending into it.
    """
    v, mt = base_sphere(**kw)
    v = np.asarray(v, float).copy()
    r = np.linalg.norm(v, axis=1)
    z = v[:, 2]
    inside = z > cap
    if not inside.any():
        return v, mt
    # t: 0 at the cap rim, 1 at the pole
    t = np.clip((z[inside] - cap) / (r[inside].max() - cap + 1e-12), 0.0, 1.0)
    xy = v[inside, :2]
    rho = np.linalg.norm(xy, axis=1, keepdims=True)
    dirn = xy / np.maximum(rho, 1e-12)
    rim = float(np.sqrt(max(1.0 - cap * cap, 1e-9)))          # xy radius of the cap boundary
    # THE WIDTH IS CONSTANT IN t: waist at the tip, and blended to the rim over the first fifth so the
    # neck is a neck and not a step the mesh cannot represent.
    neck = np.clip(t / 0.2, 0.0, 1.0)[:, None]
    v[inside, :2] = dirn * (rim * (1 - neck) + waist * neck)
    v[inside, 2] = cap + t * length
    return v, mt


def _cylinder_on_axis(v, a, cap, length, waist, best=None, u=None):
    """Map the polar cap about axis `a` onto a cylinder. The one construction every tube here uses.

    Shared so a tube, a fork's prong and one of many tubes are the SAME shape by construction. If they
    were built three ways, a metric that behaved differently on them would be ambiguous between a real
    difference and a fixture difference.
    """
    r = np.linalg.norm(v, axis=1)
    if u is None:
        u = v / np.maximum(r[:, None], 1e-12)
    if best is None:
        best = u @ a
    sel = best > cap
    if not sel.any():
        return v
    e1 = np.cross(a, [0.0, 1.0, 0.0])
    if np.linalg.norm(e1) < 1e-9:
        e1 = np.cross(a, [1.0, 0.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    rim = float(np.sqrt(max(1.0 - cap * cap, 1e-9)))
    t = np.clip((best[sel] - cap) / (1.0 - cap + 1e-12), 0.0, 1.0)
    c1, c2 = v[sel] @ e1, v[sel] @ e2
    rho = np.hypot(c1, c2)
    d1, d2 = c1 / np.maximum(rho, 1e-12), c2 / np.maximum(rho, 1e-12)
    neck = np.clip(t / 0.2, 0.0, 1.0)
    rad = rim * (1 - neck) + waist * neck
    along = cap + t * length
    v[sel] = (a[None, :] * along[:, None] + e1[None, :] * (d1 * rad)[:, None]
              + e2[None, :] * (d2 * rad)[:, None])
    return v


def branched(length=2.0, trunk=0.55, tilt=34.0, waist=0.26, cap=0.80, **kw):
    """ONE trunk that FORKS: one protrusion base, two tips. Okuda's branching phenotype.

    THE DISTINCTION THAT COST FOUR ATTEMPTS, and the classifier had it right all along.
    `morphology.classify` calls a shape *branched* when it counts MORE TIPS THAN BASES -- a fork from a
    single trunk -- and *tube* when tips and bases match. My fourth version built two cylinders from two
    separate polar caps, which reads as `n_tubes` 2, `n_tips` 2, morphology *tube*: two tubes on one
    body, correctly identified, and not a branch. That construction is now `multi_tube`, which is its
    own useful fixture.

    A fork therefore has to share a base. This grows the trunk as a cylinder from `cap` to `trunk` along
    z, then continues as TWO cylinders about axes tilted either side -- so the two prongs leave from one
    neck and the crotch between them is the trunk's own surface.
    """
    v, mt = base_sphere(**kw)
    v = np.asarray(v, float).copy()
    r = np.linalg.norm(v, axis=1)
    u = v / np.maximum(r[:, None], 1e-12)
    z = u[:, 2]
    inside = z > cap
    if not inside.any():
        return v, mt
    t = np.clip((z - cap) / (1.0 - cap + 1e-12), 0.0, 1.0)
    rim = float(np.sqrt(max(1.0 - cap * cap, 1e-9)))
    th = np.deg2rad(float(tilt))

    # THE TRUNK: a single cylinder, exactly as `tubed` builds one.
    stem = inside & (t <= trunk)
    if stem.any():
        tt = t[stem] / max(trunk, 1e-9)
        xy = v[stem, :2]
        rho = np.linalg.norm(xy, axis=1, keepdims=True)
        dirn = xy / np.maximum(rho, 1e-12)
        neck = np.clip(tt / 0.3, 0.0, 1.0)[:, None]
        v[stem, :2] = dirn * (rim * (1 - neck) + waist * neck)
        v[stem, 2] = cap + tt * (trunk * length)

    # THE PRONGS: two cylinders leaving the trunk's top, one per side of x.
    fork = inside & (t > trunk)
    if fork.any():
        z0 = cap + trunk * length
        tf = (t[fork] - trunk) / max(1.0 - trunk, 1e-9)
        sgn = np.where(v[fork, 0] >= 0, 1.0, -1.0)
        axis_x = sgn * np.sin(th)
        run = tf * (1.0 - trunk) * length
        # each prong keeps a constant cross-section about its own tilted axis
        ang = np.arctan2(v[fork, 1], np.abs(v[fork, 0]) + 1e-12)
        v[fork, 0] = axis_x * run + sgn * waist * np.cos(ang) * np.cos(th)
        v[fork, 1] = waist * np.sin(ang)
        v[fork, 2] = z0 + np.cos(th) * run - waist * np.cos(ang) * np.sin(th) * sgn * sgn
    return v, mt


def multi_tube(n=5, length=1.9, waist=0.24, cap=None, area_frac=0.34, **kw):
    """`n` separate tubes on one body -- the fixture for COUNTING.

    Cedric, 5 August: *"can you add a geometry with many tubes to let count the tubes"*. Every other
    fixture has zero, one or two, so `n_tubes` was only ever checked against small numbers -- and the
    defect it had (counting the ends of an ellipsoid) was a COUNT error. A fixture with a known count of
    five is the only thing that can show whether the count is right rather than merely non-zero.

    Axes are a Fibonacci arrangement, so they are maximally separated for their number and no two tubes
    merge: a merge would make the true count ambiguous, which is the mistake the six-spot fixture made
    before its seeds were separated.

    `cap` SCALES WITH `n` SO THE BODY STAYS THE MAJORITY, and getting this wrong produced a false
    finding I nearly shipped. At a fixed cap of 0.80 each tube claims about 10% of the sphere, so seven
    tubes is 70% of the tissue -- and then the MEDIAN radius is itself a tube cell. Every protrusion
    metric collapsed to zero at n >= 7 and I read it as the outlier test defeating itself; the real
    cause was a fixture that is a starburst rather than a body with tubes. A spherical cap subtending
    polar angle t covers (1 - cos t)/2 of the sphere, so `cap` is solved from `area_frac`: the tubes
    together take at most a third of the surface whatever `n` is, and the body remains what the
    detector's median is measuring.
    """
    if cap is None:
        # (1 - cap)/2 per cap x n = area_frac  ->  cap = 1 - 2*area_frac/n
        cap = float(np.clip(1.0 - 2.0 * area_frac / max(int(n), 1), 0.55, 0.97))
    from mesh_ops import fib_sphere
    v, mt = base_sphere(**kw)
    v = np.asarray(v, float).copy()
    axes = np.asarray(fib_sphere(int(n), 1.0), float)
    r = np.linalg.norm(v, axis=1)
    u0 = v / np.maximum(r[:, None], 1e-12)
    proj = np.stack([u0 @ (a / np.linalg.norm(a)) for a in axes], axis=1)
    which = np.argmax(proj, axis=1)
    for k, a in enumerate(axes):
        a = a / np.linalg.norm(a)
        best = np.where(which == k, proj[:, k], -1.0)
        v = _cylinder_on_axis(v, a, cap, length, waist, best=best, u=u0)
    return v, mt


def undulated(amp=0.22, k=6, **kw):
    """Many shallow waves over the whole surface: elevated shape index, no dominant feature.

    This is Okuda's undulation, and it is the fixture that stops `protr` being read as "a tube":
    a lobed sphere raises it above 1 without any finger existing.
    """
    v, mt = base_sphere(**kw)
    v = v.copy()
    r = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / np.maximum(r, 1e-12)
    phi = np.arctan2(u[:, 1], u[:, 0])
    f = 1.0 + amp * np.cos(k * phi) * np.cos(k * np.arcsin(np.clip(u[:, 2], -1, 1)))
    return u * (r * f[:, None]), mt


def self_intersecting(depth=1.6, cap=0.6, **kw):
    """A polar cap pushed THROUGH the centre -- the one thing a physical tissue cannot do.

    `genus` cannot see this: Euler characteristic is combinatorial, so it reports a sphere however
    deeply the shell is folded through itself. `ray_single_frac` is the measurement that can, and this
    fixture exists to prove it does.
    """
    v, mt = base_sphere(**kw)
    v = v.copy()
    z = v[:, 2]
    t = np.clip((z - cap) / (1.0 - cap), 0.0, 1.0)
    s = t * t * (3 - 2 * t)
    v[:, 2] = z - s * depth                             # inward, past the origin
    return v, mt


GEOMETRIES = {"sphere": sphere, "prolate": prolate, "oblate": oblate, "undulated": undulated,
              "tubed": tubed, "branched": branched, "multi_tube": multi_tube,
              "self_intersecting": self_intersecting}


def measure(fixture, act=None, a_sw=None, **kw):
    """Run the real registry over a fixture. No mocking: this is `frame_metrics` as the round calls it."""
    v, mt = GEOMETRIES[fixture](**kw)
    from tissue_analysis import frame_metrics
    with contextlib.redirect_stdout(io.StringIO()):
        return frame_metrics(v, mt, act=act, a_sw=a_sw)


# ================================================================ what each shape must report

def test_sphere_is_the_null():
    """EVERY SHAPE METRIC HAS A KNOWN VALUE HERE, which is what makes it the reference. A metric that
    cannot report "no" correctly cannot report "yes" credibly."""
    print("\na sphere reads as a sphere")
    m = measure("sphere")
    check(abs(m["protr"] - 1.0) < 0.06, f"protr {m['protr']} is not ~1.0 on a sphere")
    check(abs(m["gyr_prolate"] - 1.0) < 0.08, f"gyr_prolate {m['gyr_prolate']} is not ~1.0")
    check(abs(m["gyr_oblate"]) < 0.05, f"gyr_oblate {m['gyr_oblate']} is not ~0")
    check(abs(m["reduced_volume"] - 1.0) < 0.05,
          f"reduced_volume {m['reduced_volume']} is not ~1.0 -- a sphere is the maximum")
    check(m["ray_single_frac"] == 1.0,
          f"ray_single_frac {m['ray_single_frac']} -- a convex shell must cross every ray once")
    check(m["genus"] == 0, f"genus {m['genus']} is not 0")
    check(m.get("n_tubes", 0) == 0, f"n_tubes {m.get('n_tubes')} -- a sphere has no tube")
    check(m["shape_idx_min"] >= 3.5449 - 1e-3,
          f"shape_idx_min {m['shape_idx_min']} is below the geometric floor 2*sqrt(pi) -- the ruler "
          f"is lying, and this is the check that can prove it")


def test_prolate_is_elongated_without_a_protrusion():
    """The distinction Okuda's phenotype axis needs: a long axis is not a tip."""
    print("\na rugby ball is prolate, and is NOT a tube")
    s, m = measure("sphere"), measure("prolate", aspect=2.0)
    check(m["gyr_prolate"] > 1.5, f"gyr_prolate {m['gyr_prolate']} did not rise on a 2:1 ellipsoid")
    check(m["gyr_oblate"] < 0.1, f"gyr_oblate {m['gyr_oblate']} -- a prolate shape is not oblate")
    check(m["reduced_volume"] < 0.99,
          f"reduced_volume {m['reduced_volume']} -- elongation costs excess area")
    check(m.get("n_tubes", 0) == 0, f"n_tubes {m.get('n_tubes')} -- there is no tube here")
    check(m["protr"] < 1.5, f"protr {m['protr']} is tube-like on a shape with no protrusion "
                            f"(sphere reads {s['protr']})")


def test_oblate_is_seen_at_all():
    print("\na collapsed disc is oblate")
    m = measure("oblate", aspect=2.5)
    check(m["gyr_oblate"] > 0.15, f"gyr_oblate {m['gyr_oblate']} did not rise on a flattened shell")
    # MY EXPECTATION WAS WRONG, NOT THE METRIC. `gyr_prolate` is l1/mean(l2,l3), and on an oblate
    # disc l3 collapses while l1 and l2 stay large -- so the ratio rises to 1.73 here. That is correct
    # arithmetic and a real trap: an agent told "gyr_prolate grows with elongation" would read a
    # collapsing vesicle as a tube. What actually separates them is `gyr_oblate`, so that is what is
    # asserted, and the caveat is now in the metric's own docstring.
    check(m["gyr_prolate"] > 1.3,
          f"gyr_prolate {m['gyr_prolate']} -- l3 collapses on a disc, so this rises too")
    check(m["gyr_oblate"] > 10 * measure("prolate", aspect=2.0)["gyr_oblate"],
          f"gyr_oblate does not separate a disc ({m['gyr_oblate']}) from a rod -- and it is the only "
          f"metric that can")
    check(m.get("n_tubes", 0) == 0, f"n_tubes {m.get('n_tubes')} on a flattened disc")


def test_a_tube_is_seen_as_a_tube():
    """THE CAMPAIGN'S WHOLE TARGET. If these fail, no run can ever be believed to have made one."""
    print("\na tube reads as a tube")
    s, m = measure("sphere"), measure("tubed", length=2.2)
    check(m["protr"] > s["protr"] + 0.2, f"protr {m['protr']} vs sphere {s['protr']} -- a finger "
                                         f"three cell-diameters long moved it by nothing")
    check(m["protr"] > 1.3, f"protr {m['protr']} is below the 1.3 this campaign treats as a tube")
    check(m["protr_p99"] >= m["protr"],
          f"protr_p99 {m['protr_p99']} < protr {m['protr']} -- p99 cannot be under p95")
    check(m["gyr_prolate"] > 1.1, f"gyr_prolate {m['gyr_prolate']} did not rise")
    check(m.get("n_tips", 0) >= 1, f"n_tips {m.get('n_tips')} -- the classifier found no tip")
    check(m.get("protrusion_aspect_max", 0) > 1.0,
          f"protrusion_aspect_max {m.get('protrusion_aspect_max')} -- a finger is not a bulge")
    check(m["ray_single_frac"] == 1.0,
          f"ray_single_frac {m['ray_single_frac']} -- this fixture does not self-intersect, so a "
          f"value below 1 means the ray test reports folding that is not there")


def test_a_fork_on_a_trunk_is_INVISIBLE_to_n_tips():
    """THE CAMPAIGN CANNOT COUNT ITS OWN TARGET PHENOTYPE, and this fixture is how that was found.

    Okuda's branching figure is a TUBE THAT SPLITS: one trunk, one base, two tips. `n_tips` clusters
    the outermost cells of a protrusion by direction **about the tissue centroid**, within a 25-degree
    cone -- and a trunk puts those tips far from the centroid, where angular separation compresses.
    Swept over the fork's tilt:

        tilt            20     34     45     60     75   degrees
        half-angle    11.2   15.3   18.4   22.4   25.0   from the centroid
        n_tips           1      1      1      1      1

    At 75 degrees the prongs are splayed almost horizontally and it STILL reads one. So a fork on a
    trunk is not merely hard to detect, it is unreachable at any tilt, and `morphology.classify` can
    never return *branched* for the shape the paper shows.

    An earlier fixture of mine did read two tips -- because it had NO trunk: two cylinders straight off
    the body, which is `multi_tube(n=2)` and correctly classified *tube*, not a branch. Confusing the
    two is what kept this hidden.

    THE FIX IS NOT APPLIED HERE. Clustering tips about the protrusion's OWN base instead of the tissue
    centroid would see the fork, but that changes a measurement the campaign has been recording, and
    such a change is a decision rather than a tidy-up. The limitation is pinned instead: if `n_tips`
    ever reads 2 on this fixture, the clustering was changed and every past "no branching" result was
    measured with a different ruler.
    """
    print("\na fork on a trunk is invisible to n_tips -- pinned, not worked around")
    for tilt in (34, 60, 75):
        m = measure("branched", tilt=tilt)
        check(m.get("n_tips") == 1,
              f"n_tips {m.get('n_tips')} on a {tilt}-degree fork -- if this is now 2 the tip "
              f"clustering was changed and past 'no branching' results are not comparable")
    m = measure("branched")
    check(m.get("n_tubes", 0) >= 1, f"n_tubes {m.get('n_tubes')} -- the trunk itself must be seen")
    check(m["protr"] > 1.3, f"protr {m['protr']} -- the fork is a protrusion whatever its tip count")
    # two tubes off the body ARE distinguishable, which is the contrast that locates the limitation
    two = measure("multi_tube", n=2)
    check(two.get("n_tips") == 2,
          f"n_tips {two.get('n_tips')} for two tubes off the body -- these are separable, and a fork "
          f"on a trunk is not; the difference is distance from the centroid, not shape")


def test_chemistry_metrics_on_known_fields():
    """The pattern metrics, against fields whose answer is arithmetic rather than a shape."""
    print("\nthe chemistry metrics on fields with known answers")
    v, mt = base_sphere()
    nF = mt["nF"]

    flat = np.full(nF, 0.4)
    m = measure("sphere", act=flat, a_sw=0.2)
    check(m["act_cv"] == 0.0, f"act_cv {m['act_cv']} on a UNIFORM field -- there is no pattern")
    check(m["act_alive"] == 0, f"act_alive {m['act_alive']} on a uniform field")
    check(m["red_frac"] == 1.0, f"red_frac {m['red_frac']} -- every cell is above a_sw=0.2")
    check("corr_act_rad" not in m,
          f"corr_act_rad {m.get('corr_act_rad')} was computed on a dead field -- the refusal is the "
          f"whole point, and Pearson will happily return a confident number")

    half = np.where(np.arange(nF) % 2 == 0, 1.0, 0.0)
    m2 = measure("sphere", act=half, a_sw=0.5)
    check(abs(m2["act_mean"] - 0.5) < 0.02, f"act_mean {m2['act_mean']} on a half-on field")
    check(m2["act_cv"] > 0.9, f"act_cv {m2['act_cv']} -- a half-on field is maximally patterned")
    check(abs(m2["red_frac"] - 0.5) < 0.02, f"red_frac {m2['red_frac']} is not ~0.5")
    check(m2["act_alive"] == 1, f"act_alive {m2['act_alive']} on a live pattern")

    # A FIELD THAT TRACKS RADIUS, on a shape that HAS a radius range: corr_act_rad must find it.
    vt, mtt = tubed(length=2.2)
    rad = np.linalg.norm(vt, axis=1)
    from tissue_analysis import _cell_centroids, frame_metrics
    with contextlib.redirect_stdout(io.StringIO()):
        cen, radl, live = _cell_centroids(vt, mtt)
        m3 = frame_metrics(vt, mtt, act=radl.astype(float), a_sw=float(np.median(radl[live])))
    check(m3.get("corr_act_rad", 0) > 0.9,
          f"corr_act_rad {m3.get('corr_act_rad')} -- the activator IS the radius here, so a "
          f"correlation below 1 means the pairing is wrong")
    check(m3.get("act_at_tip", 0) > 1.0,
          f"act_at_tip {m3.get('act_at_tip')} -- the field is highest at the tip by construction")


def test_the_table():
    """Every fixture, every headline metric, in one table. Not an assertion -- a thing to READ.

    A number that is merely inside a tolerance can still be wrong in a way no bound catches, and the
    campaign has twice reported a rail as a result. Printing the whole grid is how a rail becomes
    visible: a column that does not move across seven shapes is not measuring shape.
    """
    print("\nthe grid -- read it, and look for a column that does not move")
    cols = ["protr", "protr_p99", "r_cv", "gyr_prolate", "gyr_oblate", "reduced_volume",
            "n_tubes", "n_tips", "protrusion_aspect_max", "ray_single_frac", "shape_idx_p95"]
    print("       " + "".join(f"{c[:9]:>10}" for c in cols))
    seen = {c: set() for c in cols}
    for name in GEOMETRIES:
        m = measure(name)
        row = []
        for c in cols:
            v = m.get(c)
            seen[c].add(None if v is None else round(float(v), 3))
            row.append("    -" if v is None else f"{float(v):10.3f}")
        print(f"  {name[:5]:<5}" + "".join(row))
    dead = [c for c, vs in seen.items() if len(vs) == 1]
    check(not dead, f"metric(s) identical across all seven shapes -- not measuring shape: {dead}")


# ================================================================ every metric, or a stated reason

# WHY A COVERAGE TEST AND NOT JUST MORE ASSERTIONS. Cedric: "test all metrics against known geometry."
# The registry declares 67 quantities and the assertions above touch about half, so the untested half
# was invisible -- exactly the shape of every defect this phase found. A metric is now either checked
# on a fixture or listed here WITH A REASON, and the list is asserted to be exhaustive. Adding a metric
# without doing one or the other fails this test.
EXEMPT = {
    # ---- need a RUN, not a frame: these are reduced from the whole series by run_one
    "act_alive_frac": "run-level: a fraction over frames, undefined on one frame",
    "act_extinct_frame": "run-level: the frame a pattern died on",
    "act_peak_frame": "run-level: the frame the activator peaked on",
    "div_blocked": "run-level: whether the reservoir ever refused a division",
    "div_blocked_first_frame": "run-level: the frame it first refused",
    "buf_full": "run-level: whether the tissue reached its vertex buffer",
    "mech_p_ratio": "needs the mechanics probe, which solves a pressure field -- not a geometry",
    "Q_drop": "needs the quasi-static relax probe, which re-runs the simulation",
    # ---- rejected: measured to lie, kept nameable only so a prediction on one gets its reason
    "ta_aspect_len_over_diam": "rejected (F15/F16) -- no longer produced",
    "ta_tube_len_final": "rejected (F15/F16) -- no longer produced",
    "retention": "rejected (F15/F16) -- no longer produced",
    "autocorr_hops_uncalibrated": "rejected (F010, uncalibrated) -- not a length",
    # ---- a string, not a number
    "morph_why": "the classifier's prose reason; asserted through `morphology` instead",
}


def test_every_metric_is_produced_on_a_geometry():
    """Every registry metric appears on a fixture, or is EXEMPT with a reason. No silent gaps."""
    print("\nevery metric is measured on a geometry, or exempt with a reason")
    import metrics as M
    v, mt = base_sphere()
    nF = mt["nF"]
    # a spotted field so the pattern and coupling metrics are genuinely exercised
    rng = np.random.default_rng(4)
    cen, radl, live = None, None, None
    from tissue_analysis import _cell_centroids
    with contextlib.redirect_stdout(io.StringIO()):
        cen, radl, live = _cell_centroids(v, mt)
    u = cen / np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-12)
    seeds = u[rng.choice(nF, size=6, replace=False)]
    spotted = np.exp(8.0 * (u @ seeds.T)).max(axis=1)
    spotted /= spotted.max()

    seen = {}
    for name in list(GEOMETRIES):
        seen.update({k: v2 for k, v2 in measure(name, act=spotted, a_sw=0.35).items()})
    # the tube fixture with a radius-tracking field, for the coupling family
    vt, mtt = tubed(length=2.2)
    from tissue_analysis import frame_metrics
    with contextlib.redirect_stdout(io.StringIO()):
        _c, rt, lt = _cell_centroids(vt, mtt)
        seen.update(frame_metrics(vt, mtt, act=rt.astype(float),
                                  a_sw=float(np.median(rt[lt]))))

    missing, unexplained = [], []
    for m in M.REGISTRY.values():
        if m.name in seen:
            continue
        if m.name in EXEMPT:
            continue
        (unexplained if m.conditional else missing).append(m.name)
    check(not missing, f"registry metrics that NO geometry produced and nothing exempts: {missing}")
    if unexplained:
        print(f"       conditional and absent here (declared): {unexplained}")
    stale = sorted(set(EXEMPT) - {m.name for m in M.REGISTRY.values()})
    check(not stale, f"EXEMPT names no longer in the registry -- the list has drifted: {stale}")
    covered = len([m for m in M.REGISTRY.values() if m.name in seen])
    print(f"       {covered}/{len(M.REGISTRY)} produced on a geometry, "
          f"{len(EXEMPT)} exempt with a reason")


def test_the_pattern_metrics_on_a_spotted_field():
    """`n_spots` and `spot_spacing_cells` are the only pattern LENGTH the campaign can compare with the
    paper ("about five spots on a 2000-cell ball"), and neither was ever checked against a field with a
    known number of spots."""
    print("\nsix painted spots are counted as spots")
    v, mt = base_sphere()
    nF = mt["nF"]
    from tissue_analysis import _cell_centroids
    with contextlib.redirect_stdout(io.StringIO()):
        cen, _r, _l = _cell_centroids(v, mt)
    u = cen / np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-12)
    # SEEDS THAT CANNOT MERGE. My first version drew six at RANDOM and got a count of 3, which a loose
    # bound accepted -- and which was untestable either way: two random spots near each other
    # legitimately merge into one, so the fixture could not distinguish an undercount from a merge. A
    # Fibonacci arrangement is maximally separated by construction, so the answer IS six and any other
    # count is the metric's error rather than the fixture's ambiguity.
    from mesh_ops import fib_sphere
    seeds = np.asarray(fib_sphere(6, 1.0), float)
    act = np.exp(14.0 * (u @ seeds.T)).max(axis=1)
    act /= act.max()
    m = measure("sphere", act=act, a_sw=0.5)
    n = m.get("n_spots")
    check(n is not None, "n_spots was not produced on a spotted field")
    check(n == 6, f"n_spots {n} for six maximally separated painted spots -- Okuda's own comparison "
                  f"is 'about five spots on a 2000-cell ball', so a count that misses by a factor "
                  f"cannot be held against a paper")
    check(m.get("spot_spacing_cells") is None or m["spot_spacing_cells"] > 0,
          f"spot_spacing_cells {m.get('spot_spacing_cells')} is not a positive length")
    flat = measure("sphere", act=np.full(nF, 0.4), a_sw=0.5)
    check(flat.get("n_spots", 0) <= 1,
          f"n_spots {flat.get('n_spots')} on a UNIFORM field -- there are no spots to count")


if __name__ == "__main__":
    print(f"metrics against generated geometries -- {N_CELLS} cells, jitter {JITTER}")
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            FAIL.append(f"{fn.__name__} raised {e}")
    print("\n" + "=" * 62)
    print(f"  {len(FAIL)} failure(s)" if FAIL else "  all checks passed")
    for f in FAIL:
        print("   - " + f)
    sys.exit(1 if FAIL else 0)
