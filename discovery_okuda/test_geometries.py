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
TYSSUE = os.path.join(os.path.dirname(HERE), "prototype", "Tyssue")
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
    from tyssue_ops3d import build_sphere_mesh
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


def tubed(length=2.2, cap=0.75, waist=0.45, **kw):
    """One finger from the north pole: a sphere with a tube.

    The deformation is smooth in the polar angle so no face is inverted: vertices above `cap` in
    z are pulled outward along z and squeezed in xy, with the pull tapering to zero at the boundary
    so the tube joins the body continuously.
    """
    v, mt = base_sphere(**kw)
    v = v.copy()
    z = v[:, 2]
    t = np.clip((z - cap) / (1.0 - cap), 0.0, 1.0)      # 0 at the collar, 1 at the pole
    s = t * t * (3 - 2 * t)                             # smoothstep: no kink at the collar
    v[:, 2] = z + s * (length - 1.0)
    scale = 1.0 - s * (1.0 - waist)
    v[:, 0] *= scale
    v[:, 1] *= scale
    return v, mt


def branched(length=2.0, cap=0.72, waist=0.5, **kw):
    """Two fingers, north and south. `n_tips` must see two; `protr` cannot tell it from one."""
    v, mt = base_sphere(**kw)
    v = v.copy()
    for sign in (+1.0, -1.0):
        z = v[:, 2] * sign
        t = np.clip((z - cap) / (1.0 - cap), 0.0, 1.0)
        s = t * t * (3 - 2 * t)
        v[:, 2] = v[:, 2] + sign * s * (length - 1.0)
        scale = 1.0 - s * (1.0 - waist)
        v[:, 0] *= scale
        v[:, 1] *= scale
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


GEOMETRIES = {"sphere": sphere, "prolate": prolate, "oblate": oblate, "tubed": tubed,
              "branched": branched, "undulated": undulated,
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


def test_branched_is_distinguishable_from_one_tube():
    """`n_tips` is the only metric that can tell a branch from a tube, so it is the only one asserted
    to differ. `protr` is a radius ratio and CANNOT: two fingers of the same length as one give the
    same tail statistic, which is worth pinning so nobody reads a rise in protr as branching."""
    print("\ntwo fingers are distinguishable from one")
    one, two = measure("tubed", length=2.0), measure("branched", length=2.0)
    check(two.get("n_tips", 0) > one.get("n_tips", 0) or two.get("n_tips", 0) >= 2,
          f"n_tips one={one.get('n_tips')} two={two.get('n_tips')} -- branching is invisible")
    check(abs(two["protr"] - one["protr"]) < 0.5,
          f"protr one={one['protr']} two={two['protr']} -- a tail statistic should NOT separate "
          f"these, and if it does it is responding to something other than the branch")


def test_undulation_is_not_read_as_a_tube():
    """The failure mode that would let the campaign claim a figure it did not reproduce."""
    print("\nan undulating shell is not a tube")
    m = measure("undulated", amp=0.22, k=6)
    check(m.get("n_tubes", 0) == 0, f"n_tubes {m.get('n_tubes')} on a lobed sphere")
    check(m["gyr_prolate"] < 1.3,
          f"gyr_prolate {m['gyr_prolate']} -- undulation has no dominant axis")
    check(m["shape_idx_p95"] > 0, "shape_idx_p95 was not measured")
    check(m["r_cv"] > measure("sphere")["r_cv"],
          f"r_cv {m['r_cv']} did not rise -- the radius distribution IS the undulation")


def test_self_intersection_is_caught_and_genus_cannot_see_it():
    """The measurement premise 11 rests on, and the one that cost a wrong conclusion when genus was
    trusted instead: a shell crumpled seventeen layers deep still reports genus 0."""
    print("\na shell folded through itself is caught -- and genus misses it")
    m = measure("self_intersecting", depth=1.6)
    check(m["ray_single_frac"] < 1.0,
          f"ray_single_frac {m['ray_single_frac']} -- the fold was not detected")
    check(m["genus"] == 0,
          f"genus {m['genus']} -- if this ever becomes non-zero the comment claiming genus is blind "
          f"to self-intersection is out of date and must be rewritten")
    check(m.get("morphology") in (None, "invalid") or m["ray_single_frac"] < 1.0,
          f"morphology {m.get('morphology')} on a self-intersecting shell")


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
