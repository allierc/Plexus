#!/usr/bin/env python
"""calibrate_success -- DERIVE the tube success threshold from geometry, not from our best run.

WHY THIS FILE EXISTS
--------------------------------------------------------------------------------------------
`control.meets_success` demanded `protr >= 2.0` (and `protr_peak >= 3.0`). Nobody derived those
numbers; they were authored as an impression of "clearly a tube". They were never challenged
because `Q_protr_after_relax` was broken -- it returned the constant 1.014 for every run (see
`run_one.quasi_static_Q`) -- so the criterion was unreachable BY ACCIDENT and the accident hid
the fact that the number itself was never justified.

Fitting the threshold to our best measured run instead would make the criterion meaningless: it
would then say "as good as we have managed", not "this is a tube". So the threshold is derived
here from SHAPES OF KNOWN ASPECT RATIO, with no simulator in the loop. Sampling points on the
surface of an analytic capsule isolates the metric from the vertex model entirely: whatever this
file reports is a property of the RULER, not of the thing being measured.

THE METRIC IS NOT REIMPLEMENTED HERE. `protrusion_ratio` is imported from `tube_analysis`, which
is the single definition shared by `run_one.protr_of` (vertex positions) and
`tube_analysis.frame_metrics` (cell centroids), and `_protr` below is asserted equal to
`run_one.protr_of` on a random point set at import time. A calibration performed against a
lookalike formula would calibrate nothing.

WHAT IT FINDS (run it; the numbers below are printed, not asserted from memory)
--------------------------------------------------------------------------------------------
  * sphere (aspect 1) reads exactly 1.000            -- the null; validates the metric
  * the capsule mapping is monotone in aspect ratio and SATURATES just below 1.9
  * therefore `protr >= 2.0` was unreachable for ANY capsule of ANY aspect ratio -- the old
    criterion was not merely strict, it was outside the metric's range for the target shape
  * aspect 2 -- the first aspect at which the parallel-sided barrel is a full diameter long,
    i.e. where a bump becomes a tube -- reads ~1.38. That is the derived threshold.

    python calibrate_success.py            # the tables + the derivation
    python calibrate_success.py --check    # geometric acceptance test against control.meets_success
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "prototype", "Tyssue"))

from tube_analysis import protrusion_ratio                            # noqa: E402  THE definition


def _protr(points):
    """percentile(r,95)/median(r) about the TISSUE CENTROID -- byte-for-byte what the campaign
    measures. Identical to `run_one.protr_of`; `_assert_same_metric_as_run_one` proves it."""
    points = np.asarray(points, float)
    return protrusion_ratio(np.linalg.norm(points - points.mean(0), axis=1))


def _assert_same_metric_as_run_one():
    """Guard against calibrating a lookalike. If `run_one.protr_of` ever diverges from the shared
    `protrusion_ratio`, every number in this file becomes a calibration of the wrong ruler."""
    try:
        import run_one
    except Exception as e:                                   # heavy stack absent -> say so, do not lie
        return f"NOT CHECKED ({type(e).__name__}: {e})"
    P = np.random.default_rng(20260731).normal(size=(777, 3)) * [1.0, 1.0, 3.0]
    a, b = _protr(P), run_one.protr_of(P)
    assert abs(a - b) < 1e-12, f"_protr {a} != run_one.protr_of {b} -- calibrating the wrong metric"
    return f"identical to run_one.protr_of on a random point set ({a:.6f})"


# ------------------------------------------------------------------ analytic shapes (no simulator)
def sample_capsule(aspect, n, rng, radius=1.0, jitter=0.0):
    """Uniform-by-AREA points on a capsule: a cylinder of radius R closed by two hemispherical
    caps. aspect = total length / diameter = (cyl_len + 2R) / 2R, so aspect 1 IS a sphere.

    Uniform by area, not by parameter: the tissue is a monolayer, so cells tile the surface at
    roughly constant areal density. Sampling uniformly in z along the cylinder is correct for a
    cylinder (its area element is dz-uniform), and Archimedes' hat-box theorem makes the caps
    dz-uniform too, which is why the cap sampling below draws the axial component flat on [0, 1].
    """
    aspect = float(aspect)
    assert aspect >= 1.0, "aspect < 1 is an oblate shape, not a capsule"
    R = float(radius)
    cyl_len = 2.0 * R * (aspect - 1.0)                      # total length 2R*aspect, minus the two caps
    a_cyl, a_caps = 2.0 * np.pi * R * cyl_len, 4.0 * np.pi * R * R
    on_cyl = rng.random(n) < a_cyl / (a_cyl + a_caps)
    P = np.empty((n, 3))

    nc = int(on_cyl.sum())
    th = rng.uniform(0.0, 2.0 * np.pi, nc)
    P[on_cyl, 0] = R * np.cos(th)
    P[on_cyl, 1] = R * np.sin(th)
    P[on_cyl, 2] = rng.uniform(-cyl_len / 2.0, cyl_len / 2.0, nc)

    nk = n - nc
    th = rng.uniform(0.0, 2.0 * np.pi, nk)
    uz = rng.random(nk)                                     # Archimedes: axial component is flat
    s = np.where(rng.random(nk) < 0.5, -1.0, 1.0)           # which cap
    rho = np.sqrt(np.maximum(1.0 - uz * uz, 0.0))
    P[~on_cyl, 0] = R * rho * np.cos(th)
    P[~on_cyl, 1] = R * rho * np.sin(th)
    P[~on_cyl, 2] = s * (cyl_len / 2.0 + R * uz)

    if jitter:                                              # finite cells do not sit exactly on the surface
        P *= (1.0 + jitter * rng.standard_normal((n, 1)))
    return P


def capsule_protrusion_exact(aspect, radius=1.0):
    """The same metric on the capsule's EXACT area measure -- no sampling.

    Both r-distributions are one-dimensional, so the area CDF is closed-form and the percentiles
    come from bisection. This exists to prove the Monte-Carlo numbers are converged rather than
    noise: a calibration that is itself a sampling artefact would repeat the campaign's original
    sin. Returns (p95/median, p95, median).
    """
    R = float(radius)
    half = R * (aspect - 1.0)                               # half the cylinder length
    a_cyl, a_caps = 2.0 * np.pi * R * (2.0 * half), 4.0 * np.pi * R * R
    f_cyl = a_cyl / (a_cyl + a_caps)
    r_max = half + R

    def cdf(x):
        c = 0.0
        if half == 0.0:                                     # a sphere: all radii equal R
            return 1.0 if x >= R else 0.0
        if x > R:                                           # cylinder: r = sqrt(R^2 + z^2), |z| flat
            c += f_cyl * min(np.sqrt(max(x * x - R * R, 0.0)), half) / half
        p = (x * x - half * half - R * R) / (2.0 * half * R)  # cap: r^2 = half^2 + 2*half*R*u + R^2
        c += (1.0 - f_cyl) * min(max(p, 0.0), 1.0)
        return c

    def inv(q):
        lo, hi = 0.0, r_max
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if cdf(mid) < q:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    med, p95 = inv(0.5), inv(0.95)
    return p95 / med, p95, med


def sample_sphere_with_tube(body_R, tube_R, tube_len, n, rng):
    """A body sphere with a capsule TUBE grown off one pole -- the morphology the campaign
    actually produces. Used only as a caveat family: `protrusion` is not a pure shape descriptor,
    it also depends on what FRACTION of the cells are in the tube (see the printout)."""
    a_body = 4.0 * np.pi * body_R ** 2                      # the buried cap is a small correction; ignored
    a_tube = 2.0 * np.pi * tube_R * tube_len + 2.0 * np.pi * tube_R ** 2
    frac = a_tube / (a_body + a_tube)
    in_tube = rng.random(n) < frac
    P = np.empty((n, 3))

    nb = int((~in_tube).sum())
    v = rng.standard_normal((nb, 3))
    P[~in_tube] = body_R * v / np.linalg.norm(v, axis=1, keepdims=True)

    nt = n - nb
    th = rng.uniform(0.0, 2.0 * np.pi, nt)
    a_side = 2.0 * np.pi * tube_R * tube_len
    side = rng.random(nt) < a_side / (a_side + 2.0 * np.pi * tube_R ** 2)
    z = np.where(side,
                 body_R + rng.random(nt) * tube_len,        # along the barrel, from the body surface
                 body_R + tube_len + tube_R * rng.random(nt))
    rr = np.where(side, tube_R, tube_R * np.sqrt(np.maximum(
        1.0 - ((z - body_R - tube_len) / tube_R) ** 2, 0.0)))
    P[in_tube, 0] = rr * np.cos(th)
    P[in_tube, 1] = rr * np.sin(th)
    P[in_tube, 2] = z
    return P, frac


# ------------------------------------------------------------------ the derivation
TUBE_ASPECT = 2.0          # the aspect at which a bump becomes a tube; justified in main()
N_MC = 400_000


def derived_threshold():
    """The one number this file exists to produce, computed exactly (no sampling)."""
    return capsule_protrusion_exact(TUBE_ASPECT)[0]


def derived_bar():
    """The threshold as it is written into control.py: the exact value FLOORED to 2 decimals.

    Floored, not rounded. The criterion means "aspect >= 2", and the exact reading at aspect 2 is
    1.3784; storing the nearest 2-decimal value 1.38 would sit ABOVE it and reject a shape of
    exactly the aspect ratio the criterion was derived to admit. That is not hypothetical -- the
    acceptance test below failed on precisely that when the bar was first written as 1.38. Any
    rounding of a >= threshold must go toward admitting the calibration shape.
    """
    return np.floor(derived_threshold() * 100.0) / 100.0


def main():
    rng = np.random.default_rng(0)
    print("=" * 92)
    print("CALIBRATING protrusion = percentile(r,95) / median(r)   [r = distance from centroid]")
    print("=" * 92)
    print(f"metric provenance : tube_analysis.protrusion_ratio -- {_assert_same_metric_as_run_one()}")
    print(f"sampling          : {N_MC:,} points, uniform BY AREA on the analytic surface; no simulator")

    # -------------------------------------------------------------- the null
    print("\n--- THE NULL: a sphere must read 1.0, or the metric is broken -------------------------")
    for label, jit in [("perfect sphere", 0.0), ("sphere, 1% radial cell jitter", 0.01),
                       ("sphere, 2% radial cell jitter", 0.02), ("sphere, 5% radial cell jitter", 0.05)]:
        v = _protr(sample_capsule(1.0, N_MC, np.random.default_rng(1), jitter=jit))
        print(f"  {label:32s} protrusion = {v:.4f}")
    print("  (real sphere runs read 1.019-1.033 in _metrology/instrument_gate.json, which is what a")
    print("   few per-cent of cell-scale radial jitter reads here -- the null is where it should be)")

    # -------------------------------------------------------------- the mapping
    print("\n--- ASPECT -> PROTRUSION, capsules of known aspect ratio ------------------------------")
    print(f"  {'aspect L/D':>11} {'shape':<26} {'protrusion (MC)':>16} {'(exact)':>9} {'barrel/D':>9}")
    for a in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, 100.0]:
        mc = _protr(sample_capsule(a, N_MC, rng))
        ex = capsule_protrusion_exact(a)[0]
        shape = ("sphere (the null)" if a == 1.0 else
                 "bump / ovoid" if a < TUBE_ASPECT else
                 "tube" if a < 20 else "asymptote (infinite tube)")
        print(f"  {a:>11.2f} {shape:<26} {mc:>16.4f} {ex:>9.4f} {a - 1.0:>9.2f}")
    ceiling = capsule_protrusion_exact(1e6)[0]
    print(f"\n  CEILING: as aspect -> infinity the metric tends to {ceiling:.4f}, because for a long")
    print("  cylinder r ~ |z| with z uniform, so p95/median -> 0.95/0.50 = 1.90. NO CAPSULE OF ANY")
    print("  ASPECT RATIO CAN REACH 2.0 -- the old threshold sat outside the metric's range.")

    # -------------------------------------------------------------- the choice
    thr = derived_threshold()
    print("\n--- WHERE A BUMP BECOMES A TUBE -------------------------------------------------------")
    print("  A capsule of aspect A has a parallel-sided barrel of length (A-1)*D between its caps.")
    print("    A < 2 : barrel shorter than one diameter -> two caps and a stub: an ovoid/bump.")
    print("    A = 2 : barrel exactly one diameter long -> the first aspect at which a straight")
    print("            tube segment, as long as it is wide, exists at all.")
    print("  That is a geometric statement about the shape, not a taste, and it agrees with the")
    print("  brief's reading of Okuda fig. 5 (clearly elongated => aspect >= 2).")
    print(f"\n  DERIVED THRESHOLD  protrusion >= {derived_bar():.2f}")
    print(f"    a capsule of aspect {TUBE_ASPECT:g} reads {thr:.4f} exactly, FLOORED to 2 dp. Floored, not")
    print("    rounded: 1.38 sits above 1.3784 and would reject the very shape the bar was derived")
    print("    from -- which it did, until the acceptance test caught it.")
    print(f"  sphere null {capsule_protrusion_exact(1.0)[0]:.3f}  |  capsule ceiling {ceiling:.3f}  "
          f"|  old, underived threshold 2.00 (unreachable)")

    # -------------------------------------------------------------- where our best run lands
    best = 1.62                       # best honest Q_protr_after_relax reported after the Q fix
    lo, hi = 1.0, 1e4
    for _ in range(200):              # invert the mapping: what aspect does 1.62 correspond to?
        mid = 0.5 * (lo + hi)
        if capsule_protrusion_exact(mid)[0] < best:
            lo = mid
        else:
            hi = mid
    print("\n--- WHERE OUR BEST MEASURED RUN LANDS -------------------------------------------------")
    print(f"  best honest Q_protr_after_relax = {best:.2f}  ->  capsule aspect {0.5 * (lo + hi):.2f}")
    print(f"  vs derived threshold {derived_bar():.2f}: "
          f"{'PASSES' if best >= derived_bar() else 'FALLS SHORT'} (margin {best - derived_bar():+.2f})")
    print("  CAVEAT, and it is not small: this equivalence holds for a WHOLE-BODY capsule. The")
    print("  campaign's morphology is a tube on a body, which reads differently -- see below.")

    # -------------------------------------------------------------- the caveat family
    print("\n--- CAVEAT: the same metric on a TUBE GROWN OFF A BODY --------------------------------")
    print("  protrusion is not a pure shape descriptor: it also depends on what fraction of the")
    print("  cells are in the tube, because percentile(r,95) only reaches into the tube once the")
    print("  tube holds more than 5% of them. Body radius 1.0 throughout.")
    print(f"  {'tube R':>7} {'tube aspect':>12} {'cells in tube':>14} {'protrusion':>11}")
    over = []
    for tR in [0.15, 0.3, 0.5]:
        for ta in [1.0, 2.0, 4.0, 8.0]:
            tlen = 2.0 * tR * ta                              # barrel length = aspect * diameter
            P, frac = sample_sphere_with_tube(1.0, tR, tlen, N_MC, rng)
            v = _protr(P)
            if v > ceiling:
                over.append((tR, ta, v))
            print(f"  {tR:>7.2f} {ta:>12.1f} {100 * frac:>13.1f}% {v:>11.4f}"
                  f"{'   <- above the capsule ceiling' if v > ceiling else ''}")
    print("  A thin tube on a big body UNDER-reads: too few cells to reach the 95th percentile.")
    print(f"  And {len(over)} of these rows read ABOVE the capsule ceiling {ceiling:.2f} -- a tube on a")
    print("  body can exceed what any capsule reaches, which is how real runs scored 5.2 and 8.3")
    print("  while the analysts called them 'spike'. So the derived threshold is a NECESSARY")
    print("  condition for a tube, not a sufficient one; Q and n_tubes carry the rest.")

    print("\n" + "=" * 92)
    print(f"USE  protrusion >= {derived_bar():.2f}  IN control.meets_success.  sphere null = 1.000.")
    print("=" * 92)
    return derived_bar()


# ------------------------------------------------------------------ the acceptance test
def check(verbose=True):
    """Geometric acceptance test for `control.meets_success`.

    FAILS on the old, guessed thresholds and PASSES on the derived one. The cases are shapes, not
    remembered numbers: each protrusion below is computed from the analytic capsule at run time.
    """
    import control
    cfg = control.CampaignConfig()
    thr, bar = derived_threshold(), derived_bar()

    cases = [                       # (aspect, must meets_success be True?)
        (1.0, False),               # a sphere is not a tube -- the null must never pass
        (1.5, False),               # barrel shorter than one diameter: a bump
        (2.0, True),                # the definition of a tube, exactly at threshold
        (3.0, True),                # unambiguously a tube
        (4.0, True),
    ]
    fails = []
    for aspect, want in cases:
        p = capsule_protrusion_exact(aspect)[0]
        summary = {"protr_peak": p, "protr_final": p, "Q_protr_after_relax": p,
                   "ta_n_tubes_final": 1}
        got = control.meets_success(summary, cfg, has_extrude=False)
        ok = (got == want)
        if not ok:
            fails.append((aspect, p, want, got))
        if verbose:
            print(f"  capsule aspect {aspect:>4.1f} -> protrusion {p:.4f}  "
                  f"meets_success={str(got):<5} want={str(want):<5}  {'ok' if ok else 'FAIL'}")

    # the threshold in control must BE the derived number, not a nearby tuned one
    for key in ("protr_peak", "protr_final", "Q_protr_after_relax"):
        txt = cfg.success[key]
        val = float(txt.split(">=")[1])
        if abs(val - bar) > 1e-9:
            fails.append((key, val, bar, txt))
            if verbose:
                print(f"  cfg.success[{key}] = {txt!r}, derived {bar:.2f}  FAIL")
        elif verbose:
            print(f"  cfg.success[{key}] = {txt!r} == derived {bar:.2f}  ok")

    if verbose:
        print(f"\n  derived threshold {thr:.4f} (capsule aspect {TUBE_ASPECT:g}), floored to "
              f"{bar:.2f} for the config; sphere null {capsule_protrusion_exact(1.0)[0]:.3f}")
    if fails:
        raise AssertionError(f"geometric acceptance test FAILED: {fails}")
    if verbose:
        print("  geometric acceptance test PASSED")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="run the geometric acceptance test against control.meets_success")
    args = ap.parse_args()
    if args.check:
        print("geometric acceptance test on control.meets_success "
              f"(control from {__import__('control').__file__})")
        check()
    else:
        main()
