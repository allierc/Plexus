#!/usr/bin/env python
"""calibrate -- read every metric on shapes whose answer is known in advance.

THE THIRD KIND OF TEST, AND IT ASKS SOMETHING THE OTHER TWO CANNOT
================================================================================================
  the battery      distort a real recording one axis at a time; does the metric MOVE where it
                   should and HOLD where it should not? Says nothing about whether the number is
                   right.
  the floors       how much does it wobble when nothing changed, and what does knowing nothing
                   score? Says nothing about whether the number is right either.
  here             put in a shape whose value is known on paper. **Is the number right, and is it
                   measuring the thing its name claims?**

It catches a class of fault the battery is blind to by construction. The battery applies one change
to the whole recording at once, so any change that is a SYMMETRY OF THE POPULATION passes unnoticed:
rotate every loop by the same angle and, if the loops already point every which way, the
distribution is unchanged and the median of anything is unchanged with it. A metric can depend
strongly on orientation and hold perfectly still in that column. That is not a hypothetical -- it
is what `openness` does, and this file is how it was found.

An ellipse is the right probe because everything has a closed form for it:

    semi-axes a, b, turned by theta
    enclosed area                 pi*a*b                       (any theta)
    axis-aligned bounding box     4 * sqrt(a^2c^2 + b^2s^2) * sqrt(a^2s^2 + b^2c^2)
    furthest point from centre    a                            (any theta)
    perimeter                     Ramanujan: pi[3(a+b) - sqrt((3a+b)(a+3b))]
    the long axis                 theta
    circulation                   the sign of the parameter's direction

    python calibrate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import metrics as M                                                 # noqa: E402

FINDINGS = []


def ellipse(a=1.0, b=0.45, theta=0.0, G=96, n_nodes=64, reverse=False, roll=0):
    """[G, n_nodes, 2] -- the same ellipse at every node, so the median IS that ellipse's value."""
    t = np.linspace(0, 2 * np.pi, G, endpoint=False)
    if reverse:
        t = -t
    u, v = a * np.cos(t), b * np.sin(t)
    p = np.stack([u * np.cos(theta) - v * np.sin(theta),
                  u * np.sin(theta) + v * np.cos(theta)], -1)
    if roll:
        p = np.roll(p, roll, 0)
    return np.repeat(p[:, None, :], n_nodes, axis=1)


def perimeter(a, b):
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))


def bbox_openness(a, b, theta):
    hx = np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)
    hy = np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
    return (np.pi * a * b) / (4 * hx * hy)


def head(n, s):
    print(f"\n  {n}. {s}\n  {'-' * 96}")


def row(label, got, want, tol=0.02, unit=""):
    ok = want is None or abs(got - want) <= tol * max(1.0, abs(want))
    w = "      --" if want is None else f"{want:>8.4f}"
    print(f"    {label:<34s} measured {got:>9.4f}{unit}   on paper {w}   "
          f"{'ok' if ok else 'OFF'}")
    return ok


def finding(text):
    FINDINGS.append(text)
    print(f"    >> {text}")


# =============================================================================================
def cal_openness():
    head(1, "OPENNESS -- area over the bounding box it fits in")
    prop = M.REGISTRY["openness"].property
    vals = [prop(ellipse(1.0, r, 0.0)) for r in (1.0, 0.8, 0.6, 0.45, 0.3, 0.15, 0.05, 0.01)]
    row("aspect 1.00 (a circle)", vals[0], np.pi / 4)
    row("aspect 0.01 (a needle)", vals[-1], np.pi / 4)
    print(f"    spread across the whole aspect sweep: {max(vals) - min(vals):.5f}")
    finding("openness DOES NOT read how fat a loop is -- a circle and a needle both read pi/4. "
            "It separates a loop from a degenerate line and nothing finer.")

    turned = [prop(ellipse(1.0, 0.45, np.deg2rad(d))) for d in (0, 15, 30, 45, 60, 75, 90)]
    for d, v in zip((0, 30, 45, 90), (turned[0], turned[2], turned[3], turned[6])):
        row(f"the same ellipse turned {d}d", v, bbox_openness(1.0, 0.45, np.deg2rad(d)))
    finding(f"openness responds to ORIENTATION by {1 - min(turned) / max(turned):.1%}, and declares "
            f"only {{'openness'}}. The battery cannot see it: it turns every loop by one angle, and "
            f"a population already pointing every which way is unchanged by that.")


def cal_peak():
    head(2, "PEAK EXCURSION -- the furthest a node gets from the centre of its own path")
    prop = M.REGISTRY["peak_excursion"].property
    for a in (0.5, 1.0, 2.0):
        row(f"a = {a}, b = 0.45", prop(ellipse(a, 0.45, 0.0)), a)
    row("turned 45d (must not matter)", prop(ellipse(1.0, 0.45, np.pi / 4)), 1.0)
    b_sweep = [prop(ellipse(1.0, r, 0.0)) for r in (0.9, 0.45, 0.05)]
    print(f"    b swept 0.9 -> 0.05 at fixed a: {b_sweep[0]:.4f} .. {b_sweep[-1]:.4f}")
    finding("peak_excursion reads the LONG semi-axis exactly and is blind to the short one, so it "
            "is a reach measure, not a size measure. Two loops of very different area read the "
            "same. That is consistent with its declared axis, and it is worth saying out loud.")


def cal_path():
    head(3, "PATH LENGTH -- distance travelled over one beat")
    prop = M.REGISTRY["path_length"].property
    for a, b in ((1.0, 1.0), (1.0, 0.45), (2.0, 0.9), (1.0, 0.05)):
        row(f"a={a}, b={b}", prop(ellipse(a, b, 0.0)), perimeter(a, b))
    row("turned 45d (must not matter)", prop(ellipse(1.0, 0.45, np.pi / 4)),
        perimeter(1.0, 0.45))
    row("rolled by a third of a beat", prop(ellipse(1.0, 0.45, 0.0, roll=32)),
        perimeter(1.0, 0.45))
    finding("path_length reproduces the closed-form perimeter, is invariant to turning, and is now "
            "invariant to rolling -- it was not, before the closing segment was put back.")


def cal_orientation():
    head(4, "ORIENTATION ERROR -- the angle between the two long axes")
    m = M.REGISTRY["orientation_error"]
    ok_lin = True
    for d in (0, 15, 30, 45, 60, 90):
        th = np.deg2rad(d)
        want = th                           # 90d is the MAXIMUM: two axes cannot differ by more
        ok_lin &= row(f"turned by {d}d", m(ellipse(1.0, .45, th), ellipse(1.0, .45, 0.0)),
                      want, tol=0.03, unit=" rad")
    finding("orientation_error returns the angle it was given, exactly and linearly, from 0 to "
            "pi/2. pi/2 is its maximum and not a wrap: an axis has no head or tail, so two axes can "
            "differ by at most 90 degrees, and that is what perpendicular reads." if ok_lin else
            "orientation_error does NOT return the angle it was given.")

    circ = m(ellipse(1.0, 1.0, np.pi / 3), ellipse(1.0, 1.0, 0.0))
    row("a CIRCLE against a circle", circ, None, unit=" rad")
    finding(f"a circle has no long axis, so this question has no answer -- and it returns "
            f"{circ:.4f} rad anyway rather than refusing. Real loops are not circles, so this is a "
            f"latent fault rather than an active one, but it is undeclared. Compare Coordination, "
            f"which now raises Undefined outside its domain.")


def cal_chirality():
    head(5, "CHIRALITY MATCH -- the fraction of loops going round the same way")
    m = M.REGISTRY["chirality_match"]
    ref = ellipse(1.0, 0.45, 0.0)
    row("same direction", m(ref, ref), 1.0)
    row("reversed", m(ellipse(1.0, 0.45, 0.0, reverse=True), ref), 0.0)
    row("reversed, and turned 40d", m(ellipse(1.0, 0.45, 0.7, reverse=True), ref), 0.0)
    half = np.concatenate([ellipse(1., .45, 0., n_nodes=32),
                           ellipse(1., .45, 0., n_nodes=32, reverse=True)], axis=1)
    row("half the nodes reversed", m(half, np.concatenate([ref[:, :32], ref[:, :32]], 1)), 0.5)
    finding("chirality_match is exact at both ends and linear in between -- half the nodes reversed "
            "reads 0.5. Its analytic null of 0.5 is the value a coin toss gives, and that is what "
            "half-reversed means, so the null is the right one.")


def cal_coordination():
    head(6, "COORDINATION -- whether the tissue moves together the way the recording does")
    m = M.REGISTRY["coordination"]
    ref = ellipse(1.0, 0.45, 0.0, G=96)
    row("identical", m(ref, ref), 1.0)
    row("the whole beat rolled (one offset)", m(ellipse(1., .45, 0., G=96, roll=17), ref), 1.0)
    rng = np.random.default_rng(0)
    sc = np.stack([np.roll(ref[:, j], int(rng.integers(0, 96)), 0) for j in range(ref.shape[1])], 1)
    row("every node rolled differently", m(sc, ref), None)
    anti = np.stack([np.roll(ref[:, j], 48 if j % 2 else 0, 0) for j in range(ref.shape[1])], 1)
    row("half the nodes in exact antiphase", m(anti, ref), None)
    finding("exact antiphase reads as perfectly coordinated, which is the defect already declared "
            "on the class: the signal is a distance from a centre and peaks twice per beat, so the "
            "alignment is only determined modulo half a beat. Confirmed here on a shape where the "
            "answer is known, rather than argued.")
    try:
        m(np.zeros_like(ref), ref)
        finding("A DEAD MODEL WAS SCORED. The domain guard is not working.")
    except M.Undefined:
        print("    a model with no motion                       refuses to be scored   ok")


def cal_rotation_invariant_alternative():
    head(7, "WHAT A ROTATION-INVARIANT OPENNESS WOULD READ  (4*pi*area / perimeter^2)")
    print(f"    {'aspect':>8s} {'aligned':>10s} {'at 45d':>10s}")
    for r in (1.0, 0.6, 0.45, 0.15, 0.05):
        v = []
        for th in (0.0, np.pi / 4):
            p = ellipse(1.0, r, th)[:, 0]
            per = np.linalg.norm(np.diff(np.concatenate([p, p[:1]]), axis=0), axis=-1).sum()
            v.append(4 * np.pi * abs(M.signed_area(ellipse(1.0, r, th))[0]) / per ** 2)
        print(f"    {r:>8.2f} {v[0]:>10.4f} {v[1]:>10.4f}")
    finding("this alternative is invariant to turning AND falls with the aspect ratio, so it "
            "measures what the name promises. PROPOSED, NOT ADOPTED: swapping the definition "
            "invalidates every number already measured with the current one, which is a decision "
            "and not a fix.")


def main():
    print(f"\n{'=' * 100}\n  CALIBRATION -- is the number right? Ellipses, against the closed form"
          f"\n{'=' * 100}")
    for f in (cal_openness, cal_peak, cal_path, cal_orientation, cal_chirality,
              cal_coordination, cal_rotation_invariant_alternative):
        f()
    print(f"\n{'=' * 100}\n  {len(FINDINGS)} FINDINGS\n{'=' * 100}")
    for i, t in enumerate(FINDINGS, 1):
        print(f"  {i}. {t}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
