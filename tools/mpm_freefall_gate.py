#!/usr/bin/env python
"""GATE: does a body in a unit-ful MPM fall the way Galileo says it does?

THE POINT OF A 100 m SCENE. Every closed form in this file is one anybody can check by hand, and at
this scale the answer takes SECONDS rather than the hundredths a 0.1 m box gives -- 3.86 s to fall
73 m, against 0.078 s for a 30 mm drop in a 100 mm box. That matters twice: a human can see whether
the movie runs at the right speed, and the measurement has three digits of dynamic range instead of
one frame's worth.

It also removes the one compromise this branch has had to declare everywhere else. Water's own bulk
modulus, 2.2 GPa, is unusable at 0.1 m -- the hydrostatic strain there is 1.78e-07, which is 1.5
FLOAT32 EPSILONS, and it costs 59,330 substeps a frame. At 100 m the SAME water is 4.15e-04, i.e.
3,485 epsilons, and 75 substeps, because hydrostatic pressure scales with depth: 883 kPa against
392 Pa. So this spec declares `bulk_modulus: 2.2e9` and nothing about it is weakly compressible.

THREE CLOSED FORMS:
    time to ground     t = sqrt(2h/g),         h the release height above the floor
    impact speed       v = sqrt(2gh) = g*t
    acceleration       d2y/dt2 = -g, fitted over the flight, which must be g and not 0.977*g
                       (that 2.3% is what a Stokes `drag` costs, and this spec sets drag 0)

WHAT WOULD MAKE THIS FAIL, so a pass means something: a dx that did not follow the world box (the
grid would be 1 unit wide over a 100 m box); a body force applied inside a clamped mass division; a
substep that does not tile general.dt; or a particle mass not tied to the declared density.

    python tools/mpm_freefall_gate.py --spec si_freefall
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="si_freefall")
    ap.add_argument("--type", default="si_material")
    ap.add_argument("--data", default=None)
    a = ap.parse_args()

    from plexus.paths import graphs_data_path
    d = a.data or graphs_data_path(a.type, a.spec)
    s = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
    g = next(float(o["g"]) for o in s["operators"] if o["op"] == "gravity")
    W = float(s["general"]["world"][1])
    ng = int(list(s["fields"].values())[0]["n_grid"])
    dx = W / ng
    dt = float(s["general"]["dt"])
    up = int((s.get("plotting") or {}).get("up_axis", 1))

    P = np.load(os.path.join(d, "trajectory.npz"))["mpm_particle__pos"]
    T = P.shape[0]
    lo = P[:, :, up].min(axis=1)                     # the body's LOWEST point, frame by frame
    com = P[:, :, up].mean(axis=1)
    floor = 2.0 * dx                                 # the gather's own position clamp

    h = float(lo[0] - floor)
    t_cf = math.sqrt(2.0 * h / g)
    # first frame whose lowest point has reached the floor, refined by linear interpolation
    hit = int(np.argmax(lo <= floor + 1e-9))
    if hit > 0:
        y0, y1 = lo[hit - 1], lo[hit]
        frac = (y0 - floor) / max(y0 - y1, 1e-12)
        t_hit = (hit - 1 + frac) * dt
    else:
        t_hit = float("nan")
    # acceleration from the centre of mass over the flight, before any part of it lands
    n = max(4, int(0.8 * t_cf / dt))
    t = np.arange(2, n) * dt
    c = np.linalg.lstsq(np.vstack([np.ones_like(t), t, t * t]).T, com[2:n], rcond=None)[0]
    acc = -2 * c[2]
    v_cf = math.sqrt(2 * g * h)
    v_meas = -(com[n - 1] - com[n - 3]) / (2 * dt)

    print(f"\n  {a.spec}: a {(P.shape[1] * float(s['sets']['mpm_particle']['particle_mass']) / float(s['sets']['mpm_particle']['density'])) ** (1/3):.1f} m cube of water "
          f"in a {W:.0f} m box, released {h:.2f} m above the floor\n")
    print(f"  {'quantity':<26}{'measured':>14}{'closed form':>14}{'error':>10}{'':>7}")
    print("  " + "-" * 72)
    rows = [("time to ground (s)", t_hit, t_cf, 2.0),
            ("acceleration (m/s^2)", acc, g, 1.0),
            ("speed at 0.8t (m/s)", v_meas, g * 0.8 * t_cf, 3.0)]
    ok = True
    for lab, m, cf, tol in rows:
        e = abs(m / cf - 1) * 100 if cf else float("nan")
        p = e <= tol
        ok &= p
        print(f"  {lab:<26}{m:>14.5g}{cf:>14.5g}{e:>9.3f}%{'  PASS' if p else '  FAIL':>7}")
    print(f"\n  {'ALL PASS' if ok else 'FAILURES ABOVE'}   "
          f"(the movie is {T - 1} frames x {dt} s = {(T - 1) * dt:.1f} s of world, real time)\n")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "src"))
    main()
