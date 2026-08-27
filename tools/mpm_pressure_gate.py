#!/usr/bin/env python
"""GATE: does the liquid carry the hydrostatic pressure p = rho g d, in pascals?

WHY THIS NEEDS ITS OWN RUN. Pressure is not a recorded quantity. It is not in `trajectory.npz` and
it is not a grid channel -- it lives in the deformation gradient, p = K(1 - J) with J = det(F), and
`F` is per-particle state the recorder does not store. A first attempt at this test tried to
reconstruct J from the shape of the settled column instead, and that is CIRCULAR: inverting the
linear hydrostatic profile to get a pressure returns rho*g*d by construction, whatever the run did.
It measured the arithmetic, not the simulation, and its numbers are discarded.

So the run happens here, with a hook that reads `p.F` live.

THE CLOSED FORM, and it is the most direct statement that a run is in SI at all:

    p(d) = rho g d          d the depth below the free surface, p in PASCALS

Nothing is fitted. rho comes from the spec, g comes from the spec, d is measured, and the pressure
is K(1 - det F) with K the declared bulk modulus. At rho 1000, g 9.81 and d 20 mm the answer is
196.2 Pa and there is nowhere to hide.

WHAT MAKES IT FAIL, so a pass means something:
  * a settled column is required -- the profile is hydrostatic only at rest, and a sloshing one
    carries dynamic pressure the formula knows nothing about. The tool reports the drift and
    refuses to grade an unsettled run.
  * the TOP of the column is not hydrostatic either: the free surface is one cell thick and the
    shallowest bin sits inside it, so bins shallower than 2*dx are reported but not graded.
  * K must be large enough that 1 - J is resolvable in float32. At K = 1e5 and d = 20 mm,
    1 - J = 1.96e-3, which is 16,000 float32 epsilons -- fine. At water's own 2.2 GPa in a 0.1 m
    box it would be 1.5 epsilons, and this test would be measuring quantisation noise.

    python tools/mpm_pressure_gate.py --spec si_gate --frames 1200 --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="si_gate")
    ap.add_argument("--type", default="si_material")
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--particles", type=int, default=250000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tol", type=float, default=5.0)
    a = ap.parse_args()

    import numpy as np
    import torch
    import yaml

    import plexus.operators  # noqa: F401
    import plexus.operators.mpm_warp  # noqa: F401
    from plexus import engine as E
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
    from plexus.schema import load

    s = yaml.safe_load(open(os.path.join(ROOT, "config", a.type, a.spec + ".yaml")))
    s["general"]["n_frames"] = int(a.frames)
    s["general"]["save_data"] = False
    s["sets"]["mpm_particle"]["per_parent"] = int(a.particles)
    rho = float(s["sets"]["mpm_particle"]["density"])
    K = float(list(s["sets"]["cell"]["types"].values())[0]["bulk_modulus"])
    g = next(float(o["g"]) for o in s["operators"] if o.get("op") == "gravity")
    W = float(s["general"]["world"][1])
    dx = W / int(list(s["fields"].values())[0]["n_grid"])
    up = int((s.get("plotting") or {}).get("up_axis", 1))

    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f); f.close()
    CFL(f.name)
    sim = load(f.name); os.unlink(f.name)

    # keep the last N frames of (height, pressure) so the profile can be averaged over the tail
    keep, tail = [], max(8, a.frames // 20)

    def on_frame(H, t):
        if t < a.frames - tail:
            return
        p = H.level("mpm_particle")
        J = torch.linalg.det(p.F.detach())
        y = p.get("pos")[:, up].detach()
        keep.append((y.cpu().numpy().copy(), (K * (1.0 - J)).cpu().numpy().copy()))

    E.run(sim, out_path=None, device=a.device, progress=False, on_frame=on_frame)

    Y = np.concatenate([k[0] for k in keep])
    P = np.concatenate([k[1] for k in keep])
    top = float(np.quantile(Y, 0.999))
    lvl = [float(np.quantile(k[0], 0.999)) for k in keep]
    drift = abs(lvl[-1] - lvl[0])

    print(f"\n  {a.spec}: rho {rho:g} kg/m^3, g {g:g} m/s^2, K {K:.3g} Pa, dx {dx * 1000:.3f} mm")
    print(f"  free surface {top * 1000:.2f} mm, drift over the graded window {drift * 1000:.3f} mm"
          f"  ({'settled' if drift < 2 * dx else 'NOT SETTLED -- not gradeable'})\n")
    print(f"  {'depth (mm)':>12}{'measured p (Pa)':>18}{'rho*g*d (Pa)':>16}{'error':>10}{'':>8}")
    print("  " + "-" * 66)
    ok = drift < 2 * dx
    for d_mm in (5, 10, 15, 20, 25):
        d = d_mm / 1000.0
        band = (Y > top - d - dx) & (Y < top - d + dx)      # a one-cell band at this depth
        if band.sum() < 200:
            continue
        pm = float(np.median(P[band]))
        cf = rho * g * d
        e = abs(pm / cf - 1) * 100
        graded = d >= 2 * dx
        ok &= (e <= a.tol) if graded else True
        mark = ("  PASS" if e <= a.tol else "  FAIL") if graded \
            else "  (inside the free surface, not graded)"
        print(f"  {d_mm:>12}{pm:>18.1f}{cf:>16.1f}{e:>9.2f}%{mark}")
    print(f"\n  {'ALL PASS' if ok else 'FAILURES ABOVE'}  (tol {a.tol:g}%)\n")


if __name__ == "__main__":
    main()
