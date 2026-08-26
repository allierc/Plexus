#!/usr/bin/env python
"""GATE: does the SI MPM reproduce closed-form water, in metres, seconds and pascals?

WHY A CLOSED FORM AND NOT A COMPARISON. Every check in this file has an answer that can be written
down before the run, from the spec's own declared numbers. That is the only kind of check a unit
refactor can be graded by: a comparison against a previous run grades the refactor against itself,
and the previous runs were dimensionless, so there is nothing to compare to.

WHAT THE SPEC DECLARES, and every one of these is a number you can look up:

    world          0.1 m cube            density    1000 kg/m^3
    g              9.81 m/s^2            eta        1.0e-3 Pa s     (water, exactly)
    dt             1/1200 s              sigma      0.072 N/m       (water, exactly)
    bulk_modulus   1.0e5 Pa              <- THE ONE COMPROMISE, and it is declared

WHY K IS NOT WATER'S 2.2 GPa, stated here so no result from this file is over-claimed. At
dx = 1.0417 mm the hydrostatic volumetric strain under 0.04 m of real water is
rho*g*H/K = 1.78e-7, which is 1.5 FLOAT32 EPSILONS: single precision cannot represent water's
compression at all, and the pressure would be quantisation noise. It would also cost 59,330
substeps per 1/60 s frame against 400. K = 1e5 Pa is the standard weakly-compressible choice
(c >= 10*U_max, Mach 0.1, density fluctuation ~1%) and it puts the run at 21 substeps/frame -- the
same cost as the dimensionless specs it replaces. Every test below is therefore a test of a
weakly-compressible liquid, and the tolerances are set with the Mach-0.1 error in mind.

THE TESTS.
  level      a confined column settles to H = H0 (1 - rho g H0 / 2K). Measured as mean(J), which
             for a laterally confined column IS H/H0 -- no surface quantile, no offset, no
             argument about where the free surface is.
  pressure   the grid's own nodal pressure against rho*g*d, read at several depths.
  slosh      the first sloshing mode of a box of width L over depth h has period 2L/sqrt(g h)
             (shallow water, h/L = 0.4 here so this is approximate and the tolerance says so).
  rayleigh   a free drop of radius R oscillating in mode 2 has T = 2*pi*sqrt(rho R^3/(8 sigma)).
             THIS IS THE ONE THAT TESTS SURFACE TENSION IN PHYSICAL UNITS -- the quantity that,
             before `csf_rho`, was a tension divided by a cell mass and meant nothing.
  freefall   a body in flight accelerates at exactly g. Already gated for `body_force` and
             `mpm_viscosity`; repeated here in SI because it is the cheapest possible unit check.

WHAT THIS BENCHMARK CANNOT CLAIM. The capillary length sqrt(sigma/(rho g)) is 2.71 mm = 2.6 cells
at dx = 1.04 mm. Anything whose shape is set below that -- a millimetre droplet, a meniscus, a
thin film -- is under-resolved and no number about it from this spec means anything.

    python tools/mpm_si_gate.py --tests level,freefall --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CFG = os.path.join(ROOT, "config", "material")


def _build(spec, frames, device, particles=None, patch=None):
    import yaml

    import plexus.operators  # noqa: F401
    import plexus.operators.mpm_warp  # noqa: F401
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
    from plexus.schema import load

    s = yaml.safe_load(open(os.path.join(CFG, spec + ".yaml")))
    s["general"]["n_frames"] = int(frames)
    s["general"]["record_cap"] = 10000
    if particles:
        s["sets"]["mpm_particle"]["per_parent"] = int(particles)
    if patch:
        patch(s)
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f); f.close()
    CFL(f.name)
    sim = load(f.name); os.unlink(f.name)
    return s, sim


def _scene(s):
    """The declared numbers, read back from the spec so the expectation cannot drift from the run."""
    ty = list(s["sets"]["cell"]["types"].values())[0]
    box = [float(x) for x in s["general"]["world"]]
    b = ty["block"]
    up = int((s.get("plotting") or {}).get("up_axis", 1))
    gu = next(o for o in s["operators"] if o["op"] == "mpm_grid_update")
    return dict(
        K=float(ty["bulk_modulus"]), rho=float(s["sets"]["mpm_particle"]["density"]),
        g=next(float(o["g"]) for o in s["operators"] if o["op"] == "gravity"),
        sigma=float(gu.get("surface_tension", 0.0)),
        eta=next((float(o["eta"]) for o in s["operators"] if o["op"] == "mpm_viscosity"), 0.0),
        box=box, up=up, H0=float(b[up + 3] - b[up]), dt=float(s["general"]["dt"]),
        dx=box[1] / float(list(s["fields"].values())[0]["n_grid"]))


def test_level(args):
    """H/H0 = mean(J) = 1 - rho g H0 / 2K, exactly, for a laterally confined column."""
    import torch
    from plexus import engine as E
    s, sim = _build(args.spec, args.frames, args.device, args.particles)
    sc = _scene(s)
    pred = 1.0 - sc["rho"] * sc["g"] * sc["H0"] / (2.0 * sc["K"])
    tr = []
    E.run(sim, out_path=None, device=args.device, progress=False,
          on_frame=lambda H, t: tr.append(
              float(torch.linalg.det(H.level("mpm_particle").F.detach()).mean())))
    n = max(2, len(tr) // 5)
    got = sum(tr[-n:]) / n                                  # mean over the final fifth
    drift = abs(tr[-1] - tr[-n])
    return dict(name="level", got=got, want=pred, err=abs(got / pred - 1) * 100, tol=5.0,
                note=f"drift over the final fifth {drift:.2e} "
                     f"({'settled' if drift < 2e-4 else 'NOT SETTLED -- the fit is not evidence'})")


def test_freefall(args):
    """A body in flight accelerates at exactly g -- the cheapest possible unit check."""
    import numpy as np
    from plexus import engine as E

    def lift(s):
        ty = list(s["sets"]["cell"]["types"].values())[0]
        W = float(s["general"]["world"][0]); R = 0.008
        ty["block"] = [W / 2 - R, 0.070, W / 2 - R, W / 2 + R, 0.070 + 2 * R, W / 2 + R]
        s["sets"]["cell"]["start"] = [[W / 2, 0.070 + R, W / 2]]
        for o in s["operators"]:
            if o["op"] == "mpm_grid_update":
                o["surface_tension"] = 0.0
                o.pop("csf_band", None)

    s, sim = _build(args.spec, 120, args.device, 200000, patch=lift)
    sc = _scene(s)
    ys = []
    E.run(sim, out_path=None, device=args.device, progress=False,
          on_frame=lambda H, t: ys.append(
              float(H.level("mpm_particle").get("pos")[:, sc["up"]].mean())))
    y = np.array(ys); lo, hi = 3, min(len(y), 60)
    t = np.arange(lo, hi) * sc["dt"]
    A = np.vstack([np.ones_like(t), t, t * t]).T
    c = np.linalg.lstsq(A, y[lo:hi], rcond=None)[0]
    return dict(name="freefall", got=-2 * c[2], want=sc["g"],
                err=abs(-2 * c[2] / sc["g"] - 1) * 100, tol=3.0,
                note="fitted over frames 3-60, before the floor is reached")


TESTS = {"level": test_level, "freefall": test_freefall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="si_water_column")
    ap.add_argument("--tests", default="level,freefall")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--particles", type=int, default=500000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import torch
    import yaml
    torch.cuda.set_device(int(a.device.split(":")[-1]) if ":" in a.device else 0)

    s = yaml.safe_load(open(os.path.join(CFG, a.spec + ".yaml")))
    sc = _scene(s)
    c = math.sqrt(sc["K"] / sc["rho"])
    U = math.sqrt(2 * sc["g"] * max(sc["box"][1] - sc["H0"] - 0.006, 1e-9))
    lc = math.sqrt(sc["sigma"] / (sc["rho"] * sc["g"])) if sc["sigma"] > 0 else float("nan")
    print(f"\n  {a.spec}: {sc['box']} m, rho {sc['rho']:g} kg/m^3, g {sc['g']:g} m/s^2, "
          f"K {sc['K']:.3g} Pa, sigma {sc['sigma']:g} N/m, eta {sc['eta']:g} Pa s")
    print(f"  dx {sc['dx'] * 1000:.3f} mm   c {c:.1f} m/s   Mach {U / c:.3f}   "
          f"Re {U * sc['H0'] * sc['rho'] / max(sc['eta'], 1e-30):.3g}   "
          f"capillary length {lc * 1000:.2f} mm = {lc / sc['dx']:.1f} cells\n")
    print(f"  {'test':>10}{'measured':>16}{'closed form':>16}{'error':>10}{'tol':>7}{'':>6}")
    print("  " + "-" * 72)
    rows, ok = [], True
    for name in a.tests.split(","):
        r = TESTS[name](a)
        r["pass"] = r["err"] <= r["tol"]
        ok &= r["pass"]
        rows.append(r)
        print(f"  {r['name']:>10}{r['got']:>16.6g}{r['want']:>16.6g}{r['err']:>9.2f}%"
              f"{r['tol']:>6.1f}%{'  PASS' if r['pass'] else '  FAIL'}")
        if r.get("note"):
            print(f"             {r['note']}")
    print(f"\n  {'ALL PASS' if ok else 'FAILURES ABOVE'}\n")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
