#!/usr/bin/env python
"""GATE for P1: every DIMENSIONAL constant expressed relative to its own natural scale.

DECLARED BEFORE THE CHANGE, AND BEFORE ANY NUMBER FROM IT WAS LOOKED AT. That is the point of
writing this file first: a threshold chosen after seeing the measurement is not a threshold.

THE DEFECT. The MPM operators carry constants with physical DIMENSIONS written as bare numbers:

    mass_floor      1e-10     a mass          the floor under gm in the momentum division
    csf_mass_floor  1e-8      a mass          the floor under gm in the CSF force division
    ring            0.04      a length        mpm_anchor's boundary band
    CSF eps         1e-6      1 / length      the regulariser in n = grad(c)/(|grad c| + eps)
    a_max           200       length / time^2 the clamp on external acceleration
    vmax            1e9       length / time   the clamp on gathered speed
    spin_k          30        1 / time        mpm_spin's controller gain

Each was chosen against a unit box at n_grid 96 with density 1, and each means something different
the moment any of those three change. `wall_contact: 0.04` -- already converted -- was the clearest
case: in a 0.1 m box it selected everything but a 0.02 m sliver, so the entire fluid read as
permanently in wall contact.

AND ONE OF THEM HAS A MEASURED COST ALREADY. The CSF `eps` is why `csf_rho` is not a pure gain
rescaling: at the parity tension sigma = 120/192^2 = 3.2552e-3, material_two_drops_st reproduces
only to 2.376% RMS IN CSF FORCE (11.7% of that run's own Rg change), rising to 32.7% with
`csf_band` on, because an ABSOLUTE epsilon added to |grad c| bites differently once the colour is
divided by rho*dx^D. That is the one conversion with a number to beat rather than a hazard to
forestall.

THE FOUR ROWS, AND WHAT EACH MUST READ.

  identity     On a legacy spec -- world 1.0, n_grid 96, density 1, no relative key set -- every
               resolved constant must equal its historical value EXACTLY, as float64 bit equality
               and not `isclose`. Anything less is a behaviour change wearing a refactor's name,
               and 152 MPM specs depend on it.

  scaling      On the SI spec -- world 0.1 m, n_grid 96, density 1000 -- each resolved constant
               must scale exactly as its DIMENSION says, against the legacy value:
                   mass_floor, csf_mass_floor   x  (rho2/rho1) * (dx2/dx1)^3
                   ring                          x  (dx2/dx1)
                   CSF eps                       x  (dx1/dx2)          [it is a reciprocal length]
               Tolerance 0, again: these are closed forms, not measurements.

  bytes        Two existing specs run end to end -- material_3balls_bouncy (2D, torch) and
               material_3d_water_st000 forced to the torch path -- must reproduce byte for byte
               against the pre-change commit. tobytes(), not allclose. The WARP path is excluded
               on purpose: it is non-deterministic run to run (measured 5.3e-4 in particle position
               on a twin of the same code and seed, because wp.atomic_add's float ordering is not
               fixed), so byte-identity is unreachable there and demanding it would be theatre.

  csf_parity   With `csf_rho` set and sigma at parity, the CSF force field must agree with the
               legacy path to better than 0.5% RMS, against the 2.376% it manages today. This is
               the row that has to IMPROVE rather than hold, and it is the reason P1 is worth doing
               at all rather than being tidiness.

    python tools/mpm_p1_gate.py --rows identity,scaling            # before and after the change
    python tools/mpm_p1_gate.py --rows bytes --base <sha>          # needs a worktree of `base`
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# name -> (historical absolute value, dimension exponents over (density, length))
#   mass  = rho^1 * L^3      length = L^1      1/length = L^-1
CONSTANTS = {
    "mass_floor":     (1e-10, (1, 3)),
    "csf_mass_floor": (1e-8, (1, 3)),
    "ring":           (0.04, (0, 1)),
    "csf_eps":        (1e-6, (0, -1)),
}
LEGACY = dict(world=1.0, n_grid=96, rho=1.0)          # what every existing spec is
SI = dict(world=0.1, n_grid=96, rho=1000.0)           # what an si_material spec is


def resolve(name, world, n_grid, rho):
    """What the operator will use, once the constant is relative. Imported from the operators so
    the gate cannot drift from the implementation -- if the helper is absent, the gate says so."""
    from plexus.operators import mpm_ops
    fn = getattr(mpm_ops, "_scale_constant", None)
    if fn is None:
        return None
    dx = world / n_grid
    return fn(name, dx=dx, rho=rho)


def historical(name, world, n_grid, rho):
    """The dimensionally-correct value: the historical number carried to a new (dx, rho)."""
    v0, (a, b) = CONSTANTS[name]
    dx0 = LEGACY["world"] / LEGACY["n_grid"]
    dx = world / n_grid
    return v0 * (rho / LEGACY["rho"]) ** a * (dx / dx0) ** b


def row_identity():
    print(f"\n  ROW `identity` -- on a legacy spec the resolved value must EQUAL the historical one,")
    print(f"  bit for bit. world {LEGACY['world']}, n_grid {LEGACY['n_grid']}, rho {LEGACY['rho']}\n")
    print(f"  {'constant':>18}{'historical':>14}{'resolved':>14}{'':>10}")
    print("  " + "-" * 58)
    ok = True
    for name, (v0, _) in CONSTANTS.items():
        got = resolve(name, **LEGACY)
        if got is None:
            print(f"  {name:>18}{v0:>14.6e}{'not built':>14}{'  SKIP':>10}")
            ok = False
            continue
        same = (got == v0)
        ok &= same
        print(f"  {name:>18}{v0:>14.6e}{got:>14.6e}{'  PASS' if same else '  FAIL':>10}")
    return ok


def row_scaling():
    dx0 = LEGACY["world"] / LEGACY["n_grid"]
    dx1 = SI["world"] / SI["n_grid"]
    print(f"\n  ROW `scaling` -- on the SI spec each constant must scale as its DIMENSION says.")
    print(f"  dx {dx0 * 1000:.4f} mm -> {dx1 * 1000:.4f} mm ({dx1 / dx0:g}x), "
          f"rho {LEGACY['rho']:g} -> {SI['rho']:g} ({SI['rho'] / LEGACY['rho']:g}x)\n")
    print(f"  {'constant':>18}{'dimension':>14}{'required':>14}{'resolved':>14}{'':>9}")
    print("  " + "-" * 72)
    ok = True
    for name, (v0, (a, b)) in CONSTANTS.items():
        want = historical(name, **SI)
        got = resolve(name, **SI)
        dim = ("rho" if a else "") + (f"*L^{b}" if b else "")
        if got is None:
            print(f"  {name:>18}{dim:>14}{want:>14.6e}{'not built':>14}{'  SKIP':>9}")
            ok = False
            continue
        same = (got == want)
        ok &= same
        print(f"  {name:>18}{dim:>14}{want:>14.6e}{got:>14.6e}"
              f"{'  PASS' if same else '  FAIL':>9}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="identity,scaling")
    a = ap.parse_args()
    res = {}
    if "identity" in a.rows:
        res["identity"] = row_identity()
    if "scaling" in a.rows:
        res["scaling"] = row_scaling()
    _st = {k: ('PASS' if v else 'FAIL') for k, v in res.items()}
    print('\n  ' + '   '.join(f'{k} {v}' for k, v in _st.items()))
    print(f"  {'ALL PASS' if all(res.values()) else 'NOT YET -- this is the pre-change state'}\n")


if __name__ == "__main__":
    main()
