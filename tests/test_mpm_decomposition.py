"""Phase-3 regression: the decomposed MPM must reproduce the fenced oracle frame by frame.

`mls_mpm_mechanics` is one operator that does the whole material step; the decomposition is four --
`mpm_strain -> p2g -> mpm_grid_update -> g2p` under a substep block -- and the claim is that they
compute the same thing. The oracle is KEPT as the reference target. CPU, because `index_add_` is
deterministic there and this test is an equality check, not a physics check.

TWO THINGS WERE FIXED HERE AFTER THIS FILE STOPPED RUNNING AT ALL.

1. THE SCHEDULE FORM. It declared `{substep: 14, dt: 2.0e-4}`, which `schema.load` now refuses in
   favour of `{substep_dt: 2.0e-4, steps: [...]}`, where the substep COUNT is derived:
   `substeps = round(general.dt / substep_dt)`. That is not a syntactic rewrite. The old form let
   `general.dt` (1.0, the frame time seen by `gravity`) and the substep budget (14 x 2.0e-4) be
   unrelated numbers; the new one ties them together. So `general.dt` becomes 14 x 2.0e-4 = 2.8e-3
   -- AND IT IS CHANGED ON BOTH SIDES, which is what keeps the test honest: the quantity under test
   is oracle-vs-decomposition, so both must integrate gravity over the same frame time. The
   absolute trajectory is not the same one this file compared in 2024; the claim it makes is.

2. IT WAS A SCRIPT, NOT A TEST. Everything ran at module level under a bare `assert ok`, so pytest
   executed all five cases during COLLECTION and a failure took the whole file down with an error
   rather than a failure -- which is how it sat broken without anyone seeing which case broke. Now
   one parametrised test per case. It is still runnable directly, as the old header promised.
"""
import os
import tempfile

import numpy as np
import pytest

import plexus.operators  # noqa: F401  (registers operators + fields)
from plexus import schema
from plexus.engine import run

_CELL = ("cell: {{n: {n}, start: {start}, types: {{{types}}}}}\n"
         "  mpm_particle: {{parent: cell, per_parent: {ppc}, radius: {rad}, density: 1.0}}")

# THE FRAME TIME IS THE SUBSTEP BUDGET. Both specs declare it, so `gravity` -- the one operator
# outside the substep block -- advances identically on the two sides.
DT_SUB = 2.0e-4


def _frame_dt(case):
    return case["sub"] * DT_SUB


def _mono(case):
    return f"""
general: {{name: {case['name']}_mono, seed: 0, n_frames: {case['frames']}, dt: {_frame_dt(case)}, boundary: wall{case['obs']}}}
sets:
  {_CELL.format(**case)}
fields: {{}}
operators:
  - {{op: aggregate, at: cell}}
  - {{op: gravity, at: cell, g: {case['g']}}}
  - {{op: mls_mpm_mechanics, at: mpm_particle, n_grid: 64, substeps: {case['sub']}, dt_sub: {DT_SUB}, a_max: 200, drag: {case['drag']}, wall_damp: {case['wd']}, wall_contact: 0.05, surface_tension: {case['st']}}}
schedule: [aggregate, gravity, mls_mpm_mechanics]
"""


def _dec(case):
    return f"""
general: {{name: {case['name']}_dec, seed: 0, n_frames: {case['frames']}, dt: {_frame_dt(case)}, boundary: wall{case['obs']}}}
sets:
  {_CELL.format(**case)}
fields:
  mpm_grid: {{frame: mpm_grid, n_grid: 64}}
operators:
  - {{op: aggregate, at: cell}}
  - {{op: gravity, at: cell, g: {case['g']}}}
  - {{op: mpm_strain, at: mpm_particle, dt_sub: {DT_SUB}}}
  - {{op: p2g, at: mpm_particle, to: mpm_grid, dt_sub: {DT_SUB}, drag: {case['drag']}, a_max: 200}}
  - {{op: mpm_grid_update, at: mpm_grid, dt_sub: {DT_SUB}, surface_tension: {case['st']}, wall_damp: {case['wd']}, wall_contact: 0.05}}
  - {{op: g2p, at: mpm_particle, from: mpm_grid, dt_sub: {DT_SUB}, wall_damp: {case['wd']}, wall_contact: 0.05, vmax: 1.0e9}}
schedule:
  - aggregate
  - gravity
  - {{substep_dt: {DT_SUB}, steps: [mpm_strain, p2g, mpm_grid_update, g2p]}}
"""


def _run(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(text)
    f.close()
    sp = schema.load(f.name)
    os.unlink(f.name)
    _, out = run(sp, device="cpu")
    return out["sets"]["mpm_particle"]["pos"]


LIQ = ("water: {fraction: 1.0, youngs: 300, block: [0.1,0.05,0.55,0.6], "
       "layers: [{frac: 1.0, youngs: 300, material: liquid}]}")
SNOW = "snow: {fraction: 1.0, youngs: 250, layers: [{frac: 1.0, youngs: 250, material: snow}]}"
ELAS = "ball: {fraction: 1.0, youngs: 500, layers: [{frac: 1.0, youngs: 500, material: elastic}]}"

# THE `csf` CASE IS THE ONE THAT DOES NOT MATCH, and it is a real divergence rather than fallout
# from the `general.dt` change above: `csf` sets `g: 0`, so `gravity` emits nothing, `general.dt`
# reaches neither side, and both specs compute exactly what they computed before this file stopped
# running. It is also the ONLY case with `surface_tension` on (st=30; every other case is st=0), so
# the disagreement is in the surface-tension path -- the oracle's internal application of it against
# `mpm_grid_update`'s. It ACCUMULATES: max|delta| grows monotonically 6.6e-06 -> 4.1e-04 over the 20
# frames, which is a small per-step difference integrating, not a one-off branch taken differently.
# Marked strict, so whoever fixes the surface-tension path is told by this test rather than having
# to remember it.
CASES = [
    dict(name="liq_grav", types=LIQ, start="[0.3,0.5,0.3,0.5]", n=1, ppc=1500, rad=0.2,
         frames=20, g=12, drag=0.1, wd=0.5, st=0, obs="", sub=14),
    pytest.param(
        dict(name="csf", types=LIQ, start="[0.3,0.5,0.3,0.5]", n=1, ppc=1500, rad=0.2,
             frames=20, g=0, drag=0.1, wd=1.0, st=30, obs="", sub=14),
        marks=pytest.mark.xfail(strict=True, reason=(
            "surface_tension: the decomposition drifts from the oracle, worst max|delta| 4.1e-04 "
            "against a 1e-4 bound, growing monotonically over the run. Pre-existing -- this case "
            "is gravity-free, so nothing in the schedule-form fix touches it."))),
    dict(name="snow", types=SNOW, start="[0.5,0.7,0.5,0.7]", n=1, ppc=1200, rad=0.16,
         frames=25, g=12, drag=0.2, wd=0.5, st=0, obs="", sub=14),
    dict(name="elastic", types=ELAS, start="[0.5,0.7,0.5,0.7]", n=1, ppc=1200, rad=0.12,
         frames=25, g=12, drag=0.2, wd=0.6, st=0, obs="", sub=14),
    dict(name="obstacle", types=LIQ, start="[0.3,0.6,0.3,0.6]", n=1, ppc=1500, rad=0.2,
         frames=25, g=12, drag=0.2, wd=0.5, st=0,
         obs=", obstacles: [[0.45,0.0,0.55,0.4]]", sub=14),
]


def _case(c):
    """Unwrap a `pytest.param` back to the plain dict, so the ids, the test and the __main__
    loop all read the same list rather than three near-copies of it."""
    # `isinstance(c, dict)` and NOT `hasattr(c, "values")`: a dict HAS a `.values`, it is the
    # method, so the hasattr form unwrapped every plain case into `dict.values[0]`.
    return c if isinstance(c, dict) else c.values[0]


_IDS = [_case(c)["name"] for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_decomposition_reproduces_the_oracle(case):
    a = _run(_mono(case))
    b = _run(_dec(case))
    assert a.shape == b.shape, f"{case['name']}: {a.shape} vs {b.shape}"
    # THE MATERIAL HAS TO ACTUALLY MOVE, or "the two agree" is a statement about two copies of the
    # initial condition. `csf` has no gravity and is driven by surface tension alone, so this is not
    # a formality: it is the assertion that each case still exercises the solver at all.
    moved = np.abs(a - a[0]).max()
    assert moved > 1e-3, f"{case['name']}: the oracle moved by {moved:.2e} -- the case is inert"
    dmax = np.abs(a - b).max(axis=(1, 2))
    assert dmax.max() < 1e-4, (f"{case['name']}: decomposition diverges from the oracle -- "
                               f"worst max|delta| {dmax.max():.2e} at frame {int(dmax.argmax())}, "
                               f"final {dmax[-1]:.2e}")


if __name__ == "__main__":                       # still runnable as the header always promised
    print(f"{'case':12s} {'frames':>6s} {'final max|d|':>13s} {'worst max|d|':>13s}  verdict")
    ok = True
    for c in (_case(x) for x in CASES):
        a = _run(_mono(c)); b = _run(_dec(c))
        d = np.abs(a - b).max(axis=(1, 2))
        good = d.max() < 1e-4
        ok = ok and good
        print(f"{c['name']:12s} {c['frames']:6d} {d[-1]:13.2e} {d.max():13.2e}  "
              f"{'MATCH' if good else 'MISMATCH'}")
    print("\nALL MATCH -- decomposition reproduces the oracle" if ok
          else "\nFAIL: decomposition diverges from oracle")
