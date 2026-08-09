#!/usr/bin/env python
"""Certify every mode of the death gate against a population whose answer is known.

Cedric, 9 August: "can you generate specific data to test each every mode of the death gate."

WHY THE SELECTOR AND NOT THE WHOLE OPERATOR. `apoptosis_3d` is three steps -- mark, shrink until
T1 sheds the cell to a triangle, extrude -- and only the first is a rule. A full run conflates the
rule with the pathway and with whatever the chemistry is doing, which is exactly how four of these
modes were reported as "no effect" when they were simply unreachable: at a shared `stall_frac` of
0.5, `competition`, `stalled`, `smaller` and `older` removed ZERO cells over 900 frames while
`dimmer` removed 379. Nothing in that run said whether the rules were wrong or the number was.
Here the marked set is read directly, so a silent rule is visibly silent and against a population
that should have made it speak.

EVERY MODE HAS A NEGATIVE CASE, and for the local ones it is the important one. A local rule must
NOT fire on a UNIFORM field however extreme -- that is the whole property that distinguishes it
from a global threshold, and it is the property both global rules lacked:

    chem_low  marked every cell below a fraction of the field maximum, so when the pattern
              weakened it marked the tissue: 2,000 cells shrank to 21.6% of volume, nothing was
              extruded.
    stalled   compared against the population median and marked nobody on a tissue where every
              slow cell was equally slow.

    python test_apoptosis_modes.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from tyssue_ops3d import build_sphere_mesh                       # noqa: E402
from tyssue_topology_ops3d import rings_from_flat_3d             # noqa: E402
from plexus.models.registry import get_operator                  # noqa: E402
import tyssue_ops3d                                              # noqa: E402,F401

N_CELLS = 400


class _Level:
    """The two attributes `_marked` touches on the cell set."""
    def __init__(self, nF, chem=None):
        self.state_schema = {"chem": (0, 2)} if chem is not None else {}
        self.state = torch.zeros(nF, 2) if chem is None else torch.stack(
            [torch.as_tensor(chem, dtype=torch.float32), torch.zeros(nF)], dim=1)
        self.occ = None


class _H:
    def __init__(self, cell):
        self._cell = cell

    def level(self, _name):
        return self._cell


def _mesh(nF_target=N_CELLS):
    """A seeded vesicle, plus the per-cell arrays the modes read. Uniform by default, so any
    firing is caused by what a test deliberately changes and by nothing else."""
    verts, es, et, ef, nF = build_sphere_mesh(nF_target, 5.0, 0.15, 0)
    pos = np.asarray(verts, float)
    rings = rings_from_flat_3d(es, et, ef, nF)
    cen = np.array([pos[r].mean(0) for r in rings])
    v = np.full(nF, 0.25)
    m = {"E_srce": torch.as_tensor(es), "E_trgt": torch.as_tensor(et),
         "E_face": torch.as_tensor(ef), "nF": nF, "Nv": len(pos),
         "V0f": torch.as_tensor(v), "Vbirth": torch.as_tensor(v.copy()),
         "age": torch.full((nF,), 20.0), "v_ref": 0.25, "cen_np": cen}
    return m, nF, cen


def _op(**kw):
    cls = get_operator("apoptosis_3d")
    return cls(dict(kw), device="cpu")


def _neighbours(m, nF, f):
    """The cells sharing an edge with f -- used to BUILD the fixtures, so a test states its
    population in the same terms the rule reads it."""
    from tyssue_ops3d import ShapeEnergy3D
    twin = ShapeEnergy3D._twin_faces(m["E_srce"].long(), m["E_trgt"].long(),
                                     m["E_face"].long(), int(m["Nv"])).numpy()
    ef = m["E_face"].numpy()
    return sorted({int(twin[k]) for k in range(len(ef))
                   if int(ef[k]) == f and 0 <= twin[k] < nF and twin[k] != f})


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


# --------------------------------------------------------------------- the absolute modes
@case("list marks exactly the named cell")
def _t():
    m, nF, _ = _mesh()
    got = _op(mode="list", cells=[7])._marked(m, _H(_Level(nF)), nF)
    return got == {7}, f"{sorted(got)}"


@case("cone marks a contiguous cap of the predicted size")
def _t():
    m, nF, cen = _mesh(2000)
    got = _op(mode="cone", cone_deg=22.8)._marked(m, _H(_Level(nF)), nF)
    # 1000*(1-cos t) cells on a 2,000-cell sphere -> 78 at 22.8 degrees; the seeded mesh is
    # jittered, so allow the spread a real Fibonacci sphere has
    return 60 <= len(got) <= 100, f"{len(got)} cells, predicted ~78"


@case("band marks a ring, and n_bands multiplies it")
def _t():
    m, nF, _ = _mesh(2000)
    one = _op(mode="band", band_deg=8.0, n_bands=1)._marked(m, _H(_Level(nF)), nF)
    nine = _op(mode="band", band_deg=4.0, n_bands=9)._marked(m, _H(_Level(nF)), nF)
    return len(one) > 0 and len(nine) > 2 * len(one), f"1 band {len(one)}, 9 bands {len(nine)}"


@case("small marks only the cells below the absolute threshold")
def _t():
    m, nF, _ = _mesh()
    v = m["V0f"].clone(); v[[3, 11, 29]] = 0.05          # 0.2 x v_ref, under a 0.35 cut
    m["V0f"] = v
    got = _op(mode="small", small_frac=0.35)._marked(m, _H(_Level(nF)), nF)
    return got == {3, 11, 29}, f"{sorted(got)}"


@case("small marks NOBODY when every cell is equally small")
def _t():
    m, nF, _ = _mesh()
    m["V0f"] = torch.full((nF,), 0.05)                   # the whole tissue shrank
    m["v_ref"] = 0.05                                    # ... and v_ref is its own median
    got = _op(mode="small", small_frac=0.35)._marked(m, _H(_Level(nF)), nF)
    return got == set(), f"{len(got)} marked on a uniformly small tissue"


# --------------------------------------------------------------------- the local family
@case("competition marks the slow cell among fast neighbours")
def _t():
    m, nF, _ = _mesh()
    vb = m["Vbirth"].clone(); v = m["V0f"].clone()
    v[:] = 0.25 * 1.5                                    # everyone grew 50% since birth
    victim = 40
    v[victim] = 0.25 * 1.02                              # except this one: 2%
    m["V0f"] = v; m["Vbirth"] = vb
    got = _op(mode="competition")._marked(m, _H(_Level(nF)), nF)
    return victim in got, f"victim {'in' if victim in got else 'MISSING from'} {len(got)} marked"


@case("competition marks NOBODY when the whole tissue is equally slow")
def _t():
    m, nF, _ = _mesh()
    m["V0f"] = m["Vbirth"] * 1.02                        # everyone grew 2%, nobody is a loser
    got = _op(mode="competition")._marked(m, _H(_Level(nF)), nF)
    return got == set(), f"{len(got)} marked on a uniformly slow tissue"


@case("smaller marks the small cell among normal neighbours")
def _t():
    m, nF, _ = _mesh()
    v = m["V0f"].clone(); victim = 55
    v[victim] = 0.25 * 0.4                               # 40% of its neighbours
    m["V0f"] = v
    got = _op(mode="smaller")._marked(m, _H(_Level(nF)), nF)
    return victim in got, f"victim {'in' if victim in got else 'MISSING from'} {len(got)} marked"


@case("smaller marks NOBODY on a uniform tissue")
def _t():
    m, nF, _ = _mesh()
    got = _op(mode="smaller")._marked(m, _H(_Level(nF)), nF)
    return got == set(), f"{len(got)} marked on a uniform tissue"


@case("dimmer marks the dim cell beside a bright patch")
def _t():
    m, nF, _ = _mesh()
    a = np.full(nF, 0.02)
    bright = _neighbours(m, nF, 60)
    a[bright] = 1.0                                      # a lit ring around cell 60
    got = _op(mode="dimmer")._marked(m, _H(_Level(nF, chem=a)), nF)
    return 60 in got, f"cell 60 {'in' if 60 in got else 'MISSING from'} {len(got)} marked"


@case("dimmer marks NOBODY on a uniformly dim field -- the chem_low failure")
def _t():
    m, nF, _ = _mesh()
    got = _op(mode="dimmer")._marked(m, _H(_Level(nF, chem=np.full(nF, 0.001))), nF)
    return got == set(), f"{len(got)} marked -- a global rule would mark all {nF}"


@case("older marks the cell that stopped cycling")
def _t():
    m, nF, _ = _mesh()
    ag = m["age"].clone(); victim = 77
    ag[victim] = 200.0                                   # 10x its neighbours
    m["age"] = ag
    got = _op(mode="older")._marked(m, _H(_Level(nF)), nF)
    return victim in got, f"victim {'in' if victim in got else 'MISSING from'} {len(got)} marked"


@case("older marks NOBODY when the whole tissue is old")
def _t():
    m, nF, _ = _mesh()
    m["age"] = torch.full((nF,), 500.0)
    got = _op(mode="older")._marked(m, _H(_Level(nF)), nF)
    return got == set(), f"{len(got)} marked on a uniformly old tissue"


@case("crowded fires above its neighbour count and not below")
def _t():
    m, nF, _ = _mesh()
    hi = _op(mode="crowded", n_max=99)._marked(m, _H(_Level(nF)), nF)      # unreachable
    lo = _op(mode="crowded", n_max=4)._marked(m, _H(_Level(nF)), nF)       # most cells
    return hi == set() and len(lo) > 0, f"n_max 99 -> {len(hi)}, n_max 4 -> {len(lo)}"


@case("min_age spares a cell that is too young to judge")
def _t():
    m, nF, _ = _mesh()
    v = m["V0f"].clone(); victim = 55
    v[victim] = 0.25 * 0.4
    m["V0f"] = v
    m["age"] = torch.zeros(nF)                           # everyone just born
    got = _op(mode="smaller", min_age=4)._marked(m, _H(_Level(nF)), nF)
    return got == set(), f"{len(got)} marked despite age 0 < min_age 4"


def main():
    print("CERTIFYING every mode of the death gate against a population whose answer is known\n")
    bad = 0
    for name, fn in CASES:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {str(e)[:70]}"
        bad += not ok
        print(f"  [{'ok ' if ok else 'BAD'}] {name:<62} {detail}")
    print(f"\n  {len(CASES) - bad}/{len(CASES)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
