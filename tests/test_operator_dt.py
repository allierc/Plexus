"""Every velocity-emitting operator divides by `general.dt`, not by a dt it declares itself.

WHY THIS TEST EXISTS. `cell_mechanics` relaxes the shape energy for `relax_iters` descent steps and
returns a VELOCITY; the engine then multiplies that velocity by `general.dt` to advance the
positions. The two divisions have to cancel, so the divisor must be the harness's dt. Both
implementations instead read a `dt:` the SPEC declared, defaulting to 1.0 -- and on a spec with
`general.dt = 0.0032` that silently ran the tissue 312x slower than every other operator in the same
schedule. Nothing raised, nothing warned; the epithelium simply did not relax, and the run looked
like a modelling problem rather than a units bug.

WHAT IS ASSERTED, AND WHY IT IS THE RIGHT INVARIANT. The per-frame DISPLACEMENT `x - x0` is pure
geometry -- `relax_iters` steps of `-eta*mu*grad(E)`, with no dt anywhere in it. So the emitted
velocity must be exactly that displacement over `general.dt`, which gives two properties that hold
BY CONSTRUCTION and cannot both survive a declared divisor:

    (1)  v * general.dt  is INDEPENDENT of general.dt      -- the step the engine actually applies;
    (2)  a `dt:` in the operator's own params changes NOTHING.

Property (2) is the regression: before the fix, halving the declared `dt` doubled the velocity.
"""
import tempfile

import numpy as np
import pytest
import torch
import yaml

from plexus.engine import build
from plexus.models.registry import get_operator
from plexus.schema import load as load_spec


def _spheroid(dt, declared_dt=None, implementation=None):
    """A 24-cell vesicle, seeded and relaxed for one frame -- the smallest thing with a mesh.

    Written out and read back through `schema.load`, not hand-assembled, so the probe goes through
    the SAME parser a real spec does: `implementation:` has to resolve to a variant, the params have
    to survive `_RESERVED` filtering, and a key the schema would reject here would be rejected in a
    spec too.
    """
    mech = {"op": "cell_mechanics", "at": "vertex", "cell_set": "cell",
            "K_A": 1.0, "K_P": 1.0, "K_V": 2.0, "K_R": 0.4, "Lambda": 0.5, "Gamma": 0.1,
            "p0": 3.5, "relax_iters": 4}
    if declared_dt is not None:
        mech["dt"] = declared_dt
    if implementation is not None:
        # `model:`, not `implementation:` -- a monolayer cell with its own 3D volume is a different
        # biological hypothesis at this slot, not the same biology computed differently.
        mech["model"] = implementation
    raw = {
        "general": {"name": "dt_probe", "seed": 0, "n_frames": 1, "dt": dt, "dim": 3,
                    "world": [40.0, 40.0, 40.0]},
        "sets": {"vertex": {"n": 2048, "mesh": "half_edge", "cell_set": "cell"},
                 "cell": {"n": 512, "state": {"area": {"width": 1}, "cen": {"width": 3}}}},
        "fields": {},
        "operators": [{"op": "mesh_seed", "at": "vertex", "before_frame": 1, "cell_set": "cell",
                       "n_cells": 24, "radius": 5.0, "jitter": 0.18, "p0": 3.5, "seed": 0},
                      {"op": "cell_geometry", "at": "cell"}, mech],
        "schedule": ["mesh_seed", "cell_geometry", "cell_mechanics"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(raw, f)
        path = f.name
    sim = load_spec(path)
    return build(sim, device="cpu"), sim


def _step(HS):
    """Seed, then take the velocity `cell_mechanics` emits on the first real frame.

    The operators are instantiated the way `engine.run` does it -- `get_operator(name, variant)`
    with the spec's params plus `_at` -- because instances live in `run`'s local list, not on the
    Hierarchy, and this test deliberately does NOT go through `run`: it needs the raw emission,
    before the engine multiplies by dt and integrates it away.
    """
    H, sim = HS
    v = None
    for o in sim.operators:
        op = get_operator(o.op, variant=o.impl)(
            {**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}, "cpu")
        out = op.forward(H)
        if o.op == "cell_mechanics":
            v = out["vertex"] if isinstance(out, dict) else out
    return v.detach().clone()


IMPLS = [None, "monolayer"]        # the default mid-surface model, and the 3D-volume monolayer


@pytest.mark.parametrize("impl", IMPLS)
def test_declared_dt_is_ignored(impl):
    """(2) A `dt:` in the operator's own params does not scale the emitted velocity.

    This is the regression. With the old `self.dt` divisor, `dt: 0.5` doubled every velocity while
    `general.dt` stayed at 1.0, so the tissue moved twice as far per frame than the same spec
    without the line -- a knob that looked like a time step and behaved like a gain.
    """
    v_none = _step(_spheroid(dt=1.0, declared_dt=None, implementation=impl))
    v_half = _step(_spheroid(dt=1.0, declared_dt=0.5, implementation=impl))
    v_big = _step(_spheroid(dt=1.0, declared_dt=312.5, implementation=impl))
    assert torch.allclose(v_none, v_half, atol=0, rtol=0), "declared dt=0.5 changed the velocity"
    assert torch.allclose(v_none, v_big, atol=0, rtol=0), "declared dt=312.5 changed the velocity"


@pytest.mark.parametrize("impl", IMPLS)
def test_velocity_tracks_general_dt(impl):
    """(1) `v * general.dt` -- the displacement the engine applies -- is invariant to general.dt.

    Equivalently the velocity scales as 1/general.dt: halve the harness's step and the operator
    reports twice the velocity, so the same geometric relaxation lands per frame. A divisor that
    ignored the harness would leave `v` unchanged and the applied step halved.
    """
    for a, b in ((1.0, 0.0032), (1.0, 0.5)):
        va = _step(_spheroid(dt=a, implementation=impl))
        vb = _step(_spheroid(dt=b, implementation=impl))
        da, db = va * a, vb * b
        assert torch.allclose(da, db, atol=1e-10, rtol=1e-6), (
            f"{impl or 'default'}: displacement changed with general.dt "
            f"({a} -> {b}); max |delta| = {(da - db).abs().max():.3e}")
        # and the velocity itself really did rescale -- otherwise the test above passes trivially
        # on an operator that emits zero.
        assert va.abs().max() > 0, "the probe relaxed nothing; the invariant is vacuous"
        assert np.isclose(float(vb.abs().max() / va.abs().max()), a / b, rtol=1e-5)
