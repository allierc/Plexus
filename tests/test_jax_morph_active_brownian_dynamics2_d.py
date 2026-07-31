"""Property tests for the `reorient` operator (the rotational-diffusion leg of jax-morph's
ActiveBrownianDynamics2D).

These assert properties statable WITHOUT the reference -- limits, a conservation law, and a
symmetry that the operator's OWN contract fixes, not any oracle number:

* ZERO-DIFFUSION LIMIT -- rot_diffusion = 0 fixes the heading (exact no-op, no crash).
* NORM CONSERVATION -- rotational diffusion is a PLANAR ROTATION, so a unit heading stays
  exactly unit. This is the property that separates a correct rotation from a naive additive
  perturbation `h += noise` (which would grow/shrink |h|).
* ZERO-DRIFT SYMMETRY -- the angle increment has mean 0: the wander is equally likely CW/CCW,
  no systematic turning (the source scores dtheta against mean 0).
* DIFFUSION SCALING -- the angle increment's variance is 2 * rot_diffusion * dt (its own noise
  contract, std_r = sqrt(2 D_r dt); NOT a value fitted to the oracle).
* OCCUPANCY / MASK GATE -- dead slots and masked-out cells keep their heading.

None of these check agreement with the oracle -- they test the operator's contract.
"""
import math
import types

import torch

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator

import plexus.operators.candidates.jax_morph_active_brownian_dynamics2_d  # noqa: F401  (registers `reorient`)


def _world(n, *, dt=1.0, seed=0, occ=None, heading=None):
    """A one-set world: `n` cells with a unit `heading` buffer (pos/vel state is unused by
    reorient). `heading=None` seeds random unit vectors; pass a [n,2] array to fix them."""
    state = torch.zeros(n, 4)                                    # pos(2) + vel(2), untouched here
    schema = {"pos": (0, 2), "vel": (2, 4)}
    occ = torch.ones(n) if occ is None else torch.as_tensor(occ, dtype=torch.float32)
    lvl = Level("cell", state=state, state_schema=schema, occ=occ)
    if heading is None:
        g = torch.Generator().manual_seed(seed + 1000)
        h = torch.randn(n, 2, generator=g)
        h = h / h.norm(dim=1, keepdim=True).clamp(min=1e-9)
    else:
        h = torch.as_tensor(heading, dtype=torch.float32)
    lvl.register_buffer("heading", h)
    H = Hierarchy()
    H.add_level(lvl)
    H.dim = 2
    H.config = types.SimpleNamespace(dt=dt)
    H.rng = torch.Generator().manual_seed(seed)
    return H, lvl


def _op(params=None):
    return get_operator("reorient")((params or {}), "cpu")


def _signed_angle(h_old, h_new):
    """Signed rotation angle from h_old to h_new per cell = atan2(cross, dot), in (-pi, pi]."""
    cross = h_old[:, 0] * h_new[:, 1] - h_old[:, 1] * h_new[:, 0]
    dot = h_old[:, 0] * h_new[:, 0] + h_old[:, 1] * h_new[:, 1]
    return torch.atan2(cross, dot)


def test_zero_diffusion_is_a_noop():
    """rot_diffusion = 0 -> std_r = 0 -> dtheta = 0: the heading is fixed exactly, and the
    op returns no delta (it steers `heading` in place)."""
    H, lvl = _world(16, seed=1)
    before = lvl.heading.clone()
    out = _op({"rot_diffusion": 0.0})(H, lvl.active)
    assert out == {}                                            # heading-steer: returns no integrable delta
    assert torch.equal(lvl.heading, before)                    # heading unchanged, bit-for-bit


def test_norm_is_conserved():
    """Rotational diffusion is a planar ROTATION, so a unit heading stays unit after a step,
    even at a large diffusion rate -- the property a naive additive `h += noise` would break."""
    H, lvl = _world(2048, seed=2)
    assert torch.allclose(lvl.heading.norm(dim=1), torch.ones(2048), atol=1e-6)   # starts unit
    _op({"rot_diffusion": 5.0})(H, lvl.active)
    assert torch.allclose(lvl.heading.norm(dim=1), torch.ones(2048), atol=1e-5)   # stays unit


def test_heading_actually_rotates():
    """A positive rate genuinely turns the heading (guards against a silent no-op): with D_r>0
    essentially every cell's heading moves."""
    H, lvl = _world(1024, seed=3)
    before = lvl.heading.clone()
    _op({"rot_diffusion": 1.0})(H, lvl.active)
    moved = (lvl.heading - before).norm(dim=1) > 1e-6
    assert int(moved.sum()) > 1000                             # ~all of them turned


def test_zero_drift_and_diffusion_scaling():
    """Over a large ensemble the per-cell angle increment is N(0, 2 D_r dt): mean ~ 0
    (zero-drift symmetry) and variance ~ 2 D_r dt (the operator's own noise contract)."""
    n = 40000
    D_r, dt = 0.05, 1.0
    H, lvl = _world(n, dt=dt, seed=7)
    before = lvl.heading.clone()
    _op({"rot_diffusion": D_r})(H, lvl.active)
    dtheta = _signed_angle(before, lvl.heading)                # recovered rotation angle per cell
    expected_var = 2.0 * D_r * dt                              # std_r^2, from std_r = sqrt(2 D_r dt)
    assert abs(float(dtheta.mean())) < 0.01                    # no systematic CW/CCW turning
    assert math.isclose(float(dtheta.var(unbiased=True)), expected_var, rel_tol=0.05)


def test_variance_scales_with_dt():
    """The angle-increment variance is linear in dt (Brownian scaling): doubling dt doubles it."""
    D_r = 0.05
    v = {}
    for dt in (1.0, 2.0):
        H, lvl = _world(40000, dt=dt, seed=11)
        before = lvl.heading.clone()
        _op({"rot_diffusion": D_r})(H, lvl.active)
        v[dt] = float(_signed_angle(before, lvl.heading).var(unbiased=True))
    assert math.isclose(v[2.0] / v[1.0], 2.0, rel_tol=0.05)    # var(2 dt) / var(dt) == 2


def test_dead_and_masked_slots_are_unchanged():
    """Dormant slots (occ=0) and masked-out live cells draw dtheta = 0 -> identity rotation:
    their heading is untouched, while masked-in live cells rotate."""
    occ = [1, 1, 1, 1, 0, 0, 0, 0]                             # last four dormant
    H, lvl = _world(8, seed=5, occ=occ)
    before = lvl.heading.clone()
    mask = torch.tensor([True, True, False, False, False, False, False, False])  # only cells 0,1 eligible
    _op({"rot_diffusion": 2.0})(H, mask)
    # masked-out (cells 2,3) and all dormant slots (4..7) keep their heading exactly
    assert torch.equal(lvl.heading[2:], before[2:])
    # the two masked-in live cells actually turned
    assert torch.all((lvl.heading[:2] - before[:2]).norm(dim=1) > 1e-6)
