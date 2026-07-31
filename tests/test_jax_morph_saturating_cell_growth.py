"""Property tests for grow_radius (candidate jax_morph_saturating_cell_growth).

We assert properties statable WITHOUT the JAX reference -- facts about the saturating ODE
dr/dt = k(1 - r/R) itself, not fitted reference numbers:

  * the exact closed-form endpoint (the ODE's known analytic solution) through Plexus's
    `radius += dt*delta` first-order integration convention;
  * UNCONDITIONAL STABILITY -- the headline distinction from forward Euler: even a huge dt never
    overshoots R (an Euler step with the same dt would blow past it);
  * the fixed point at R, the k=0 no-op, sign symmetry about R, the multi-step asymptote,
    dormant-cell masking, and the registration/routing.
"""
import math

import torch
import pytest

import plexus.operators.candidates.jax_morph_saturating_cell_growth as m   # noqa: F401  self-registers grow_radius
from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_contract, get_operator
from plexus.models.state import (
    Block, StateSchema, NONE, FIRST_ORDER, SECOND_ORDER_COORDINATE, SECOND_ORDER_RATE,
    BOUNDARY_WORLD, BOUNDARY_FREE,
)


class _Cfg:
    def __init__(self, dt):
        self.dt = dt


def _make_H(N=5, dt=0.5, device="cpu"):
    """A one-cell-set Hierarchy: pos/vel (spatial coordinate) + a first-order `radius` block +
    a non-integrated `growth_rate` driver block. `pos` being the coordinate makes `radius` a
    NON-coordinate first-order integrand (radius += dt*delta)."""
    schema = StateSchema([
        Block("pos", 2, role="coordinate", integration=SECOND_ORDER_COORDINATE, boundary=BOUNDARY_WORLD),
        Block("vel", 2, role="rate", integration=SECOND_ORDER_RATE, record=False),
        Block("radius", 1, integration=FIRST_ORDER, boundary=BOUNDARY_FREE),
        Block("growth_rate", 1, integration=NONE, boundary=BOUNDARY_FREE),
    ])
    state = torch.zeros(N, schema.dim, device=device)
    cell = Level("cell", state=state, state_schema=schema)
    H = Hierarchy()
    H.add_level(cell)
    H.config = _Cfg(dt)
    return H, cell


def _set(cell, name, value):
    a, b = cell.state_schema[name]
    cell.state[:, a:b] = value


def _get(cell, name):
    a, b = cell.state_schema[name]
    return cell.state[:, a:b]


def _op(**params):
    return get_operator("grow_radius")({"_at": "cell", **params}, device="cpu")


def test_registration_and_routing():
    """grow_radius is a growth-family lateral op that writes the non-coordinate `radius` block."""
    c = get_contract("grow_radius")
    assert c.kind == "lateral" and c.family == "growth" and c.set == "cell"
    cls = get_operator("grow_radius")
    assert cls.EMIT == "velocity" and cls.INTEGRAND == "radius"   # first-order delta on a non-coordinate block
    assert cls.MAPS == []                                         # per-cell autonomous ODE, no coupling
    assert cls.signature()["writes"] == ["radius"]


def test_matches_analytic_ode_solution():
    """One step reproduces the ODE's KNOWN solution r(dt) = R - (R-r0)exp(-k dt/R) via the
    engine convention new_radius = r0 + dt*delta. This is a fact about dr/dt=k(1-r/R), not a
    reference number."""
    N, dt, R, k = 6, 0.7, 1.3, 0.9
    H, cell = _make_H(N=N, dt=dt)
    r0 = torch.linspace(0.05, 1.0, N).unsqueeze(1)               # a spread of starting sizes
    _set(cell, "radius", r0)
    _set(cell, "growth_rate", torch.full((N, 1), k))
    delta = _op(max_radius=R).forward(H)["cell"]
    new_r = r0 + dt * delta                                      # engine integration: radius += dt*delta
    analytic = R - (R - r0) * math.exp(-k * dt / R)
    assert torch.allclose(new_r, analytic, atol=1e-6), (new_r - analytic).abs().max()


def test_unconditional_stability_no_overshoot():
    """The exact flow never overshoots R for ANY dt -- the property that separates it from
    forward Euler. A huge dt still lands inside (r0, R); an Euler step with the same dt would
    blow past R."""
    R, k, r0 = 1.0, 5.0, 0.2
    for dt in (0.1, 1.0, 10.0, 1e3):
        H, cell = _make_H(N=1, dt=dt)
        _set(cell, "radius", torch.tensor([[r0]]))
        _set(cell, "growth_rate", torch.tensor([[k]]))
        delta = _op(max_radius=R).forward(H)["cell"]
        new_r = (r0 + dt * float(delta))                        # our exact-flow endpoint
        euler = r0 + k * (1.0 - r0 / R) * dt                    # naive forward Euler with the same dt
        # monotone up, never past R (1e-5 tolerance absorbs float32 roundoff at huge dt; the real
        # contrast is Euler's gross overshoot below, not sub-ulp noise at the saturation point).
        assert r0 < new_r < R + 1e-5, (dt, new_r)
        if dt >= 1.0:
            assert new_r < euler - 1e-3                         # Euler overshoots where the exact flow saturates


def test_fixed_point_at_target():
    """r == R is a fixed point: no growth at the target size (dr = 0)."""
    R = 0.8
    H, cell = _make_H(N=4, dt=0.5)
    _set(cell, "radius", torch.full((4, 1), R))
    _set(cell, "growth_rate", torch.full((4, 1), 2.0))
    delta = _op(max_radius=R).forward(H)["cell"]
    assert torch.allclose(delta, torch.zeros(4, 1), atol=1e-7)


def test_sign_symmetry_about_target():
    """Below R the cell GROWS (delta>0); above R it RELAXES DOWN (delta<0) -- dr/dt=k(1-r/R)
    drives r toward R from either side."""
    R = 1.0
    H, cell = _make_H(N=2, dt=0.3)
    _set(cell, "radius", torch.tensor([[0.4], [1.6]]))          # one below, one above target
    _set(cell, "growth_rate", torch.full((2, 1), 1.5))
    delta = _op(max_radius=R).forward(H)["cell"]
    assert delta[0, 0] > 0 and delta[1, 0] < 0


def test_zero_rate_is_a_noop():
    """No growth_rate + default rate=0 -> k=0 -> exactly zero delta (byte no-op)."""
    H, cell = _make_H(N=3, dt=0.5)
    _set(cell, "radius", torch.tensor([[0.1], [0.3], [0.6]]))   # growth_rate left at 0
    delta = _op(max_radius=1.0).forward(H)["cell"]
    assert torch.equal(delta, torch.zeros(3, 1))


def test_multistep_asymptote_to_target():
    """Iterating radius += dt*delta drives every cell toward R and holds there (saturation)."""
    R, k, dt = 1.0, 0.7, 0.5
    H, cell = _make_H(N=4, dt=dt)
    _set(cell, "radius", torch.tensor([[0.05], [0.2], [0.5], [0.9]]))
    _set(cell, "growth_rate", torch.full((4, 1), k))
    op = _op(max_radius=R)
    for _ in range(400):
        delta = op.forward(H)["cell"]
        _set(cell, "radius", _get(cell, "radius") + dt * delta)
    r = _get(cell, "radius")
    assert torch.allclose(r, torch.full((4, 1), R), atol=1e-4)
    assert (r < R + 1e-9).all()                                # approached from below, never overshot


def test_dormant_cells_get_no_increment():
    """occ==0 cells receive a zero delta regardless of their size/rate."""
    H, cell = _make_H(N=5, dt=0.5)
    _set(cell, "radius", torch.full((5, 1), 0.2))
    _set(cell, "growth_rate", torch.full((5, 1), 3.0))
    cell.occ[3:] = 0.0                                          # retire the last two cells
    delta = _op(max_radius=1.0).forward(H)["cell"]
    assert torch.allclose(delta[3:], torch.zeros(2, 1), atol=1e-9)
    assert (delta[:3] > 0).all()                               # live cells still grow


def test_end_to_end_through_engine_integrate():
    """The returned delta, run through the real engine `_integrate`, lands on the analytic
    endpoint -- confirming the first-order `_delta_blocks` (radius) routing composes."""
    from plexus.engine import _integrate
    N, dt, R, k = 4, 0.5, 1.2, 1.1
    H, cell = _make_H(N=N, dt=dt)
    r0 = torch.tensor([[0.1], [0.3], [0.6], [0.9]])
    _set(cell, "radius", r0)
    _set(cell, "growth_rate", torch.full((N, 1), k))
    op = _op(max_radius=R)
    H.emit_order = {}                                          # no coordinate op on cell in this minimal harness
    H.world_size = torch.tensor([10.0, 10.0]); H.dim = 2
    H.zero_delta()
    for lvl, d in op.forward(H).items():
        H.add_delta(lvl, d, op.INTEGRAND)                      # engine main-loop routing (block = "radius")
    _integrate(H, dt)
    analytic = R - (R - r0) * math.exp(-k * dt / R)
    assert torch.allclose(_get(cell, "radius"), analytic, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
