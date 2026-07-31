"""Property tests for the `regulate` operator (jax-morph ODEController).

These assert properties statable WITHOUT the reference -- exact solutions of the
drive-frozen scalar ODE the operator self-solves -- so they test the machinery
(adaptive self-solve accuracy, the inc/dt EMIT scaling, the algebraic sigmoid, the
degradation term, the frozen forcing input), not agreement with the oracle.

Key reference-free fact: when `W_gene = 0` the regulatory drive is constant in the
gene value, so each gene obeys the LINEAR scalar ODE

    dg/dt = s - gamma * g ,     s = sigma(W_in @ u + b)   (constant, u frozen)

whose exact solution is g(t) = g* + (g0 - g*) exp(-gamma t), g* = s/gamma. The
operator must reproduce this endpoint to solver tolerance, and its returned delta,
run through the engine's first-order step g += dt*delta, must land on g(dt).
"""
import math

import torch

from plexus.schema import Spec, OpSpec, Selector
from plexus.engine import build, _integrate, _resolve_emit
from plexus.models.registry import get_operator

import plexus.operators.candidates.jax_morph_odecontroller  # noqa: F401  (registers `regulate`)


def _cell_sim(n_gene, dt, params, n_in=0):
    """A one-set world: `n` cells carrying a first-order `gene` block (width n_gene)
    and, optionally, a frozen `drive` block (width n_in, integration=none)."""
    state = {"gene": {"width": n_gene, "integration": "first_order", "boundary": "free"}}
    if n_in:
        state["drive"] = {"width": n_in, "integration": "none", "boundary": "free", "record": False}
    sets = {"cell": {"n": 4, "state": state}}
    op = OpSpec(op="regulate", on=Selector("cell"), params={**params, "_at": "cell"})
    sim = Spec(name="reg", seed=0, n_frames=1, dt=dt, sets=sets, fields={},
               operators=[op], schedule=["regulate"])
    H = build(sim, device="cpu")
    H.emit_order = _resolve_emit(sim, H)
    return sim, H


def _op(params):
    return get_operator("regulate")({**params, "_at": "cell"}, "cpu")


def test_linear_decay_increment_is_exact():
    """W_gene=0, b=0, gamma=1, no drive -> dg/dt = 0.5 - g (sigma(0)=0.5), exactly.
    Starting at g0=0, the endpoint is g(dt) = 0.5(1 - e^{-dt}); the operator's
    self-solved increment, applied as g0 + dt*delta, must match it to tolerance."""
    dt = 0.7
    sim, H = _cell_sim(1, dt, {"gamma": 1.0})
    cell = H.level("cell")
    cell.get("gene")[:] = 0.0
    op = _op({"gamma": 1.0})
    H.zero_delta()
    delta = op(H, cell.active)["cell"]
    g_next = 0.0 + dt * delta                              # the engine's first-order step
    g_exact = 0.5 * (1.0 - math.exp(-dt))
    assert torch.allclose(g_next, torch.full_like(g_next, g_exact), atol=1e-4)


def test_engine_integration_lands_on_analytic_endpoint():
    """The same ODE, but driven through the engine's _integrate (the g += dt*delta path)
    from a nonzero start g0=0.3: gene block must equal g(dt) = 0.5 + (0.3-0.5)e^{-dt}."""
    dt = 1.3
    sim, H = _cell_sim(1, dt, {"gamma": 1.0})
    cell = H.level("cell")
    cell.get("gene")[:] = 0.3
    op = _op({"gamma": 1.0})
    H.zero_delta()
    H.add_delta("cell", op(H, cell.active)["cell"], op.INTEGRAND)
    _integrate(H, sim.dt)
    g_exact = 0.5 + (0.3 - 0.5) * math.exp(-dt)
    assert torch.allclose(cell.get("gene"), torch.full((4, 1), g_exact), atol=1e-4)


def test_fixed_point_is_stationary():
    """At the fixed point g* = sigma(0)/gamma = 0.5/gamma the derivative vanishes, so the
    self-solved increment is ~0 (a genuine equilibrium, not a coincidence of the endpoint)."""
    gamma = 2.0
    sim, H = _cell_sim(1, 0.9, {"gamma": gamma})
    cell = H.level("cell")
    cell.get("gene")[:] = 0.5 / gamma
    op = _op({"gamma": gamma})
    H.zero_delta()
    delta = op(H, cell.active)["cell"]
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)


def test_frozen_drive_forces_the_fixed_point_and_is_unchanged():
    """The sensed drive enters INSIDE the sigmoid (source wins over the paper): with
    W_gene=0, W_in=[[k]], b=0, gamma=1, the fixed point is g* = sigma(k*u). Seeding g0=g*
    gives ~0 delta, confirming the input path; and the `drive` block (integration=none)
    is untouched by the step -- it is read-only and frozen across the solve."""
    k, u = 1.5, 0.8
    g_star = 0.5 * (1.0 + (k * u) / math.hypot(1.0, k * u))     # algebraic sigmoid(k*u)
    sim, H = _cell_sim(1, 1.0, {"gamma": 1.0, "W_gene": [[0.0]], "W_in": [[k]], "inputs": "drive"}, n_in=1)
    cell = H.level("cell")
    cell.get("gene")[:] = g_star
    cell.get("drive")[:] = u
    op = _op({"gamma": 1.0, "W_gene": [[0.0]], "W_in": [[k]], "inputs": "drive"})
    H.zero_delta()
    delta = op(H, cell.active)["cell"]
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)          # at the driven fixed point
    assert torch.allclose(cell.get("drive"), torch.full((4, 1), u))           # drive frozen / read-only


def test_dead_cells_hold_their_gene_state():
    """A dormant cell (occ=0) gets a zero delta -- its heritable gene state is frozen."""
    sim, H = _cell_sim(2, 0.5, {"gamma": 1.0})
    cell = H.level("cell")
    cell.get("gene")[:] = 0.1
    cell.occ[2:] = 0.0                                         # retire the last two slots
    op = _op({"gamma": 1.0})
    H.zero_delta()
    delta = op(H, cell.active)["cell"]
    assert torch.count_nonzero(delta[2:]) == 0
    assert torch.all(delta[:2].abs() > 0)                     # live cells do move
