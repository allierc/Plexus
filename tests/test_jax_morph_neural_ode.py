"""Property tests for regulate:neural_ode (candidate jax_morph_neural_ode).

We assert properties statable WITHOUT the JAX reference -- a known-solution limit, a zero
limit, a sign, dormant-masking, and the shared-contract registration. The key test injects a
LINEAR vector field dy/dt = -k*y (a configured, known RHS -- not a fitted reference number)
and checks the operator reproduces analytic exponential decay y(dt) = y0*exp(-k*dt) through
Plexus's `g += dt*delta` integration convention.
"""
import math

import torch
import torch.nn as nn
import pytest

# Import the sibling FIRST so `connectionist` stays the shipped default of the `regulate`
# contract (default == first-registered implementation); our own tests select `neural_ode`
# explicitly, so they are order-independent, but the sibling's tests call get_operator(
# "regulate") with no implementation and rely on connectionist being the default.
import plexus.operators.candidates.jax_morph_odecontroller as _oc      # noqa: F401  sibling connectionist (default)
import plexus.operators.candidates.jax_morph_neural_ode as m          # noqa: F401  self-registers neural_ode
from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_contract, get_operator
from plexus.models.state import (
    Block, StateSchema, NONE, FIRST_ORDER, SECOND_ORDER_COORDINATE, SECOND_ORDER_RATE,
    BOUNDARY_WORLD, BOUNDARY_FREE,
)


class _Cfg:
    def __init__(self, dt):
        self.dt = dt


def _make_H(N=5, in_w=2, gene_w=3, dt=0.5, device="cpu"):
    """A one-cell-set Hierarchy: pos/vel (spatial coordinate) + a driver block `u` + an
    evolving first-order `gene` block. `pos` being the coordinate makes `gene` a
    NON-coordinate first-order integrand."""
    schema = StateSchema([
        Block("pos", 2, role="coordinate", integration=SECOND_ORDER_COORDINATE, boundary=BOUNDARY_WORLD),
        Block("vel", 2, role="rate", integration=SECOND_ORDER_RATE, record=False),
        Block("u", in_w, integration=NONE, boundary=BOUNDARY_FREE),
        Block("gene", gene_w, integration=FIRST_ORDER, boundary=BOUNDARY_FREE),
    ])
    state = torch.zeros(N, schema.dim, device=device)
    cell = Level("cell", state=state, state_schema=schema)
    H = Hierarchy()
    H.add_level(cell)
    H.config = _Cfg(dt)
    return H, cell


def _set_block(cell, name, value):
    a, b = cell.state_schema[name]
    cell.state[:, a:b] = value


def _get_block(cell, name):
    a, b = cell.state_schema[name]
    return cell.state[:, a:b]


def _linear_decay_net(n_in, n_gene, k):
    """A single linear layer realizing dy/dt = -k*y with NO dependence on the drivers u.
    Input layout is [u (n_in cols) | y (n_gene cols)]."""
    in_dim = n_in + n_gene
    net = nn.Sequential(nn.Linear(in_dim, n_gene))
    lin = net[0]
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.zero_()
        lin.weight[:, n_in:] = -k * torch.eye(n_gene)          # -k on the y-columns only
    return net


def _op(**params):
    return get_operator("regulate", "neural_ode")({"_at": "cell", **params}, device="cpu")


def test_registration_shared_contract():
    """neural_ode and connectionist are two implementations of the ONE `regulate` contract."""
    c = get_contract("regulate")
    assert c.kind == "exchange" and c.family == "fields" and c.set == "cell"
    assert {"neural_ode", "connectionist"} <= set(c.implementations)
    cls = get_operator("regulate", "neural_ode")
    assert cls.EMIT == "velocity" and cls.INTEGRAND == "gene"     # shared routing with the sibling
    assert cls.MAPS == [] and cls.DIFFERENTIABLE is True


def test_linear_field_matches_analytic_exponential_decay():
    """dy/dt = -k*y integrated over dt gives y0*exp(-k*dt) to solver tolerance, via the
    engine's rate convention (new_gene = gene0 + dt*delta)."""
    torch.manual_seed(0)
    N, in_w, gene_w, dt, k = 6, 2, 3, 0.7, 1.3
    H, cell = _make_H(N=N, in_w=in_w, gene_w=gene_w, dt=dt)
    g0 = torch.randn(N, gene_w)
    _set_block(cell, "gene", g0)
    _set_block(cell, "u", torch.randn(N, in_w) * 3.0)            # the net ignores u -> result must not depend on it

    op = _op(inputs="u", state="gene", net=_linear_decay_net(in_w, gene_w, k))
    out = op.forward(H)
    delta = out["cell"]
    new_g = g0 + dt * delta                                      # engine integration: g += dt*delta
    analytic = g0 * math.exp(-k * dt)
    assert torch.allclose(new_g, analytic, atol=1e-4, rtol=1e-4), (new_g - analytic).abs().max()
    assert (new_g.abs() <= g0.abs() + 1e-6).all()               # decay moves the field toward zero


def test_end_to_end_through_engine_integrate():
    """The returned delta, run through the real engine `_integrate`, lands on the analytic
    endpoint -- confirming the first-order `_delta_blocks` routing composes with the engine."""
    from plexus.engine import _integrate
    N, in_w, gene_w, dt, k = 4, 1, 2, 0.5, 1.2
    H, cell = _make_H(N=N, in_w=in_w, gene_w=gene_w, dt=dt)
    g0 = torch.randn(N, gene_w)
    _set_block(cell, "gene", g0)
    op = _op(inputs="u", state="gene", net=_linear_decay_net(in_w, gene_w, k))
    H.emit_order = {}                                           # no coordinate op on cell in this minimal harness
    H.world_size = torch.tensor([10.0, 10.0]); H.dim = 2
    H.zero_delta()
    deltas = op.forward(H)
    for lvl, d in deltas.items():
        H.add_delta(lvl, d, op.INTEGRAND)                       # engine main-loop routing
    _integrate(H, dt)
    new_g = _get_block(cell, "gene")
    assert torch.allclose(new_g, g0 * math.exp(-k * dt), atol=1e-4, rtol=1e-4)


def test_driver_freezing_is_invariant_when_field_ignores_u():
    """The RHS closes over u once; here the net ignores u, so different u give the same
    increment -- drivers enter only through the frozen packed vector."""
    N, in_w, gene_w, dt, k = 4, 2, 2, 0.4, 0.9
    net = _linear_decay_net(in_w, gene_w, k)
    g0 = torch.randn(N, gene_w)

    def run(u_scale):
        H, cell = _make_H(N=N, in_w=in_w, gene_w=gene_w, dt=dt)
        _set_block(cell, "gene", g0)
        _set_block(cell, "u", torch.full((N, in_w), float(u_scale)))
        return _op(inputs="u", state="gene", net=net).forward(H)["cell"].clone()

    assert torch.allclose(run(0.0), run(5.0), atol=1e-6)


def test_zero_vector_field_gives_zero_delta():
    """A vanishing RHS (zeroed final layer) integrates to no change -> ~zero rate."""
    N, in_w, gene_w, dt = 4, 1, 2, 0.5
    net = nn.Sequential(nn.Linear(in_w + gene_w, gene_w))
    with torch.no_grad():
        net[0].weight.zero_(); net[0].bias.zero_()
    H, cell = _make_H(N=N, in_w=in_w, gene_w=gene_w, dt=dt)
    _set_block(cell, "gene", torch.randn(N, gene_w))
    delta = _op(inputs="u", state="gene", net=net).forward(H)["cell"]
    # exactly-zero derivative -> no change; only float32 solver roundoff (~1e-6) remains
    assert torch.allclose(delta, torch.zeros(N, gene_w), atol=1e-5)


def test_dormant_cells_get_no_increment():
    """occ==0 cells receive a zero increment regardless of their state."""
    N, in_w, gene_w, dt, k = 5, 1, 2, 0.6, 1.1
    net = _linear_decay_net(in_w, gene_w, k)
    H, cell = _make_H(N=N, in_w=in_w, gene_w=gene_w, dt=dt)
    _set_block(cell, "gene", torch.randn(N, gene_w) + 2.0)
    cell.occ[3:] = 0.0                                          # retire the last two cells
    delta = _op(inputs="u", state="gene", net=net).forward(H)["cell"]
    assert torch.allclose(delta[3:], torch.zeros(2, gene_w), atol=1e-7)
    assert (delta[:3].abs() > 0).any()                          # live cells still move


def test_hidden_split_integrates_whole_block():
    """hidden_size names the leading latent columns of the ONE `gene` block; the whole block
    is solved as one coupled vector (the source's concat(hidden, outputs))."""
    N, in_w, gene_w, dt, k = 3, 1, 4, 0.5, 0.8
    H, cell = _make_H(N=N, in_w=in_w, gene_w=gene_w, dt=dt)
    g0 = torch.randn(N, gene_w)
    _set_block(cell, "gene", g0)
    op = _op(inputs="u", state="gene", hidden_size=2, net=_linear_decay_net(in_w, gene_w, k))
    delta = op.forward(H)["cell"]
    new_g = g0 + dt * delta
    assert delta.shape == (N, gene_w)                           # hidden + outputs integrated together
    assert torch.allclose(new_g, g0 * math.exp(-k * dt), atol=1e-4, rtol=1e-4)


def test_lazy_mlp_shape_validation():
    """An injected net of the wrong shape is rejected at forward, mirroring the source ctor."""
    N, in_w, gene_w = 3, 2, 3
    bad = nn.Sequential(nn.Linear(in_w + gene_w + 1, gene_w))   # wrong in_size
    H, cell = _make_H(N=N, in_w=in_w, gene_w=gene_w)
    _set_block(cell, "gene", torch.randn(N, gene_w))
    with pytest.raises(ValueError):
        _op(inputs="u", state="gene", net=bad).forward(H)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
