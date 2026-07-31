"""Property tests for the `regulate:connectionist` candidate (jax-morph gene circuit).

Every assertion here is stated from the operator's OWN definition -- the algebraic
sigmoid's range, its value at 0, and the fixed point of the zero-interaction circuit --
never from the JAX reference's numbers. The differential test against the oracle comes
later; nothing here is fitted to it.
"""
import torch

from plexus.schema import Spec, OpSpec, Selector
from plexus.engine import build, _integrate, _resolve_emit
from plexus.operators.candidates.jax_morph_gene_network_connectionist import (
    RegulateConnectionist, _rescaled_sigmoid,
)


def test_production_term_is_sigmoid_bounded():
    """The regulatory production is sigma_alg(drive) in (0, 1) for ANY drive, so the
    vector field satisfies 0 < dg/dt + gamma*g < 1 elementwise -- a definitional bound on
    the algebraic sigmoid, independent of the parameters or the integrator."""
    torch.manual_seed(0)
    n_gene, in_size, N, gamma = 4, 3, 16, 0.1
    op = RegulateConnectionist({
        "W_gene": torch.randn(n_gene, n_gene) * 3.0,   # large drives -> deep into saturation
        "W_in": torch.randn(n_gene, in_size) * 3.0,
        "b": torch.randn(n_gene) * 3.0,
        "gamma": gamma,
    }, "cpu")
    g = torch.randn(N, n_gene) * 5.0
    u = torch.randn(N, in_size) * 5.0
    production = op.vector_field(g, u) + gamma * g      # dg/dt + gamma*g = sigma_alg(drive)
    assert (production > 0.0).all()
    assert (production < 1.0).all()


def test_rescaled_sigmoid_shape():
    """sigma_alg(0) = 0.5 exactly, monotone, strictly inside (0,1) for moderate drives; at
    drives that would overflow a naive x/sqrt(1+x*x) the guard keeps it FINITE in [0,1]
    (it saturates to exactly 0/1, never NaN)."""
    x = torch.tensor([-2.0, 0.0, 2.0])
    s = _rescaled_sigmoid(x)
    assert torch.allclose(s[1], torch.tensor(0.5))
    assert (s > 0.0).all() and (s < 1.0).all()
    assert torch.all(s[1:] >= s[:-1])                  # monotone non-decreasing
    big = _rescaled_sigmoid(torch.tensor([-1e30, 1e30]))
    assert torch.isfinite(big).all()                   # overflow guard: no NaN
    assert (big >= 0.0).all() and (big <= 1.0).all()


def test_zero_interaction_circuit_is_not_inert():
    """Defaults (W/W_in/b = zeros) are NOT a no-op: with gamma=0.1 the drive is zero, so
    production = sigma_alg(0) = 0.5 and dg/dt = 0.5 - 0.1*g drives every gene toward the
    fixed point g* = 0.5/gamma = 5.0. Derived purely from this operator's definition."""
    op = RegulateConnectionist({"gamma": 0.1}, "cpu")  # all matrices default to zeros
    u = torch.zeros(3, 0)
    dg0 = op.vector_field(torch.zeros(3, 2), u)
    assert torch.allclose(dg0, torch.full_like(dg0, 0.5))            # 0.5 - 0.1*0
    g_star = torch.full((3, 2), 5.0)
    assert torch.allclose(op.vector_field(g_star, u), torch.zeros_like(g_star), atol=1e-6)


def test_self_solve_contracts_toward_fixed_point_through_engine():
    """End-to-end: the operator self-solves the macro-step and the engine's first-order
    step (g += dt*delta) recovers exactly the internally integrated endpoint g(dt), which
    for the zero-interaction circuit is strictly CLOSER to g* = 5.0 than g(0) was (and on
    the same side) -- a contraction any accurate integrator must produce."""
    params = {"gamma": 0.1, "substeps": 8}
    sets = {"cell": {"n": 2, "state": {"gene": {"width": 1, "integration": "first_order",
                                                "boundary": "free"}}}}
    ops = [OpSpec(op="regulate", impl="connectionist", on=Selector("cell"), params=params)]
    sim = Spec(name="reg", seed=0, n_frames=1, dt=1.0, sets=sets, fields={},
               operators=ops, schedule=["regulate"])
    H = build(sim, device="cpu")
    H.emit_order = _resolve_emit(sim, H)
    cell = H.level("cell")
    g0 = torch.tensor([0.0, 10.0])                     # one below, one above the fixed point
    cell.state[:, 0] = g0

    op = RegulateConnectionist({**params, "_at": "cell"}, "cpu")
    H.zero_delta()
    delta = op(H, None)["cell"]
    H.add_delta("cell", delta, op.INTEGRAND)
    _integrate(H, sim.dt)
    g_new = cell.get("gene").squeeze(-1)

    # the engine result equals a direct internal self-solve (mean-rate * dt cancels)
    g_end = op._solve(g0[:, None], g0.new_zeros(2, 0), 1.0).squeeze(-1)
    assert torch.allclose(g_new, g_end, atol=1e-6)
    # contraction toward g* = 5.0, on the original side
    assert ((g_new - 5.0).abs() < (g0 - 5.0).abs()).all()
    assert (torch.sign(g_new - 5.0) == torch.sign(g0 - 5.0)).all()


def test_dormant_cells_do_not_evolve():
    """A cell with occ = 0 (a dormant buffer slot) returns a zero gene delta -- the
    occupancy mask gates evolution, so freeing/parking a slot never spuriously regulates."""
    op = RegulateConnectionist({"gamma": 0.1, "_at": "cell"}, "cpu")

    class _Lvl:
        occ = torch.tensor([1.0, 0.0])
        def get(self, _):
            return torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    class _H:
        config = type("C", (), {"dt": 1.0})
        def level(self, _):
            return _Lvl()

    delta = op(_H(), None)["cell"]
    assert torch.allclose(delta[1], torch.zeros(2))    # dormant slot: no change
    assert not torch.allclose(delta[0], torch.zeros(2))
