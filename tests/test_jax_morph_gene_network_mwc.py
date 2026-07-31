"""Property tests for the `regulate:mwc` gene-network operator (candidate).

These check structural invariants of the MWC vector field
`dg/dt = rho*sigmoid(F0 + sum H*ln(1+g/K)) - g/tau` that are stated WITHOUT the
reference (no oracle numbers, no fitted constants):

  * saturation BOUND -- the production term is strictly in (0, rho) for any input;
  * the inert circuit's fixed-point LIMIT (all params zero -> dg = 0.5 - g);
  * the activation/inhibition SIGN convention of H;
  * the restorative-decay ASYMMETRY on a negative concentration.

The differ compares us to the oracle separately; these fix the mechanism, not its numbers.
"""
import torch

import plexus.operators.candidates.jax_morph_gene_network_mwc as gm
from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator
from plexus.models.state import Block, StateSchema, BOUNDARY_FREE, FIRST_ORDER, NONE


def _cell(gene, sensed=None):
    """A minimal cell set carrying a `gene` block (+ optional FIXED `sensed` block)."""
    gene = torch.as_tensor(gene, dtype=torch.float32)
    n_gene = gene.shape[1]
    blocks = [Block("gene", n_gene, integration=FIRST_ORDER, boundary=BOUNDARY_FREE)]
    cols = [gene]
    if sensed is not None:
        sensed = torch.as_tensor(sensed, dtype=torch.float32)
        blocks.append(Block("sensed", sensed.shape[1], integration=NONE, boundary=BOUNDARY_FREE))
        cols.append(sensed)
    lvl = Level("cell", state=torch.cat(cols, dim=1), state_schema=StateSchema(blocks))
    H = Hierarchy()
    H.add_level(lvl)
    return H


def _op(**params):
    params.setdefault("_at", "cell")
    params.setdefault("gene", "gene")
    return get_operator("regulate", "mwc")(params, "cpu")


def test_production_is_bounded_by_rho():
    """THE headline limit: rho*sigmoid(.) is strictly in (0, rho) for ANY genes, drivers,
    and (finite) parameters -- the defining property of saturating production, and what
    separates this from an unbounded linear drive. Reconstruct production = dg + g/tau."""
    torch.manual_seed(0)
    N, n_gene, n_in = 8, 3, 2
    g = torch.randn(N, n_gene)                       # arbitrary, includes negatives
    u = torch.randn(N, n_in)
    log_rho = torch.randn(n_gene)
    log_tau = torch.randn(n_gene)
    op = _op(sensed="sensed",
             log_rho=log_rho.tolist(), log_tau=log_tau.tolist(),
             F0=torch.randn(n_gene).tolist(),
             H_gene=torch.randn(n_gene, n_gene).tolist(),
             log_K_gene=torch.randn(n_gene, n_gene).tolist(),
             H_in=torch.randn(n_gene, n_in).tolist(),
             log_K_in=torch.randn(n_gene, n_in).tolist())
    dg = op(_cell(g, sensed=u), None)["cell"]
    rho = gm._positive_from_log(log_rho)
    tau = gm._positive_from_log(log_tau)
    production = dg + g / tau                         # decay uses raw g, so this is exactly rho*sigmoid(.)
    assert torch.isfinite(dg).all()
    assert (production > 0).all()
    assert (production < rho).all()                  # rho [n_gene] broadcasts over the batch


def test_inert_circuit_relaxes_to_half():
    """Default (all params zero): rho=tau=1, H=0, F0=0, so dg = sigmoid(0) - g = 0.5 - g.
    Every gene has its fixed point at 0.5 and relaxes toward it (a limit + a fixed point)."""
    g = torch.tensor([[0.0, 0.2], [0.5, 0.9], [1.0, 0.3]])
    dg = _op(sensed="sensed")(_cell(g, sensed=torch.zeros(3, 1)), None)["cell"]
    assert torch.allclose(dg, 0.5 - g, atol=1e-6)
    assert abs(float(dg[1, 0])) < 1e-6               # g = 0.5 is stationary


def test_activation_and_inhibition_signs():
    """H > 0 is activating (raises production), H < 0 is inhibitory (lowers it): with a
    positive gene present, a +self-coupling must push dg UP and a -self-coupling DOWN,
    relative to the inert circuit."""
    g = [[1.0]]
    base = _op()(_cell(g), None)["cell"]
    up = _op(H_gene=[[2.0]])(_cell(g), None)["cell"]
    down = _op(H_gene=[[-2.0]])(_cell(g), None)["cell"]
    assert float(up) > float(base) > float(down)


def test_decay_is_restorative_on_a_negative_gene():
    """The intentional asymmetry: occupancy clamps genes to >=0, but DECAY uses the raw
    (negative) state. So a gene at -0.5 gets dg = sigmoid(0) - (-0.5) = 1.0 -- larger than
    the 0.5 production baseline: decay pushes a negative concentration back toward 0."""
    dg = _op()(_cell([[-0.5]]), None)["cell"]
    assert float(dg) > 0.5
    assert torch.allclose(dg, torch.tensor([[1.0]]), atol=1e-6)


def test_dormant_cells_do_not_evolve():
    """A cell with occ=0 (retired slot) returns a zero derivative even though its inert
    dg would be 0.5."""
    H = _cell([[0.0], [0.0]])
    H.level("cell").occ[1] = 0.0
    dg = _op()(H, None)["cell"]
    assert float(dg[0]) != 0.0 and float(dg[1]) == 0.0


def test_engine_integrates_the_gene_block():
    """End-to-end through the real engine: the operator returns dg/dt and the engine
    Euler-integrates the first-order `gene` block (x += dt*delta), exactly the `signal`
    routing. Inert genes start at 0 -> dg/dt = 0.5 -> after one dt=1 step gene = 0.5."""
    from plexus.engine import build, _integrate, _resolve_emit
    from plexus.schema import OpSpec, Selector, Spec

    sets = {"cell": {"n": 2, "state": {
        "gene": {"width": 1, "integration": "first_order", "boundary": "free"},
        "sensed": {"width": 1, "integration": "none", "record": False}}}}
    ops = [OpSpec(op="regulate", impl="mwc", on=Selector("cell"),
                  params={"gene": "gene", "sensed": "sensed"})]
    sim = Spec(name="reg", seed=0, n_frames=1, dt=1.0, sets=sets, fields={},
               operators=ops, schedule=["regulate"])
    H = build(sim, "cpu")
    H.emit_order = _resolve_emit(sim, H)
    op = _op(sensed="sensed")
    H.zero_delta()
    H.add_delta("cell", op(H, None)["cell"], op.INTEGRAND)
    _integrate(H, sim.dt)
    assert torch.allclose(H.level("cell").get("gene").squeeze(-1),
                          torch.tensor([0.5, 0.5]), atol=1e-6)
