"""Property tests for the `morphogen:free_space_greens_function` operator (candidate).

These fix STRUCTURAL invariants of the steady screened-diffusion field
c_i = sum_j alive_j G(r_ij, a_j) S_j, each stated WITHOUT the reference (no oracle
numbers, no fitted constants -- the differ compares us to the oracle separately):

  * SUPERPOSITION -- the map S -> c is linear (the defining property of a Green's-function
    steady-state solve): c(alpha S) = alpha c(S) and c(S1 + S2) = c(S1) + c(S2);
  * SIGN -- non-negative sources give a non-negative field (the kernel G is positive);
  * SCREENING -- raising the degradation K shortens the range (a distant cell reads less);
  * SELF-FIELD -- the i == j diagonal contributes: a lone source reads its own secretion;
  * ALIVE-MASKING (twice) -- a dead cell neither emits nor carries a field;
  * the FIELD SOLVE does not move cells (pos is invariant -- the integration guard).
"""
import torch

import plexus.operators.candidates.jax_morph_free_screened_diffusion as m  # noqa: F401  registers `morphogen`
from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator
from plexus.models.state import (
    Block, StateSchema, BOUNDARY_FREE, BOUNDARY_WORLD, NONE, SECOND_ORDER_COORDINATE,
)


def _cell(pos, radius, secretion):
    """A minimal cell set carrying pos + a per-cell radius, secretion_rate, chemical block."""
    pos = torch.as_tensor(pos, dtype=torch.float32)
    N, D = pos.shape
    rad = torch.as_tensor(radius, dtype=torch.float32).reshape(N, 1)
    S = torch.as_tensor(secretion, dtype=torch.float32)
    if S.dim() == 1:
        S = S[:, None]
    ns = S.shape[1]
    blocks = [
        Block("pos", D, role="coordinate", integration=SECOND_ORDER_COORDINATE, boundary=BOUNDARY_WORLD),
        Block("radius", 1, integration=NONE, boundary=BOUNDARY_FREE),
        Block("secretion_rate", ns, integration=NONE, boundary=BOUNDARY_FREE),
        Block("chemical", ns, integration=NONE, boundary=BOUNDARY_FREE),
    ]
    state = torch.cat([pos, rad, S, torch.zeros(N, ns)], dim=1)
    lvl = Level("cell", state=state, state_schema=StateSchema(blocks))
    H = Hierarchy()
    H.add_level(lvl)
    return H


def _op(**params):
    params.setdefault("_at", "cell")
    return get_operator("morphogen", "free_space_greens_function")(params, "cpu")


def _run(pos, radius, secretion, **params):
    """Run the operator once on a fresh set and return the written `chemical` block [N, ns]."""
    H = _cell(pos, radius, secretion)
    _op(**params)(H, None)
    return H.level("cell").get("chemical")


def test_superposition_is_linear():
    """THE headline property: the steady-state solve S -> c is LINEAR (a Green's-function
    superposition), so scaling every source scales the field and two source patterns add.
    Kernel-independent, so tested in 2-D (exercises the modified-Bessel K0/K1 path)."""
    torch.manual_seed(0)
    pos = torch.rand(6, 2)
    radius = torch.full((6,), 0.05)
    S1 = torch.rand(6, 1)
    S2 = torch.rand(6, 1)
    D, K = 1.0, 1.0                                   # screened (K > 0 required in 2-D)
    c1 = _run(pos, radius, S1, diffusion=D, degradation=K)
    c2 = _run(pos, radius, S2, diffusion=D, degradation=K)
    c_scaled = _run(pos, radius, 3.0 * S1, diffusion=D, degradation=K)
    c_sum = _run(pos, radius, S1 + S2, diffusion=D, degradation=K)
    assert torch.isfinite(c1).all()
    assert torch.allclose(c_scaled, 3.0 * c1, atol=1e-5)        # homogeneity
    assert torch.allclose(c_sum, c1 + c2, atol=1e-5)            # additivity


def test_nonnegative_sources_give_nonnegative_field():
    """Sign: with S >= 0, D > 0, K >= 0 the Green's function is positive, so the whole field
    is >= 0. A morphogen concentration is never negative. Tested in 3-D (K = 0 admitted there)."""
    torch.manual_seed(1)
    pos = torch.rand(8, 3)
    radius = torch.full((8,), 0.03)
    S = torch.rand(8, 1)                              # non-negative sources
    c = _run(pos, radius, S, diffusion=1.0, degradation=0.5)
    assert torch.isfinite(c).all()
    assert (c >= 0).all()
    assert (c > 0).any()                             # not trivially zero


def test_screening_shortens_the_range():
    """Limit / monotonicity: a stronger degradation K (larger inverse screening length
    kappa = sqrt(K/D)) makes a DISTANT cell read LESS of a source. Two cells: a unit source
    and a silent receiver one unit away; the receiver's field falls as K rises."""
    pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    radius = [0.05, 0.05]
    S = [[1.0], [0.0]]                               # cell 0 emits, cell 1 is silent
    weak = _run(pos, radius, S, diffusion=1.0, degradation=0.25)
    strong = _run(pos, radius, S, diffusion=1.0, degradation=4.0)
    assert float(weak[1, 0]) > float(strong[1, 0]) > 0.0        # receiver reads less under stronger screening


def test_self_field_is_included():
    """The i == j diagonal contributes (r_eff = max(0, a) = a): a lone, isolated source reads
    its OWN secretion -- unlike a neighbour sum that drops the self term."""
    c = _run([[0.5, 0.5, 0.5]], [0.05], [[1.0]], diffusion=1.0, degradation=1.0)
    assert float(c[0, 0]) > 0.0


def test_dead_cells_neither_emit_nor_carry():
    """Alive-masking applied TWICE: a dead SOURCE (occ=0) emits nothing onto the others, and a
    dead RECEIVER carries zero field -- even with a large secretion rate on the dead slot."""
    pos = [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]
    radius = [0.05, 0.05, 0.05]
    S = [[10.0], [0.0], [0.0]]                       # cell 0 has a big source rate
    H = _cell(pos, radius, S)
    H.level("cell").occ[0] = 0.0                     # ...but is dead
    _op(diffusion=1.0, degradation=1.0)(H, None)
    c = H.level("cell").get("chemical")
    assert float(c[0, 0]) == 0.0                     # dead receiver holds zero
    assert float(c[1, 0]) == 0.0 and float(c[2, 0]) == 0.0   # dead source emits nothing
    # sanity: the SAME configuration with cell 0 alive DOES drive the others
    live = _run(pos, radius, S, diffusion=1.0, degradation=1.0)
    assert float(live[1, 0]) > 0.0 and float(live[2, 0]) > 0.0


def test_field_solve_does_not_move_cells():
    """The quasistatic solve writes only the `chemical` block: positions are invariant (it is a
    derived readout, not a force). This is what the engine's frame-0 integration guard checks."""
    torch.manual_seed(2)
    pos = torch.rand(5, 2)
    H = _cell(pos, torch.full((5,), 0.05), torch.rand(5, 1))
    before = H.level("cell").get("pos").clone()
    _op(diffusion=1.0, degradation=1.0)(H, None)
    assert torch.equal(H.level("cell").get("pos"), before)


def test_multi_species_are_independent():
    """Per-species D/K vmap: two channels with different coefficients are solved independently
    (channel 0's field is unchanged by channel 1's sources)."""
    torch.manual_seed(3)
    pos = torch.rand(5, 3)
    radius = torch.full((5,), 0.04)
    S = torch.rand(5, 2)
    both = _run(pos, radius, S, diffusion=[1.0, 2.0], degradation=[0.5, 1.5])
    ch0_only = _run(pos, radius, torch.stack([S[:, 0], torch.zeros(5)], dim=1),
                    diffusion=[1.0, 2.0], degradation=[0.5, 1.5])
    assert torch.allclose(both[:, 0], ch0_only[:, 0], atol=1e-6)   # channel 0 independent of channel 1's sources
    assert not torch.allclose(both[:, 1], ch0_only[:, 1])          # channel 1 differs (its sources were zeroed)
