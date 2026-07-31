"""Property tests for the `adhere` operator, `harmonic` implementation (candidate; jax-morph
Harmonic pair potential).

`adhere:harmonic` is a finite-range shifted harmonic spring over live cell pairs: a repulsive
core (r < sigma) and an adhesive tail (sigma < r < r_c), with the per-pair contact distance
sigma = r_i + r_j read off the physical `radius`. It emits the overdamped drift velocity
v = mobility * F, F(r) = k*(sigma - r) truncated at r_c = r_cutoff_frac * sigma.

Every assertion is stated WITHOUT the reference -- a conservation law, a sign, a limit at
contact, a symmetry, the energy/force relation -- so none can be met by fitting the oracle's
numbers:

  * ENERGY-DEFINED FORCE (the contract property) -- the emitted velocity equals -grad of the
    operator's own total_energy w.r.t. position (force = -grad U), to autodiff precision;
  * ZERO AT CONTACT -- a pair exactly at r = sigma feels no force (the mechanical rest state);
  * THREE REGIMES + HARD CUTOFF -- compressed pairs repel, stretched (sigma < r < r_c) pairs
    adhere, and pairs at r >= r_c feel exactly zero (the C0 truncation);
  * MOMENTUM CONSERVATION -- a central pairwise force sums to zero over the whole cluster
    (Newton's third law), so a translation-free cluster has no net drift;
  * DEAD / MASKED cells neither move nor exert a force (occ + `at:` masking);
  * POSITION IS NOT MUTATED -- forward returns a delta and never Euler-steps pos itself.
"""
import math
import types

import torch

import plexus.operators.candidates.jax_morph_harmonic as m  # noqa: F401  registers `adhere:harmonic`
from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator


def _world(pos, *, radius=0.5, occ=None, dt=1.0, requires_grad=False):
    """A one-set cell world carrying a pos block, a per-cell `radius` buffer, and occupancy.

    `pos` is an [N, D] tensor / nested list; `radius` a scalar or [N] tensor."""
    pos = torch.as_tensor(pos, dtype=torch.float32)
    n, dim = pos.shape
    state = pos.clone()
    if requires_grad:
        state.requires_grad_(True)
    occ = torch.ones(n) if occ is None else torch.as_tensor(occ, dtype=torch.float32)
    lvl = Level("cell", state=state, state_schema={"pos": (0, dim)}, occ=occ)
    rad = torch.as_tensor(radius, dtype=torch.float32)
    if rad.ndim == 0:
        rad = rad.expand(n).clone()
    lvl.register_buffer("radius", rad)
    H = Hierarchy()
    H.add_level(lvl)
    H.dim = dim
    H.config = types.SimpleNamespace(dt=dt)
    H.rng = torch.Generator().manual_seed(0)
    return H, lvl


def _op(**params):
    params.setdefault("_at", "cell")
    return get_operator("adhere", "harmonic")(params, "cpu")


def _velocity(pos, *, mask=None, **params):
    H, lvl = _world(pos, radius=params.pop("radius", 0.5), occ=params.pop("occ", None))
    return _op(**params)(H, mask)["cell"]


def test_force_is_negative_gradient_of_energy():
    """The contract property: `adhere` is energy-defined, and the emitted drift is the negative
    position-gradient of its own total_energy. Build one grad-tracking world, differentiate the
    energy, and compare to the operator's analytic force (mobility = 1 so velocity = force). A
    non-trivial cluster (compressed, stretched, and beyond-cutoff pairs all present)."""
    torch.manual_seed(1)
    pos = torch.rand(24, 2) * 3.0                       # box ~3 = 3*sigma: a mix of all three regimes
    rad = 0.3 + 0.4 * torch.rand(24)                    # heterogeneous radii -> per-pair sigma
    kfield = 0.5 + torch.rand(24)                       # per-cell stiffness -> the arithmetic-mean mix

    # analytic force from forward (no grad needed)
    H, lvl = _world(pos, radius=rad)
    lvl.register_buffer("kcell", kfield.clone())
    force = _op(mobility=1.0, k_field="kcell")(H, None)["cell"]

    # -grad(total_energy) w.r.t. position on an identical grad-tracking world
    Hg, lvlg = _world(pos, radius=rad, requires_grad=True)
    lvlg.register_buffer("kcell", kfield.clone())
    E = _op(mobility=1.0, k_field="kcell").total_energy(Hg)
    (grad,) = torch.autograd.grad(E, lvlg.state)
    assert torch.allclose(force, -grad, atol=1e-4), (force - (-grad)).abs().max()


def test_zero_force_at_contact():
    """At r = sigma the parabola is at its minimum: the pair force is exactly zero (the rest
    state). Two unit-sigma cells placed exactly sigma apart feel no drift."""
    v = _velocity([[0.0, 0.0], [1.0, 0.0]], radius=0.5)   # sigma = 1.0, r = 1.0
    assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)


def test_three_regimes_and_hard_cutoff():
    """Repulsion when compressed, adhesion when stretched, and exactly zero past the cutoff.
    sigma = 1.0, r_c = 2.5*sigma = 2.5. Cell 1 sits at (r, 0), so its x-velocity sign reports the
    regime: + is apart (repel), - is together (adhere)."""
    # compressed r = 0.8 < sigma -> repel: cell 1 pushed +x, cell 0 pushed -x, equal & opposite
    v = _velocity([[0.0, 0.0], [0.8, 0.0]], radius=0.5)
    assert v[1, 0] > 1e-6 and v[0, 0] < -1e-6
    assert math.isclose(float(v[1, 0]), float(-v[0, 0]), rel_tol=1e-5)
    # stretched sigma < r = 1.5 < r_c -> adhere: cell 1 pulled -x, cell 0 pulled +x
    v = _velocity([[0.0, 0.0], [1.5, 0.0]], radius=0.5)
    assert v[1, 0] < -1e-6 and v[0, 0] > 1e-6
    # beyond r = 3.0 > r_c = 2.5 -> exactly zero (C0 hard truncation)
    v = _velocity([[0.0, 0.0], [3.0, 0.0]], radius=0.5)
    assert torch.allclose(v, torch.zeros_like(v), atol=1e-8)


def test_momentum_conservation():
    """A conservative central pairwise force obeys Newton's third law, so the total force over
    the cluster is zero -- no self-propulsion, no net drift. Holds for any positions / radii /
    per-cell stiffness (heterogeneous here) because the pair force on i is minus that on j."""
    torch.manual_seed(2)
    pos = torch.rand(50, 2) * 2.5
    rad = 0.3 + 0.5 * torch.rand(50)
    v = _velocity(pos, radius=rad, k=1.7)
    assert torch.allclose(v.sum(dim=0), torch.zeros(2), atol=1e-4), v.sum(dim=0)


def test_dead_and_masked_cells_neither_move_nor_push():
    """A dead cell (occ = 0) draws no drift AND exerts no force on its neighbours; the `at:`
    mask likewise gates who is driven. Compare a live pair to the same pair with a THIRD cell
    that is dead (must be inert) vs alive-but-masked-out (still exerts force -- masking gates the
    ACTOR, not the source)."""
    base = [[0.0, 0.0], [0.8, 0.0]]                       # a compressed live pair
    v_pair = _velocity(base, radius=0.5)

    # add a nearby DEAD cell: it must not change the live pair's drift and must not move itself
    pos3 = base + [[0.4, 0.3]]
    v_dead = _velocity(pos3, radius=0.5, occ=[1, 1, 0])
    assert torch.allclose(v_dead[:2], v_pair, atol=1e-6)  # dead cell exerts no force
    assert torch.count_nonzero(v_dead[2]) == 0            # dead cell does not move

    # same geometry, cell 2 ALIVE but masked out of the actor set: it is not driven itself, yet
    # it still SOURCES a force on the live pair (so their drift changes vs the dead case).
    H, lvl = _world(pos3, radius=0.5)
    mask = torch.tensor([True, True, False])
    v_masked = _op(k=1.0)(H, mask)["cell"]
    assert torch.count_nonzero(v_masked[2]) == 0          # masked-out cell not driven
    assert not torch.allclose(v_masked[:2], v_dead[:2], atol=1e-6)  # but it still pushes the pair


def test_step_does_not_mutate_position():
    """forward returns a velocity DELTA and never integrates pos itself (the engine does that):
    the frame-0 integration guard would flag any in-place pos write."""
    H, lvl = _world([[0.0, 0.0], [0.8, 0.0], [1.6, 0.2]], radius=0.5)
    before = lvl.get("pos").clone()
    _op(k=1.0)(H, None)
    assert torch.equal(lvl.get("pos"), before)


def test_cutoff_must_sit_beyond_contact():
    """Faithful construction check: r_cutoff_frac <= 1 leaves no finite interaction range (the
    down-shift is ill-defined); the source raises, and so must this."""
    raised = False
    try:
        _op(r_cutoff_frac=1.0)
    except ValueError:
        raised = True
    assert raised


def test_dimension_generic_3d():
    """The pairwise force is dimension-generic (reads D = pos.shape[-1]): it runs in 3-D, still
    conserves momentum, and still repels a compressed pair along the separation axis."""
    torch.manual_seed(3)
    pos = torch.rand(30, 3) * 2.5
    v = _velocity(pos, radius=0.4, k=1.0)
    assert v.shape == (30, 3)
    assert torch.allclose(v.sum(dim=0), torch.zeros(3), atol=1e-4)
    # a compressed pair on the z-axis pushes apart along z
    v2 = _velocity([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]], radius=0.5)
    assert v2[1, 2] > 1e-6 and v2[0, 2] < -1e-6
