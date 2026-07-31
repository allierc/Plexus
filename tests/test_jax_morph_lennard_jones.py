"""Property tests for the `adhere:lennard_jones` operator (candidate; jax-morph LennardJones).

`adhere:lennard_jones` is the 12-6 member of the pairwise cell-cell mechanics family: a
conservative pair energy U(r) = eps((sigma/r)^12 - 2 (sigma/r)^6) (r_min form) over live non-self
cell pairs, its adhesive tail smoothly truncated on [r_onset_frac, r_cutoff_frac]*sigma, whose
force = -grad(energy) is emitted as an overdamped velocity. sigma = r_i + r_j is the size-consistent
contact distance. Every assertion below is stated WITHOUT the reference -- a limit, a sign, a
symmetry, a conservation law -- so none can be satisfied by fitting the oracle's numbers:

  * WELL AT CONTACT (the headline / r_min discriminator) -- two cells separated by exactly
    sigma = r_i + r_j feel ZERO force (the well minimum sits at contact, NOT at 2^(1/6) sigma as
    the textbook 4-eps form would place it);
  * STABLE EQUILIBRIUM -- inside contact the pair repels (pushed apart), just outside it adheres
    (pulled together), so contact is a stable rest separation;
  * HARD CORE -- the repulsive force strengthens monotonically as the overlap grows;
  * NEWTON'S THIRD LAW -- a two-body pair feels equal and opposite forces;
  * MOMENTUM CONSERVATION -- over any configuration the net force sums to zero (a conservative
    pairwise energy: no self-propulsion);
  * SMOOTH CUTOFF -- beyond r_cutoff_frac*sigma the force is exactly zero;
  * SIZE CONSISTENCY -- doubling both radii doubles the rest separation (sigma tracks cell size);
  * ALIVE / MASK MASKING -- a dead (occ = 0) or masked-out cell emits a zero velocity;
  * POSITION IS NOT MUTATED -- the step returns a delta and never Euler-steps pos itself.
"""
import math
import types

import torch

import plexus.operators.candidates.jax_morph_lennard_jones as m  # noqa: F401  registers adhere:lennard_jones
from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator


def _world(pos, radius, *, occ=None, dt=1.0, seed=0):
    """A one-set cell world: positions [n, dim], a per-cell `radius` buffer, occupancy, config.dt."""
    pos = torch.as_tensor(pos, dtype=torch.float32)
    n, dim = pos.shape
    occ = torch.ones(n) if occ is None else torch.as_tensor(occ, dtype=torch.float32)
    lvl = Level("cell", state=pos.clone(), state_schema={"pos": (0, dim)}, occ=occ)
    lvl.register_buffer("radius", torch.as_tensor(radius, dtype=torch.float32))
    H = Hierarchy()
    H.add_level(lvl)
    H.dim = dim
    H.periodic = False                                    # unbounded: minimum_image is a no-op
    H.config = types.SimpleNamespace(dt=dt)
    H.rng = torch.Generator().manual_seed(seed)
    return H, lvl


def _op(**params):
    params.setdefault("_at", "cell")
    return get_operator("adhere", implementation="lennard_jones")(params, "cpu")


def _vel(pos, radius, *, occ=None, mask=None, **params):
    """Run one step and return the emitted velocity delta [n, dim] (detached)."""
    H, lvl = _world(pos, radius, occ=occ)
    with torch.no_grad():
        out = _op(**params)(H, mask)
    return out["cell"].detach()


def _pair(d, r0=0.5, r1=0.5, **params):
    """Velocity delta for a two-cell pair along x at separation `d` (cell 0 at origin)."""
    return _vel([[0.0, 0.0], [d, 0.0]], [r0, r1], **params)


def test_well_minimum_is_at_contact_not_textbook_sigma():
    """THE headline / r_min discriminator. The well minimum is at the contact distance
    sigma = r_i + r_j, where the force is exactly zero -- NOT at 2^(1/6) sigma (where the textbook
    4-eps form ((sigma/r)^12 - (sigma/r)^6) places its minimum). At contact the force vanishes; at
    2^(1/6) sigma (the wrong form's rest point) it is clearly nonzero (still adhesive)."""
    sigma = 1.0                                                   # r0 + r1 = 0.5 + 0.5
    v_contact = _pair(sigma, epsilon=1.0)
    assert torch.allclose(v_contact, torch.zeros_like(v_contact), atol=1e-5)   # zero force at contact
    v_textbook = _pair(2.0 ** (1.0 / 6.0) * sigma, epsilon=1.0)  # where the 4-eps form's well sits
    assert v_textbook[1, 0].abs() > 1e-3                          # our r_min form is NOT at rest there


def test_contact_is_a_stable_equilibrium():
    """Contact is a STABLE rest separation: compressed (r < sigma) the pair repels -- cell 1 is
    pushed to +x, away from cell 0 -- and stretched just past contact (sigma < r < r_onset*sigma)
    it adheres, cell 1 pulled back toward cell 0 in -x. The restoring sign flips across contact."""
    inside = _pair(0.9, epsilon=1.0)                             # r < sigma -> repulsion
    outside = _pair(1.2, epsilon=1.0)                            # sigma < r < 1.5 sigma -> adhesion
    assert inside[1, 0] > 0.0                                    # cell 1 pushed away (+x)
    assert outside[1, 0] < 0.0                                   # cell 1 pulled back (-x)
    # equal-and-opposite within each pair (checked in full below), so cell 0 mirrors cell 1
    assert inside[0, 0] < 0.0 and outside[0, 0] > 0.0


def test_repulsive_core_strengthens_with_overlap():
    """The r^-12 hard core: as two cells overlap more (r decreasing below sigma) the repulsive
    force grows monotonically -- a soft touch is a weak push, a deep overlap a hard one."""
    f = [float(_pair(d, epsilon=1.0)[1, 0]) for d in (0.95, 0.90, 0.85, 0.80)]
    assert all(fi > 0.0 for fi in f)                             # all repulsive
    assert f[0] < f[1] < f[2] < f[3]                            # strengthens as overlap grows


def test_newton_third_law_two_body():
    """A two-body interaction obeys Newton's third law: the forces are equal and opposite (the
    energy depends only on the separation, so its gradient is antisymmetric across the pair)."""
    for d in (0.85, 1.0, 1.3, 2.0):
        v = _pair(d, epsilon=1.0)
        assert torch.allclose(v[0], -v[1], atol=1e-6)


def test_total_force_conserves_momentum():
    """A conservative pairwise energy exerts NO net force on the system: summed over all cells the
    velocity is zero (uniform mobility, no self-propulsion). A jittered grid (contact-scale spacing,
    no deep overlaps that would blow the r^-12 core past float32's cancellation headroom)."""
    torch.manual_seed(0)
    gx, gy = torch.meshgrid(torch.arange(4.0), torch.arange(3.0), indexing="ij")
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1) * 1.3    # spacing 1.3 ~ contact scale
    pos = pos + 0.15 * torch.randn_like(pos)                            # generic (not a lattice fixed point)
    radius = torch.full((pos.shape[0],), 0.5)                          # sigma = 1.0
    v = _vel(pos, radius, epsilon=1.5)
    assert v.abs().max() > 0.0                                          # the cells genuinely interact
    assert v.sum(dim=0).norm() < 1e-4 * v.norm()                       # net force ~ 0 (float roundoff only)


def test_smooth_cutoff_beyond_r_off_is_zero():
    """The sigma-relative smooth cutoff truncates the tail: beyond r_cutoff_frac*sigma the force is
    EXACTLY zero, while inside the transition window (r_onset < r < r_cutoff) the adhesive tail is
    still felt (small but nonzero)."""
    inside_window = _pair(2.0, epsilon=1.0)                      # r_on=1.5 < 2.0 < r_off=2.5
    beyond = _pair(2.6, epsilon=1.0)                            # r > r_off = 2.5 sigma
    assert inside_window[1, 0].abs() > 0.0                       # tail still acts (adhesive)
    assert inside_window[1, 0] < 0.0                            # ... and it is attractive
    assert torch.allclose(beyond, torch.zeros_like(beyond), atol=1e-7)


def test_cutoff_fraction_ordering_is_checked():
    """Faithful to the source's construction-time guard: r_onset_frac must be < r_cutoff_frac for a
    valid smooth-cutoff window; an inverted window raises rather than silently misbehaving."""
    try:
        _op(r_onset_frac=2.5, r_cutoff_frac=1.5)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_size_consistency_rest_separation_tracks_radius():
    """sigma = r_i + r_j, so the rest separation tracks cell SIZE with no range knob to retune.
    Doubling both radii (0.5 -> 1.0, sigma 1.0 -> 2.0) moves the zero-force contact from r = 1.0 to
    r = 2.0; at the old contact r = 1.0 the now-larger cells strongly repel."""
    v_new_contact = _pair(2.0, r0=1.0, r1=1.0, epsilon=1.0)     # r = sigma = 2.0 -> equilibrium
    assert torch.allclose(v_new_contact, torch.zeros_like(v_new_contact), atol=1e-5)
    v_old_contact = _pair(1.0, r0=1.0, r1=1.0, epsilon=1.0)     # r = 1.0 = 0.5 sigma -> deep overlap
    assert v_old_contact[1, 0] > 0.0                            # strong repulsion at the old rest point


def test_dead_and_masked_cells_emit_nothing():
    """A dead cell (occ = 0) exerts and feels no force, and the `at:` selector mask gates who moves:
    a masked-out live cell is left untouched. Cell 1 is dead, cell 2 is masked out."""
    pos = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.9, 0.9]]
    radius = [0.5, 0.5, 0.5, 0.5]
    mask = torch.tensor([True, True, False, True])              # cell 2 masked out (though live)
    v = _vel(pos, radius, occ=[1, 0, 1, 1], mask=mask, epsilon=1.0)
    assert torch.count_nonzero(v[1]) == 0                       # dead cell: no velocity
    assert torch.count_nonzero(v[2]) == 0                       # masked-out live cell: no velocity
    assert torch.count_nonzero(v[0]) > 0                        # live, masked-in, has a live neighbour


def test_step_does_not_mutate_position():
    """The interaction returns a velocity DELTA and never integrates pos itself (the engine does):
    positions are unchanged by a forward pass. This is what the frame-0 integration guard checks."""
    H, lvl = _world([[0.0, 0.0], [0.9, 0.0]], [0.5, 0.5])
    before = lvl.get("pos").clone()
    with torch.no_grad():
        _op(epsilon=1.0)(H, None)
    assert torch.equal(lvl.get("pos"), before)


def test_dimension_generic_3d():
    """The energy is dimension-generic: a 3-D pair rests at contact (zero force) and repels when
    compressed, with no special-casing of the spatial dimension."""
    contact = _vel([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.5, 0.5], epsilon=1.0)
    assert torch.allclose(contact, torch.zeros_like(contact), atol=1e-5)
    overlap = _vel([[0.0, 0.0, 0.0], [0.85, 0.0, 0.0]], [0.5, 0.5], epsilon=1.0)
    assert overlap[1, 0] > 0.0 and math.isclose(float(overlap[0, 0]), -float(overlap[1, 0]), abs_tol=1e-6)
