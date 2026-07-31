"""Property tests for the `cell_divide:volume_conserving` operator (jax-morph Division step).

These assert properties statable WITHOUT the reference -- the Bernoulli-hazard limits, the >=0 clip,
and the geometric/conservation laws the refinement adds over the isotropic default:

* VOLUME CONSERVATION -- the two daughters conserve the mother's d-volume: r_mother = r_daughter =
  r*m with m = 2^(-1/d), so r_mother^d + r_daughter^d == r_old^d (each daughter half the volume).
* JUST-TOUCHING GEOMETRY -- the daughters sit exactly touching: centre distance = sum of radii.
* CAPACITY IS A HARD WALL -- surplus dividers past a full buffer are dropped and accumulated into the
  GLOBAL division_overflow counter, and occupancy never exceeds the buffer (no crash).
* LINEAGE -- a woken daughter slot records born=1 and mother=parent-slot; both reset each step.
* ORIENTED LIMIT -- at a large orientation_snr with a fixed axis the split aligns with that axis.

None of these check agreement with the oracle -- they test the operator's contract.
"""
import math
import types

import torch

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator

import plexus.operators.candidates.jax_morph_division  # noqa: F401  (registers cell_divide:volume_conserving)


def _world(n, rate, *, dt=1.0, seed=0, occ=None, radius=0.5, axis=None, dim=2):
    """A one-set world: `n` cells with pos+vel state and per-cell radius / division_rate (and an
    optional division_axis) buffers. `rate`/`radius` may be scalar (uniform) or length-n sequences."""
    state = torch.zeros(n, 2 * dim)                                   # pos(dim) + vel(dim)
    schema = {"pos": (0, dim), "vel": (dim, 2 * dim)}
    occ = torch.ones(n) if occ is None else torch.as_tensor(occ, dtype=torch.float32)
    lvl = Level("cell", state=state, state_schema=schema, occ=occ)
    dr = torch.full((n,), float(rate)) if isinstance(rate, (int, float)) else torch.as_tensor(rate, dtype=torch.float32)
    rr = torch.full((n,), float(radius)) if isinstance(radius, (int, float)) else torch.as_tensor(radius, dtype=torch.float32)
    lvl.register_buffer("division_rate", dr)
    lvl.register_buffer("radius", rr)
    if axis is not None:
        lvl.register_buffer("division_axis", torch.as_tensor(axis, dtype=torch.float32))
    H = Hierarchy()
    H.add_level(lvl)
    H.dim = dim
    H.config = types.SimpleNamespace(dt=dt)
    H.rng = torch.Generator().manual_seed(seed)
    return H, lvl


def _op(params=None):
    return get_operator("cell_divide", "volume_conserving")((params or {}), "cpu")


def test_zero_rate_is_a_noop():
    """division_rate = 0 -> p = 0 -> nobody divides: occupancy unchanged, no lineage, no overflow."""
    H, lvl = _world(8, rate=0.0, occ=[1, 1, 1, 1, 0, 0, 0, 0])
    before = lvl.occ.clone()
    out = _op()(H, lvl.active)
    assert out == {}                                                 # structural: returns no delta
    assert torch.equal(lvl.occ, before)                             # nobody woken
    assert torch.count_nonzero(lvl.born) == 0                       # no births recorded
    assert torch.all(lvl.mother == -1)                             # no parent recorded
    assert float(lvl.division_overflow) == 0.0                     # nothing dropped


def test_negative_rate_is_clipped_to_no_division():
    """A negative division_rate (e.g. an unconstrained controller output) clips to 0, so p = 0 and
    nobody divides -- rather than a negative probability that would NaN a score."""
    H, lvl = _world(8, rate=-5.0, occ=[1, 1, 1, 1, 0, 0, 0, 0])
    before = lvl.occ.clone()
    _op()(H, lvl.active)
    assert torch.equal(lvl.occ, before)
    assert torch.count_nonzero(lvl.born) == 0


def test_volume_is_conserved():
    """A single certain divider halves its volume into two equal daughters: r_mother = r_daughter =
    r_old * 2^(-1/d), so r_mother^d + r_daughter^d == r_old^d (dimension-dependent factor)."""
    for dim in (2, 3):
        rates = [1e3] + [0.0] * 7                                    # only cell 0 divides
        H, lvl = _world(8, rate=rates, radius=0.5, occ=[1, 1, 1, 1, 0, 0, 0, 0], dim=dim)
        r_old = float(lvl.radius[0])
        _op()(H, lvl.active)
        daughter = int((lvl.born > 0.5).nonzero(as_tuple=True)[0].item())
        m = 2.0 ** (-1.0 / dim)
        assert math.isclose(float(lvl.radius[0]), r_old * m, rel_tol=1e-5)      # mother halved
        assert math.isclose(float(lvl.radius[daughter]), r_old * m, rel_tol=1e-5)  # daughter equal
        vol = float(lvl.radius[0]) ** dim + float(lvl.radius[daughter]) ** dim
        assert math.isclose(vol, r_old ** dim, rel_tol=1e-5)                    # d-volume conserved


def test_daughters_are_just_touching():
    """The offset uses the NEW radius, so the two daughters end exactly touching: the centre
    distance equals the sum of the two (equal) daughter radii, = 2 * r_old * m."""
    rates = [1e3] + [0.0] * 7
    H, lvl = _world(8, rate=rates, radius=0.4, occ=[1, 1, 1, 1, 0, 0, 0, 0], seed=2)
    r_old = float(lvl.radius[0])
    _op()(H, lvl.active)
    daughter = int((lvl.born > 0.5).nonzero(as_tuple=True)[0].item())
    x_m = lvl.state[0, 0:2]
    x_d = lvl.state[daughter, 0:2]
    centre_dist = float((x_m - x_d).norm())
    sum_radii = float(lvl.radius[0] + lvl.radius[daughter])
    assert math.isclose(centre_dist, sum_radii, rel_tol=1e-5)                   # just touching
    assert math.isclose(centre_dist, 2.0 * r_old * 2.0 ** (-0.5), rel_tol=1e-5) # = 2 r m in 2D


def test_daughters_centred_on_the_mother():
    """Mother -> x+offset and daughter -> x-offset, so the pair's midpoint is the mother's
    pre-division centre (a symmetric split)."""
    rates = [1e3] + [0.0] * 7
    H, lvl = _world(8, rate=rates, occ=[1, 1, 1, 1, 0, 0, 0, 0], seed=5)
    x0 = lvl.state[0, 0:2].clone()                                              # mother starts at origin (zeros)
    _op()(H, lvl.active)
    daughter = int((lvl.born > 0.5).nonzero(as_tuple=True)[0].item())
    mid = 0.5 * (lvl.state[0, 0:2] + lvl.state[daughter, 0:2])
    assert torch.allclose(mid, x0, atol=1e-6)


def test_lineage_recorded_and_occupancy_grows():
    """Each committed divider wakes a dormant slot; that slot records born=1 and mother=parent-slot,
    with born=0 / mother=-1 everywhere else. Live count increases by the number of divisions."""
    rates = [1e3, 1e3, 0, 0, 0, 0, 0, 0]                                        # two certain dividers
    H, lvl = _world(8, rate=rates, occ=[1, 1, 1, 1, 0, 0, 0, 0])
    n_live_before = int((lvl.occ > 0).sum())
    _op()(H, lvl.active)
    daughters = (lvl.born > 0.5).nonzero(as_tuple=True)[0]
    assert daughters.numel() == 2                                              # two births
    assert int((lvl.occ > 0).sum()) == n_live_before + 2                       # live count up by 2
    for s in daughters.tolist():
        parent = int(lvl.mother[s])
        assert parent in (0, 1)                                                # parent is one of the dividers
        assert float(lvl.occ[s]) == 1.0                                        # daughter slot woken
    others = torch.ones(8, dtype=torch.bool)
    others[daughters] = False
    assert torch.all(lvl.mother[others] == -1)                                # no parent recorded elsewhere
    assert torch.all(lvl.born[others] == 0.0)


def test_capacity_overflow_is_capped_and_accumulated():
    """When more cells divide than there are free slots, the surplus dividers are DROPPED (never a
    crash), occupancy never exceeds the buffer, and the dropped count accumulates into the GLOBAL
    division_overflow across macro-steps."""
    # 6 certain dividers, only 2 free slots -> 2 commit, 4 drop.
    H, lvl = _world(8, rate=1e3, occ=[1, 1, 1, 1, 1, 1, 0, 0])
    _op()(H, lvl.active)
    assert int((lvl.occ > 0).sum()) == 8                                       # buffer filled, not exceeded
    assert float(lvl.division_overflow) == 4.0                                 # 6 dividers - 2 slots
    # a second step: now the buffer is full (0 free), all 8 dividers drop -> overflow accumulates.
    _op()(H, lvl.active)
    assert int((lvl.occ > 0).sum()) == 8                                       # still no over-fill
    assert float(lvl.division_overflow) == 12.0                               # 4 + 8, a running counter


def test_oriented_split_aligns_with_the_axis():
    """At a large orientation_snr the placement direction collapses onto the per-cell division_axis:
    the mother->daughter separation is parallel to the axis (its perpendicular component ~ 0)."""
    n = 8
    axis = torch.zeros(n, 2)
    axis[0, 0] = 1.0                                                            # cell 0 splits along +x
    rates = [1e3] + [0.0] * (n - 1)
    H, lvl = _world(n, rate=rates, radius=0.5, occ=[1, 1, 1, 1, 0, 0, 0, 0],
                    axis=axis, seed=1)
    op = _op({"orientation_snr": 1e3})
    op(H, lvl.active)
    daughter = int((lvl.born > 0.5).nonzero(as_tuple=True)[0].item())
    sep = (lvl.state[0, 0:2] - lvl.state[daughter, 0:2])                        # separation vector
    # essentially all of the separation is along x; the y (perpendicular) component is ~0.
    assert abs(float(sep[0])) > 1e-3
    assert abs(float(sep[1])) / abs(float(sep[0])) < 1e-2


def test_isotropic_when_no_axis_and_no_snr():
    """orientation_snr = 0 (and no division_axis field) -> pure isotropic placement still divides
    and conserves volume: the refinement is additive, a zero axis is a safe no-op for orientation."""
    rates = [1e3] + [0.0] * 7
    H, lvl = _world(8, rate=rates, radius=0.5, occ=[1, 1, 1, 1, 0, 0, 0, 0], seed=3)
    r_old = float(lvl.radius[0])
    _op()(H, lvl.active)                                                        # snr defaults to 0, no axis buffer
    daughter = int((lvl.born > 0.5).nonzero(as_tuple=True)[0].item())
    m = 2.0 ** (-0.5)
    assert math.isclose(float(lvl.radius[daughter]), r_old * m, rel_tol=1e-5)   # still volume-conserving
    dist = float((lvl.state[0, 0:2] - lvl.state[daughter, 0:2]).norm())
    assert math.isclose(dist, 2.0 * r_old * m, rel_tol=1e-5)                    # still just-touching


def test_mask_restricts_who_divides():
    """The `at:` selector mask gates eligibility: with a huge hazard only masked-in live cells
    divide; masked-out live cells are untouched (no daughter, radius unchanged)."""
    H, lvl = _world(8, rate=1e3, radius=0.5, occ=[1, 1, 1, 1, 0, 0, 0, 0])
    mask = torch.zeros(8, dtype=torch.bool)
    mask[0] = True                                                             # only cell 0 eligible
    r_before = lvl.radius.clone()
    _op()(H, mask)
    assert torch.count_nonzero(lvl.born) == 1                                  # exactly one division
    assert int(lvl.mother[(lvl.born > 0.5)][0]) == 0                           # by cell 0
    assert math.isclose(float(lvl.radius[1]), float(r_before[1]), rel_tol=1e-6)  # masked-out cell 1 unchanged
