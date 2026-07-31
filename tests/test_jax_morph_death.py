"""Property tests for the `apoptose` operator (jax-morph Death step).

These assert properties statable WITHOUT the reference -- limits of the Bernoulli hazard, the
clip guard, and the structural conservation laws (only live cells die, occupancy is monotone
non-increasing, the death record matches the retired slots) -- so they test the operator's
contract, not agreement with the oracle. The hazard is p = 1 - exp(-clip(death_rate, 0) * dt):
at rate 0 it is 0 (no-op), at a huge rate it is ~1 (certain death), and a negative rate clips
to 0 (no death).
"""
import types

import torch

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator

import plexus.operators.candidates.jax_morph_death  # noqa: F401  (registers `apoptose`)


def _world(n, rate, dt=1.0, seed=0, occ=None):
    """A one-set world: `n` cells carrying pos+vel state, a per-cell `death_rate` buffer, and
    an occupancy mask. `rate` may be a scalar (uniform) or a length-n sequence."""
    state = torch.zeros(n, 4)                                     # pos(2) + vel(2)
    occ = torch.ones(n) if occ is None else torch.as_tensor(occ, dtype=torch.float32)
    lvl = Level("cell", state=state, state_schema={"pos": (0, 2), "vel": (2, 4)}, occ=occ)
    if isinstance(rate, (int, float)):
        dr = torch.full((n,), float(rate))
    else:
        dr = torch.as_tensor(rate, dtype=torch.float32)
    lvl.register_buffer("death_rate", dr)
    H = Hierarchy()
    H.add_level(lvl)
    H.config = types.SimpleNamespace(dt=dt)
    H.rng = torch.Generator().manual_seed(seed)
    return H, lvl


def _op(params=None):
    return get_operator("apoptose")((params or {}), "cpu")


def test_zero_rate_is_a_noop():
    """death_rate = 0 -> p = 0 -> no cell dies: occupancy unchanged, death record all zero."""
    H, lvl = _world(8, rate=0.0)
    before = lvl.occ.clone()
    out = _op()(H, lvl.active)
    assert out == {}                                             # structural: returns no delta
    assert torch.equal(lvl.occ, before)                         # nobody retired
    assert torch.count_nonzero(lvl.death) == 0                  # no deaths recorded this step


def test_negative_rate_is_clipped_to_no_death():
    """A negative death_rate (e.g. an unconstrained controller output) clips to 0, so p = 0 and
    no cell dies -- rather than a negative probability that would NaN a score."""
    H, lvl = _world(8, rate=-5.0)
    before = lvl.occ.clone()
    _op()(H, lvl.active)
    assert torch.equal(lvl.occ, before)
    assert torch.count_nonzero(lvl.death) == 0


def test_certain_death_retires_every_live_cell():
    """A huge hazard drives p ~ 1: every eligible live cell dies deterministically, regardless of
    the seed. Occupancy goes all-zero and the death record marks exactly the cells that were live."""
    for seed in (0, 1, 7):
        H, lvl = _world(16, rate=1e3, seed=seed)
        was_live = lvl.occ > 0
        _op()(H, lvl.active)
        assert torch.all(lvl.occ == 0)                          # all live cells retired
        assert torch.equal(lvl.death > 0.5, was_live)           # death record = the newly-dead mask


def test_only_live_cells_die_and_count_is_monotone():
    """Dormant slots (occ = 0) are ineligible: they stay dormant, are never recorded as newly
    dead, and are never revived. The live count is monotone non-increasing (apoptose never wakes
    a slot -- it is cell_divide's inverse)."""
    occ = [1, 1, 0, 1, 0, 0, 1, 1]                              # four live, four dormant
    H, lvl = _world(8, rate=1e3, occ=occ)
    dormant = lvl.occ == 0
    n_live_before = int((lvl.occ > 0).sum())
    _op()(H, lvl.active)
    assert torch.all(lvl.occ[dormant] == 0)                     # dormant slots untouched (not revived)
    assert torch.all(lvl.death[dormant] == 0)                   # a dormant slot never "newly dies"
    assert int((lvl.occ > 0).sum()) <= n_live_before            # live count only decreases


def test_death_record_matches_the_retired_slots():
    """At a moderate hazard some cells die and some survive. The float `death` record is 1 exactly
    on the slots whose occupancy flipped 1 -> 0 this step, and 0 elsewhere; occupancy only ever
    decreases (no slot is woken)."""
    H, lvl = _world(64, rate=0.7, dt=1.0, seed=3)
    occ_before = lvl.occ.clone()
    _op()(H, lvl.active)
    flipped = (occ_before > 0) & (lvl.occ == 0)                 # slots retired this step
    assert torch.equal(lvl.death > 0.5, flipped)               # death record == the flip mask
    assert lvl.death.dtype.is_floating_point                    # float record (summable downstream)
    assert torch.all(lvl.occ <= occ_before)                    # monotone: nothing woken
    assert 0 < int(flipped.sum()) < 64                         # a genuine mix (not a degenerate all/none)


def test_mask_restricts_eligibility():
    """The `at:` selector mask gates who may die: with a huge hazard, only masked-in live cells are
    retired; masked-out live cells survive (their occupancy and death record are untouched)."""
    H, lvl = _world(10, rate=1e3)
    mask = torch.zeros(10, dtype=torch.bool)
    mask[:4] = True                                             # only the first four are eligible
    _op()(H, mask)
    assert torch.all(lvl.occ[:4] == 0)                         # eligible live cells died
    assert torch.all(lvl.occ[4:] > 0)                          # masked-out cells survived
    assert torch.all(lvl.death[4:] == 0)                       # and were not recorded as dead
