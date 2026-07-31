"""Property tests for the `adhere:hertzian` operator (jax-morph Hertzian contact potential).

These assert properties statable WITHOUT the reference -- they come from the calculus of the
conservative Hertzian energy U(r) = (2/5) eps (1 - r/sigma)^(5/2), not from the oracle's numbers:

* PURELY REPULSIVE, ANALYTIC FORCE -- two overlapping cells push apart with magnitude exactly
  f(r) = (eps/sigma)(1 - r/sigma)^(3/2). This is -dU/dr done by hand; matching it verifies that
  autodiff of the energy WITH the 2/5 prefactor yields a unit force coefficient (the whole point
  of the magic 2/5 = 1/exponent), and that the force points away from the neighbour.
* COMPACT SUPPORT -- at and beyond contact (r >= sigma = r_i + r_j) the force is exactly zero;
  there is no adhesive tail and no cutoff parameter (self-truncating at contact).
* C2 SOFTNESS -- the force -> 0 continuously as r -> sigma^-, and its magnitude falls monotonically
  as the overlap shrinks. (Contrast soft_sphere, whose force slope stays finite at contact.)
* NEWTON'S THIRD LAW -- a conservative pair energy gives equal-and-opposite forces, so the total
  force over a live cluster sums to zero (momentum conservation).
* SIZE-CONSISTENT CONTACT -- the contact distance is the ADDITIVE r_i + r_j, so growing a radius
  widens the interaction range; the same overlap fraction gives the same force fraction.
* DEAD-CELL MASKING -- masking is EXTERNAL to the per-pair law: a dead cell neither feels nor
  exerts a force, and its presence does not perturb the live pair.

None of these check agreement with the oracle -- they test the operator's contract.
"""
import math
import types

import torch

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator

import plexus.operators.candidates.jax_morph_hertzian  # noqa: F401  (registers adhere:hertzian)


def _world(pos, radius=0.5, occ=None, dim=2):
    """A one-set world: cells at `pos` [N, dim] with a per-cell `radius` buffer, pos+vel state."""
    pos = torch.as_tensor(pos, dtype=torch.float32)
    n = pos.shape[0]
    state = torch.zeros(n, 2 * dim)
    state[:, :dim] = pos
    schema = {"pos": (0, dim), "vel": (dim, 2 * dim)}
    occ = torch.ones(n) if occ is None else torch.as_tensor(occ, dtype=torch.float32)
    lvl = Level("cell", state=state, state_schema=schema, occ=occ)
    rr = torch.full((n,), float(radius)) if isinstance(radius, (int, float)) else torch.as_tensor(radius, dtype=torch.float32)
    lvl.register_buffer("radius", rr)
    H = Hierarchy()
    H.add_level(lvl)
    H.dim = dim
    H.config = types.SimpleNamespace(dt=1.0)
    return H, lvl


def _op(params=None):
    return get_operator("adhere", "hertzian")((params or {}), "cpu")


def _hertz_force(r, sigma, eps):
    """The analytic radial force magnitude f(r) = (eps/sigma)(1 - r/sigma)^(3/2), 0 beyond contact.
    Derived by hand from -dU/dr with U = (2/5) eps (1 - r/sigma)^(5/2) -- NOT read from the source."""
    if r >= sigma:
        return 0.0
    return (eps / sigma) * (1.0 - r / sigma) ** 1.5


def test_overlap_is_repulsive_with_analytic_magnitude():
    """Two cells overlapping within contact push APART, with the exact Hertzian force magnitude."""
    eps, r_cell = 1.0, 0.5
    sigma = 2.0 * r_cell                                     # = 1.0
    s = 0.5                                                  # centre separation < sigma (overlap)
    H, lvl = _world([[0.0, 0.0], [s, 0.0]], radius=r_cell)
    out = _op({"epsilon": eps})(H, lvl.active)["cell"]      # [2, 2] velocity == force (mobility 1)
    f = _hertz_force(s, sigma, eps)                          # (1/1)(1 - 0.5)^1.5 = 0.35355...
    # cell 0 pushed toward -x (away from cell 1 at +x); cell 1 toward +x, same magnitude.
    assert torch.allclose(out[0], torch.tensor([-f, 0.0]), atol=1e-5)
    assert torch.allclose(out[1], torch.tensor([+f, 0.0]), atol=1e-5)
    assert f > 0.0                                           # strictly repulsive inside contact


def test_beyond_contact_is_zero_force():
    """At and beyond contact (r >= sigma) the compact potential exerts exactly zero force -- no tail."""
    H, lvl = _world([[0.0, 0.0], [1.2, 0.0]], radius=0.5)   # s = 1.2 > sigma = 1.0
    out = _op({"epsilon": 1.0})(H, lvl.active)["cell"]
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)
    # exactly at contact (r = sigma) the force is also zero (compact, continuous).
    H2, lvl2 = _world([[0.0, 0.0], [1.0, 0.0]], radius=0.5)
    out2 = _op({"epsilon": 1.0})(H2, lvl2.active)["cell"]
    assert torch.allclose(out2, torch.zeros_like(out2), atol=1e-6)


def test_force_vanishes_softly_and_monotonically_at_contact():
    """C2 softness: |force| falls monotonically to 0 as the overlap shrinks toward contact."""
    eps, r_cell = 1.0, 0.5
    sigma = 1.0
    seps = [0.2, 0.5, 0.8, 0.98]                             # increasing separation -> shrinking overlap
    mags = []
    for s in seps:
        H, lvl = _world([[0.0, 0.0], [s, 0.0]], radius=r_cell)
        out = _op({"epsilon": eps})(H, lvl.active)["cell"]
        mags.append(float(out[1].norm()))
        assert math.isclose(mags[-1], _hertz_force(s, sigma, eps), rel_tol=1e-4)
    assert all(mags[k] > mags[k + 1] for k in range(len(mags) - 1))   # strictly decreasing
    assert mags[-1] < 1e-2                                            # nearly zero just below contact


def test_newtons_third_law_sum_of_forces_is_zero():
    """A conservative pair energy gives equal-and-opposite forces: the cluster's total force = 0."""
    torch.manual_seed(0)
    pos = torch.rand(12, 2) * 0.6                            # dense cluster (many overlaps within sigma=1)
    H, lvl = _world(pos, radius=0.5)
    out = _op({"epsilon": 2.0})(H, lvl.active)["cell"]
    assert out.abs().max() > 0.0                            # forces are actually acting
    assert torch.allclose(out.sum(dim=0), torch.zeros(2), atol=1e-4)


def test_size_consistent_contact_distance():
    """sigma = r_i + r_j is additive: unequal radii set the contact distance, and the same overlap
    FRACTION (r/sigma) yields the same force FRACTION -- growth widens the range, no retuning."""
    eps = 1.0
    # pair A: radii 0.5, 0.5 -> sigma 1.0, at r = 0.6 (overlap fraction 0.6)
    Ha, la = _world([[0.0, 0.0], [0.6, 0.0]], radius=[0.5, 0.5])
    fa = float(_op({"epsilon": eps})(Ha, la.active)["cell"][1].norm())
    # pair B: radii 0.7, 0.3 -> sigma 1.0 too, same r = 0.6 -> identical force (sigma is what matters)
    Hb, lb = _world([[0.0, 0.0], [0.6, 0.0]], radius=[0.7, 0.3])
    fb = float(_op({"epsilon": eps})(Hb, lb.active)["cell"][1].norm())
    assert math.isclose(fa, fb, rel_tol=1e-5)
    # pair C: radii 0.75, 0.75 -> sigma 1.5, at r = 0.9 (same fraction 0.6) -> force scales by 1/sigma
    Hc, lc = _world([[0.0, 0.0], [0.9, 0.0]], radius=[0.75, 0.75])
    fc = float(_op({"epsilon": eps})(Hc, lc.active)["cell"][1].norm())
    assert math.isclose(fc, _hertz_force(0.9, 1.5, eps), rel_tol=1e-4)


def test_dead_cells_are_masked_externally():
    """A dead cell neither feels nor exerts a force, and does not perturb the live overlapping pair:
    the live pair's force equals what it feels with the dead cell absent."""
    eps, r_cell, s = 1.0, 0.5, 0.5
    # cells 0,1 overlap and are alive; cell 2 sits right on top of cell 0 but is DEAD.
    H, lvl = _world([[0.0, 0.0], [s, 0.0], [0.0, 0.0]], radius=r_cell, occ=[1, 1, 0])
    out = _op({"epsilon": eps})(H, lvl.active)["cell"]
    assert torch.allclose(out[2], torch.zeros(2), atol=1e-7)          # dead cell emits nothing
    # the live pair is unaffected by the phantom dead overlapper (would-be huge force is masked):
    f = _hertz_force(s, 2 * r_cell, eps)
    assert torch.allclose(out[0], torch.tensor([-f, 0.0]), atol=1e-5)
    assert torch.allclose(out[1], torch.tensor([+f, 0.0]), atol=1e-5)


def test_per_cell_epsilon_mixes_by_arithmetic_mean():
    """A per-cell stiffness field is combined per pair by the arithmetic mean 0.5*(eps_i + eps_j),
    so eps=(2,4) acts like a shared eps=3 -- distinct from the additive sigma rule."""
    r_cell, s = 0.5, 0.5
    sigma = 1.0
    lvl_eps = [2.0, 4.0]
    H, lvl = _world([[0.0, 0.0], [s, 0.0]], radius=r_cell)
    lvl.register_buffer("stiffness", torch.tensor(lvl_eps))
    out = _op({"epsilon_field": "stiffness"})(H, lvl.active)["cell"]
    f = _hertz_force(s, sigma, 3.0)                                   # mean(2,4) = 3
    assert torch.allclose(out[1], torch.tensor([+f, 0.0]), atol=1e-5)


def test_force_is_differentiable_wrt_positions():
    """The emitted force is a real autodiff of the energy: gradients flow to positions under grad."""
    H, lvl = _world([[0.0, 0.0], [0.5, 0.0]], radius=0.5)
    lvl.state.requires_grad_(True)
    out = _op({"epsilon": 1.0})(H, lvl.active)["cell"]
    loss = out.pow(2).sum()
    loss.backward()
    assert lvl.state.grad is not None
    assert torch.isfinite(lvl.state.grad).all()                      # no NaN from the fractional power
    assert lvl.state.grad.abs().sum() > 0.0                          # the force genuinely depends on pos
