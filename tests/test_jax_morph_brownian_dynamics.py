"""Property tests for the `agitate` operator (candidate; jax-morph BrownianDynamics bath).

`agitate` emits ONLY the thermal (Brownian) leg of an overdamped Langevin step: a zero-drift
isotropic Gaussian kick returned as a `velocity` v = sqrt(2 kT / (gamma dt)) * xi, so that the
engine's `pos += dt * v` lands the Wiener displacement dx = sqrt(2 kT dt / gamma) * xi. Every
assertion below is stated WITHOUT the reference -- a limit, a scaling law, a conservation, a
symmetry -- so none can be satisfied by fitting the oracle's numbers:

  * kT = 0 LIMIT -- no bath: the emitted velocity is exactly zero (pure relaxation regime);
  * WIENER sqrt(dt) SCALING (the headline) -- with the noise held fixed, the DISPLACEMENT
    scales as sqrt(dt); quartering dt halves the displacement (a naive dt-scaled or
    dt-independent noise would not);
  * FDT AMPLITUDE -- the displacement scales as sqrt(kT) and as 1/sqrt(gamma) (the
    fluctuation-dissipation coupling std = sqrt(2 kT dt / gamma));
  * EINSTEIN DIFFUSION + ISOTROPY -- over many cells the per-dim displacement variance is
    2 kT dt / gamma (D = kT/gamma), the mean kick is zero, and no direction is preferred;
  * ALIVE / MASK MASKING -- a dead (occ = 0) or masked-out cell draws a zero kick;
  * POSITION IS NOT MUTATED -- the step returns a delta and never Euler-steps pos itself
    (the engine's integration invariant).
"""
import math

import types

import torch

import plexus.operators.candidates.jax_morph_brownian_dynamics as m  # noqa: F401  registers `agitate`
from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator


def _world(n, *, dim=2, dt=1.0, seed=0, occ=None):
    """A one-set cell world: `n` cells carrying a pos block, occupancy, config.dt and a seeded rng."""
    state = torch.zeros(n, dim)                                       # pos only (all the operator reads)
    occ = torch.ones(n) if occ is None else torch.as_tensor(occ, dtype=torch.float32)
    lvl = Level("cell", state=state, state_schema={"pos": (0, dim)}, occ=occ)
    H = Hierarchy()
    H.add_level(lvl)
    H.dim = dim
    H.config = types.SimpleNamespace(dt=dt)
    H.rng = torch.Generator().manual_seed(seed)
    return H, lvl


def _op(**params):
    params.setdefault("_at", "cell")
    return get_operator("agitate")(params, "cpu")


def _velocity(n, *, dim=2, dt=1.0, seed=0, mask=None, **params):
    """Run one step on a fresh (freshly-seeded) world and return the emitted velocity delta [n, dim]."""
    H, lvl = _world(n, dim=dim, dt=dt, seed=seed)
    out = _op(**params)(H, mask)
    return out["cell"]


def test_zero_temperature_is_no_bath():
    """kT = 0 is the deterministic gradient-descent relaxation limit: the thermal kick vanishes
    exactly (only a separate drift operator would then move cells)."""
    v = _velocity(64, kT=0.0, gamma=1.0, dt=0.5)
    assert torch.count_nonzero(v) == 0


def test_wiener_sqrt_dt_scaling_of_the_displacement():
    """THE headline property. Holding the noise xi fixed (same seed), the emitted velocity scales as
    1/sqrt(dt) so the DISPLACEMENT dt*v scales as sqrt(dt): quartering dt halves the displacement.
    This is the Euler-Maruyama (Wiener) discretization -- a dt-independent or dt-scaled noise fails."""
    kT, gamma, dt = 0.7, 1.3, 0.4
    v_big = _velocity(256, dt=dt, seed=11, kT=kT, gamma=gamma)
    v_small = _velocity(256, dt=dt / 4.0, seed=11, kT=kT, gamma=gamma)      # SAME seed -> same xi
    disp_big = dt * v_big
    disp_small = (dt / 4.0) * v_small
    assert torch.allclose(disp_small, 0.5 * disp_big, atol=1e-6)            # sqrt(dt/4)/sqrt(dt) = 1/2
    # sanity: the VELOCITY itself did change (it is genuinely dt-dependent, not a bare constant kick)
    assert not torch.allclose(v_small, v_big, atol=1e-6)
    assert torch.allclose(v_small, 2.0 * v_big, atol=1e-6)                  # v ~ 1/sqrt(dt)


def test_fluctuation_dissipation_amplitude_scaling():
    """The displacement amplitude follows std = sqrt(2 kT dt / gamma): with the noise fixed, it
    scales as sqrt(kT) (hotter -> larger kicks) and as 1/sqrt(gamma) (draggier -> smaller kicks)."""
    dt = 0.3
    base = dt * _velocity(200, dt=dt, seed=7, kT=0.25, gamma=1.0)
    hot = dt * _velocity(200, dt=dt, seed=7, kT=1.0, gamma=1.0)             # 4x kT -> 2x displacement
    draggy = dt * _velocity(200, dt=dt, seed=7, kT=0.25, gamma=4.0)         # 4x gamma -> 1/2 displacement
    assert torch.allclose(hot, 2.0 * base, atol=1e-6)
    assert torch.allclose(draggy, 0.5 * base, atol=1e-6)


def test_einstein_diffusion_constant_and_isotropy():
    """Over many cells the per-dimension displacement variance is 2 kT dt / gamma (diffusion
    D = kT/gamma, the Einstein relation), the mean kick is zero (no drift), and both dimensions
    carry the same variance (isotropic -- no preferred direction). Statistical, generous tol."""
    n, kT, gamma, dt = 40000, 0.5, 2.0, 0.8
    v = _velocity(n, dim=2, dt=dt, seed=3, kT=kT, gamma=gamma)
    disp = dt * v                                                            # what the engine applies
    expected_var = 2.0 * kT * dt / gamma                                     # = 2 D dt, D = kT/gamma
    var = disp.var(dim=0, unbiased=True)
    mean = disp.mean(dim=0)
    assert abs(float(mean[0])) < 0.02 and abs(float(mean[1])) < 0.02         # zero drift
    assert math.isclose(float(var[0]), expected_var, rel_tol=0.05)          # Einstein variance, dim 0
    assert math.isclose(float(var[1]), expected_var, rel_tol=0.05)          # ... and dim 1 (isotropy)
    assert math.isclose(float(var[0]), float(var[1]), rel_tol=0.05)         # no preferred direction


def test_dead_and_masked_cells_draw_no_kick():
    """A dead cell (occ = 0) gets a zero kick, and the `at:` selector mask gates who is agitated:
    live cells outside the mask are left untouched while masked-in live cells move."""
    H, lvl = _world(6, dt=0.5, seed=1, occ=[1, 0, 1, 1, 0, 1])
    mask = torch.tensor([True, True, False, True, True, True])              # cell 2 masked out (though live)
    v = _op(kT=0.5, gamma=1.0)(H, mask)["cell"]
    dead_or_masked = [1, 4, 2]                                              # occ=0 (1,4) or masked-out (2)
    for i in dead_or_masked:
        assert torch.count_nonzero(v[i]) == 0
    live_and_masked_in = [0, 3, 5]
    assert all(torch.count_nonzero(v[i]) > 0 for i in live_and_masked_in)


def test_step_does_not_mutate_position():
    """The bath returns a velocity DELTA and never integrates pos itself (the engine does that):
    positions are unchanged by a forward pass. This is what the frame-0 integration guard checks."""
    H, lvl = _world(32, dt=0.5, seed=2)
    before = lvl.get("pos").clone()
    _op(kT=0.9, gamma=1.0)(H, None)
    assert torch.equal(lvl.get("pos"), before)


def test_dimension_generic_3d():
    """The kick is dimension-generic: it runs in 3-D and stays isotropic (all three dims share the
    diffusion variance), with no special-casing of the spatial dimension."""
    n, kT, gamma, dt = 40000, 0.4, 1.0, 0.5
    v = _velocity(n, dim=3, dt=dt, seed=4, kT=kT, gamma=gamma)
    var = (dt * v).var(dim=0, unbiased=True)
    expected_var = 2.0 * kT * dt / gamma
    for d in range(3):
        assert math.isclose(float(var[d]), expected_var, rel_tol=0.05)


def test_n_space_dim_assert_catches_mismatch():
    """Faithful surprise: the source RAISES when the built n_space_dim disagrees with the state's
    real spatial dimension (it sizes the kick). A 2-D world with n_space_dim=3 must raise, not
    silently broadcast."""
    H, lvl = _world(8, dim=2, dt=0.5)
    try:
        _op(kT=0.5, n_space_dim=3)(H, None)
        raised = False
    except ValueError:
        raised = True
    assert raised
