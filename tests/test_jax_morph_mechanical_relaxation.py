"""Property tests for the `relax` operator (jax-morph MechanicalRelaxation / FIRE).

These assert properties statable WITHOUT the reference -- they follow from what a mechanical
relaxation IS (a minimizer of an interaction energy to a force balance), not from the oracle's
numbers:

* FORCE BALANCE -- after relaxation the residual force inf-norm is <= f_tol: the returned
  configuration is a genuine mechanical equilibrium grad_x U(x*) = 0 (the defining property).
* ENERGY DESCENT -- U(x*) <= U(x0): a minimizer never raises the energy.
* OVERLAP RESOLUTION (sign) -- two soft-sphere cells starting inside contact (r0 < sigma) are
  pushed APART to (about) contact; repulsion separates them, it does not pull them together.
* NoForce NO-OP (limit) -- `potential: none` leaves every position exactly unchanged: every
  configuration is already an equilibrium.
* TRANSLATION SYMMETRY -- the energy depends only on pairwise distances, so relaxing a
  translated initial condition gives the translated equilibrium (in free space).
* QUASISTATIC, dt-INDEPENDENT -- the operator emits (x* - x0)/dt, so ONE engine macro-step
  `pos += dt*v` lands the cluster on the SAME force-balanced x* for any dt.

None of these check agreement with the oracle -- they test the operator's contract.
"""
import types

import torch

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator

import plexus.operators.candidates.jax_morph_mechanical_relaxation as R  # noqa: F401 (registers `relax`)
from plexus.operators.candidates.jax_morph_mechanical_relaxation import (
    _compact_repulsion,
    _force,
    _safe_norm,
    relax_equilibrium,
)


def _world(pos, radius=0.5, occ=None, dim=2, dt=1.0):
    """A one-set world: cells at `pos` [N, dim] with a per-cell `radius` buffer, pos+vel state."""
    pos = torch.as_tensor(pos, dtype=torch.float32)
    n = pos.shape[0]
    state = torch.zeros(n, 2 * dim)
    state[:, :dim] = pos
    schema = {"pos": (0, dim), "vel": (dim, 2 * dim)}
    occ = torch.ones(n) if occ is None else torch.as_tensor(occ, dtype=torch.float32)
    lvl = Level("cell", state=state, state_schema=schema, occ=occ)
    rr = (torch.full((n,), float(radius)) if isinstance(radius, (int, float))
          else torch.as_tensor(radius, dtype=torch.float32))
    lvl.register_buffer("radius", rr)
    H = Hierarchy()
    H.add_level(lvl)
    H.dim = dim
    H.config = types.SimpleNamespace(dt=dt)
    return H, lvl


def _op(params=None):
    return get_operator("relax")((params or {}), "cpu")


def _soft_sphere_energy(radius, eps=1.0):
    """The soft-sphere overlap energy U = 0.5 eps (1 - r/sigma)^2 summed over non-self pairs."""
    rad = torch.as_tensor(radius, dtype=torch.float32)

    def E(x):
        rr = rad if rad.ndim == 1 else rad.reshape(1).expand(x.shape[0])   # scalar -> per-cell vector
        disp = x[:, None, :] - x[None, :, :]
        r = _safe_norm((disp * disp).sum(-1))
        sigma = rr[:, None] + rr[None, :]
        u = _compact_repulsion(r, sigma, eps, 2.0, 0.5)
        eye = torch.eye(x.shape[0], dtype=torch.bool)
        u = torch.where(~eye, u, torch.zeros_like(u))
        return 0.5 * u.sum()

    return E


# --------------------------------------------------------------------------- #
#  the module-level FIRE (relax_equilibrium): the pure-solver properties
# --------------------------------------------------------------------------- #
def test_relaxation_reaches_a_force_balance_and_lowers_energy():
    """FIRE drives a dense overlapping cluster to grad U ~ 0, without ever raising the energy."""
    torch.manual_seed(0)
    x0 = torch.rand(10, 2) * 0.6          # dense cluster, radius 0.5 -> many overlaps within sigma=1
    E = _soft_sphere_energy(0.5, eps=1.0)
    f_tol = 1e-3
    x_star = relax_equilibrium(E, x0, f_tol=f_tol, max_steps=2000)
    residual = float(_force(E, x_star).abs().max())
    assert residual <= f_tol                       # a genuine mechanical equilibrium
    assert float(E(x_star)) <= float(E(x0)) + 1e-6  # energy never increased


def test_overlap_is_resolved_apart_not_together():
    """Two soft-sphere cells starting inside contact are pushed APART to (at least) contact sigma."""
    E = _soft_sphere_energy([0.5, 0.5], eps=1.0)   # sigma = 1.0
    x0 = torch.tensor([[0.0, 0.0], [0.5, 0.0]])    # overlap: separation 0.5 < sigma
    x_star = relax_equilibrium(E, x0, f_tol=1e-3, max_steps=2000)
    sep = float((x_star[1] - x_star[0]).norm())
    assert sep > 0.5                               # they separated (repulsion, not attraction)
    assert sep >= 0.99                             # to at least the contact distance sigma = 1.0
    # every r >= sigma is a zero-force equilibrium (compact support); FIRE's momentum may coast
    # just past contact into that flat region -- so the stop is a genuine force balance either way.
    assert float(_force(E, x_star).abs().max()) <= 1e-3
    com0, com1 = x0.mean(0), x_star.mean(0)
    assert torch.allclose(com0, com1, atol=1e-4)   # equal-and-opposite: centre of mass conserved


def test_translation_symmetry_in_free_space():
    """Relaxing a translated initial condition gives the translated equilibrium (distance-only U)."""
    E = _soft_sphere_energy(0.5, eps=1.0)
    x0 = torch.tensor([[0.0, 0.0], [0.4, 0.1], [0.1, 0.45]])
    shift = torch.tensor([3.0, -2.0])
    xa = relax_equilibrium(E, x0, f_tol=1e-4, max_steps=2000)
    xb = relax_equilibrium(E, x0 + shift, f_tol=1e-4, max_steps=2000)
    assert torch.allclose(xb, xa + shift, atol=1e-4)


# --------------------------------------------------------------------------- #
#  the operator forward: the quasistatic emit + the NoForce no-op
# --------------------------------------------------------------------------- #
def test_noforce_is_a_noop():
    """`potential: none` (the reference NoForce) emits zero -- positions pass through unchanged."""
    H, lvl = _world([[0.0, 0.0], [0.5, 0.0]], radius=0.5)
    out = _op({"potential": "none"})(H, lvl.active)["cell"]
    assert torch.allclose(out, torch.zeros_like(out), atol=0.0)


def test_forward_lands_on_equilibrium_in_one_macro_step():
    """One engine step pos += dt*v puts the cluster at a force balance -- the quasistatic contract."""
    H, lvl = _world([[0.0, 0.0], [0.5, 0.0]], radius=0.5, dt=1.0)
    x0 = lvl.get("pos").clone()
    vel = _op({"potential": "soft_sphere", "epsilon": 1.0, "f_tol": 1e-3})(H, lvl.active)["cell"]
    x_new = x0 + H.config.dt * vel                 # what the engine's `velocity` integration does
    E = _soft_sphere_energy(0.5, eps=1.0)
    assert float(_force(E, x_new).abs().max()) <= 1e-3   # landed on the equilibrium in ONE step


def test_quasistatic_is_dt_independent():
    """The emitted velocity scales as 1/dt, so the landed configuration is the SAME for any dt."""
    landed = []
    for dt in (0.25, 1.0, 4.0):
        H, lvl = _world([[0.0, 0.0], [0.5, 0.0], [0.2, 0.4]], radius=0.5, dt=dt)
        x0 = lvl.get("pos").clone()
        vel = _op({"potential": "soft_sphere", "f_tol": 1e-3})(H, lvl.active)["cell"]
        landed.append(x0 + dt * vel)               # dt * (x* - x0)/dt == x* - x0, for any dt
    assert torch.allclose(landed[0], landed[1], atol=1e-5)
    assert torch.allclose(landed[1], landed[2], atol=1e-5)


def test_dead_cells_do_not_move():
    """A dead cell emits zero relaxation velocity (occupancy-masked), whatever the live pair does."""
    H, lvl = _world([[0.0, 0.0], [0.5, 0.0], [0.25, 0.0]], radius=0.5, occ=[1, 1, 0])
    out = _op({"potential": "soft_sphere"})(H, lvl.active)["cell"]
    assert torch.allclose(out[2], torch.zeros(2), atol=0.0)   # dead cell held fixed
    assert out[:2].abs().max() > 0.0                          # the live overlapping pair does relax
