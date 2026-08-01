r"""relax -- cell (lateral, quasistatic). Drive cell POSITIONS to a mechanical equilibrium.

BIOLOGY. Between the discrete events of a morphogenesis step (a cell divides, secretes,
changes adhesion) the tissue is assumed to sit at MECHANICAL EQUILIBRIUM: every cell has
settled into the force-balanced arrangement its neighbours and its own size dictate. `relax`
is that settling. Given an interaction potential U over the cell positions, it moves the
cluster to the configuration x* where the net force on every (alive) cell vanishes,
grad_x U(x*) = 0 -- a genuine force balance, not a single nudge. This is the paper's
"mechanical relaxation" framework slot (Deshpande, Mottes et al. 2025, Fig. 1a): after each
chemistry/growth event the cells relax before the next event is evaluated.

QUASISTATIC, NOT DYNAMIC (the contract distinction, and the key emit subtlety). The registered
force laws -- `adhere` (soft_sphere / hertzian / morse / ...), `attraction_repulsion`, the
`agitate` bath -- are DYNAMIC: each emits ONE overdamped velocity per macro-step and the engine
takes a single Euler step `pos += dt * v`, so the cluster creeps toward equilibrium over many
steps. `relax` is QUASISTATIC: it runs an INNER minimizer (FIRE) all the way to the force
balance and lands the cluster on x* in a SINGLE macro-step. Plexus has no quasistatic phase and
only `_integrate` may write pos (the integration invariant), so we express the overwrite through
the delta contract: the engine multiplies our emitted velocity by dt, so to land the finite
displacement (x* - x0) in one step we emit

    v = (x* - x0) / dt        ->   pos += dt * v = pos + (x* - x0) = x*   (exact, for any dt).

That 1/dt factor is the SIGNATURE of a quasistatic step (it reaches equilibrium regardless of
dt), exactly as `agitate` emits a 1/sqrt(dt) velocity to realize a Wiener increment. A
reimplementer who emits the raw displacement, or the force itself, gets a dynamic creep, not a
relaxation.

POTENTIAL-AGNOSTIC SOLVER (orthogonal to the force law). `relax` is NOT a force law; it WRAPS
one. `potential:` selects the energy family it relaxes -- `soft_sphere` (default), `hertzian`,
`harmonic`, `morse`, `lennard_jones`, or `none` (the reference's NoForce: every configuration is
already an equilibrium, so relaxation is a no-op and positions pass through). The same `relax`
step runs any of them; the force law (which interaction) and the solver (how to advance under it)
are orthogonal axes that compose. The energies are the sigma-relative, size-consistent
(sigma = r_i + r_j) pair potentials of the source's potential library, autodiffed to forces.

FIRE (the minimizer). The forward is the Fast Inertial Relaxation Engine (Bitzek 2006), a
momentum-accelerated gradient descent with an adaptive timestep, NOT plain gradient descent: a
velocity is carried and mixed toward the instantaneous force direction, the timestep grows on a
run of downhill steps and is reset (with the velocity) whenever a step goes uphill. It STOPS at a
force tolerance |grad U|_inf <= f_tol (a real equilibrium), with `max_steps` only a fallback
bound. The FIRE schedule constants (dt_start, dt_max, n_min, f_inc, f_dec, alpha_start, f_alpha)
are the standard Bitzek values, hardcoded (only max_steps / f_tol are exposed), matching the
source.

FORWARD-ONLY / GRADIENTS. The solver path is not differentiated: `_fire_to_tol` runs under
detached tensors and this implementation is marked DIFFERENTIABLE=False. The source differentiates
the EQUILIBRIUM (not the FIRE path) by the implicit function theorem -- dx*/dp = -H^{-1}
d(grad U)/dp solved by conjugate gradient on the physical (deformation) subspace, with the
rigid-body gauge modes (a global translation and rotation of the alive cells) projected out so
they carry no gradient. Reproducing that adjoint as a torch.autograd.Function (a projected-Hessian
CG solve) is the promotion follow-up; the forward equilibration -- the part the engine runs and the
part this candidate is tested on -- is faithful here.

WHY NEW. No registered contract carries "relax to mechanical equilibrium". The closest,
`attraction_repulsion`, is a per-step velocity emitter over an edge graph -- a dynamic forward
map; widening it to an equilibrium SOLVER changes its temporal mode (quasistatic vs dynamic), its
output (a converged x* vs one velocity), and its backward (implicit-diff of a fixed point vs
forward autodiff of a force). The campaign-local `adhere` contract is the FORCE LAW `relax`
wraps, not `relax` itself; the two lie on orthogonal axes and compose. `agitate` is the sibling
stepper's THERMAL bath (kT>0, a Wiener increment); `relax` is the deterministic (kT=0)
equilibration.

SOURCE vs PAPER (rule 5, SOURCE WINS). The paper's Methods (p. 14) call this "gradient descent
energy minimization of the Morse potential for a FIXED NUMBER OF STEPS (except in the case where
we learn cell adhesion)". The SOURCE differs on three axes: the minimizer is FIRE (momentum +
adaptive dt), not plain descent; it runs to a force TOLERANCE (a real equilibrium), not a fixed
count; and the gradient is the implicit-function-theorem sensitivity of the EQUILIBRIUM, not
backprop through the descent. The parenthetical "(except ... cell adhesion)" marks exactly the
regime where fixed-step solver-path gradients break down -- the problem the source's implicit-diff
solves in general.

Translated from papers/jax-morph/jax_morph/physics/mechanics/relaxation.py:L221
(MechanicalRelaxation; forward `_fire_to_tol` at :L23, implicit-diff backward `_relax_bwd` at
:L158, rigid-mode projection `_rigid_body_modes` at :L93) and the potential library
physics/mechanics/potentials.py (Morse/SoftSphere/Hertzian/Harmonic/LennardJones,
`_compact_repulsion` at :L30). Torch, not JAX. FIRE: Bitzek, E. et al. (2006) Phys. Rev. Lett.
97:170201. Paper: Deshpande, Mottes et al., Nat. Comput. Sci. (2025) -- paper is fixed-step
descent, SOURCE is FIRE-to-tolerance + implicit-diff; SOURCE WINS.
"""
from __future__ import annotations

import torch

from plexus.geometry import minimum_image
from plexus.models.base import Lateral
from plexus.models.registry import register_operator


# --------------------------------------------------------------------------- #
#  AD-safe helpers (verbatim ports of the source's ad_utils, so the fractional
#  powers stay finite at dead-dead padded pairs and the self-diagonal)
# --------------------------------------------------------------------------- #
def _safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a / b` where `b != 0`, else 0 -- finite gradient at `b == 0` (double-`where`)."""
    safe_b = torch.where(b != 0.0, b, torch.ones_like(b))
    return torch.where(b != 0.0, a / safe_b, torch.zeros_like(a))


def _safe_norm(d2: torch.Tensor) -> torch.Tensor:
    """`sqrt(d2)` where `d2 > 0`, else 0 -- finite gradient at `d2 == 0` (the self-diagonal)."""
    safe = torch.where(d2 > 0.0, d2, torch.ones_like(d2))
    return torch.where(d2 > 0.0, torch.sqrt(safe), torch.zeros_like(d2))


def _compact_repulsion(r, sigma, eps, exponent: float, prefactor: float):
    """`prefactor * eps * (1 - r/sigma)**exponent` for `r < sigma`, else 0 (value AND grad safe)."""
    base = 1.0 - _safe_divide(r, sigma)
    safe = torch.where(base > 0.0, base, torch.ones_like(base))     # power only sees a positive base
    return torch.where(base > 0.0, prefactor * eps * safe ** exponent, torch.zeros_like(base))


def _smooth_cutoff(r, r_on, r_off):
    """Multiplicative C1 isotropic cutoff (jax-md form): 1 at r<=r_on, 0 at r>=r_off."""
    r2, ron2, roff2 = r * r, r_on * r_on, r_off * r_off
    s = _safe_divide((roff2 - r2) ** 2 * (roff2 + 2.0 * r2 - 3.0 * ron2), (roff2 - ron2) ** 3)
    return torch.where(r < r_on, torch.ones_like(r), torch.where(r < r_off, s, torch.zeros_like(r)))


# The size-consistent pair energies of the source's potential library. Each is a sigma-relative
# (sigma = r_i + r_j) energy; the force comes from autodiff, exactly as the source does. `eps` is
# the coupling (well depth / stiffness / spring constant), scalar or an [N, N] per-pair matrix.
_POTENTIALS = ("none", "soft_sphere", "hertzian", "harmonic", "morse", "lennard_jones")
_EPS_DEFAULT = {"soft_sphere": 1.0, "hertzian": 1.0, "harmonic": 1.0, "morse": 3.0,
                "lennard_jones": 1.0, "none": 0.0}


def _pair_energy(potential, r, sigma, eps, alpha, r_onset_frac, r_cutoff_frac):
    """Elementwise pair energy for the selected `potential` at separation `r` [N, N]."""
    if potential == "soft_sphere":                                 # harmonic overlap, compact at contact
        return _compact_repulsion(r, sigma, eps, 2.0, 0.5)
    if potential == "hertzian":                                    # elastic contact, compact at contact
        return _compact_repulsion(r, sigma, eps, 2.5, 0.4)
    if potential == "harmonic":                                    # shifted parabola truncated at r_cut
        r_cut = r_cutoff_frac * sigma
        u = 0.5 * eps * ((r - sigma) ** 2 - (r_cut - sigma) ** 2)
        return torch.where(r < r_cut, u, torch.zeros_like(u))
    if potential == "morse":                                       # adhesive well, min -eps at contact
        e = 1.0 - torch.exp(-alpha * (r - sigma))
        u = eps * (e * e - 1.0)
        return u * _smooth_cutoff(r, r_onset_frac * sigma, r_cutoff_frac * sigma)
    if potential == "lennard_jones":                               # r^-12 core + r^-6 adhesive tail
        x = _safe_divide(sigma, r)
        u = eps * (x ** 12 - 2.0 * x ** 6)
        return u * _smooth_cutoff(r, r_onset_frac * sigma, r_cutoff_frac * sigma)
    raise ValueError(f"relax: unknown potential {potential!r}; choose one of {_POTENTIALS}.")


# --------------------------------------------------------------------------- #
#  FIRE: forward-only relaxation to a force tolerance (the source's _fire_to_tol)
# --------------------------------------------------------------------------- #
def _force(energy_fn, x: torch.Tensor) -> torch.Tensor:
    """`f = -grad_x U` at `x`, via autodiff of the energy on a fresh detached leaf.

    Locally re-enables grad (the engine runs generation under `torch.no_grad`) and returns a
    DETACHED force, so the FIRE path is never part of any outer graph (forward-only, matching the
    source's `jax.custom_vjp`, which replaces the solver-path autodiff with the implicit adjoint).
    """
    with torch.enable_grad():
        xl = x.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(energy_fn(xl), xl)
    return -g.detach()


def _fire_to_tol(energy_fn, x0: torch.Tensor, max_steps: int = 500, f_tol: float = 1e-3, *,
                 dt_start: float = 0.01, dt_max: float = 0.1, n_min: int = 5,
                 f_inc: float = 1.1, f_dec: float = 0.5, alpha_start: float = 0.1,
                 f_alpha: float = 0.99) -> torch.Tensor:
    """Relax `x0` to a mechanical equilibrium of `energy_fn` with FIRE (Bitzek 2006).

    STOP when the force inf-norm |grad U|_inf <= f_tol (a genuine force balance) OR the iteration
    count reaches `max_steps` (a fallback bound; the last iterate is returned anyway). The FIRE
    schedule mixes a carried velocity toward the force direction, grows the timestep on a run of
    `n_min` downhill steps, and zeros the velocity / shrinks the timestep on an uphill step. All
    reductions (power, force norm, velocity norm) are GLOBAL over the whole array, as in the source.
    """
    x = x0.detach().clone()
    v = torch.zeros_like(x)
    dt, alpha, n_pos = float(dt_start), float(alpha_start), 0
    f = _force(energy_fn, x)
    i = 0
    while i < max_steps and float(f.abs().max()) > f_tol:
        uphill = float((f * v).sum()) <= 0.0                       # power P = f . v; uphill when P <= 0
        if uphill:
            n_pos, dt, alpha, v = 0, dt * f_dec, alpha_start, torch.zeros_like(v)
        else:
            n_pos += 1
            if n_pos > n_min:                                      # a run of downhill steps: accelerate
                dt, alpha = min(dt * f_inc, dt_max), alpha * f_alpha
        v = v + dt * f                                             # (velocity was zeroed above on uphill)
        fnorm = f.norm() + 1e-30
        v = (1.0 - alpha) * v + alpha * (f / fnorm) * v.norm()     # mix toward the force direction
        x = x + dt * v                                             # semi-implicit Euler position update
        f = _force(energy_fn, x)                                   # force at the new x, carried to the test
        i += 1
    return x


def relax_equilibrium(energy_fn, x0: torch.Tensor, *, max_steps: int = 500,
                      f_tol: float = 1e-3) -> torch.Tensor:
    """Relax `x0` to the equilibrium `grad_x U = 0` of `energy_fn` (the source's public entry).

    Mirrors ``relax_equilibrium`` in the source: a forward-only FIRE to a force tolerance. Pass any
    scalar-valued ``energy_fn(x) -> tensor``; the gradient defines the force.
    """
    return _fire_to_tol(energy_fn, x0, max_steps, f_tol)


@register_operator("relax", family="mechanics", set="cell", kind="lateral")
class Relax(Lateral):
    EMIT = "velocity"                            # quasistatic: emit (x* - x0)/dt; engine lands pos on x*
    # typed signature (Plexus2 sec. 2.1): a within-set cell -> cell morphism. Reads pos (the FIRE
    # start), radius (sigma = r_i + r_j), alive (pair masking), and any per-cell stiffness field;
    # writes the pos delta. Dense pairwise energy -- no named gather map.
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["pos", "radius", "alive"]      # + `epsilon_field` when the coupling is per-cell
    WRITES = ["pos"]
    MAPS = []
    SUPPORTED_DIMS = [2, 3]                       # energy + minimum_image are dimension-generic (D = pos.shape[-1])
    # This IMPLEMENTATION is forward-only: gradients do NOT flow through the FIRE path, and the
    # source's implicit-diff equilibrium adjoint is not reproduced here (a follow-up). An inverse
    # loop filtering capabilities() for `differentiable` correctly skips this implementation.
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = []                          # `potential` defaults to soft_sphere; all knobs optional
    MECHANISM_TAGS = ["mechanical_equilibration", "quasistatic_relaxation", "energy_minimization",
                      "force_balance", "fire_minimizer", "excluded_volume"]
    PARAM_ROLES = {"potential": "interaction_potential", "epsilon": "coupling_strength",
                   "alpha": "morse_well_width", "max_steps": "solver_iteration_cap",
                   "f_tol": "equilibrium_tolerance", "epsilon_field": "per_cell_coupling_block"}
    REFERENCE = (
        "jax-morph MechanicalRelaxation, physics/mechanics/relaxation.py:L221 "
        "(FIRE forward `_fire_to_tol` at :L23, implicit-diff backward `_relax_bwd` at :L158, "
        "rigid-mode projection `_rigid_body_modes` at :L93); potential library "
        "physics/mechanics/potentials.py (`_compact_repulsion` at :L30, Morse/SoftSphere/"
        "Hertzian/Harmonic/LennardJones). FIRE: Bitzek, E. et al. (2006) Phys. Rev. Lett. "
        "97:170201. Deshpande, Mottes et al. (2025) Nat. Comput. Sci. -- paper is fixed-step "
        "gradient descent (Methods p. 14), source is FIRE-to-tolerance + implicit-diff; SOURCE WINS."
    )

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.potential = str(params.get("potential", "soft_sphere")).lower()
        if self.potential not in _POTENTIALS:
            raise ValueError(f"relax: unknown potential {self.potential!r}; choose from {_POTENTIALS}.")
        self.epsilon = float(params.get("epsilon", _EPS_DEFAULT[self.potential]))   # coupling (eps / k)
        self.eps_field = params.get("epsilon_field", None)          # optional per-cell coupling block/buffer
        self.alpha = float(params.get("alpha", 2.8))                # morse well width (unused by others)
        self.r_onset_frac = float(params.get("r_onset_frac", 1.5))  # smooth-cutoff onset (morse / LJ)
        self.r_cutoff_frac = float(params.get("r_cutoff_frac", 2.5))  # cutoff end (morse / LJ / harmonic)
        self.max_steps = int(params.get("max_steps", 500))          # FIRE fallback bound (NOT the stop test)
        self.f_tol = float(params.get("f_tol", 1e-3))               # force tolerance -> genuine equilibrium
        self.radius0 = float(params.get("radius", 0.5))             # fallback uniform radius if no buffer

    def _read_epsilon(self, lvl, n, dev):
        """Per-pair coupling: a per-cell field mixed by the arithmetic mean, else the shared scalar."""
        if self.eps_field is None:
            return self.epsilon
        if self.eps_field in getattr(lvl, "state_schema", {}):
            v = lvl.get(self.eps_field)[:, 0]
        else:
            v = getattr(lvl, self.eps_field)
        v = v.reshape(n).to(dev)
        return 0.5 * (v[:, None] + v[None, :])                      # arithmetic-mean mix (finite grad at zeros)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        n = lvl.n
        a, b = lvl.state_schema["pos"]
        pos = lvl.state[:, a:b]                                     # [N, D] view of the integrated state

        # NoForce: every configuration is already an equilibrium -> a no-op (positions pass through).
        if self.potential == "none":
            return {self.at: torch.zeros_like(pos)}

        periodic = getattr(H, "periodic", False)
        world = getattr(H, "world_size", getattr(H, "world_width", 1.0))

        radius = getattr(lvl, "radius", None)
        if radius is None:
            radius = torch.full((n,), self.radius0, device=dev)
        radius = radius.reshape(n).to(dev)
        sigma = radius[:, None] + radius[None, :]                   # [N, N] contact distance r_i + r_j

        eps_pair = self._read_epsilon(lvl, n, dev)                 # scalar or [N, N]

        alive = lvl.occ > 0
        eye = torch.eye(n, dtype=torch.bool, device=dev)
        pair_mask = alive[:, None] & alive[None, :] & ~eye         # live non-self pairs

        def energy(x):
            disp = minimum_image(x[:, None, :] - x[None, :, :], periodic, world)   # [N, N, D]
            r = _safe_norm((disp * disp).sum(-1))                                  # [N, N], 0 on the diagonal
            u = _pair_energy(self.potential, r, sigma, eps_pair, self.alpha,
                             self.r_onset_frac, self.r_cutoff_frac)
            u = torch.where(pair_mask, u, torch.zeros_like(u))                      # mask self + dead pairs
            return 0.5 * u.sum()                                                   # each unordered pair once

        x0 = pos.detach()
        x_star = _fire_to_tol(energy, x0, self.max_steps, self.f_tol)              # relax to equilibrium

        # Quasistatic emit: the engine integrates a velocity as `pos += dt * v`, so emitting
        # (x* - x0)/dt lands pos exactly on the equilibrium x* in ONE macro-step, for any dt.
        cfg = getattr(H, "config", None)
        dt = float(getattr(cfg, "dt", 1.0)) if cfg is not None else 1.0
        if dt <= 0.0:
            dt = 1.0
        vel = (x_star - x0) / dt
        vel = vel * lvl.occ[:, None]                                # dead cells emit nothing
        if mask is not None:
            vel = vel * mask[:, None].float()
        return {self.at: vel}
