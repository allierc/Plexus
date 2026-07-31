"""adhere:lennard_jones -- cell <-> cell mechanics (the Lennard-Jones implementation).

BIOLOGY. Two cells interact through their membranes: they cannot interpenetrate (a hard
excluded-volume core) and, once in contact, they stick (a short-range adhesive tail). The
Lennard-Jones law in ``r_min`` form captures both with one 12-6 well. Writing the contact
distance as the sum of the two cell radii, sigma = r_i + r_j, the pair energy is

    U(r) = eps ( (sigma/r)^12 - 2 (sigma/r)^6 )      minimum -eps EXACTLY at r = sigma,

a steep r^-12 repulsive core minus a 2 r^-6 adhesive tail, its adhesive tail truncated by a
smooth sigma-relative cutoff (S = 1 inside r_onset_frac*sigma, ramping C1 to 0 at
r_cutoff_frac*sigma, defaults 1.5 and 2.5). Because sigma is built from the two radii, the
excluded volume AND the adhesive well track cell SIZE with no interaction-range knob to retune
as a cell grows or divides.

THE r_min FORM (the headline subtlety, faithfully preserved). This is NOT the textbook
``4 eps ((sigma/r)^12 - (sigma/r)^6)``. Here sigma is the location of the MINIMUM (the contact
distance r_i + r_j), the well value is exactly -eps, and the ZERO-CROSSING sits below contact.
The textbook 4-eps form instead puts its minimum at 2^(1/6) sigma with depth -eps there -- a
silent, plausible-looking mistake that would place cell equilibrium at the wrong separation and
break size-consistency. The property test below asserts the well is at contact (zero force at
r = sigma), which the 4-eps form fails.

ROUTING. ``kind=lateral, family=interaction, set=cell`` -- a within-set pairwise force, the
sibling of ``attraction_repulsion`` / ``cohesion`` / ``separation`` and the twin of
``adhere:hertzian`` / ``adhere:soft_sphere``. It is a CONSERVATIVE energy: rather than hand-code
the radial force we build the total pair energy and take the force by AUTODIFF, exactly as the
source does (``forces = -grad(total_energy)``) -- there is NO analytic force and NO jax-md
dependency in the source either. The paper's mechanics is OVERDAMPED (gradient-descent
``MechanicalRelaxation`` / overdamped ``BrownianDynamics``), so the force is emitted as a velocity
(``EMIT="velocity"``, velocity = mobility * F, mobility = 1 by default); the engine integrates it
and it simply sums with any other velocity a cell carries (glide, chemotax, another interaction,
the ``agitate`` thermal bath). This matches ``attraction_repulsion`` (the registered near-miss),
which also emits an overdamped velocity.

DENSE, SIZE-CONSISTENT, MASKED. The sum runs over all live non-self pairs (dense N x N,
half-summed with the 0.5 factor so each unordered pair counts once -- dropping it double-counts
the energy and hence the forces), the source's ``neighbor_sum`` seam. Two guards keep the
autodiff finite everywhere (both ported verbatim from the source):
* ``x = safe_divide(sigma, r)`` returns 0 (not inf) at r = 0, so the self-diagonal and any
  dead-dead padded pair give U = 0 rather than a NaN that a naive sigma/r would inject even
  through a masked entry (0 * inf = NaN in the backward pass);
* ``_smooth_cutoff`` divides by (r_off^2 - r_on^2)^3, which is 0 for a dead-dead pair
  (sigma = 0 -> r_on = r_off = 0); a ``safe_divide`` guards that 0/0.
Dead cells and self-pairs are ALSO removed by an EXTERNAL alive-mask on the energy (not inside
the per-pair law): the diagonal and dead pairs already evaluate to a clean 0 here, but the mask
keeps the seam identical to the repulsion-only siblings (whose compact core does score a spurious
value on a dead-dead pair) and documents the neighbor_sum contract. The contact distance uses the
ADDITIVE rule sigma = r_i + r_j (NOT the arithmetic mean); the per-cell well depth ``epsilon``,
when it is a field, is combined per pair by the ARITHMETIC MEAN 0.5*(eps_i + eps_j) (base ``mix``,
chosen over a geometric / Lorentz-Berthelot rule for its finite gradient at zeros/dead cells).

THE WHOLE-ENERGY CUTOFF. The smooth cutoff S multiplies the WHOLE energy (the core included),
not just the r^-6 tail. It is inert on the core only because the default r_on = 1.5*sigma > sigma
keeps S = 1 for r < sigma; a reimplementer who lowers ``r_onset_frac`` below 1 would start cutting
into the repulsive core. S is C1 (value and first derivative continuous), so the force stays
continuous with a small slope kink at the window edges. LJ has NO width/steepness knob (the 12-6
exponents fix the shape), so ``epsilon`` and the two cutoff fractions are the only tunables -- it
cannot be tuned to reproduce a given Morse alpha.

CONTRACT (``adhere``). This is the ``lennard_jones`` implementation of the pairwise cell-cell
mechanics contract that the family shares with ``soft_sphere`` (which minted it) and ``hertzian``:
a Lateral force between two cells whose RANGE is set by their physical radii (sigma = r_i + r_j
read from ``radius``), defined by a conservative pair energy autodiffed to a force, with a per-cell
OR shared coupling. LJ is an ADHESIVE member (a hard r^-12 core minus a 2 r^-6 tail, minimum -eps
at contact), so ``adhere`` names it directly; the repulsion-only ``soft_sphere`` / ``hertzian`` are
the adhesion-off limit of the SAME signature, and ``morse`` / ``harmonic`` are the other adhesive
members. NOT ``attraction_repulsion`` (the registered near-miss): that is the D'Orsogna
self-propelled law -- a hand-coded velocity from a two-Gaussian over an edge graph, a FIXED global
width sigma, per-TYPE params, and NO radius read -- so it cannot express a size-consistent contact
sigma = r_i + r_j nor a per-cell coupling.

SOURCE vs PAPER (rule 5, SOURCE WINS). The paper (Deshpande, Mottes et al. 2025; Methods p. 9
'Mechanical interactions', SI eq. V_ij p. 14, Fig. 1d) defines a SINGLE mechanical potential, the
ADHESIVE Morse well V_morse(r) = eps[(1 - exp(-alpha (r - sigma)))^2 - 1]. Lennard-Jones is a
code-only alternative in the potential library, exercised by NO paper experiment -- a reimplementer
targeting the paper would never instantiate this class; it is recorded as the source/paper
divergence. The per-cell virial pressure the source's base derives from this energy is a SEPARATE
mechanism (VirialStress, which writes a ``stress`` field) and is out of scope here, consistent with
Morse / Hertzian / SoftSphere.

Reference: jax-morph ``LennardJones``, papers/jax-morph/jax_morph/physics/mechanics/potentials.py:L419
(pair_energy -> eps*(x^12 - 2 x^6) * _smooth_cutoff at :L455; _smooth_cutoff at :L42; base
total_energy/forces at :L228/:L84; sigma = r_i + r_j at :L452). Physics: Lennard-Jones, J. E. (1924)
Proc. R. Soc. Lond. A 106:463-477. Paper: Deshpande, Mottes et al., "Engineering morphogenesis of
cell clusters with differentiable programming", Nat. Comput. Sci. (2025) -- Lennard-Jones is ABSENT
(paper mechanics is the adhesive Morse potential; SOURCE WINS).
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image


def _safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a / b` where `b != 0`, else 0 -- with a finite gradient at `b == 0` (the double-`where`
    trick: the division only ever sees a nonzero denominator). Mirrors jax-morph's `safe_divide`,
    which guards the r = 0 self-diagonal (sigma/r) and a dead-dead padded pair (sigma = 0 in the
    cutoff denominator) that would otherwise NaN a masked pair's gradient (0 * inf) in the
    backward pass."""
    safe_b = torch.where(b != 0.0, b, torch.ones_like(b))
    return torch.where(b != 0.0, a / safe_b, torch.zeros_like(a))


def _safe_norm(d2: torch.Tensor) -> torch.Tensor:
    """`sqrt(d2)` where `d2 > 0`, else 0 -- finite gradient at `d2 == 0` (the self-diagonal). The
    torch analogue of jax-morph's `safe_norm`: `sqrt` at 0 has an infinite derivative, so the
    zero-separation diagonal must be routed around it."""
    safe = torch.where(d2 > 0.0, d2, torch.ones_like(d2))
    return torch.where(d2 > 0.0, torch.sqrt(safe), torch.zeros_like(d2))


def _smooth_cutoff(r: torch.Tensor, r_on: torch.Tensor, r_off: torch.Tensor) -> torch.Tensor:
    """Multiplicative isotropic cutoff (jax-md form): a switch from 1 (r <= r_on) to 0 (r >= r_off).

    ``S(r) = (r_off^2 - r^2)^2 (r_off^2 + 2 r^2 - 3 r_on^2) / (r_off^2 - r_on^2)^3`` on the
    transition window. It is C1 (value and first derivative continuous at both ends), so the force
    stays continuous with a small slope kink at the window edges. A verbatim port of jax-morph's
    ``_smooth_cutoff`` (potentials.py:L42); ``_safe_divide`` guards the denominator for a dead-dead
    padded pair (sigma = 0 gives r_on = r_off = 0), which would otherwise be 0/0 and NaN a masked
    pair's gradient."""
    r2, ron2, roff2 = r * r, r_on * r_on, r_off * r_off
    s = _safe_divide((roff2 - r2) ** 2 * (roff2 + 2.0 * r2 - 3.0 * ron2), (roff2 - ron2) ** 3)
    return torch.where(r < r_on, torch.ones_like(s), torch.where(r < r_off, s, torch.zeros_like(s)))


@register_operator("adhere", family="interaction", set="cell", kind="lateral",
                   implementation="lennard_jones")
class LennardJones(Lateral):
    EMIT = "velocity"                           # overdamped: velocity = mobility * F, engine-integrated
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["position", "radius", "epsilon"]   # epsilon is a shared scalar param unless `epsilon_field` names a per-cell block
    WRITES = ["position"]                        # emits the interaction force as a velocity delta on `pos`
    MAPS = []                                    # dense within-set pairwise (N x N); no named gather map
    SUPPORTED_DIMS = [2, 3]                       # dimension-generic (reads D = pos.shape[-1]); energy is D-agnostic
    REQUIRES_PARAMS = []                          # epsilon defaults to 1.0 -- the sole optional coupling, like the source
    MECHANISM_TAGS = ["excluded_volume", "short_range_repulsion", "cell_cell_adhesion",
                      "adhesive_tail", "lennard_jones_potential", "size_consistent_contact",
                      "smooth_cutoff"]
    PARAM_ROLES = {"epsilon": "well_depth", "mobility": "overdamped_mobility",
                   "epsilon_field": "per_cell_well_depth_block", "radius": "fallback_cell_radius",
                   "r_onset_frac": "cutoff_onset_fraction", "r_cutoff_frac": "cutoff_end_fraction"}
    REFERENCE = (
        "jax-morph LennardJones, physics/mechanics/potentials.py:L419 "
        "(pair_energy -> eps*(x^12 - 2 x^6)*_smooth_cutoff at :L455; _smooth_cutoff at :L42; "
        "base total_energy/forces at :L228/:L84; sigma = r_i + r_j at :L452); "
        "Lennard-Jones, J. E. (1924) Proc. R. Soc. Lond. A 106:463-477; "
        "Deshpande, Mottes et al. (2025) Nat. Comput. Sci. (Lennard-Jones ABSENT; paper uses Morse -- SOURCE WINS)."
    )

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.epsilon = float(params.get("epsilon", 1.0))          # shared well depth (default 1.0)
        self.eps_field = params.get("epsilon_field", None)        # optional per-cell well-depth block/buffer name
        self.mobility = float(params.get("mobility", 1.0))        # overdamped mobility 1/gamma (velocity = mobility*F)
        self.radius0 = float(params.get("radius", 0.5))           # fallback uniform radius if no `radius` buffer
        # sigma-relative smooth-cutoff window (multiples of the contact distance), as in the source:
        self.r_onset_frac = float(params.get("r_onset_frac", 1.5))    # S = 1 inside this (tail untouched)
        self.r_cutoff_frac = float(params.get("r_cutoff_frac", 2.5))  # energy exactly 0 beyond this
        if not self.r_onset_frac < self.r_cutoff_frac:            # the source's construction-time check
            raise ValueError(
                f"LennardJones needs r_onset_frac < r_cutoff_frac for a smooth cutoff window, "
                f"got r_onset_frac={self.r_onset_frac}, r_cutoff_frac={self.r_cutoff_frac}.")

    def _read_epsilon(self, lvl, n, dev):
        """Per-pair well depth eps_ij as an [N, N] matrix: a per-cell field mixed by the arithmetic
        mean 0.5*(eps_i + eps_j), else the shared scalar broadcast. The additive contact distance
        keeps its OWN rule (r_i + r_j) -- do not mix sigma with the mean."""
        if self.eps_field is None:
            return self.epsilon                                   # scalar broadcasts over the [N, N] energy
        # read the named per-cell scalar (a state block if present, else a registered buffer)
        if self.eps_field in getattr(lvl, "state_schema", {}):
            v = lvl.get(self.eps_field)[:, 0]
        else:
            v = getattr(lvl, self.eps_field)
        v = v.reshape(n).to(dev)
        return 0.5 * (v[:, None] + v[None, :])                    # arithmetic-mean mix (finite grad at zeros)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        n = lvl.n
        a, b = lvl.state_schema["pos"]
        pos = lvl.state[:, a:b]                                   # [N, D] (a view of the integrated state)

        periodic = getattr(H, "periodic", False)
        world = getattr(H, "world_size", getattr(H, "world_width", 1.0))

        # contact distance sigma = r_i + r_j (ADDITIVE, size-consistent). Fall back to a uniform
        # radius if the set carries no per-cell `radius` buffer (keeps a radius-less spec runnable).
        radius = getattr(lvl, "radius", None)
        if radius is None:
            radius = torch.full((n,), self.radius0, device=dev)
        radius = radius.reshape(n).to(dev)
        sigma = radius[:, None] + radius[None, :]                 # [N, N]
        r_on = self.r_onset_frac * sigma                          # sigma-relative cutoff window
        r_off = self.r_cutoff_frac * sigma

        eps_pair = self._read_epsilon(lvl, n, dev)               # scalar or [N, N]

        # live non-self pair mask (the `neighbor_sum` seam): drop the self-diagonal and any pair
        # touching a dead cell, OUTSIDE the per-pair energy.
        alive = (lvl.occ > 0)
        eye = torch.eye(n, dtype=torch.bool, device=dev)
        pair_mask = alive[:, None] & alive[None, :] & ~eye        # [N, N]

        def energy(x):
            disp = minimum_image(x[:, None, :] - x[None, :, :], periodic, world)   # [N, N, D]
            d2 = (disp * disp).sum(-1)                                             # [N, N]
            r = _safe_norm(d2)                                                     # [N, N], 0 on the diagonal
            q = _safe_divide(sigma, r)                                             # sigma/r, 0 at r=0 (no inf)
            u_lj = eps_pair * (q ** 12 - 2.0 * q ** 6)                             # r_min form: min -eps at r=sigma
            u = u_lj * _smooth_cutoff(r, r_on, r_off)                              # tail truncated (whole-energy cutoff)
            u = torch.where(pair_mask, u, torch.zeros_like(u))                     # mask self + dead pairs
            return 0.5 * u.sum()                                                   # each unordered pair once

        # force = -grad_positions E, by autodiff of the conservative pair energy (the source's
        # `forces = -jax.grad(total_energy)`; there is no analytic LJ force in the source). Under a
        # differentiable rollout `pos` already carries grad and the force stays connected
        # (create_graph); otherwise differentiate a leaf clone cheaply inside a local enable_grad.
        outer_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            x = pos if pos.requires_grad else pos.detach().requires_grad_(True)
            E = energy(x)
            (grad_x,) = torch.autograd.grad(E, x, create_graph=outer_grad)
        force = -grad_x                                          # [N, D]

        vel = self.mobility * force * lvl.occ[:, None]           # overdamped velocity; dead cells emit nothing
        if mask is not None:
            vel = vel * mask[:, None].float()
        return {self.at: vel}
