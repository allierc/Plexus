"""adhere:soft_sphere -- cell <-> cell excluded volume (the harmonic SoftSphere implementation).

BIOLOGY. Two cells that overlap push apart: their membranes resist compression. The
harmonic soft sphere is the canonical soft-disk / active-matter excluded-volume law --
purely REPULSIVE (no adhesive tail), compact (exactly zero once the surfaces separate),
and the STIFFEST-at-contact of the compact family. Writing the contact distance as the
sum of the two cell radii, sigma = r_i + r_j, the pair energy is

    U(r) = (eps/2) (1 - r/sigma)^2       for r < sigma,   0 at and beyond contact,

so the excluded volume tracks cell SIZE: as a cell grows or divides its radius changes and
sigma follows, with no interaction-range parameter to retune. The exponent-2 (harmonic) core
makes the FORCE vanish at contact but leaves its SLOPE finite there (a C1 energy) -- that
nonzero force slope is the whole distinction from `adhere:hertzian`, whose 5/2 exponent makes
BOTH the force and its slope vanish at contact (C2, "deformable elastic"). This is the
adhesion-OFF, harmonic member of `adhere`.

ROUTING. `kind=lateral, family=interaction, set=cell` -- a within-set pairwise force, the
sibling of `attraction_repulsion` / `cohesion` / `separation`. It is a CONSERVATIVE energy:
rather than hand-code the radial force, we build the total pair energy and take the force by
AUTODIFF, exactly as the source does (`forces = -grad(total_energy)`). That is faithful to
the mechanism AND it realizes the one subtlety of the constant 1/2 -- see below. The paper's
mechanics is OVERDAMPED (gradient-descent `MechanicalRelaxation` / overdamped `BrownianDynamics`),
so the force is emitted as a velocity `EMIT="velocity"` (mobility * F, mobility=1 by default);
the engine integrates it, and it simply sums with any other velocity a cell carries (glide,
chemotax, another interaction). This matches `attraction_repulsion` (the registered near-miss),
which also emits an overdamped velocity.

WHY AUTODIFF, NOT A HAND-WRITTEN FORCE. The prefactor 1/2 = 1/exponent is not cosmetic: it
normalizes the radial force to a unit-per-sigma coefficient,

    f(r) = -dU/dr = (eps/sigma) (1 - r/sigma)      for r < sigma,   else 0,

so a reimplementer who writes `U = eps (1 - r/sigma)^2` and then differentiates gets a force
2x too strong. Taking the force by autodiff of the energy WITH the 1/2 in place reproduces the
intended f = (eps/sigma)(1 - r/sigma) automatically, and keeps the operator a genuine
conservative interaction (`hertzian` uses the same trick with 2/5 for exponent 5/2). Note the
TWO independent halves: this per-pair 1/2 (the harmonic prefactor) is DISTINCT from the leading
1/2 in the total energy (each undirected pair counted once over the dense N x N sum); folding
them into one under-counts either the energy or the stiffness.

DENSE, SIZE-CONSISTENT, MASKED. The sum runs over all live non-self pairs (dense N x N,
half-summed so each unordered pair counts once), the source's `neighbor_sum` seam. Two guards
keep the autodiff finite everywhere (both taken verbatim from the source, and shared with the
fractional-exponent siblings whose gradients they actually protect):
* a double-`where` on the base of the power -- the power only ever sees a strictly positive
  base, so its gradient is finite for r >= sigma too. At the integer exponent 2 this looks
  redundant, but it is what keeps the sibling Hertzian's 5/2 power NaN-free at r >= sigma under
  the source's always-on `jax_debug_nans`; it is kept here so the family shares one helper.
* a `safe_divide` for r/sigma and a `safe_norm` for the separation, so a dead-dead padded pair
  (sigma = 0) and the self-diagonal (r = 0) have a defined, finite gradient.
Dead cells and self-pairs are removed by an EXTERNAL alive-mask on the energy, not inside the
per-pair law: on the self-diagonal r = 0 gives base = 1 and the law returns a spurious 0.5*eps,
and for a dead-dead pair sigma = 0 gives base = 1 > 0 and again a spurious 0.5*eps -- both of
which the mask cancels; dropping the mask would inject phantom repulsion (a constant self-energy
offset and repulsion between dead cells). The contact distance uses the ADDITIVE rule
sigma = r_i + r_j (NOT the arithmetic mean); the per-cell stiffness `epsilon`, when it is a
field, is combined per pair by the ARITHMETIC MEAN 0.5*(eps_i + eps_j) (chosen over a geometric
mean for its finite gradient at zeros/dead cells). So within one pair epsilon mixes by MEAN
while sigma mixes by SUM -- two different combination rules.

CONTRACT (`adhere`, minted here). SoftSphere is the member that MINTS the pairwise cell-cell
CONTACT mechanics contract `adhere`: a Lateral force between two cells whose RANGE is set by
their physical radii (sigma = r_i + r_j read from `radius`), defined by a compact soft-core
energy autodiffed to a force, with a per-cell OR shared strength. The adhesive members (Morse,
LennardJones) add an attractive tail; SoftSphere and Hertzian switch it off -- Hertzian
(exponent 5/2) and Harmonic join SoftSphere as implementations of the SAME contract, differing
only in the pair-energy law U(r). NOT `attraction_repulsion` (the registered near-miss): that
is the D'Orsogna self-propelled law -- a hand-coded velocity from a two-Gaussian over an edge
graph, a FIXED global width sigma, per-TYPE params, and NO radius read -- so it cannot express a
size-consistent contact sigma = r_i + r_j nor a per-cell strength.

SOURCE vs PAPER (rule 5, SOURCE WINS). The paper (Deshpande, Mottes et al. 2025; Methods p. 9,
SI eq. V_ij p. 14, Fig. 1) names its cells "adhesive soft spheres" but realizes them with the
ADHESIVE Morse well (V_ij = eps[(1 - exp(-alpha(r - sigma)))^2 - 1]), NOT this harmonic
(eps/2)(1 - r/sigma)^2. This SoftSphere is a code-only alternative in the potential library --
purely repulsive, so it cannot reproduce the paper's cell-cell adhesion at all; the paper
describes it in name only. The per-cell virial pressure the source's base derives from this
energy is a SEPARATE mechanism (VirialStress, which writes a `stress` field) and is out of
scope here.

Reference: jax-morph `SoftSphere`, papers/jax-morph/jax_morph/physics/mechanics/potentials.py:L315
(pair_energy -> `_compact_repulsion(r, sigma, eps, 2.0, 0.5)` at :L342; base `total_energy`/
`forces` at :L228/:L84; sigma = r_i + r_j at :L337). Physics: Durian, D. J. (1995) Phys. Rev.
Lett. 75:4780 (harmonic soft disks / bubble model). Paper: Deshpande, Mottes et al.,
"Engineering morphogenesis of cell clusters with differentiable programming", Nat. Comput. Sci.
(2025) -- harmonic SoftSphere is ABSENT (paper mechanics is the adhesive Morse potential; SOURCE WINS).
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image


def _safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a / b` where `b != 0`, else 0 -- with a finite gradient at `b == 0` (the double-`where`
    trick: the division only ever sees a nonzero denominator). Mirrors jax-morph's `safe_divide`,
    which guards a dead-dead padded pair (sigma = 0) that would otherwise NaN the gradient."""
    safe_b = torch.where(b != 0.0, b, torch.ones_like(b))
    return torch.where(b != 0.0, a / safe_b, torch.zeros_like(a))


def _safe_norm(d2: torch.Tensor) -> torch.Tensor:
    """`sqrt(d2)` where `d2 > 0`, else 0 -- finite gradient at `d2 == 0` (the self-diagonal). The
    torch analogue of jax-morph's `safe_norm`: `sqrt` at 0 has an infinite derivative, so the
    zero-separation diagonal must be routed around it."""
    safe = torch.where(d2 > 0.0, d2, torch.ones_like(d2))
    return torch.where(d2 > 0.0, torch.sqrt(safe), torch.zeros_like(d2))


def _compact_repulsion(r, sigma, eps, exponent: float, prefactor: float):
    """`prefactor * eps * (1 - r/sigma)**exponent` for `r < sigma`, else 0 (value AND grad safe).

    Verbatim port of jax-morph's `_compact_repulsion` (potentials.py:L30). `_safe_divide` handles
    a dead-dead padded pair (sigma = 0); the double `where` evaluates the (possibly fractional)
    power only on a strictly positive base, so the gradient is finite for `r >= sigma` too."""
    base = 1.0 - _safe_divide(r, sigma)
    safe = torch.where(base > 0.0, base, torch.ones_like(base))   # the power only sees a positive base
    return torch.where(base > 0.0, prefactor * eps * safe ** exponent, torch.zeros_like(base))


@register_operator("adhere", family="interaction", set="cell", kind="lateral",
                   implementation="soft_sphere")
class SoftSphere(Lateral):
    EMIT = "velocity"                           # overdamped: velocity = mobility * F, engine-integrated
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["pos", "radius", "epsilon", "alive"]   # epsilon is a shared scalar param unless `epsilon_field` names a per-cell block
    WRITES = ["pos"]                        # emits the contact force as a velocity delta on `pos`
    MAPS = []                                    # dense within-set pairwise (N x N); no named gather map
    SUPPORTED_DIMS = [2, 3]                       # dimension-generic (reads D = pos.shape[-1]); energy is D-agnostic
    REQUIRES_PARAMS = []                          # epsilon defaults to 1.0 -- the sole optional coupling, like the source
    MECHANISM_TAGS = ["excluded_volume", "short_range_repulsion", "harmonic_soft_core",
                      "compact_support", "size_consistent_contact"]
    PARAM_ROLES = {"epsilon": "repulsion_strength", "mobility": "overdamped_mobility",
                   "epsilon_field": "per_cell_strength_block", "radius": "fallback_cell_radius"}
    REFERENCE = (
        "jax-morph SoftSphere, physics/mechanics/potentials.py:L315 "
        "(pair_energy -> _compact_repulsion(r, sigma, eps, 2.0, 0.5) at :L342; "
        "base total_energy/forces at :L228/:L84; sigma = r_i + r_j at :L337); "
        "Durian, D. J. (1995) Phys. Rev. Lett. 75:4780 (harmonic soft disks); "
        "Deshpande, Mottes et al. (2025) Nat. Comput. Sci. (harmonic SoftSphere ABSENT; paper uses Morse -- SOURCE WINS)."
    )

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.epsilon = float(params.get("epsilon", 1.0))          # shared repulsion strength (default 1.0)
        self.eps_field = params.get("epsilon_field", None)        # optional per-cell strength block/buffer name
        self.mobility = float(params.get("mobility", 1.0))        # overdamped mobility 1/gamma (velocity = mobility*F)
        self.radius0 = float(params.get("radius", 0.5))           # fallback uniform radius if no `radius` buffer
        # the compact-repulsion constants are FIXED by the harmonic soft-sphere law, not tunables:
        self._exponent = 2.0                                      # energy ~ overlap^2 (harmonic core)
        self._prefactor = 0.5                                     # 1/2 = 1/exponent -> force coefficient = eps/sigma

    def _read_epsilon(self, lvl, n, dev):
        """Per-pair strength eps_ij as an [N, N] matrix: a per-cell field mixed by the arithmetic
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
            radius = torch.ones(n, device=dev) * self.radius0  # ones*p, not full(p): `full` rejects a tensor fill value, blocking a learnable knob
        radius = radius.reshape(n).to(dev)
        sigma = radius[:, None] + radius[None, :]                 # [N, N]

        eps_pair = self._read_epsilon(lvl, n, dev)               # scalar or [N, N]

        # live non-self pair mask (the `neighbor_sum` seam): drop the self-diagonal and any pair
        # touching a dead cell, OUTSIDE the per-pair energy (the self-diagonal and a dead-dead pair
        # each score a spurious 0.5*eps, not zero).
        alive = (lvl.occ > 0)
        eye = torch.eye(n, dtype=torch.bool, device=dev)
        pair_mask = alive[:, None] & alive[None, :] & ~eye        # [N, N]

        def energy(x):
            disp = minimum_image(x[:, None, :] - x[None, :, :], periodic, world)   # [N, N, D]
            d2 = (disp * disp).sum(-1)                                             # [N, N]
            r = _safe_norm(d2)                                                     # [N, N], 0 on the diagonal
            u = _compact_repulsion(r, sigma, eps_pair, self._exponent, self._prefactor)
            u = torch.where(pair_mask, u, torch.zeros_like(u))                     # mask self + dead pairs
            return 0.5 * u.sum()                                                   # each unordered pair once

        # force = -grad_positions E, by autodiff of the conservative pair energy (the source's
        # `forces = -jax.grad(total_energy)`). Under a differentiable rollout `pos` already carries
        # grad, so the force stays CONNECTED to the outer graph (create_graph); otherwise it is a
        # detached leaf clone differentiated cheaply inside a local enable_grad.
        connected = pos.requires_grad and torch.is_grad_enabled()
        with torch.enable_grad():
            x = pos if connected else pos.detach().requires_grad_(True)
            E = energy(x)
            (grad_x,) = torch.autograd.grad(E, x, create_graph=connected)
        force = -grad_x                                          # [N, D], repulsive (points away from neighbours)

        vel = self.mobility * force * lvl.occ[:, None]           # overdamped velocity; dead cells emit nothing
        if mask is not None:
            vel = vel * mask[:, None].float()
        return {self.at: vel}
