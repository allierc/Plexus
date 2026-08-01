r"""adhere:harmonic -- cell (lateral). A pairwise cell-cell MECHANICAL interaction: a
finite-range harmonic spring between every live cell pair, with a repulsive core and an
adhesive tail, the force read off the physical cell radius so it stays size-consistent as
cells grow and divide.

BIOLOGY. Cells in a cluster push apart when they overlap (excluded volume) and pull together
when adhesion molecules on their membranes bridge a small gap. `adhere` is that pairwise
mechanics; the `harmonic` implementation gives it the simplest shape that carries BOTH halves
at once -- a parabola with its minimum sitting exactly at the contact distance
sigma = r_i + r_j (the sum of the two cell radii):

    U(r) = 0.5 * k * [ (r - sigma)^2 - (r_c - sigma)^2 ]     for r < r_c
    U(r) = 0                                                 for r >= r_c
      with cutoff  r_c = r_cutoff_frac * sigma

The bracket is a plain spring energy 0.5*k*(r-sigma)^2 SHIFTED DOWN by 0.5*k*(r_c-sigma)^2 so
that it vanishes exactly at the cutoff. That down-shift is the whole point: it makes the well
minimum NEGATIVE at contact (depth 0.5*k*(r_c-sigma)^2) and lets the energy reach 0 at r_c, so
the finite-range interaction is ADHESIVE (attractive for sigma < r < r_c), not a bare repulsive
spring. Drop the shift and you lose both the adhesion and the truncation.

THE FORCE (what this operator emits). The radial pair force is the negative energy gradient,

    F(r) = -dU/dr = k * (sigma - r)      for r < r_c,   0 beyond,

so the force on cell i from a neighbour j is  F(r_ij) * (x_i - x_j)/r_ij, summed over live
j != i:
  * r < sigma  (compressed) -> sigma - r > 0 -> force points APART   -> repulsion (excluded volume)
  * sigma < r < r_c (stretched) -> sigma - r < 0 -> force points TOGETHER -> adhesion
  * r >= r_c    -> zero.
At contact r = sigma the force is exactly zero -- the mechanical rest state. The interaction is
a CONSERVATIVE central force, so it conserves total momentum (Newton's third law: the pair
force on i is minus the pair force on j); a spatially uniform cluster feels no net drift.

ROUTING (why `velocity`, and how it composes). The reference potential returns a FORCE; the
wrapping overdamped-dynamics step turns it into motion by the mobility 1/gamma
(dx = -(grad U / gamma) dt = (F/gamma) dt). Following this campaign's split of that composite
(see `agitate`, the Brownian bath), the drift lives HERE: `adhere` emits the overdamped drift
velocity v = mobility * F (mobility = 1/gamma, default 1.0 = the reference's gamma = 1), and the
engine integrates pos += dt * v. Schedule it alongside `agitate` and their velocities simply
sum, reconstructing one Euler-Maruyama step of overdamped Langevin dynamics; with no bath it is
the deterministic gradient-descent relaxation of the paper's mechanics. It is a single-body-
free, within-set pairwise morphism -> kind `lateral`, family `interaction`, set `cell`.

SIZE-CONSISTENT, PER-CELL. Two combining rules coexist in one potential. The contact distance
uses an ADDITIVE rule sigma = r_i + r_j read from the physical per-cell `radius`, so both the
excluded-volume core and the adhesion range TRACK cell growth and division (load-bearing for a
morphogenesis model whose cells grow and divide). The stiffness k is either a shared scalar or a
per-cell field, symmetrised per pair by the arithmetic-MEAN mix 0.5*(k_i + k_j) -- deliberately
NOT the geometric (Lorentz-Berthelot) mean, so the gradient stays finite at zeros and dead cells
(no sqrt). A single k sets BOTH the repulsion slope and the adhesion depth at once (a narrower
knob set than Morse's independent depth/steepness).

FAITHFUL SURPRISE: C0-ONLY CUTOFF. Unlike the Morse / Lennard-Jones members of `adhere` (which
multiply by a C1 smooth cutoff) and unlike the purely-repulsive SoftSphere / Hertzian shapes
(compact C1 tails), `harmonic` hard-truncates the energy with a bare where at r_c, so the FORCE
is DISCONTINUOUS there: it jumps from k*(sigma - r_c) = -k*(r_c - sigma) up to 0. This is
preserved verbatim, not "fixed" with a smoothing (a smoothing would change the force law); it is
harmless in practice because r_c = 2.5*sigma sits well beyond the resting contact distance.

WHY `new`, contract `adhere` (NOT an alias of attraction_repulsion). Harmonic is one force-law
SHAPE of a pairwise cell-cell mechanical-interaction contract the frozen language lacks -- the
same contract its sibling SoftSphere named `adhere`. It reads the physical `radius` to build a
per-pair contact distance, takes a per-cell stiffness, and is ENERGY-defined (force =
-grad(total_energy)). The nearest frozen operator, attraction_repulsion, is a D'Orsogna self-
propelled-particle law: a fixed GLOBAL sigma with per-TYPE params, no `radius`, a hand-coded
two-Gaussian velocity, set = particle. Same "long pull, short push" force profile, different
typed signature -- so a new contract, with the five PairwisePotential shapes (Morse, SoftSphere,
Hertzian, Harmonic, LennardJones) collapsing to that ONE contract, differing only in U(r).

SOURCE vs PAPER (rule 5, SOURCE WINS). NOT IN THE PAPER: the paper (Deshpande, Mottes et al.
2025, Methods p. 9 and SI 'Mechanical Interactions' p. 14) defines a SINGLE mechanical
potential, the Morse well V_ij(r) = eps*[(1 - exp(-alpha(r - sigma)))^2 - 1]. Harmonic -- and
SoftSphere, Hertzian, Lennard-Jones -- are library-only alternative pair potentials sharing the
same sigma = r_i + r_j contact convention; a reimplementer from the paper alone would implement
only Morse and never encounter this class. The SOURCE carries it, and it normalizes to the same
`adhere` contract Morse's true home is.

Translated from papers/jax-morph/jax_morph/physics/mechanics/potentials.py:L375 (Harmonic;
pair_energy at :L412-416, base total_energy/forces at :L228-266). Physics: a shifted finite-range
harmonic (soft-disk adhesion) potential. Torch, not JAX.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("adhere", family="interaction", set="cell", kind="lateral",
                   implementation="harmonic")
class AdhereHarmonic(Lateral):
    r"""`harmonic` implementation of the `adhere` contract: a finite-range shifted harmonic
    spring (repulsive core + adhesive tail), the force emitted as an overdamped drift velocity.

    Siblings under the same contract differ only in the pair-energy law U(r): Morse and
    Lennard-Jones are the other adhesion-ON shapes (C1 smooth cutoff); SoftSphere and Hertzian
    are the adhesion-OFF, purely-repulsive shapes. Select this one with
    `{op: adhere, implementation: harmonic}`."""

    EMIT = "velocity"                            # a force -> overdamped drift v = mobility*F; engine integrates pos += dt*v
    # typed signature (Plexus2 sec. 2.1): a within-set pairwise cell -> cell morphism. Reads
    # pos + the physical radius (to form the per-pair contact distance) + alive (occ, to drop
    # dead/padded slots); writes the pos delta. No gather MAP: the pairwise reduction is a dense
    # N x N over the set itself, not a traversal of a named map.
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["pos", "radius", "alive"]
    WRITES = ["pos"]
    MAPS = []
    SUPPORTED_DIMS = [2, 3]                       # dimension-generic: reads D = pos.shape[-1], no hard-coded 2
    REQUIRES_PARAMS = []                          # every knob optional (k / r_cutoff_frac / radius default to the source's)
    MECHANISM_TAGS = ["excluded_volume", "cell_adhesion", "pairwise_potential",
                      "size_consistent_contact", "short_range_repulsion", "finite_range_attraction",
                      "energy_defined_force"]
    PARAM_ROLES = {"k": "interaction_stiffness", "r_cutoff_frac": "interaction_range",
                   "mobility": "overdamped_mobility", "radius": "fallback_cell_radius",
                   "k_field": "per_cell_stiffness_field"}
    REFERENCE = (
        "Deshpande, Mottes, Vidal Saez, Kicheva & Hiscock (2025), 'Engineering morphogenesis of "
        "cell clusters with differentiable programming', Nat Comput Sci -- the paper (Methods p. 9, "
        "SI 'Mechanical Interactions' p. 14) defines ONLY the Morse potential; Harmonic is a "
        "library-only pair-potential shape (SOURCE WINS, rule 5). Translated from "
        "papers/jax-morph/jax_morph/physics/mechanics/potentials.py:L375 (Harmonic.pair_energy "
        ":L412-416; PairwisePotential.total_energy/forces :L228-266, sigma = r_i + r_j, "
        "arithmetic-mean k mix). Physics: a shifted finite-range harmonic (soft-disk) potential."
    )

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")                        # the set this acts on (engine-injected)
        self.k = self.tunable(params.get("k"), 1.0)                       # spring stiffness (scalar fallback / uniform)
        self.k_field = params.get("k_field", None)                 # optional per-cell stiffness field (block or buffer)
        self.r_cutoff_frac = self.tunable(params.get("r_cutoff_frac"), 2.5)  # cutoff as a multiple of contact distance sigma
        self.mobility = self.tunable(params.get("mobility"), 1.0)         # 1/gamma: force -> overdamped drift velocity
        self.radius0 = self.tunable(params.get("radius"), 0.5)            # uniform fallback radius if the set carries no `radius`
        # faithful construction check: the clamp must sit BEYOND contact, else the "finite range"
        # is empty and the down-shift is ill-defined (source raises the same way).
        if not self.r_cutoff_frac > 1.0:
            raise ValueError(
                f"adhere:harmonic needs r_cutoff_frac > 1 (the cutoff must sit beyond contact), "
                f"got r_cutoff_frac={self.r_cutoff_frac}.")
        if self.mobility < 0.0:
            raise ValueError(f"adhere:harmonic mobility (1/gamma) must be >= 0, got {self.mobility}.")
        self._eps2 = 1e-24                                         # floor on the squared distance (self/coincident pairs)

    def _read_scalar(self, lvl, name):
        """A per-cell [N] view of a named quantity: a state block if the schema declares one,
        else a registered per-node buffer of that name, else None (same recipe as grow_radius)."""
        if name is None:
            return None
        if name in lvl.state_schema:
            v = lvl.get(name)
        else:
            v = getattr(lvl, name, None)
        if v is None or not torch.is_tensor(v):
            return None
        return v.reshape(v.shape[0], -1)[:, 0]                     # [N] (first component if width > 1)

    def _pair_geometry(self, lvl, pos):
        """Per-pair (N,N) arrays shared by the force and the energy: safe distance r, contact
        distance sigma = r_i + r_j, mixed stiffness k_ij, the interaction mask (live, non-self,
        inside cutoff), the per-pair occupancy, and the displacement pos_i - pos_j."""
        N = pos.shape[0]
        dev = pos.device
        rad = self._read_scalar(lvl, "radius")
        if rad is None:
            rad = torch.ones(N, device=dev, dtype=pos.dtype) * self.radius0  # ones*p, not full(p): `full` rejects a tensor fill value, blocking a learnable knob
        sigma = rad[:, None] + rad[None, :]                        # [N,N] additive contact rule

        kv = self._read_scalar(lvl, self.k_field)                 # per-cell k, if a field was named
        if kv is None:
            k_ij = torch.ones(N, N, device=dev, dtype=pos.dtype) * self.k  # ones*p, not full(p): `full` rejects a tensor fill value, blocking a learnable knob
        else:
            k_ij = 0.5 * (kv[:, None] + kv[None, :])               # arithmetic-mean mix (no sqrt: finite grad)

        diff = pos[:, None, :] - pos[None, :, :]                   # [N,N,D]  pos_i - pos_j (points j -> i)
        if getattr(self, "_periodic", False):
            diff = torch.remainder(diff + 0.5, 1.0) - 0.5          # minimum image on the unit torus
        sq = (diff * diff).sum(-1)                                 # [N,N] squared distance
        r = torch.sqrt(sq.clamp(min=self._eps2))                  # safe norm: diagonal r -> ~0 with zero grad

        occ = lvl.occ.to(pos.dtype)
        occ_pair = occ[:, None] * occ[None, :]                    # 0 unless BOTH cells are live
        r_c = self.r_cutoff_frac * sigma
        eye = torch.eye(N, device=dev, dtype=torch.bool)
        within = (r < r_c) & (~eye)                               # inside cutoff, non-self (dead pairs masked by occ)
        return diff, r, sigma, k_ij, r_c, occ_pair, within

    def total_energy(self, H):
        """Total interaction energy 0.5 * sum_{i,j live, i!=j} U(r_ij) -- the conservative energy
        whose negative position-gradient is the force this operator emits. Exposed for tests /
        diagnostics (the `adhere` contract is energy-defined); `forward` uses the analytic force."""
        lvl = H.level(self.at)
        self._periodic = getattr(H, "periodic", False)
        pos = lvl.get("pos")
        _, r, sigma, k_ij, r_c, occ_pair, within = self._pair_geometry(lvl, pos)
        u = 0.5 * k_ij * ((r - sigma) ** 2 - (r_c - sigma) ** 2)   # 0 at r_c, negative inside (down-shifted)
        u = torch.where(within, u, torch.zeros_like(u)) * occ_pair
        return 0.5 * u.sum()                                       # 0.5: each unordered pair counted once

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        self._periodic = getattr(H, "periodic", False)
        pos = lvl.get("pos")                                       # [N, D]
        diff, r, sigma, k_ij, r_c, occ_pair, within = self._pair_geometry(lvl, pos)

        # radial force magnitude F(r) = -dU/dr = k*(sigma - r), truncated to the interaction mask.
        f_mag = k_ij * (sigma - r)                                 # [N,N]  >0 repulsive (r<sigma), <0 adhesive
        f_mag = torch.where(within, f_mag, torch.zeros_like(f_mag)) * occ_pair
        unit = diff / r[..., None]                                 # [N,N,D] from j to i (r is floored, safe)
        force = (f_mag[..., None] * unit).sum(dim=1)              # [N, D] sum over neighbours j

        v = self.mobility * force                                  # overdamped drift velocity v = F/gamma
        v = v * lvl.occ.to(v.dtype)[:, None]                      # dead cells receive no drift
        if mask is not None:                                      # `at:` may restrict the acting subset
            v = v * mask.to(v.dtype)[:, None]
        return {self.at: v}
