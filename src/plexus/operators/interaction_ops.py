"""Lateral laws on a particle set, and the Rewire that builds the relation they read.

Every contract here acts on one `particle` set and reads that set's `edge_index`, the neighbour
relation; `radius_graph` is the Rewire that writes it, and it must be scheduled before any law
below it.

In the order they appear below:

    attraction_repulsion  lateral   two competing Gaussians: dispersed, clustered or crystalline
    squared_law           lateral   inverse square: gravity between masses, Coulomb between charges
    cohesion              lateral   steer toward the mean neighbour position        (boids)
    separation            lateral   steer away from the closest neighbours          (boids)
    velocity_align        lateral   steer toward the mean neighbour velocity        (boids, Vicsek)
    stillinger_weber      lateral   a two-body well plus a three-body angular penalty
    radius_graph          rewire    the relation: which particles count as neighbours

then the alternative implementations, which change only the numerics:

    squared_law[warp]     one Warp kernel: O(N^2) arithmetic, O(N) memory
    squared_law[mesh]     particle-mesh: deposit, FFT Poisson, gather -- O(N + M log M)
"""
from __future__ import annotations
import torch
from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image
from plexus.geometry import neighbour_mean
from plexus.models.base import Rewire
from plexus.geometry import edges_radius_blockwise

import math

# `warp` is optional. Two of `squared_law`'s implementations are Warp kernels; `HAVE_WARP` gates
# every kernel definition below so the module still imports where warp is absent. One guard and
# one `wp.init()` for the whole module.
try:
    import warp as wp
    wp.init()
    HAVE_WARP = True
except Exception:
    HAVE_WARP = False


@register_operator("attraction_repulsion", family="interaction", set="particle", kind="lateral")
class AttractionRepulsion(Lateral):
    """Soft-core attraction-repulsion: a smooth pairwise law whose two competing Gaussians,
    a long-range pull and a short-range push, set the phase the set settles into.

    particle -[neighbour relation]-> particle: reads pos, emits a velocity.

        dx_i/dt = mean_j  f(r_ij) (x_j - x_i)
        f(r)    = p1 exp(-(r^2)^p2 / 2 sigma^2)  -  p3 exp(-(r^2)^p4 / 2 sigma^2)

    r_ij = |x_j - x_i| is the distance between the pair and sigma the interaction length, both
    in world units. p1 is the strength of the long-range pull and p2 its range exponent; p3 is
    the strength of the short-range push and p4 its range exponent. All four are dimensionless
    and carried per receiver type, so different types obey different laws in the same set. The
    pull dominating at large r and the push at small r is what fixes a stable spacing instead
    of a collapse; the balance of the two is the phase (dispersed, clustered, crystalline).

    Emits a velocity because the law is overdamped: friction is assumed to dominate inertia,
    so the force IS the velocity. `aggr: mean` (the default) divides by the live-neighbour
    count, making the result independent of local density; `aggr: sum` does not, so a denser
    region then feels a proportionally larger drift.

    Reference: D'Orsogna, M. R., Chuang, Y. L., Bertozzi, A. L. & Chayes, L. S. (2006).
    Self-propelled particles with soft-core interactions: patterns, stability and collapse.
    Phys. Rev. Lett. 96:104302.
    """

    EMIT = "velocity"             # overdamped: the force IS the velocity, no inertia
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic (reads D = pos.shape[-1])
    REQUIRES_PARAMS = ["sigma"]                 # the cutoff lives on the radius_graph rewire op
    REQUIRES_TYPE_PROPS = ["p"]                 # per-type force-law params [p1,p2,p3,p4]
    MECHANISM_TAGS = ["long_range_attraction", "short_range_repulsion", "coarsening", "lattice_forming"]
    PARAM_ROLES = {"sigma": "interaction_length", "noise": "exploration_noise",
                   "p": "[pull_strength, pull_range, push_strength, push_range] per type"}
    REFERENCE = ("D'Orsogna, M. R., Chuang, Y. L., Bertozzi, A. L. & Chayes, L. S. (2006). "
                 "Self-propelled particles with soft-core interactions: patterns, stability "
                 "and collapse. Phys. Rev. Lett. 96:104302.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.sigma = float(params["sigma"])
        self.aggr = params.get("aggr", "mean")               # mean (default, matches the reference) or sum
        self.noise = float(params.get("noise", 0.0))         # isotropic velocity noise (off by default)
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        occ = lvl.occ
        N, D = pos.shape[0], pos.shape[-1]
        ei = lvl.edge_index                                  # [2, E]: row0 = receiver i, row1 = neighbour j
        if ei.numel() == 0:
            return {self.at: torch.zeros(N, D, device=pos.device)}
        i, j = ei[0], ei[1]

        d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False),
                          getattr(H, "world_size", getattr(H, "world_width", 1.0)))   # j - i  [E, D]
        r2 = (d * d).sum(-1)                                  # [E]
        p = lvl.type_params[lvl.node_type[i]]                # receiver-type params [E, 4]
        s2 = 2.0 * self.sigma ** 2
        f = (p[:, 0] * torch.exp(-(r2 ** p[:, 1]) / s2)
             - p[:, 2] * torch.exp(-(r2 ** p[:, 3]) / s2))   # [E]
        f = f * occ[j]                                       # ignore dormant neighbours

        dpos = torch.zeros(N, D, device=pos.device)
        dpos.index_add_(0, i, f[:, None] * d)                # aggregate at the receiver
        if self.aggr == "mean":                              # average over neighbours (density-independent)
            deg = torch.zeros(N, device=pos.device).index_add_(0, i, occ[j])
            dpos = dpos / deg.clamp(min=1.0)[:, None]
        dpos = dpos * occ[:, None]
        if self.noise > 0.0:                                 # exploratory noise on the overdamped velocity
            dpos = dpos + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None),
                                                   device=pos.device) * occ[:, None]
        if mask is not None:
            dpos = dpos * mask[:, None].float()
        return {self.at: dpos}


def _inv_square_sum(pos, src, soft2):
    """The unsigned all-pairs sum of `SquaredLaw`, before the law's own sign and coupling:

        pull_i = sum_j src_j (x_j - x_i) / (|x_j - x_i|^2 + soft2)^(3/2)

    A per-particle vector, not a Plexus `Field` -- no grid is involved. The caller scales it by
    the signed strength and the receiver's coupling to get the acceleration. `src` folds in
    occupancy, so a dormant particle contributes nothing. Written per dimension, so only [N, N]
    arrays appear and never an [N, N, D] one, and kept a free function so a fusing backend can
    collapse the reduction and materialise neither."""
    N, D = pos.shape
    r2 = torch.full((N, N), soft2, device=pos.device, dtype=pos.dtype)
    for k in range(D):
        dk = pos[:, k].unsqueeze(0) - pos[:, k].unsqueeze(1)       # dk[i,j] = pos[j]-pos[i]
        r2 = r2 + dk * dk
    inv_r3 = r2.clamp(min=1e-12).pow(-1.5)                          # diagonal dk=0 -> 0 contribution
    pull = torch.empty(N, D, device=pos.device, dtype=pos.dtype)
    for k in range(D):
        dk = pos[:, k].unsqueeze(0) - pos[:, k].unsqueeze(1)
        pull[:, k] = (dk * inv_r3) @ src
    return pull


@register_operator("squared_law", family="interaction", set="particle", kind="lateral")
class SquaredLaw(Lateral):
    """The inverse-square law between particles: Newtonian gravity between masses, or
    Coulomb electrostatics between signed charges, as one contract with two conventions.

    particle -[all pairs, or the neighbour relation]-> particle: reads pos and one per-type
    source property (mass or charge), emits an acceleration.

        a_i = s k c_i  sum_j  q_j (x_j - x_i) / (|x_j - x_i|^2 + eps^2)^(3/2)

    q_j is the source property of the neighbour -- mass for gravity, signed charge for Coulomb
    -- and c_i the receiver's coupling to it. The two laws differ only in s and c_i:

        law: gravity   s = +1,  c_i = 1     the receiver's own mass cancels (equivalence
                                            principle), so every body falls the same way
        law: coulomb   s = -1,  c_i = q_i   the receiver's own charge scales its acceleration,
                                            and the sign makes like charges repel

    k is the strength constant (Newton's G, or the Coulomb constant) in whatever units the
    specification is written in; eps is the Plummer softening length, in world units, which
    caps the force a near-coincident pair can exert -- eps = 0 gives the bare 1/r^2. `clamp`
    is a further cap on the magnitude of a_i itself, a stability device rather than physics.

    `all_pairs: true` sums over every pair, which is what a long-range law means and costs
    O(N^2); it requires open boundaries, since there is no minimum image over all pairs.
    `all_pairs: false` sums only over the neighbour relation, turning the same law into a
    screened short-range one whose cutoff is the rewire's radius.

    Reference: Newton, I. (1687). Philosophiae Naturalis Principia Mathematica (the
    inverse-square law of gravitation); Coulomb, C.-A. (1785). Premier memoire sur
    l'electricite et le magnetisme. Hist. Acad. R. Sci., 569-577. Plummer softening:
    Plummer, H. C. (1911). Mon. Not. R. Astron. Soc. 71:460-470.
    """

    EMIT = "acceleration"                        # second-order: masses and charges have inertia
    SUPPORTED_DIMS = [2, 3]                       # dimension-generic (reads D = pos.shape[-1])
    REQUIRES_PARAMS = []                          # no required params — all knobs optional (defaults in __init__)
    OPTIONAL_TYPE_PROPS = ["charge", "mass"]     # reads ONE (self.coupling), chosen by `law`
    MECHANISM_TAGS = ["inverse_square", "electrostatics", "gravity", "newtonian_gravity",
                      "long_range", "self_gravity"]
    PARAM_ROLES = {"law": "coulomb (signed charge, like-repel) | gravity (mass, attract)",
                   "k": "strength constant (Coulomb constant / Newton's G)",
                   "coupling": "per-type source property (charge|mass)",
                   "softening": "Plummer softening length eps (0 = pure 1/r^3)",
                   "all_pairs": "sum over ALL pairs (O(N^2), long-range) vs the neighbour graph",
                   "clamp": "max |acceleration| (0 = unbounded)"}
    REFERENCE = ("Newton, I. (1687). Philosophiae Naturalis Principia Mathematica; "
                 "Coulomb, C.-A. (1785). Premier memoire sur l'electricite et le magnetisme. "
                 "Hist. Acad. R. Sci., 569-577; softening: Plummer, H. C. (1911). "
                 "Mon. Not. R. Astron. Soc. 71:460-470.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.law = str(params.get("law", "coulomb"))               # coulomb | gravity
        if self.law not in ("coulomb", "gravity"):
            raise ValueError(f"squared_law: law must be 'coulomb' or 'gravity', got {self.law!r}")
        self.k = float(params.get("k", 1.0))                       # strength (Coulomb const / Newton G)
        self.coupling = str(params.get("coupling",                # per-type source property
                                       "charge" if self.law == "coulomb" else "mass"))
        self.soft = float(params.get("softening", 0.0))           # Plummer eps (0 = pure 1/r^3)
        self.all_pairs = bool(params.get("all_pairs", False))     # O(N^2) long-range vs neighbour graph
        self.clamp = float(params.get("clamp", 0.0))              # optional cap on |a| (0 = off)
        # physical conventions bundled by `law`: (sign) like-repel vs attract; (receiver) whether the
        # receiver's own coupling charge scales its acceleration (Coulomb) or cancels (gravity).
        self.sign = -1.0 if self.law == "coulomb" else 1.0
        self._recv_coupled = (self.law == "coulomb")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        occ = lvl.occ
        N, D = pos.shape[0], pos.shape[-1]
        s = getattr(lvl, self.coupling, None)                     # per-particle source charge (charge|mass)
        if s is None:
            raise ValueError(f"squared_law(law={self.law}) needs per-type property "
                             f"{self.coupling!r} on {self.at!r}; declare it in the set's `types`.")

        if self.all_pairs:
            # --- long-range O(N^2): pull_i = Σ_j src_j (r_j-r_i)/denom ; a_i = sign*k*recv_i*pull_i
            if getattr(H, "periodic", False):
                raise ValueError("squared_law all_pairs=True supports only open/free boundaries "
                                 "(no minimum-image over all pairs); use a neighbour graph if periodic.")
            src = s * occ                                          # dormant particles contribute nothing
            pull = _inv_square_sum(pos, src, self.soft ** 2)
            recv = s if self._recv_coupled else torch.ones(N, device=pos.device, dtype=pos.dtype)
            acc = (self.sign * self.k) * recv[:, None] * pull
        else:
            # --- neighbour graph O(E) over Level.edge_index (row0 = receiver i, row1 = source j)
            ei = lvl.edge_index
            if ei.numel() == 0:
                return {self.at: torch.zeros(N, D, device=pos.device)}
            i, j = ei[0], ei[1]
            d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False),
                              getattr(H, "world_size", getattr(H, "world_width", 1.0)))   # j - i  [E, D]
            r = d.norm(dim=-1).clamp(min=1e-6)                    # |d| (off zero via the graph's min_radius)
            if self.law == "coulomb" and self.soft == 0.0:
                # unsoftened Coulomb written out directly: the same expression as the general
                # branch below, spelled so the arithmetic is bit-for-bit the original one.
                coef = -self.k * s[i] * s[j] / (r ** 3) * occ[j]
            else:
                inv = (r * r + self.soft ** 2).pow(-1.5) if self.soft > 0 else r.pow(-3)
                recv = s[i] if self._recv_coupled else 1.0
                coef = self.sign * self.k * recv * s[j] * inv * occ[j]
            acc = torch.zeros(N, D, device=pos.device).index_add_(0, i, coef[:, None] * d)

        if self.clamp > 0:                                        # optional stability cap on |a|
            mag = acc.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            acc = acc * (mag.clamp(max=self.clamp) / mag)
        acc = acc * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


@register_operator("cohesion", family="interaction", set="particle", kind="lateral")
class Cohesion(Lateral):
    """Cohesion, the first boids steering rule: accelerate toward the mean position of the
    live neighbours, which is what holds a flock together against the other two rules.

    particle -[neighbour relation]-> particle: reads pos, emits an acceleration.

        d2x_i/dt2 = a w_i  mean_j (x_j - x_i)

    The neighbour mean of (x_j - x_i) is the offset from the particle to its neighbours'
    centroid, in world units. `scale` is a, the cohesion strength, in inverse time squared
    -- unbounded in r, so this is the rule that dominates at long range. w_i is the per-type
    `cohesion` weight, dimensionless, letting one type in a mixed set flock more tightly
    than another.

    Emits an acceleration, not a velocity: a boid has momentum, and steering rules add to it.

    Reference: Reynolds, C. W. (1987). Flocks, herds and schools: a distributed behavioral
    model. SIGGRAPH Comput. Graph. 21(4):25-34.
    """

    EMIT = "acceleration"                            # second-order: a boid has momentum
    SUPPORTED_DIMS = [2, 3]                          # neighbour_mean is N-D; the rule is dimension-generic
    REQUIRES_PARAMS = []                             # no required params — `scale` optional
    REQUIRES_TYPE_PROPS = ["cohesion"]
    MECHANISM_TAGS = ["cohesion", "collective_motion"]
    PARAM_ROLES = {"scale": "cohesion_strength"}
    REFERENCE = ("Reynolds, C. W. (1987). Flocks, herds and schools: a distributed "
                 "behavioral model. SIGGRAPH Comput. Graph. 21(4):25-34.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("scale", 0.5e-5))     # PDE_B cohesion scale a1
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        w = lvl.cohesion
        acc = neighbour_mean(lvl.get("pos"), lvl.occ, lvl.edge_index,
                             getattr(H, "periodic", False),
                             getattr(H, "world_size", getattr(H, "world_width", 1.0)),
                             lambda i, j, d: w[i, None] * self.a * d)
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


@register_operator("separation", family="interaction", set="particle", kind="lateral")
class Separation(Lateral):
    """Separation, the second boids steering rule: accelerate away from close neighbours, which
    is what stops `cohesion` from collapsing the flock to a point.

    particle -[neighbour relation]-> particle: reads pos, emits an acceleration.

        d2x_i/dt2 = -a w_i  mean_j (x_j - x_i) / |x_j - x_i|^2

    Dividing by the squared distance makes each contribution fall off as 1/r, so the rule is
    short-range where `cohesion` is long-range, and the two balance at a preferred spacing.
    `scale` is a, the separation strength, in world units squared over time squared (it carries
    the length the 1/|d| leaves behind). w_i is the per-type `separation` weight, dimensionless.
    Division by zero is prevented upstream: the rewire's `min_radius` keeps |d| off zero.

    Reference: Reynolds, C. W. (1987). Flocks, herds and schools: a distributed behavioral
    model. SIGGRAPH Comput. Graph. 21(4):25-34.
    """

    EMIT = "acceleration"                            # second-order: a boid has momentum
    SUPPORTED_DIMS = [2, 3]                          # neighbour_mean is N-D; the rule is dimension-generic
    REQUIRES_PARAMS = []                             # no required params — `scale` optional (separation is a type prop)
    REQUIRES_TYPE_PROPS = ["separation"]
    MECHANISM_TAGS = ["short_range_repulsion", "collision_avoidance"]
    PARAM_ROLES = {"scale": "separation_strength"}
    REFERENCE = ("Reynolds, C. W. (1987). Flocks, herds and schools: a distributed "
                 "behavioral model. SIGGRAPH Comput. Graph. 21(4):25-34.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("scale", 1e-8))       # PDE_B separation scale a3
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        w = lvl.separation

        def msg(i, j, d):
            d2 = (d * d).sum(-1, keepdim=True)          # |d|^2 (> 0 via the graph's min_radius)
            return -w[i, None] * self.a * d / d2

        acc = neighbour_mean(lvl.get("pos"), lvl.occ, lvl.edge_index,
                             getattr(H, "periodic", False),
                             getattr(H, "world_size", getattr(H, "world_width", 1.0)), msg)
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


@register_operator("velocity_align", "alignment", family="interaction", set="particle", kind="lateral")
class VelocityAlign(Lateral):
    """Velocity alignment: accelerate toward the mean velocity of the neighbours. The third
    boids steering rule, and on its own the Vicsek model -- the one term that turns a set of
    independently moving particles into a collectively moving one.

    particle -[neighbour relation]-> particle: reads pos and vel, emits an acceleration.

        d2x_i/dt2 = a  ( sum_j g_ij (v_j - v_i) ) / ( sum_j g_ij )  +  eta xi_i

    a is the alignment strength, in inverse time, since (v_j - v_i) already carries a velocity.
    g_ij is the weight the receiver gives that neighbour: `gate: none` sets every g_ij to 1, so
    all neighbours inside the rewire's radius count equally (Vicsek and boids both do this);
    `gate: contact` instead applies a smoothstep falling from 1 to 0 between (1 - softness) r
    and r, in world units, so influence fades with distance rather than stopping at a cliff.
    eta is `noise`, isotropic Vicsek noise on the alignment, and xi_i a standard normal vector;
    it is the control parameter of the order-disorder transition, and 0 by default.

    With `per_type: true` the whole term is scaled by the receiver's own `alignment` weight,
    dimensionless -- the boids special case, where types align to different degrees.

    Reference: Vicsek, T., Czirok, A., Ben-Jacob, E., Cohen, I. & Shochet, O. (1995). Novel
    type of phase transition in a system of self-driven particles. Phys. Rev. Lett.
    75:1226-1229. The per-type variant follows Reynolds, C. W. (1987). SIGGRAPH Comput.
    Graph. 21(4):25-34.
    """

    EMIT = "acceleration"                       # second-order: a boid has momentum
    SUPPORTED_DIMS = [2, 3]                     # velocity neighbour-mean is dimension-generic
    REQUIRES_PARAMS = []                        # no required params — all knobs optional (defaults in __init__)
    OPTIONAL_TYPE_PROPS = ["alignment"]        # read per-receiver only when `per_type: true` (boids)
    MECHANISM_TAGS = ["velocity_alignment", "collective_motion", "vicsek"]
    PARAM_ROLES = {"a": "alignment_strength", "gate": "neighbour_weighting",
                   "r": "contact_radius", "noise": "orientation_noise"}
    REFERENCE = ("Vicsek, T., Czirok, A., Ben-Jacob, E., Cohen, I. & Shochet, O. (1995). "
                 "Novel type of phase transition in a system of self-driven particles. "
                 "Phys. Rev. Lett. 75:1226-1229; per-type variant: Reynolds, C. W. (1987). "
                 "SIGGRAPH Comput. Graph. 21(4):25-34.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("a", params.get("scale", 5e-4)))    # alignment scale (`scale`: legacy alias)
        self.gate = str(params.get("gate", "none"))                   # "none" (Vicsek/boids) | "contact"
        self.r = float(params.get("r", 0.05))                         # contact radius (gate="contact")
        self.softness = float(params.get("softness", 0.5))            # falloff band [0,1]; 0 = hard cutoff
        self.per_type = bool(params.get("per_type", False))           # boids special case: per-type weight
        self.weight_prop = str(params.get("weight", "alignment"))     # which type property holds the weight
        self.noise = float(params.get("noise", 0.0))                  # isotropic Vicsek orientation noise (off by default)
        self.at = params.get("_at", "particle")
        if self.gate not in ("none", "contact"):
            raise ValueError(f"alignment: gate must be 'none' or 'contact', got {self.gate!r}")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos, vel, occ = lvl.get("pos"), lvl.get("vel"), lvl.occ
        N, D = vel.shape[0], vel.shape[-1]
        ei = lvl.edge_index                                 # row0 = receiver i, row1 = neighbour j
        if ei.numel() == 0:
            return {self.at: torch.zeros(N, D, device=vel.device)}
        i, j = ei[0], ei[1]

        if self.gate == "contact":                          # smoothstep contact gate: 1 -> 0 over [r_in, r]
            d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False),
                              getattr(H, "world_size", getattr(H, "world_width", 1.0)))
            dist = d.norm(dim=-1)
            r_in = (1.0 - self.softness) * self.r
            t = ((self.r - dist) / max(self.r - r_in, 1e-12)).clamp(0.0, 1.0)
            gate = t * t * (3.0 - 2.0 * t)
        else:                                               # gate="none": every neighbour equal (Vicsek/boids)
            gate = torch.ones(ei.shape[1], device=vel.device)
        w = gate * occ[j]                                   # gate + mask dormant neighbours

        msg = self.a * (vel[j] - vel[i]) * w[:, None]
        if self.per_type:                                   # boids: per-receiver alignment weight
            pw = getattr(lvl, self.weight_prop)[i]
            msg = msg * pw[:, None]
        acc = torch.zeros(N, D, device=vel.device).index_add_(0, i, msg)
        deg = torch.zeros(N, device=vel.device).index_add_(0, i, w)
        acc = (acc / deg.clamp(min=1.0)[:, None]) * occ[:, None]       # (weighted) mean over neighbours
        if self.noise > 0.0:                                           # Vicsek angular noise: order vs disorder
            acc = acc + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None),
                                                 device=vel.device) * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


_A, _B, _P = 7.049556277, 0.6022245584, 4.0


@register_operator("stillinger_weber", set="particle", kind="lateral", family="interaction")
class StillingerWeber(Lateral):
    """The Stillinger-Weber potential: a two-body well plus a three-body penalty on bond
    ANGLES. The angular term is the point -- it is what makes a liquid tetrahedral, and so
    what lets one particle per molecule reproduce water's freezing without any hydrogens.

    particle -[its own min-image neighbour list]-> particle: reads pos, emits an acceleration.
    The force is the autograd gradient of the energy, so the operator is differentiable.

        E   = eps [ sum_{i<j} phi2(r_ij) + lam sum_i sum_{j<k} h(r_ij) h(r_ik) (cos t_jik - cos t0)^2 ]
        phi2(r) = A (B r^-p - 1) exp(1 / (r - a))
        h(r)    = exp(gamma / (r - a))
        d2x_i/dt2 = -dE/dx_i                        (unit mass)

    Lengths are in units of the particle diameter sigma, which this implementation fixes at 1,
    so r_ij is a dimensionless reduced distance. a is the cutoff in those units (1.8 for mW
    water); both exponentials go to zero as r approaches a from below, which is what makes the
    potential and its force strictly finite-ranged and continuous at the cutoff. A, B and p are
    the fixed Stillinger-Weber shape constants (7.0496, 0.6022, 4), not tunable here.

    t_jik is the angle at particle i subtended by neighbours j and k, and cos t0 the preferred
    cosine: -1/3, i.e. 109.47 degrees, the tetrahedral angle. lam is the tetrahedral strength,
    dimensionless -- the single most consequential parameter, since it weighs the angular
    penalty against the two-body well and thereby how strongly the network resists non-
    tetrahedral packing (23.15 for mW water). gamma sets the range over which a neighbour still
    counts as bonded for the angular term. eps is the overall energy scale.

    `maxnb` is the padded neighbour count per particle, a buffer size and not physics; too small
    silently truncates the neighbour list.

    Reference: Stillinger, F. H. & Weber, T. A. (1985). Computer simulation of local order in
    condensed phases of silicon. Phys. Rev. B 31:5262-5271. The mW monatomic water parameters
    (lam 23.15, a 1.8, gamma 1.2) are from Molinero, V. & Moore, E. B. (2009). Water modeled as
    an intermediate element between carbon and silicon. J. Phys. Chem. B 113:4008-4016.
    """

    EMIT = "acceleration"                       # second-order: -grad E / m, Newtonian, m = 1
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic (reads D = pos.shape[-1])
    DIFFERENTIABLE = True                        # force = -grad E by autograd
    INPUTS = ["particle"]; OUTPUTS = ["particle"]
    READS = ["pos"]; WRITES = ["vel"]
    MAPS = []                                    # builds its own min-image neighbour list (implicit rewire)
    REQUIRES_PARAMS = []                         # all params default to mW water; none mandatory
    MECHANISM_TAGS = ["tetrahedral_network", "three_body", "angular_interaction",
                      "stillinger_weber", "monatomic_water", "directional_bonding",
                      "mechanical_interaction"]
    PARAM_ROLES = {"lam": "tetrahedral_strength", "cos0": "preferred_cos_angle",
                   "gamma": "three_body_range", "a": "cutoff_over_sigma", "eps": "energy_scale",
                   "maxnb": "neighbour_buffer"}
    REFERENCE = ("Stillinger, F. H. & Weber, T. A. (1985). Computer simulation of local "
                 "order in condensed phases of silicon. Phys. Rev. B 31:5262-5271; mW water "
                 "parameters: Molinero, V. & Moore, E. B. (2009). Water modeled as an "
                 "intermediate element between carbon and silicon. J. Phys. Chem. B "
                 "113:4008-4016.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.lam = float(params.get("lam", 23.15))          # mW water default
        self.cos0 = float(params.get("cos0", -1.0 / 3.0))   # tetrahedral (109.47 deg)
        self.gamma = float(params.get("gamma", 1.2))
        self.a = float(params.get("a", 1.8))                # cutoff / sigma
        self.eps = float(params.get("eps", 1.0))            # overall energy scale
        self.maxnb = int(params.get("maxnb", 40))           # padded neighbour count

    # --- geometry ---------------------------------------------------------- #
    def _box(self, H, D):
        ws = getattr(H, "world_size", None)
        if ws is None:
            return None, False
        return ws[:D], bool(getattr(H, "periodic", False))

    def _min_image(self, dv, box, periodic):
        return dv - box * torch.round(dv / box) if (periodic and box is not None) else dv

    def _neighbours(self, pos, box, periodic):
        with torch.no_grad():
            d = self._min_image(pos[:, None, :] - pos[None, :, :], box, periodic)
            r = d.norm(dim=-1); r.fill_diagonal_(1e9)
            k = min(self.maxnb, pos.shape[0] - 1)
            rr, idx = torch.topk(r, k, largest=False)
            return idx, rr < self.a

    # --- energy (two-body + three-body) ------------------------------------ #
    def _energy(self, pos, nb, valid, box, periodic):
        a = self.a
        d = self._min_image(pos[:, None, :] - pos[nb], box, periodic)     # [N,k,D]
        r = d.norm(dim=-1)
        inside = valid & (r < a) & (r > 1e-4)
        rr = r.clamp(min=1e-4)
        arg2 = torch.where(inside, 1.0 / (rr - a), torch.full_like(rr, -1e9))
        phi2 = _A * (_B * rr.pow(-_P) - 1.0) * torch.exp(arg2)
        E2 = 0.5 * (phi2 * inside).sum()                                  # symmetric list -> halve
        u = d / rr.unsqueeze(-1)
        cos = torch.einsum("nmc,nkc->nmk", u, u)                          # [N,k,k]
        arg3 = torch.where(inside, self.gamma / (rr - a), torch.full_like(rr, -1e9))
        h = torch.exp(arg3)                                              # ~0 outside cutoff
        k = h.shape[1]
        triu = torch.triu(torch.ones(k, k, device=pos.device), diagonal=1).bool()
        pair = (h[:, :, None] * h[:, None, :]) * (cos - self.cos0) ** 2
        E3 = self.lam * (pair * triu[None]).sum()
        return self.eps * (E2 + E3)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        N, D = pos.shape[0], pos.shape[-1]
        box, periodic = self._box(H, D)
        if box is not None:
            box = box.to(pos.device)
        nb, valid = self._neighbours(pos, box, periodic)
        with torch.enable_grad():                                        # engine runs under no_grad
            p = pos.detach().requires_grad_(True)
            E = self._energy(p, nb, valid, box, periodic)
            grad, = torch.autograd.grad(E, p)
        acc = torch.nan_to_num(-grad) * lvl.occ[:, None]                 # force / m (m = 1); dormant -> 0
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


@register_operator("radius_graph", family="topology", set="particle", kind="rewire")
class RadiusGraph(Rewire):
    """The neighbour relation: two particles interact when they are close enough. Every
    lateral law in this module reads what this writes, so it is scheduled before them.

    particle -> particle: reads pos, writes the set's `edge_index` in place.

        E = { (i, j) : r_min < |x_j - x_i| <= r_max,  both live }

    r_max is `radius` and r_min the optional `min_radius`, both in world units. r_max is the
    interaction cutoff -- the physical claim that beyond it the law is negligible -- and
    changing it changes the model, not the discretization. r_min excludes a pair that is
    closer than it, which keeps a 1/|d| law such as `separation` off its singularity. The
    relation is symmetric: every pair appears in both directions, and there are no self-edges.
    Under periodic boundaries the distance is the minimum-image one.

    `block` is how many receivers are scored per pass, a memory knob and not physics: the
    O(N^2) distance matrix is never materialised, which is what carries this to 1e4-1e5 nodes.

    Reference: none -- a cutoff neighbour list is standard practice, not a result. Plexus
    (this work).
    """

    EMIT = None                                 # rewire: rebuilds edge_index in place, returns no delta
    SUPPORTED_DIMS = [2, 3]                      # pairwise distances are dimension-generic
    REQUIRES_PARAMS = ["radius"]
    MECHANISM_TAGS = ["radius_graph", "neighbor_search", "rewire"]
    PARAM_ROLES = {"min_radius": "inner_cutoff_radius", "block": "block_size"}
    REFERENCE = "Plexus (this work); a cutoff neighbour list is standard practice."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.r_max = float(params["radius"])
        self.r_min = float(params.get("min_radius", 0.0))
        self.block = int(params.get("block", 2048))
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        lvl.edge_index = edges_radius_blockwise(
            lvl.get("pos"), lvl.occ, self.r_min, self.r_max,
            periodic=getattr(H, "periodic", False),
            world_width=getattr(H, "world_size", getattr(H, "world_width", 1.0)),
            block=self.block,
        )
        return {}


# ==========================================================================================================
#  implementations -- same biology, different numerics
# ==========================================================================================================
# `squared_law[implementation: warp]` -- the all-pairs inverse square without the [N, N] matrices.
if HAVE_WARP:

    @wp.kernel
    def _pull3(P: wp.array(dtype=wp.vec3), S: wp.array(dtype=float), soft2: float, n: int,
               OUT: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        pi = P[i]
        a = wp.vec3(0.0, 0.0, 0.0)
        for j in range(n):
            d = P[j] - pi
            r2 = wp.dot(d, d) + soft2
            # The same 1e-12 floor on r^2 the torch path uses, and what makes the j == i term
            # safe: unsoftened, the self term has r^2 = 0, so the floor caps the reciprocal at a
            # finite 1e18 and it is then multiplied by d = 0. A particle pulls on itself by 0.
            if r2 < 1.0e-12:
                r2 = 1.0e-12
            a = a + d * (S[j] / (r2 * wp.sqrt(r2)))
        OUT[i] = a

    @wp.kernel
    def _pull2(P: wp.array(dtype=wp.vec2), S: wp.array(dtype=float), soft2: float, n: int,
               OUT: wp.array(dtype=wp.vec2)):
        i = wp.tid()
        pi = P[i]
        a = wp.vec2(0.0, 0.0)
        for j in range(n):
            d = P[j] - pi
            r2 = wp.dot(d, d) + soft2
            if r2 < 1.0e-12:
                r2 = 1.0e-12
            a = a + d * (S[j] / (r2 * wp.sqrt(r2)))
        OUT[i] = a


def _launch(kernel, n, dev, inputs):
    """`wp.launch` on pytorch's current stream -- see mpm_warp._wp_launch for why this matters."""
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(dev)),
                         sync_enter=False, sync_exit=False):
        wp.launch(kernel, dim=n, device=f"cuda:{dev.index or 0}", inputs=inputs)


@register_operator("squared_law", implementation="warp", family="interaction",
                   set="particle", kind="lateral")
class SquaredLawWarp(SquaredLaw):
    """Same law, as one Warp kernel: still O(N^2) arithmetic, but O(N) memory, because each
    thread accumulates its own sum and no [N, N] array is ever allocated. That is what lifts
    the particle count off the memory ceiling the torch paths hit.

    CUDA-only, and 2D or 3D only. Everything else -- the neighbour-graph branch, a CPU tensor,
    a fourth dimension -- falls back to the torch implementation rather than failing, because
    `implementation: warp` in a specification is a preference and not a demand that the run
    stop if it cannot be met. Not differentiable: the kernel has no autograd path."""

    MECHANISM_TAGS = SquaredLaw.MECHANISM_TAGS + ["fused_kernel"]
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev = pos.device
        if not HAVE_WARP or not pos.is_cuda or not self.all_pairs:
            # The neighbour-graph branch is O(E) and materialises nothing, so there is nothing here
            # for it to win; a non-CUDA device has no kernel at all. Both defer, rather than
            # failing, because `implementation: warp` on a spec is a preference and not a demand
            # that the run stop if it cannot be met.
            return SquaredLaw.forward(self, H, mask)
        if getattr(H, "periodic", False):
            raise ValueError("squared_law all_pairs=True supports only open/free boundaries")

        occ = lvl.occ
        N, D = pos.shape[0], pos.shape[-1]
        s = getattr(lvl, self.coupling, None)
        if s is None:
            raise ValueError(f"squared_law(law={self.law}) needs per-type property "
                             f"{self.coupling!r} on {self.at!r}; declare it in the set's `types`.")
        if D not in (2, 3):
            return SquaredLaw.forward(self, H, mask)

        P = pos.contiguous().float()
        S = (s * occ).contiguous().float()
        pull = torch.empty_like(P)
        vec = wp.vec3 if D == 3 else wp.vec2
        _launch(_pull3 if D == 3 else _pull2, N, dev,
                [wp.from_torch(P, dtype=vec), wp.from_torch(S),
                 float(self.soft ** 2), int(N), wp.from_torch(pull, dtype=vec)])

        recv = s if self._recv_coupled else torch.ones(N, device=dev, dtype=pos.dtype)
        acc = (self.sign * self.k) * recv[:, None] * pull.to(pos.dtype)
        if self.clamp > 0:
            mag = acc.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            acc = acc * (mag.clamp(max=self.clamp) / mag)
        acc = acc * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


# `squared_law[implementation: mesh]` -- particle-mesh gravity: O(N), not O(N^2).
if HAVE_WARP:

    @wp.kernel
    def _cic_deposit(P: wp.array(dtype=wp.vec3), M: wp.array(dtype=float),
                     lo: wp.vec3, inv_h: float, n: int, RHO: wp.array(dtype=float)):
        """Cloud-in-cell: split each mass over the 8 cells of its containing cube, by volume."""
        i = wp.tid()
        x = (P[i] - lo) * inv_h
        ix = int(wp.floor(x[0]))
        iy = int(wp.floor(x[1]))
        iz = int(wp.floor(x[2]))
        fx = x[0] - float(ix)
        fy = x[1] - float(iy)
        fz = x[2] - float(iz)
        m = M[i]
        for a in range(2):
            wx = 1.0 - fx
            if a == 1:
                wx = fx
            gx = ix + a
            if gx >= 0 and gx < n:
                for b in range(2):
                    wy = 1.0 - fy
                    if b == 1:
                        wy = fy
                    gy = iy + b
                    if gy >= 0 and gy < n:
                        for c in range(2):
                            wz = 1.0 - fz
                            if c == 1:
                                wz = fz
                            gz = iz + c
                            if gz >= 0 and gz < n:
                                wp.atomic_add(RHO, (gx * n + gy) * n + gz, m * wx * wy * wz)

    @wp.kernel
    def _cic_gather(P: wp.array(dtype=wp.vec3), lo: wp.vec3, inv_h: float, n: int,
                    AX: wp.array(dtype=float), AY: wp.array(dtype=float), AZ: wp.array(dtype=float),
                    OUT: wp.array(dtype=wp.vec3)):
        """The same weights, read back: the transfer is its own transpose, which is what keeps
        momentum conserved to round-off rather than leaking it into the grid."""
        i = wp.tid()
        x = (P[i] - lo) * inv_h
        ix = int(wp.floor(x[0]))
        iy = int(wp.floor(x[1]))
        iz = int(wp.floor(x[2]))
        fx = x[0] - float(ix)
        fy = x[1] - float(iy)
        fz = x[2] - float(iz)
        a = wp.vec3(0.0, 0.0, 0.0)
        for p in range(2):
            wx = 1.0 - fx
            if p == 1:
                wx = fx
            gx = ix + p
            if gx >= 0 and gx < n:
                for q in range(2):
                    wy = 1.0 - fy
                    if q == 1:
                        wy = fy
                    gy = iy + q
                    if gy >= 0 and gy < n:
                        for r in range(2):
                            wz = 1.0 - fz
                            if r == 1:
                                wz = fz
                            gz = iz + r
                            if gz >= 0 and gz < n:
                                w = wx * wy * wz
                                k = (gx * n + gy) * n + gz
                                a = a + wp.vec3(AX[k] * w, AY[k] * w, AZ[k] * w)
        OUT[i] = a


def _launch(kernel, n, dev, inputs):
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(dev)),
                         sync_enter=False, sync_exit=False):
        wp.launch(kernel, dim=n, device=f"cuda:{dev.index or 0}", inputs=inputs)


@register_operator("squared_law", implementation="mesh", family="interaction",
                   set="particle", kind="lateral")
class SquaredLawMesh(SquaredLaw):
    """Same law, solved on a grid instead of pair by pair: deposit the masses cloud-in-cell,
    solve Poisson in k-space, take the gradient in k-space, gather back with the same weights.
    O(N + M log M) for M = n_grid^3 cells, against O(N^2), which is what makes 1e7 bodies
    reachable. The deposit and gather being transposes of one another is what keeps total
    momentum conserved to round-off rather than leaking it into the grid.

    Two limits worth knowing before reading a result from it. The grid is padded to twice the
    cloud's own extent so the nearest periodic image sits a full box away and the boundary is
    effectively isolated. And the method has a softening of its own, about 1.5 cells wide,
    whether or not the specification asked for one -- so a specification declaring a smaller
    `softening` is running a smoother force than it wrote down, and the operator says so once
    on the first call.

    3D only, gravity only, CUDA-only, not differentiable; anything else falls back to torch.

    Reference: Hockney, R. W. & Eastwood, J. W. (1988). Computer Simulation Using Particles.
    Adam Hilger (the particle-mesh method and cloud-in-cell assignment)."""

    MECHANISM_TAGS = SquaredLaw.MECHANISM_TAGS + ["particle_mesh", "fft_poisson"]
    REFERENCE = (SquaredLaw.REFERENCE + " Particle-mesh method and cloud-in-cell assignment: "
                 "Hockney, R. W. & Eastwood, J. W. (1988). Computer Simulation Using "
                 "Particles. Adam Hilger.")
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    PARAM_ROLES = dict(SquaredLaw.PARAM_ROLES,
                       n_grid="cells per axis of the (zero-padded) FFT grid")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.n_grid = int(params.get("n_grid", 256))
        if self.n_grid & (self.n_grid - 1):
            raise ValueError(f"squared_law[mesh]: n_grid must be a power of two, got {self.n_grid}")
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        if not HAVE_WARP or not pos.is_cuda or not self.all_pairs or pos.shape[-1] != 3:
            return SquaredLaw.forward(self, H, mask)
        if self.law != "gravity":
            # Coulomb has SIGNED charge, which Poisson handles perfectly well -- but the receiver
            # coupling differs and no spec needs it yet, so refuse rather than quietly get it wrong.
            raise ValueError("squared_law[mesh] implements law: gravity (use warp for coulomb)")

        dev, N = pos.device, pos.shape[0]
        occ = lvl.occ
        s = getattr(lvl, self.coupling, None)
        if s is None:
            raise ValueError(f"squared_law[mesh] needs per-particle {self.coupling!r}")
        n = self.n_grid
        P = pos.contiguous().float()
        M = (s * occ).contiguous().float()

        # --- the box: twice the cloud's own extent, so the nearest periodic image is a box away ---
        lo_p = P.amin(0)
        hi_p = P.amax(0)
        span = (hi_p - lo_p).amax().clamp(min=1e-9) * 1.02
        h = float(2.0 * span / n)                       # cell size on the PADDED grid
        lo = (0.5 * (lo_p + hi_p) - 0.5 * n * h).contiguous()

        rho = torch.zeros(n * n * n, device=dev, dtype=torch.float32)
        _launch(_cic_deposit, N, dev,
                [wp.from_torch(P, dtype=wp.vec3), wp.from_torch(M),
                 wp.vec3(*[float(v) for v in lo.tolist()]), float(1.0 / h), int(n),
                 wp.from_torch(rho)])
        rho = rho.view(n, n, n) / (h ** 3)               # mass -> density

        # --- Poisson in k-space, then the gradient in k-space (no finite differences) ---
        kx = torch.fft.fftfreq(n, d=h, device=dev, dtype=torch.float32) * (2.0 * math.pi)
        kz = torch.fft.rfftfreq(n, d=h, device=dev, dtype=torch.float32) * (2.0 * math.pi)
        k2 = (kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
        k2[0, 0, 0] = 1.0                                # the mean density has no force; phi_0 = 0
        rho_k = torch.fft.rfftn(rho, dim=(0, 1, 2))
        del rho
        # a_k = i k * (4 pi G rho_k / k^2); the i is a swap of real and imaginary parts
        base = rho_k * ((4.0 * math.pi * self.k) / k2)
        base[0, 0, 0] = 0.0
        # Apply the specification's `softening` only when it is the coarser of the two scales.
        # The grid already softens at ~1.5 cells whether or not anyone asked; a specification
        # asking for MORE than that and not getting it would run a sharper force than it
        # declared, which for a disc galaxy is the difference between a smooth potential and one
        # that scatters stars off shot noise in its own mass field. Note the form differs:
        # multiplying the kernel by exp(-k^2 eps^2 / 2) is GAUSSIAN softening of scale eps, where
        # `softening` is Plummer elsewhere in this contract. The two agree in the far field and
        # differ by roughly 15% in force magnitude at r ~ eps.
        self._eps_used = max(float(self.soft), 1.5 * h)
        if self.soft > 1.5 * h:
            base = base * torch.exp(-0.5 * k2 * (self.soft ** 2))
        del rho_k, k2
        acc_g = []
        for ax, kv in ((0, kx[:, None, None]), (1, kx[None, :, None]), (2, kz[None, None, :])):
            acc_g.append(torch.fft.irfftn(base * (1j * kv), s=(n, n, n), dim=(0, 1, 2))
                         .contiguous().view(-1))
        del base

        out = torch.empty_like(P)
        _launch(_cic_gather, N, dev,
                [wp.from_torch(P, dtype=wp.vec3),
                 wp.vec3(*[float(v) for v in lo.tolist()]), float(1.0 / h), int(n),
                 wp.from_torch(acc_g[0]), wp.from_torch(acc_g[1]), wp.from_torch(acc_g[2]),
                 wp.from_torch(out, dtype=wp.vec3)])
        del acc_g

        if not self._said:
            self._said = True
            # Report the softening actually in force, once. Cloud-in-cell plus a k-space
            # gradient softens at roughly 1.5 cells; a specification that declared
            # `softening: 0.15` and is handed 2.3 is running physics it did not write down,
            # and printing it is the only place that becomes visible.
            eff = 1.5 * h
            note = ""
            if self.soft > 0:
                note = (f"; the spec asked for {self.soft:g} = {self.soft / eff:.2f}x that"
                        if eff > 0 else "")
            print(f"[mesh] particle-mesh gravity: {n}^3 grid, cell {h:.4g}, cloud spans "
                  f"{float(span) / h:.0f} cells of the {n} (the rest is the zero pad that makes the "
                  f"boundary isolated). Force is softened at ~{eff:.4g}{note}", flush=True)

        acc = out.to(pos.dtype)
        if self.clamp > 0:
            mag = acc.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            acc = acc * (mag.clamp(max=self.clamp) / mag)
        acc = acc * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}
