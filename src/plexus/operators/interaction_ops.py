"""Pairwise laws, and the relation they act over.

    radius_graph          the relation: who is near whom (a rewire, and it comes first)
    attraction_repulsion  the two-term law the prototype scenarios are built on
    squared_law           inverse-square: gravity between bodies, Coulomb between charges
    cohesion / separation / velocity_align    the three boids terms, one operator each
    stillinger_weber      a three-body potential with an angular term (autograd)

THE GRAPH IS IN THIS FILE ON PURPOSE. Every law below it reads `edge_index`, and which relation
they read is the single most consequential thing about a spec that uses them -- a cutoff radius
changes a flock into a crystal. It is not a utility that happens to live elsewhere.
"""
from __future__ import annotations
import torch
from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image
from plexus.geometry import neighbour_mean
from plexus.models.base import Rewire
from plexus.geometry import edges_radius_blockwise


# ==========================================================================================================
# FROM `discovery_okuda/ops/attraction_repulsion.py` -- Analytic attraction-repulsion: a smooth, per-type pairwise interaction law.
# ==========================================================================================================
@register_operator("attraction_repulsion", family="interaction", set="particle", kind="lateral")
class AttractionRepulsion(Lateral):
    EMIT = "velocity"             # emits a velocity (overdamped law)
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic (reads D = pos.shape[-1])
    REQUIRES_PARAMS = ["sigma"]                 # the cutoff lives on the radius_graph rewire op
    REQUIRES_TYPE_PROPS = ["p"]                 # per-type force-law params [p1,p2,p3,p4]
    # mechanism-search metadata: the long-range Gaussian (p1,p2) is the pull, the
    # short-range Gaussian (p3,p4) the push; their balance sets the phase.
    MECHANISM_TAGS = ["long_range_attraction", "short_range_repulsion", "coarsening", "lattice_forming"]
    PARAM_ROLES = {"sigma": "interaction_length", "noise": "exploration_noise",
                   "p": "[pull_strength, pull_range, push_strength, push_range] per type"}
    REFERENCE = "D'Orsogna, M. R. et al. (2006). Self-propelled particles with soft-core interactions. Phys. Rev. Lett. 96:104302."

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


# ==========================================================================================================
# FROM `discovery_okuda/ops/squared_law.py` -- squared_law -- the inverse-square law between particles: electrostatics (Coulomb) OR gravity (Newton).
# ==========================================================================================================
def _inv_square_sum(pos, src, soft2):
    """All-pairs inverse-square PULL at each particle: pull_i = Σ_j src_j (r_j-r_i)/denom,
    denom = (|r_j-r_i|^2 + soft2)^(3/2). This is a per-particle VECTOR (the summed inverse-square
    interaction), NOT a Plexus `Field` (grid entity) -- no grid is involved; the caller scales it
    by the signed strength and the receiver's coupling to get the acceleration. Per-dimension so
    only [N,N] matrices appear; a free function so torch.compile can FUSE the reduction (the [N,N]
    r2/inv_r3 never materialise). `src` folds in occupancy (dormant = 0). (Mocz getAcc generalised
    to any source charge.)"""
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


_inv_square_sum_compiled = None


def _get_inv_square_sum(compile):
    """Return the (optionally torch.compiled) all-pairs inverse-square-sum kernel, compiling once."""
    global _inv_square_sum_compiled
    if not compile:
        return _inv_square_sum
    if _inv_square_sum_compiled is None:
        _inv_square_sum_compiled = torch.compile(_inv_square_sum)
    return _inv_square_sum_compiled


@register_operator("squared_law", family="interaction", set="particle", kind="lateral")
class SquaredLaw(Lateral):
    EMIT = "acceleration"                        # emits an acceleration (charges/masses have inertia)
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
                   "compile": "torch.compile the all-pairs kernel (big N)",
                   "clamp": "max |acceleration| (0 = unbounded)"}
    REFERENCE = "Newton, I. (1687). Principia (inverse-square law); Coulomb, C.-A. (1785)."

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
        self.compile = bool(params.get("compile", False))         # torch.compile the all-pairs kernel
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
            pull = _get_inv_square_sum(self.compile)(pos, src, self.soft ** 2)
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
                # exact legacy Coulomb path -- byte-identical to the pre-merge squared_law
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


# ==========================================================================================================
# FROM `discovery_okuda/ops/cohesion.py` -- cohesion -- a boids steering rule (Lateral, second-derivative).
# ==========================================================================================================
@register_operator("cohesion", family="interaction", set="particle", kind="lateral")
class Cohesion(Lateral):
    EMIT = "acceleration"
    SUPPORTED_DIMS = [2, 3]                          # neighbour_mean is N-D; the rule is dimension-generic
    REQUIRES_PARAMS = []                             # no required params — `scale` optional
    REQUIRES_TYPE_PROPS = ["cohesion"]
    MECHANISM_TAGS = ["cohesion", "collective_motion"]
    PARAM_ROLES = {"scale": "cohesion_strength"}
    REFERENCE = "Reynolds, C. W. (1987). Flocks, herds and schools: a distributed behavioral model. SIGGRAPH Comput. Graph. 21(4):25-34."

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


# ==========================================================================================================
# FROM `discovery_okuda/ops/separation.py` -- separation -- a boids steering rule (Lateral, second-derivative).
# ==========================================================================================================
@register_operator("separation", family="interaction", set="particle", kind="lateral")
class Separation(Lateral):
    EMIT = "acceleration"
    SUPPORTED_DIMS = [2, 3]                          # neighbour_mean is N-D; the rule is dimension-generic
    REQUIRES_PARAMS = []                             # no required params — `scale` optional (separation is a type prop)
    REQUIRES_TYPE_PROPS = ["separation"]
    MECHANISM_TAGS = ["short_range_repulsion", "collision_avoidance"]
    PARAM_ROLES = {"scale": "separation_strength"}
    REFERENCE = "Reynolds, C. W. (1987). Flocks, herds and schools. SIGGRAPH Comput. Graph. 21(4):25-34."

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


# ==========================================================================================================
# FROM `discovery_okuda/ops/velocity_align.py` -- velocity_align (was alignment) -- Vicsek-style neighbour velocity alignment (the NOMINAL model);
# ==========================================================================================================
@register_operator("velocity_align", "alignment", family="interaction", set="particle", kind="lateral")
class VelocityAlign(Lateral):                    # (alias `alignment`, one migration cycle)
    EMIT = "acceleration"            # emits an acceleration
    SUPPORTED_DIMS = [2, 3]                     # velocity neighbour-mean is dimension-generic
    REQUIRES_PARAMS = []                        # no required params — all knobs optional (defaults in __init__)
    OPTIONAL_TYPE_PROPS = ["alignment"]        # read per-receiver only when `per_type: true` (boids)
    MECHANISM_TAGS = ["velocity_alignment", "collective_motion", "vicsek"]
    PARAM_ROLES = {"a": "alignment_strength", "gate": "neighbour_weighting",
                   "r": "contact_radius", "noise": "orientation_noise"}
    REFERENCE = "Vicsek, T. et al. (1995). Novel type of phase transition in a system of self-driven particles. Phys. Rev. Lett. 75:1226-1229."

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


# ==========================================================================================================
# FROM `discovery_okuda/ops/stillinger_weber.py` -- Stillinger--Weber interaction: a two-body well + a THREE-BODY angular penalty.
# ==========================================================================================================
_A, _B, _P = 7.049556277, 0.6022245584, 4.0


@register_operator("stillinger_weber", set="particle", kind="lateral", family="interaction",
                   implementation="autograd")
class StillingerWeber(Lateral):
    EMIT = "acceleration"                       # d2x/dt2 = force / m  (Newtonian, engine-integrated)
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
    REFERENCE = ("Stillinger, F. H. & Weber, T. A. (1985). Phys. Rev. B 31:5262; "
                 "mW water: Molinero, V. & Moore, E. B. (2009). J. Phys. Chem. B 113:4008. "
                 "Promoted from plexus prototype/ice (mw_forces).")

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


# ==========================================================================================================
# FROM `discovery_okuda/ops/graph.py` -- Relation-building (rewire) operators: construct a Level's `edge_index`.
# ==========================================================================================================
@register_operator("radius_graph", family="topology", set="particle", kind="rewire")
class RadiusGraph(Rewire):
    """Set `Level.edge_index` to all live pairs within `radius` (optionally beyond
    `min_radius`). Blockwise build -> scales to 1e4-1e5 nodes; minimum-image under
    periodic BC. Run before a pairwise lateral operator in the schedule."""
    EMIT = None                                 # rewire: rebuilds edge_index; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]                      # pairwise distances are dimension-generic
    REQUIRES_PARAMS = ["radius"]
    MECHANISM_TAGS = ["radius_graph", "neighbor_search", "rewire"]
    PARAM_ROLES = {"min_radius": "inner_cutoff_radius", "block": "block_size",
                   "compile": "torch.compile the O(N^2) block distance+mask kernel"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.r_max = float(params["radius"])
        self.r_min = float(params.get("min_radius", 0.0))
        self.block = int(params.get("block", 2048))
        self.compile = bool(params.get("compile", False))    # torch.compile the O(N^2) distance kernel
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        lvl.edge_index = edges_radius_blockwise(
            lvl.get("pos"), lvl.occ, self.r_min, self.r_max,
            periodic=getattr(H, "periodic", False),
            world_width=getattr(H, "world_size", getattr(H, "world_width", 1.0)),
            block=self.block, compile=self.compile,
        )
        return {}
