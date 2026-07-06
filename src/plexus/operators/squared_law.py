"""squared_law -- the inverse-square law between particles: electrostatics (Coulomb) OR gravity (Newton).

ONE Lateral, second-derivative operator (EMIT=acceleration) for every pairwise
inverse-square interaction. Coulomb and Newtonian gravity share the *same contract*
-- an add-aggregated force a_i = Σ_j (strength) (r_j - r_i) / |r_j - r_i|^3 that the
engine integrates as an acceleration -- and differ only in options, so they are the
same operator, not two:

    law=coulomb (default)  a_i = Σ_j  -k q_i q_j (r_j-r_i) / |r_j-r_i|^3
                           signed `charge`; LIKE charges repel, OPPOSITE attract.
                           Receiver carries its own charge q_i (inertial mass = 1).
    law=gravity            a_i = Σ_j  +k m_j (r_j-r_i) / (|r_j-r_i|^2 + eps^2)^(3/2)
                           `mass`-weighted, ALWAYS attractive, softened. The receiver
                           factor is 1 (equivalence principle: acceleration is
                           independent of the test mass, m_i/m_i = 1).

The two knobs behind `law` are physical, not cosmetic: (sign) like-repel vs.
attract, and (receiver) whether the receiver's own coupling charge scales its
acceleration -- q_i for Coulomb (charge != inertial mass), 1 for gravity (charge =
inertial mass, so it cancels). `coupling` names the per-type source property (`charge`
or `mass`, defaulted by `law`); the engine broadcasts it to a per-particle buffer.

Two ranges, orthogonal to `law`:
    all_pairs=False (default)  sum over the neighbour graph `Level.edge_index` left by a
                               `rewire` op (radius_graph); O(E), a screened/short-range
                               interaction (plasma). Coulomb's default.
    all_pairs=True             sum over ALL pairs; O(N^2), long-range with no cutoff
                               (gravity has no screening). `compile` fuses the [N,N]
                               reduction with torch.compile (~23x, so N~5e4 fits ~0.01 GB;
                               the r2/inv_r3 matrices never materialise).

`softening` eps>0 replaces 1/r^3 with 1/(r^2+eps^2)^(3/2) (Plummer softening) so close
pairs stay finite without a neighbour cutoff. eps=0 recovers pure 1/r^3 (the radius
graph's `min_radius`, or the diagonal guard in the all-pairs kernel, keeps it finite).

Dimension-generic (SUPPORTED_DIMS [2,3]) -- the law reads D = pos.shape[-1].

Provenance:
  * Coulomb branch ported from ParticleGraph `PDE_E` (electrostatic inverse-square).
  * gravity branch ported from Philip Mocz, "Create Your Own N-body Simulation (With
    Python)" (2020), vendored at papers/nbody-python/ (MIT) -- his softened `getAcc`.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image


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


@register_operator("squared_law", level="particle", kind="lateral")
class SquaredLaw(Lateral):
    EMIT = "acceleration"                        # emits an acceleration (charges/masses have inertia)
    SUPPORTED_DIMS = [2, 3]                       # dimension-generic (reads D = pos.shape[-1])
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
